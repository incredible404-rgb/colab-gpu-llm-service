"""Unit tests for TFE keep-alive classification, idle detection, and launch detach."""
import ast
import contextlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import requests

import colab_cli
import colab_keepalive as ka


class PingClassifyTests(unittest.TestCase):
    def test_read_timeout_is_success(self):
        self.assertEqual(ka.classify_ping(None, requests.exceptions.ReadTimeout('timed out')), 'ok')

    def test_connect_timeout_is_transient(self):
        err = requests.exceptions.ConnectTimeout('connect timed out')
        self.assertEqual(ka.classify_ping(None, err), 'transient')

    def test_http_status(self):
        self.assertEqual(ka.classify_ping(200, None), 'ok')
        self.assertEqual(ka.classify_ping(204, None), 'ok')
        self.assertEqual(ka.classify_ping(404, None), 'http_4xx')
        self.assertEqual(ka.classify_ping(401, None), 'http_4xx')
        self.assertEqual(ka.classify_ping(503, None), 'transient')

    def test_url(self):
        self.assertEqual(
            ka.keep_alive_url('https://colab.research.google.com', 'gpu-g4-s-example'),
            'https://colab.research.google.com/tun/m/gpu-g4-s-example/keep-alive/')


class IdleTests(unittest.TestCase):
    def test_first_sample_starts_the_clock(self):
        active, fp = ka.inference_active([{'is_processing': False, 'n_past': 10}], None)
        self.assertTrue(active)
        self.assertEqual(fp[1], 10)

    def test_unchanged_slot_is_idle(self):
        _, fp = ka.inference_active([{'is_processing': False, 'n_past': 10}], None)
        active, fp2 = ka.inference_active([{'is_processing': False, 'n_past': 10}], fp)
        self.assertFalse(active)
        self.assertEqual(fp2, fp)

    def test_processing_or_token_change_or_timestamp(self):
        _, fp = ka.inference_active([{'is_processing': False, 'n_past': 10}], None)
        self.assertTrue(ka.inference_active([{'is_processing': True, 'n_past': 10}], fp)[0])
        self.assertTrue(ka.inference_active([{'is_processing': False, 'n_past': 11}], fp)[0])
        self.assertTrue(ka.inference_active(
            [{'is_processing': False, 'n_past': 10, 'last_request_time': '2026-11-06T12:00:00Z'}], fp)[0])


class KernelBeatTests(unittest.TestCase):
    def test_beat_is_a_short_colab_exec(self):
        self.assertEqual(ka.beat_argv('s', 'b.py'), ['colab', 'exec', '-s', 's', '-f', 'b.py', '--timeout', '60'])
        ast.parse(ka.BEAT_CODE)
        self.assertLessEqual(ka.BEAT_INTERVAL_S, 30)
        self.assertEqual(ka.PING_INTERVAL_S % ka.BEAT_INTERVAL_S, 0)

    def test_kernel_beat_results(self):
        real = subprocess.run
        try:
            subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a[0], 0, b'', b'')
            self.assertEqual(ka.kernel_beat('s', 'b.py'), (True, 'rc0'))
            subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a[0], 1, b'', b'Session not found')
            ok, detail = ka.kernel_beat('s', 'b.py')
            self.assertFalse(ok)
            self.assertIn('Session not found', detail)

            def boom(*a, **k):
                raise subprocess.TimeoutExpired(a[0], 1)
            subprocess.run = boom
            self.assertEqual(ka.kernel_beat('s', 'b.py'), (False, 'timeout'))
        finally:
            subprocess.run = real


class RotateLogTests(unittest.TestCase):
    def test_small_log_is_left_alone_and_big_log_rolls_to_dot_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'keepalive-s.log'
            self.assertFalse(ka.rotate_log(log))  # missing file
            log.write_bytes(b'x' * 100)
            self.assertFalse(ka.rotate_log(log, max_bytes=100))
            self.assertTrue(log.exists())
            log.write_bytes(b'y' * 101)
            self.assertTrue(ka.rotate_log(log, max_bytes=100))
            self.assertFalse(log.exists())
            self.assertEqual((Path(tmp) / 'keepalive-s.log.1').read_bytes(), b'y' * 101)
            log.write_bytes(b'z' * 200)
            self.assertTrue(ka.rotate_log(log, max_bytes=100))
            self.assertEqual((Path(tmp) / 'keepalive-s.log.1').read_bytes(), b'z' * 200)  # .1 is replaced, not piled up

    def test_replace_failure_is_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'k.log'
            log.write_bytes(b'x' * 500)
            with mock.patch.object(ka.os, 'replace', side_effect=PermissionError('in use')):
                self.assertFalse(ka.rotate_log(log, max_bytes=10))
            self.assertTrue(log.exists())


class RefreshEndpointTests(unittest.TestCase):
    @staticmethod
    def _state(local, endpoints):
        store = mock.Mock()
        store.get.return_value = local
        client = mock.Mock()
        client.list_assignments.return_value = [
            types.SimpleNamespace(endpoint=ep, runtime_proxy_info=types.SimpleNamespace(token=tok, url=url))
            for ep, tok, url in endpoints]
        return types.SimpleNamespace(store=store, client=client)

    @staticmethod
    def _apply(local, info):
        local.token, local.url = info.token, info.url
        local.token_expires_at = object()  # the real helper stamps now()+ttl, so it differs on every call

    def test_unchanged_proxy_info_does_not_rewrite_sessions_json(self):
        local = types.SimpleNamespace(endpoint='ep', token='t1', url='u1', token_expires_at=None)
        state = self._state(local, [('ep', 't1', 'u1')])
        self.assertEqual(ka.refresh_endpoint(state, 's', 'ep', self._apply), 'ep')
        state.store.add.assert_not_called()

    def test_new_token_or_url_is_persisted(self):
        for tok, url in (('t2', 'u1'), ('t1', 'u2')):
            local = types.SimpleNamespace(endpoint='ep', token='t1', url='u1', token_expires_at=None)
            state = self._state(local, [('ep', tok, url)])
            self.assertEqual(ka.refresh_endpoint(state, 's', 'ep', self._apply), 'ep')
            state.store.add.assert_called_once_with(local)
            self.assertEqual((local.token, local.url), (tok, url))

    def test_gone_assignment_and_missing_local_session(self):
        local = types.SimpleNamespace(endpoint='ep', token='t', url='u', token_expires_at=None)
        self.assertIsNone(ka.refresh_endpoint(self._state(local, [('other', 't', 'u')]), 's', 'ep', self._apply))
        state = self._state(None, [('cur', 't', 'u')])
        self.assertEqual(ka.refresh_endpoint(state, 's', 'cur', self._apply), 'cur')
        state.store.add.assert_not_called()


class DaemonLoopTests(unittest.TestCase):
    """Drive main() for a few simulated minutes with every outside call faked."""

    def _run(self, *, log_preload=b'', idle_min=0.5):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name)
        cfg = home / '.config/colab-cli'
        cfg.mkdir(parents=True)
        if log_preload:
            (cfg / 'keepalive-s.log').write_bytes(log_preload)
        local = types.SimpleNamespace(endpoint='ep', token='t', url='u', token_expires_at=None)
        store = mock.Mock()
        store.get.return_value = local
        client = mock.Mock(colab_domain='https://colab.example')
        client.session.credentials = None
        client.session.get.return_value = types.SimpleNamespace(status_code=200)
        client.list_assignments.return_value = [
            types.SimpleNamespace(endpoint='ep', runtime_proxy_info=types.SimpleNamespace(token='t', url='u'))]
        fake_state = types.SimpleNamespace(store=store, client=client)
        clock = {'now': 1000.0}
        sleeps = []

        def fake_sleep(seconds):
            sleeps.append(seconds)
            clock['now'] += seconds

        slots = mock.Mock()
        slots.get.return_value.json.return_value = [{'is_processing': False, 'n_past': 5}]
        sessions_created = []

        def make_session():
            sessions_created.append(slots)
            return slots

        beats = []
        stops = []

        def apply_proxy(local_, info):
            local_.token, local_.url = info.token, info.url
            local_.token_expires_at = object()

        # main() reads <repo>/.secrets.json relative to __file__; point that at an empty scratch tree.
        fake_file = str(home / 'repo/scripts/colab_keepalive.py')
        with mock.patch.object(ka, '__file__', fake_file), \
             mock.patch('pathlib.Path.home', return_value=home), \
             mock.patch.dict(os.environ, {'TAILNET_FQDN': 'qwen.example.ts.net', 'LLAMA_API_KEY': 'k'}), \
             mock.patch.object(ka, '_import_colab', return_value=(Exception, lambda: fake_state, apply_proxy, None)), \
             mock.patch.object(ka, 'kernel_beat', side_effect=lambda s, f: beats.append(clock['now']) or (True, 'rc0')), \
             mock.patch.object(ka, '_stop_session', side_effect=lambda s, log: stops.append(s)), \
             mock.patch.object(ka.requests, 'Session', side_effect=make_session), \
             mock.patch.object(ka.requests, 'get', side_effect=AssertionError('slots must use the shared session')), \
             mock.patch.object(ka.time, 'sleep', side_effect=fake_sleep), \
             mock.patch.object(ka.time, 'monotonic', side_effect=lambda: clock['now']):
            rc = ka.main(['colab_keepalive.py', 's', str(idle_min)])
        return types.SimpleNamespace(rc=rc, cfg=cfg, store=store, client=client, slots=slots, sessions=sessions_created,
                                     beats=beats, stops=stops, sleeps=sleeps)

    def test_beats_every_cycle_pings_every_other_and_shares_one_slots_connection(self):
        run = self._run()
        self.assertEqual(run.rc, 0)
        self.assertEqual(len(run.beats), 3)  # 30 s cadence: beat, beat, beat+ping 60 s in
        self.assertEqual(run.client.session.get.call_count, 2)  # TFE pings at cycle 1 and 3
        self.assertEqual(len(run.sessions), 1)
        self.assertEqual(run.slots.get.call_count, 2)
        run.slots.close.assert_called_once()
        self.assertEqual(run.stops, ['s'])  # idle guard fired once the unchanged /slots sample aged past 0.5 min
        run.store.add.assert_not_called()  # proxy token/url never changed
        self.assertFalse((run.cfg / 'keepalive-s.json').exists())
        self.assertFalse((run.cfg / 'keepalive-s-beat.py').exists())
        self.assertTrue(all(0 <= s <= ka.BEAT_INTERVAL_S for s in run.sleeps))

    def test_oversized_log_is_rotated_on_start(self):
        run = self._run(log_preload=b'old line\n' * 40_000)
        self.assertTrue((run.cfg / 'keepalive-s.log.1').exists())
        self.assertLess((run.cfg / 'keepalive-s.log').stat().st_size, ka.LOG_MAX_BYTES)
        self.assertIn('keep_alive_started', (run.cfg / 'keepalive-s.log').read_text(encoding='utf-8'))


class CliStartupAndProbeTests(unittest.TestCase):
    def test_importing_colab_cli_loads_no_heavy_modules(self):
        probe = (
            'import sys; sys.path.insert(0, %r); import colab_cli; '
            'print(sorted(m for m in ("requests", "colab_kernel", "secrets_config", "sqlite3") if m in sys.modules))'
            % str(Path(colab_cli.__file__).resolve().parent))
        out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, '[]')

    def test_lazy_attributes_still_resolve_for_importers(self):
        import colab_kernel
        self.assertIs(colab_cli.ALL_MODELS, colab_kernel.ALL_MODELS)
        self.assertIs(colab_cli.MODELS, colab_kernel.MODELS)
        self.assertIs(colab_cli.requests, requests)
        self.assertTrue(callable(colab_cli.load_config))
        self.assertIsInstance(colab_cli.FIELDS, dict)
        with self.assertRaises(AttributeError):
            colab_cli.no_such_attribute

    def test_unknown_slug_lists_choices(self):
        with self.assertRaises(SystemExit) as ctx:
            colab_cli.main(['start', 'no-such-model'])
        self.assertIn('qwen38-huihui-v2', str(ctx.exception))

    def test_pid_alive_without_spawning_a_process(self):
        with mock.patch.object(colab_cli.subprocess, 'run', side_effect=AssertionError('tasklist must not run')):
            self.assertTrue(colab_cli._pid_alive(os.getpid()))
            self.assertFalse(colab_cli._pid_alive(0))
            self.assertFalse(colab_cli._pid_alive(-5))
            self.assertFalse(colab_cli._pid_alive(2 ** 22 + 12345))
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            self.assertTrue(colab_cli._pid_alive(child.pid))
        finally:
            child.kill()
            child.wait()
        self.assertFalse(colab_cli._pid_alive(child.pid))

    def test_pid_alive_falls_back_to_tasklist(self):
        done = subprocess.CompletedProcess([], 0, stdout='python.exe  4242 Console  1  9,000 K\n', stderr='')
        with mock.patch.object(colab_cli, '_win_pid_alive', side_effect=OSError('no ctypes')), \
             mock.patch.object(colab_cli.subprocess, 'run', return_value=done) as run:
            self.assertTrue(colab_cli._pid_alive(4242))
            self.assertFalse(colab_cli._pid_alive(77))
        self.assertEqual(run.call_args.args[0][:3], ['tasklist', '/FI', 'PID eq 77'])


class CliStopOrderingTests(unittest.TestCase):
    def _stop(self, names, *, live=(), scheduled_text='', state_sessions=()):
        log = []
        lock = threading.Lock()
        sessions_started = threading.Event()

        def note(item):
            with lock:
                log.append(item)

        def fake_sessions():
            note('sessions')
            sessions_started.set()
            time.sleep(0.2)
            return list(live)

        def fake_query():
            note('query-start')
            self.assertTrue(sessions_started.wait(5), 'schtasks query must overlap `colab sessions`')
            note('query-end')
            return scheduled_text

        def fake_stop_keepalive(session):
            time.sleep(0.05)
            note('keepalive-stopped ' + session)

        def fake_colab(*args, **kw):
            note('colab ' + ' '.join(args))
            return ' stopped %s \n' % args[2]

        out = io.StringIO()
        with mock.patch.object(colab_cli, 'sessions', side_effect=fake_sessions), \
             mock.patch.object(colab_cli, '_schtasks_query_text', side_effect=fake_query), \
             mock.patch.object(colab_cli, 'keepalive_state_sessions', return_value=list(state_sessions)), \
             mock.patch.object(colab_cli, 'stop_keepalive', side_effect=fake_stop_keepalive), \
             mock.patch.object(colab_cli, 'colab', side_effect=fake_colab), \
             mock.patch.object(colab_cli, 'usage', return_value={'balance': 5.0, 'rate': 0.0, 'active': 0}), \
             contextlib.redirect_stdout(out):
            rc = colab_cli.cmd_stop(names)
        return rc, log, out.getvalue()

    def test_stop_all_overlaps_probes_and_kills_every_daemon_before_any_colab_stop(self):
        rc, log, out = self._stop(
            ['--all'], live=['a', '?'], scheduled_text='"\\ColabKeepalive-b","N/A","Ready"\n', state_sessions=['c'])
        self.assertEqual(rc, 0)
        colab_idx = [i for i, item in enumerate(log) if item.startswith('colab ')]
        keepalive_idx = [i for i, item in enumerate(log) if item.startswith('keepalive-stopped')]
        self.assertEqual(len(keepalive_idx), 3)
        self.assertGreater(min(colab_idx), max(keepalive_idx))
        self.assertEqual([log[i] for i in colab_idx], ['colab stop -s a', 'colab stop -s b', 'colab stop -s c'])
        self.assertNotIn('colab stop -s ?', log)
        self.assertEqual(out.splitlines(), ['stopped a', 'stopped b', 'stopped c',
                                            'Active assignments: 0 | Burn rate: 0.00 CU/h | Balance: 5.00 CU'])

    def test_named_stop_touches_nothing_else(self):
        rc, log, out = self._stop(['solo'])
        self.assertEqual(rc, 0)
        self.assertEqual(log, ['keepalive-stopped solo', 'colab stop -s solo'])
        self.assertEqual(out.splitlines()[0], 'stopped solo')


class LaunchDetachTests(unittest.TestCase):
    def test_breakaway_flag_and_argv(self):
        self.assertEqual(colab_cli.CREATE_BREAKAWAY_FROM_JOB, 0x01000000)
        argv = colab_cli.keepalive_argv('qwen38-fn-nl', 45, 'breakaway')
        self.assertTrue(argv[0].lower().endswith('pythonw.exe'))
        self.assertEqual(argv[-3:], ['qwen38-fn-nl', '45', 'breakaway'])

    def test_schtasks_command_fits_tr_limit(self):
        tr = subprocess.list2cmdline(colab_cli.keepalive_argv('ka-cpu-test', 45, 'schtasks'))
        self.assertLess(len(tr), 261)
        self.assertIn('colab_keepalive.py', tr)
        self.assertIn('schtasks', tr.split())

    def test_launch_exec_sets_detached_and_parses(self):
        code = colab_cli.launch_exec_code('print("cell")\n')
        ast.parse(code)
        self.assertIn('LAUNCH_DETACHED', code)
        self.assertIn('print("cell")', code)
        self.assertNotIn('python -u', code)

    def test_notebook_first_cell_is_the_setup_cell(self):
        cell = colab_cli.notebook_first_cell('qwen38-huihui-v2')
        self.assertIn('start_new_session=True', cell)
        self.assertIn('DEBUG_VERIFY = False', cell)
        self.assertNotIn('display(HTML', cell)
        self.assertNotIn('qwenOpenWebui', cell)

    def test_detached_spawn_skips_health_wait(self):
        cell = colab_cli.notebook_first_cell('qwen38-huihui-v2')
        tree = ast.parse(cell)
        spawn = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or 'LAUNCH_DETACHED' not in ast.dump(node.test):
                continue
            body = '\n'.join(ast.get_source_segment(cell, stmt) or '' for stmt in node.body)
            if 'server_just_started = True' in body and 'while ' not in body:
                spawn = node
                body_src = body
                break
        self.assertIsNotNone(spawn, 'detached branch after llama-server spawn not found')
        self.assertIn('LAUNCH_DETACHED: llama-server pid', body_src)
        orelse = '\n'.join(ast.get_source_segment(cell, stmt) or '' for stmt in spawn.orelse)
        self.assertIn('while time.monotonic() - t0 < 600', orelse)
        self.assertIn('server_just_started = True', orelse)
        bg = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_bg_finish')
        bg_src = ast.get_source_segment(cell, bg)
        self.assertIn("os.environ.get('LAUNCH_DETACHED')", bg_src)
        self.assertIn('health()', bg_src)
        self.assertIn('threading.Thread(target=_bg_finish, daemon=True).start()', cell)

    def test_launch_health_url_is_raw_tcp_service(self):
        src = inspect.getsource(colab_cli.cmd_launch)
        self.assertIn("'http://%s:443/health'", src)
        self.assertNotIn('https://%s/health', src)

    def test_stop_all_includes_orphan_keepalive_tasks(self):
        text = (
            '"TaskName","Next Run Time","Status"\n'
            '"\\ColabKeepalive-qwen38-fn-nl","2026/11/6 23:59:00","Ready"\n'
            '"\\Microsoft\\Windows\\Foo","N/A","Ready"\n'
        )
        self.assertEqual(colab_cli.keepalive_sessions_in_schtasks_query(text), ['qwen38-fn-nl'])
        self.assertEqual(
            colab_cli.stop_targets([], ['qwen38-fn-nl'], ['qwen38-fn-nl', 'ka-cpu-test'], ['ghost']),
            ['qwen38-fn-nl', 'ka-cpu-test', 'ghost'])
        self.assertEqual(
            colab_cli.stop_targets(['qwen38-fn-nl'], [], ['other'], ['ghost']),
            ['qwen38-fn-nl'])
        self.assertEqual(ka.schtasks_task_name('qwen38-fn-nl'), colab_cli.keepalive_task_name('qwen38-fn-nl'))

    def test_cmd_launch_rejects_sanitized_placeholders_before_allocating(self):
        ensured = []
        with mock.patch.object(colab_cli, '_config', return_value={'tailnet_fqdn': 'qwen.ts.net', 'api_key': 'k'}), \
             mock.patch.object(colab_cli, 'notebook_first_cell', return_value='KEY = "<YOUR_LLAMA_API_KEY>"\n'), \
             mock.patch.object(colab_cli, '_ensure_session', side_effect=lambda *a: ensured.append(a)):
            with self.assertRaises(SystemExit) as ctx:
                colab_cli.cmd_launch('qwen38-huihui-v2', 'G4', 45)
            self.assertIn('placeholders', str(ctx.exception))
        self.assertEqual(ensured, [])

    def test_cmd_start_releases_vm_when_keepalive_fails(self):
        stopped = []
        with mock.patch.object(colab_cli, '_config', return_value={'tailnet_fqdn': 'qwen.ts.net', 'api_key': 'k'}), \
             mock.patch.object(colab_cli, 'headless_setup', return_value='print(1)\n'), \
             mock.patch.object(colab_cli, 'usage', return_value={'balance': 100.0, 'rate': 0.0, 'active': 0}), \
             mock.patch.object(colab_cli, 'sessions', return_value=['qwen38-huihui-v2']), \
             mock.patch.object(colab_cli, 'start_keepalive', side_effect=SystemExit('daemon failed')), \
             mock.patch.object(colab_cli, 'cmd_stop', side_effect=lambda names: stopped.append(names)):
            with self.assertRaises(SystemExit):
                colab_cli.cmd_start('qwen38-huihui-v2', 'G4', 45, False, False)
        self.assertEqual(stopped, [['qwen38-huihui-v2']])

    def test_reuse_active_easycolab_session_as_slug(self):
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            cli_dir = Path(tmp)
            (cli_dir / 'sessions.json').write_text(json.dumps([
                {'name': 'easycolab', 'accelerator': 'G4', 'endpoint': 'gpu-g4-s-123', 'url': 'https://example', 'token': 'tok'}
            ]), encoding='utf-8')
            with mock.patch.object(colab_cli, 'CLI_DIR', cli_dir):
                ok = colab_cli._reuse_active_session_as('qwen38-huihui-v2', 'G4', ['easycolab'])
            self.assertTrue(ok)
            saved = json.loads((cli_dir / 'sessions.json').read_text(encoding='utf-8'))
            self.assertEqual([s['name'] for s in saved], ['easycolab', 'qwen38-huihui-v2'])
            self.assertEqual(saved[1]['endpoint'], 'gpu-g4-s-123')


if __name__ == '__main__':
    unittest.main()
