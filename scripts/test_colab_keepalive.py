"""Unit tests for TFE keep-alive classification, idle detection, and launch detach."""
import ast
import inspect
import json
import subprocess
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
        cell = colab_cli.notebook_first_cell('qwen38-fn-nl')
        self.assertIn('start_new_session=True', cell)
        self.assertIn('DEBUG_VERIFY = False', cell)
        self.assertNotIn('display(HTML', cell)
        self.assertNotIn('qwenOpenWebui', cell)

    def test_detached_spawn_skips_health_wait(self):
        cell = colab_cli.notebook_first_cell('qwen38-fn-nl')
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
                colab_cli.cmd_launch('qwen38-fn', 'G4', 45)
            self.assertIn('placeholders', str(ctx.exception))
        self.assertEqual(ensured, [])

    def test_cmd_start_releases_vm_when_keepalive_fails(self):
        stopped = []
        with mock.patch.object(colab_cli, '_config', return_value={'tailnet_fqdn': 'qwen.ts.net', 'api_key': 'k'}), \
             mock.patch.object(colab_cli, 'headless_setup', return_value='print(1)\n'), \
             mock.patch.object(colab_cli, 'usage', return_value={'balance': 100.0, 'rate': 0.0, 'active': 0}), \
             mock.patch.object(colab_cli, 'sessions', return_value=['qwen38-fn']), \
             mock.patch.object(colab_cli, 'start_keepalive', side_effect=SystemExit('daemon failed')), \
             mock.patch.object(colab_cli, 'cmd_stop', side_effect=lambda names: stopped.append(names)):
            with self.assertRaises(SystemExit):
                colab_cli.cmd_start('qwen38-fn', 'G4', 45, False, False)
        self.assertEqual(stopped, [['qwen38-fn']])


if __name__ == '__main__':
    unittest.main()
