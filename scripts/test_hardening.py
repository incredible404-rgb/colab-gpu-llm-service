"""Second-round first-run hardening, all offline: config shared by the keep-alive daemon, runners export guard,
wake success, release on every launch failure, reused sessions are never stopped, unauthorized CLI, PATH fallback,
preflight, Colab-yoga activity in the beat, private notebook output, service name. No network, no Colab, no GPU."""
import ast
import contextlib
import http.server
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

import colab_cli
import colab_keepalive as ka
import colab_kernel as ck
import colab_path
import secrets_config as sc
import test_colab_keepalive as tck

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'qwen38-huihui-v2'
CFG = {
    'tailnet_fqdn': 'qwen.example.ts.net',
    'api_key': 'test-key',
    'ts_client_id': 'id',
    'ts_client_secret': 'secret',
    'local_tailscale_ip': '192.0.2.1',
}
PLACEHOLDER_FILE = {'tailnet_fqdn': 'qwen.<YOUR_TAILNET>.ts.net', 'api_key': '<YOUR_LLAMA_API_KEY>'}


def setUpModule():
    # Lifecycle integration has its own isolated tests; these offline suites must not stop the user's bridge.
    patcher = mock.patch.object(colab_cli, 'local_mcp', return_value={'ok': True})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class KeepaliveConfigTests(unittest.TestCase):
    """The daemon resolves tailnet name and key like `start` does."""

    def _slots_call(self, **kw):
        helper = tck.DaemonLoopTests()
        helper.addCleanup = self.addCleanup
        return tck.DaemonLoopTests._run(helper, **kw)

    def test_environment_value_replaces_a_placeholder_from_the_file(self):
        run = self._slots_call(secrets_json=PLACEHOLDER_FILE)
        self.assertEqual(run.rc, 0)
        url = run.slots.get.call_args_list[0][0][0]
        self.assertEqual(url, 'http://qwen.example.ts.net:443/slots')
        self.assertEqual(run.slots.get.call_args_list[0][1]['headers'], {'Authorization': 'Bearer k'})

    def test_real_file_value_is_used_as_before(self):
        run = self._slots_call(secrets_json={'tailnet_fqdn': 'qwen.mine.ts.net', 'api_key': 'file-key'})
        self.assertEqual(run.slots.get.call_args_list[0][0][0], 'http://qwen.mine.ts.net:443/slots')

    def test_nothing_usable_stops_the_daemon_without_touching_the_session(self):
        run = self._slots_call(secrets_json=PLACEHOLDER_FILE, env_extra={'TAILNET_FQDN': '', 'LLAMA_API_KEY': ''})
        self.assertEqual(run.rc, 1)
        self.assertEqual(run.stops, [])
        self.assertIn('missing_tailnet_fqdn_or_api_key', (run.cfg / 'keepalive-s.log').read_text(encoding='utf-8'))


class RunnersExportGuardTests(unittest.TestCase):
    def _export(self, config):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / 'colab_kernel_runners.json'
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                result = ck.export_yoga_runners(config, dst)
            return result, dst.exists(), out.getvalue(), err.getvalue()

    def test_placeholders_are_refused_and_named(self):
        cfg = {**CFG, 'api_key': '<YOUR_LLAMA_API_KEY>', 'tailnet_fqdn': 'qwen.<YOUR_TAILNET>.ts.net'}
        result, written, out, err = self._export(cfg)
        self.assertFalse(result)
        self.assertFalse(written)
        self.assertNotIn('Exported', out)
        self.assertIn('api_key', err)
        self.assertIn('tailnet_fqdn', err)
        self.assertNotIn('ts_client_id', err)

    def test_empty_required_value_is_refused(self):
        result, written, out, err = self._export({**CFG, 'ts_client_secret': ''})
        self.assertFalse(result)
        self.assertFalse(written)
        self.assertIn('ts_client_secret', err)

    def test_complete_config_still_exports(self):
        result, written, out, _ = self._export(CFG)
        self.assertTrue(result)
        self.assertTrue(written)
        self.assertIn('Exported Colab-yoga runners to:', out)

    def test_main_exits_non_zero_when_the_guard_refuses(self):
        placeholder_sources = {f: 'placeholder' for f in sc.FIELDS}
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(sys, 'argv', ['colab_kernel.py', SLUG]), \
                mock.patch.object(ck, 'generate_one'), \
                mock.patch.object(ck, 'load_config', return_value=({f: p for f, (_, p) in sc.FIELDS.items()}, False,
                                                                 Path(td), placeholder_sources)), \
                mock.patch.object(ck, 'yoga_runners_path', return_value=Path(td) / 'colab_kernel_runners.json'), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ck.main(), 1)


class WakeTests(unittest.TestCase):
    def _wake(self, rc, text, sessions_now=('easycolab',)):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / 'Colab-yoga.exe'
            exe.write_text('')
            with mock.patch.object(colab_cli, 'YOGA_EXE', exe), mock.patch.object(colab_cli, 'YOGA_PY', Path(td) / 'p.exe'), \
                    mock.patch.object(colab_cli, 'run_streamed', return_value=(rc, text)), \
                    mock.patch.object(colab_cli, 'sessions', return_value=list(sessions_now)), \
                    mock.patch.object(colab_cli, 'start_keepalive') as ka_start, quiet():
                result = colab_cli.cmd_yoga(['wake', '--idle-min', '30'])
        return result, ka_start

    def test_wake_url_line_means_success(self):
        result, ka_start = self._wake(0, 'uploading\nWAKE_URL: https://x\n')
        self.assertEqual(result, 0)
        ka_start.assert_called_once_with('easycolab', 30)

    def test_exit_zero_without_wake_url_is_a_failure_and_starts_no_keepalive(self):
        result, ka_start = self._wake(0, 'WAKE_ERROR: no auth\n')
        self.assertNotEqual(result, 0)
        ka_start.assert_not_called()

    def test_non_zero_exit_is_a_failure_even_with_a_wake_url(self):
        result, ka_start = self._wake(3, 'WAKE_URL: https://x\n')
        self.assertEqual(result, 3)
        ka_start.assert_not_called()

    def test_wake_succeeded_parsing(self):
        self.assertTrue(colab_cli.wake_succeeded('a\n  WAKE_URL: u\nb'))
        self.assertFalse(colab_cli.wake_succeeded('see WAKE_URL: in the docs'))
        self.assertFalse(colab_cli.wake_succeeded(''))


class LaunchReleaseTests(unittest.TestCase):
    """cmd_launch releases a runtime it allocated on every failure; a reused session is left running."""

    def _launch(self, *, created=True, exec_results=(0, 0), exec_error=None, health='ok', keepalive_running=False):
        stopped, daemons = [], []
        results = list(exec_results)
        clock = {'t': 0.0}

        def fake_monotonic():
            clock['t'] += 2000.0 if health == 'timeout' else 0.01
            return clock['t']

        fake_time = types.SimpleNamespace(monotonic=fake_monotonic, sleep=lambda s: None, time=time.time,
                                          strftime=time.strftime)

        def fake_exec(slug, code, timeout_s):
            if exec_error is not None and len(results) == len(exec_results):
                results.pop(0)
                raise exec_error
            return results.pop(0) if results else 1

        response = mock.Mock(status_code=200 if health == 'ok' else 503)
        with mock.patch.object(colab_cli, '_config', return_value={'tailnet_fqdn': 'q.ts.net', 'api_key': 'k'}), \
                mock.patch.object(colab_cli, 'notebook_first_cell', return_value='print(1)\n'), \
                mock.patch.object(colab_cli, 'preflight'), \
                mock.patch.object(colab_cli, 'usage', return_value={'balance': 5.0, 'rate': 0.0, 'active': 0}), \
                mock.patch.object(colab_cli, '_ensure_session', return_value=created), \
                mock.patch.object(colab_cli, '_keepalive_running', return_value=keepalive_running), \
                mock.patch.object(colab_cli, 'start_keepalive'), \
                mock.patch.object(colab_cli, 'stop_keepalive', side_effect=daemons.append), \
                mock.patch.object(colab_cli, 'exec_stream', side_effect=fake_exec), \
                mock.patch.object(colab_cli, 'cmd_stop', side_effect=lambda names: stopped.append(names)), \
                mock.patch.object(colab_cli, 'time', fake_time), \
                mock.patch('requests.get', return_value=response), quiet():
            try:
                rc = colab_cli.cmd_launch(SLUG, 'G4', 45)
                raised = None
            except BaseException as err:  # noqa: BLE001 - the tests look at what escaped
                rc, raised = None, err
        return rc, raised, stopped, daemons

    def test_success_releases_nothing(self):
        rc, raised, stopped, _ = self._launch()
        self.assertEqual((rc, raised, stopped), (0, None, []))

    def test_failed_exec_releases(self):
        rc, raised, stopped, _ = self._launch(exec_results=(7,))
        self.assertEqual((rc, stopped), (1, [[SLUG]]))

    def test_dead_kernel_releases(self):
        rc, raised, stopped, _ = self._launch(exec_results=(0, 1, 1, 1))
        self.assertEqual((rc, stopped), (1, [[SLUG]]))

    def test_health_timeout_releases(self):
        rc, raised, stopped, _ = self._launch(health='timeout')
        self.assertEqual((rc, stopped), (1, [[SLUG]]))

    def test_ctrl_c_during_exec_releases_and_propagates(self):
        rc, raised, stopped, _ = self._launch(exec_error=KeyboardInterrupt())
        self.assertIsInstance(raised, KeyboardInterrupt)
        self.assertEqual(stopped, [[SLUG]])

    def test_reused_session_is_not_stopped_but_its_new_daemon_is(self):
        rc, raised, stopped, daemons = self._launch(created=False, exec_results=(7,))
        self.assertEqual((rc, stopped, daemons), (1, [], [SLUG]))

    def test_reused_session_with_a_running_daemon_keeps_both(self):
        rc, raised, stopped, daemons = self._launch(created=False, exec_results=(7,), keepalive_running=True)
        self.assertEqual((rc, stopped, daemons), (1, [], []))


class StartReusedSessionTests(unittest.TestCase):
    def _start(self, exec_rc):
        stopped = []
        with mock.patch.object(colab_cli, '_config', return_value={'tailnet_fqdn': 'q.ts.net', 'api_key': 'k'}), \
                mock.patch.object(colab_cli, 'headless_setup', return_value='print(1)\n'), \
                mock.patch.object(colab_cli, 'preflight'), \
                mock.patch.object(colab_cli, 'usage', return_value={'balance': 5.0, 'rate': 0.0, 'active': 1}), \
                mock.patch.object(colab_cli, '_ensure_session', return_value=False), \
                mock.patch.object(colab_cli, '_keepalive_running', return_value=True), \
                mock.patch.object(colab_cli, 'start_keepalive'), \
                mock.patch.object(colab_cli, 'exec_stream', return_value=exec_rc), \
                mock.patch.object(colab_cli, 'remote_check', return_value=False), \
                mock.patch.object(colab_cli, 'cmd_stop', side_effect=lambda names: stopped.append(names)), quiet():
            rc = colab_cli.cmd_start(SLUG, 'G4', 45, False, False)
        return rc, stopped

    def test_failed_start_on_a_reused_session_reports_and_stops_nothing(self):
        rc, stopped = self._start(1)
        self.assertEqual((rc, stopped), (1, []))
        rc, stopped = self._start(0)  # exec ok, remote check fails
        self.assertEqual((rc, stopped), (1, []))


class StopScopeTests(unittest.TestCase):
    def _stop(self, names, active, unnamed=None, usages=None):
        """`unnamed` is the list of "[?]" endpoint lists successive `colab sessions` calls return."""
        leftovers, released = [], []
        listings = [list(x) for x in (unnamed if unnamed is not None else [[]])]
        usage_calls = list(usages or [])

        def fake_unnamed():
            return listings.pop(0) if len(listings) > 1 else listings[0]

        def fake_usage():
            if usage_calls:
                return usage_calls.pop(0) if len(usage_calls) > 1 else usage_calls[0]
            return {'balance': 1.0, 'rate': 10.0, 'active': active}

        with mock.patch.object(colab_cli, 'stop_keepalive'), \
                mock.patch.object(colab_cli, 'colab', return_value='stopped\n'), \
                mock.patch.object(colab_cli, 'sessions', return_value=['?']), \
                mock.patch.object(colab_cli, '_schtasks_query_text', return_value=''), \
                mock.patch.object(colab_cli, 'keepalive_state_sessions', return_value=[]), \
                mock.patch.object(colab_cli.time, 'sleep'), \
                mock.patch.object(colab_cli, 'usage', side_effect=fake_usage), \
                mock.patch.object(colab_cli, 'unnamed_endpoints', side_effect=fake_unnamed), \
                mock.patch.object(colab_cli, 'unassign_endpoints',
                                  side_effect=lambda eps: released.append(list(eps)) or True), \
                mock.patch.object(colab_cli.subprocess, 'run', side_effect=AssertionError('no subprocess in stop')), \
                mock.patch.object(colab_cli, '_release_leftovers',
                                  side_effect=lambda u: leftovers.append(u) or {**u, 'active': 0, 'rate': 0.0}), \
                quiet() as out:
            rc = colab_cli.cmd_stop(names)
        return rc, leftovers, released, out.getvalue()

    def test_single_name_never_escalates_to_releasing_the_account(self):
        rc, leftovers, released, out = self._stop(['solo'], active=1)
        self.assertEqual(leftovers, [])
        self.assertEqual(released, [])  # only named sessions are left: nothing to sweep
        self.assertEqual(rc, 1)
        self.assertIn('local name', out)
        self.assertIn('stop --all', out)

    def test_single_name_releases_only_confirmed_unnamed_endpoints(self):
        done = {'balance': 1.0, 'rate': 0.0, 'active': 0}
        busy = {'balance': 1.0, 'rate': 10.0, 'active': 2}
        # Listings: first look, recheck after the pause, post-release verification. ghost-b appears only once.
        rc, leftovers, released, out = self._stop(
            ['solo'], active=2, unnamed=[['ghost-a', 'ghost-b'], ['ghost-a'], []], usages=[busy] * 5 + [done])
        self.assertEqual(released, [['ghost-a']])
        self.assertEqual(leftovers, [])
        self.assertEqual(rc, 0)
        self.assertIn('ghost-a', out)
        self.assertNotIn('ghost-b', out)

    def test_named_session_survives_a_single_name_stop(self):
        # Colab-yoga's easycolab keeps one assignment alive; the endpoint list holds only unnamed ones.
        busy = {'balance': 1.0, 'rate': 5.0, 'active': 1}
        rc, leftovers, released, out = self._stop(['solo'], active=1, unnamed=[[], []], usages=[busy])
        self.assertEqual((released, leftovers, rc), ([], [], 1))

    def test_a_ghost_that_cannot_be_released_is_reported_with_the_manual_route(self):
        busy = {'balance': 1.0, 'rate': 10.0, 'active': 1}
        rc, _, released, out = self._stop(['solo'], active=1, unnamed=[['g'], ['g'], ['g']], usages=[busy])
        self.assertEqual(released, [['g']])
        self.assertEqual(rc, 1)
        self.assertIn('could not be released', out)
        self.assertIn('Manage sessions', out)

    def test_stop_all_still_releases_leftovers(self):
        rc, leftovers, released, _ = self._stop(['--all'], active=1)
        self.assertEqual(len(leftovers), 1)
        self.assertEqual(released, [])
        self.assertEqual(rc, 0)


class UnnamedReleaseTests(unittest.TestCase):
    def test_release_unnamed_waits_then_confirms_before_unassigning(self):
        pauses, released = [], []
        listings = [['a', 'b'], ['b', 'c']]
        with mock.patch.object(colab_cli, 'unnamed_endpoints', side_effect=lambda: listings.pop(0)), \
                mock.patch.object(colab_cli, 'unassign_endpoints', side_effect=lambda eps: released.append(list(eps)) or True), \
                quiet():
            got = colab_cli.release_unnamed(sleep=pauses.append)
        self.assertEqual(got, ['b'])
        self.assertEqual(released, [['b']])
        self.assertEqual(pauses, [colab_cli.UNNAMED_RECHECK_S])

    def test_nothing_unnamed_means_no_pause_and_no_unassign(self):
        pauses = []
        with mock.patch.object(colab_cli, 'unnamed_endpoints', return_value=[]), \
                mock.patch.object(colab_cli, 'unassign_endpoints', side_effect=AssertionError('nothing to release')):
            self.assertEqual(colab_cli.release_unnamed(sleep=pauses.append), [])
        self.assertEqual(pauses, [])

    def test_endpoints_come_only_from_unnamed_lines(self):
        out = ('[?] gpu-ghost | Hardware: G4 | Shape: High-RAM | Variant: GPU\n'
               '[easycolab] gpu-yoga | Hardware: G4 | Shape: High-RAM | Variant: GPU\n'
               '[qwen38-fn] m-s-x | Hardware: CPU | Shape: Standard | Variant: DEFAULT\n')
        with mock.patch.object(colab_cli, 'colab', return_value=out):
            self.assertEqual(colab_cli.unnamed_endpoints(), ['gpu-ghost'])
            self.assertEqual(colab_cli.sessions(), ['?', 'easycolab', 'qwen38-fn'])

    def test_unassign_runs_the_cli_client_per_endpoint_without_stop_all(self):
        with tempfile.TemporaryDirectory() as td:
            py = Path(td) / 'python.exe'
            py.write_text('')
            seen = {}

            def fake_run(argv, **kw):
                seen['argv'], seen['kw'] = argv, kw
                return subprocess.CompletedProcess(argv, 0, 'released gpu-ghost\n', '')

            with mock.patch.object(colab_cli, 'CLI_PYTHON', py), \
                    mock.patch.object(colab_cli.subprocess, 'run', side_effect=fake_run), quiet():
                self.assertTrue(colab_cli.unassign_endpoints(['gpu-ghost']))
        argv = seen['argv']
        self.assertEqual(argv[1:3], ['-I', '-c'])
        self.assertEqual(argv[4:], ['gpu-ghost'])
        self.assertIn('state.client.unassign(ep)', argv[3])
        self.assertNotIn('stop_all', ' '.join(argv))
        self.assertIs(seen['kw']['stdin'], subprocess.DEVNULL)

    def test_unassign_without_the_cli_interpreter_does_nothing(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(colab_cli, 'CLI_PYTHON', Path(td) / 'missing.exe'), \
                mock.patch.object(colab_cli.subprocess, 'run', side_effect=AssertionError('must not run')), quiet():
            self.assertFalse(colab_cli.unassign_endpoints(['x']))
            self.assertTrue(colab_cli.unassign_endpoints([]))

    def test_embedded_unassign_code_is_valid_python(self):
        ast.parse(colab_cli._UNASSIGN_CODE)


class ColabInvocationTests(unittest.TestCase):
    def test_stdin_is_closed_and_the_resolved_path_is_used(self):
        seen = {}

        def fake_run(argv, **kw):
            seen['argv'], seen['kw'] = argv, kw
            return subprocess.CompletedProcess(argv, 0, 'ok', '')

        with mock.patch.object(colab_cli, 'find_colab', return_value='C:/tools/colab.exe'), \
                mock.patch.object(colab_cli.subprocess, 'run', side_effect=fake_run):
            self.assertEqual(colab_cli.colab('usage'), 'ok')
        self.assertEqual(seen['argv'], ['C:/tools/colab.exe', 'usage'])
        self.assertIs(seen['kw']['stdin'], subprocess.DEVNULL)

    def test_unauthorized_cli_becomes_one_clear_line_with_the_link(self):
        out = ('\nTo authorize colab-cli, visit this URL in any browser:\n\n'
               '  https://accounts.google.com/o/oauth2/auth?client_id=abc&state=s\n\n'
               'Enter the authorization code: Traceback ... EOFError')
        with mock.patch.object(colab_cli.subprocess, 'run',
                               return_value=subprocess.CompletedProcess([], 1, '', out)):
            with self.assertRaises(SystemExit) as cm:
                colab_cli.colab('usage')
        msg = str(cm.exception)
        self.assertEqual(len(msg.splitlines()), 1)
        self.assertIn('not authorized', msg)
        self.assertIn('SKILL.md section 0', msg)
        self.assertIn('https://accounts.google.com/o/oauth2/auth?client_id=abc&state=s', msg)

    def test_timeout_with_the_authorization_prompt_is_also_reported_as_unauthorized(self):
        err = subprocess.TimeoutExpired(['colab'], 180, output=b'', stderr=b'To authorize colab-cli, visit ...')
        with mock.patch.object(colab_cli.subprocess, 'run', side_effect=err):
            with self.assertRaises(SystemExit) as cm:
                colab_cli.colab('sessions')
        self.assertIn('not authorized', str(cm.exception))

    def test_plain_timeout_and_missing_binary_are_one_line_errors(self):
        with mock.patch.object(colab_cli.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['colab'], 5)):
            with self.assertRaises(SystemExit) as cm:
                colab_cli.colab('usage', timeout=5)
        self.assertIn('timed out after 5s', str(cm.exception))
        with mock.patch.object(colab_cli.subprocess, 'run', side_effect=FileNotFoundError()):
            with self.assertRaises(SystemExit) as cm:
                colab_cli.colab('usage')
        self.assertIn('python -m uv tool install', str(cm.exception))

    def test_a_failing_command_keeps_its_output(self):
        with mock.patch.object(colab_cli.subprocess, 'run',
                               return_value=subprocess.CompletedProcess([], 2, 'x', 'boom')):
            with self.assertRaises(SystemExit) as cm:
                colab_cli.colab('stop', '-s', 'a')
        self.assertIn('boom', str(cm.exception))


class FindColabTests(unittest.TestCase):
    def setUp(self):
        colab_path.find_colab.cache_clear()
        self.addCleanup(colab_path.find_colab.cache_clear)

    def _find(self, *, which=None, home=None, env=None, uv_dir=None):
        env = {k: v for k, v in os.environ.items() if k not in ('UV_TOOL_BIN_DIR', 'XDG_BIN_HOME')} | (env or {})
        with mock.patch.object(colab_path.shutil, 'which', side_effect=lambda n: (which or {}).get(n)), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(colab_path.Path, 'home', return_value=Path(home or tempfile.gettempdir()) / 'nohome'), \
                mock.patch.object(colab_path, '_uv_bin_dir', return_value=uv_dir):
            return colab_path.find_colab()

    def test_path_wins(self):
        self.assertEqual(self._find(which={'colab': 'C:/on/path/colab.exe'}), 'C:/on/path/colab.exe')

    def test_falls_back_to_the_uv_bin_directory_under_home(self):
        with tempfile.TemporaryDirectory() as td:
            bin_dir = Path(td) / 'nohome' / '.local' / 'bin'
            bin_dir.mkdir(parents=True)
            exe = bin_dir / ('colab.exe' if os.name == 'nt' else 'colab')
            exe.write_text('')
            self.assertEqual(self._find(home=td), str(exe))

    def test_uv_tool_bin_dir_variable_and_uv_query_are_consulted(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as other:
            exe = Path(td) / ('colab.exe' if os.name == 'nt' else 'colab')
            exe.write_text('')
            self.assertEqual(self._find(env={'UV_TOOL_BIN_DIR': td}), str(exe))
            colab_path.find_colab.cache_clear()
            exe2 = Path(other) / ('colab.exe' if os.name == 'nt' else 'colab')
            exe2.write_text('')
            self.assertEqual(self._find(uv_dir=other), str(exe2))

    def test_nothing_found_returns_the_plain_name(self):
        self.assertEqual(self._find(), 'colab')

    def test_uv_query_parses_the_directory_line(self):
        done = subprocess.CompletedProcess([], 0, stdout='C:\\Users\\u\\.local\\bin\n')
        with mock.patch.object(colab_path.shutil, 'which', return_value='uv'), \
                mock.patch.object(colab_path.subprocess, 'run', return_value=done) as run:
            self.assertEqual(colab_path._uv_bin_dir(), 'C:\\Users\\u\\.local\\bin')
        self.assertEqual(run.call_args[0][0], ['uv', 'tool', 'dir', '--bin'])

    def test_keepalive_resolves_colab_the_same_way(self):
        self.assertIs(ka.find_colab, colab_path.find_colab)
        with mock.patch.object(ka, 'find_colab', return_value='C:/x/colab.exe'):
            self.assertEqual(ka.beat_argv('s', 'f.py')[0], 'C:/x/colab.exe')
            with mock.patch.object(ka.subprocess, 'run') as run:
                ka._stop_session('s', lambda m: None)
            self.assertEqual(run.call_args[0][0], ['C:/x/colab.exe', 'stop', '-s', 's'])


class PreflightTests(unittest.TestCase):
    CONFIG = {'ts_client_id': 'the-id', 'ts_client_secret': 'the-secret', 'tailnet_fqdn': 'q.ts.net', 'api_key': 'k'}

    def _preflight(self, *, pythonw=True, ts_state='Running', ts_exe='C:/ts/tailscale.exe', oauth=(200, {'access_token': 't'}),
                   oauth_error=None, skip=False):
        with tempfile.TemporaryDirectory() as td:
            fake_pythonw = Path(td) / 'pythonw.exe'
            if pythonw:
                fake_pythonw.write_text('')
            status = subprocess.CompletedProcess([], 0, stdout=json.dumps({'BackendState': ts_state}))
            resp = mock.Mock(status_code=oauth[0])
            resp.json.return_value = oauth[1]
            import requests
            post = mock.Mock(return_value=resp, side_effect=oauth_error)
            with mock.patch.object(colab_cli, 'CLI_PYTHONW', fake_pythonw), \
                    mock.patch.object(colab_cli, '_tailscale_exe', return_value=ts_exe), \
                    mock.patch.object(colab_cli.subprocess, 'run', return_value=status) as run, \
                    mock.patch.object(requests, 'post', post), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                try:
                    colab_cli.preflight(self.CONFIG, skip)
                    error = None
                except SystemExit as err:
                    error = str(err)
        return error, post, run, out.getvalue()

    def test_everything_fine_posts_the_documented_oauth_request(self):
        error, post, run, out = self._preflight()
        self.assertIsNone(error)
        self.assertEqual(post.call_args[0][0], 'https://api.tailscale.com/api/v2/oauth/token')
        self.assertEqual(post.call_args[1]['data'], {'client_id': 'the-id', 'client_secret': 'the-secret'})
        self.assertEqual(run.call_args[0][0], ['C:/ts/tailscale.exe', 'status', '--json'])

    def test_missing_interpreter_is_reported_with_the_install_command(self):
        error, *_ = self._preflight(pythonw=False)
        self.assertIn('nothing was allocated', error)
        self.assertIn('pythonw.exe', error)
        self.assertIn('python -m uv tool install', error)

    def test_tailscale_offline_or_missing(self):
        error, *_ = self._preflight(ts_state='Stopped')
        self.assertIn('Stopped', error)
        error, *_ = self._preflight(ts_exe=None)
        self.assertIn('tailscale CLI not found', error)

    def test_oauth_rejection_fails_without_printing_the_secret(self):
        error, *_ = self._preflight(oauth=(401, {'message': 'invalid client'}))
        self.assertIn('HTTP 401', error)
        self.assertNotIn('the-secret', error)
        self.assertNotIn('the-id', error)

    def test_network_error_from_this_pc_is_a_warning_not_a_failure(self):
        import requests
        error, _, _, out = self._preflight(oauth_error=requests.ConnectionError('offline'))
        self.assertIsNone(error)
        self.assertIn('not treated as a failure', out)

    def test_all_problems_are_listed_together(self):
        error, *_ = self._preflight(pythonw=False, ts_state='NeedsLogin', oauth=(403, {}))
        self.assertEqual(error.count('\n  - '), 3)

    def test_skip_does_nothing(self):
        error, post, run, out = self._preflight(pythonw=False, ts_state='Stopped', skip=True)
        self.assertIsNone(error)
        post.assert_not_called()
        run.assert_not_called()
        self.assertIn('skipped', out)

    def test_failed_preflight_stops_start_and_launch_before_anything_is_billed(self):
        boom = SystemExit('preflight failed, nothing was allocated')
        with mock.patch.object(colab_cli, '_config', return_value=self.CONFIG), \
                mock.patch.object(colab_cli, 'headless_setup', return_value='x'), \
                mock.patch.object(colab_cli, 'notebook_first_cell', return_value='x'), \
                mock.patch.object(colab_cli, 'preflight', side_effect=boom) as pre, \
                mock.patch.object(colab_cli, 'usage', side_effect=AssertionError('no colab call')), \
                mock.patch.object(colab_cli, '_ensure_session', side_effect=AssertionError('no allocation')):
            with self.assertRaises(SystemExit):
                colab_cli.cmd_start(SLUG, 'G4', 45, False, False)
            with self.assertRaises(SystemExit):
                colab_cli.cmd_launch(SLUG, 'G4', 45)
        self.assertEqual(pre.call_count, 2)

    def test_skip_flag_reaches_preflight(self):
        with mock.patch.object(colab_cli, 'cmd_start', return_value=0) as start, \
                mock.patch.object(colab_cli, 'cmd_launch', return_value=0) as launch:
            colab_cli.main(['start', SLUG, '--skip-preflight'])
            colab_cli.main(['launch', SLUG])
        self.assertTrue(start.call_args[0][-1])
        self.assertFalse(launch.call_args[0][-1])


class BeatCodeTests(unittest.TestCase):
    """The code that runs inside the VM, executed here against a stand-in for Colab-yoga's :7860."""

    def _run_code(self, port):
        with tempfile.TemporaryDirectory() as td:
            code = ka.BEAT_CODE.replace('/content/.keepalive_beat', (Path(td) / 'beat').as_posix()).replace('7860', str(port))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(code, '<beat>', 'exec'), {})
            self.assertTrue((Path(td) / 'beat').exists())
        return out.getvalue()

    def _serve(self, payload):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_beat_prints_the_three_yoga_fields(self):
        port = self._serve({'idle_remaining_seconds': 1200, 'is_downloading': True, 'is_loading_vram': False,
                            'gpu_name': 'ignored'})
        text = self._run_code(port)
        self.assertEqual(ka.parse_yoga_status(text),
                         {'idle_remaining_seconds': 1200, 'is_downloading': True, 'is_loading_vram': False})

    def test_without_yoga_the_beat_prints_nothing_and_does_not_raise(self):
        with socket_free_port() as port:
            started = time.monotonic()
            text = self._run_code(port)
        self.assertEqual(text, '')
        self.assertLess(time.monotonic() - started, 5)

    def test_beat_defines_only_one_global_in_the_kernel(self):
        ns = {}
        with tempfile.TemporaryDirectory() as td:
            code = ka.BEAT_CODE.replace('/content/.keepalive_beat', (Path(td) / 'b').as_posix())
            exec(compile(code, '<beat>', 'exec'), ns)
        self.assertEqual({k for k in ns if not k.startswith('__')}, {'_keepalive_beat'})

    def test_parse_from_text_and_from_notebook_outputs(self):
        line = 'YOGA_STATUS {"idle_remaining_seconds": 5, "is_downloading": false, "is_loading_vram": true}'
        want = {'idle_remaining_seconds': 5, 'is_downloading': False, 'is_loading_vram': True}
        self.assertEqual(ka.parse_yoga_status('noise\n' + line + '\n'), want)
        self.assertEqual(ka.parse_yoga_status([{'output_type': 'stream', 'name': 'stdout', 'text': line + '\n'}]), want)
        self.assertEqual(ka.parse_yoga_status([{'output_type': 'stream', 'text': ['x\n', line + '\n']}]), want)
        for bad in ('', None, 'YOGA_STATUS not json', 'YOGA_STATUS [1]', [{'output_type': 'error'}]):
            self.assertIsNone(ka.parse_yoga_status(bad))


@contextlib.contextmanager
def socket_free_port():
    import socket
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    yield port


class YogaActivityTests(unittest.TestCase):
    def test_download_or_vram_load_is_activity(self):
        self.assertTrue(ka.yoga_activity({'is_downloading': True, 'idle_remaining_seconds': 100}, None)[0])
        self.assertTrue(ka.yoga_activity({'is_loading_vram': True}, None)[0])

    def test_idle_countdown_going_up_is_activity_going_down_is_not(self):
        active, remaining = ka.yoga_activity({'idle_remaining_seconds': 1500}, 1470)
        self.assertTrue(active)
        self.assertEqual(remaining, 1500)
        self.assertFalse(ka.yoga_activity({'idle_remaining_seconds': 1440}, 1470)[0])
        self.assertFalse(ka.yoga_activity({'idle_remaining_seconds': 1471}, 1470)[0])  # int rounding slack

    def test_first_reading_and_disabled_timer_are_not_activity(self):
        self.assertEqual(ka.yoga_activity({'idle_remaining_seconds': 1500}, None), (False, 1500))
        self.assertFalse(ka.yoga_activity({'idle_remaining_seconds': -1}, 100)[0])
        self.assertFalse(ka.yoga_activity({'idle_remaining_seconds': 600}, -1)[0])

    def test_no_reading_changes_nothing_and_forgets_the_countdown(self):
        self.assertEqual(ka.yoga_activity(None, 500), (False, None))
        self.assertEqual(ka.yoga_activity({'idle_remaining_seconds': 'x', 'is_downloading': None}, 5), (False, None))


def yoga_line(remaining=1800, downloading=False, loading=False):
    return 'YOGA_STATUS ' + json.dumps({'idle_remaining_seconds': remaining, 'is_downloading': downloading,
                                        'is_loading_vram': loading})


class DaemonYogaTests(unittest.TestCase):
    def _run(self, **kw):
        helper = tck.DaemonLoopTests()
        helper.addCleanup = self.addCleanup
        return tck.DaemonLoopTests._run(helper, **kw)

    def test_without_yoga_the_idle_guard_is_unchanged(self):
        run = self._run()
        self.assertEqual(len(run.beats), 3)
        self.assertEqual(run.stops, ['s'])

    def test_a_download_in_progress_postpones_the_idle_stop(self):
        run = self._run(beat_fn=lambda n: (True, 'rc0', yoga_line(downloading=True) if n <= 8 else ''))
        self.assertEqual(run.stops, ['s'])
        self.assertGreater(len(run.beats), 8)
        self.assertIn('yoga_activity downloading=True', (run.cfg / 'keepalive-s.log').read_text(encoding='utf-8'))

    def test_an_extended_idle_countdown_postpones_the_idle_stop(self):
        # idle limit 1 min, so the baseline stops at beat 3. Readings 1790, 1760, then "extend" back to 1800 at beat 3.
        base = self._run(idle_min=1.0)
        self.assertEqual(len(base.beats), 3)
        readings = {1: 1790, 2: 1760, 3: 1800}
        run = self._run(idle_min=1.0, beat_fn=lambda n: (True, 'rc0', yoga_line(readings[n]) if n in readings else ''))
        self.assertEqual(run.stops, ['s'])
        self.assertEqual(len(run.beats), 5)

    def test_a_download_postpones_the_server_never_up_stop_too(self):
        base = self._run(slots_error=ConnectionError('down'))
        self.assertEqual(base.stops, ['s'])
        run = self._run(slots_error=ConnectionError('down'),
                        beat_fn=lambda n: (True, 'rc0', yoga_line(downloading=True) if n <= 8 else ''))
        self.assertEqual(run.stops, ['s'])
        self.assertGreater(len(run.beats), len(base.beats))

    def test_in_process_beat_reads_yoga_from_the_outputs_and_exec_beat_from_stdout(self):
        line = yoga_line(downloading=True) + '\n'
        state = types.SimpleNamespace(store=mock.Mock(), get_session=mock.Mock(return_value=types.SimpleNamespace(
            url='u', token='t', kernel_id='k', session_id='s')))

        class RT:
            def __init__(self, *a, **k):
                pass

            def execute_code(self, code, timeout=None):
                return [{'output_type': 'stream', 'name': 'stdout', 'text': line}]

            def stop(self):
                pass

        beater = ka.InProcessBeat(state, 's', runtime_cls=RT, exec_beat=lambda *a: (True, 'rc0', line))
        self.assertEqual(beater.beat('f'), (True, 'inproc'))
        self.assertTrue(beater.yoga['is_downloading'])
        broken = ka.InProcessBeat(state, 's', runtime_cls=mock.Mock(side_effect=OSError('x')),
                                  exec_beat=lambda *a: (True, 'rc0', line))
        ok, detail = broken.beat('f')
        self.assertTrue(ok)
        self.assertIn('; exec rc0', detail)
        self.assertTrue(broken.yoga['is_downloading'])  # the fallback beat carried it
        failed = ka.InProcessBeat(state, 's', runtime_cls=mock.Mock(side_effect=OSError('x')),
                                  exec_beat=lambda *a: (False, 'rc1', line))
        failed.beat('f')
        self.assertIsNone(failed.yoga)  # a failed beat tells nothing


class PrivateNotebooksTests(unittest.TestCase):
    def test_default_output_dir_depends_on_sanitized(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / '.secrets.json'
            path.write_text('{}', encoding='utf-8')
            with mock.patch.object(sc, 'SECRETS_FILE', path):
                _, _, private_dir, _ = sc.load_config([])
                _, _, clean_dir, _ = sc.load_config(['--sanitized'])
                _, _, explicit, _ = sc.load_config(['--output-dir', td])
        self.assertEqual(private_dir, ROOT / 'notebooks' / 'private')
        self.assertEqual(clean_dir, ROOT / 'notebooks')
        self.assertEqual(explicit, Path(td).resolve())

    def test_loading_the_config_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(sc, 'PRIVATE_NOTEBOOKS_DIR', Path(td) / 'private'), \
                    mock.patch.object(sc, 'SECRETS_FILE', Path(td) / 'none.json'):
                sc.load_config([])
            self.assertFalse((Path(td) / 'private').exists())

    def test_private_directory_is_git_ignored(self):
        lines = (ROOT / '.gitignore').read_text(encoding='utf-8').splitlines()
        self.assertIn('notebooks/private/', lines)

    def test_generate_one_creates_the_directory_and_still_mirrors_a_default_private_build(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as drive:
            out = Path(td) / 'notebooks' / 'private'
            cfg = {**CFG, **{s['drive_key']: 'id' for s in ck.ALL_MODELS.values()}, 'notebooks_dir': drive}
            with mock.patch.object(ck, 'ROOT_DIR', Path(td)), \
                    mock.patch.object(ck, 'load_config', return_value=(cfg, False, out, {})), \
                    mock.patch.object(ck, 'report'), mock.patch.object(ck, 'guard_sync'), \
                    mock.patch.object(ck, 'sync_requested'), quiet():
                ck.generate_one(SLUG, [])
            name = ck.ALL_MODELS[SLUG]['notebook']
            self.assertTrue((out / name).exists())
            self.assertEqual((Path(drive) / name).read_bytes(), (out / name).read_bytes())

    def test_a_custom_output_dir_is_not_mirrored(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as drive:
            cfg = {**CFG, **{s['drive_key']: 'id' for s in ck.ALL_MODELS.values()}, 'notebooks_dir': drive}
            with mock.patch.object(ck, 'ROOT_DIR', Path(td)), \
                    mock.patch.object(ck, 'load_config', return_value=(cfg, False, Path(td) / 'elsewhere', {})), \
                    mock.patch.object(ck, 'report'), mock.patch.object(ck, 'guard_sync'), \
                    mock.patch.object(ck, 'sync_requested'), quiet():
                ck.generate_one(SLUG, [])
            self.assertEqual(list(Path(drive).iterdir()), [])

    def test_launch_prefers_the_private_build_and_falls_back_to_notebooks(self):
        name = ck.ALL_MODELS[SLUG]['notebook']
        with tempfile.TemporaryDirectory() as td:
            nb, private = Path(td) / 'notebooks', Path(td) / 'notebooks' / 'private'
            private.mkdir(parents=True)
            with mock.patch.object(colab_cli, 'NOTEBOOKS', nb), mock.patch.object(colab_cli, 'PRIVATE_NOTEBOOKS', private):
                self.assertEqual(colab_cli.notebook_path(SLUG), private / name)  # nothing yet: private is the hint
                (nb / name).write_text('{}', encoding='utf-8')
                self.assertEqual(colab_cli.notebook_path(SLUG), nb / name)  # existing layout keeps working
                (private / name).write_text('{}', encoding='utf-8')
                self.assertEqual(colab_cli.notebook_path(SLUG), private / name)

    def test_launch_reads_the_cell_from_the_private_build(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = {**CFG, **{s['drive_key']: 'id' for s in ck.ALL_MODELS.values()}}
            private = Path(td) / 'notebooks' / 'private'
            with mock.patch.object(ck, 'ROOT_DIR', Path(td)), \
                    mock.patch.object(ck, 'load_config', return_value=(cfg, False, private, {})), \
                    mock.patch.object(ck, 'report'), mock.patch.object(ck, 'guard_sync'), \
                    mock.patch.object(ck, 'sync_requested'), quiet():
                ck.generate_one(SLUG, [])
            with mock.patch.object(colab_cli, 'NOTEBOOKS', Path(td) / 'notebooks'), \
                    mock.patch.object(colab_cli, 'PRIVATE_NOTEBOOKS', private):
                cell = colab_cli.notebook_first_cell(SLUG)
        self.assertIn('qwen.example.ts.net', cell)
        self.assertIn('DEBUG_VERIFY = False', cell)

    def test_missing_notebook_tells_how_to_build_it(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(colab_cli, 'NOTEBOOKS', Path(td)), mock.patch.object(colab_cli, 'PRIVATE_NOTEBOOKS', Path(td) / 'p'):
                with self.assertRaises(SystemExit) as cm:
                    colab_cli.notebook_first_cell(SLUG)
        self.assertIn('colab_kernel.py ' + SLUG, str(cm.exception))


class ServiceNameAndSecretsTests(unittest.TestCase):
    def _cell(self, config):
        nb = ck.build_notebook_dict(ck.MODELS[SLUG], config)
        return ''.join(next(c['source'] for c in nb['cells'] if c['metadata'].get('id') == 'persistent-setup'))

    def test_default_service_is_qwen(self):
        self.assertIn("SERVICE = 'svc:qwen'\n", self._cell(CFG))

    def test_custom_service_name_follows_the_config(self):
        cell = self._cell({**CFG, 'service_name': 'llm', 'tailnet_fqdn': 'llm.example.ts.net'})
        self.assertIn("SERVICE = 'svc:llm'\n", cell)
        self.assertNotIn('svc:qwen', cell)
        ast.parse(cell)

    def test_fqdn_must_start_with_the_service_name(self):
        with self.assertRaises(SystemExit) as cm:
            self._cell({**CFG, 'service_name': 'llm'})
        self.assertIn('must start with the Service name "llm."', str(cm.exception))
        with self.assertRaises(SystemExit):
            self._cell({**CFG, 'service_name': 'Bad Name'})

    def test_sanitized_build_keeps_the_placeholder_domain_with_any_service_name(self):
        placeholders = {f: p for f, (_, p) in sc.FIELDS.items()}
        self.assertEqual(ck.service_name({**placeholders, 'service_name': 'llm'}), 'llm')

    def test_no_api_key_flag_reaches_llama_server(self):
        # Documentation says the tailnet is the only gate; if this ever changes the docs must change with it.
        self.assertNotIn('--api-key', self._cell(CFG))

    def _load(self, stored):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / '.secrets.json'
            path.write_text(json.dumps(stored), encoding='utf-8')
            env = {k: v for k, v in os.environ.items() if k not in [e for e, _ in sc.FIELDS.values()]}
            with mock.patch.object(sc, 'SECRETS_FILE', path), mock.patch.dict(os.environ, env, clear=True):
                return sc.load_config([])

    def test_empty_string_means_not_configured(self):
        config, _, _, sources = self._load({**{k: 'v' for k in sc.REQUIRED}, 'local_tailscale_ip': '',
                                            'webui_system_prompt': '', 'notebooks_dir': '   ', 'service_name': ''})
        self.assertEqual(config['local_tailscale_ip'], '<YOUR_LOCAL_TAILSCALE_IP>')
        self.assertEqual(sources['local_tailscale_ip'], 'placeholder')
        self.assertEqual(config['notebooks_dir'], '')
        self.assertEqual(sc.missing_required(sources), [])

    def test_empty_required_value_is_reported_missing_not_a_crash(self):
        _, _, _, sources = self._load({**{k: 'v' for k in sc.REQUIRED}, 'api_key': ''})
        self.assertEqual(sc.missing_required(sources), ['api_key'])

    def test_non_string_values_are_still_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            self._load({'api_key': 5})
        self.assertIn('api_key', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
