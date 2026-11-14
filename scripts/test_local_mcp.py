"""Lifecycle and ownership contracts; fixtures never touch the user's bridge or GPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_mcp as m
import colab_cli as cli


class LocalMcpLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='local-mcp-unit-')
        self.root = Path(self.tmp.name)
        self.path = self.root / 'config.json'
        self.env = mock.patch.dict(os.environ, {'LOCAL_MCP_CONFIG': str(self.path), 'LOCAL_MCP_PORT': '34265'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def config(self):
        self.path.write_text(json.dumps({'token': 'test-only-owned-bridge-token', 'cwd': str(self.root), 'custom': 7}), encoding='utf-8')

    def test_configuration_preserves_user_choices_and_token(self):
        self.config()
        first = self.path.read_bytes()
        self.assertEqual(m.ensure_config()['custom'], 7)
        self.assertEqual(self.path.read_bytes(), first)

    def test_first_start_configuration_generates_a_stable_strong_token(self):
        with mock.patch.object(m, 'default_cwd', return_value=str(self.root)):
            first = m.ensure_config()
            second = m.ensure_config()
        self.assertGreaterEqual(len(first['token']), 40)
        self.assertEqual(first, second)

    def test_status_does_not_create_configuration(self):
        with mock.patch.object(m, 'probe', return_value=None):
            self.assertFalse(m.status()['running'])
        self.assertFalse(self.path.exists())

    def test_foreign_listener_is_neither_started_over_nor_stopped(self):
        with mock.patch.object(m, 'probe', return_value={'server': 'foreign'}), \
                mock.patch.object(m.subprocess, 'Popen') as spawn, mock.patch.object(m, '_terminate') as kill:
            self.assertFalse(m.start()['ok'])
            self.assertFalse(m.stop()['ok'])
            spawn.assert_not_called()
            kill.assert_not_called()

    def test_existing_bridge_with_another_token_is_not_adopted(self):
        self.config()
        with mock.patch.object(m, 'probe', return_value={'server': 'local_mcp'}), \
                mock.patch.object(m, '_post_initialize', return_value=401), mock.patch.object(m.subprocess, 'Popen') as spawn:
            self.assertFalse(m.start()['ok'])
            spawn.assert_not_called()

    def test_stop_requires_an_authenticated_pid(self):
        self.config()
        with mock.patch.object(m, 'probe', return_value={'server': 'local_mcp'}), mock.patch.object(m, '_terminate') as kill:
            self.assertFalse(m.stop()['ok'])
            kill.assert_not_called()

    def test_changed_pid_is_not_killed(self):
        self.config()
        with mock.patch.object(m, 'probe', side_effect=[{'server': 'local_mcp'}, {'pid': 10}, {'pid': 11}]), \
                mock.patch.object(m, '_terminate') as kill:
            self.assertFalse(m.stop()['ok'])
            kill.assert_not_called()

    def test_status_output_excludes_the_token(self):
        self.config()
        out = io.StringIO()
        with mock.patch.object(m, 'probe', return_value=None), contextlib.redirect_stdout(out):
            self.assertEqual(m.main(['status']), 0)
        result = json.loads(out.getvalue())
        self.assertTrue(result['ok'])
        self.assertNotIn('test-only-owned-bridge-token', out.getvalue())


class LocalMcpBindAndPrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='local-mcp-unit-')
        self.root = Path(self.tmp.name)
        self.path = self.root / 'config.json'
        env = {k: v for k, v in os.environ.items() if k not in ('LOCAL_MCP_HOST', 'LOCAL_TAILSCALE_IP')}
        env.update(LOCAL_MCP_CONFIG=str(self.path), LOCAL_MCP_PORT='34265', LOCAL_MCP_LOG=str(self.root / 'log.txt'))
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _ip(self, value):
        with mock.patch('secrets_config.optional_setting', return_value=value):
            return m.tailscale_bind_ip()

    def test_only_a_real_ipv4_counts_as_a_tailscale_address(self):
        self.assertEqual(self._ip('100.64.0.7'), '100.64.0.7')
        for value in ('', '<YOUR_LOCAL_TAILSCALE_IP>', '127.0.0.1', '0.0.0.0', 'host.ts.net'):
            self.assertEqual(self._ip(value), '', value)

    def test_default_bind_is_loopback_plus_the_tailscale_ip(self):
        with mock.patch.object(m, 'tailscale_bind_ip', return_value=''):
            self.assertEqual(m.listen_env(), {'LOCAL_MCP_HOST': '', 'LOCAL_MCP_EXTRA_HOST': ''})
        with mock.patch.object(m, 'tailscale_bind_ip', return_value='100.64.0.7'):
            self.assertEqual(m.listen_env(), {'LOCAL_MCP_HOST': '', 'LOCAL_MCP_EXTRA_HOST': '100.64.0.7'})

    def test_explicit_host_is_passed_through_alone(self):
        with mock.patch.dict(os.environ, {'LOCAL_MCP_HOST': '0.0.0.0'}), \
                mock.patch.object(m, 'tailscale_bind_ip', return_value='100.64.0.7'):
            self.assertEqual(m.listen_env(), {'LOCAL_MCP_HOST': '0.0.0.0', 'LOCAL_MCP_EXTRA_HOST': ''})

    def test_control_address_follows_the_bind_and_maps_wildcards_to_loopback(self):
        for host, expected in (('', '127.0.0.1'), ('0.0.0.0', '127.0.0.1'),
                               ('::', '::1'), ('::1', '::1'), ('127.0.0.2', '127.0.0.2')):
            with self.subTest(host=host), mock.patch.dict(os.environ, {'LOCAL_MCP_HOST': host}):
                self.assertEqual(m.control_host(), expected)
                authority = '[' + expected + ']' if ':' in expected else expected
                self.assertEqual(m.url(), 'http://%s:34265/mcp' % authority)

    def test_host_flag_sets_the_explicit_bind(self):
        with mock.patch.dict(m._COMMANDS, {'status': lambda: {'ok': True}}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(['status', '--host', '0.0.0.0']), 0)
        self.assertEqual(os.environ.get('LOCAL_MCP_HOST'), '0.0.0.0')

    def test_missing_pi_fails_before_any_config_is_written(self):
        out = io.StringIO()
        with mock.patch.object(m, 'probe', return_value=None), mock.patch.object(m, '_node', return_value='node'), \
                mock.patch.object(m, '_pi_dir', side_effect=SystemExit('@earendil-works/pi-coding-agent not found')), \
                mock.patch.object(m.subprocess, 'Popen') as spawn, contextlib.redirect_stdout(out):
            self.assertEqual(m.main(['start']), 1)
        spawn.assert_not_called()
        self.assertFalse(self.path.exists())
        self.assertIn('pi-coding-agent not found', json.loads(out.getvalue())['error'])

    def test_missing_node_fails_before_any_config_is_written(self):
        with mock.patch.object(m, 'probe', return_value=None), mock.patch.object(m.shutil, 'which', return_value=None), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(['start']), 1)
        self.assertFalse(self.path.exists())

    def test_pi_dir_honours_the_explicit_root(self):
        pkg = self.root / 'pi'
        pkg.mkdir()
        (pkg / 'package.json').write_text('{}', encoding='utf-8')
        with mock.patch.dict(os.environ, {'LOCAL_MCP_PI_ROOT': str(pkg)}):
            self.assertEqual(m._pi_dir(str(self.root / 'node.exe')), pkg)

    @unittest.skipIf(os.name == 'nt', 'pathlib cannot build POSIX paths on Windows')
    def test_pi_dir_finds_node_prefix_lib_node_modules_on_posix(self):
        prefix = self.root / 'homebrew'
        node_bin = prefix / 'bin' / 'node'
        pkg = prefix / 'lib' / 'node_modules' / '@earendil-works' / 'pi-coding-agent'
        node_bin.parent.mkdir(parents=True)
        node_bin.write_text('', encoding='utf-8')
        pkg.mkdir(parents=True)
        (pkg / 'package.json').write_text('{}', encoding='utf-8')
        env = {k: v for k, v in os.environ.items() if k not in ('LOCAL_MCP_PI_ROOT', 'npm_config_prefix')}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(m.os, 'name', 'posix'):
            self.assertEqual(m._pi_dir(str(node_bin)), pkg.resolve())

    def test_large_log_keeps_only_its_tail(self):
        log = self.root / 'log.txt'
        log.write_bytes(b'old line\n' * 200_000)  # ~1.8 MB
        m._rotate_log()
        self.assertLessEqual(log.stat().st_size, m.LOG_KEEP_BYTES)
        self.assertTrue(log.read_bytes().startswith(b'old line\n'))
        small = b'small\n' * 10
        log.write_bytes(small)
        m._rotate_log()
        self.assertEqual(log.read_bytes(), small)


class StartBridgeOnDemandTests(unittest.TestCase):
    def _start(self, ip, auto=None):
        env = {k: v for k, v in os.environ.items() if k != 'COLAB_LOCAL_MCP_AUTO'}
        if auto is not None:
            env['COLAB_LOCAL_MCP_AUTO'] = auto
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(m, 'tailscale_bind_ip', return_value=ip), \
                mock.patch.object(cli, 'local_mcp', return_value={'ok': True, 'started': True, 'url': 'u'}) as helper, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            cli.start_local_mcp()
        return helper, out.getvalue()

    def test_without_a_tailscale_ip_nothing_starts_or_prints(self):
        helper, out = self._start('')
        helper.assert_not_called()
        self.assertEqual(out, '')

    def test_with_a_tailscale_ip_the_bridge_starts(self):
        helper, _ = self._start('100.64.0.7')
        helper.assert_called_once_with('start')

    def test_auto_flag_starts_without_a_tailscale_ip(self):
        helper, _ = self._start('', auto='1')
        helper.assert_called_once_with('start')


class SharedBridgeCleanupTests(unittest.TestCase):
    def test_another_gpu_assignment_keeps_the_shared_bridge(self):
        with mock.patch.object(cli, 'usage', return_value={'active': 1}), mock.patch.object(cli, 'local_mcp') as helper:
            cli.stop_local_mcp()
        helper.assert_not_called()

    def test_unknown_account_state_keeps_the_shared_bridge(self):
        with mock.patch.object(cli, 'usage', side_effect=RuntimeError('offline')), \
                mock.patch.object(cli, 'local_mcp') as helper, contextlib.redirect_stdout(io.StringIO()):
            cli.stop_local_mcp()
        helper.assert_not_called()

    def test_confirmed_empty_account_stops_through_the_helper(self):
        with mock.patch.object(cli, 'usage', return_value={'active': 0}), \
                mock.patch.object(cli, 'local_mcp', return_value={'ok': True}) as helper:
            cli.stop_local_mcp()
        helper.assert_called_once_with('stop', timeout=15.0)

    def test_existing_keeper_still_checks_the_local_bridge(self):
        with mock.patch.object(cli, 'start_local_mcp') as start, \
                mock.patch.object(cli, 'keepalive_state', return_value={'pid': 12}), \
                mock.patch.object(cli, '_pid_alive', return_value=True), contextlib.redirect_stdout(io.StringIO()):
            cli.start_keepalive('fixture', 30)
        start.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
