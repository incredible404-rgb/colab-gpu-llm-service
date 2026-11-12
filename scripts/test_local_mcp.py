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


class SharedBridgeCleanupTests(unittest.TestCase):
    def test_another_gpu_assignment_keeps_the_shared_bridge(self):
        with mock.patch.object(cli, 'usage', return_value={'active': 1}), mock.patch.object(cli, 'local_mcp') as helper:
            cli.stop_novel_mcp()
        helper.assert_not_called()

    def test_unknown_account_state_keeps_the_shared_bridge(self):
        with mock.patch.object(cli, 'usage', side_effect=RuntimeError('offline')), \
                mock.patch.object(cli, 'local_mcp') as helper, contextlib.redirect_stdout(io.StringIO()):
            cli.stop_novel_mcp()
        helper.assert_not_called()

    def test_confirmed_empty_account_stops_through_the_helper(self):
        with mock.patch.object(cli, 'usage', return_value={'active': 0}), \
                mock.patch.object(cli, 'local_mcp', return_value={'ok': True}) as helper:
            cli.stop_novel_mcp()
        helper.assert_called_once_with('stop', timeout=15.0)

    def test_existing_keeper_still_checks_the_local_bridge(self):
        with mock.patch.object(cli, 'start_novel_mcp') as start, \
                mock.patch.object(cli, 'keepalive_state', return_value={'pid': 12}), \
                mock.patch.object(cli, '_pid_alive', return_value=True), contextlib.redirect_stdout(io.StringIO()):
            cli.start_keepalive('fixture', 30)
        start.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
