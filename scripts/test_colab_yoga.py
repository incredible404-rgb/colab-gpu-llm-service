"""Colab-yoga integration: install discovery, legacy `easycolab` alias, runners export (no network, no Colab)."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import colab_cli
import colab_kernel as ck

CFG = {
    'tailnet_fqdn': 'qwen.example.ts.net',
    'api_key': 'test-key',
    'ts_client_id': 'id',
    'ts_client_secret': 'secret',
    'local_tailscale_ip': '192.0.2.1',
}


class CliYogaTests(unittest.TestCase):
    def test_missing_install_reports_where_to_install(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(colab_cli, 'YOGA_EXE', Path(td) / 'Colab-yoga.exe'):
                with self.assertRaises(SystemExit) as cm:
                    colab_cli.cmd_yoga(['gui'])
        msg = str(cm.exception)
        self.assertIn('Colab-yoga not found', msg)
        self.assertIn('github.com/incredible404-rgb/Colab-yoga', msg)
        self.assertIn('COLAB_YOGA_DIR', msg)
        self.assertNotIn('EasyColab', msg)

    def test_yoga_and_legacy_easycolab_route_to_same_handler(self):
        for name in ('yoga', 'colab-yoga', 'easycolab'):
            with mock.patch.object(colab_cli, 'cmd_yoga', return_value=0) as m:
                self.assertEqual(colab_cli.main([name, 'status']), 0)
            m.assert_called_once_with(['status'])

    def test_no_legacy_install_fallback(self):
        # Only Colab-yoga is probed; the old EasyColab layout must not be consulted any more.
        self.assertFalse(hasattr(colab_cli, 'EASYCOLAB_DIR'))
        self.assertEqual(colab_cli.YOGA_EXE.name, 'Colab-yoga.exe')

    def test_wake_keeps_alive_the_yoga_session(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / 'Colab-yoga.exe'
            exe.write_text('')
            with mock.patch.object(colab_cli, 'YOGA_EXE', exe), \
                    mock.patch.object(colab_cli, 'YOGA_PY', Path(td) / 'python.exe'), \
                    mock.patch.object(colab_cli.subprocess, 'run', return_value=mock.Mock(returncode=0)), \
                    mock.patch.object(colab_cli, 'sessions', return_value=['other', 'easycolab']), \
                    mock.patch.object(colab_cli, 'start_keepalive') as ka:
                self.assertEqual(colab_cli.cmd_yoga(['wake', '--gpu', 'G4', '--idle-min', '30']), 0)
        ka.assert_called_once_with('easycolab', 30)


class RunnersExportTests(unittest.TestCase):
    def test_path_follows_env_override(self):
        with mock.patch.dict(os.environ, {'COLAB_YOGA_DIR': 'X:/yoga'}):
            self.assertEqual(ck.yoga_runners_path(), Path('X:/yoga') / 'colab_kernel_runners.json')

    def test_default_path_is_colab_yoga_only(self):
        env = {k: v for k, v in os.environ.items() if k != 'COLAB_YOGA_DIR'}
        env['LOCALAPPDATA'] = 'C:/Users/u/AppData/Local'
        with mock.patch.dict(os.environ, env, clear=True):
            p = ck.yoga_runners_path()
        self.assertEqual(p.parts[-3:], ('Programs', 'Colab-yoga', 'colab_kernel_runners.json'))

    def test_missing_install_dir_is_skipped_with_hint(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / 'Colab-yoga' / 'colab_kernel_runners.json'
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ck.export_yoga_runners(CFG, dst)
            self.assertFalse(dst.parent.exists())
        out = buf.getvalue()
        self.assertIn('skipped', out)
        self.assertIn('github.com/incredible404-rgb/Colab-yoga', out)

    def test_writes_runners_for_every_active_model(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / 'colab_kernel_runners.json'
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                ck.export_yoga_runners(CFG, dst)
            data = json.loads(dst.read_text(encoding='utf-8'))
        self.assertEqual(list(data), list(ck.MODELS))
        for slug, item in data.items():
            self.assertEqual(item['alias'], ck.MODELS[slug]['alias'])
            self.assertTrue(item['code'])
        self.assertIn('Colab-yoga runners', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
