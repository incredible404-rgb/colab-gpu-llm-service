"""Colab-yoga integration: install discovery, legacy `easycolab` alias, runners export (no network, no Colab)."""
import contextlib
import io
import json
import os
import sys
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
        expected = 'Colab-yoga.app' if sys.platform == 'darwin' else ('Colab-yoga.exe' if os.name == 'nt' else 'Colab-yoga')
        self.assertEqual(colab_cli.YOGA_EXE.name, expected)

    def test_macos_gui_launches_app_bundle_via_open(self):
        with tempfile.TemporaryDirectory() as td:
            app = Path(td) / 'Colab-yoga.app'
            app.mkdir()
            launched = []
            with mock.patch.object(colab_cli.sys, 'platform', 'darwin'), \
                    mock.patch.object(colab_cli, 'YOGA_DIR', Path(td)), \
                    mock.patch.object(colab_cli, 'YOGA_EXE', app), \
                    mock.patch.object(colab_cli.subprocess, 'Popen', side_effect=lambda argv, **kw: launched.append((argv, kw))), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(colab_cli.cmd_yoga(['gui']), 0)
            self.assertEqual(launched, [(['open', '-a', str(app)], {'cwd': str(Path(td))})])

    def test_wake_keeps_alive_the_yoga_session(self):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / 'Colab-yoga.exe'
            exe.write_text('')
            with mock.patch.object(colab_cli, 'YOGA_EXE', exe), \
                    mock.patch.object(colab_cli, 'YOGA_PY', Path(td) / 'python.exe'), \
                    mock.patch.object(colab_cli, 'run_streamed', return_value=(0, 'boot\nWAKE_URL: http://x\n')), \
                    mock.patch.object(colab_cli, 'sessions', return_value=['other', 'easycolab']), \
                    mock.patch.object(colab_cli, 'start_keepalive') as ka:
                self.assertEqual(colab_cli.cmd_yoga(['wake', '--gpu', 'G4', '--idle-min', '30']), 0)
        ka.assert_called_once_with('easycolab', 30)


class RunnersExportTests(unittest.TestCase):
    def test_path_follows_env_override(self):
        with mock.patch.dict(os.environ, {'COLAB_YOGA_DIR': 'X:/yoga'}):
            self.assertEqual(ck.yoga_runners_path(), Path('X:/yoga') / 'colab_kernel_runners.json')

    def test_default_path_is_the_per_user_colab_yoga_dir(self):
        env = {k: v for k, v in os.environ.items() if k != 'COLAB_YOGA_DIR'}
        env['LOCALAPPDATA'] = 'C:/Users/u/AppData/Local'
        with mock.patch.dict(os.environ, env, clear=True):
            p = ck.yoga_runners_path()
        self.assertEqual(p, Path('C:/Users/u/AppData/Local') / 'Colab-yoga' / 'colab_kernel_runners.json')

    def test_without_localappdata_the_install_dir_is_the_fallback(self):
        env = {k: v for k, v in os.environ.items() if k not in ('COLAB_YOGA_DIR', 'LOCALAPPDATA')}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(ck.sys, 'platform', 'win32'):
            p = ck.yoga_runners_path()
        self.assertEqual(p.parts[-3:], ('Programs', 'Colab-yoga', 'colab_kernel_runners.json'))

    def test_macos_default_path_uses_application_support_and_creates_on_demand(self):
        env = {k: v for k, v in os.environ.items() if k not in ('COLAB_YOGA_DIR', 'LOCALAPPDATA')}
        with tempfile.TemporaryDirectory() as td:
            app_sup = Path(td) / 'Library/Application Support'
            app_sup.mkdir(parents=True)
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(ck.sys, 'platform', 'darwin'), \
                    mock.patch.object(ck.Path, 'home', return_value=Path(td)):
                p = ck.yoga_runners_path()
                self.assertEqual(p, app_sup / 'Colab-yoga' / 'colab_kernel_runners.json')
                self.assertTrue(ck._yoga_dir_created_on_demand())
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(ck.export_yoga_runners(CFG))
                self.assertTrue(p.is_file())
                self.assertTrue((app_sup / 'Colab-yoga' / 'local_mcp_bridge.json').is_file())

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

    def test_runners_carry_the_display_name_tags_and_note(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / 'colab_kernel_runners.json'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(ck.export_yoga_runners(CFG, dst))
            runners = json.loads(dst.read_text(encoding='utf-8'))
        for slug, spec in ck.MODELS.items():
            for key in ('name', 'tags', 'note'):
                self.assertEqual(runners[slug][key], spec['display'][key])
            self.assertNotIn('display', runners[slug])

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


class ZeroConfigExportTests(unittest.TestCase):
    """Default export into %LOCALAPPDATA%/Colab-yoga; LOCALAPPDATA always points at a temp dir here."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.local = Path(self.tmp.name) / 'Local'
        env = {k: v for k, v in os.environ.items() if k != 'COLAB_YOGA_DIR'}
        env['LOCALAPPDATA'] = str(self.local)
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _export(self, config=CFG):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            result = ck.export_yoga_runners(config)
        return result, out.getvalue()

    def test_creates_the_data_dir_and_writes_runners_and_bridge(self):
        result, out = self._export()
        data_dir = self.local / 'Colab-yoga'
        self.assertTrue(result)
        self.assertEqual(list(json.loads((data_dir / 'colab_kernel_runners.json').read_text(encoding='utf-8'))),
                         list(ck.MODELS))
        bridge = json.loads((data_dir / 'local_mcp_bridge.json').read_text(encoding='utf-8'))
        self.assertEqual(Path(bridge['script']), Path(ck.__file__).resolve().with_name('local_mcp.py'))
        self.assertTrue(Path(bridge['script']).is_file())
        self.assertIn('Exported Colab-yoga runners to:', out)

    def test_placeholders_create_nothing(self):
        result, _ = self._export({**CFG, 'api_key': '<YOUR_LLAMA_API_KEY>'})
        self.assertFalse(result)
        self.assertFalse((self.local / 'Colab-yoga').exists())

    def test_stale_copies_in_the_install_dir_and_app_dir_are_refreshed(self):
        programs = self.local / 'Programs' / 'Colab-yoga'
        programs.mkdir(parents=True)
        (programs / 'colab_kernel_runners.json').write_text('{}', encoding='utf-8')
        portable = Path(self.tmp.name) / 'portable'
        portable.mkdir()
        (portable / 'colab_kernel_runners.json').write_text('{}', encoding='utf-8')
        (self.local / 'Colab-yoga').mkdir(parents=True)
        (self.local / 'Colab-yoga' / 'install.json').write_text(json.dumps({'appDir': str(portable)}), encoding='utf-8')
        self.assertTrue(self._export()[0])
        primary = (self.local / 'Colab-yoga' / 'colab_kernel_runners.json').read_text(encoding='utf-8')
        self.assertEqual((programs / 'colab_kernel_runners.json').read_text(encoding='utf-8'), primary)
        self.assertEqual((portable / 'colab_kernel_runners.json').read_text(encoding='utf-8'), primary)
        self.assertFalse((programs / 'local_mcp_bridge.json').exists())

    def test_app_dir_and_install_dir_without_an_old_copy_are_left_alone(self):
        programs = self.local / 'Programs' / 'Colab-yoga'
        programs.mkdir(parents=True)
        portable = Path(self.tmp.name) / 'portable'
        portable.mkdir()
        (self.local / 'Colab-yoga').mkdir(parents=True)
        (self.local / 'Colab-yoga' / 'install.json').write_text(json.dumps({'appDir': str(portable)}), encoding='utf-8')
        self.assertTrue(self._export()[0])
        self.assertEqual(list(portable.iterdir()), [])
        self.assertEqual(list(programs.iterdir()), [])

    def test_colab_yoga_dir_is_created_and_skips_other_copies(self):
        target = Path(self.tmp.name) / 'unzipped' / 'Colab-yoga'
        programs = self.local / 'Programs' / 'Colab-yoga'
        programs.mkdir(parents=True)
        with mock.patch.dict(os.environ, {'COLAB_YOGA_DIR': str(target)}):
            self.assertTrue(self._export()[0])
        self.assertTrue((target / 'colab_kernel_runners.json').is_file())
        self.assertTrue((target / 'local_mcp_bridge.json').is_file())
        self.assertFalse((programs / 'colab_kernel_runners.json').exists())
        self.assertFalse((self.local / 'Colab-yoga').exists())


class ResolveYogaDirTests(unittest.TestCase):
    def _resolve(self, local, **extra):
        env = {k: v for k, v in os.environ.items() if k != 'COLAB_YOGA_DIR'}
        env.update(LOCALAPPDATA=str(local), **extra)
        with mock.patch.dict(os.environ, env, clear=True):
            return colab_cli.resolve_yoga_dir()

    def test_order_env_then_install_json_then_programs(self):
        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / 'Local'
            self.assertEqual(self._resolve(local), local / 'Programs' / 'Colab-yoga')
            portable = Path(td) / 'anywhere'
            (local / 'Colab-yoga').mkdir(parents=True)
            (local / 'Colab-yoga' / 'install.json').write_text(json.dumps({'appDir': str(portable)}), encoding='utf-8')
            self.assertEqual(self._resolve(local), local / 'Programs' / 'Colab-yoga')  # no bridge there yet
            (portable / 'bridge').mkdir(parents=True)
            (portable / 'bridge' / 'colab_yoga_bridge.py').write_text('', encoding='utf-8')
            self.assertEqual(self._resolve(local), portable)
            self.assertEqual(self._resolve(local, COLAB_YOGA_DIR='X:/yoga'), Path('X:/yoga'))

    def test_broken_install_json_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            local = Path(td)
            (local / 'Colab-yoga').mkdir()
            (local / 'Colab-yoga' / 'install.json').write_text('not json', encoding='utf-8')
            self.assertEqual(self._resolve(local), local / 'Programs' / 'Colab-yoga')


if __name__ == '__main__':
    unittest.main()
