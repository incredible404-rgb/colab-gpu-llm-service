"""First-run safety: an unfinished .secrets.json must stop before a GPU is billed, a console code page or a crash
must not strand an allocated runtime, and nothing private is assumed (mirror dir, Magpie, system prompt, file bridge).
No network, no Colab, no Drive writes."""
import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import colab_cli
import colab_kernel as ck
import secrets_config as sc
import stdio_utf8

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'qwen38-huihui-v2'
ENV_NAMES = [env for env, _ in sc.FIELDS.values()]
CFG = {
    'tailnet_fqdn': 'qwen.example.ts.net',
    'api_key': 'test-key',
    'ts_client_id': 'id',
    'ts_client_secret': 'secret',
    'local_tailscale_ip': '192.0.2.1',
}


def _clean_env():
    return {k: v for k, v in os.environ.items() if k not in ENV_NAMES}


def _ui_cfg(nb):
    src = ''.join(next(c['source'] for c in nb['cells'] if c['metadata'].get('id') == 'persistent-setup'))
    line = next(l for l in src.splitlines() if l.startswith('UI_CFG_DATA = json.loads('))
    return json.loads(ast.literal_eval(line[len('UI_CFG_DATA = json.loads('):-1])), src


class PlaceholderTests(unittest.TestCase):
    def _load(self, stored, env=None):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / '.secrets.json'
            path.write_text(json.dumps(stored), encoding='utf-8')
            with mock.patch.object(sc, 'SECRETS_FILE', path), \
                    mock.patch.dict(os.environ, {**_clean_env(), **(env or {})}, clear=True):
                return sc.load_config(['--output-dir', td])

    def test_is_placeholder(self):
        for v in ('<YOUR_LLAMA_API_KEY>', 'qwen.<YOUR_TAILNET>.ts.net', '<anything>', '', '  ', None, '<your_x>'):
            self.assertTrue(sc.is_placeholder(v), v)
        for v in ('sk-abc', 'qwen.my-tailnet.ts.net', '198.51.100.9'):
            self.assertFalse(sc.is_placeholder(v), v)

    def test_example_file_copied_verbatim_counts_as_missing(self):
        example = json.loads((ROOT / '.secrets.example.json').read_text(encoding='utf-8'))
        _, _, _, sources = self._load(example)
        self.assertEqual(sc.missing_required(sources), list(sc.REQUIRED))

    def test_real_values_are_not_missing_and_env_fills_a_placeholder(self):
        stored = {'api_key': '<YOUR_LLAMA_API_KEY>', 'ts_client_id': 'a', 'ts_client_secret': 'b', 'tailnet_fqdn': 'q.x.ts.net'}
        config, _, _, sources = self._load(stored, {'LLAMA_API_KEY': 'from-env'})
        self.assertEqual(sc.missing_required(sources), [])
        self.assertEqual((config['api_key'], sources['api_key']), ('from-env', 'env'))

    def test_optional_fields_never_warn_and_never_block_sync(self):
        config, _, _, sources = self._load({k: 'v' for k in sc.REQUIRED})
        self.assertEqual(config['webui_system_prompt'], '')
        self.assertEqual(config['notebooks_dir'], '')
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            sc.report(config, False, sources)
        self.assertNotIn('webui_system_prompt', err.getvalue())
        self.assertNotIn('notebooks_dir', err.getvalue())

    def test_free_text_setting_may_contain_angle_brackets(self):
        stored = {k: 'v' for k in sc.REQUIRED}
        stored['webui_system_prompt'] = 'Answer as <your friend> would.'
        config, _, _, _ = self._load(stored)
        self.assertEqual(config['webui_system_prompt'], 'Answer as <your friend> would.')

    def test_start_exits_before_any_colab_call_on_example_secrets(self):
        example = (ROOT / '.secrets.example.json').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / '.secrets.json'
            path.write_text(example, encoding='utf-8')
            with mock.patch.object(sc, 'SECRETS_FILE', path), \
                    mock.patch.dict(os.environ, _clean_env(), clear=True), \
                    mock.patch.object(colab_cli, 'colab', side_effect=AssertionError('colab must not run')), \
                    mock.patch.object(colab_cli, 'usage', side_effect=AssertionError('usage must not run')), \
                    mock.patch.object(colab_cli, '_ensure_session', side_effect=AssertionError('no allocation')):
                with self.assertRaises(SystemExit) as cm:
                    colab_cli.cmd_start(SLUG, 'G4', 45, False, False)
                with self.assertRaises(SystemExit):
                    colab_cli.cmd_launch(SLUG, 'G4', 45)
        msg = str(cm.exception)
        self.assertIn('missing api_key, ts_client_id, ts_client_secret, tailnet_fqdn', msg)
        self.assertIn('nothing was allocated', msg)

    def test_launch_ignores_unset_optional_placeholders(self):
        cell = 'KEY = "real"\nLOCAL = "<YOUR_LOCAL_TAILSCALE_IP>"\n'
        with mock.patch.object(colab_cli, '_config', return_value={'tailnet_fqdn': 'q.ts.net', 'api_key': 'k'}), \
                mock.patch.object(colab_cli, 'notebook_first_cell', return_value=cell), \
                mock.patch.object(colab_cli, 'usage', side_effect=RuntimeError('past the placeholder check')):
            with self.assertRaises(RuntimeError):
                colab_cli.cmd_launch(SLUG, 'G4', 45)


class StartReleasesTests(unittest.TestCase):
    def _start(self, *, created, keep_on_fail=False, exec_error=None):
        stopped = []
        with mock.patch.object(colab_cli, '_config', return_value={'tailnet_fqdn': 'q.ts.net', 'api_key': 'k'}), \
                mock.patch.object(colab_cli, 'headless_setup', return_value='print(1)\n'), \
                mock.patch.object(colab_cli, 'usage', return_value={'balance': 100.0, 'rate': 0.0, 'active': 0}), \
                mock.patch.object(colab_cli, '_ensure_session', return_value=created), \
                mock.patch.object(colab_cli, 'start_keepalive'), \
                mock.patch.object(colab_cli, 'exec_stream', side_effect=exec_error), \
                mock.patch.object(colab_cli, 'cmd_stop', side_effect=lambda names: stopped.append(names)), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                colab_cli.cmd_start(SLUG, 'G4', 45, False, keep_on_fail)
            except BaseException as err:
                return stopped, err
        return stopped, None

    def test_unicode_error_after_allocation_releases_the_new_runtime(self):
        err = UnicodeEncodeError('cp1252', '\u4e2d', 0, 1, 'character maps to <undefined>')
        stopped, raised = self._start(created=True, exec_error=err)
        self.assertIs(raised, err)
        self.assertEqual(stopped, [[SLUG]])

    def test_keyboard_interrupt_also_releases(self):
        stopped, raised = self._start(created=True, exec_error=KeyboardInterrupt())
        self.assertIsInstance(raised, KeyboardInterrupt)
        self.assertEqual(stopped, [[SLUG]])

    def test_reused_session_is_left_running(self):
        stopped, raised = self._start(created=False, exec_error=RuntimeError('boom'))
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(stopped, [])

    def test_keep_on_fail_keeps_the_runtime(self):
        stopped, raised = self._start(created=True, keep_on_fail=True, exec_error=RuntimeError('boom'))
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(stopped, [])

    def test_failed_start_releases_once(self):
        stopped, raised = self._start(created=True, exec_error=[1])
        self.assertIsNone(raised)
        self.assertEqual(stopped, [[SLUG]])

    def test_launch_releases_a_new_runtime_on_unexpected_error(self):
        stopped = []
        with mock.patch.object(colab_cli, '_config', return_value={'tailnet_fqdn': 'q.ts.net', 'api_key': 'k'}), \
                mock.patch.object(colab_cli, 'notebook_first_cell', return_value='print(1)\n'), \
                mock.patch.object(colab_cli, 'usage', return_value={'balance': 1.0, 'rate': 0.0, 'active': 0}), \
                mock.patch.object(colab_cli, '_ensure_session', return_value=True), \
                mock.patch.object(colab_cli, 'start_keepalive', side_effect=ValueError('boom')), \
                mock.patch.object(colab_cli, 'cmd_stop', side_effect=lambda names: stopped.append(names)), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                colab_cli.cmd_launch(SLUG, 'G4', 45)
        self.assertEqual(stopped, [[SLUG]])

    def test_failed_colab_new_cleans_up_what_it_may_have_allocated(self):
        stopped = []
        with mock.patch.object(colab_cli, 'sessions', return_value=[]), \
                mock.patch.object(colab_cli, 'colab', side_effect=SystemExit('colab new failed')), \
                mock.patch.object(colab_cli, 'cmd_stop', side_effect=lambda names: stopped.append(names)), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                colab_cli._ensure_session(SLUG, 'G4')
        self.assertEqual(stopped, [[SLUG]])


class Utf8Tests(unittest.TestCase):
    def test_force_utf8_makes_a_cp1252_stream_accept_chinese(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding='cp1252', errors='strict')
        with self.assertRaises(UnicodeEncodeError):
            stream.write('\u4e2d\u6587')
            stream.flush()
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding='cp1252', errors='strict')
        with mock.patch.object(sys, 'stdout', stream), mock.patch.object(sys, 'stderr', stream):
            stdio_utf8.force_utf8()
            print('\u4e2d\u6587 \U0001f680')
            stream.flush()
        self.assertEqual(raw.getvalue().decode('utf-8').strip(), '\u4e2d\u6587 \U0001f680')

    def test_cli_entry_survives_a_legacy_console_code_page(self):
        code = ('import sys; sys.path.insert(0, %r); import colab_cli; colab_cli.force_utf8(); print("\\u4e2d\\u6587")'
                % str(Path(__file__).resolve().parent))
        env = {**os.environ, 'PYTHONIOENCODING': 'cp1252', 'PYTHONUTF8': '0'}
        proc = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode('utf-8', 'replace'))
        self.assertEqual(proc.stdout.decode('utf-8').strip(), '\u4e2d\u6587')


class StopLeftoverTests(unittest.TestCase):
    def _run(self, *, yoga, usages):
        calls = []
        usages = list(usages)
        with tempfile.TemporaryDirectory() as td:
            py, bridge = Path(td) / 'python.exe', Path(td) / 'bridge.py'
            if yoga:
                py.write_text('')
                bridge.write_text('')
            with mock.patch.object(colab_cli, 'YOGA_PY', py), mock.patch.object(colab_cli, 'YOGA_BRIDGE', bridge), \
                    mock.patch.object(colab_cli, 'usage', side_effect=lambda: usages.pop(0)), \
                    mock.patch.object(colab_cli.subprocess, 'run', side_effect=lambda argv, **kw: calls.append(argv)), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                result = colab_cli._release_leftovers({'active': 1, 'rate': 10.0})
        return calls, result, out.getvalue()

    def test_yoga_bridge_goes_first_and_no_browser_is_needed(self):
        calls, result, _ = self._run(yoga=True, usages=[{'active': 0, 'rate': 0.0}])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:2], ['-u'])
        self.assertEqual(calls[0][-1], 'stop_all')
        self.assertEqual(result['active'], 0)

    def test_browser_fallback_only_when_the_bridge_is_absent_or_not_enough(self):
        calls, _, _ = self._run(yoga=False, usages=[{'active': 0, 'rate': 0.0}])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][-2].endswith('colab_sync.py'))
        calls, _, out = self._run(yoga=True, usages=[{'active': 1, 'rate': 10.0}, {'active': 1, 'rate': 10.0}])
        self.assertEqual(len(calls), 2)
        self.assertIn('Manage sessions', out)


class KernelPortabilityTests(unittest.TestCase):
    def test_positional_args_skips_the_output_dir_value(self):
        self.assertEqual(ck.positional_args([SLUG, '--output-dir', 'out']), [SLUG])
        self.assertEqual(ck.positional_args(['--output-dir', 'out', SLUG, '--sanitized']), [SLUG])
        self.assertEqual(ck.positional_args(['--output-dir=out', SLUG]), [SLUG])
        self.assertEqual(ck.positional_args(['--all', '--output-dir', 'out']), [])

    def test_main_does_not_treat_output_dir_as_a_slug(self):
        seen = []
        with mock.patch.object(sys, 'argv', ['colab_kernel.py', SLUG, '--sanitized', '--output-dir', 'somewhere']), \
                mock.patch.object(ck, 'generate_one', side_effect=lambda slug, argv=None: seen.append(slug)):
            ck.main()
        self.assertEqual(seen, [SLUG])

    def test_bridge_ip(self):
        self.assertEqual(ck.bridge_ip({'local_tailscale_ip': '198.51.100.7'}), '198.51.100.7')
        for value in ('<YOUR_LOCAL_TAILSCALE_IP>', '', 'not-an-ip', '999.1.1.1', None):
            self.assertEqual(ck.bridge_ip({'local_tailscale_ip': value}), '', value)
        self.assertEqual(ck.bridge_ip({}), '')

    def test_no_file_bridge_without_a_real_ip(self):
        spec = ck.MODELS[SLUG]
        for ip in ('<YOUR_LOCAL_TAILSCALE_IP>', ''):
            nb = ck.build_notebook_dict(spec, {**CFG, 'local_tailscale_ip': ip})
            ui, src = _ui_cfg(nb)
            self.assertNotIn('mcpServers', ui)
            self.assertIn('LOCAL_MCP_IP = ""', src)
            ast.parse(src)
        ui, src = _ui_cfg(ck.build_notebook_dict(spec, CFG))
        self.assertEqual(json.loads(ui['mcpServers'])[0]['name'], 'local_novel_fs')
        self.assertIn('LOCAL_MCP_IP = "192.0.2.1"', src)

    def test_system_prompt_is_empty_unless_configured(self):
        spec = ck.MODELS[SLUG]
        ui, src = _ui_cfg(ck.build_notebook_dict(spec, CFG))
        self.assertNotIn('systemMessage', ui)
        self.assertFalse(ui.get('showSystemMessage', False))
        ui, _ = _ui_cfg(ck.build_notebook_dict(spec, {**CFG, 'webui_system_prompt': 'be brief'}))
        self.assertEqual(ui['systemMessage'], 'be brief')
        self.assertTrue(ui['showSystemMessage'])
        blob = json.dumps(ck.build_notebook_dict(spec, CFG), ensure_ascii=False)
        self.assertNotIn('Colab\u521b\u4f5c', blob)

    def test_drive_mirror_is_opt_in(self):
        with mock.patch.object(ck, '_probe_my_drive_notebooks', return_value=Path('D:/My Drive/Colab Notebooks')) as probe:
            self.assertIsNone(ck.drive_mirror_dir(CFG, ['--all']))
            probe.assert_not_called()
            self.assertEqual(ck.drive_mirror_dir({**CFG, 'notebooks_dir': 'E:/nb'}, ['--all']), Path('E:/nb'))
            self.assertEqual(ck.drive_mirror_dir(CFG, ['--mirror-drive']), Path('D:/My Drive/Colab Notebooks'))

    def test_generate_one_does_not_copy_a_private_build_to_drive_by_default(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as drive:
            out = Path(td) / 'notebooks'
            with mock.patch.object(ck, 'ROOT_DIR', Path(td)), \
                    mock.patch.object(ck, 'load_config', return_value=({**CFG, **{s['drive_key']: 'id' for s in ck.ALL_MODELS.values()}}, False, out, {})), \
                    mock.patch.object(ck, 'report'), mock.patch.object(ck, 'guard_sync'), \
                    mock.patch.object(ck, 'sync_requested'), \
                    mock.patch.object(ck, '_probe_my_drive_notebooks', return_value=Path(drive)), \
                    contextlib.redirect_stdout(io.StringIO()):
                out.mkdir()
                ck.generate_one(SLUG, [])
            self.assertEqual(list(Path(drive).iterdir()), [])
            self.assertTrue(any(out.iterdir()))


class MagpieLauncherTests(unittest.TestCase):
    def test_missing_magpie_skips_with_a_message(self):
        out = io.StringIO()
        with mock.patch.object(ck, '_magpie_launcher', return_value=None), \
                mock.patch.object(ck.subprocess, 'run', side_effect=AssertionError('no process may start')), \
                contextlib.redirect_stdout(out):
            ck.sync_magpie_models()
        self.assertIn('skipped', out.getvalue())

    def test_launcher_prefers_a_real_executable_on_path(self):
        with mock.patch.object(ck.shutil, 'which', side_effect=lambda n: 'C:/tools/magpie.exe' if n == 'magpie' else None):
            self.assertEqual(ck._magpie_launcher(), ('C:/tools/magpie.exe',))

    def test_launcher_uses_bash_only_when_the_shell_finds_magpie(self):
        which = {'magpie': None, 'bash': 'C:/git/bash.exe'}
        found = subprocess.CompletedProcess([], 0, stdout='/home/u/.local/bin/magpie\n')
        with mock.patch.object(ck.shutil, 'which', side_effect=which.get), \
                mock.patch.object(ck.subprocess, 'run', return_value=found):
            self.assertEqual(ck._magpie_launcher(), ('C:/git/bash.exe', '-lc'))
        with mock.patch.object(ck.shutil, 'which', side_effect=which.get), \
                mock.patch.object(ck.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, stdout='')):
            self.assertIsNone(ck._magpie_launcher())

    def test_no_machine_specific_paths_left_in_the_kernel_source(self):
        source = (Path(ck.__file__)).read_text(encoding='utf-8')
        for private in ('Program Files', '.local/bin/magpie', 'Colab\u521b\u4f5c'):
            self.assertNotIn(private, source)


class YogaRunnersHintTests(unittest.TestCase):
    def test_skip_message_names_both_ways_to_fix_it(self):
        with tempfile.TemporaryDirectory() as td:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ck.export_yoga_runners(CFG, Path(td) / 'missing' / 'colab_kernel_runners.json')
        text = out.getvalue()
        self.assertIn('COLAB_YOGA_DIR', text)
        self.assertIn('Programs', text)

    def test_runners_follow_colab_yoga_dir(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {'COLAB_YOGA_DIR': td}):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                ck.export_yoga_runners(CFG)
            self.assertTrue((Path(td) / 'colab_kernel_runners.json').exists())
        self.assertIn('Exported Colab-yoga runners to:', out.getvalue())


class NovelRootTests(unittest.TestCase):
    def test_root_resolution_order(self):
        import local_novel_mcp as m
        with mock.patch.dict(os.environ, {'NOVEL_ROOT': str(Path.home() / 'envroot')}):
            self.assertEqual(m._novel_root(), (Path.home() / 'envroot').resolve())
        env = {k: v for k, v in os.environ.items() if k != 'NOVEL_ROOT'}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(m, 'optional_setting', return_value='~/from-secrets'):
            self.assertEqual(m._novel_root(), (Path.home() / 'from-secrets').resolve())
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(m, 'optional_setting', return_value=''):
            self.assertEqual(m._novel_root(), (Path.home() / 'Documents/colab-novel').resolve())


class SyncTierTests(unittest.TestCase):
    def test_browsermcp_tier_is_skipped_when_not_configured(self):
        import colab_sync
        tried = []
        with mock.patch.object(colab_sync, 'BMCP_BRIDGE', None), mock.patch.object(colab_sync, '_backend', None), \
                mock.patch.object(colab_sync, '_bmcp_eval', side_effect=lambda *a, **k: tried.append('bmcp')), \
                mock.patch.object(colab_sync, '_pw_eval', side_effect=lambda *a, **k: tried.append('pw') or {'ok': 1}):
            ok, result = colab_sync._tiered('() => 1', {'drive_id': '__ANY__', 'url': 'u'})
        self.assertTrue(ok)
        self.assertEqual(tried, ['pw'])
        colab_sync._backend = None


if __name__ == '__main__':
    unittest.main()
