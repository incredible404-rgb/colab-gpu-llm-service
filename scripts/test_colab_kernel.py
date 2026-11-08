"""Generation tests for the 3.8fn-only kernel (no network, no Drive writes)."""
import ast
import json
import subprocess
import unittest
from unittest import mock

import colab_kernel as ck

CFG = {
    'tailnet_fqdn': 'qwen.example.ts.net',
    'api_key': 'test-key',
    'ts_client_id': 'id',
    'ts_client_secret': 'secret',
    'local_tailscale_ip': '100.64.0.1',
}


def _cell(nb, cell_id):
    return ''.join(next(c['source'] for c in nb['cells'] if c['metadata'].get('id') == cell_id))


def _runtime_spec(src):
    line = next(l for l in src.splitlines() if l.startswith('SPEC = json.loads('))
    return json.loads(ast.literal_eval(line[len('SPEC = json.loads('):-1]))


class RegistryTests(unittest.TestCase):
    def test_only_38fn_quants(self):
        self.assertEqual(list(ck.MODELS), ['qwen38-fn', 'qwen38-fn-nl'])

    def test_nl_shares_everything_but_weights(self):
        fn, nl = ck.MODELS['qwen38-fn'], ck.MODELS['qwen38-fn-nl']
        differ = {k for k in fn if fn[k] != nl.get(k)}
        self.assertEqual(differ, {'slug', 'notebook', 'drive_key', 'env_key', 'placeholder', 'alias',
                                  'title', 'banner', 'root_dir', 'ts_hostname', 'ple_shard_name', 'files'})
        self.assertEqual([f for f in fn['files'] if f['role'] in ('mmproj', 'draft')],
                         [f for f in nl['files'] if f['role'] in ('mmproj', 'draft')])

    def test_ple_shard_registered_and_first_to_tmpfs(self):
        for spec in ck.MODELS.values():
            ple = [f for f in spec['files'] if f['path'].endswith(spec['ple_shard_name'])]
            self.assertEqual(len(ple), 1)
            self.assertEqual(ple[0]['priority'], 1)
            self.assertEqual(spec['files'][0]['role'], 'model')
            self.assertTrue(all(len(f['sha256']) == 64 and f['size'] > 0 for f in spec['files']))

    def test_effort_prompts_pairwise_distinct(self):
        for spec in ck.MODELS.values():
            texts = list(spec['effort_prompts'].values())
            self.assertEqual(set(spec['effort_prompts']), {'low', 'medium', 'high', 'xhigh'})
            self.assertEqual(len(set(texts)), 4)


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.nbs = {slug: ck.build_notebook_dict(spec, CFG) for slug, spec in ck.MODELS.items()}

    def test_cells_parse_and_prod_path_clean(self):
        for nb in self.nbs.values():
            for c in nb['cells']:
                if c['cell_type'] == 'code':
                    ast.parse(''.join(c['source']))
            ck._assert_prod_startup(_cell(nb, 'persistent-setup'))
            ck._selftest_boot_fp(_cell(nb, 'persistent-setup'))

    def test_runtime_spec_has_no_generator_only_keys(self):
        for slug, nb in self.nbs.items():
            spec = _runtime_spec(_cell(nb, 'persistent-setup'))
            self.assertEqual(spec['slug'], slug)
            self.assertFalse(set(ck._GEN_ONLY_KEYS) & set(spec))
            self.assertTrue(all('sha256' not in f for f in spec['files']))

    def test_shard_dir_follows_local_subdir(self):
        src = _cell(self.nbs['qwen38-fn-nl'], 'persistent-setup')
        self.assertNotIn("ROOT / 'IQ4_XS'", src)
        self.assertIn('st_blocks * 512 >= size', src)

    def test_hot_switch_lists_only_38fn(self):
        for nb in self.nbs.values():
            sw = _cell(nb, 'hot-switch')
            options = json.loads(sw.split('# @param ', 1)[1].split('\n', 1)[0])
            self.assertEqual(sorted(options), ['qwen38-fn', 'qwen38-fn-nl'])

    def test_no_other_model_traces(self):
        for nb in self.nbs.values():
            blob = json.dumps(nb, ensure_ascii=False)
            for gone in ('Qwopus', 'Nemotron', 'gemma-4', 'template_family', 'penalty_args'):
                self.assertNotIn(gone, blob)


class MagpieSyncTests(unittest.TestCase):
    def test_sync_keeps_models_outside_registry(self):
        listed = '  qwen-colab/Old-Model  low/medium\n  qwen-colab/Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS  low\n  other/x\n'
        calls = []

        def fake_run(argv, **kw):
            calls.append(argv[-1])
            return subprocess.CompletedProcess(argv, 0, stdout=listed if argv[-1].endswith('model list') else '')

        with mock.patch.object(ck.subprocess, 'run', side_effect=fake_run):
            ck.sync_magpie_models()
        replace = next(c for c in calls if 'provider models qwen-colab' in c).split()
        self.assertIn('Old-Model', replace)
        for spec in ck.MODELS.values():
            self.assertIn(spec['alias'], replace)
        self.assertEqual(len(replace), len(set(replace)))


if __name__ == '__main__':
    unittest.main()
