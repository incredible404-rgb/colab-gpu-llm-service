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
    'local_tailscale_ip': '192.0.2.1',
}


def _cell(nb, cell_id):
    return ''.join(next(c['source'] for c in nb['cells'] if c['metadata'].get('id') == cell_id))


def _runtime_spec(src):
    line = next(l for l in src.splitlines() if l.startswith('SPEC = json.loads('))
    return json.loads(ast.literal_eval(line[len('SPEC = json.loads('):-1]))


class RegistryTests(unittest.TestCase):
    def test_only_38fn_quants(self):
        self.assertEqual(list(ck.MODELS), ['qwen38-huihui-v2', 'gemma4-31b', 'qwen38-27b-rvn', 'glm53-flash'])

    def test_ple_shard_registered_and_first_to_tmpfs(self):
        for spec in ck.ALL_MODELS.values():
            if spec['ple_shard_name']:
                ple = [f for f in spec['files'] if f['path'].endswith(spec['ple_shard_name'])]
                self.assertEqual(len(ple), 1)
                self.assertEqual(ple[0]['priority'], 1)
            self.assertEqual(spec['files'][0]['role'], 'model')
            self.assertTrue(all(len(f['sha256']) == 64 and f['size'] > 0 for f in spec['files']))

    def test_effort_prompts_pairwise_distinct(self):
        for spec in ck.ALL_MODELS.values():
            texts = list(spec['effort_prompts'].values())
            self.assertEqual(set(spec['effort_prompts']), {'low', 'medium', 'high', 'xhigh'})
            self.assertEqual(len(set(texts)), 4)

    def test_archived_models_build_cleanly_on_latest_kernel(self):
        for slug, spec in ck.ARCHIVED_MODELS.items():
            nb = ck.build_notebook_dict(spec, CFG, with_switch=False)
            for c in nb['cells']:
                if c['cell_type'] == 'code':
                    ast.parse(''.join(c['source']))
            ck._assert_prod_startup(_cell(nb, 'persistent-setup'))


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
        src = _cell(self.nbs['qwen38-huihui-v2'], 'persistent-setup')
        self.assertNotIn("ROOT / 'IQ4_XS'", src)
        self.assertIn('st_blocks * 512 >= size', src)

    def test_hot_switch_lists_only_38fn(self):
        for nb in self.nbs.values():
            sw = _cell(nb, 'hot-switch')
            options = json.loads(sw.split('# @param ', 1)[1].split('\n', 1)[0])
            self.assertEqual(sorted(options), ['gemma4-31b', 'glm53-flash', 'qwen38-27b-rvn', 'qwen38-huihui-v2'])

    def test_unsloth_high_shim_stripped_before_effort_patch(self):
        src = _cell(self.nbs['qwen38-huihui-v2'], 'persistent-setup')
        tree = ast.parse(src)
        stmts = [ast.get_source_segment(src, n) for n in tree.body
                 if isinstance(n, (ast.Assign, ast.For, ast.If))
                 and any(tok in (ast.get_source_segment(src, n) or '')
                         for tok in ('universal_effort_block', 'fam =', 'old_effort_block'))]
        patch_code = '\n'.join(s for s in stmts if s)
        base_tmpl = (
            "{%- set reasoning_instructions = '' %}\n"
            "{%- if enable_thinking is undefined or enable_thinking is true %}\n"
            "    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n"
            "__SHIM__"
            "    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}\n"
            "        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}\n"
            "    {%- endif %}\n"
            "    {%- if resolved_reasoning_effort == 'xhigh' %}\n"
            "        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}\n"
            "    {%- elif resolved_reasoning_effort == 'low' %}\n"
            "        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}\n"
            "    {%- endif %}\n"
            "{%- endif %}\n"
            "{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}\n"
        )
        shim = (
            "    {%- if resolved_reasoning_effort == 'high' %}\n"
            "        {%- set resolved_reasoning_effort = 'xhigh' %}\n"
            "    {%- endif %}\n"
        )
        for sample in (base_tmpl.replace('__SHIM__', ''), base_tmpl.replace('__SHIM__', shim)):
            ns = {'SPEC': _runtime_spec(src), 'tmpl_str': sample}
            exec(patch_code, ns)
            self.assertIn("('xhigh', 'high', 'medium', 'low')", ns['tmpl_str'])
            self.assertNotIn("resolved_reasoning_effort = 'xhigh'", ns['tmpl_str'])

        gemma_tmpl = (
            "{%- set enable_thinking = enable_thinking | default(false) -%}\n"
            "    {%- if enable_thinking -%}\n"
            "        {{- '<|think|>\\n' -}}\n"
            "        {%- set ns.prev_message_type = 'think' -%}\n"
            "    {%- endif -%}\n"
        )
        ns_g = {'SPEC': _runtime_spec(_cell(self.nbs['gemma4-31b'], 'persistent-setup')), 'tmpl_str': gemma_tmpl}
        exec(patch_code, ns_g)
        self.assertIn('default(true)', ns_g['tmpl_str'])
        self.assertIn("reasoning_instructions + '\\n'", ns_g['tmpl_str'])

        glm_tmpl = (
            "[gMASK]<sop>\n"
            "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}\n"
            "{%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}\n"
            "{%- set clear_thinking = clear_thinking if clear_thinking is defined else false -%}\n"
        )
        ns_glm = {'SPEC': _runtime_spec(_cell(self.nbs['glm53-flash'], 'persistent-setup')), 'tmpl_str': glm_tmpl}
        exec(patch_code, ns_glm)
        self.assertIn("('xhigh', 'high', 'medium', 'low')", ns_glm['tmpl_str'])
        self.assertIn("'max' if resolved_reasoning_effort == 'xhigh'", ns_glm['tmpl_str'])
        self.assertIn("clear_thinking if clear_thinking is defined else true", ns_glm['tmpl_str'])

    def test_no_other_model_traces(self):
        for nb in self.nbs.values():
            blob = json.dumps(nb, ensure_ascii=False)
            for gone in ('Qwopus', 'Nemotron', 'Ortenzya'):
                self.assertNotIn(gone, blob)


class MagpieSyncTests(unittest.TestCase):
    def test_sync_keeps_models_outside_registry_and_prunes_archived(self):
        listed = '  qwen-colab/Old-Model  low/medium\n  qwen-colab/Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS  low\n  other/x\n'
        specs = '  qwen-colab/*  context 262144\n  qwen-colab/Custom-External  context 262144\n'
        calls = []

        def fake_run(argv, **kw):
            calls.append(argv[-1])
            if argv[-1].endswith('model list'):
                out = listed
            elif argv[-1].endswith('model specs'):
                out = specs
            else:
                out = ''
            return subprocess.CompletedProcess(argv, 0, stdout=out)

        with mock.patch.object(ck, '_magpie_launcher', return_value=('bash', '-lc')), \
                mock.patch.object(ck.subprocess, 'run', side_effect=fake_run):
            ck.sync_magpie_models()
        replace = next(c for c in calls if 'provider models qwen-colab' in c).split()
        self.assertIn('Old-Model', replace)
        self.assertIn('Custom-External', replace)
        self.assertNotIn('Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS', replace)
        self.assertNotIn('*', replace)
        for spec in ck.MODELS.values():
            self.assertIn(spec['alias'], replace)
        self.assertEqual(len(replace), len(set(replace)))


if __name__ == '__main__':
    unittest.main()
