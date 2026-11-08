"""Colab G4 kernel & notebook generator for Qwen3.8-Flash-Next (Swift-1.5 Heretic V2).

Single source of truth for the 3.8fn notebooks (IQ4_XS-plus `qwen38-fn`, IQ4_NL `qwen38-fn-nl`).
Other models were moved out on 2026-11-06; the multi-model kernel is frozen in
archive/multi-model-kernel/ and their last generated notebooks stay in notebooks/.
- Runtime kernel: llama.cpp b12032, Tailscale svc:qwen + SOCKS5 bridge to local Windows
  novel MCP (local_tailscale_ip:3426), persistent --ui-config-file + --ui-mcp-proxy, /dev/shm tmpfs
  allocator, 32-stream zero-GIL downloader, millisecond size check (no startup SHA-256),
  16 MiB GGUF template patcher, production path DEBUG_VERIFY=False (banner right after /health;
  CUDA warmup, tmpfs release and Tailscale finish in the background; LAUNCH_DETACHED=1 returns
  once llama-server is spawned and that same daemon thread waits for /health before warmup),
  per-model-card sampling, post-load release of GPU-only tmpfs copies into a larger prompt cache,
  in-runtime hot switch between the registered quants, 7-cell layout, and 45-min idle guard.
- Auto-distribution: generates notebooks/<name>.ipynb, mirrors to G:/My Drive/Colab Notebooks/,
  auto-resolves new Google Drive cloud_id from DriveFS SQLite DB into .secrets.json, AST-validates
  all code cells, and syncs to Colab Web (--sync-colab) and Magpie (--sync-magpie, additive).

Usage:
  python scripts/colab_kernel.py --all [--sync-colab] [--sync-magpie]
  python scripts/colab_kernel.py <slug> [--sanitized] [--output-dir DIR] [--sync-colab]
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_sync import sync_requested
from secrets_config import FIELDS, SECRETS_FILE, guard_sync, load_config, report
from webui_console import console_code

ROOT_DIR = Path(__file__).resolve().parents[1]
DRIVE_NB_DIR = Path('G:/My Drive/Colab Notebooks')
LOCAL_MCP_PORT = 3426


def _drivefs_db() -> Path | None:
    """DriveFS metadata DB of this Windows account (DRIVEFS_DB overrides; else the most recently used profile)."""
    if os.environ.get('DRIVEFS_DB'):
        return Path(os.environ['DRIVEFS_DB'])
    base = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData/Local'))) / 'Google/DriveFS'
    dbs = sorted(base.glob('*/metadata_sqlite_db'), key=lambda p: p.stat().st_mtime, reverse=True)
    return dbs[0] if dbs else None

# 3.8fn 注册表：同一份权重的不同量化，采样/模板/MTP 共用。
MODELS: dict[str, dict] = {
    'qwen38-fn': {
        'slug': 'qwen38-fn',
        'notebook': 'qwen3.8-flash-next-g4-colab.ipynb',
        'drive_key': 'g4_drive_id',
        'env_key': 'G4_DRIVE_ID',
        'placeholder': '<YOUR_G4_NOTEBOOK_DRIVE_ID>',
        'alias': 'Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS',
        'title': 'Swift-1.5-Qwen3.8-Flash-Next · Colab G4 中文长篇小说共创与全能主力服务（Heretic V2 原生破限 IQ4_XS-plus）',
        'banner': '🔓 Heretic V2 原生破限 125B，满速运行',
        'root_dir': '/content/qwen38-fn',
        'ts_hostname': 'qwen-g4-colab',
        'min_vram_mib': 85000,
        'ram_reserve_gib': 95,  # Reserve 95 GiB for 26.82 GiB PLE MAP_POPULATE + CUDA pinned buffers
        'ple_shard_name': 'Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00002-of-00006.gguf',
        # 采样按 Qwen3.8-Flash-Next 模型卡「思考模式」默认值（Swift-1.5 仓库没另给）。
        'temp': '1.0',
        'top_p': '0.95',
        'top_k': '20',
        'min_p': '0.0',
        # 官方模板默认 xhigh，low 不关思考只要求想得短；high 是我们补的一档。
        'reasoning_effort_default': 'xhigh',
        'effort_prompts': {'xhigh': '用中文思考，尽量深入。', 'high': '用中文思考。', 'medium': '用中文思考，简短些。', 'low': '用中文思考，越短越好。'},
        'mtp_n_max': 3,
        'mtp_p_min': '0.48',
        'files': [
            {
                'role': 'model',
                'repo': 'jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer',
                'rev': '76208ff7308ccebd887eec5293d3cab305091514',
                'path': 'IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00001-of-00006.gguf',
                'local_subdir': 'IQ4_XS',
                'size': 539_422_592,
                'sha256': 'b963118b880fe3b7413e53b474c1d8d3faefc1ae0d2d6bae7a7c3e9f68217b46',
                'priority': 2,
            },
            {
                'role': 'shard',
                'repo': 'jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer',
                'rev': '76208ff7308ccebd887eec5293d3cab305091514',
                'path': 'IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00002-of-00006.gguf',
                'local_subdir': 'IQ4_XS',
                'size': 28_800_138_432,
                'sha256': '0e9027aa7de4817c91b5954039b563484466392e29a6b051b096f9c47a71a1c6',
                'priority': 1,  # PLE table shard -> top priority for /dev/shm tmpfs
            },
            {
                'role': 'shard',
                'repo': 'jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer',
                'rev': '76208ff7308ccebd887eec5293d3cab305091514',
                'path': 'IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00003-of-00006.gguf',
                'local_subdir': 'IQ4_XS',
                'size': 16_681_220_224,
                'sha256': '95c25e4ada4fa9ce4380683e55fb9f55aa8e411dcca13bec789f3cc106ffd403',
                'priority': 3,
            },
            {
                'role': 'shard',
                'repo': 'jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer',
                'rev': '76208ff7308ccebd887eec5293d3cab305091514',
                'path': 'IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00004-of-00006.gguf',
                'local_subdir': 'IQ4_XS',
                'size': 16_984_320_096,
                'sha256': '5458253ef234558243c522c8d5289b923fe28601ec6aca26239de14f03ed1951',
                'priority': 3,
            },
            {
                'role': 'shard',
                'repo': 'jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer',
                'rev': '76208ff7308ccebd887eec5293d3cab305091514',
                'path': 'IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00005-of-00006.gguf',
                'local_subdir': 'IQ4_XS',
                'size': 16_662_985_152,
                'sha256': 'ae9f3f684493f0107f0aa01dd603844266ee8b9a53bad53475761f2acc31202a',
                'priority': 3,
            },
            {
                'role': 'shard',
                'repo': 'jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer',
                'rev': '76208ff7308ccebd887eec5293d3cab305091514',
                'path': 'IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00006-of-00006.gguf',
                'local_subdir': 'IQ4_XS',
                'size': 15_036_859_296,
                'sha256': 'bcebb4fecfaaf887c2a763c252ecca17749efe175e6c9c1c45d555c0e68906c0',
                'priority': 3,
            },
            {
                'role': 'mmproj',
                'repo': 'ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF',
                'rev': '0142554c963a8653912c0c8283f05f13f63653f3',
                'path': 'mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf',
                'size': 907_543_520,
                'sha256': 'cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60',
                'priority': 2,
            },
            {
                'role': 'draft',
                'repo': 'unsloth/Qwen3.8-Flash-Next-GGUF',
                'rev': '766911a6b7369840a91dbcd95f9f997acaab6cd6',
                'path': 'MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf',
                'size': 2_786_204_800,
                'sha256': '8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231',
                'priority': 2,
            },
        ],
    },
}

# 同一份 Swift-1.5 Heretic V2 权重的 IQ4_NL 量化：热尾层 down 近 8 bit、44–46 层 gate/up 5 bit，显存部分 80.1 GB
# （IQ4_XS-plus 为 65.9 GB）。采样、模板、MTP 与现版完全一致，只换权重，方便和 qwen38-fn 做 A/B；
# 第 2 片同样只装 per_layer_token_embd（N-gram 表），所以沿用常驻内存盘的处理。
_NL_REPO = 'jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer'
_NL_REV = 'bd3bca068a544d459cf42902bf21a0fac00ec853'
_NL_NAME = 'Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_NL-{:05d}-of-00008.gguf'
_NL_SHARDS = [
    (3_208_309_664, '9344341eab2c2d9d59c8a4a9533c8cfc515547675d7a139d17957968a3019c5b'),
    (28_800_138_432, '56a185873792b560485ee971ba5c4f9c5558def5e3e70dbf0f8925e9849eb513'),
    (13_700_060_352, 'e6c22d9c0a22e39fb25decc81d8862667d8184907c623b182d9e2771ebfb9e03'),
    (13_766_580_576, 'a8e29ba699751bf05250824b7075aee2383048b39290ab55fb50c677925a1a37'),
    (13_643_236_928, 'afa6f9afdeccef869b8f9ed1b3800246a626a9ff4e34859926800ed17e0cbdec'),
    (13_670_914_048, 'da51b3a9d3f485a54e69f16338f7f1dc21f0032b470116c75c508f1422446848'),
    (13_751_645_536, 'a232c44cfc9fe5555ab5dde1c629120eada854f598a8c26247aac552b68697c2'),
    (8_352_778_560, '2148d4cf9c7d03c953ec3c95fc28f23cbad13547be4ebfd4ecf4d590d1498cd3'),
]
MODELS['qwen38-fn-nl'] = {
    **MODELS['qwen38-fn'],
    'slug': 'qwen38-fn-nl',
    'notebook': 'qwen3.8-flash-next-nl-g4-colab.ipynb',
    'drive_key': 'g4nl_drive_id',
    'env_key': 'G4NL_DRIVE_ID',
    'placeholder': '<YOUR_G4NL_NOTEBOOK_DRIVE_ID>',
    'alias': 'Swift-1.5-Qwen3.8-Flash-Next-IQ4_NL',
    'title': 'Swift-1.5-Qwen3.8-Flash-Next · Colab G4 中文长篇小说共创与全能主力服务（Heretic V2 原生破限 IQ4_NL 高精度版）',
    'banner': '🔓 Heretic V2 原生破限 125B · IQ4_NL 高精度版，满速运行',
    'root_dir': '/content/qwen38-fn-nl',
    'ts_hostname': 'qwen-g4nl-colab',
    'ple_shard_name': _NL_NAME.format(2),
    'files': [
        {
            'role': 'model' if i == 1 else 'shard',
            'repo': _NL_REPO,
            'rev': _NL_REV,
            'path': _NL_NAME.format(i),
            'local_subdir': 'IQ4_NL',
            'size': size,
            'sha256': sha,
            'priority': 1 if i == 2 else (2 if i == 1 else 3),
        }
        for i, (size, sha) in enumerate(_NL_SHARDS, start=1)
    ] + [f for f in MODELS['qwen38-fn']['files'] if f['role'] in ('mmproj', 'draft')],
}

# effort_prompts 四档文本必须两两不同：llama.cpp 靠渲染差异探测 reasoning_effort_levels，/props 门禁要求报齐四档。
# 只在生成端用、不写进首格 SPEC 的登记项。
_GEN_ONLY_KEYS = ('notebook', 'drive_key', 'env_key', 'placeholder', 'title', 'banner')

WEBUI_SYSTEM_PROMPT = (
    '你是中文长篇小说共创搭档。\n'
    '1. 开了 local_novel_fs 时，动笔前先读 写作要求.md 和正文末尾（tail_lines=150）。新章或下半章用 write_novel_file 追加；重写上一轮用 replace_last_write，重写整章用 replace_last_chapter；小改用 edit_novel_file；撤回用 undo_novel_write。没开工具就直接在对话里写。\n'
    '2. 查原作用 grep_reference；设定冲突以用户当前指令为准。\n'
    '3. 用户说「收工」时，用 edit_novel_file 把新定的设定、进度和伏笔更新进 写作要求.md，只改变化项。\n'
    '4. 用户点名时才读技能：grill/拷问 读 skill:grilling，优化提示词 读 skill:prompting，retro/复盘 读 skill:retro。'
)


def _lines(text: str) -> list[str]:
    raw = text.strip('\n').split('\n')
    return [line + '\n' for line in raw]


def _ensure_drive_id(spec: dict, config: dict, sanitized: bool) -> str:
    key = spec['drive_key']
    if key not in FIELDS:
        FIELDS[key] = (spec['env_key'], spec['placeholder'])
    val = config.get(key)
    if sanitized or (val and not val.startswith('<YOUR_')):
        return val or spec['placeholder']
    drivefs_db = _drivefs_db()
    if drivefs_db is None or not drivefs_db.exists():
        return spec['placeholder']
    nb_name = spec['notebook']
    for _ in range(15):
        conn = sqlite3.connect(f'file:{drivefs_db.as_posix()}?mode=ro', uri=True)
        cur = conn.cursor()
        cur.execute(
            '''
            SELECT s.cloud_id FROM items i
            JOIN stable_ids s ON i.stable_id = s.stable_id
            JOIN stable_parents sp ON i.stable_id = sp.item_stable_id
            JOIN items p ON sp.parent_stable_id = p.stable_id
            WHERE i.local_title = ? AND p.local_title = 'Colab Notebooks' AND i.trashed = 0
            ''',
            (nb_name,),
        )
        row = cur.fetchone()
        conn.close()
        if row and row[0] and not row[0].startswith('local_'):
            cloud_id = row[0]
            if SECRETS_FILE.exists():
                sec = json.loads(SECRETS_FILE.read_text(encoding='utf-8'))
                sec[key] = cloud_id
                SECRETS_FILE.write_text(json.dumps(sec, indent=2) + '\n', encoding='utf-8', newline='')
            config[key] = cloud_id
            return cloud_id
        time.sleep(1)
    return spec['placeholder']


_PROD_FORBIDDEN = ('/props', 'sha256', 'spell')
_FOUR_EFFORTS = {'low', 'medium', 'high', 'xhigh'}


def _is_debug_test(test: ast.AST) -> bool:
    return isinstance(test, ast.Name) and test.id == 'DEBUG_VERIFY'


def _assert_prod_startup(src: str) -> list[str]:
    """DEBUG_VERIFY=False 路径不得出现 /props、sha256、spell 或四档推理请求。"""
    tree = ast.parse(src)
    if not tree.body or not isinstance(tree.body[0], ast.Assign):
        raise RuntimeError('首格第一条语句必须是 DEBUG_VERIFY = False')
    first = tree.body[0]
    if not (
        len(first.targets) == 1
        and isinstance(first.targets[0], ast.Name)
        and first.targets[0].id == 'DEBUG_VERIFY'
        and isinstance(first.value, ast.Constant)
        and first.value.value is False
    ):
        raise RuntimeError('首格 DEBUG_VERIFY 必须默认 False')

    class _V(ast.NodeVisitor):
        def __init__(self):
            self.in_debug = 0
            self.http: list[str] = []

        def visit_If(self, node):
            if _is_debug_test(node.test):
                self.in_debug += 1
                try:
                    for stmt in node.body:
                        self.visit(stmt)
                finally:
                    self.in_debug -= 1
                for stmt in node.orelse:
                    self.visit(stmt)
                return
            self.generic_visit(node)

        def visit_Constant(self, node):
            if self.in_debug or not isinstance(node.value, str):
                return
            for bad in _PROD_FORBIDDEN:
                if bad in node.value:
                    raise RuntimeError('DEBUG_VERIFY=False 路径出现 %s: %s' % (bad, node.value[:160]))
            if node.value.startswith('http') or '/v1/' in node.value:
                self.http.append(node.value)

        def visit_Attribute(self, node):
            if not self.in_debug and node.attr in ('sha256', 'spell'):
                raise RuntimeError('DEBUG_VERIFY=False 路径出现 .%s' % node.attr)
            self.generic_visit(node)

        def visit_For(self, node):
            if not self.in_debug and isinstance(node.iter, (ast.Tuple, ast.List, ast.Set)):
                vals = [e.value for e in node.iter.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                if _FOUR_EFFORTS.issubset(vals):
                    blob = ast.dump(node)
                    if 'completions' in blob or 'chat' in blob:
                        raise RuntimeError('DEBUG_VERIFY=False 路径出现四档推理请求')
            self.generic_visit(node)

    checker = _V()
    checker.visit(tree)
    completions = [u for u in checker.http if 'chat/completions' in u]
    if completions != ['http://127.0.0.1:%d/v1/chat/completions']:
        raise RuntimeError('生产路径推理 URL 应只有一次预热，实际 %s' % completions)
    return checker.http


def _selftest_boot_fp(src: str) -> None:
    tree = ast.parse(src)
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('_fp_args', '_boot_fp'):
            seg = ast.get_source_segment(src, node)
            if not seg:
                raise RuntimeError('无法提取 %s' % node.name)
            chunks.append(seg)
    if len(chunks) != 2:
        raise RuntimeError('首格缺少 _fp_args/_boot_fp')
    ns: dict = {}
    exec('import hashlib\nfrom pathlib import Path\n' + '\n\n'.join(chunks), ns)
    args = ['-m', 'model.gguf', '--temp', '0.8', '--cache-ram', '100', '--port', '8081']
    base = ns['_boot_fp'](args, 'TMPL', 'UI')
    other_cache = ns['_boot_fp'](['-m', 'model.gguf', '--temp', '0.8', '--cache-ram', '99999', '--port', '8081'], 'TMPL', 'UI')
    other_tmpl = ns['_boot_fp'](args, 'TMPL2', 'UI')
    with_bin = ns['_boot_fp'](['/opt/llama-server', *args], 'TMPL', 'UI')
    if not (base == other_cache == with_bin and base != other_tmpl):
        raise RuntimeError('启动指纹不符合预期: %s %s %s %s' % (base, other_cache, other_tmpl, with_bin))
    sample_ui = 'U' * 2000
    sample_tmpl = 'T' * 8000
    t0 = time.perf_counter()
    for _ in range(200):
        ns['_boot_fp'](args, sample_tmpl, sample_ui)
    elapsed_ms = (time.perf_counter() - t0) / 200 * 1000
    if elapsed_ms > 5:
        raise RuntimeError('启动指纹过慢: %.2fms' % elapsed_ms)
    print('boot fingerprint %.3f ms (sha1, cache-ram ignored)' % elapsed_ms)


def build_notebook_dict(spec: dict, config: dict, with_switch: bool = True) -> dict:
    fqdn = config['tailnet_fqdn']
    api_key = config['api_key']
    ts_id = config['ts_client_id']
    ts_secret = config['ts_client_secret']
    alias = spec['alias']
    slug = spec['slug']
    root_dir = spec['root_dir']

    overview_md = f"""# {spec['title']}

- **模型别名**：`{alias}`（262,144 上下文 / 65,536 最大输出 / 16–64 GiB Prompt Cache + 槽位持久化）
- **内存回收**：权重进显存后删掉内存盘里只供显存用的分片副本（生产模式在后台做），空出的内存一半给 Prompt Cache；服务不重启就不用重新下载。
- **启动**：首格 `DEBUG_VERIFY = False`（调试新模型/改配置时设 True）。生产模式在 `/health` 返回 200 后立刻出就绪横幅；CUDA 预热、内存盘副本释放和 Tailscale 收尾放到后台。调试模式才做 /props 四档门禁、阻塞预热和额外大小核对。不跑拼写探针，也不做 SHA-256。
- **热切换**：“热切换模型”格不释放 G4 就换跑另一个已登记模型，省掉重新分配运行时和装 llama.cpp / Tailscale 的时间。
- **采样参数**：`--temp {spec['temp']} --top-p {spec['top_p']} --top-k {spec['top_k']} --min-p {spec['min_p']}`（按模型卡官方默认值）
- **WebUI 持久化与本机文件直通**：服务端内置 `--ui-config-file` 与 `--ui-mcp-proxy`，自动加载小说共创 System Prompt、关闭后台 LLM 标题生成防缓存冲刷，并通过 Tailscale SOCKS5 隧道直连本机 `~/Documents/Colab创作`（`local_novel_fs` 工具可在 WebUI 输入框一键开关）。
- **固定入口（所有笔记本分时复用 `svc:qwen`）**：
  - 网页：`http://{fqdn}:443/`（首格就绪自动弹出，点首格停机按钮自动关闭）
  - 接口：`http://{fqdn}:443/v1`
  - 运行时目录：`{root_dir}`
"""

    # b12032 WebUI（tools/ui）采样键是顶层蛇形，不是嵌套的 samplers 对象：
    # repeat_penalty / repeat_last_n / presence_penalty / frequency_penalty /
    # dry_multiplier / dry_base / dry_allowed_length / dry_penalty_last_n，另加采样链顺序键 samplers。
    # --ui-config-file 只在浏览器第一次访问（localStorage 还没有配置）时写入；
    # 之后 localStorage 的 userOverrides 会盖过服务端 CLI。采样项留空时请求不带这些字段，由 llama-server CLI 决定。
    # 这里不把采样数值写进 ui 配置，只由 CLI 决定；浏览器里存过采样覆盖或旧系统提示时，
    # 要在 WebUI 点 Reset to Default，或清掉该站点的 llama.cpp 配置 localStorage，否则请求体仍会覆盖 CLI。
    ui_cfg_dict = {
        'systemMessage': WEBUI_SYSTEM_PROMPT,
        'showSystemMessage': True,
        'showMessageStats': True,
        'enableContinueGeneration': True,
        'pasteLongTextToFileLen': 50000,
        'excludeReasoningFromContext': True,
        'renderThinkingAsMarkdown': True,
        'renderUserContentAsMarkdown': True,
        'titleGenerationUseLLM': False,
        'titleGenerationUseFirstLine': True,
        'mcpServers': json.dumps(
            [
                {
                    'id': 'local-novel-fs',
                    'name': 'local_novel_fs',
                    'displayName': '本机小说目录 (Colab创作)',
                    'url': f'http://127.0.0.1:{LOCAL_MCP_PORT}/mcp',
                    'enabled': True,
                    'useProxy': True,
                }
            ],
            ensure_ascii=False,
        ),
    }

    runtime_spec = {k: v for k, v in json.loads(json.dumps(spec)).items() if k not in _GEN_ONLY_KEYS}
    for item in runtime_spec['files']:
        item.pop('sha256', None)  # 启动不再哈希；登记表里的摘要留在 MODELS，不写进首格
    spec_json = json.dumps(runtime_spec, ensure_ascii=False, indent=2)
    ui_cfg_json = json.dumps(ui_cfg_dict, ensure_ascii=False, indent=2)

    oneclick_code = f"""# 一键启动（由 colab_kernel.py 统一内核生成）：tmpfs 内存盘 + 32路下载 + 字节大小校验 + WebUI 本机文件桥 + Tailscale
# 调试新模型/改配置时设 True
DEBUG_VERIFY = False
import concurrent.futures, hashlib, json, os, shutil, socket, struct, subprocess, sys, threading, time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter

t_boot = time.monotonic()
SPEC = json.loads({json.dumps(spec_json)})
UI_CFG_DATA = json.loads({json.dumps(ui_cfg_json)})

ROOT = Path(SPEC['root_dir'])
DIST = ROOT / 'llama-dist'
MODEL_ALIAS = SPEC['alias']
KEY_FILE = ROOT / '.llama-api-key'
PORT = 8081
KEY = {json.dumps(api_key)}
FILES = SPEC['files']
CPP_TAG = 'b12032'
CPP_URL = 'https://github.com/ggml-org/llama.cpp/releases/download/' + CPP_TAG

def _local_path(f):
    sub = f.get('local_subdir')
    return (ROOT / sub / Path(f['path']).name) if sub else (ROOT / Path(f['path']).name)

MODEL = _local_path(FILES[0])
MMPROJ = next((_local_path(f) for f in FILES if f['role'] == 'mmproj'), None)
DRAFT = next((_local_path(f) for f in FILES if f['role'] == 'draft'), None)
# PLE 表所在目录跟着各模型的 local_subdir 走（IQ4_XS / IQ4_NL …），不能写死。
SHARD_DIR = next((_local_path(f).parent for f in FILES if Path(f['path']).name == SPEC.get('ple_shard_name')), ROOT)

# 服务把权重读进显存后，/dev/shm 里只供显存用的副本就是冗余内存；删掉后靠这份记录（绑定服务 PID）在重跑本格时跳过重新下载。
RELEASE_STATE = ROOT / 'released-copies.json'
HEAD_CACHE = ROOT / 'model-head.bin'
TMPFS_PREFIXES = ('/dev/shm/', '/var/colab/')
FILE_NAMES = {{_local_path(f).name for f in FILES}}

def _server_proc():
    # 本模型的 llama-server：(pid, argv)；没有则 (None, [])
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdigit():
            continue
        try:
            args = (procdir / 'cmdline').read_bytes().decode(errors='replace').split('\\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if args and Path(args[0]).name == 'llama-server' and str(MODEL) in args:
            return int(procdir.name), args
    return None, []

def _server_pid():
    return _server_proc()[0]

def _proc_identity(pid):
    # boot_id + 进程启动时刻，防止重启后的同模型服务恰好复用旧 PID 时误用释放记录
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        stat = Path('/proc/%d/stat' % pid).read_text()
        return boot + ':' + stat.rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError, TypeError):
        return None

def _cache_ram_of(args):
    try:
        return int(args[args.index('--cache-ram') + 1])
    except (ValueError, IndexError):
        return None

def _stop_llama_servers():
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdigit():
            continue
        try:
            args = (procdir / 'cmdline').read_bytes().decode(errors='replace').split('\\0')
            if args and Path(args[0]).name == 'llama-server':
                os.kill(int(procdir.name), 15)
                for _ in range(60):
                    stat = (procdir / 'stat').read_text() if procdir.exists() else ''
                    if not stat or stat.split(') ', 1)[1].startswith('Z'):
                        break
                    time.sleep(0.5)
                else:
                    os.kill(int(procdir.name), 9)
                    time.sleep(2)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    old_proc = globals().get('qwen_server')
    if old_proc is not None:
        try:
            old_proc.wait(timeout=5)
        except Exception:
            pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        alive = []
        for procdir in Path('/proc').iterdir():
            try:
                if procdir.name.isdigit() and Path((procdir / 'cmdline').read_bytes().decode(errors='replace').split('\\0')[0]).name == 'llama-server':
                    alive.append(procdir.name)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        if not alive:
            return
        time.sleep(1)
    raise RuntimeError('旧 llama-server 停不掉（PID ' + ', '.join(alive) + '），已中止；请用停机按钮释放运行时后重来。')

def _mem_available():
    for l in Path('/proc/meminfo').read_text().splitlines():
        if l.startswith('MemAvailable:'):
            return int(l.split()[1]) * 1024
    return 0

SERVICE = 'svc:qwen'
SERVICE_FQDN = {json.dumps(fqdn)}
SERVICE_TAG = 'tag:colab'
TS_HOSTNAME = SPEC['ts_hostname']
TS_CLIENT_ID = {json.dumps(ts_id)}
TS_CLIENT_SECRET = {json.dumps(ts_secret)}
TS = '/content/tsbin/tailscale'
TS_SOCK = '/var/run/tailscale/tailscaled.sock'
TS_STATE = Path('/content/tailscaled.state')
TS_DIR = Path('/content/tsstatedir')

def sh(c):
    subprocess.run(c, shell=True, check=True)

def health():
    try:
        return requests.get('http://127.0.0.1:%d/health' % PORT, timeout=2).status_code == 200
    except Exception:
        return False

gpu_info = subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits').strip()
if not gpu_info or ',' not in gpu_info:
    raise RuntimeError('未检测到 NVIDIA GPU，请在右上角切换到 G4 GPU 运行时。输出：' + gpu_info)
gpu_name, gpu_mib_str = [x.strip() for x in gpu_info.split(',', 1)]
gpu_mib = int(gpu_mib_str)
print(f'检测到 GPU: {{gpu_name}} ({{gpu_mib / 1024:.1f}} GiB VRAM)', flush=True)
if gpu_mib < SPEC['min_vram_mib']:
    raise RuntimeError(f'当前 GPU 显存仅 {{gpu_mib / 1024:.1f}} GiB（{{gpu_name}}），低于要求的 {{SPEC["min_vram_mib"] / 1024:.0f}} GiB，请切换到 G4 高内存运行时。')

ROOT.mkdir(parents=True, exist_ok=True)
for _d in sorted(set(_local_path(f).parent for f in FILES)):
    _d.mkdir(parents=True, exist_ok=True)
DIST.mkdir(parents=True, exist_ok=True)
KEY_FILE.write_text(KEY + '\\n', encoding='utf-8', newline='')
KEY_FILE.chmod(0o600)
AUTH = {{'Authorization': 'Bearer ' + KEY}}

def prepare_llama_cpp():
    if next(DIST.rglob('llama-server'), None) is not None:
        return
    print(f'[并行任务] 下载并解压 llama.cpp {{CPP_TAG}} (CUDA 12.8) ...', flush=True)
    sh(f'cd {{DIST}} && '
       f'curl -fsSL -O {{CPP_URL}}/llama-{{CPP_TAG}}-bin-ubuntu-cuda-12.8-x64.tar.gz && '
       f'curl -fsSL -O {{CPP_URL}}/cudart-llama-{{CPP_TAG}}-bin-ubuntu-cuda-12.8-x64.tar.gz && '
       f'tar -xzf llama-{{CPP_TAG}}-bin-ubuntu-cuda-12.8-x64.tar.gz && '
       f'tar -xzf cudart-llama-{{CPP_TAG}}-bin-ubuntu-cuda-12.8-x64.tar.gz && '
       f'rm -f *.tar.gz')
    print(f'[并行任务] llama.cpp {{CPP_TAG}} 就绪', flush=True)

def ts(*a, timeout=60):
    return subprocess.run([TS, '--socket=' + TS_SOCK, *a],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)

def ts_state():
    try:
        return json.loads(ts('status', '--json').stdout).get('BackendState', '')
    except Exception:
        return ''

def _start_mcp_bridge():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(('127.0.0.1', {LOCAL_MCP_PORT}))
    except OSError:
        return
    srv.listen(32)
    def _pipe(a, b):
        try:
            while (d := a.recv(65536)):
                b.sendall(d)
        except Exception:
            pass
        finally:
            try: a.close()
            except Exception: pass
            try: b.close()
            except Exception: pass
    while True:
        try:
            cli, _ = srv.accept()
            up = socket.create_connection(('127.0.0.1', 1055), timeout=10)
            up.sendall(bytes([5, 1, 0]))
            assert up.recv(2) == bytes([5, 0])
            up.sendall(bytes([5, 1, 0, 1]) + socket.inet_aton({json.dumps(config['local_tailscale_ip'])}) + struct.pack('>H', {LOCAL_MCP_PORT}))
            assert up.recv(10)[1] == 0
            threading.Thread(target=_pipe, args=(cli, up), daemon=True).start()
            threading.Thread(target=_pipe, args=(up, cli), daemon=True).start()
        except Exception:
            pass

def prepare_tailscale():
    if not Path(TS).exists():
        sh('curl -fsSL https://pkgs.tailscale.com/stable/tailscale_1.104.1_amd64.tgz | tar -xz -C /content && '
           'mv /content/tailscale_1.104.1_amd64 /content/tsbin')
    TS_DIR.mkdir(parents=True, exist_ok=True)
    Path('/var/run/tailscale').mkdir(parents=True, exist_ok=True)
    if ts_state() in ('', 'NoState', 'Stopped'):
        subprocess.Popen(
            ['/content/tsbin/tailscaled', '--tun=userspace-networking',
             '--socks5-server=localhost:1055', '--state=' + str(TS_STATE),
             '--statedir=' + str(TS_DIR), '--socket=' + TS_SOCK],
            stdout=open('/content/tailscaled.log', 'a', encoding='utf-8', newline=''),
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        for _ in range(40):
            if Path(TS_SOCK).exists():
                break
            time.sleep(0.5)
        for _ in range(40):
            if ts_state() in ('Running', 'NeedsLogin'):
                break
            time.sleep(0.5)
    if ts_state() != 'Running':
        token = requests.post(
            'https://api.tailscale.com/api/v2/oauth/token',
            data={{'client_id': TS_CLIENT_ID, 'client_secret': TS_CLIENT_SECRET}}, timeout=30
        ).json()['access_token']
        key = requests.post(
            'https://api.tailscale.com/api/v2/tailnet/-/keys',
            headers={{'Authorization': 'Bearer ' + token}},
            json={{'capabilities': {{'devices': {{'create': {{
                'reusable': False, 'ephemeral': True, 'preauthorized': True, 'tags': [SERVICE_TAG]
            }}}}}}}}, timeout=30
        ).json()['key']
        ts('up', '--auth-key=' + key, '--hostname=' + TS_HOSTNAME,
           '--advertise-tags=' + SERVICE_TAG, '--accept-dns=false', '--timeout=60s', timeout=120)
        for _ in range(120):
            if ts_state() == 'Running':
                break
            time.sleep(0.5)
    ts('serve', 'reset')
    ts('serve', '--bg', '--service=' + SERVICE, '--tcp=443', 'tcp://127.0.0.1:%d' % PORT, timeout=90)
    ts('serve', '--bg', '--https=443', 'http://127.0.0.1:%d' % PORT, timeout=90)
    threading.Thread(target=_start_mcp_bridge, daemon=True).start()
    print(f'[并行任务] Tailscale Service ({{SERVICE_FQDN}}) 与本机小说文件 MCP 隧道已就绪', flush=True)

bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
fut_cpp = bg_executor.submit(prepare_llama_cpp)
fut_ts = bg_executor.submit(prepare_tailscale)

subprocess.run(['mount', '-o', 'remount,size=115G', '/dev/shm'], capture_output=True)
tmpfs_dirs = []
for mline in Path('/proc/mounts').read_text().splitlines():
    mparts = mline.split()
    if len(mparts) >= 3 and mparts[2] == 'tmpfs' and mparts[1] in ('/dev/shm', '/var/colab'):
        tmpfs_dirs.append(Path(mparts[1]))
mp_free = {{mp: shutil.disk_usage(mp).free for mp in tmpfs_dirs}}
meminfo_txt = Path('/proc/meminfo').read_text()
avail_kb = next((int(l.split()[1]) for l in meminfo_txt.splitlines() if l.startswith('MemAvailable:')), 0)
ram_budget = max(0, avail_kb * 1024 - SPEC['ram_reserve_gib'] * (1024**3))

released_reuse = set()
if RELEASE_STATE.exists():
    try:
        rel_state = json.loads(RELEASE_STATE.read_text(encoding='utf-8'))
        rel_ok = (isinstance(rel_state, dict) and isinstance(rel_state.get('pid'), int)
                  and isinstance(rel_state.get('files'), list)
                  and all(isinstance(n, str) and n in FILE_NAMES for n in rel_state['files']))
    except Exception:
        rel_state, rel_ok = {{}}, False
    pid_now = _server_pid()
    if rel_ok and pid_now == rel_state['pid'] and rel_state.get('proc_id') and _proc_identity(pid_now) == rel_state['proc_id']:
        released_reuse = set(rel_state['files'])
        print(f'服务仍在运行：{{len(released_reuse)}} 个已释放内存副本的分片无需重新下载', flush=True)
    else:
        if not rel_ok:
            print('⚠️ released-copies.json 无法识别，缺失的分片会重新下载', flush=True)
        RELEASE_STATE.unlink(missing_ok=True)
if not released_reuse and _server_pid() and any(not _local_path(f).exists() for f in FILES):
    # 记录不可用、服务还在且文件已删：先停服务再补下载，避免服务占着内存时又往内存盘塞几十 GiB。
    print('⚠️ 服务在运行，但分片副本已删且释放记录不可用：先停掉旧服务，再重新下载。', flush=True)
    _stop_llama_servers()

sorted_files = sorted(FILES, key=lambda x: x.get('priority', 2))
all_targets = [
    ('https://huggingface.co/' + f['repo'] + '/resolve/' + f['rev'] + '/' + f['path'],
     _local_path(f), f['size'])
    for f in sorted_files
]
downloads = []
tmpfs_bytes = 0
disk_bytes = 0
for orig_url, logical_path, size in all_targets:
    if logical_path.name in released_reuse:
        continue
    if logical_path.exists() and logical_path.stat().st_size == size:
        continue
    if logical_path.is_symlink() or logical_path.exists():
        logical_path.unlink(missing_ok=True)
    real_path = logical_path
    if ram_budget >= size:
        for mp in tmpfs_dirs:
            if mp_free.get(mp, 0) >= size + 4 * (1024**3):
                cand = mp / SPEC['slug'] / logical_path.name
                cand.parent.mkdir(parents=True, exist_ok=True)
                # 只看大小会把 truncate 出来的稀疏占位文件（下载中断、上次报错）当成已下载；tmpfs 的 st_blocks 是实际占用。
                if cand.exists() and cand.stat().st_size == size and cand.stat().st_blocks * 512 >= size:
                    logical_path.symlink_to(cand)
                    mp_free[mp] -= size
                    ram_budget -= size
                    real_path = None
                    break
                cand.unlink(missing_ok=True)
                with cand.open('wb') as f:
                    f.truncate(size)
                logical_path.symlink_to(cand)
                mp_free[mp] -= size
                ram_budget -= size
                tmpfs_bytes += size
                real_path = cand
                break
    if real_path is None:
        continue
    if real_path == logical_path:
        free_disk = shutil.disk_usage('/content').free
        if free_disk < disk_bytes + size + 6 * (1024**3):
            raise RuntimeError('本地 /content 与 tmpfs 空间不足，请重建运行时。')
        with real_path.open('wb') as f:
            f.truncate(size)
        disk_bytes += size
    downloads.append((orig_url, str(real_path), size))

freshly_downloaded = bool(downloads)
if downloads:
    need_bytes = sum(size for _, _, size in downloads)
    print(f'待并发下载: {{need_bytes / (1024**3):.2f}} GiB ({{len(downloads)}} 个文件) → DDR5 内存盘 tmpfs: {{tmpfs_bytes / (1024**3):.1f}} GiB | 本地磁盘: {{disk_bytes / (1024**3):.1f}} GiB', flush=True)
    CHUNK = 64 * 1024 * 1024
    hf_tok = os.environ.get('HF_TOKEN', '').strip()
    if not hf_tok:
        try:
            from google.colab import userdata
            hf_tok = (userdata.get('HF_TOKEN') or '').strip()
        except Exception:
            hf_tok = ''

    def _resolve_cdn(item):
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone
        orig_url, real_path, size = item
        req_hdr = {{'Range': 'bytes=0-0'}}
        if hf_tok:
            req_hdr['Authorization'] = 'Bearer ' + hf_tok
        for attempt in range(5):
            try:
                with requests.get(orig_url, headers=req_hdr, stream=True, timeout=30) as r:
                    if r.status_code == 429:
                        ra = r.headers.get('Retry-After', '')
                        try:
                            wait_s = float(ra)
                        except ValueError:
                            try:
                                wait_s = max(1.0, (parsedate_to_datetime(ra) - datetime.now(timezone.utc)).total_seconds())
                            except Exception:
                                wait_s = 2 ** (attempt + 1)
                        time.sleep(min(wait_s, 60.0))
                        continue
                    r.raise_for_status()
                    if r.status_code != 206:
                        raise RuntimeError(f'Range probe expected 206, got {{r.status_code}}')
                    cdn_url = r.url
                    return [(cdn_url, orig_url, real_path, start, min(start + CHUNK, size) - 1)
                            for start in range(0, size, CHUNK)]
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** (attempt + 1))

    range_tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(downloads)) as rpool:
        for chunk_list in rpool.map(_resolve_cdn, downloads):
            range_tasks.extend(chunk_list)

    tasks_file = Path('/tmp/colab_kernel_dl_tasks.json')
    tasks_file.write_text(json.dumps({{'need_bytes': need_bytes, 'tasks': range_tasks}}), encoding='utf-8', newline='')
    dl_script = Path('/tmp/colab_kernel_fast_dl.py')
    dl_script.write_text('''import concurrent.futures, json, multiprocessing as mp, os, sys, time
import requests
from requests.adapters import HTTPAdapter

data = json.loads(open(sys.argv[1], encoding="utf-8").read())
need_bytes = data["need_bytes"]
tasks = data["tasks"]
N_PROC = 8
N_THREADS = 4

done_bytes = mp.Value("q", 0)
t_start = time.monotonic()
last_print = mp.Value("d", t_start)

def worker_batch(batch):
    sess = requests.Session()
    adapter = HTTPAdapter(pool_connections=N_THREADS + 4, pool_maxsize=N_THREADS + 4)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)

    def fetch_one(task):
        cdn_url, orig_url, target, start, end = task
        expected = end - start + 1
        for attempt in range(5):
            url = cdn_url if attempt < 2 else orig_url
            try:
                with sess.get(url, headers={{"Range": f"bytes={{start}}-{{end}}"}}, stream=True, timeout=(15, 120)) as resp:
                    if resp.status_code != 206:
                        raise RuntimeError(f"HTTP {{resp.status_code}}")
                    fd = os.open(target, os.O_RDWR)
                    try:
                        pos = start
                        for blk in resp.iter_content(8 * 1024 * 1024):
                            if blk:
                                os.pwrite(fd, blk, pos)
                                pos += len(blk)
                    finally:
                        os.close(fd)
                    if pos - start != expected:
                        raise RuntimeError(f"short read {{pos - start}} != {{expected}}")
                with done_bytes.get_lock():
                    done_bytes.value += expected
                    now = time.monotonic()
                    if now - last_print.value >= 8:
                        rate = done_bytes.value / max(now - t_start, 0.1) / (1024**2)
                        print(f"  并发下载进度: {{done_bytes.value / need_bytes:.0%}} ({{done_bytes.value / (1024**3):.1f}} / {{need_bytes / (1024**3):.1f}} GiB)，均速 {{rate:.0f}} MB/s", flush=True)
                        last_print.value = now
                return
            except Exception as err:
                if attempt == 4:
                    raise RuntimeError(f"{{os.path.basename(target)}} [{{start}}-{{end}}] 失败: {{err}}") from err
                time.sleep(1.5 * (attempt + 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as tpool:
        futs = [tpool.submit(fetch_one, t) for t in batch]
        for f in concurrent.futures.as_completed(futs):
            f.result()

if __name__ == "__main__":
    batches = [tasks[i::N_PROC] for i in range(N_PROC) if tasks[i::N_PROC]]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=len(batches)) as pool:
        pool.map(worker_batch, batches)
''', encoding='utf-8', newline='')
    t_dl_start = time.monotonic()
    subprocess.run([sys.executable, str(dl_script), str(tasks_file)], check=True)
    dl_sec = max(time.monotonic() - t_dl_start, 0.1)
    print(f'全部权重下载完成：用时 {{dl_sec:.1f}}s，平均 {{need_bytes / (1024**2) / dl_sec:.0f}} MB/s', flush=True)
else:
    print(f'全部 {{len(FILES)}} 个权重与附件已存在于内存盘/本地，跳过下载。', flush=True)

def verify_hashes():
    # 仅调试模式调用。下载决策时已经对过一次字节大小；这里再 stat 一遍。
    try:
        for item in FILES:
            path = _local_path(item)
            if path.name in released_reuse:
                continue
            actual = path.stat().st_size if path.exists() else -1
            if actual != item['size']:
                if path.exists():
                    path.resolve().unlink(missing_ok=True)
                    path.unlink(missing_ok=True)
                raise RuntimeError('权重存在性或字节大小校验失败，请重新运行首格：' + path.name)
            print('SIZE OK:', path.name, item['size'], flush=True)
    except BaseException:
        if globals().get('qwen_server') is not None:
            qwen_server.terminate()
        raise

PARALLEL_SLOTS = 1
SLOT_CONTEXT_TOKENS = 262144
TOTAL_CONTEXT_TOKENS = SLOT_CONTEXT_TOKENS * PARALLEL_SLOTS
UBATCH_SIZE = 2048
MAX_OUTPUT_TOKENS = 65536
SLOT_SAVE_DIR = ROOT / 'slot-saves'
SLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CHAT_TMPL = ROOT / 'chat-template-clean.jinja'
UI_CFG = ROOT / 'webui-config.json'
new_ui_text = json.dumps(UI_CFG_DATA, ensure_ascii=False, indent=2) + '\\n'
old_ui_on_disk = UI_CFG.read_text(encoding='utf-8') if UI_CFG.exists() else ''

ple_name = SPEC.get('ple_shard_name')
if ple_name:
    ple_real = str((SHARD_DIR / ple_name).resolve())
    ple_in_tmpfs = ple_real.startswith(TMPFS_PREFIXES)
else:
    ple_in_tmpfs = False

# PLE 表分片在运行期由 CPU 直接 mmap 读取，必须常驻；其余落在内存盘的分片加载后可删。
release_keep = {{ple_name}} if ple_in_tmpfs else set()
release_candidates = [f for f in FILES
                      if _local_path(f).name not in release_keep
                      and _local_path(f).exists()
                      and str(_local_path(f).resolve()).startswith(TMPFS_PREFIXES)]
releasable_bytes = sum(f['size'] for f in release_candidates)
# 只用于新启动的服务；已在运行的服务保留自己的 --cache-ram（configured_server 不比较这一项，免得为改缓存上限重启）。
# 只拿回收内存的一半给提示缓存，另一半留作余量（mmap 模式下 CPU 张量所在分片删文件后仍占内存）。
PROMPT_CACHE_MIB = min(65536, 16384 + releasable_bytes // (2 * 1024**2))

def release_gpu_only_copies():
    pid, args = _server_proc()
    todo = [f for f in release_candidates if _local_path(f).exists()]
    if not pid or not todo or not health():
        return
    names = sorted({{_local_path(f).name for f in todo}} | released_reuse)
    tmp_state = RELEASE_STATE.with_suffix('.tmp')
    tmp_state.write_text(json.dumps({{'pid': pid, 'proc_id': _proc_identity(pid), 'files': names}}), encoding='utf-8', newline='')
    os.replace(tmp_state, RELEASE_STATE)
    if DEBUG_VERIFY:
        before = _mem_available()
    for f in todo:
        lp = _local_path(f)
        lp.resolve().unlink(missing_ok=True)
        lp.unlink(missing_ok=True)
    expect = sum(f['size'] for f in todo)
    if DEBUG_VERIFY:
        time.sleep(1)
        gained = _mem_available() - before
        cap = _cache_ram_of(args) or 0
        print(f'已删除 {{len(todo)}} 个显存专用分片的内存盘副本：可用内存 +{{gained / 1024**3:.1f}} GiB（预计 {{expect / 1024**3:.1f}} GiB）| 当前服务提示缓存上限 {{cap / 1024:.0f}} GiB', flush=True)
        if gained < expect * 0.5:
            print('⚠️ 实际回收不到预计的一半：部分分片仍被服务映射，提示缓存别塞满；服务重启后才会真正归还。', flush=True)
    else:
        print(f'已删除 {{len(todo)}} 个显存专用分片的内存盘副本（预计 {{expect / 1024**3:.1f}} GiB）', flush=True)

if MODEL.exists():
    with MODEL.open('rb') as f:
        raw_head = f.read(16 * 1024 * 1024)
    HEAD_CACHE.write_bytes(raw_head)
else:
    raw_head = HEAD_CACHE.read_bytes()
tmpl_key = b'tokenizer.chat_template'
key_pos = raw_head.index(struct.pack('<Q', len(tmpl_key)) + tmpl_key)
val_pos = key_pos + 8 + len(tmpl_key)
assert struct.unpack_from('<I', raw_head, val_pos)[0] == 8, 'tokenizer.chat_template 不是 GGUF 字符串类型'
tmpl_len = struct.unpack_from('<Q', raw_head, val_pos + 4)[0]
assert val_pos + 12 + tmpl_len <= len(raw_head), '模板超出前 16 MiB'
tmpl_str = raw_head[val_pos + 12:val_pos + 12 + tmpl_len].decode('utf-8')

# 官方模板只有 xhigh/medium/low 三档；换成四档，并按 effort_prompts 在 system 顶部注入一句中文引导。
universal_effort_block = '''{{%- set reasoning_instructions = '' %}}
{{%- if enable_thinking is undefined or enable_thinking is true %}}
    {{%- set resolved_reasoning_effort = reasoning_effort|default('@default@') %}}
    {{%- if resolved_reasoning_effort not in ('xhigh', 'high', 'medium', 'low') %}}
        {{{{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh, high, medium, and low (default @default@).') }}}}
    {{%- endif %}}
    {{%- if resolved_reasoning_effort == 'xhigh' %}}
        {{%- set reasoning_instructions = '@xhigh@' %}}
    {{%- elif resolved_reasoning_effort == 'high' %}}
        {{%- set reasoning_instructions = '@high@' %}}
    {{%- elif resolved_reasoning_effort == 'medium' %}}
        {{%- set reasoning_instructions = '@medium@' %}}
    {{%- elif resolved_reasoning_effort == 'low' %}}
        {{%- set reasoning_instructions = '@low@' %}}
    {{%- endif %}}
{{%- endif %}}'''
for _lvl, _txt in SPEC['effort_prompts'].items():
    universal_effort_block = universal_effort_block.replace('@%s@' % _lvl, _txt)
universal_effort_block = universal_effort_block.replace('@default@', SPEC.get('reasoning_effort_default') or 'high')

old_effort_block = '''{{%- if enable_thinking is undefined or enable_thinking is true %}}
    {{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}}
    {{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}}
        {{{{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}}}
    {{%- endif %}}
    {{%- if resolved_reasoning_effort == 'xhigh' %}}
        {{%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}}
    {{%- elif resolved_reasoning_effort == 'low' %}}
        {{%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}}
    {{%- endif %}}
{{%- endif %}}'''
assert old_effort_block in tmpl_str, '未找到预期的 reasoning_effort 模板块'
tmpl_str = tmpl_str.replace(old_effort_block, universal_effort_block, 1)
old_preserve_line = '{{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}}'
assert old_preserve_line in tmpl_str, '未找到预期的 preserve_thinking 模板行'
# 官方默认 preserve_thinking 未定义即为真，历史轮次会被套上空 <think></think>。
# 只保留当前工具链里 reasoning_content 非空的轮次，避免 few-shot 诱导模型秒关思考。
# 生成起手块不改：官方只在 enable_thinking=false 时关思考，low 照常思考。
tmpl_str = tmpl_str.replace(
    old_preserve_line,
    '{{%- if loop.index0 > ns.last_query_index and reasoning_content %}}',
    1,
)
old_tmpl_on_disk = CHAT_TMPL.read_text(encoding='utf-8') if CHAT_TMPL.exists() else ''
CHAT_TMPL.write_text(tmpl_str, encoding='utf-8', newline='')
UI_CFG.write_text(new_ui_text, encoding='utf-8', newline='')

def _fp_args(args):
    # 指纹跳过二进制路径和 --cache-ram（只改缓存上限不重启）。
    out = []
    i = 1 if args and Path(args[0]).name == 'llama-server' else 0
    while i < len(args):
        a = args[i]
        if not a:
            i += 1
            continue
        if a == '--cache-ram':
            i += 2
            continue
        out.append(a)
        i += 1
    return out

def _boot_fp(args, tmpl, ui):
    blob = chr(0).join(_fp_args(args)) + chr(0) + tmpl + chr(0) + ui
    return hashlib.sha1(blob.encode()).hexdigest()

load_flags = ['--lazy-mode', 'off'] if ple_in_tmpfs else ['-lm', 'none', '--lazy-mode', 'off']
server_args = ['-m', str(MODEL)]
if MMPROJ:
    server_args += ['--mmproj', str(MMPROJ)]
if DRAFT and SPEC.get('mtp_n_max'):
    server_args += ['--model-draft', str(DRAFT), '--spec-type', 'draft-mtp',
                    '--spec-draft-n-max', str(SPEC['mtp_n_max']),
                    '--spec-draft-p-min', str(SPEC['mtp_p_min'])]
server_args += [
    '--slot-save-path', str(SLOT_SAVE_DIR),
    '--device', 'CUDA0',
    '-ngl', '99',
    *load_flags,
    '--cache-ram', str(PROMPT_CACHE_MIB),
    '-c', str(TOTAL_CONTEXT_TOKENS),
    '--parallel', str(PARALLEL_SLOTS),
    '--host', '127.0.0.1',
    '--port', str(PORT),
    '--alias', MODEL_ALIAS,
    '--jinja',
    '--chat-template-file', str(CHAT_TMPL),
    '--ui-config-file', str(UI_CFG),
    '--ui-mcp-proxy',
    '--reasoning', 'on',
]
if SPEC.get('reasoning_effort_default'):
    server_args += ['--reasoning-effort', SPEC['reasoning_effort_default']]
server_args += [
    '--temp', SPEC['temp'],
    '--top-p', SPEC['top_p'],
    '--top-k', SPEC['top_k'],
    '--min-p', SPEC['min_p'],
    '--flash-attn', 'on',
    '--cache-type-k', 'q8_0',
    '--cache-type-v', 'q8_0',
    '--batch-size', '4096',
    '--ubatch-size', str(UBATCH_SIZE),
    '--predict', str(MAX_OUTPUT_TOKENS),
]
expected_fp = _boot_fp(server_args, tmpl_str, new_ui_text)

def configured_server():
    # 已在跑且启动指纹一致就跳过重启。指纹 = sha1(argv 去掉二进制和 --cache-ram + 模板 + webui-config)。
    if not health():
        return False
    _pid, args = _server_proc()
    if not args or _boot_fp(args, old_tmpl_on_disk, old_ui_on_disk) != expected_fp:
        return False
    if DEBUG_VERIFY:
        props = requests.get('http://127.0.0.1:%d/props' % PORT, headers=AUTH, timeout=10).json()
        if props.get('default_generation_settings', {{}}).get('n_ctx') != SLOT_CONTEXT_TOKENS:
            return False
        if props.get('total_slots') != PARALLEL_SLOTS:
            return False
        if props.get('chat_template_caps', {{}}).get('reasoning_effort_levels') != ['low', 'medium', 'high', 'xhigh']:
            return False
    return True

server_just_started = False
if health() and not configured_server():
    print('检测到旧配置或不同模型服务：重启 llama-server ...', flush=True)
    if not released_reuse:
        _stop_llama_servers()

if configured_server():
    print(f'模型已在运行：{{MODEL_ALIAS}} / 单路 × {{SLOT_CONTEXT_TOKENS}} 上下文 / {{(_cache_ram_of(_server_proc()[1]) or 0) / 1024:.0f}} GiB Prompt Cache', flush=True)
else:
    missing_now = [_local_path(f).name for f in FILES if not _local_path(f).exists()]
    if released_reuse or missing_now:
        _stop_llama_servers()
        RELEASE_STATE.unlink(missing_ok=True)
        raise RuntimeError('需要重新启动服务，但缺少上次为扩大提示缓存而删掉的分片副本：'
                           + ', '.join(sorted(set(missing_now) | released_reuse))
                           + '。旧服务已停，请再运行一次本格，会重新下载这些分片。')
    _stop_llama_servers()
    if ple_name and not freshly_downloaded and not ple_in_tmpfs:
        meminfo = Path('/proc/meminfo').read_text()
        cached_kb = next((int(l.split()[1]) for l in meminfo.splitlines() if l.startswith('Cached:')), 0)
        if cached_kb < 40 * 1024 * 1024:
            print('预热 PLE 表所在分片到主机内存 Page Cache ...', flush=True)
            with (SHARD_DIR / ple_name).open('rb') as f:
                while f.read(32 * 1024 * 1024):
                    pass

    fut_cpp.result()
    SERVER = next(DIST.rglob('llama-server'))
    env = os.environ.copy()
    libdirs = {{str(SERVER.parent)}}
    for pat in ('libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12'):
        libdirs.update(str(p.parent) for p in DIST.rglob(pat))
    env['LD_LIBRARY_PATH'] = ':'.join(sorted(libdirs)) + ':' + env.get('LD_LIBRARY_PATH', '')
    cmd = [str(SERVER), *server_args]
    log = (ROOT / 'server.log').open('a', encoding='utf-8', newline='')
    log.write(chr(10) + '=== START %s ===' % time.strftime('%Y-%m-%d %H:%M:%S') + chr(10))
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    qwen_server = proc
    t0 = time.monotonic()
    # LAUNCH_DETACHED=1（colab_cli launch）：进程已 start_new_session，不等本机 /health，交给后面的后台线程。
    if os.environ.get('LAUNCH_DETACHED') == '1':
        print('LAUNCH_DETACHED: llama-server pid %s' % proc.pid, flush=True)
        server_just_started = True
    else:
        print(f'加载 {{MODEL_ALIAS}} 到 GPU ...', flush=True)
        while time.monotonic() - t0 < 600:
            if proc.poll() is not None:
                raise RuntimeError((ROOT / 'server.log').read_text(errors='replace')[-3500:])
            if health():
                break
            time.sleep(2 if DEBUG_VERIFY else 0.5)
        else:
            proc.terminate()
            raise RuntimeError(f'模型加载超时，请查看 {{ROOT}}/server.log')
        server_just_started = True

def _warmup_cuda():
    # 第一次推理会捕获 CUDA graph，短提示通常要数秒到十几秒，不是零点几秒；放后台不挡就绪横幅。只发这一次 max_tokens=4。
    requests.post('http://127.0.0.1:%d/v1/chat/completions' % PORT, headers=AUTH, timeout=600,
                  json={{'model': MODEL_ALIAS,
                        'messages': [{{'role': 'user', 'content': '请只回复“就绪”两个字。'}}],
                        'max_tokens': 4, 'temperature': 0, 'reasoning_effort': 'low'}})

if DEBUG_VERIFY:
    if server_just_started:
        print('服务就绪 %.0fs，核验四档思考能力并发送预热请求初始化 CUDA 图 ...' % (time.monotonic() - t0), flush=True)
        props_now = requests.get('http://127.0.0.1:%d/props' % PORT, headers=AUTH, timeout=10).json()
        levels_now = props_now.get('chat_template_caps', {{}}).get('reasoning_effort_levels', [])
        if levels_now != ['low', 'medium', 'high', 'xhigh']:
            proc.terminate()
            raise RuntimeError(f'思考档位探测未达四档标准 (期望 low/medium/high/xhigh): {{levels_now}}')
        print('思考档位门禁通过：', levels_now, '| 默认档位：', SPEC.get('reasoning_effort_default', 'high'), flush=True)
        _warmup_cuda()
    verify_hashes()
    release_gpu_only_copies()
    fut_ts.result()
    status = ts('serve', 'status').stdout
    if SERVICE_FQDN not in status:
        raise RuntimeError('Service 未生效，serve status:' + chr(10) + status)
else:
    # 就绪横幅不等这些：预热、删内存盘副本、等 Tailscale。
    def _bg_finish():
        try:
            if server_just_started:
                if os.environ.get('LAUNCH_DETACHED') == '1':
                    # 主线程已返回。预热必须先等本机 /health，否则连接失败会跳过后面的内存盘释放和 Tailscale 收尾。
                    deadline = time.monotonic() + 600
                    while time.monotonic() < deadline:
                        if proc.poll() is not None:
                            raise RuntimeError((ROOT / 'server.log').read_text(errors='replace')[-3500:])
                        if health():
                            break
                        time.sleep(0.5)
                    else:
                        raise RuntimeError('模型加载超时，请查看 %s/server.log' % ROOT)
                _warmup_cuda()
            release_gpu_only_copies()
            fut_cpp.result()
            fut_ts.result()
        except Exception as exc:
            print('后台收尾失败：', exc, flush=True)
    threading.Thread(target=_bg_finish, daemon=True).start()
    print('生产模式：预热、释放内存盘副本、Tailscale 在后台收尾，不挡住下面的就绪横幅。', flush=True)

print('=' * 64)
print('总启动耗时： %.1fs' % (time.monotonic() - t_boot), flush=True)
print('网页版：     http://%s:443/' % SERVICE_FQDN, flush=True)
print('接口：       http://%s:443/v1' % SERVICE_FQDN, flush=True)
print('模型名：     %s' % MODEL_ALIAS, flush=True)
print('=' * 64)
""" + console_code(
        6,
        '服务已就绪！',
        'G4',
        f'<b>当前状态：</b><span style="color:#5cb85c">{spec["banner"]}</span> — 用完点黄色按钮立即释放 GPU，WebUI 标签页会随之自动关闭',
    )

    inspect_code = f"""# 可选：查看当前配置、GPU 显存与主机内存占用
import json, subprocess
from pathlib import Path
import requests

root = Path(globals().get('SPEC', {{}}).get('root_dir', {json.dumps(root_dir)}))
hdr = {{'Authorization': 'Bearer ' + (root / '.llama-api-key').read_text().strip()}}
r = requests.get('http://127.0.0.1:8081/props', headers=hdr, timeout=10)
r.raise_for_status()
props = r.json()
p = props['default_generation_settings']
print('每槽上下文:', p['n_ctx'], 'tokens | 并发槽位数:', props.get('total_slots'), '| WebUI 配置已下发:', bool(props.get('ui_settings') or props.get('webui_settings')))
for procdir in Path('/proc').iterdir():
    if not procdir.name.isdigit():
        continue
    try:
        args = (procdir / 'cmdline').read_bytes().decode().split('\\0')
        if args and Path(args[0]).name == 'llama-server':
            flags = ('--alias', '--mmproj', '--model-draft', '-c', '--parallel', '--predict', '--spec-draft-n-max', '--spec-draft-p-min', '--reasoning', '--reasoning-effort', '--cache-ram', '--ui-config-file', '--temp', '--top-p', '--top-k', '--min-p', '--repeat-penalty', '--repeat-last-n', '--presence-penalty', '--frequency-penalty', '--dry-multiplier', '--dry-base', '--dry-allowed-length', '--dry-penalty-last-n', '--batch-size', '--ubatch-size', '--cache-type-k', '--cache-type-v', '-lm', '--lazy-mode', '-ot')
            print('实际启动参数:', json.dumps({{k: args[args.index(k) + 1] for k in flags if k in args}}, ensure_ascii=False))
            print('WebUI MCP 代理已启用:', '--ui-mcp-proxy' in args)
    except (FileNotFoundError, PermissionError, IndexError):
        pass
print('GPU:', subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader'))
print('Host RAM:', subprocess.getoutput('free -h | head -n 2'))
for name in ('released-copies.json',):
    p = root / name
    if p.exists():
        print(name + ':', p.read_text(encoding='utf-8')[:400])
"""

    verify_code = f"""# 【调试用 · 手动运行】全链路验收：接口 / 思考档位切换 / 速度 / 中文小说共创 / 工具调用
import json, time
from pathlib import Path
import requests

CUR_SPEC = globals().get('SPEC', {{}})
ROOT = Path(CUR_SPEC.get('root_dir', {json.dumps(root_dir)}))
MODEL_ALIAS = CUR_SPEC.get('alias', {json.dumps(alias)})
base = 'http://127.0.0.1:8081'
headers = {{'Authorization': 'Bearer ' + (ROOT / '.llama-api-key').read_text().strip()}}

models = requests.get(base + '/v1/models', headers=headers, timeout=15).json()
print('模型：', [m['id'] for m in models['data']])
assert MODEL_ALIAS in [m['id'] for m in models['data']]

novel = ('你是我的中文小说共创搭档。下面这段武侠场景需要你接着往下写：\\n'
         '　　「风从断崖口灌进来，卷起满地碎雪。沈砚之握着半截断剑，指节发白。」\\n'
         '请续写 200 字左右：只写正文，不要解释思路，以动作细节表现人物心虚但嘴硬。')

for effort in ('high', 'xhigh'):
    t0 = time.monotonic()
    r = requests.post(base + '/v1/chat/completions', headers=headers, timeout=600,
                      json={{'model': MODEL_ALIAS, 'messages': [{{'role': 'user', 'content': novel}}],
                            'reasoning_effort': effort, 'temperature': float(CUR_SPEC.get('temp', {json.dumps(spec['temp'])})), 'max_tokens': 512}})
    r.raise_for_status()
    d = r.json()
    msg = d['choices'][0]['message']
    tm = d.get('timings') or {{}}
    print(f'=== effort={{effort}} | wall={{time.monotonic()-t0:.2f}}s | tok/s={{tm.get("predicted_per_second", 0):.1f}} ===')
    print('REASONING:', (msg.get('reasoning_content') or '(none)')[:160].replace('\\n', ' '))
    print('CONTENT:', (msg.get('content') or '')[:240].replace('\\n', ' '))

tools = [{{'type': 'function', 'function': {{'name': 'get_weather', 'description': '查询城市天气',
          'parameters': {{'type': 'object', 'properties': {{'city': {{'type': 'string'}}}}, 'required': ['city']}}}}}}]
rt = requests.post(base + '/v1/chat/completions', headers=headers, timeout=120,
                   json={{'model': MODEL_ALIAS, 'messages': [{{'role': 'user', 'content': '亚特兰大现在天气如何？用 get_weather 工具查。'}}],
                         'reasoning_effort': 'low', 'temperature': 0, 'tools': tools, 'max_tokens': 128}}).json()
print('TOOL_CALLS:', json.dumps(rt['choices'][0]['message'].get('tool_calls'), ensure_ascii=False))
"""

    stop_code = r"""# 长程写作任务防断连护航（前台保活 + 45分钟无请求自动停机省 CU）或立即点击按钮释放 G4 GPU
import ipywidgets as widgets
import os, subprocess, threading, time
import requests
from google.colab import runtime
from IPython.display import display
from html import escape

IDLE_LIMIT_MIN = 45
stop_button = widgets.Button(description='🛑 立即结束 G4 GPU 运行时', button_style='warning')
guard_button = widgets.Button(description='🛡️ 启动长程防断连护航 (45m空闲自停)', button_style='info')
stop_status = widgets.HTML(value='长程写作/共创时可点蓝色按钮保持前台心跳防 Colab 90m 断连（连续 45 分钟无推理请求会自动释放 G4 省 CU）；用完点黄色按钮立即释放。')
_guard_running = False

def stop_runtime(_=None):
    global _guard_running
    _guard_running = False
    stop_button.disabled = True
    guard_button.disabled = True
    stop_button.description = '已请求释放'
    stop_status.value = ('正在关闭服务并释放 G4 GPU。此提示断开后不再刷新；'
                         '请在“代码执行程序 → 管理会话”确认没有活动会话。')
    for name in ('qwen_server', 'tunnel_proc'):
        process = globals().get(name)
        if process is not None and process.poll() is None:
            process.terminate()
    tsctl = '/content/tsbin/tailscale'
    if os.path.exists(tsctl):
        try:
            subprocess.run([tsctl, '--socket=/var/run/tailscale/tailscaled.sock', 'down'],
                           timeout=5, capture_output=True)
        except Exception as error:
            print('节点下线未确认，将直接释放运行时：', type(error).__name__)
    try:
        runtime.unassign()
    except Exception as error:
        stop_button.disabled = False
        stop_button.description = '重试释放 GPU'
        stop_status.value = ('<b>释放请求失败：</b>' + escape(str(error)) +
                             '<br>请重试，或使用“代码执行程序 → 断开连接并删除运行时”。')

def start_guard(_):
    global _guard_running
    if _guard_running:
        return
    _guard_running = True
    guard_button.disabled = True
    guard_button.description = '🛡️ 长程护航运行中...'
    def _loop():
        last_active = time.monotonic()
        last_tokens = -1
        while _guard_running:
            try:
                slots = requests.get('http://127.0.0.1:8081/slots', timeout=5).json()
                busy = any(s.get('is_processing') for s in slots)
                cur_tokens = sum(int(s.get('n_past', 0)) for s in slots)
                if busy or (last_tokens != -1 and cur_tokens != last_tokens):
                    last_active = time.monotonic()
                last_tokens = cur_tokens
                idle_min = (time.monotonic() - last_active) / 60.0
                if idle_min >= IDLE_LIMIT_MIN:
                    stop_status.value = f'<b>已连续空闲 {idle_min:.0f} 分钟，触发自动停机释放 G4 GPU 省 CU...</b>'
                    stop_runtime()
                    return
                stop_status.value = f'<b>🛡️ 长程护航生效中：</b>槽位状态={"🔥 推理中" if busy else "🟢 待命"} | 当前缓存={cur_tokens} tokens | 已空闲={idle_min:.1f}/{IDLE_LIMIT_MIN} 分钟'
            except Exception as e:
                stop_status.value = f'<b>护航探测异常：</b>{escape(str(e))}'
            time.sleep(30)
    threading.Thread(target=_loop, daemon=True).start()

stop_button.on_click(stop_runtime)
guard_button.on_click(start_guard)
display(widgets.VBox([widgets.HBox([guard_button, stop_button]), stop_status]))
"""

    stop_md = """**结束状态与用量**

释放 G4 GPU 后，右上角显示“重新连接”表示已断开；进入 **代码执行程序 → 管理会话**，显示 **没有正在进行的会话** 即表示 G4 已释放。

计费按运行时挂着的时间算，空闲一样扣。Colab G4（RTX PRO 6000 Blackwell 96GB）约 **8.71–10.05 计算单元/小时**。

本地提示：`python scripts/colab_meter.py status` 实时查 CU 余额与燃烧率；`python scripts/colab_meter.py stop --all` 强杀全部残留实例并打印双轨 CU 账单。
"""

    nb = {
        'nbformat': 4,
        'nbformat_minor': 0,
        'metadata': {
            'colab': {
                'provenance': [],
                'gpuType': 'G4',
                'machine_shape': 'hm',
            },
            'accelerator': 'GPU',
            'kernelspec': {
                'display_name': 'Python 3',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
            },
        },
        'cells': [
            {'cell_type': 'markdown', 'metadata': {'id': 'overview'}, 'source': _lines(overview_md)},
            {'cell_type': 'code', 'metadata': {'id': 'persistent-setup'}, 'source': _lines(oneclick_code), 'execution_count': None, 'outputs': []},
            {'cell_type': 'code', 'metadata': {'id': 'inspect-props'}, 'source': _lines(inspect_code), 'execution_count': None, 'outputs': []},
            {'cell_type': 'code', 'metadata': {'id': f'verify-{slug}'}, 'source': _lines(verify_code), 'execution_count': None, 'outputs': []},
            *([{'cell_type': 'code', 'metadata': {'id': 'hot-switch'}, 'source': _lines(build_switch_code(spec, config)), 'execution_count': None, 'outputs': []}] if with_switch else []),
            {'cell_type': 'code', 'metadata': {'id': 'stop-runtime'}, 'source': _lines(stop_code), 'execution_count': None, 'outputs': []},
            {'cell_type': 'markdown', 'metadata': {'id': 'stop-status-help'}, 'source': _lines(stop_md)},
        ],
    }

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    setup_src = ''.join(next(c['source'] for c in nb['cells'] if c['metadata'].get('id') == 'persistent-setup'))
    _assert_prod_startup(setup_src)

    return nb


SWITCH_TEMPLATE = r"""# 热切换模型：不释放 G4，停掉当前模型、清掉它的文件，再执行目标模型的一键启动（权重重新下载，约 2–4 分钟）
TARGET = __DEFAULT__  # @param __OPTIONS__
import os, shutil, signal, subprocess, time
from pathlib import Path
import requests

SWITCH = __PAYLOAD__

def _llama_pids():
    pids = []
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdigit():
            continue
        try:
            args = (procdir / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if args and Path(args[0]).name == 'llama-server':
            pids.append(int(procdir.name))
    return pids

def _switch():
    if TARGET not in SWITCH:
        raise ValueError('未知模型 %s，可选：%s' % (TARGET, ', '.join(SWITCH)))
    info = SWITCH[TARGET]
    try:
        served = [m['id'] for m in requests.get('http://127.0.0.1:8081/v1/models', timeout=5,
                  headers={'Authorization': 'Bearer ' + globals().get('KEY', '')}).json()['data']]
    except Exception:
        served = []
    if globals().get('SPEC', {}).get('slug') == TARGET and info['alias'] in served:
        print('%s 已在运行且健康，无需切换。' % info['alias'])
        return False
    gpu_mib = int(subprocess.getoutput('nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits').split()[0])
    if gpu_mib < info['min_vram_mib']:
        raise RuntimeError('当前显卡 %.0f GiB，跑不了 %s（至少 %.0f GiB），未做任何改动。'
                           % (gpu_mib / 1024, info['alias'], info['min_vram_mib'] / 1024))
    for pid in _llama_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    old = globals().get('qwen_server')
    if old is not None:
        try:
            old.wait(timeout=60)
        except Exception:
            pass
    deadline = time.monotonic() + 60
    while _llama_pids() and time.monotonic() < deadline:
        time.sleep(1)
    for pid in _llama_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    while _llama_pids() and time.monotonic() < deadline:
        time.sleep(1)
    if _llama_pids():
        raise RuntimeError('旧 llama-server 杀不掉，切换中止（文件未删）。请用停机按钮释放运行时后重来。')
    globals()['qwen_server'] = None
    target_root = Path(info['root_dir'])
    target_root.mkdir(parents=True, exist_ok=True)
    for slug, other in SWITCH.items():
        if slug == TARGET:
            continue
        for base in ('/dev/shm', '/var/colab'):
            shutil.rmtree(Path(base) / slug, ignore_errors=True)
        old_root = Path(other['root_dir'])
        if (old_root / 'llama-dist').is_dir() and not (target_root / 'llama-dist').exists():
            shutil.move(str(old_root / 'llama-dist'), str(target_root / 'llama-dist'))
        shutil.rmtree(old_root, ignore_errors=True)
    time.sleep(3)
    print('已停掉旧模型并清空其内存盘与本地文件，开始启动 %s ...' % info['alias'], flush=True)
    return True

if _switch():
    exec(compile(SWITCH[TARGET]['code'], '<switch:%s>' % TARGET, 'exec'), globals())
    print('切换完成：客户端里把模型名换成 %s；上方“查看配置”和“调试用全链路验收”两格会自动跟着新模型。' % SWITCH[TARGET]['alias'], flush=True)
"""


def build_switch_code(spec: dict, config: dict) -> str:
    payload = {}
    for slug, other in MODELS.items():
        other_nb = build_notebook_dict(other, config, with_switch=False)
        setup = next(c for c in other_nb['cells'] if c['metadata'].get('id') == 'persistent-setup')
        payload[slug] = {
            'alias': other['alias'],
            'root_dir': other['root_dir'],
            'min_vram_mib': other['min_vram_mib'],
            'code': ''.join(setup['source']),
        }
    options = [s for s in MODELS if s != spec['slug']] + [spec['slug']]
    return (SWITCH_TEMPLATE
            .replace('__DEFAULT__', repr(options[0]))
            .replace('__OPTIONS__', json.dumps(options))
            .replace('__PAYLOAD__', repr(payload)))


def generate_one(slug: str, argv: list[str] | None = None) -> tuple[dict, str]:
    argv = sys.argv[1:] if argv is None else argv
    spec = MODELS[slug]
    if spec['drive_key'] not in FIELDS:
        FIELDS[spec['drive_key']] = (spec['env_key'], spec['placeholder'])
    config, sanitized, output_dir, sources = load_config(argv)
    report(config, sanitized, sources)
    guard_sync(sanitized, sources, config)

    nb = build_notebook_dict(spec, config)
    setup_src = ''.join(next(c['source'] for c in nb['cells'] if c['metadata'].get('id') == 'persistent-setup'))
    prod_urls = _assert_prod_startup(setup_src)
    _selftest_boot_fp(setup_src)
    print('Prod-path HTTP (no /props):', prod_urls)
    nb_json = json.dumps(nb, ensure_ascii=False, indent=1) + '\n'
    nb_path = output_dir / spec['notebook']
    nb_path.write_text(nb_json, encoding='utf-8', newline='')
    print('Generated:', nb_path)

    if not sanitized and '--sync-colab' not in argv and output_dir == ROOT_DIR / 'notebooks' and DRIVE_NB_DIR.exists():
        drive_nb = DRIVE_NB_DIR / spec['notebook']
        drive_nb.write_text(nb_json, encoding='utf-8', newline='')
        assert drive_nb.read_bytes() == nb_path.read_bytes()
        sha = hashlib.sha256(drive_nb.read_bytes()).hexdigest()
        print('Mirrored to Drive:', drive_nb, 'SHA-256:', sha[:16] + '...')

    drive_id = _ensure_drive_id(spec, config, sanitized)
    sync_requested(nb, drive_id, label=slug)
    return nb, drive_id


def sync_magpie_models():
    # `magpie provider models <id> <ids…>` 是整体替换：先读现有模型再取并集，移出注册表的模型不会被顺手删掉。
    bash_bin = r'C:\Program Files\Git\bin\bash.exe'
    listed = subprocess.run([bash_bin, '-lc', '~/.local/bin/magpie model list'],
                            capture_output=True, text=True, encoding='utf-8', check=True).stdout
    existing = [tok.split('/', 1)[1] for tok in listed.split() if tok.startswith('qwen-colab/')]
    aliases = [m['alias'] for m in MODELS.values()]
    keep = list(dict.fromkeys(existing + aliases))
    subprocess.run([bash_bin, '-lc', '~/.local/bin/magpie provider models qwen-colab ' + ' '.join(keep)], check=True)
    for alias in aliases:
        subprocess.run(
            [
                bash_bin,
                '-lc',
                f'~/.local/bin/magpie model spec qwen-colab/{alias} --context 262144 --output 65536 --vision on --reasoning low,medium,high,xhigh',
            ],
            check=True,
        )


def run_single(slug: str):
    generate_one(slug)


def main():
    args = sys.argv[1:]
    targets = list(MODELS.keys()) if ('--all' in args or not [a for a in args if not a.startswith('-')]) else [a for a in args if not a.startswith('-')]
    for slug in targets:
        if slug not in MODELS:
            raise SystemExit(f'Unknown model slug: {slug}. Available: {", ".join(MODELS)}')
        generate_one(slug, args)
    if '--sync-magpie' in args:
        sync_magpie_models()


if __name__ == '__main__':
    main()
