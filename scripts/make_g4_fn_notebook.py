"""Generate the Colab G4 notebook for Comprehensive #1: Swift-1.5-Qwen3.8-Flash-Next Heretic V2 (IQ4_XS-plus, jackjusko, native uncensored) + MTP + mmproj.

Does NOT modify the 27B Literary files. Reuses the Tailscale Service (svc:qwen) and API key
from notebooks/qwen3.8-27b-pi-colab.ipynb (falls back to this notebook's own previous copy)
for time-shared operation. Usage: python scripts/make_g4_fn_notebook.py [--sanitized] [--output-dir DIR] [--sync-colab [--run-cell1] [--release-runtime]]
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_sync import sync_requested
from webui_console import console_code

from secrets_config import guard_sync, load_config, report

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG, SANITIZED, OUTPUT_DIR, CONFIG_SOURCES = load_config()
report(CONFIG, SANITIZED, CONFIG_SOURCES)
guard_sync(SANITIZED, CONFIG_SOURCES, CONFIG)
LIT_NB = OUTPUT_DIR / 'qwen3.8-27b-pi-colab.ipynb'
G4_NB = OUTPUT_DIR / 'qwen3.8-flash-next-g4-colab.ipynb'
G4_USAGE = ROOT_DIR / 'references/qwen38-flash-next-g4.md'

LIT_DRIVE_ID = CONFIG['lit_drive_id']
G4_DRIVE_ID = CONFIG['g4_drive_id']
SERVICE_FQDN_VALUE = CONFIG['tailnet_fqdn']
TS_CLIENT_ID_VALUE = CONFIG['ts_client_id']
API_KEY = CONFIG['api_key']
TS_SECRET = CONFIG['ts_client_secret']

def lines(text: str) -> list[str]:
    raw = text.strip('\n').split('\n')
    return [line + '\n' for line in raw]

OVERVIEW_MD = """# Qwen3.8-Flash-Next · Colab G4 代码与长程软件工程服务（Comprehensive #1：Swift-1.5 Heretic V2 原生破限 IQ4_XS-plus + 单路 262K 工程特化）

模型 `jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer`（`IQ4_XS-plus`，6 分片共 94.70 GB / 88.20 GiB，**原生 Heretic v2 破限，针对代码/架构/Agent 工具调用链/逆向工程专门校准 imatrix，无需任何 LoRA**），配 Unsloth `mtp-Qwen3.8-Flash-Next-Q4_K_M` 投机头与 ukisai 自带的 `mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16` 视觉投影器，**单路 × 262,144 token 满血上下文（`PARALLEL_SLOTS=1`，整张卡的显存带宽只服务你一条流）**，单卡 **Colab G4（RTX PRO 6000 Blackwell 96GB + ~180GB RAM）**。

**代码 / 逆向 / 长程软件工程 Agent 专项调优参数**

- `--temp 0.25 --top-p 0.85 --top-k 20 --min-p 0.05`：代码与工具调用高确定性采样，压低长程多步编辑时的标识符漂移与 JSON Schema 语法错误，且使代码场景下 MTP 投机解码接受率跃升至 60%+。
- `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.55`：针对高可预测性代码语法块优化的 4 步投机解码。
- `--dry-multiplier 0.5 --dry-base 1.75 --dry-allowed-length 6 --dry-penalty-last-n 2048`：放行 6 tokens 以内的合法代码重复（如 `import`、结构体字段、类型注解），同时精准掐断长程 Agent 偶尔出现的重复失败命令死循环。
- `--cache-ram 16384` + `--slot-save-path /content/qwen38-fn/slot-saves` + `--batch-size 4096 --ubatch-size 2048 --predict 65536`：16 GiB 主机内存 Prompt Cache + 槽位持久化 + **65,536 tokens 单次最大输出**，大仓库长上下文瞬间恢复，长补丁/多文件重构永不截断。
- 不带 `--api-key-file`：`svc:qwen` 仅 tailnet 内可见，网页端打开即可聊天，无需在设置里粘贴 key。

**与 Literary #1 的关系（分时复用固定入口，零干扰）**

- 网页：`http://__SERVICE_FQDN__:443/`（首格结束时自动弹出；点首格的停机按钮会一并关闭该标签页）
- 接口：`http://__SERVICE_FQDN__:443/v1`
- 模型名：`Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS`（服务端不校验 key，客户端带不带 `Authorization` 都行）
- 运行时目录：`/content/qwen38-fn`（与 Literary #1 的 `/content/qwen27b-lit` 完全隔离）
- 分时复用 `svc:qwen`：同一时刻只开一个运行时；后启动的节点自动以 `tag:colab` 接管 `svc:qwen`，Magpie 与 `pi` 无需改地址，直接在 `/model` 中切换模型 ID 即可。

**启动加速设计（DDR5 内存盘 `tmpfs` 直写 + 32 路多进程零 GIL 下载 + `MAP_POPULATE` 零拷贝映射）**

1. **绕过 `/content` 持久盘写限速墙（206–353 MiB/s）**：自动扩容并分配 `/dev/shm` 与 `/var/colab` 两个 DDR5 `tmpfs` 内存盘，把权重与附件优先直接写进物理内存（>20 GB/s，零磁盘 I/O）并软链回 `/content/qwen38-fn`。
2. **8 进程 × 4 线程（32 路并发流）+ `os.pwrite` 原子直写**：独立多进程下载器绕开单 Python 进程 GIL 锁与 SSL 解密瓶颈，并行解析一次 HF AWS CDN 签名直链后 32 路直写内存盘。
3. **`MAP_POPULATE`（`--lazy-mode off`）零拷贝映射 PLE 表**：分片 2（28.8 GB）只含一个张量 `per_layer_token_embd.weight`（26.82 GiB 的 PLE / N-gram 表），第一优先写入 `/dev/shm`，`llama-server` 原地映射并预填页表项，省去 `-lm none` 的二次内存拷贝，推理期 0 缺页。
4. **首格一站式控制台**：启动完成后在首格底部先显示「🚀 服务已就绪」横幅（含 WebUI 链接），其下常驻 **「🛑 结束 G4 运行时（停机省 CU）」** 按钮，无需再翻页运行其他单元格。
"""

OVERVIEW_MD = OVERVIEW_MD.replace('__SERVICE_FQDN__', SERVICE_FQDN_VALUE)

ONECLICK_CODE = rf"""# 一键启动（Colab G4 内存盘极速版）：tmpfs 内存盘 + 32路多进程下载 + llama.cpp + Tailscale
import concurrent.futures, json, os, shutil, struct, subprocess, sys, threading, time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter

t_boot = time.monotonic()
ROOT = Path('/content/qwen38-fn')
DIST = ROOT / 'llama-dist'
SHARD_DIR = ROOT / 'IQ4_XS'
MODEL_ALIAS = 'Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS'
KEY_FILE = ROOT / '.llama-api-key'
PORT = 8081
KEY = '{API_KEY}'

HERETIC_REV = '76208ff7308ccebd887eec5293d3cab305091514'
HERETIC_HF = 'https://huggingface.co/jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer/resolve/' + HERETIC_REV
UKISAI_REV = '0142554c963a8653912c0c8283f05f13f63653f3'
UKISAI_HF = 'https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF/resolve/' + UKISAI_REV
UNSLOTH_REV = '766911a6b7369840a91dbcd95f9f997acaab6cd6'
UNSLOTH_HF = 'https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/' + UNSLOTH_REV
CPP_TAG = 'b12032'
CPP_URL = 'https://github.com/ggml-org/llama.cpp/releases/download/' + CPP_TAG

# 6 个分片共 94_704_945_792 B（88.20 GiB），仓库内路径 IQ4_XS/<文件名>。分片 1（0.5 GB）只含元数据与 4 个小张量；
# 分片 2（28.8 GB）只含一个张量 per_layer_token_embd.weight（26.82 GiB 的 PLE 表）；分片 3–6 是纯 GPU 张量
SHARDS = [
    ('Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00001-of-00006.gguf', 539_422_592, 'b963118b880fe3b7413e53b474c1d8d3faefc1ae0d2d6bae7a7c3e9f68217b46'),
    ('Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00002-of-00006.gguf', 28_800_138_432, '0e9027aa7de4817c91b5954039b563484466392e29a6b051b096f9c47a71a1c6'),
    ('Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00003-of-00006.gguf', 16_681_220_224, '95c25e4ada4fa9ce4380683e55fb9f55aa8e411dcca13bec789f3cc106ffd403'),
    ('Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00004-of-00006.gguf', 16_984_320_096, '5458253ef234558243c522c8d5289b923fe28601ec6aca26239de14f03ed1951'),
    ('Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00005-of-00006.gguf', 16_662_985_152, 'ae9f3f684493f0107f0aa01dd603844266ee8b9a53bad53475761f2acc31202a'),
    ('Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00006-of-00006.gguf', 15_036_859_296, 'bcebb4fecfaaf887c2a763c252ecca17749efe175e6c9c1c45d555c0e68906c0'),
]
PLE_SHARD = SHARDS[1]
MODEL = SHARD_DIR / SHARDS[0][0]
DRAFT = ROOT / 'mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf'
DRAFT_SIZE = 2_786_204_800
MMPROJ = ROOT / 'mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf'
MMPROJ_SIZE = 907_543_520

SERVICE = 'svc:qwen'
SERVICE_FQDN = '{SERVICE_FQDN_VALUE}'
SERVICE_TAG = 'tag:colab'
TS_HOSTNAME = 'qwen-g4-colab'
TS_CLIENT_ID = '{TS_CLIENT_ID_VALUE}'
TS_CLIENT_SECRET = '{TS_SECRET}'
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

# 1. 硬件前置校验：确保是 96GB 级 GPU（G4 / RTX PRO 6000）且磁盘空间充足
gpu_info = subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits').strip()
if not gpu_info or ',' not in gpu_info:
    raise RuntimeError('未检测到 NVIDIA GPU，请在右上角切换到 G4 GPU 运行时。输出：' + gpu_info)
gpu_name, gpu_mib_str = [x.strip() for x in gpu_info.split(',', 1)]
gpu_mib = int(gpu_mib_str)
print(f'检测到 GPU: {{gpu_name}} ({{gpu_mib / 1024:.1f}} GiB VRAM)', flush=True)
if gpu_mib < 85000:
    raise RuntimeError(f'当前 GPU 显存仅 {{gpu_mib / 1024:.1f}} GiB（{{gpu_name}}），不足以全量加载 Swift-1.5 Heretic V2 IQ4_XS-plus (GPU 侧约 61.4 GiB 权重 + 1×262K KV)。请切换到 G4 (96GB) 运行时！')

ROOT.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)
DIST.mkdir(parents=True, exist_ok=True)
KEY_FILE.write_text(KEY + '\n')
KEY_FILE.chmod(0o600)
AUTH = {{'Authorization': 'Bearer ' + KEY}}

# 2. 后台任务 A（与权重下载并行）：准备 llama.cpp b12032
def prepare_llama_cpp():
    if next(DIST.rglob('llama-server'), None) is not None:
        return
    print(f'[并行任务] 下载并解压 llama.cpp {{CPP_TAG}} (CUDA 12.8) ...', flush=True)
    sh(f'cd {{DIST}} && curl -fsSL -O {{CPP_URL}}/llama-{{CPP_TAG}}-bin-ubuntu-cuda-12.8-x64.tar.gz'
       f' && curl -fsSL -O {{CPP_URL}}/cudart-llama-{{CPP_TAG}}-bin-ubuntu-cuda-12.8-x64.tar.gz'
       f' && tar -xzf llama-{{CPP_TAG}}-bin-ubuntu-cuda-12.8-x64.tar.gz'
       f' && tar -xzf cudart-llama-{{CPP_TAG}}-bin-ubuntu-cuda-12.8-x64.tar.gz'
       f' && rm -f *.tar.gz')
    print(f'[并行任务] llama.cpp {{CPP_TAG}} 就绪', flush=True)

# 3. 后台任务 B（与权重下载并行）：启动 Tailscale 并预先挂载 svc:qwen（提前完成自动审批与证书签发）
def ts(*a, timeout=60):
    return subprocess.run([TS, '--socket=' + TS_SOCK, *a], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)

def ts_state():
    try:
        return json.loads(ts('status', '--json').stdout).get('BackendState', '')
    except Exception:
        return ''

def prepare_tailscale():
    if not Path(TS).exists():
        subprocess.run('curl -fsSL https://pkgs.tailscale.com/stable/tailscale_1.104.1_amd64.tgz | tar -xz -C /content'
                       ' && mv /content/tailscale_1.104.1_amd64 /content/tsbin', shell=True, check=True)
    TS_DIR.mkdir(parents=True, exist_ok=True)
    Path('/var/run/tailscale').mkdir(parents=True, exist_ok=True)
    if ts_state() in ('', 'NoState', 'Stopped'):
        subprocess.Popen(['/content/tsbin/tailscaled', '--tun=userspace-networking', '--socks5-server=localhost:1055',
                          '--state=' + str(TS_STATE), '--statedir=' + str(TS_DIR), '--socket=' + TS_SOCK],
                         stdout=open('/content/tailscaled.log', 'a'), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        for _ in range(40):
            if Path(TS_SOCK).exists():
                break
            time.sleep(0.5)
        for _ in range(20):
            if ts_state() in ('Running', 'NeedsLogin'):
                break
            time.sleep(1)
    if ts_state() != 'Running':
        token = requests.post('https://api.tailscale.com/api/v2/oauth/token',
                              data={{'client_id': TS_CLIENT_ID, 'client_secret': TS_CLIENT_SECRET}},
                              timeout=30).json()['access_token']
        key = requests.post('https://api.tailscale.com/api/v2/tailnet/-/keys',
                            headers={{'Authorization': 'Bearer ' + token}},
                            json={{'capabilities': {{'devices': {{'create': {{
                                'reusable': False, 'ephemeral': True, 'preauthorized': True,
                                'tags': [SERVICE_TAG]}}}}}}}},
                            timeout=30).json()['key']
        ts('up', '--auth-key=' + key, '--hostname=' + TS_HOSTNAME, '--advertise-tags=' + SERVICE_TAG,
           '--accept-dns=false', '--timeout=60s', timeout=120)
        for _ in range(30):
            if ts_state() == 'Running':
                break
            time.sleep(2)
    ts('serve', 'reset')
    ts('serve', '--bg', '--service=' + SERVICE, '--tcp=443', 'tcp://127.0.0.1:%d' % PORT, timeout=90)
    ts('serve', '--bg', '--https=443', 'http://127.0.0.1:%d' % PORT, timeout=90)
    print(f'[并行任务] Tailscale Service ({{SERVICE_FQDN}}) 隧道已就绪', flush=True)

bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
fut_cpp = bg_executor.submit(prepare_llama_cpp)

# 4. 内存盘 (tmpfs: /dev/shm + /var/colab) 智能分配 + 8进程×4线程 (32路) 零 GIL 极速并发下载
# Colab 的 /content 挂在 ~235GiB GCE 持久盘上，宿主机写吞吐上限仅 ~206-353 MiB/s。
# G4 拥有 176 GiB DDR5 主机内存，优先把权重直写进 tmpfs 内存盘并软链回 /content/qwen38-fn，彻底绕开磁盘写限速，
# 同时让 llama-server 直接在内存盘上 MAP_POPULATE 零拷贝映射 26.82 GiB 的 PLE 表（分片 2，省去 -lm none 的二次内存拷贝）。
subprocess.run(['mount', '-o', 'remount,size=115G', '/dev/shm'], capture_output=True)
tmpfs_dirs = []
for mline in Path('/proc/mounts').read_text().splitlines():
    mparts = mline.split()
    if len(mparts) >= 3 and mparts[2] == 'tmpfs' and mparts[1] in ('/dev/shm', '/var/colab'):
        tmpfs_dirs.append(Path(mparts[1]))
mp_free = {{mp: shutil.disk_usage(mp).free for mp in tmpfs_dirs}}
meminfo_txt = Path('/proc/meminfo').read_text()
avail_kb = next((int(l.split()[1]) for l in meminfo_txt.splitlines() if l.startswith('MemAvailable:')), 0)
ram_budget = max(0, avail_kb * 1024 - 95 * (1024**3))  # 预留 95GiB 给 PLE MAP_POPULATE + CUDA_Host 锁页缓冲 + 16GiB Prompt Cache，剩余空间才放 tmpfs

# 按内存盘优先级排列：分片 2 (28.8GB，只含 26.82GiB PLE 表) 必须第一优先进 tmpfs，其次是元数据分片 1 (0.5GB) 与小组件，最后是纯 GPU 分片 3–6
all_targets = [
    (f'{{HERETIC_HF}}/IQ4_XS/{{PLE_SHARD[0]}}', SHARD_DIR / PLE_SHARD[0], PLE_SHARD[1]),
    (f'{{HERETIC_HF}}/IQ4_XS/{{SHARDS[0][0]}}', SHARD_DIR / SHARDS[0][0], SHARDS[0][1]),
    (f'{{UNSLOTH_HF}}/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf', DRAFT, DRAFT_SIZE),
    (f'{{UKISAI_HF}}/mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf', MMPROJ, MMPROJ_SIZE),
] + [(f'{{HERETIC_HF}}/IQ4_XS/{{name}}', SHARD_DIR / name, size) for name, size, _ in SHARDS[2:]]

downloads = []
tmpfs_bytes = 0
disk_bytes = 0
for orig_url, logical_path, size in all_targets:
    if logical_path.exists() and logical_path.stat().st_size == size:
        continue
    if logical_path.is_symlink() or logical_path.exists():
        logical_path.unlink(missing_ok=True)
    real_path = logical_path
    if ram_budget >= size:
        for mp in tmpfs_dirs:
            if mp_free.get(mp, 0) >= size + 4 * (1024**3):
                cand = mp / 'qwen38-fn' / logical_path.name
                cand.parent.mkdir(parents=True, exist_ok=True)
                if cand.exists() and cand.stat().st_size == size:
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

    # 并行解析全部文件的 CDN 签名直链（0.3s 完成）
    CHUNK = 64 * 1024 * 1024
    hf_tok = os.environ.get('HF_TOKEN', '').strip()
    if not hf_tok:
        try:
            from google.colab import userdata
            hf_tok = (userdata.get('HF_TOKEN') or '').strip()
        except Exception:
            hf_tok = ''
    def _resolve_cdn(item):
        orig_url, real_path, size = item
        req_hdr = {{'Range': 'bytes=0-0'}}
        if hf_tok:
            req_hdr['Authorization'] = 'Bearer ' + hf_tok
        cdn_url = orig_url
        for attempt in range(5):
            try:
                with requests.get(orig_url, headers=req_hdr, stream=True, timeout=30) as r:
                    if r.status_code == 429:
                        wait_s = float(r.headers.get('Retry-After', 2 ** (attempt + 1)))
                        time.sleep(min(wait_s, 30.0))
                        continue
                    r.raise_for_status()
                    cdn_url = r.url
                    break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return [
            (cdn_url, orig_url, real_path, start, min(start + CHUNK, size) - 1)
            for start in range(0, size, CHUNK)
        ]

    range_tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(downloads)) as rpool:
        for chunk_list in rpool.map(_resolve_cdn, downloads):
            range_tasks.extend(chunk_list)

    # 写出独立多进程下载脚本：8 个独立 Python 进程 × 4 线程 = 32 路并发流 + os.pwrite 原子直写，彻底绕开单进程 GIL
    tasks_file = Path('/tmp/qwen38_dl_tasks.json')
    tasks_file.write_text(json.dumps({{'need_bytes': need_bytes, 'tasks': range_tasks}}), encoding='utf-8')
    dl_script = Path('/tmp/qwen38_fast_dl.py')
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
''', encoding='utf-8')

    t_dl_start = time.monotonic()
    subprocess.run([sys.executable, str(dl_script), str(tasks_file)], check=True)
    dl_sec = max(time.monotonic() - t_dl_start, 0.1)
    print(f'全部权重下载完成：用时 {{dl_sec:.1f}}s，平均 {{need_bytes / (1024**2) / dl_sec:.0f}} MB/s', flush=True)

# 权重下载完成后再启动 Tailscale（与 llama-server 显存加载并行，避开 32 路并发流抢网卡导致握手超时）
fut_ts = bg_executor.submit(prepare_tailscale)

# 等待 llama.cpp 解压完成
fut_cpp.result()

# 5. 启动或复用 llama-server（代码、逆向与长程软件工程 Agent 特化）
PARALLEL_SLOTS = 1
SLOT_CONTEXT_TOKENS = 262144
TOTAL_CONTEXT_TOKENS = SLOT_CONTEXT_TOKENS * PARALLEL_SLOTS
MAX_OUTPUT_TOKENS = 65536
MTP_DRAFT_TOKENS = 4
PROMPT_CACHE_MIB = 16384
SLOT_SAVE_DIR = ROOT / 'slot-saves'
SLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CHAT_TMPL = ROOT / 'chat-template-clean.jinja'
ple_real = str((SHARD_DIR / PLE_SHARD[0]).resolve())
ple_in_tmpfs = ple_real.startswith(('/dev/shm/', '/var/colab/'))

# 原模板（Qwen 官方 xhigh 默认档 + 只保留最后一轮 <think>）会把客户端选的档位拉满，并在多轮追问时剥离历史思考导致缓存失效。
# 只读分片 1 的前 16 MiB 取出 tokenizer.chat_template（模板在文件偏移约 10.9 MB 处，64 KiB 读不到；分片 1 本体 0.5 GB，不整文件读入内存），
# 按 GGUF 字符串的长度前缀精确截取，不依赖任何厂商标记（jackjusko 不是 Unsloth GGUF，模板末尾没有 Unsloth 标记）：
# 1) 改档位映射：medium 为默认，high 不加额外反刍指令，xhigh 降级为 high；
# 2) 强制保留历史轮次的 <think> 块，避免多轮对话切换用户消息时剥离历史思考导致 KV 与线性注意力缓存失效重算。
# 放在重启判断之前：提取失败只会抛错，不会停掉正在运行的旧服务。
with MODEL.open('rb') as f:
    raw_head = f.read(16 * 1024 * 1024)
tmpl_key = b'tokenizer.chat_template'
key_pos = raw_head.index(struct.pack('<Q', len(tmpl_key)) + tmpl_key)
val_pos = key_pos + 8 + len(tmpl_key)
assert struct.unpack_from('<I', raw_head, val_pos)[0] == 8, 'tokenizer.chat_template 不是 GGUF 字符串类型'
tmpl_len = struct.unpack_from('<Q', raw_head, val_pos + 4)[0]
assert val_pos + 12 + tmpl_len <= len(raw_head), '模板超出前 16 MiB，请加大读取范围'
tmpl_str = raw_head[val_pos + 12:val_pos + 12 + tmpl_len].decode('utf-8')
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
new_effort_block = '''{{%- if enable_thinking is undefined or enable_thinking is true %}}
    {{%- set resolved_reasoning_effort = reasoning_effort|default('medium') %}}
    {{%- if resolved_reasoning_effort == 'xhigh' %}}
        {{%- set resolved_reasoning_effort = 'high' %}}
    {{%- endif %}}
    {{%- if resolved_reasoning_effort not in ('high', 'medium', 'low') %}}
        {{{{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are high, medium (default), and low.') }}}}
    {{%- endif %}}
    {{%- if resolved_reasoning_effort == 'medium' %}}
        {{%- set reasoning_instructions = 'Reasoning effort is set to medium. Keep your thinking concise and focused on the essential steps without repeating yourself.' %}}
    {{%- elif resolved_reasoning_effort == 'low' %}}
        {{%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}}
    {{%- endif %}}
{{%- endif %}}'''
assert old_effort_block in tmpl_str, '未找到预期的 reasoning_effort 模板块'
tmpl_str = tmpl_str.replace(old_effort_block, new_effort_block, 1)
old_preserve_line = '{{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}}'
assert old_preserve_line in tmpl_str, '未找到预期的 preserve_thinking 模板行'
tmpl_str = tmpl_str.replace(old_preserve_line, '{{%- if true %}}', 1)
old_tmpl_on_disk = CHAT_TMPL.read_text(encoding='utf-8') if CHAT_TMPL.exists() else ''
CHAT_TMPL.write_text(tmpl_str, encoding='utf-8')

def configured_server():
    if old_tmpl_on_disk != tmpl_str or not health():
        return False
    props = requests.get('http://127.0.0.1:%d/props' % PORT, headers=AUTH, timeout=10).json()
    if props.get('default_generation_settings', {{}}).get('n_ctx') != SLOT_CONTEXT_TOKENS:
        return False
    if props.get('total_slots') != PARALLEL_SLOTS:
        return False
    expected = {{
        '--mmproj': str(MMPROJ),
        '--alias': MODEL_ALIAS,
        '--spec-draft-n-max': str(MTP_DRAFT_TOKENS),
        '--spec-draft-p-min': '0.55',
        '--predict': str(MAX_OUTPUT_TOKENS),
        '-c': str(TOTAL_CONTEXT_TOKENS),
        '--parallel': str(PARALLEL_SLOTS),
        '--slot-save-path': str(SLOT_SAVE_DIR),
        '--cache-ram': str(PROMPT_CACHE_MIB),
        '--chat-template-file': str(CHAT_TMPL),
        '--reasoning-effort': 'medium',
        '--lazy-mode': 'off',
        '--temp': '0.25',
        '--top-p': '0.85',
        '--top-k': '20',
        '--min-p': '0.05',
        '--dry-multiplier': '0.5',
        '--dry-allowed-length': '6',
        '--dry-penalty-last-n': '2048',
        '--batch-size': '4096',
        '--ubatch-size': '2048',
    }}
    if not ple_in_tmpfs:
        expected['-lm'] = 'none'
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdigit():
            continue
        try:
            args = (procdir / 'cmdline').read_bytes().decode().split('\0')
            if args and Path(args[0]).name == 'llama-server' and str(MODEL) in args:
                # 旧版带 --api-key-file / --no-context-shift / --lora-scaled 的进程必须重启
                if any(flag in args for flag in ('--api-key-file', '--no-context-shift', '--lora-scaled')):
                    return False
                return all(flag in args and args[args.index(flag) + 1] == value
                           for flag, value in expected.items())
        except (FileNotFoundError, PermissionError, IndexError):
            pass
    return False

if health() and not configured_server():
    print('检测到旧配置或不同模型服务：重启 llama-server ...', flush=True)
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdigit():
            continue
        try:
            args = (procdir / 'cmdline').read_bytes().decode().split('\0')
            if args and Path(args[0]).name == 'llama-server':
                os.kill(int(procdir.name), 15)
                for _ in range(60):
                    stat = (procdir / 'stat').read_text() if procdir.exists() else ''
                    if not stat or stat.split(') ', 1)[1].startswith('Z'):
                        break
                    time.sleep(0.5)
                else:
                    os.kill(int(procdir.name), 9)
        except (FileNotFoundError, ProcessLookupError):
            pass

if configured_server():
    print('模型已在运行：Swift-1.5-Qwen3.8-Flash-Next Heretic V2 IQ4_XS-plus（原生破限） + 视觉 mmproj-BF16 + MTP Q4_K_M / 单路 × 262K 上下文 / 16 GiB Prompt Cache', flush=True)
else:
    if not freshly_downloaded and not ple_in_tmpfs:
        meminfo = Path('/proc/meminfo').read_text()
        cached_kb = next((int(l.split()[1]) for l in meminfo.splitlines() if l.startswith('Cached:')), 0)
        if cached_kb < 40 * 1024 * 1024:
            print('预热 PLE (N-gram) 表所在分片到主机内存 Page Cache ...', flush=True)
            with (SHARD_DIR / PLE_SHARD[0]).open('rb') as f:
                while f.read(32 * 1024 * 1024):
                    pass

    SERVER = next(DIST.rglob('llama-server'))
    env = os.environ.copy()
    libdirs = {{str(SERVER.parent)}}
    for pat in ('libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12'):
        libdirs.update(str(p.parent) for p in DIST.rglob(pat))
    env['LD_LIBRARY_PATH'] = ':'.join(sorted(libdirs)) + ':' + env.get('LD_LIBRARY_PATH', '')
    load_flags = ['--lazy-mode', 'off'] if ple_in_tmpfs else ['-lm', 'none', '--lazy-mode', 'off']
    cmd = [
        str(SERVER),
        '-m', str(MODEL),
        '--mmproj', str(MMPROJ),
        '--model-draft', str(DRAFT),
        '--spec-type', 'draft-mtp',
        '--spec-draft-n-max', str(MTP_DRAFT_TOKENS),
        '--spec-draft-p-min', '0.55',
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
        '--reasoning', 'on',
        '--reasoning-effort', 'medium',
        '--temp', '0.25',
        '--top-p', '0.85',
        '--top-k', '20',
        '--min-p', '0.05',
        '--dry-multiplier', '0.5',
        '--dry-base', '1.75',
        '--dry-allowed-length', '6',
        '--dry-penalty-last-n', '2048',
        '--flash-attn', 'on',
        '--cache-type-k', 'q8_0',
        '--cache-type-v', 'q8_0',
        '--batch-size', '4096',
        '--ubatch-size', '2048',
        '--predict', str(MAX_OUTPUT_TOKENS),
    ]
    log = (ROOT / 'server.log').open('a')
    log.write('\n=== START %s ===\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    qwen_server = proc
    t0 = time.monotonic()
    mode_desc = 'tmpfs 内存盘 MAP_POPULATE 零拷贝映射 PLE 表' if ple_in_tmpfs else 'CUDA_Host 锁页内存加载 PLE 表'
    print(f'加载 Swift-1.5 IQ4_XS 模型到 G4 (96GB VRAM + {{mode_desc}}) ...', flush=True)
    while time.monotonic() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError((ROOT / 'server.log').read_text(errors='replace')[-3500:])
        if health():
            break
        time.sleep(2)
    else:
        proc.terminate()
        raise RuntimeError('模型加载超时，请查看 /content/qwen38-fn/server.log')
    print('服务就绪 %.0fs，发送多样化长批次请求预热 CUDA/MTP/MoE/QSA 图 ...' % (time.monotonic() - t0), flush=True)
    warmup_text = (
        'System warmup check for Swift-1.5-Qwen3.8-Flash-Next on RTX PRO 6000 Blackwell: '
        'verify MoE expert routing, QSA k-pool lightning indexer, PLE gather, and MTP draft=4 verification. '
        '请简短计算 17 + 25 并只回复“就绪”。\n'
    ) * 16
    requests.post('http://127.0.0.1:%d/v1/chat/completions' % PORT, headers=AUTH, timeout=600,
                  json={{'model': MODEL_ALIAS,
                        'messages': [{{'role': 'user', 'content': warmup_text}}],
                        'max_tokens': 16, 'temperature': 0, 'reasoning_effort': 'low'}})

# 等待后台 Tailscale Service 绑定完成
fut_ts.result()
status = ts('serve', 'status').stdout
if SERVICE_FQDN not in status:
    raise RuntimeError('Service 未生效，serve status:\n' + status)

print('=' * 64)
print('总启动耗时： %.1fs' % (time.monotonic() - t_boot), flush=True)
print('网页版：     http://%s:443/' % SERVICE_FQDN, flush=True)
print('接口：       http://%s:443/v1' % SERVICE_FQDN, flush=True)
print('模型名：     %s' % MODEL_ALIAS, flush=True)
print('API key：    %s（服务端不校验，Magpie / pi 保留原值即可，网页端无需填写）' % KEY, flush=True)
print('=' * 64)
print('当前配置：Swift-1.5 Heretic V2 IQ4_XS-plus (94.7GB 原生破限，代码/架构/逆向工程校准，无 LoRA，PLE 分片 tmpfs 内存盘 MAP_POPULATE 零拷贝常驻) + mmproj-BF16 + MTP Q4_K_M (draft 4, p_min 0.55)，单路 × 262144 上下文，16 GiB Prompt Cache + 槽位持久化，batch 4096 / ubatch 2048，65536 输出上限，默认 medium 思考（支持 low/medium/high，保留历史思考防重算），代码高确定性采样 temp 0.25 / top-p 0.85 / top-k 20 / min-p 0.05，Q8 KV，工程防死循环 DRY 0.5 / base 1.75 / 允许长度 6 / 回看 2048', flush=True)
""" + console_code(6, '服务已就绪！', 'G4', '<b>当前状态：</b><span style="color:#5cb85c">🔓 Heretic V2 原生破限，满速运行</span> — 用完点黄色按钮立即释放 GPU，WebUI 标签页会随之自动关闭')

INSPECT_CODE = r"""# 可选：查看当前配置、GPU 显存与主机内存占用（服务端不校验 key，带不带 Authorization 都行）
import json, subprocess
from pathlib import Path
import requests

root = Path('/content/qwen38-fn')
hdr = {'Authorization': 'Bearer ' + (root / '.llama-api-key').read_text().strip()}
r = requests.get('http://127.0.0.1:8081/props', headers=hdr, timeout=10)
r.raise_for_status()
props = r.json()
p = props['default_generation_settings']
print('每槽上下文:', p['n_ctx'], 'tokens | 并发槽位数:', props.get('total_slots'), '| 总 KV 池:', p['n_ctx'] * int(props.get('total_slots', 1)), 'tokens')
for procdir in Path('/proc').iterdir():
    if not procdir.name.isdigit():
        continue
    try:
        args = (procdir / 'cmdline').read_bytes().decode().split('\0')
        if args and Path(args[0]).name == 'llama-server':
            flags = ('--alias', '--mmproj', '-c', '--parallel', '--predict', '--spec-draft-n-max', '--reasoning', '--reasoning-effort', '--cache-ram', '--dry-multiplier', '--dry-base', '--dry-allowed-length', '--dry-penalty-last-n', '--batch-size', '--ubatch-size', '--cache-type-k', '--cache-type-v', '-lm', '--lazy-mode')
            print('实际启动参数:', json.dumps({k: args[args.index(k) + 1] for k in flags if k in args}, ensure_ascii=False))
            print('已移除的旧参数仍在命令行中:', [f for f in ('--api-key-file', '--no-context-shift', '--lora-scaled') if f in args] or '无')
    except (FileNotFoundError, PermissionError, IndexError):
        pass
print('GPU:', subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader'))
print('Host RAM:', subprocess.getoutput('free -h | head -n 2'))
"""

VERIFY_CODE = r"""# 全链路验收：接口 / 思考 / 速度 / MTP / 工具调用
import json, time
from pathlib import Path
import requests

ROOT = Path('/content/qwen38-fn')
MODEL_ALIAS = 'Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS'
base = 'http://127.0.0.1:8081'
headers = {'Authorization': 'Bearer ' + (ROOT / '.llama-api-key').read_text().strip()}

models = requests.get(base + '/v1/models', headers=headers, timeout=15).json()
print('模型：', [m['id'] for m in models['data']])

def call(tag, messages, max_tokens=64, effort='medium', tools=None, stream=False):
    body = {'model': MODEL_ALIAS, 'messages': messages,
            'reasoning_effort': effort, 'temperature': 0,
            'max_tokens': max_tokens, 'stream': stream}
    if tools:
        body['tools'] = tools
    t = time.monotonic()
    r = requests.post(base + '/v1/chat/completions', headers=headers, json=body, timeout=600)
    if stream:
        ttft = None
        text = []
        n = 0
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith('data: '):
                continue
            payload = line[6:]
            if payload == '[DONE]':
                break
            tok = json.loads(payload)['choices'][0].get('delta', {}).get('content') or ''
            if tok and ttft is None:
                ttft = time.monotonic() - t
            text.append(tok)
            n += 1
        print('--- %s (stream) ttft=%.2fs wall=%.2fs chunks=%d' % (tag, ttft or -1, time.monotonic() - t, n))
        print('CONTENT:', ''.join(text)[:200].replace('\\n', ' | '))
        return None
    d = r.json()
    wall = time.monotonic() - t
    msg = d['choices'][0]['message']
    print('--- %s wall=%.2fs' % (tag, wall))
    print('REASONING:', (msg.get('reasoning_content') or '(none)')[:200].replace('\\n', ' | '))
    print('CONTENT:', (msg.get('content') or '')[:200].replace('\\n', ' | '))
    if msg.get('tool_calls'):
        print('TOOL_CALLS:', json.dumps(msg['tool_calls'], ensure_ascii=False)[:300])
    print('USAGE:', json.dumps(d.get('usage')))
    tm = d.get('timings') or {}
    keep = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tm.items()
            if k in ('prompt_n', 'predicted_n', 'prompt_per_second', 'predicted_per_second')}
    print('TIMINGS:', json.dumps(keep))
    return d

q = '请先简短思考，再回答：17 × 23 等于多少？只给最终数字。'
call('cold-1', [{'role': 'user', 'content': q}])
call('warm-2', [{'role': 'user', 'content': q}])
call('warm-3', [{'role': 'user', 'content': '用三句话说明 Qwen3.8-Flash-Next 的 PLE (N-gram) 表为什么适合留在主机内存。'}], max_tokens=180)
call('stream', [{'role': 'user', 'content': '从 1 数到 20，用空格分隔。'}], max_tokens=80, stream=True)
tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': '查询城市天气',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
call('tools', [{'role': 'user', 'content': '亚特兰大现在天气如何？用 get_weather 工具查。'}], tools=tools)

log = (ROOT / 'server.log').read_text(errors='replace').splitlines()
spec = [l for l in log if any(w in l.lower() for w in ('draft', 'speculat', 'accepted', 'mtp'))]
print('MTP_LOG_LINES:', len(spec))
print('\\n'.join(spec[-6:]))
"""

STOP_CODE = r"""# 长程工程任务防断连护航（前台保活 + 45分钟无请求自动停机省 CU）或立即点击按钮释放 G4 GPU
import ipywidgets as widgets
import os, subprocess, threading, time
import requests
from google.colab import runtime
from IPython.display import display
from html import escape

IDLE_LIMIT_MIN = 45
stop_button = widgets.Button(description='🛑 立即结束 G4 GPU 运行时', button_style='warning')
guard_button = widgets.Button(description='🛡️ 启动长程防断连护航 (45m空闲自停)', button_style='info')
stop_status = widgets.HTML(value='跑长程代码/工程任务时可点蓝色按钮保持前台心跳防 Colab 90m 断连（连续 45 分钟无推理请求会自动释放 G4 省 CU）；用完点黄色按钮立即释放。')
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

STOP_MD = """**结束状态与用量**

释放 G4 GPU 后，右上角显示“重新连接”表示已断开；进入 **代码执行程序 → 管理会话**，显示 **没有正在进行的会话** 即表示 G4 已释放。

计费按运行时挂着的时间算，空闲一样扣。Colab G4（RTX PRO 6000 Blackwell 96GB）约 **8.71–8.90 计算单元/小时**（A100 40GB 约 5.37 CU/小时）。

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
        {'cell_type': 'markdown', 'metadata': {'id': 'overview'}, 'source': lines(OVERVIEW_MD)},
        {'cell_type': 'code', 'metadata': {'id': 'persistent-setup'}, 'source': lines(ONECLICK_CODE), 'execution_count': None, 'outputs': []},
        {'cell_type': 'code', 'metadata': {'id': 'inspect-props'}, 'source': lines(INSPECT_CODE), 'execution_count': None, 'outputs': []},
        {'cell_type': 'code', 'metadata': {'id': 'verify-fn'}, 'source': lines(VERIFY_CODE), 'execution_count': None, 'outputs': []},
        {'cell_type': 'code', 'metadata': {'id': 'stop-runtime'}, 'source': lines(STOP_CODE), 'execution_count': None, 'outputs': []},
        {'cell_type': 'markdown', 'metadata': {'id': 'stop-status-help'}, 'source': lines(STOP_MD)},
    ],
}

G4_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='')
print('Generated:', G4_NB)

sync_requested(nb, G4_DRIVE_ID, label='qwen38-fn')
