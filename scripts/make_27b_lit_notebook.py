"""Generate the Colab notebook for Literary #1: Hemmingway-1-Heretic Q8_0 (native Heretic ARA uncensored) + Chinese-novel SFT LoRA baked into the weights at boot + MTP + mmproj.

Writes notebooks/qwen3.8-27b-pi-colab.ipynb. Reuses the Tailscale Service (svc:qwen) and API key from the
previous copy of that notebook (or from the G4 notebook) for time-shared operation with Comprehensive #1.
Usage: python scripts/make_27b_lit_notebook.py [--sanitized] [--output-dir DIR] [--sync-colab [--run-cell1] [--release-runtime]]
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
guard_sync(SANITIZED, CONFIG_SOURCES)
LIT_NB = OUTPUT_DIR / 'qwen3.8-27b-pi-colab.ipynb'
G4_NB = OUTPUT_DIR / 'qwen3.8-flash-next-g4-colab.ipynb'

LIT_DRIVE_ID = CONFIG['lit_drive_id']
G4_DRIVE_ID = CONFIG['g4_drive_id']
SERVICE_FQDN_VALUE = CONFIG['tailnet_fqdn']
TS_CLIENT_ID_VALUE = CONFIG['ts_client_id']
API_KEY = CONFIG['api_key']
TS_SECRET = CONFIG['ts_client_secret']


def lines(text: str) -> list[str]:
    raw = text.strip('\n').split('\n')
    return [line + '\n' for line in raw]


OVERVIEW_MD = """# Qwen3.8-27B · Colab 中文小说创作与长程打磨服务（Literary #1：Hemmingway-1 Heretic Q8_0 原生破限 + 永久固化 8000 万字中文小说 LoRA）

主权重 `darrellbest/Hemmingway-1-Heretic-GGUF`（`Hemmingway-1-Heretic-Q8_0.gguf`，29.05 GB，**原生 Heretic ARA 破限**），并在首格下载校验后通过 GPU (`torch.cuda`) **原地将 `Moeblack/Qwen3.8-27B-chinese-taste-sft-lora`（8000 万字中文长篇小说 SFT）256 个矩阵永久烘焙进 `Q8_0` 权重**——启动 `llama-server` 时**不挂载任何 `--lora-scaled`**，彻底消除运行期 LoRA 算子减速，兼具 **EQ-Bench 4 全球第 3 的海明威拟人底子 + 8000 万字中文长句文风（长句率 10%→25%）+ 83–87 tok/s 满速输出**！配 `Hemmingway-1-Heretic-mmproj-BF16` 视觉投影器与 `mtp-Q4_0` 投机头（`draft-mtp`，`n_max=3, p_min=0.45`）。默认 `low` 思考档，写作任务不浪费思考 token。

**中文长篇小说创作 / 长程打磨参数**

- `--parallel 1`，整张卡的显存带宽只服务你一条流。
- `--cache-ram 16384` + `--slot-save-path /content/qwen27b-lit/slot-saves`：16 GiB 主机内存 Prompt Cache + 槽位持久化，网页端回退分支、多轮改稿重新生成时直接恢复 KV 与 Gated DeltaNet 状态，不重算长前缀。
- `--batch-size 4096 --ubatch-size 2048 --predict 65536`：单次最大生成提升至 **65,536 tokens**，一次塞入数万字设定集或整章长程连写永不截断。
- `--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 4 --dry-penalty-last-n 1024`：放过 2–4 个 token 的中文人名、成语与保留的原稿句子，仍拦截局部复读循环。
- 不带 `--api-key-file`：`svc:qwen` 仅 tailnet 内可见，网页端打开即可聊天，无需在设置里粘贴 key。

**显存自适应（首格自动判断）**

| GPU | 并发 × 每槽上下文 | 总 KV 池 | ubatch |
|---|---|---|---|
| G4 / RTX PRO 6000 96GB（默认） | 1 × 262,144 | 262,144 | 2048 |
| A100 40GB（回退） | 1 × 131,072 | 131,072 | 512 |
| 低于 35 GB | 报错并提示切换到 G4 或 A100 | — | — |

**与 Comprehensive #1 的关系（分时复用固定入口，零干扰）**

- 网页：`http://__SERVICE_FQDN__:443/`（首格结束时自动弹出；点首格的停机按钮会一并关闭该标签页）
- 接口：`http://__SERVICE_FQDN__:443/v1`
- 模型名：`Qwen3.8-27B-Antislop-ZH-Q8_0`（服务端不校验 key，客户端带不带 `Authorization` 都行）
- 运行时目录：`/content/qwen27b-lit`（与 Comprehensive #1 的 `/content/qwen38-fn` 完全隔离）
- 分时复用 `svc:qwen`：同一时刻只开一个运行时；后启动的节点自动以 `tag:colab` 接管 `svc:qwen`。

**启动加速设计（DDR5 内存盘 `tmpfs` 直写 + 32 路多进程零 GIL 下载）**

1. **绕过 `/content` 持久盘写限速墙**：全部 4 个文件（共 29.8 GiB）直接写进 `/dev/shm` 内存盘并软链回 `/content/qwen27b-lit`。
2. **8 进程 × 4 线程（32 路并发流）+ `os.pwrite` 原子直写**：并行解析一次 HF CDN 签名直链后 32 路直写内存盘。
3. **后台 SHA-256 校验**：新下载的文件在 `llama-server` 加载权重的同时并行校验哈希，校验通过才宣布启动完成。
4. **首格控制台**：先显示「🚀 服务已就绪」横幅（含 WebUI 链接），其下是 `🛑 结束 GPU 运行时 (省 CU)` 按钮，点击后关闭 WebUI 标签页、停服务并释放运行时。
"""

OVERVIEW_MD = OVERVIEW_MD.replace('__SERVICE_FQDN__', SERVICE_FQDN_VALUE)

ONECLICK_CODE = rf"""# 一键启动（Colab 内存盘极速版）：tmpfs 内存盘 + 32路多进程下载 + 后台 SHA-256 + llama.cpp + Tailscale
import concurrent.futures, hashlib, json, os, shutil, struct, subprocess, sys, threading, time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter

t_boot = time.monotonic()
ROOT = Path('/content/qwen27b-lit')
DIST = ROOT / 'llama-dist'
MODEL_ALIAS = 'Qwen3.8-27B-Antislop-ZH-Q8_0'
KEY_FILE = ROOT / '.llama-api-key'
PORT = 8081
KEY = '{API_KEY}'

HEMMINGWAY_REV = 'd15d84a09627b54b70f77b26af46524c7dda438f'
HEMMINGWAY_HF = 'https://huggingface.co/darrellbest/Hemmingway-1-Heretic-GGUF/resolve/' + HEMMINGWAY_REV
ZH_REV = '6c8950c79ade177b6fcf74c6c921369f5715b225'
ZH_HF = 'https://huggingface.co/Moeblack/Qwen3.8-27B-chinese-taste-sft-lora/resolve/' + ZH_REV
PI_REV = '97caacb971d9246520fa8db5282a06bd36b103df'
PI_HF = 'https://huggingface.co/bytkim/Qwen3.8-27B-pi-GGUF/resolve/' + PI_REV
CPP_TAG = 'b12032'
CPP_URL = 'https://github.com/ggml-org/llama.cpp/releases/download/' + CPP_TAG

MODEL = ROOT / 'Hemmingway-1-Heretic-Q8_0.gguf'
MODEL_SIZE = 29_047_084_288
MODEL_SHA = '1e76479ae4f05530035d788453b27fc76bcb7a46faa7a6bcee6e2b69b0c3bb18'
LORA_ZH = ROOT / 'qwen3.8-27b-chinese-taste-sft-lora-q8_0.gguf'
LORA_ZH_SIZE = 15_995_264
LORA_ZH_SHA = '02211ccec9e44a0c753ea37682105b4f0faf39d37512ac8f931d9dd3164b3c3d'
MMPROJ = ROOT / 'Hemmingway-1-Heretic-mmproj-BF16.gguf'
MMPROJ_SIZE = 931_145_792
MMPROJ_SHA = '64a7db66b2cb3031c5fd9e008a78cabc5b4043c83244d8a011edea3f8239a935'
DRAFT = ROOT / 'mtp-Qwen3.8-27B-pi-Q4_0.gguf'
DRAFT_SIZE = 2_008_054_464
DRAFT_SHA = '6ce59430911676c54217db078ed51ffa0408831a15e2acf4436302c4e8643381'
EXPECTED_SHA = {{MODEL: MODEL_SHA, LORA_ZH: LORA_ZH_SHA, MMPROJ: MMPROJ_SHA, DRAFT: DRAFT_SHA}}

SERVICE = 'svc:qwen'
SERVICE_FQDN = '{SERVICE_FQDN_VALUE}'
SERVICE_TAG = 'tag:colab'
TS_HOSTNAME = 'qwen-lit-colab'
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

# 1. 硬件前置校验与显存自适应：G4 96GB 单路 262K；A100 40GB 单路 131K；更小的卡直接报错
gpu_info = subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits').strip()
if not gpu_info or ',' not in gpu_info:
    raise RuntimeError('未检测到 NVIDIA GPU，请在右上角把运行时切换到 G4 GPU（96GB）或 A100 GPU（40GB）。输出：' + gpu_info)
gpu_name, gpu_mib_str = [x.strip() for x in gpu_info.split(',', 1)]
gpu_mib = int(gpu_mib_str)
print(f'检测到 GPU: {{gpu_name}} ({{gpu_mib / 1024:.1f}} GiB VRAM)', flush=True)
PARALLEL_SLOTS = 1
if gpu_mib >= 80000:
    SLOT_CONTEXT_TOKENS = 262144
    UBATCH_SIZE = 2048
    gpu_tier = 'G4 96GB：单路 × 262144'
elif gpu_mib >= 35000:
    SLOT_CONTEXT_TOKENS = 131072
    UBATCH_SIZE = 512
    gpu_tier = 'A100 40GB 回退：单路 × 131072'
else:
    raise RuntimeError(f'当前 GPU 显存仅 {{gpu_mib / 1024:.1f}} GiB（{{gpu_name}}），装不下 Q8_0 主权重 + MTP + 128K KV。请切换到 G4 (96GB) 或 A100 (40GB) 运行时！')
TOTAL_CONTEXT_TOKENS = SLOT_CONTEXT_TOKENS * PARALLEL_SLOTS
print(f'显存档位：{{gpu_tier}}（总 KV 池 {{TOTAL_CONTEXT_TOKENS}} tokens，ubatch {{UBATCH_SIZE}}）', flush=True)

ROOT.mkdir(parents=True, exist_ok=True)
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

# 3. 后台任务 B（权重下载完成后启动）：启动 Tailscale 并挂载 svc:qwen（自动审批与证书签发）
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

# 4. 后台任务 C：新下载文件的 SHA-256 校验（与 llama-server 加载权重并行；未通过则不宣布启动完成）
def verify_sha256(fresh):
    def one(path):
        digest = hashlib.sha256()
        buf = bytearray(16 * 1024 * 1024)
        view = memoryview(buf)
        with path.open('rb') as f:
            while True:
                n = f.readinto(buf)
                if not n:
                    break
                digest.update(view[:n])
        return path, digest.hexdigest()
    t_hash = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fresh)) as pool:
        got = list(pool.map(one, fresh))
    bad = [(p.name, h, EXPECTED_SHA[p]) for p, h in got if h != EXPECTED_SHA[p]]
    return bad, time.monotonic() - t_hash

bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
fut_cpp = bg_executor.submit(prepare_llama_cpp)

# 5. 内存盘 (tmpfs: /dev/shm + /var/colab) 智能分配 + 8进程×4线程 (32路) 零 GIL 极速并发下载
# Colab 的 /content 挂在 GCE 持久盘上，宿主机写吞吐上限仅 ~206-353 MiB/s。
# 全部 4 个文件共 29.8 GiB，直接写进 DDR 内存盘并软链回 /content/qwen27b-lit，彻底绕开磁盘写限速。
subprocess.run(['mount', '-o', 'remount,size=115G', '/dev/shm'], capture_output=True)
tmpfs_dirs = []
for mline in Path('/proc/mounts').read_text().splitlines():
    mparts = mline.split()
    if len(mparts) >= 3 and mparts[2] == 'tmpfs' and mparts[1] in ('/dev/shm', '/var/colab'):
        tmpfs_dirs.append(Path(mparts[1]))
mp_free = {{mp: shutil.disk_usage(mp).free for mp in tmpfs_dirs}}
meminfo_txt = Path('/proc/meminfo').read_text()
avail_kb = next((int(l.split()[1]) for l in meminfo_txt.splitlines() if l.startswith('MemAvailable:')), 0)
ram_budget = max(0, avail_kb * 1024 - 45 * (1024**3))

# 优先级：主权重最先进内存盘，其次是中文网文 LoRA、视觉投影器与 MTP 头
all_targets = [
    (f'{{HEMMINGWAY_HF}}/Hemmingway-1-Heretic-Q8_0.gguf', MODEL, MODEL_SIZE),
    (f'{{ZH_HF}}/qwen3.8-27b-chinese-taste-sft-lora-q8_0.gguf', LORA_ZH, LORA_ZH_SIZE),
    (f'{{HEMMINGWAY_HF}}/Hemmingway-1-Heretic-mmproj-BF16.gguf', MMPROJ, MMPROJ_SIZE),
    (f'{{PI_HF}}/mtp/mtp-Qwen3.8-27B-pi-Q4_0.gguf', DRAFT, DRAFT_SIZE),
]

downloads = []
fresh_files = []
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
                cand = mp / 'qwen27b-lit' / logical_path.name
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
    fresh_files.append(logical_path)

freshly_downloaded = bool(downloads)
fut_sha = None
if downloads:
    need_bytes = sum(size for _, _, size in downloads)
    print(f'待并发下载: {{need_bytes / (1024**3):.2f}} GiB ({{len(downloads)}} 个文件) → DDR 内存盘 tmpfs: {{tmpfs_bytes / (1024**3):.1f}} GiB | 本地磁盘: {{disk_bytes / (1024**3):.1f}} GiB', flush=True)

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
    tasks_file = Path('/tmp/qwen27b_dl_tasks.json')
    tasks_file.write_text(json.dumps({{'need_bytes': need_bytes, 'tasks': range_tasks}}), encoding='utf-8')
    dl_script = Path('/tmp/qwen27b_fast_dl.py')
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
    bad_hashes, hash_sec = verify_sha256(fresh_files)
    if bad_hashes:
        raise RuntimeError('SHA-256 校验失败：' + json.dumps(bad_hashes, ensure_ascii=False))
    print(f'SHA-256 校验通过：{{len(fresh_files)}} 个新下载文件（{{hash_sec:.1f}}s）', flush=True)
    (ROOT / '.zh_lora_baked').unlink(missing_ok=True)

# 权重下载完成后再启动 Tailscale（与 GPU 烘焙及 llama-server 显存加载并行，避开 32 路并发流抢网卡导致握手超时）
fut_ts = bg_executor.submit(prepare_tailscale)

# 5.5 原地 GPU 固化 8000 万字中文小说 LoRA 到 Hemmingway-1-Heretic-Q8_0.gguf（零额外内存盘占用，彻底消除运行期 32% LoRA 减速）
BAKED_MARKER = ROOT / '.zh_lora_baked'
if not BAKED_MARKER.exists():
    import mmap, torch
    t_bake = time.monotonic()
    print('正在用 GPU 原地将 8000 万字中文小说 LoRA (256 矩阵, scale=1.0) 固化烘焙进 Q8_0 主权重 ...', flush=True)
    def _parse_gguf_dir(mm):
        pos = 24
        _, _, n_tensors, n_kv = struct.unpack_from('<4sIQQ', mm, 0)
        def _rstr():
            nonlocal pos
            n, = struct.unpack_from('<Q', mm, pos); pos += 8
            s = bytes(mm[pos:pos+n]).decode('utf-8', errors='replace'); pos += n
            return s
        def _skip(t):
            nonlocal pos
            sz = {{0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}}
            if t in sz: pos += sz[t]
            elif t == 8: _rstr()
            elif t == 9:
                et, n = struct.unpack_from('<IQ', mm, pos); pos += 12
                for _ in range(n): _skip(et)
        for _ in range(n_kv):
            _rstr()
            vt, = struct.unpack_from('<I', mm, pos); pos += 4
            _skip(vt)
        tdir = {{}}
        for _ in range(n_tensors):
            name = _rstr()
            nd, = struct.unpack_from('<I', mm, pos); pos += 4
            dims = [struct.unpack_from('<Q', mm, pos + 8*i)[0] for i in range(nd)]; pos += 8*nd
            tt, off = struct.unpack_from('<IQ', mm, pos); pos += 12
            tdir[name] = (dims, tt, off)
        return tdir, (pos + 31) // 32 * 32

    with LORA_ZH.resolve().open('rb') as fl, MODEL.resolve().open('r+b') as fb:
        mm_l = mmap.mmap(fl.fileno(), 0, access=mmap.ACCESS_READ)
        mm_b = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_WRITE)
        try:
            lt, l_off = _parse_gguf_dir(mm_l)
            bt, b_off = _parse_gguf_dir(mm_b)
            targets = sorted({{k[:-7] for k in lt if k.endswith('.lora_a')}})
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                for tname in targets:
                    (in_dim, r_dim), _, off_a = lt[tname + '.lora_a']
                    (_, out_dim), _, off_b = lt[tname + '.lora_b']
                    (_, _), _, off_w = bt[tname]
                    n_blk_a = r_dim * (in_dim // 32)
                    raw_a = torch.frombuffer(memoryview(mm_l)[l_off + off_a : l_off + off_a + n_blk_a * 34], dtype=torch.uint8).reshape(n_blk_a, 34).to(dev)
                    da = raw_a[:, :2].contiguous().view(torch.float16).to(torch.float32)
                    qa = raw_a[:, 2:].contiguous().view(torch.int8).to(torch.float32)
                    A = (da * qa).reshape(r_dim, in_dim)
                    raw_b = torch.frombuffer(memoryview(mm_l)[l_off + off_b : l_off + off_b + out_dim * r_dim * 2], dtype=torch.uint8).view(torch.float16).reshape(out_dim, r_dim).to(dev, dtype=torch.float32)
                    n_blk_w = out_dim * (in_dim // 32)
                    w_start = b_off + off_w
                    w_end = w_start + n_blk_w * 34
                    raw_w = torch.frombuffer(memoryview(mm_b)[w_start:w_end], dtype=torch.uint8).reshape(n_blk_w, 34).to(dev)
                    dw = raw_w[:, :2].contiguous().view(torch.float16).to(torch.float32)
                    qw = raw_w[:, 2:].contiguous().view(torch.int8).to(torch.float32)
                    W = (dw * qw).reshape(out_dim, in_dim)
                    W.addmm_(raw_b, A, beta=1.0, alpha=2.0)
                    W_blk = W.reshape(n_blk_w, 32)
                    d_new = (W_blk.abs().amax(dim=1, keepdim=True) / 127.0).to(torch.float16)
                    d_f32 = d_new.to(torch.float32)
                    q_new = torch.where(d_f32 > 0, torch.round(W_blk / torch.where(d_f32 > 0, d_f32, torch.ones_like(d_f32))), torch.zeros_like(W_blk)).clamp(-127, 127).to(torch.int8)
                    out_blk = torch.empty((n_blk_w, 34), dtype=torch.uint8, device=dev)
                    out_blk[:, :2] = d_new.view(torch.uint8)
                    out_blk[:, 2:] = q_new.view(torch.uint8)
                    memoryview(mm_b)[w_start:w_end] = out_blk.cpu().numpy().tobytes()
            torch.cuda.empty_cache()
        finally:
            mm_l.close()
            mm_b.close()
    BAKED_MARKER.write_text('baked=Moeblack-zh-sft-r2-alpha4-scale1.0\n')
    print(f'8000 万字中文小说 LoRA 原地烘焙完成（{{len(targets)}} 个 Q8_0 矩阵，耗时 {{time.monotonic() - t_bake:.1f}}s，运行期零 LoRA 损耗）', flush=True)

# 等待 llama.cpp 解压完成
fut_cpp.result()

# 6. 启动或复用 llama-server（中文小说创作 + 长程特化）
MAX_OUTPUT_TOKENS = 65536
MTP_DRAFT_TOKENS = 3
PROMPT_CACHE_MIB = 16384
SLOT_SAVE_DIR = ROOT / 'slot-saves'
SLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CHAT_TMPL = ROOT / 'chat-template-clean.jinja'

# 原模板（Qwen 官方 xhigh 默认档）会把客户端选的档位拉满。Hemmingway 的 GGUF 不是 Unsloth 打包，模板末尾没有 Unsloth 标记，
# 所以只读主权重的前 16 MiB 取出 tokenizer.chat_template（模板在文件偏移约 10.9 MB 处，64 KiB 读不到；主权重本体 29.0 GB，绝不能整文件读入内存），
# 按 GGUF 字符串的长度前缀精确截取，不依赖任何厂商标记：
# 1) 改档位映射：medium 为默认，high 不加额外反刍指令，xhigh 降级为 high；写作任务用 low；
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
        '--model-draft': str(DRAFT),
        '--alias': MODEL_ALIAS,
        '--spec-draft-n-max': str(MTP_DRAFT_TOKENS),
        '--spec-draft-p-min': '0.45',
        '--predict': str(MAX_OUTPUT_TOKENS),
        '-c': str(TOTAL_CONTEXT_TOKENS),
        '--parallel': str(PARALLEL_SLOTS),
        '--slot-save-path': str(SLOT_SAVE_DIR),
        '--cache-ram': str(PROMPT_CACHE_MIB),
        '--chat-template-file': str(CHAT_TMPL),
        '--reasoning-effort': 'low',
        '--temp': '0.85',
        '--top-p': '0.92',
        '--top-k': '40',
        '--min-p': '0.05',
        '--dry-multiplier': '0.8',
        '--dry-allowed-length': '4',
        '--dry-penalty-last-n': '1024',
        '--batch-size': '4096',
        '--ubatch-size': str(UBATCH_SIZE),
    }}
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
    print('模型已在运行：Hemmingway-1 Heretic Q8_0（原生破限 + 永久固化 8000 万字中文小说 LoRA） + MTP Q4_0 draft 3 + mmproj-BF16 / 单路 262K 上下文 / 16 GiB Prompt Cache / 64K 输出上限', flush=True)
else:
    SERVER = next(DIST.rglob('llama-server'))
    env = os.environ.copy()
    libdirs = {{str(SERVER.parent)}}
    for pat in ('libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12'):
        libdirs.update(str(p.parent) for p in DIST.rglob(pat))
    env['LD_LIBRARY_PATH'] = ':'.join(sorted(libdirs)) + ':' + env.get('LD_LIBRARY_PATH', '')
    cmd = [
        str(SERVER),
        '-m', str(MODEL),
        '--mmproj', str(MMPROJ),
        '--model-draft', str(DRAFT),
        '--spec-type', 'draft-mtp',
        '--spec-draft-n-max', str(MTP_DRAFT_TOKENS),
        '--spec-draft-p-min', '0.45',
        '--slot-save-path', str(SLOT_SAVE_DIR),
        '--device', 'CUDA0',
        '-ngl', '99',
        '--cache-ram', str(PROMPT_CACHE_MIB),
        '-c', str(TOTAL_CONTEXT_TOKENS),
        '--parallel', str(PARALLEL_SLOTS),
        '--host', '127.0.0.1',
        '--port', str(PORT),
        '--alias', MODEL_ALIAS,
        '--jinja',
        '--chat-template-file', str(CHAT_TMPL),
        '--reasoning', 'on',
        '--reasoning-effort', 'low',
        '--temp', '0.85',
        '--top-p', '0.92',
        '--top-k', '40',
        '--min-p', '0.05',
        '--dry-multiplier', '0.8',
        '--dry-base', '1.75',
        '--dry-allowed-length', '4',
        '--dry-penalty-last-n', '1024',
        '--flash-attn', 'on',
        '--cache-type-k', 'q8_0',
        '--cache-type-v', 'q8_0',
        '--batch-size', '4096',
        '--ubatch-size', str(UBATCH_SIZE),
        '--predict', str(MAX_OUTPUT_TOKENS),
    ]
    log = (ROOT / 'server.log').open('a')
    log.write('\n=== START %s ===\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    qwen_server = proc
    t0 = time.monotonic()
    model_real = str(MODEL.resolve())
    where = 'tmpfs 内存盘' if model_real.startswith(('/dev/shm/', '/var/colab/')) else '本地磁盘'
    print(f'加载 Hemmingway-1 Heretic Q8_0 到 {{gpu_name}}（权重来自{{where}}）...', flush=True)
    while time.monotonic() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError((ROOT / 'server.log').read_text(errors='replace')[-3500:])
        if health():
            break
        time.sleep(2)
    else:
        proc.terminate()
        raise RuntimeError('模型加载超时，请查看 /content/qwen27b-lit/server.log')
    print('服务就绪 %.0fs，发送中文长批次请求预热 CUDA/MTP 图 ...' % (time.monotonic() - t0), flush=True)
    warmup_text = (
        'System warmup check for Hemmingway-1 Heretic Q8_0 with MTP draft=6: '
        '请用一句话描述雨夜里的旧书店，并只回复“就绪”。\n'
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
print('当前配置：Hemmingway-1 Heretic Q8_0（原生 Heretic ARA 破限 + 永久固化烘焙 8000 万字中文小说 SFT LoRA，运行期零 LoRA 减速）；MTP Q4_0 (draft 3, p_min 0.45)；mmproj-BF16；%s；batch 4096 / ubatch %d；16 GiB Prompt Cache + 槽位持久化；65536 输出上限；默认 low 思考（支持 low/medium/high，保留历史思考防重算）；temp 0.85 / top-p 0.92 / top-k 40 / min-p 0.05；Q8 KV；DRY 0.8 / base 1.75 / 允许长度 4 / 回看 1024' % (gpu_tier, UBATCH_SIZE), flush=True)
print('GPU 显存：', subprocess.getoutput('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'), flush=True)
""" + console_code(7, '中文小说创作服务已就绪！', 'GPU', '<b>当前状态：</b><span style="color:#5cb85c">✍️ 海明威原生破限 + 8000万字中文小说 LoRA 已永久固化烘焙（零减速满血创作态，64K 单次输出）</span> — 用完点黄色按钮释放，WebUI 标签页会随之自动关闭')

INSPECT_CODE = r"""# 可选：查看当前配置、GPU 显存与主机内存占用，或啟動长程任务防断连护航（Keepalive + 45分钟空闲自动释放 GPU）
import json, subprocess, time
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
import requests

root = Path('/content/qwen27b-lit')
base = 'http://127.0.0.1:8081'
hdr = {'Authorization': 'Bearer ' + (root / '.llama-api-key').read_text().strip()}

r = requests.get(base + '/props', headers=hdr, timeout=10)
r.raise_for_status()
props = r.json()
p = props['default_generation_settings']
print('每槽上下文:', p['n_ctx'], 'tokens | 并发槽位数:', props.get('total_slots'), '| 总 KV 池:', p['n_ctx'] * int(props.get('total_slots', 1)), 'tokens')
print('中文小说 LoRA 烘焙状态:', (root / '.zh_lora_baked').read_text().strip() if (root / '.zh_lora_baked').exists() else '未烘焙')
for procdir in Path('/proc').iterdir():
    if not procdir.name.isdigit():
        continue
    try:
        args = (procdir / 'cmdline').read_bytes().decode().split('\0')
        if args and Path(args[0]).name == 'llama-server':
            flags = ('--alias', '--mmproj', '--model-draft', '-c', '--parallel', '--predict', '--spec-draft-n-max', '--spec-draft-p-min', '--slot-save-path', '--reasoning', '--reasoning-effort', '--temp', '--top-p', '--top-k', '--min-p', '--cache-ram', '--dry-multiplier', '--dry-base', '--dry-allowed-length', '--dry-penalty-last-n', '--batch-size', '--ubatch-size', '--cache-type-k', '--cache-type-v')
            print('实际启动参数:', json.dumps({k: args[args.index(k) + 1] for k in flags if k in args}, ensure_ascii=False))
            print('已移除的旧参数仍在命令行中:', [f for f in ('--api-key-file', '--no-context-shift', '--lora-scaled') if f in args] or '无')
    except (FileNotFoundError, PermissionError, IndexError):
        pass
print('GPU:', subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader'))
print('Host RAM:', subprocess.getoutput('free -h | head -n 2'))
"""

VERIFY_CODE = r"""# 全链路验收：接口 / 思考 / 速度 / MTP / 中文创作 / 工具调用
import json, time
from pathlib import Path
import requests

ROOT = Path('/content/qwen27b-lit')
MODEL_ALIAS = 'Qwen3.8-27B-Antislop-ZH-Q8_0'
base = 'http://127.0.0.1:8081'
headers = {'Authorization': 'Bearer ' + (ROOT / '.llama-api-key').read_text().strip()}

models = requests.get(base + '/v1/models', headers=headers, timeout=15).json()
print('模型：', [m['id'] for m in models['data']])
print('LoRA：', [(Path(a['path']).name, a['scale']) for a in requests.get(base + '/lora-adapters', headers=headers, timeout=15).json()])

def call(tag, messages, max_tokens=64, effort='low', tools=None, stream=False):
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
        print('CONTENT:', ''.join(text)[:200].replace('\n', ' | '))
        return None
    d = r.json()
    wall = time.monotonic() - t
    msg = d['choices'][0]['message']
    print('--- %s wall=%.2fs' % (tag, wall))
    print('REASONING:', (msg.get('reasoning_content') or '(none)')[:200].replace('\n', ' | '))
    print('CONTENT:', (msg.get('content') or '')[:300].replace('\n', ' | '))
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
call('creative', [{'role': 'user', 'content': '写一段 150 字左右的中文小说开头：雨夜、旧书店、一个不该出现的客人。'}], max_tokens=400)
call('stream', [{'role': 'user', 'content': '从 1 数到 20，用空格分隔。'}], max_tokens=80, stream=True)
tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': '查询城市天气',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
call('tools', [{'role': 'user', 'content': '亚特兰大现在天气如何？用 get_weather 工具查。'}], tools=tools)

log = (ROOT / 'server.log').read_text(errors='replace').splitlines()
spec = [l for l in log if any(w in l.lower() for w in ('draft', 'speculat', 'accepted', 'mtp'))]
print('MTP_LOG_LINES:', len(spec))
print('\n'.join(spec[-6:]))
"""

STOP_CODE = r"""# 长程任务防断连护航（前台保活 + 45分钟无请求自动停机省 CU）或立即点击按钮释放 GPU
import ipywidgets as widgets
import os, subprocess, threading, time
import requests
from google.colab import runtime
from IPython.display import display
from html import escape

IDLE_LIMIT_MIN = 45
stop_button = widgets.Button(description='🛑 立即结束 GPU 运行时', button_style='warning')
guard_button = widgets.Button(description='🛡️ 启动长程防断连护航 (45m空闲自停)', button_style='info')
stop_status = widgets.HTML(value='长程写作/改稿时可点蓝色按钮保持前台心跳防 Colab 90m 断连（连续 45 分钟无推理请求会自动释放 GPU 省 CU）；用完点黄色按钮立即释放。')
_guard_running = False

def stop_runtime(_=None):
    global _guard_running
    _guard_running = False
    stop_button.disabled = True
    guard_button.disabled = True
    stop_button.description = '已请求释放'
    stop_status.value = ('正在关闭服务并释放 GPU。此提示断开后不再刷新；'
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
                    stop_status.value = f'<b>已连续空闲 {idle_min:.0f} 分钟，触发自动停机释放 GPU 省 CU...</b>'
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

释放 GPU 后，右上角显示“重新连接”表示已断开；进入 **代码执行程序 → 管理会话**，显示 **没有正在进行的会话** 即表示已释放。

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
        {'cell_type': 'code', 'metadata': {'id': 'verify-lit'}, 'source': lines(VERIFY_CODE), 'execution_count': None, 'outputs': []},
        {'cell_type': 'code', 'metadata': {'id': 'stop-runtime'}, 'source': lines(STOP_CODE), 'execution_count': None, 'outputs': []},
        {'cell_type': 'markdown', 'metadata': {'id': 'stop-status-help'}, 'source': lines(STOP_MD)},
    ],
}

LIT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='')
print('Generated:', LIT_NB)

sync_requested(nb, LIT_DRIVE_ID)
