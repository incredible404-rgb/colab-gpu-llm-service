"""Experimental Colab notebook: Strata + Qwen3.8-Flash-Next Unsloth UD-IQ4_XS-v31.

Does not import or modify colab_kernel.py. The llama.cpp 3.8fn notebooks stay as they are.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from secrets_config import ROOT_DIR, guard_sync, load_config, report

DRIVE_NB_DIR = Path(os.environ.get('COLAB_NOTEBOOKS_DIR') or 'G:/My Drive/Colab Notebooks')
NOTEBOOK = 'qwen3.8-flash-next-strata-g4-colab.ipynb'

OVERVIEW = """# Strata · Qwen3.8-Flash-Next UD-IQ4_XS-v31（实验，未在本机实测）

跟现役 llama.cpp 笔记本分开。这本不加载 Swift-1.5 Heretic，也不改 `colab_kernel.py`。DeepSeek 那本先不做。

- **引擎**：Strata Terminal（安装脚本取 GitHub latest）。权重是 Unsloth `UD-IQ4_XS-v31`，不是现在那份 Heretic IQ4_XS-plus。
- **同卡社区数**（RTX PRO 6000 Blackwell 96GB，253GB 内存，Windows，Strata 0.2.11）：DFlash2、`--spec 6` 时聊天解码大约 270 tok/s，16K 预填大约 6700 tok/s。本机是 G4 96GB、约 176GB 内存，专家若整包在显存里，解码有机会接近；预填和冷启动要以这本跑出来的数为准。
- **质量**：Strata 文档写 UD-IQ4_XS 跟 Q8_0 的首 token 一致率 88.9%，困惑度 4.120 对 4.052。不是原版满血。
- **一次只答一路**。DFlash2 在多槽位上不可用；本本 `--parallel 1`。不要跟 llama.cpp 那本同时开，两边都占这一张 G4，而且都挂 `svc:qwen`。
- **磁盘**：首格要求 `/content` 至少 150 GiB 空闲（下载约 94GB，还要引擎、草稿头和打包余量）。不够就停，不会先下文件。驱动低于 580 也停（CUDA 13 引擎）。
- **入口**：`http://__FQDN_BARE__:443/` ，接口 `/v1`。模型 id 以首格打印的 `/v1/models` 为准，还没注册进 Magpie。
- **重跑**：中断后直接再跑首格，Strata 会接着未下完的文件。下载和准备可能要几十分钟，单元格会一直转。

跑完用最后一格释放 G4。空闲也在扣 CU（大约 9–10 CU/小时）。用 Colab 网页打开这本再跑。不要用 `colab_cli.py` 起：那条路要每 30 秒一次内核心跳才不会在 21–35 分钟后被回收，这本没接。
"""

SETUP = r'''# Strata 一键：检查卡和磁盘 → 安装 Terminal → --plan → 启动 → Tailscale svc:qwen
import json, os, shutil, subprocess, time
from pathlib import Path
import requests

PORT = 8081
KEY = __KEY__
FQDN = __FQDN__
TS_CLIENT_ID = __TS_ID__
TS_CLIENT_SECRET = __TS_SECRET__
DATA = Path('/content/strata-qwen')
STATE = Path('/content/strata-xdg')
MIN_FREE = 150 * (1024 ** 3)
MIN_VRAM_MIB = 90000
CANDIDATES = ('UD-IQ4_XS-v31', 'UD-IQ4_XS')

def say(msg):
    print(msg, flush=True)

def run(argv, check=True, timeout=None):
    say('+ ' + ' '.join(argv[:6]) + (' ...' if len(argv) > 6 else ''))
    result = subprocess.run(argv, timeout=timeout)
    if check and result.returncode != 0:
        raise SystemExit('命令失败，退出码 %s' % result.returncode)
    return result.returncode

os.environ['XDG_DATA_HOME'] = str(STATE)
os.environ['STRATA_DATA_DIR'] = str(STATE / 'strata')
os.environ['STRATA_HOME'] = '/content/strata-home'
STATE.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
Path(os.environ['STRATA_HOME']).mkdir(parents=True, exist_ok=True)

line = subprocess.getoutput(
    'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits'
).strip().splitlines()[0]
rest, driver = line.rsplit(',', 1)
name, mem_s = rest.rsplit(',', 1)
name, driver, mem = name.strip(), driver.strip(), int(float(mem_s.strip()))
say('GPU: %s | %s MiB | driver %s' % (name, mem, driver))
if '6000' not in name or mem < MIN_VRAM_MIB:
    raise SystemExit('不是 G4（需要 RTX PRO 6000 且显存 ≥90GB）。已停止，没有下载。')
major = int(driver.split('.')[0])
if major < 580:
    raise SystemExit('驱动 %s 低于 580，Strata CUDA 13 引擎不会加载。已停止，没有下载。' % driver)

free = shutil.disk_usage('/content').free
say('/content 剩余 %.1f GiB（门槛 150）' % (free / 1024 ** 3))
if free < MIN_FREE:
    raise SystemExit('/content 不够 150 GiB。UD-IQ4_XS 下载约 94GB，还要引擎和打包余量。已停止，没有下载。')
meminfo = Path('/proc/meminfo').read_text(encoding='utf-8', errors='replace')
avail = next((int(l.split()[1]) for l in meminfo.splitlines() if l.startswith('MemAvailable:')), 0)
say('MemAvailable %.1f GiB' % (avail / 1024 ** 2))

strata = Path.home() / '.local/bin/strata'
if not strata.exists():
    installer = Path('/tmp/strata-install.sh')
    run(['curl', '-fsSL', '-o', str(installer),
         'https://github.com/Niko1221/Strata/releases/latest/download/install.sh'])
    run(['sh', str(installer)], timeout=1800)
if not strata.exists():
    raise SystemExit('安装结束但没有 %s' % strata)
run([str(strata), '--version'], check=False)

help_txt = subprocess.run([str(strata), 'launch', '--help'], capture_output=True, text=True).stdout

def launch_argv(model, plan):
    argv = [str(strata), 'launch', '--family', 'unsloth', '--model', model,
            '--context', '256k', '--parallel', '1', '--data-dir', str(DATA),
            '--port', str(PORT), '--api-key', KEY, '--yes', '--no-tui']
    if plan:
        argv.append('--plan')
    if '--access' in help_txt:
        argv += ['--access', 'local']
    if '--vision' in help_txt:
        argv += ['--vision', 'on']
    if '--kv' in help_txt:
        argv += ['--kv', 'int8']
    if '--drafting' in help_txt:
        argv += ['--drafting', 'dflash2']
    if '--set' in help_txt:
        argv += ['--set', 'spec=6', '--set', 'spec-min-p=0.7']
    if (not plan) and ('--save-preset' in help_txt):
        argv += ['--save-preset', 'colab-g4-ud-iq4-xs']
    return argv

chosen = None
for model in CANDIDATES:
    say('--- plan %s ---' % model)
    code = run(launch_argv(model, True), check=False)
    if code == 0:
        chosen = model
        break
    say('plan 没过：%s' % model)
if chosen is None:
    raise SystemExit('UD-IQ4_XS-v31 和 UD-IQ4_XS 的 plan 都失败了。没有启动服务。')

say('--- launch %s ---' % chosen)
run(launch_argv(chosen, False))

base = 'http://127.0.0.1:%d' % PORT
headers = {'Authorization': 'Bearer ' + KEY}
ok = False
for _ in range(60):
    try:
        if requests.get(base + '/health', headers=headers, timeout=3).ok:
            ok = True
            break
        if requests.get(base + '/v1/models', headers=headers, timeout=3).ok:
            ok = True
            break
    except Exception:
        pass
    time.sleep(2)
if not ok:
    raise SystemExit('strata launch 返回了，但 /health 和 /v1/models 都没起来。退出码 0 只表示命令结束，不表示模型在服务。看上面的日志。')

models = requests.get(base + '/v1/models', headers=headers, timeout=30).json()
say('MODELS: ' + json.dumps(models, ensure_ascii=False)[:800])

TS = '/content/tsbin/tailscale'
TS_SOCK = '/var/run/tailscale/tailscaled.sock'
TS_STATE = '/content/tailscaled-strata.state'
TS_DIR = '/content/tsstatedir-strata'

def ts(*args, timeout=60):
    return subprocess.run([TS, '--socket=' + TS_SOCK, *args], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout)

def ts_state():
    try:
        return json.loads(ts('status', '--json').stdout).get('BackendState', '')
    except Exception:
        return ''

if not Path(TS).exists():
    run(['bash', '-lc',
         'curl -fsSL https://pkgs.tailscale.com/stable/tailscale_1.104.1_amd64.tgz | tar -xz -C /content && mv /content/tailscale_1.104.1_amd64 /content/tsbin'])
Path(TS_DIR).mkdir(parents=True, exist_ok=True)
Path('/var/run/tailscale').mkdir(parents=True, exist_ok=True)
if ts_state() in ('', 'NoState', 'Stopped'):
    subprocess.Popen(
        ['/content/tsbin/tailscaled', '--tun=userspace-networking', '--state=' + TS_STATE,
         '--statedir=' + TS_DIR, '--socket=' + TS_SOCK],
        stdout=open('/content/tailscaled-strata.log', 'a', encoding='utf-8', newline=''),
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(40):
        if Path(TS_SOCK).exists() and ts_state() in ('Running', 'NeedsLogin'):
            break
        time.sleep(0.5)
if ts_state() != 'Running':
    token = requests.post(
        'https://api.tailscale.com/api/v2/oauth/token',
        data={'client_id': TS_CLIENT_ID, 'client_secret': TS_CLIENT_SECRET}, timeout=30,
    ).json()['access_token']
    key = requests.post(
        'https://api.tailscale.com/api/v2/tailnet/-/keys',
        headers={'Authorization': 'Bearer ' + token},
        json={'capabilities': {'devices': {'create': {
            'reusable': False, 'ephemeral': True, 'preauthorized': True, 'tags': ['tag:colab']}}}},
        timeout=30,
    ).json()['key']
    up = ts('up', '--auth-key=' + key, '--hostname=qwen-strata-colab',
            '--advertise-tags=tag:colab', '--accept-dns=false', '--timeout=60s', timeout=120)
    if up.returncode != 0:
        raise SystemExit('tailscale up 失败：' + (up.stderr or up.stdout or '')[:400])
    for _ in range(120):
        if ts_state() == 'Running':
            break
        time.sleep(0.5)
ts('serve', 'reset')
ts('serve', '--bg', '--service=svc:qwen', '--tcp=443', 'tcp://127.0.0.1:%d' % PORT, timeout=90)
ts('serve', '--bg', '--https=443', 'http://127.0.0.1:%d' % PORT, timeout=90)
say('网页： http://%s:443/' % FQDN)
say('接口： http://%s:443/v1' % FQDN)
say('本机： %s/v1  （钥匙与现役笔记本相同）' % base)
'''

PROBE = r'''# 短探针：冷一次、再热一次。数字是墙钟，不是社区那份 270 tok/s 协议。
import json, time
import requests

PORT = 8081
KEY = __KEY__
base = 'http://127.0.0.1:%d' % PORT
headers = {'Authorization': 'Bearer ' + KEY}
models = requests.get(base + '/v1/models', headers=headers, timeout=30).json()['data']
model = models[0]['id']
print('model:', model)

def once(label):
    t0 = time.perf_counter()
    resp = requests.post(
        base + '/v1/chat/completions', headers=headers, timeout=300,
        json={'model': model, 'messages': [{'role': 'user', 'content': '用一句话回答：1+1 等于几？'}],
              'max_tokens': 64, 'temperature': 0, 'reasoning_effort': 'low'})
    dt = time.perf_counter() - t0
    resp.raise_for_status()
    body = resp.json()
    usage = body.get('usage') or {}
    out = int(usage.get('completion_tokens') or 0)
    print(label, 'wall %.2fs' % dt, 'completion_tokens', out,
          'tok/s', ('%.1f' % (out / dt)) if out and dt else 'n/a')
    msg = (body.get('choices') or [{}])[0].get('message') or {}
    print('content:', str(msg.get('content') or '')[:160].replace('\n', ' '))
    timings = body.get('timings')
    if timings:
        print('timings:', json.dumps(timings, ensure_ascii=False)[:400])

once('cold')
once('warm')
'''

STOP = r'''# 停掉 Strata 并释放 G4。断开后这格不再刷新；到「代码执行程序 → 管理会话」确认没有活动会话。
import os, subprocess
from pathlib import Path
from google.colab import runtime

strata = Path.home() / '.local/bin/strata'
env = os.environ.copy()
env['XDG_DATA_HOME'] = '/content/strata-xdg'
env['STRATA_DATA_DIR'] = '/content/strata-xdg/strata'
env['STRATA_HOME'] = '/content/strata-home'
if strata.exists():
    subprocess.run([str(strata), 'launch', '--stop'], env=env, timeout=180)
ts = '/content/tsbin/tailscale'
if os.path.exists(ts):
    subprocess.run([ts, '--socket=/var/run/tailscale/tailscaled.sock', 'down'], timeout=15)
runtime.unassign()
'''


def _lines(text: str) -> list[str]:
    raw = text.strip('\n').split('\n')
    return [line + '\n' for line in raw]


def _fill(template: str, config: dict) -> str:
    return (template
            .replace('__KEY__', json.dumps(config['api_key']))
            .replace('__FQDN__', json.dumps(config['tailnet_fqdn']))
            .replace('__TS_ID__', json.dumps(config['ts_client_id']))
            .replace('__TS_SECRET__', json.dumps(config['ts_client_secret'])))


def build_notebook(config: dict) -> dict:
    overview = OVERVIEW.replace('__FQDN_BARE__', config['tailnet_fqdn'])
    cells = [
        ('overview', 'markdown', overview),
        ('strata-setup', 'code', _fill(SETUP, config)),
        ('strata-probe', 'code', _fill(PROBE, config)),
        ('stop-runtime', 'code', STOP),
    ]
    nb_cells = []
    for cid, kind, src in cells:
        if kind == 'code':
            ast.parse(src)
        cell = {'cell_type': kind, 'metadata': {'id': cid}, 'source': _lines(src)}
        if kind == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
        nb_cells.append(cell)
    return {
        'nbformat': 4,
        'nbformat_minor': 0,
        'metadata': {
            'colab': {'provenance': [], 'gpuType': 'G4', 'machine_shape': 'hm'},
            'accelerator': 'GPU',
            'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'cells': nb_cells,
    }


def main():
    argv = sys.argv[1:]
    config, sanitized, output_dir, sources = load_config(argv)
    report(config, sanitized, sources)
    guard_sync(sanitized, sources, config)
    if not sanitized:
        missing = [k for k in ('api_key', 'ts_client_id', 'ts_client_secret', 'tailnet_fqdn')
                   if '<YOUR_' in config[k]]
        if missing:
            raise SystemExit('refusing to write a runnable notebook; missing %s' % ', '.join(missing))
    nb = build_notebook(config)
    text = json.dumps(nb, ensure_ascii=False, indent=1) + '\n'
    path = output_dir / NOTEBOOK
    path.write_text(text, encoding='utf-8', newline='')
    print('Generated:', path)
    if (not sanitized and '--sync-colab' not in argv
            and output_dir == ROOT_DIR / 'notebooks' and DRIVE_NB_DIR.exists()):
        mirror = DRIVE_NB_DIR / NOTEBOOK
        mirror.write_text(text, encoding='utf-8', newline='')
        assert mirror.read_bytes() == path.read_bytes()
        print('Mirrored to Drive:', mirror)


if __name__ == '__main__':
    main()
