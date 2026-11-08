"""One-shot boot for the Strata trial notebook.

The setup cell blocks inside `strata launch` for the whole download. A blocked
kernel cannot run the 30s heartbeat, and Colab then reclaims the VM. This
script runs preflight, install and --plan in one cell, then spawns the launch
detached and returns. Tailscale is a later short cell, after /health is up.

Not part of colab_kernel. Session name: strata-fn. Idle limit is long because
Strata does not serve llama.cpp /slots; the stock 45-minute guard would stop
a download that never answers /slots.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_cli import colab, exec_stream, sessions, start_keepalive, usage  # noqa: E402

SESSION = 'strata-fn'
IDLE_MIN = 240
ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks' / 'qwen3.8-flash-next-strata-g4-colab.ipynb'
LAUNCH_MARK = "say('--- launch %s ---' % chosen)\n"
TAIL_MARK = "TS = '/content/tsbin/tailscale'\n"

DETACH = r'''
say('--- launch %s (detached) ---' % chosen)
logf = open('/content/strata-launch.log', 'a', encoding='utf-8', newline='')
proc = subprocess.Popen(
    launch_argv(chosen, False),
    stdout=logf, stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL, start_new_session=True)
Path('/content/strata-launch.pid').write_text(str(proc.pid), encoding='utf-8')
say('LAUNCH_PID %s model %s' % (proc.pid, chosen))
'''


def _setup_cell() -> str:
    nb = json.loads(NB.read_text(encoding='utf-8'))
    cell = ''.join(next(c['source'] for c in nb['cells'] if c['metadata']['id'] == 'strata-setup'))
    at = cell.find(LAUNCH_MARK)
    if at < 0:
        raise SystemExit('launch marker missing from the Strata notebook cell')
    return cell[:at] + DETACH


def _tailscale_cell() -> str:
    nb = json.loads(NB.read_text(encoding='utf-8'))
    cell = ''.join(next(c['source'] for c in nb['cells'] if c['metadata']['id'] == 'strata-setup'))
    at = cell.find(TAIL_MARK)
    env_at = cell.find("os.environ['XDG_DATA_HOME']")
    if at < 0 or env_at < 0:
        raise SystemExit('tailscale marker missing')
    # Imports, secrets, say(), run(). The TS block below defines ts() itself.
    return cell[:env_at] + "\n" + cell[at:]


def boot() -> int:
    before = usage()
    print('[strata-boot] balance %.2f CU, %d active' % (before['balance'] or 0, before['active']), flush=True)
    if SESSION not in sessions():
        print('[strata-boot] colab new -s %s --gpu G4' % SESSION, flush=True)
        print(colab('new', '-s', SESSION, '--gpu', 'G4', timeout=900).strip(), flush=True)
    start_keepalive(SESSION, IDLE_MIN)
    rc = exec_stream(SESSION, _setup_cell(), timeout_s=2400)
    print('[strata-boot] setup cell exit %s' % rc, flush=True)
    return rc


def health() -> int:
    code = r'''
import requests
try:
    h = requests.get('http://127.0.0.1:8081/health', timeout=3)
    print('HEALTH', h.status_code)
except Exception as e:
    print('HEALTH_DOWN', type(e).__name__)
try:
    tail = open('/content/strata-launch.log', encoding='utf-8', errors='replace').read()[-1500:]
except Exception as e:
    tail = 'NOLOG %s' % e
print('LOG_TAIL_START')
print(tail)
print('LOG_TAIL_END')
'''
    return exec_stream(SESSION, code, timeout_s=90)


def tailscale() -> int:
    return exec_stream(SESSION, _tailscale_cell(), timeout_s=180)


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'boot'
    if cmd == 'boot':
        return boot()
    if cmd == 'health':
        return health()
    if cmd == 'tailscale':
        return tailscale()
    raise SystemExit('usage: boot_strata_trial.py boot|health|tailscale')


if __name__ == '__main__':
    sys.exit(main() or 0)
