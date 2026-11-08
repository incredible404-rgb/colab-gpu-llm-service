"""Run the unified kernel's models through the official Colab CLI (google-colab-cli), no browser tab.

  python scripts/colab_cli.py start  <slug> [--gpu G4] [--idle-min 45] [--no-verify] [--keep-on-fail]
  python scripts/colab_cli.py launch <slug> [--gpu G4] [--idle-min 45]
  python scripts/colab_cli.py verify <slug>
  python scripts/colab_cli.py status
  python scripts/colab_cli.py stop   [<slug> | --all]

start reuses colab_kernel.build_notebook_dict, so the CLI path and the notebook path run the same
cell-1 code; the notebook's frontend-only tail (banner, WebUI popup, stop button) is cut off.
launch reads that same cell from notebooks/<model>.ipynb (it does not rebuild it), execs it in
the kernel with LAUNCH_DETACHED=1, and polls http://<tailnet>:443/health until 200.
The Tailscale service is raw TCP to llama-server, so HTTPS on that name never completes.

Liveness: google-colab-cli 0.7.5 does not start a keep-alive daemon; a CLI runtime is pruned
21-35 min after creation unless the kernel sees fresh executions. start/launch therefore start
colab_keepalive.py outside this shell: every 30 s it runs a one-line `colab exec` (the kernel
heartbeat that keeps the VM), every 60 s it also GETs /tun/m/<endpoint>/keep-alive/, and it runs
`colab stop` after --idle-min minutes without llama-server traffic. If this PC sleeps, the
heartbeat stops and Colab reclaims the VM by itself within ~20 min.

Why background processes die: a shell `&` or subprocess.Popen without start_new_session is in
the kernel's process group and is reaped when the cell is interrupted or that shell exits.
The notebook cell already starts llama-server and tailscaled with
Popen(..., start_new_session=True, stdout/stderr to a log file, stdin DEVNULL), which is the
setsid equivalent and survives `colab exec` returning. Do not wrap the cell in a second
`python -u` process: with DEBUG_VERIFY = False the cell's warmup / Tailscale wait is a
daemon thread, and that thread is killed if the interpreter exits. It stays alive only while
the kernel process stays alive.

`launch` sets LAUNCH_DETACHED=1 before the cell. colab_kernel.py honors it: immediately after
`qwen_server = proc`, if LAUNCH_DETACHED is 1 the cell prints the llama-server pid and sets
server_just_started, skipping the `while time.monotonic() - t0 < 600` local /health wait.
Otherwise that loop is unchanged (browser runs). Either way the cell falls through to the
production branch, which starts `_bg_finish` as a daemon thread and then ends. When detached,
that thread waits for local /health before CUDA warmup so a connection error cannot skip tmpfs
release or the Tailscale future. Do not SystemExit before the thread is started, and do not
change it to non-daemon: it lives only as long as the kernel process does. Tailscale's Popen
already uses start_new_session=True.

Colab Secrets and drive.mount() do not work in CLI sessions; the kernel downloads from pinned Hugging Face
revisions and reads HF_TOKEN from the environment first, so a local HF_TOKEN is forwarded when present.
The colab CLI must be authorized once (`colab sessions` prints the OAuth URL; token in
~/.config/colab-cli/token.json). Its DEBUG log ~/.config/colab-cli/colab.log records executed code,
which includes the same credentials the generated notebooks already carry.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_kernel import MODELS, build_notebook_dict  # noqa: E402
from secrets_config import FIELDS, load_config  # noqa: E402

CONSOLE_MARKER = '就绪横幅 + WebUI 标签页 + 停机按钮'
ENV = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
CLI_DIR = Path.home() / '.config/colab-cli'
CLI_PYTHONW = Path(os.environ.get('APPDATA', str(Path.home() / 'AppData/Roaming'))) / 'uv/tools/google-colab-cli/Scripts/pythonw.exe'
KEEPALIVE = Path(__file__).with_name('colab_keepalive.py')
# Not exported by subprocess on every Python build. Win32 CREATE_BREAKAWAY_FROM_JOB.
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
NOTEBOOKS = Path(__file__).resolve().parents[1] / 'notebooks'


def colab(*args, check=True, timeout=180):
    r = subprocess.run(['colab', *args], env=ENV, timeout=timeout, text=True, encoding='utf-8',
                       errors='replace', capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit('colab %s failed (%d):\n%s' % (' '.join(args), r.returncode, (r.stdout or '') + (r.stderr or '')))
    return (r.stdout or '') + (r.stderr or '')


def sessions() -> list[str]:
    # Lines look like "[name] <endpoint> | Hardware: ..."; "[?]" is a server assignment with no local name.
    return re.findall(r'^\[([^\]]+)\] \S+ \| Hardware', colab('sessions'), re.M)


def usage() -> dict:
    out = colab('usage')
    num = lambda pat: float(m.group(1)) if (m := re.search(pat, out)) else None  # noqa: E731
    return {'balance': num(r'balance:\s*([\d.]+)'), 'rate': num(r'rate:\s*([\d.]+)'),
            'active': int(num(r'assignments:\s*(\d+)') or 0)}


def exec_stream(session: str, code: str, timeout_s: int) -> int:
    """Run code in the session's kernel, echoing output live. The temp file holds credentials and is removed."""
    fd, path = tempfile.mkstemp(prefix='colab-cli-', suffix='.py')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(code)
        # colab exec gives up after 30 s by default, far short of a model download
        proc = subprocess.Popen(['colab', 'exec', '-s', session, '-f', path, '--timeout', str(timeout_s)], env=ENV, text=True,
                                encoding='utf-8', errors='replace', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout_s
        for line in proc.stdout:
            print(line, end='', flush=True)
            if time.monotonic() > deadline:
                proc.kill()
                print('\n[colab_cli] exec timed out after %ds' % timeout_s, flush=True)
                return 124
        return proc.wait()
    finally:
        Path(path).unlink(missing_ok=True)


def keepalive_state(session: str) -> dict | None:
    f = CLI_DIR / ('keepalive-%s.json' % session)
    try:
        return json.loads(f.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    out = subprocess.run(['tasklist', '/FI', 'PID eq %d' % pid, '/NH'], capture_output=True, text=True, errors='replace').stdout
    return str(pid) in out


def keepalive_task_name(session: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9_-]', '-', session)[:40]
    return 'ColabKeepalive-' + safe


def keepalive_argv(session: str, idle_min: int, spawn: str) -> list[str]:
    return [str(CLI_PYTHONW), str(KEEPALIVE), session, str(idle_min), spawn]


def _wait_keepalive(session: str, seconds: float):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        st = keepalive_state(session)
        if st and _pid_alive(int(st.get('pid') or 0)):
            return st
        time.sleep(0.25)
    return None


def _schtasks_spawn(session: str, idle_min: int):
    """One-shot Task Scheduler start. The process is not in the caller's job, so it survives shell exit."""
    task = keepalive_task_name(session)
    # schtasks parses /TR itself; embed quotes so paths with spaces survive. No shell.
    tr = subprocess.list2cmdline(keepalive_argv(session, idle_min, 'schtasks'))
    created = subprocess.run(
        ['schtasks', '/Create', '/F', '/TN', task, '/SC', 'ONCE', '/ST', '23:59', '/RL', 'LIMITED', '/TR', tr],
        capture_output=True, text=True, encoding='utf-8', errors='replace')
    if created.returncode != 0:
        raise SystemExit('schtasks /Create failed:\n%s' % ((created.stdout or '') + (created.stderr or '')))
    ran = subprocess.run(['schtasks', '/Run', '/TN', task], capture_output=True, text=True,
                         encoding='utf-8', errors='replace')
    if ran.returncode != 0:
        raise SystemExit('schtasks /Run failed:\n%s' % ((ran.stdout or '') + (ran.stderr or '')))


def start_keepalive(session: str, idle_min: int):
    st = keepalive_state(session)
    if st and _pid_alive(int(st.get('pid') or 0)):
        print('[colab_cli] keep-alive already running (pid %d)' % st['pid'], flush=True)
        return
    if st:
        (CLI_DIR / ('keepalive-%s.json' % session)).unlink(missing_ok=True)
    if not CLI_PYTHONW.exists():
        raise SystemExit('colab CLI interpreter not found at %s (install with: uv tool install google-colab-cli)' % CLI_PYTHONW)
    flags = (getattr(subprocess, 'DETACHED_PROCESS', 0)
             | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
             | CREATE_BREAKAWAY_FROM_JOB)
    spawn = 'breakaway'
    proc = None
    try:
        proc = subprocess.Popen(
            keepalive_argv(session, idle_min, 'breakaway'), env=ENV, creationflags=flags,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    except OSError as err:
        print('[colab_cli] breakaway spawn failed (%s); using schtasks' % err, flush=True)
        proc = None
    if proc is not None:
        time.sleep(1.0)
        if proc.poll() is not None:
            print('[colab_cli] breakaway process exited %s; using schtasks' % proc.returncode, flush=True)
            proc = None
    if proc is None:
        spawn = 'schtasks'
        _schtasks_spawn(session, idle_min)
    st = _wait_keepalive(session, 25)
    if not st:
        log_path = CLI_DIR / ('keepalive-%s.log' % session)
        tail = ''
        try:
            tail = log_path.read_text(encoding='utf-8', errors='replace')[-800:]
        except OSError:
            pass
        raise SystemExit('keep-alive daemon did not stay up (spawn %s). log:\n%s' % (spawn, tail or '(none)'))
    if spawn == 'schtasks':
        # /Run has already detached pythonw. The ONCE task would otherwise stay Ready and fire at 23:59.
        delete_keepalive_task(session)
    print('[colab_cli] keep-alive started via %s (pid %d, stop after %d idle min)' % (spawn, st['pid'], idle_min), flush=True)


def delete_keepalive_task(session: str):
    """Remove the one-shot Task Scheduler entry. Does not kill a process it already started."""
    subprocess.run(
        ['schtasks', '/Delete', '/F', '/TN', keepalive_task_name(session)],
        capture_output=True, text=True, encoding='utf-8', errors='replace')


def keepalive_sessions_in_schtasks_query(text: str) -> list[str]:
    """Session names from `schtasks /Query /FO CSV` lines that are ColabKeepalive-* tasks."""
    found = []
    for match in re.finditer(r'ColabKeepalive-([A-Za-z0-9_-]{1,40})', text or ''):
        name = match.group(1)
        if name not in found:
            found.append(name)
    return found


def keepalive_state_sessions() -> list[str]:
    names = []
    for path in CLI_DIR.glob('keepalive-*.json'):
        name = path.name[len('keepalive-'):-len('.json')]
        if name and name not in names:
            names.append(name)
    return names


def stop_targets(names: list[str], live_sessions: list[str], scheduled: list[str], state_sessions: list[str]) -> list[str]:
    """Who `stop` must clean. With no name or --all, include tasks whose Colab session is already gone."""
    stop_all = not names or names == ['--all']
    if not stop_all:
        return [s for s in names if s and s != '?']
    targets = []
    for session in list(live_sessions) + list(scheduled) + list(state_sessions):
        if session and session != '?' and session not in targets:
            targets.append(session)
    return targets


def _schtasks_query_text() -> str:
    queried = subprocess.run(
        ['schtasks', '/Query', '/FO', 'CSV'],
        capture_output=True, text=True, encoding='utf-8', errors='replace')
    return (queried.stdout or '') + '\n' + (queried.stderr or '')


def stop_keepalive(session: str):
    st = keepalive_state(session)
    if st and _pid_alive(int(st.get('pid') or 0)):
        subprocess.run(['taskkill', '/F', '/PID', str(st['pid'])], capture_output=True)
    delete_keepalive_task(session)
    (CLI_DIR / ('keepalive-%s.json' % session)).unlink(missing_ok=True)


def _config():
    for spec in MODELS.values():
        FIELDS.setdefault(spec['drive_key'], (spec['env_key'], spec['placeholder']))
    config, _, _, sources = load_config([])
    needed = ('api_key', 'ts_client_id', 'ts_client_secret', 'tailnet_fqdn')
    missing = [f for f in needed if sources.get(f) == 'placeholder']
    if missing:
        raise SystemExit('missing %s in .secrets.json / environment' % ', '.join(missing))
    return config


def _cells(slug: str, config: dict) -> dict:
    nb = build_notebook_dict(MODELS[slug], config, with_switch=False)
    return {c['metadata']['id']: ''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'}


def _hf_token() -> str:
    hf = os.environ.get('HF_TOKEN', '').strip()
    hf_file = Path.home() / '.cache/huggingface/token'
    if not hf and hf_file.exists():
        hf = hf_file.read_text(encoding='utf-8').strip()
    return hf


def _cut_frontend(cell: str, where: str) -> str:
    cut = cell.find(CONSOLE_MARKER)
    if cut < 0:
        raise SystemExit('cell-1 console marker not found in %s; colab_kernel.py changed, update CONSOLE_MARKER' % where)
    return cell[:cell.rfind('\n', 0, cut) + 1]


def headless_setup(slug: str, config: dict, idle_min: int) -> str:
    setup = _cut_frontend(_cells(slug, config)['persistent-setup'], 'build_notebook_dict')
    hf = _hf_token()
    prefix = 'import os\nos.environ["HF_TOKEN"] = %s\n' % json.dumps(hf) if hf else ''
    return prefix + setup


def notebook_first_cell(slug: str) -> str:
    """First code cell of notebooks/<model>.ipynb, frontend tail removed. Does not rebuild the kernel."""
    path = NOTEBOOKS / MODELS[slug]['notebook']
    if not path.exists():
        raise SystemExit('notebook not found: %s' % path)
    nb = json.loads(path.read_text(encoding='utf-8'))
    for cell in nb.get('cells') or []:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source') or ''
        text = ''.join(src) if isinstance(src, list) else src
        return _cut_frontend(text, path.name)
    raise SystemExit('no code cell in %s' % path.name)


def launch_exec_code(cell: str) -> str:
    """Cell source run inside the kernel. LAUNCH_DETACHED=1 makes the cell return once llama-server is spawned."""
    hf = _hf_token()
    prefix = 'import os\nos.environ["LAUNCH_DETACHED"] = "1"\n'
    if hf:
        prefix += 'os.environ["HF_TOKEN"] = %s\n' % json.dumps(hf)
    return prefix + cell


def _ensure_session(slug: str, gpu: str):
    if slug in sessions():
        print('[colab_cli] reusing session %s' % slug, flush=True)
        return
    shape = [] if gpu in ('G4', 'L4') else ['--high-mem']
    print('[colab_cli] colab new -s %s --gpu %s %s' % (slug, gpu, ' '.join(shape)), flush=True)
    print(colab('new', '-s', slug, '--gpu', gpu, *shape, timeout=900).strip(), flush=True)


def cmd_launch(slug: str, gpu: str, idle_min: int):
    """new --gpu, start the TFE daemon, exec the notebook's first cell in the kernel, poll /health."""
    config = _config()
    # svc:qwen publishes raw TCP :443 to llama-server. HTTPS on that name never returns 200.
    health = 'http://%s:443/health' % config['tailnet_fqdn']
    before = usage()
    info = {'balance_before': before['balance'], 'active_before': before['active']}
    cmd_launch.last = info
    print('[colab_cli] balance %.2f CU, %d active' % (before['balance'] or 0, before['active']), flush=True)
    t_assign = time.monotonic()
    info['assign_begin'] = time.time()
    print('[colab_cli] assign_begin epoch=%.3f wall=%s' % (info['assign_begin'], time.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
    _ensure_session(slug, gpu)
    assign_s = time.monotonic() - t_assign
    info['assign_s'] = assign_s
    info['assign_done'] = time.time()
    print('[colab_cli] assign_done %.1fs epoch=%.3f wall=%s' % (assign_s, info['assign_done'], time.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
    t0 = time.monotonic()
    start_keepalive(slug, idle_min)
    t_exec = time.monotonic()
    rc = exec_stream(slug, launch_exec_code(notebook_first_cell(slug)), timeout_s=3600)
    exec_s = time.monotonic() - t_exec
    info['exec_s'] = exec_s
    print('[colab_cli] exec_returned %.1fs rc=%s wall=%s' % (exec_s, rc, time.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
    if rc != 0:
        print('[colab_cli] launch exec failed rc=%s' % rc, flush=True)
        return 1
    alive = 1
    for _ in range(3):
        alive = exec_stream(slug, 'print("KERNEL_ALIVE", __import__("os").getpid())\n', timeout_s=90)
        if alive == 0:
            break
        time.sleep(2)
    print('[colab_cli] kernel_after_exec %s' % ('alive' if alive == 0 else 'dead rc=%s' % alive), flush=True)
    if alive != 0:
        return 1
    last = None
    while time.monotonic() - t0 < 3600:
        try:
            response = requests.get(health, timeout=10)
            if response.status_code == 200:
                info['exec_start_to_health'] = time.monotonic() - t0
                info['assign_to_health'] = time.monotonic() - t_assign
                info['ready_epoch'] = time.time()
                print('[colab_cli] ready %s exec_start_to_health=%.0fs assign_to_health=%.0fs wall=%s' % (
                    health, info['exec_start_to_health'], info['assign_to_health'], time.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
                return 0
            last = 'HTTP %s' % response.status_code
        except Exception as err:
            last = str(err)
        time.sleep(5)
    print('[colab_cli] launch health timeout: %s' % last, flush=True)
    return 1


def remote_check(config: dict, slug: str) -> bool:
    base = 'http://%s:443' % config['tailnet_fqdn']
    hdr = {'Authorization': 'Bearer ' + config['api_key']}
    last = None
    for _ in range(30):
        try:
            ids = [m['id'] for m in requests.get(base + '/v1/models', headers=hdr, timeout=10).json()['data']]
            print('[colab_cli] Tailscale %s/v1/models -> %s' % (base, ids), flush=True)
            return MODELS[slug]['alias'] in ids
        except Exception as err:
            last = err
            time.sleep(5)
    print('[colab_cli] Tailscale check failed: %s' % last, flush=True)
    return False


def cmd_start(slug: str, gpu: str, idle_min: int, verify: bool, keep_on_fail: bool):
    config = _config()
    code = headless_setup(slug, config, idle_min)
    before = usage()
    print('[colab_cli] balance %.2f CU, %d active' % (before['balance'] or 0, before['active']), flush=True)
    if slug not in sessions():
        # G4 has a single 176 GiB shape and the backend rejects --high-mem for it; other GPUs need it for big models.
        shape = [] if gpu in ('G4', 'L4') else ['--high-mem']
        print('[colab_cli] colab new -s %s --gpu %s %s' % (slug, gpu, ' '.join(shape)), flush=True)
        print(colab('new', '-s', slug, '--gpu', gpu, *shape, timeout=900).strip(), flush=True)
    start_keepalive(slug, idle_min)  # before the long download: without the kernel heartbeat the VM is reclaimed in ~20 min
    t0 = time.monotonic()
    rc = exec_stream(slug, code, timeout_s=3600)
    ok = rc == 0 and remote_check(config, slug)
    if ok and verify:
        ok = exec_stream(slug, _cells(slug, config)['verify-' + slug], timeout_s=1800) == 0
    after = usage()
    print('[colab_cli] start %s in %.0fs | balance %.2f CU | rate %.2f CU/h' % (
        'OK' if ok else 'FAILED', time.monotonic() - t0, after['balance'] or 0, after['rate'] or 0), flush=True)
    if not ok and not keep_on_fail:
        print('[colab_cli] releasing the runtime so a failed start does not keep burning CU', flush=True)
        cmd_stop([slug])
    return 0 if ok else 1


def cmd_verify(slug: str):
    return exec_stream(slug, _cells(slug, _config())['verify-' + slug], timeout_s=1800)


def cmd_status():
    u = usage()
    print('Active assignments: %d | Burn rate: %.2f CU/h | Balance: %.2f CU' % (u['active'], u['rate'] or 0, u['balance'] or 0))
    for s in sessions():
        if s == '?':
            print('--- session without a local name (stop it with: stop --all)')
            continue
        st = keepalive_state(s)
        if st and _pid_alive(st['pid']):
            print('--- session %s | keep-alive pid %d, %d pings, idle %.1f/%.0f min, last %s' % (
                s, st['pid'], st['pings'], st['idle_min'], st['limit_min'], st['time']))
        else:
            print('--- session %s | NO keep-alive running: it will be reclaimed within ~20 min '
                  '(restart with: start %s, or stop it)' % (s, s))


def cmd_stop(names: list[str]):
    stop_all = not names or names == ['--all']
    scheduled = keepalive_sessions_in_schtasks_query(_schtasks_query_text()) if stop_all else []
    state_sessions = keepalive_state_sessions() if stop_all else []
    targets = stop_targets(names, sessions() if stop_all else [], scheduled, state_sessions)
    for s in targets:
        if s != '?':
            stop_keepalive(s)
            print(colab('stop', '-s', s, check=False).strip())
    for _ in range(5):
        u = usage()
        if u['active'] == 0 and not u['rate']:
            break
        time.sleep(3)
    if u['active'] and (stop_all or '?' in sessions()):
        # Assignments the CLI has no local name for ("[?]") cannot be stopped by name; kill them via /tun/m in Edge.
        print('[colab_cli] %d assignment(s) left, releasing via colab_sync --release-all' % u['active'], flush=True)
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('colab_sync.py')), '--release-all'], env=ENV)
        u = usage()
    print('Active assignments: %d | Burn rate: %.2f CU/h | Balance: %.2f CU' % (u['active'], u['rate'] or 0, u['balance'] or 0))
    return 0 if u['active'] == 0 else 1


def main(argv: list[str]):
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    opt = lambda flag, default: rest[rest.index(flag) + 1] if flag in rest else default  # noqa: E731
    pos = [a for i, a in enumerate(rest) if not a.startswith('-') and (i == 0 or rest[i - 1] not in ('--gpu', '--idle-min'))]
    if cmd in ('start', 'verify', 'launch'):
        if not pos or pos[0] not in MODELS:
            raise SystemExit('slug required, one of: ' + ', '.join(MODELS))
        if cmd == 'verify':
            return cmd_verify(pos[0])
        if cmd == 'launch':
            return cmd_launch(pos[0], opt('--gpu', 'G4'), int(opt('--idle-min', 45)))
        return cmd_start(pos[0], opt('--gpu', 'G4'), int(opt('--idle-min', 45)),
                         '--no-verify' not in rest, '--keep-on-fail' in rest)
    if cmd == 'status':
        return cmd_status()
    if cmd == 'stop':
        return cmd_stop(pos or (['--all'] if '--all' in rest else []))
    raise SystemExit('unknown command %s' % cmd)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
