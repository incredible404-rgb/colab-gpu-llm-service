"""Run the unified kernel's models through the official Colab CLI (google-colab-cli), no browser tab.

  python scripts/colab_cli.py start     [<slug>] [--gpu G4] [--idle-min 45] [--no-verify] [--keep-on-fail] [--skip-preflight]
  python scripts/colab_cli.py launch    [<slug>] [--gpu G4] [--idle-min 45] [--skip-preflight]
  python scripts/colab_cli.py verify    [<slug>]
  python scripts/colab_cli.py yoga      [gui | wake | status | stop] [--gpu G4] [--idle-min 45]   (alias: easycolab)
  python scripts/colab_cli.py status
  python scripts/colab_cli.py stop      [<slug> | --all]

start reuses colab_kernel.build_notebook_dict, so the CLI path and the notebook path run the same
cell-1 code; the notebook's frontend-only tail (banner, WebUI popup, stop button) is cut off.
launch reads that same cell from notebooks/private/<model>.ipynb, else notebooks/<model>.ipynb (it does not rebuild it), execs it in
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
The colab CLI must be authorized once, by a person at a terminal: `colab sessions` prints the OAuth URL and
waits on input() for the pasted code (token in ~/.config/colab-cli/token.json). Every colab call here runs
with stdin closed, so an unauthorized CLI fails at once with a one-line "not authorized" message instead of
hanging. Its DEBUG log ~/.config/colab-cli/colab.log records executed code,
which includes the same credentials the generated notebooks already carry.

Run it as `python -X utf8 scripts/colab_cli.py ...`; main() also forces UTF-8 on stdout/stderr, so a console
code page never turns an echoed kernel line into a UnicodeEncodeError while a GPU is allocated.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_path import INSTALL_HINT, find_colab  # noqa: E402
from stdio_utf8 import force_utf8  # noqa: E402

# `status` and `stop` are the commands run most often and need none of these: requests alone is ~170 ms of
# import time, colab_kernel (+ secrets_config, colab_sync, sqlite3) another ~35 ms. Functions import what they
# use; `import colab_cli; colab_cli.ALL_MODELS` and friends still resolve through __getattr__.
_LAZY_ATTRS = {
    'requests': 'requests', 'ALL_MODELS': 'colab_kernel', 'MODELS': 'colab_kernel',
    'build_notebook_dict': 'colab_kernel', 'FIELDS': 'secrets_config', 'load_config': 'secrets_config',
}


def __getattr__(name):
    module = _LAZY_ATTRS.get(name)
    if module is None:
        raise AttributeError('module %r has no attribute %r' % (__name__, name))
    import importlib
    mod = importlib.import_module(module)
    return mod if name == module else getattr(mod, name)


CONSOLE_MARKER = '就绪横幅 + WebUI 标签页 + 停机按钮'
ENV = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
CLI_DIR = Path.home() / '.config/colab-cli'
CLI_PYTHONW = Path(os.environ.get('APPDATA', str(Path.home() / 'AppData/Roaming'))) / 'uv/tools/google-colab-cli/Scripts/pythonw.exe'
KEEPALIVE = Path(__file__).with_name('colab_keepalive.py')
# Not exported by subprocess on every Python build. Win32 CREATE_BREAKAWAY_FROM_JOB.
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
NOTEBOOKS = Path(__file__).resolve().parents[1] / 'notebooks'
PRIVATE_NOTEBOOKS = NOTEBOOKS / 'private'  # default output of a private build; git-ignored


_AUTH_MARKERS = ('Enter the authorization code', 'To authorize colab-cli')


def _as_text(value) -> str:
    return value.decode('utf-8', 'replace') if isinstance(value, bytes) else (value or '')


def unauthorized_hint(output: str) -> str | None:
    """One-line message when the CLI's output shows it is waiting for the first OAuth code, else None."""
    if not any(marker in output for marker in _AUTH_MARKERS):
        return None
    url = re.search(r'https://accounts\.google\.com/\S+', output)
    return ('colab CLI is not authorized: a person must run `colab sessions` once in their own terminal and paste '
            'the code Google shows (SKILL.md section 0, first authorization)' + ('. Link: ' + url.group(0) if url else ''))


def colab(*args, check=True, timeout=180):
    """Run the colab CLI. stdin is closed: an unauthorized CLI would otherwise wait on input() until the timeout."""
    try:
        r = subprocess.run([find_colab(), *args], env=ENV, timeout=timeout, text=True, encoding='utf-8',
                           errors='replace', capture_output=True, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise SystemExit('colab CLI not found (not on PATH, not in the uv tool bin directory). Install: ' + INSTALL_HINT)
    except subprocess.TimeoutExpired as err:
        hint = unauthorized_hint(_as_text(err.stdout) + _as_text(err.stderr))
        raise SystemExit(hint or 'colab %s timed out after %ds' % (' '.join(args), timeout))
    out = (r.stdout or '') + (r.stderr or '')
    if check and r.returncode != 0:
        raise SystemExit(unauthorized_hint(out) or 'colab %s failed (%d):\n%s' % (' '.join(args), r.returncode, out))
    return out


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
        proc = subprocess.Popen([find_colab(), 'exec', '-s', session, '-f', path, '--timeout', str(timeout_s)], env=ENV,
                                text=True, encoding='utf-8', errors='replace', stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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


def _win_pid_alive(pid: int) -> bool:
    """OpenProcess + exit code: the same answer as tasklist without spawning a process (~0.1-0.3 s each)."""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: it exists, we just may not look inside
    try:
        code = wintypes.DWORD()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == 'nt':
        try:
            return _win_pid_alive(pid)
        except Exception:  # no ctypes / odd Python build: fall back to the slow, always-available probe
            pass
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
        raise SystemExit('colab CLI interpreter not found at %s (install with: %s)' % (CLI_PYTHONW, INSTALL_HINT))
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
    from colab_kernel import ALL_MODELS
    from secrets_config import FIELDS, load_config
    for spec in ALL_MODELS.values():
        FIELDS.setdefault(spec['drive_key'], (spec['env_key'], spec['placeholder']))
    from secrets_config import missing_required
    config, _, _, sources = load_config([])
    # Runs before any `colab new`: a value copied from .secrets.example.json counts as missing, so a
    # half-filled file never gets as far as allocating a paid GPU.
    missing = missing_required(sources)
    if missing:
        raise SystemExit('missing %s in .secrets.json / environment (empty or still a <YOUR_...> placeholder from '
                         '.secrets.example.json); nothing was allocated' % ', '.join(missing))
    return config


def _tailscale_exe() -> str | None:
    found = shutil.which('tailscale')
    if found:
        return found
    for var in ('ProgramFiles', 'ProgramW6432'):
        base = os.environ.get(var)
        cand = Path(base) / 'Tailscale' / 'tailscale.exe' if base else None
        if cand and cand.is_file():
            return str(cand)
    return None


def _local_tailscale_problem() -> str | None:
    """Why this PC is not on the tailnet, or None. Without it /health and /slots can never be reached."""
    exe = _tailscale_exe()
    if not exe:
        return 'tailscale CLI not found on this PC (install Tailscale and sign in)'
    try:
        r = subprocess.run([exe, 'status', '--json'], capture_output=True, text=True, encoding='utf-8', errors='replace',
                           timeout=20, stdin=subprocess.DEVNULL)
        state = json.loads(r.stdout or '{}').get('BackendState')
    except (OSError, subprocess.SubprocessError, ValueError) as err:
        return 'could not read `tailscale status` (%s)' % type(err).__name__
    if state != 'Running':
        return 'Tailscale on this PC is %s, not Running (open Tailscale and sign in)' % (state or 'unknown')
    return None


def _tailscale_oauth_problem(config: dict) -> str | None:
    """Ask Tailscale for an OAuth token with the configured client, as cell 1 will. Never prints the credentials.

    Only a definite rejection counts. A network error from this PC proves nothing about the VM's route.
    """
    import requests
    try:
        r = requests.post('https://api.tailscale.com/api/v2/oauth/token', timeout=15,
                          data={'client_id': config['ts_client_id'], 'client_secret': config['ts_client_secret']})
    except requests.RequestException as err:
        print('[colab_cli] preflight: could not reach api.tailscale.com from this PC (%s); not treated as a failure'
              % type(err).__name__, flush=True)
        return None
    try:
        token = r.json().get('access_token')
    except ValueError:
        token = None
    if r.status_code == 200 and token:
        return None
    return ('Tailscale rejected the OAuth client (HTTP %d): check ts_client_id / ts_client_secret in .secrets.json '
            'and that the client may write Auth Keys for tag:colab' % r.status_code)


def preflight(config: dict, skip: bool = False) -> None:
    """Local checks that otherwise show up only after the G4 is billing. Raises SystemExit before anything is allocated."""
    if skip:
        print('[colab_cli] preflight skipped (--skip-preflight)', flush=True)
        return
    problems = []
    if not CLI_PYTHONW.exists():
        problems.append('colab CLI interpreter not found at %s (the keep-alive daemon runs on it); install: %s'
                        % (CLI_PYTHONW, INSTALL_HINT))
    for check in (_local_tailscale_problem, lambda: _tailscale_oauth_problem(config)):
        problem = check()
        if problem:
            problems.append(problem)
    if problems:
        raise SystemExit('preflight failed, nothing was allocated:\n  - %s\n(--skip-preflight overrides these checks)'
                         % '\n  - '.join(problems))
    print('[colab_cli] preflight ok (CLI interpreter, Tailscale online, Tailscale OAuth client)', flush=True)


def _cells(slug: str, config: dict) -> dict:
    from colab_kernel import ALL_MODELS, build_notebook_dict
    nb = build_notebook_dict(ALL_MODELS[slug], config, with_switch=False)
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


def notebook_path(slug: str) -> Path:
    """The private build in notebooks/private/ when there is one, else notebooks/ (older private builds and the
    committed sanitized notebooks live there). The path returned for a missing file is the private one."""
    from colab_kernel import ALL_MODELS
    name = ALL_MODELS[slug]['notebook']
    for folder in (PRIVATE_NOTEBOOKS, NOTEBOOKS):
        if (folder / name).exists():
            return folder / name
    return PRIVATE_NOTEBOOKS / name


def notebook_first_cell(slug: str) -> str:
    """First code cell of the model's notebook (see notebook_path), frontend tail removed. Does not rebuild the kernel."""
    path = notebook_path(slug)
    if not path.exists():
        raise SystemExit('notebook not found: %s; generate a private build first with: '
                         'python -X utf8 scripts/colab_kernel.py %s' % (path, slug))
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


def _reuse_active_session_as(slug: str, gpu: str, active: list[str]) -> bool:
    """If another active session (e.g. the Colab-yoga one) is already running on the same GPU, alias it to slug in sessions.json."""
    sess_file = CLI_DIR / 'sessions.json'
    if not sess_file.exists():
        return False
    try:
        items = json.loads(sess_file.read_text(encoding='utf-8'))
        if not isinstance(items, list):
            return False
        named = [s for s in active if s != '?']
        donor = next((it for it in items if isinstance(it, dict) and it.get('name') in named
                      and str(it.get('accelerator') or '').upper() == gpu.upper()), None)
        if not donor:
            return False
        alias_item = dict(donor)
        alias_item['name'] = slug
        items = [it for it in items if not (isinstance(it, dict) and it.get('name') == slug)] + [alias_item]
        sess_file.write_text(json.dumps(items, indent=2) + '\n', encoding='utf-8', newline='')
        print('[colab_cli] reusing active %s session [%s] as [%s]' % (gpu, donor.get('name'), slug), flush=True)
        return True
    except Exception:
        return False


def _ensure_session(slug: str, gpu: str) -> bool:
    """Make sure a session named slug exists. True only when this call allocated a new runtime (colab new)."""
    active = sessions()
    if slug in active:
        print('[colab_cli] reusing session %s' % slug, flush=True)
        return False
    if active and _reuse_active_session_as(slug, gpu, active):
        return False
    shape = [] if gpu in ('G4', 'L4') else ['--high-mem']
    print('[colab_cli] colab new -s %s --gpu %s %s' % (slug, gpu, ' '.join(shape)), flush=True)
    try:
        print(colab('new', '-s', slug, '--gpu', gpu, *shape, timeout=900).strip(), flush=True)
    except BaseException:
        # The backend may have assigned the VM before the CLI gave up (timeout, Ctrl-C); do not leave it burning.
        print('[colab_cli] colab new did not finish cleanly; releasing anything it allocated', flush=True)
        try:
            cmd_stop([slug])
        except BaseException as err:
            print('[colab_cli] cleanup after failed colab new also failed: %s' % err, flush=True)
        raise
    return True


def _keepalive_running(session: str) -> bool:
    st = keepalive_state(session)
    return bool(st and _pid_alive(int(st.get('pid') or 0)))


def _abandon(slug: str, state: dict, reason: str, keep: bool = False) -> None:
    """Failure path shared by start and launch; runs at most once per command.

    Only a runtime this command allocated (state['created']) is released. A reused session, such as the one
    Colab-yoga owns, keeps running: the failure is reported, and only a keep-alive daemon started here is
    taken down again so its idle guard cannot later stop somebody else's session.
    """
    if state.get('released') or keep or not state.get('ensured'):
        return  # nothing to undo before the session exists
    state['released'] = True
    if state.get('created'):
        print('[colab_cli] %s; releasing the runtime so a failed run does not keep burning CU' % reason, flush=True)
        cmd_stop([slug])
        return
    print('[colab_cli] %s; the reused session %s keeps running (stop it with: stop %s)' % (reason, slug, slug), flush=True)
    if state.get('keepalive_started'):
        stop_keepalive(slug)


def _begin_keepalive(slug: str, idle_min: int, state: dict) -> None:
    state['keepalive_started'] = not _keepalive_running(slug)
    start_keepalive(slug, idle_min)


def cmd_launch(slug: str, gpu: str, idle_min: int, skip_preflight: bool = False):
    """new --gpu, start the TFE daemon, exec the notebook's first cell in the kernel, poll /health.

    Every failure after this call allocated the runtime releases it, whether it comes back as a return code
    or as an exception including Ctrl-C (same rule as cmd_start). A session that was reused is left running.
    """
    state = {'created': False, 'ensured': False, 'released': False, 'keepalive_started': False}
    try:
        rc = _cmd_launch(slug, gpu, idle_min, state, skip_preflight)
    except BaseException as err:
        _abandon(slug, state, 'unexpected %s' % type(err).__name__)
        raise
    if rc:
        _abandon(slug, state, 'launch failed')
    return rc


def _cmd_launch(slug: str, gpu: str, idle_min: int, state: dict, skip_preflight: bool = False):
    import requests
    from colab_kernel import ALL_MODELS
    config = _config()
    cell_exec_code = launch_exec_code(notebook_first_cell(slug))
    from secrets_config import FIELDS, REQUIRED
    stale = [f for f in REQUIRED if FIELDS[f][1] in cell_exec_code]
    if stale:
        # Only the four required values matter; an unset optional field (file bridge IP, Drive ids) is fine.
        raise SystemExit(
            'notebook %s still holds sanitized <YOUR_...> placeholders (%s); '
            'generate a private build first with: python -X utf8 scripts/colab_kernel.py %s' % (
                ALL_MODELS[slug]['notebook'], ', '.join(stale), slug))
    preflight(config, skip_preflight)
    # The Tailscale Service publishes raw TCP :443 to llama-server. HTTPS on that name never returns 200.
    health = 'http://%s:443/health' % config['tailnet_fqdn']
    before = usage()
    info = {'balance_before': before['balance'], 'active_before': before['active']}
    cmd_launch.last = info
    print('[colab_cli] balance %.2f CU, %d active' % (before['balance'] or 0, before['active']), flush=True)
    t_assign = time.monotonic()
    info['assign_begin'] = time.time()
    print('[colab_cli] assign_begin epoch=%.3f wall=%s' % (info['assign_begin'], time.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
    state['created'] = _ensure_session(slug, gpu)
    state['ensured'] = True
    assign_s = time.monotonic() - t_assign
    info['assign_s'] = assign_s
    info['assign_done'] = time.time()
    print('[colab_cli] assign_done %.1fs epoch=%.3f wall=%s' % (assign_s, info['assign_done'], time.strftime('%Y-%m-%d %H:%M:%S')), flush=True)
    t0 = time.monotonic()
    try:
        _begin_keepalive(slug, idle_min, state)
    except SystemExit as err:
        print('[colab_cli] %s' % err, flush=True)
        _abandon(slug, state, 'keep-alive failed')
        raise
    t_exec = time.monotonic()
    rc = exec_stream(slug, cell_exec_code, timeout_s=3600)
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
    import requests
    from colab_kernel import ALL_MODELS
    base = 'http://%s:443' % config['tailnet_fqdn']
    hdr = {'Authorization': 'Bearer ' + config['api_key']}
    last = None
    for _ in range(30):
        try:
            ids = [m['id'] for m in requests.get(base + '/v1/models', headers=hdr, timeout=10).json()['data']]
            print('[colab_cli] Tailscale %s/v1/models -> %s' % (base, ids), flush=True)
            return ALL_MODELS[slug]['alias'] in ids
        except Exception as err:
            last = err
            time.sleep(5)
    print('[colab_cli] Tailscale check failed: %s' % last, flush=True)
    return False


def cmd_start(slug: str, gpu: str, idle_min: int, verify: bool, keep_on_fail: bool, skip_preflight: bool = False):
    config = _config()
    code = headless_setup(slug, config, idle_min)
    preflight(config, skip_preflight)
    before = usage()
    print('[colab_cli] balance %.2f CU, %d active' % (before['balance'] or 0, before['active']), flush=True)
    state = {'created': False, 'ensured': False, 'released': False, 'keepalive_started': False}
    state['created'] = _ensure_session(slug, gpu)
    state['ensured'] = True

    def release(reason: str):
        _abandon(slug, state, reason, keep=keep_on_fail)

    try:
        try:
            _begin_keepalive(slug, idle_min, state)  # before the long download: without the kernel heartbeat the VM is reclaimed in ~20 min
        except SystemExit as err:
            print('[colab_cli] %s' % err, flush=True)
            release('keep-alive failed')
            raise
        t0 = time.monotonic()
        rc = exec_stream(slug, code, timeout_s=3600)
        ok = rc == 0 and remote_check(config, slug)
        if ok and verify:
            ok = exec_stream(slug, _cells(slug, config)['verify-' + slug], timeout_s=1800) == 0
        after = usage()
        print('[colab_cli] start %s in %.0fs | balance %.2f CU | rate %.2f CU/h' % (
            'OK' if ok else 'FAILED', time.monotonic() - t0, after['balance'] or 0, after['rate'] or 0), flush=True)
        if not ok:
            release('start failed')
        return 0 if ok else 1
    except BaseException as err:
        # Anything unexpected after the runtime exists (UnicodeEncodeError on a console, a dead network,
        # Ctrl-C) must not strand a billed G4. Only a runtime this call allocated is released; a reused one
        # (for example Colab-yoga's) keeps running and is reported instead.
        release('unexpected %s' % type(err).__name__)
        raise


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


def _in_background(fn):
    """Start fn on a thread now; the returned callable joins it and returns the result or re-raises its error."""
    box = {}

    def run():
        try:
            box['value'] = fn()
        except BaseException as err:
            box['error'] = err

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    def wait():
        worker.join()
        if 'error' in box:
            raise box['error']
        return box.get('value')

    return wait


def _release_leftovers(u: dict) -> dict:
    """Free assignments `colab stop` cannot name, cheapest route first; returns the fresh usage().

    1. Colab-yoga bridge `stop_all` (when installed): official unassign per endpoint, no browser.
    2. colab_sync --release-all: Colab's /tun/m endpoints through a signed-in Edge tab (needs a browser driver,
       see "Browser fallback" in SKILL.md).
    Anything still active afterwards is for the person: Colab web page > Runtime > Manage sessions.
    """
    if YOGA_PY.exists() and YOGA_BRIDGE.exists():
        print('[colab_cli] %d assignment(s) left, releasing via the Colab-yoga bridge (stop_all)' % u['active'], flush=True)
        try:
            subprocess.run([str(YOGA_PY), '-u', str(YOGA_BRIDGE), 'stop_all'], env=ENV, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as err:
            print('[colab_cli] Colab-yoga stop_all failed: %s' % err, flush=True)
        u = usage()
    if u['active']:
        print('[colab_cli] %d assignment(s) left, trying colab_sync --release-all (needs a signed-in Edge tab)' % u['active'], flush=True)
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name('colab_sync.py')), '--release-all'], env=ENV)
        u = usage()
    if u['active']:
        print('[colab_cli] still %d active: open https://colab.research.google.com > Runtime > Manage sessions and '
              'terminate them there' % u['active'], flush=True)
    return u


def cmd_stop(names: list[str]):
    stop_all = not names or names == ['--all']
    live, scheduled, state_sessions = [], [], []
    if stop_all:
        # `schtasks /Query` (slow) and `colab sessions` do not depend on each other. Only a non-colab command
        # overlaps: two colab processes at once could race on token.json, which the CLI rewrites in place.
        scheduled_text = _in_background(_schtasks_query_text)
        live = sessions()
        scheduled = keepalive_sessions_in_schtasks_query(scheduled_text())
        state_sessions = keepalive_state_sessions()
    named = [s for s in stop_targets(names, live, scheduled, state_sessions) if s != '?']
    # Every daemon goes first (taskkill + schtasks /Delete each, in parallel), then the sessions are stopped.
    for wait in [_in_background(lambda s=s: stop_keepalive(s)) for s in named]:
        wait()
    for s in named:
        print(colab('stop', '-s', s, check=False).strip())
    for _ in range(5):
        u = usage()
        if u['active'] == 0 and not u['rate']:
            break
        time.sleep(3)
    if u['active'] and stop_all:
        # Assignments the CLI has no local name for ("[?]") cannot be stopped by name.
        u = _release_leftovers(u)
    elif u['active']:
        # Stopping one name never escalates: _release_leftovers frees the whole account, Colab-yoga's session
        # included. Say what is left and how to clear it on purpose.
        print('[colab_cli] %d assignment(s) still active after stopping %s (another session, or an unnamed [?] ghost). '
              '`stop --all` releases every assignment on the account, so run it only when none of them is wanted.'
              % (u['active'], ', '.join(named) or 'nothing'), flush=True)
    print('Active assignments: %d | Burn rate: %.2f CU/h | Balance: %.2f CU' % (u['active'], u['rate'] or 0, u['balance'] or 0))
    return 0 if u['active'] == 0 else 1


_PROGRAMS = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData/Local'))) / 'Programs'
# Colab-yoga (https://github.com/incredible404-rgb/Colab-yoga) installs to %LOCALAPPDATA%/Programs/Colab-yoga;
# COLAB_YOGA_DIR overrides the location for portable or custom installs.
YOGA_DIR = Path(os.environ.get('COLAB_YOGA_DIR') or (_PROGRAMS / 'Colab-yoga'))
YOGA_EXE = YOGA_DIR / 'Colab-yoga.exe'
YOGA_PY = YOGA_DIR / 'runtime/python.exe'
YOGA_BRIDGE = YOGA_DIR / 'bridge/colab_yoga_bridge.py'
# Name under which the Colab-yoga bridge registers its session in ~/.config/colab-cli/sessions.json.
# Its current build still writes 'easycolab' (an inherited string, not an install path); both are accepted.
YOGA_SESSION_NAMES = ('colab-yoga', 'easycolab')


def run_streamed(argv: list[str]) -> tuple[int, str]:
    """Run argv with output echoed live and also returned (the bridge wake takes minutes; the log must not wait)."""
    proc = subprocess.Popen(argv, env=ENV, text=True, encoding='utf-8', errors='replace', stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for line in proc.stdout:
        print(line, end='', flush=True)
        lines.append(line)
    return proc.wait(), ''.join(lines)


def wake_succeeded(text: str) -> bool:
    return any(line.strip().startswith('WAKE_URL:') for line in (text or '').splitlines())


def cmd_yoga(rest: list[str]) -> int:
    if not YOGA_EXE.exists():
        raise SystemExit(
            'Colab-yoga not found at %s\n'
            'Install it from https://github.com/incredible404-rgb/Colab-yoga (Windows tray app; it installs to '
            '%%LOCALAPPDATA%%\\Programs\\Colab-yoga), or set COLAB_YOGA_DIR to an existing install. '
            'Colab-yoga is optional: start/launch/status/stop work without it.' % YOGA_EXE)
    opt = lambda flag, default: rest[rest.index(flag) + 1] if flag in rest else default  # noqa: E731
    pos = [a for i, a in enumerate(rest) if not a.startswith('-') and (i == 0 or rest[i - 1] not in ('--gpu', '--idle-min'))]
    sub = pos[0] if pos else 'gui'
    if sub == 'gui':
        subprocess.Popen([str(YOGA_EXE)], cwd=str(YOGA_DIR))
        print('[colab_cli] Colab-yoga GUI launched (%s)' % YOGA_EXE, flush=True)
        return 0
    if sub == 'status':
        return subprocess.run([str(YOGA_PY), '-u', str(YOGA_BRIDGE), 'auth_status'], env=ENV).returncode
    if sub == 'stop':
        return cmd_stop(['--all'])
    if sub == 'wake':
        gpu = opt('--gpu', 'G4')
        idle_min = int(opt('--idle-min', 45))
        rc, text = run_streamed([str(YOGA_PY), '-u', str(YOGA_BRIDGE), 'wake', '--gpu', gpu])
        # The bridge's wrapper exits 0 even when the wake failed (it prints WAKE_ERROR: ...), so success is the
        # WAKE_URL: line it prints once the server is up. A bridge that also returns non-zero on failure is fine.
        if rc != 0 or not wake_succeeded(text):
            print('[colab_cli] yoga wake did not report WAKE_URL; no keep-alive started. If a runtime was allocated, '
                  'check `status` and stop it if unwanted.', flush=True)
            return rc or 1
        active = [s for s in sessions() if s != '?']
        target = next((n for n in YOGA_SESSION_NAMES if n in active), active[0] if active else None)
        if target:
            start_keepalive(target, idle_min)
        return 0
    raise SystemExit('unknown yoga action %s (expected: gui | wake | status | stop)' % sub)


def main(argv: list[str]):
    force_utf8()
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    opt = lambda flag, default: rest[rest.index(flag) + 1] if flag in rest else default  # noqa: E731
    pos = [a for i, a in enumerate(rest) if not a.startswith('-') and (i == 0 or rest[i - 1] not in ('--gpu', '--idle-min'))]
    if cmd in ('yoga', 'colab-yoga', 'easycolab'):  # 'easycolab' is the pre-rename alias, kept for old scripts
        return cmd_yoga(rest)
    if cmd in ('start', 'verify', 'launch'):
        from colab_kernel import ALL_MODELS, MODELS
        slug = pos[0] if pos else next(iter(MODELS))
        if slug not in ALL_MODELS:
            raise SystemExit('slug required, one of: %s (archived: %s)' % (', '.join(MODELS), ', '.join(ALL_MODELS)))
        if cmd == 'verify':
            return cmd_verify(slug)
        if cmd == 'launch':
            return cmd_launch(slug, opt('--gpu', 'G4'), int(opt('--idle-min', 45)), '--skip-preflight' in rest)
        return cmd_start(slug, opt('--gpu', 'G4'), int(opt('--idle-min', 45)),
                         '--no-verify' not in rest, '--keep-on-fail' in rest, '--skip-preflight' in rest)
    if cmd == 'status':
        return cmd_status()
    if cmd == 'stop':
        return cmd_stop(pos or (['--all'] if '--all' in rest else []))
    raise SystemExit('unknown command %s' % cmd)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
