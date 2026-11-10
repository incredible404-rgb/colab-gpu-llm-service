"""Kernel heartbeat + TFE keep-alive + idle guard for one Colab CLI session.

Started detached by colab_cli.py (Windows: pythonw + DETACHED_PROCESS |
CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB, or a one-shot schtasks
task if the parent job forbids breakaway):

  <uv-tool>/Scripts/pythonw.exe colab_keepalive.py <session> <idle_min> [spawn]

Why: google-colab-cli 0.7.5 starts no keep-alive daemon (`colab new` only
assigns), and a CLI runtime (T4 and G4 alike) is pruned 21-35 minutes after
creation. What keeps it: a short kernel execution every 30 s (`colab exec`,
the pattern Wakeloom's local mode uses). The daemon runs that execution itself
(InProcessBeat) the way `colab exec` does - a fresh kernel websocket through
the runtime proxy, one execute_request, the socket closed with the kernel kept -
so Colab sees the same traffic without a CLI process (about 1 s of imports)
starting every 30 s; any in-process failure falls back to `colab exec`. The TFE ping and llama-server
traffic alone do not; neither does one long-running cell. A 2026-11-07 T4
A/B: control pruned at 21 min, the 30 s exec arm lived the full soak. The old
RuntimeService KeepAliveAssignment RPC now answers 403 USER_PROJECT_DENIED.

The TFE ping is kept (cheap, harmless) the way the browser and colab-vscode do:

  GET https://colab.research.google.com/tun/m/<endpoint>/keep-alive/
  header X-Colab-Tunnel: Google
  Authorization: the same user bearer token as /tun/m/assign

The VM often never answers that path. ReadTimeout means TFE already recorded
LastActiveTime and is success. A real HTTP 4xx is failure; two in a row means
the assignment is gone and this process exits. The endpoint and the runtime
proxy token (about 1 h) are refreshed from GET /tun/m/assignments each cycle.
OAuth access tokens are refreshed from ~/.config/colab-cli/token.json.

Idle stop (default 45 min) uses llama-server /slots: is_processing, a move in
n_past/n_decoded, or an explicit last-request timestamp if the build has one.
Until /slots has answered once, that clock is paused (download/load is not
idle); if it never answers within the same limit, the session is stopped so a
failed boot cannot burn CU until the 24 h cap.

Colab-yoga (optional tray app) manages the same runtime with its own idle timer. The beat code
therefore also reads its status at http://127.0.0.1:7860/api/status inside the VM (2 s limit,
failure ignored) and prints one YOGA_STATUS line. A model download or VRAM load in progress, or
an idle countdown that went up since the last reading (someone pressed "extend"), counts as
activity for both clocks above. Without Colab-yoga there is no such line and nothing changes.

The tailnet name and API key come from secrets_config.load_config, the loader `start` uses:
a <YOUR_...> placeholder in .secrets.json is skipped in favour of the environment variable.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

# Imported before _import_colab() takes this directory off sys.path (scripts/colab_cli.py would shadow the package).
import secrets_config
from colab_path import find_colab

FOUR_XX_LIMIT = 2
PING_INTERVAL_S = 60
BEAT_INTERVAL_S = 30
BEAT_TIMEOUT_S = 60
BEAT_FAIL_LOG_EVERY = 10
# One short execution. Everything lives inside a function so the kernel's globals (the notebook cell shares
# them) gain only the name _keepalive_beat. The Colab-yoga read never raises and is capped at 2 s per step.
BEAT_CODE = (
    "def _keepalive_beat():\n"
    "    import json, time, urllib.request\n"
    "    with open('/content/.keepalive_beat', 'w') as f:\n"
    "        f.write(str(time.time()))\n"
    "    try:\n"
    "        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
    "        data = json.load(opener.open('http://127.0.0.1:7860/api/status', timeout=2))\n"
    "        keys = ('idle_remaining_seconds', 'is_downloading', 'is_loading_vram')\n"
    "        print('YOGA_STATUS ' + json.dumps({k: data.get(k) for k in keys}))\n"
    "    except Exception:\n"
    "        pass\n"
    "_keepalive_beat()\n"
)
YOGA_PREFIX = 'YOGA_STATUS '
YOGA_RESET_SLACK_S = 2  # the countdown only falls between readings; more than this above the last one is a reset
CREATE_NO_WINDOW = 0x08000000
MAX_HOURS = 24
LOG_MAX_BYTES = 256 * 1024
PING_TIMEOUT_S = 10
SLOT_TOKEN_KEYS = ('n_past', 'n_decoded', 'n_prompt_tokens')
SLOT_TIME_KEYS = ('last_request_time', 't_last_used', 'last_used')


def keep_alive_url(domain: str, endpoint: str) -> str:
    return '%s/tun/m/%s/keep-alive/' % (domain.rstrip('/'), endpoint)


def schtasks_task_name(session: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9_-]', '-', session)[:40]
    return 'ColabKeepalive-' + safe


def delete_schtask(session: str) -> None:
    """Drop the one-shot launcher task. Safe while this process is still running."""
    subprocess.run(
        ['schtasks', '/Delete', '/F', '/TN', schtasks_task_name(session)],
        capture_output=True)


def classify_ping(status_code, exc) -> str:
    """'ok' | 'http_4xx' | 'transient'. ReadTimeout is ok: TFE counts the hit before the VM answers."""
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return 'ok'
    if exc is not None and not isinstance(exc, requests.exceptions.ConnectTimeout):
        if 'Read timed out' in str(exc):
            return 'ok'
    if exc is not None and status_code is None:
        return 'transient'
    if status_code is None or 200 <= status_code < 400:
        return 'ok'
    if 400 <= status_code < 500:
        return 'http_4xx'
    return 'transient'


def slots_fingerprint(slots) -> tuple:
    busy = False
    tokens = 0
    stamps = []
    for slot in slots or []:
        if not isinstance(slot, dict):
            continue
        if slot.get('is_processing'):
            busy = True
        for key in SLOT_TOKEN_KEYS:
            value = slot.get(key)
            if isinstance(value, (int, float)):
                tokens += int(value)
                break
        for key in SLOT_TIME_KEYS:
            value = slot.get(key)
            if value not in (None, ''):
                stamps.append(str(value))
    return busy, tokens, tuple(stamps)


def inference_active(slots, prev) -> tuple:
    """Whether this /slots sample is a new or ongoing request.

    The first successful sample counts as activity so the idle clock starts
    when the server appears, not when the daemon started. After that, busy,
    a token-counter change, or a changed last-request timestamp resets it.
    """
    fp = slots_fingerprint(slots)
    if prev is None:
        return True, fp
    busy, tokens, stamps = fp
    if busy or tokens != prev[1] or stamps != prev[2]:
        return True, fp
    return False, fp


def parse_yoga_status(text) -> dict | None:
    """The Colab-yoga reading from a beat's printed output (a str, or a list of nbformat outputs); None when absent."""
    if isinstance(text, (list, tuple)):
        chunks = []
        for out in text:
            if isinstance(out, dict) and out.get('output_type') == 'stream':
                part = out.get('text', '')
                chunks.append(''.join(part) if isinstance(part, list) else str(part))
        text = ''.join(chunks)
    for line in str(text or '').splitlines():
        if line.startswith(YOGA_PREFIX):
            try:
                data = json.loads(line[len(YOGA_PREFIX):])
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


def yoga_activity(status, prev_remaining) -> tuple:
    """(active, remaining) for one Colab-yoga reading.

    Active while it downloads or loads a model, or when its idle countdown is higher than at the previous
    reading (reset by /api/touch or a new idle setting). No reading (Colab-yoga not running) is never activity
    and forgets the previous countdown. -1 means its idle release is off: no countdown to compare.
    """
    if not isinstance(status, dict):
        return False, None
    remaining = status.get('idle_remaining_seconds')
    if isinstance(remaining, bool) or not isinstance(remaining, (int, float)):
        remaining = None
    busy = bool(status.get('is_downloading') or status.get('is_loading_vram'))
    reset = (remaining is not None and remaining >= 0 and prev_remaining is not None and prev_remaining >= 0
             and remaining > prev_remaining + YOGA_RESET_SLACK_S)
    return busy or reset, remaining


def rotate_log(path: Path, max_bytes: int = LOG_MAX_BYTES) -> bool:
    """Bound the log: past max_bytes it becomes <name>.1 (replacing the previous .1) and a new log starts.

    The log is appended across runs of the same session name and was never trimmed. Best effort: a reader
    holding the file open on Windows makes os.replace fail, and that is not worth stopping the daemon for.
    """
    try:
        if path.stat().st_size <= max_bytes:
            return False
        os.replace(path, path.with_name(path.name + '.1'))
    except OSError:
        return False
    return True


def _paths(session: str):
    cfg = Path.home() / '.config/colab-cli'
    return cfg, cfg / ('keepalive-%s.json' % session), cfg / ('keepalive-%s.log' % session)


def _import_colab():
    # scripts/colab_cli.py would shadow the installed colab_cli package, and a
    # process that already imported that script keeps it in sys.modules.
    here = Path(__file__).resolve().parent
    sys.path = [p for p in sys.path if Path(p or '.').resolve() != here]
    loaded = sys.modules.get('colab_cli')
    if loaded is not None and not hasattr(loaded, '__path__'):
        del sys.modules['colab_cli']
    from colab_cli.client import ColabRequestError
    from colab_cli.common import State, _apply_proxy_info
    from colab_cli.utils import get_status_code
    return ColabRequestError, State, _apply_proxy_info, get_status_code


def refresh_oauth(client, log) -> None:
    """Refresh an expired access token and persist it next to the CLI's token.json."""
    creds = getattr(client.session, 'credentials', None)
    if creds is None or not getattr(creds, 'refresh_token', None):
        return
    if not getattr(creds, 'expired', False):
        return
    from google.auth.transport.requests import Request
    creds.refresh(Request())
    path = Path.home() / '.config/colab-cli/token.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(creds.to_json(), encoding='utf-8', newline='\n')
    os.replace(tmp, path)
    log('oauth access token refreshed')


def force_refresh_oauth(client, log) -> bool:
    creds = getattr(client.session, 'credentials', None)
    if creds is None or not getattr(creds, 'refresh_token', None):
        return False
    from google.auth.transport.requests import Request
    creds.refresh(Request())
    path = Path.home() / '.config/colab-cli/token.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(creds.to_json(), encoding='utf-8', newline='\n')
    os.replace(tmp, path)
    log('oauth access token force-refreshed after 401')
    return True


def ping_endpoint(client, endpoint, ColabRequestError, get_status_code) -> tuple:
    """Return (kind, detail) for one TFE keep-alive GET.

    Detail is read_timeout or http_<status>. The bearer token is the CLI session's.
    ColabRequestError / get_status_code stay in the signature so callers and tests
    can pass the CLI helpers; a direct GET is used so a 4xx is a status, not an exception.
    """
    del ColabRequestError, get_status_code
    url = keep_alive_url(client.colab_domain, endpoint)
    headers = {
        'X-Colab-Tunnel': 'Google',
        'Accept': 'application/json',
        'X-Colab-Client-Agent': 'colab-cli',
    }
    try:
        response = client.session.get(url, headers=headers, params={'authuser': '0'}, timeout=PING_TIMEOUT_S)
    except requests.exceptions.ReadTimeout:
        return 'ok', 'read_timeout'
    except Exception as err:
        status = getattr(getattr(err, 'response', None), 'status_code', None)
        return classify_ping(status, None if status is not None else err), (
            'http_%s' % status if status is not None else type(err).__name__)
    return classify_ping(response.status_code, None), 'http_%s' % response.status_code


def beat_argv(session: str, path: str) -> list[str]:
    return [find_colab(), 'exec', '-s', session, '-f', path, '--timeout', str(BEAT_TIMEOUT_S)]


def kernel_beat(session: str, path: str) -> tuple:
    """One short kernel execution via `colab exec`. Returns (ok, detail, stdout text).

    No console window from pythonw. The text carries the YOGA_STATUS line when Colab-yoga answered.
    """
    try:
        r = subprocess.run(beat_argv(session, path), capture_output=True, timeout=BEAT_TIMEOUT_S + 30,
                           stdin=subprocess.DEVNULL,
                           creationflags=CREATE_NO_WINDOW if os.name == 'nt' else 0,
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    except subprocess.TimeoutExpired:
        return False, 'timeout', ''
    except OSError as err:  # colab missing or not runnable: a failed beat, never a dead daemon
        return False, 'exec %s: %s' % (type(err).__name__, str(err)[:120]), ''
    text = (r.stdout or b'').decode('utf-8', 'replace')
    if r.returncode == 0:
        return True, 'rc0', text
    tail = (r.stdout or b'') + (r.stderr or b'')
    return False, 'rc%d %s' % (r.returncode, tail.decode('utf-8', 'replace').strip()[-160:]), text


class InProcessBeat:
    """The kernel beat run from this process, exactly as `colab exec` runs one.

    Each beat loads the session through the CLI's State (which refreshes the runtime
    proxy token near expiry, as `colab exec` does), opens a new ColabRuntime on the
    stored kernel, executes BEAT_CODE and closes the socket without shutting the
    kernel down. A socket per beat, rather than one kept open, is deliberate: one
    long-lived execution did not keep a runtime (2026-11-07 A/B), so the beat keeps
    the CLI's connect-execute-close shape and drops only the process start.

    A beat runs on a worker thread under the hard limit the subprocess had; a hung one
    is abandoned and counts as a failure. A failed beat is retried at once with
    `colab exec`, and after FALLBACK_AFTER failures in a row beats stay on `colab exec`
    for FALLBACK_BEATS beats before in-process is tried again.
    COLAB_KEEPALIVE_BEAT=exec turns the in-process beat off.
    """

    FALLBACK_AFTER = 3
    FALLBACK_BEATS = 20

    def __init__(self, state, session: str, runtime_cls=None, exec_beat=None, hard_timeout=None):
        self.state, self.session = state, session
        self._runtime_cls = runtime_cls
        self._exec_beat = exec_beat or kernel_beat
        self._hard_timeout = BEAT_TIMEOUT_S + 30 if hard_timeout is None else hard_timeout
        self.fails = 0
        self.exec_left = 0
        self.yoga = None  # Colab-yoga reading carried by the latest beat, None when it printed none
        self.disabled = os.environ.get('COLAB_KEEPALIVE_BEAT', '').strip().lower() == 'exec'

    def via(self) -> str:
        return 'exec' if self.disabled or self.exec_left else 'inproc'

    def _runtime(self):
        if self._runtime_cls is None:
            from colab_cli.runtime import ColabRuntime
            self._runtime_cls = ColabRuntime
        return self._runtime_cls

    def _once(self) -> tuple:
        """(ok, detail, outputs) for one in-process beat."""
        s = self.state.get_session(self.session, ignore_missing_session=True)
        if s is None:
            return False, 'inproc no_session', []

        def on_kernel(kid):
            s.kernel_id = kid
            self.state.store.add(s)

        def on_session(sid):
            s.session_id = sid
            self.state.store.add(s)

        rt = self._runtime()(s.url, s.token, kernel_id=s.kernel_id, session_id=s.session_id,
                             on_kernel_started=on_kernel, on_session_started=on_session)
        try:
            outputs = rt.execute_code(BEAT_CODE, timeout=BEAT_TIMEOUT_S) or []
        finally:
            try:
                rt.stop()
            except Exception:
                pass
        err = next((o for o in outputs if o.get('output_type') == 'error'), None)
        if err is not None:
            return False, 'inproc kernel_error %s' % err.get('ename', ''), outputs
        return True, 'inproc', outputs

    def _guarded(self) -> tuple:
        box = {}

        def run():
            try:
                box['r'] = self._once()
            except BaseException as e:  # a beat must never take the daemon down
                box['r'] = (False, 'inproc %s: %s' % (type(e).__name__, str(e)[:120]), [])

        t = threading.Thread(target=run, name='kernel-beat', daemon=True)
        t.start()
        t.join(self._hard_timeout)
        if t.is_alive():
            return False, 'inproc hung', []
        return box.get('r', (False, 'inproc no_result', []))

    def _run_exec(self, beat_file: str) -> tuple:
        """The `colab exec` beat as (ok, detail, text); a stand-in that returns (ok, detail) is accepted."""
        res = self._exec_beat(self.session, beat_file)
        return res[0], res[1], (res[2] if len(res) > 2 else '')

    def beat(self, beat_file: str) -> tuple:
        """(ok, detail) for one beat; detail names the path that delivered it. self.yoga holds its Colab-yoga reading."""
        self.yoga = None
        if self.disabled or self.exec_left > 0:
            self.exec_left = max(0, self.exec_left - 1)
            ok, detail, text = self._run_exec(beat_file)
            self.yoga = parse_yoga_status(text) if ok else None
            return ok, detail
        ok, detail, outputs = self._guarded()
        if ok:
            self.fails = 0
            self.yoga = parse_yoga_status(outputs)
            return ok, detail
        self.fails += 1
        if self.fails >= self.FALLBACK_AFTER:
            self.fails, self.exec_left = 0, self.FALLBACK_BEATS
        ok2, detail2, text = self._run_exec(beat_file)
        self.yoga = parse_yoga_status(text) if ok2 else None
        return ok2, '%s; exec %s' % (detail, detail2)


def refresh_endpoint(state, session, current, apply_proxy):
    """Re-read assignments. Return the endpoint to ping, or None if this session is gone."""
    assignments = state.client.list_assignments()
    by_endpoint = {item.endpoint: item for item in assignments}
    local = state.store.get(session)
    endpoint = local.endpoint if local is not None and local.endpoint else current
    if endpoint not in by_endpoint:
        return None
    if local is not None:
        token, url = local.token, local.url
        apply_proxy(local, by_endpoint[endpoint].runtime_proxy_info)
        # Same proxy token and url: the stored expiry is still the conservative one (the CLI refreshes on
        # its own near expiry), and add() would take the exclusive lock and rewrite sessions.json for nothing.
        if local.token != token or local.url != url:
            state.store.add(local)
    return endpoint


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.stderr.write('usage: colab_keepalive.py SESSION IDLE_MIN [spawn]\n')
        return 2
    session, idle_min = argv[1], float(argv[2])
    spawn = argv[3] if len(argv) > 3 else os.environ.get('COLAB_KEEPALIVE_SPAWN', 'direct')
    cfg, state_file, log_file = _paths(session)
    cfg.mkdir(parents=True, exist_ok=True)

    rotate_log(log_file)

    def log(msg: str) -> None:
        with log_file.open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')

    # Same loader as `start` (colab_cli._config): a placeholder copied from .secrets.example.json counts as unset,
    # so an environment variable fills it in. A placeholder tailnet name would leave /slots unreachable and the
    # never-up guard would stop a runtime somebody is using.
    try:
        config = secrets_config.load_config([])[0]
    except SystemExit as err:
        log('keep_alive_stopped reason=invalid_secrets_json error=%s' % err)
        return 1
    fqdn, api_key = config['tailnet_fqdn'], config['api_key']
    if secrets_config.is_placeholder(fqdn) or secrets_config.is_placeholder(api_key):
        log('keep_alive_stopped reason=missing_tailnet_fqdn_or_api_key')
        return 1
    slots_url = 'http://%s:443/slots' % fqdn
    slots_hdr = {'Authorization': 'Bearer ' + api_key}

    def save(**kw) -> None:
        payload = {'pid': os.getpid(), 'session': session, 'spawn': spawn, 'limit_min': idle_min, **kw}
        tmp = state_file.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload), encoding='utf-8', newline='\n')
        os.replace(tmp, state_file)

    ColabRequestError, State, apply_proxy, get_status_code = _import_colab()
    state = State()
    local = state.store.get(session)
    if local is None:
        log('keep_alive_stopped reason=no_local_session')
        if spawn == 'schtasks':
            delete_schtask(session)
        return 1
    endpoint = local.endpoint
    log('keep_alive_started pid=%d session=%s endpoint=%s spawn=%s idle_min=%s' % (
        os.getpid(), session, endpoint, spawn, idle_min))
    save(endpoint=endpoint, time=time.strftime('%Y-%m-%d %H:%M:%S'), pings=0,
         last_result='starting', consecutive_4xx=0, idle_min=0.0, busy=None, server_seen=False)
    t0 = time.monotonic()
    last_active = t0
    server_seen = False
    prev_fp = None
    errors = 0
    pings = 0
    beats = beat_fails = 0
    yoga_prev = None
    yoga_was_active = False
    last_ping = None
    slots_http = requests.Session()  # one kept-alive connection to /slots instead of a new TCP connect per minute
    beat_file = cfg / ('keepalive-%s-beat.py' % session)
    beat_file.write_text(BEAT_CODE, encoding='utf-8', newline='\n')
    beater = InProcessBeat(state, session)
    exit_reason = 'loop_ended'
    try:
        while True:
            cycle_start = time.monotonic()
            ok, detail = beater.beat(str(beat_file))
            beats += 1
            if ok:
                if beat_fails:
                    log('kernel_beat recovered n=%d after %d failures' % (beats, beat_fails))
                beat_fails = 0
            else:
                beat_fails += 1
                if beat_fails == 1 or beat_fails % BEAT_FAIL_LOG_EVERY == 0:
                    log('kernel_beat_error n=%d consecutive=%d detail=%s' % (beats, beat_fails, detail))
            if '; exec ' in detail:
                log('kernel_beat_fallback n=%d detail=%s' % (beats, detail))
            # Colab-yoga reports download/load and its own idle countdown through the beat; either is use.
            yoga_active, yoga_prev = yoga_activity(beater.yoga, yoga_prev)
            if yoga_active:
                last_active = time.monotonic()
                if not yoga_was_active:
                    log('yoga_activity downloading=%s loading=%s idle_remaining=%s' % (
                        beater.yoga.get('is_downloading'), beater.yoga.get('is_loading_vram'),
                        beater.yoga.get('idle_remaining_seconds')))
            yoga_was_active = yoga_active
            if last_ping is not None and cycle_start - last_ping < PING_INTERVAL_S - 1:
                time.sleep(max(0.0, BEAT_INTERVAL_S - (time.monotonic() - cycle_start)))
                continue
            last_ping = cycle_start
            rotate_log(log_file)
            try:
                refresh_oauth(state.client, log)
                refreshed = refresh_endpoint(state, session, endpoint, apply_proxy)
            except Exception as err:
                refreshed = endpoint
                log('keep_alive_error type=assignments detail=%s' % str(err)[:200])
            if not refreshed:
                exit_reason = 'assignment_gone'
                log('keep_alive_stopped reason=assignment_gone pings=%d endpoint=%s' % (pings, endpoint))
                break
            endpoint = refreshed
            kind, detail = ping_endpoint(state.client, endpoint, ColabRequestError, get_status_code)
            if kind == 'http_4xx' and detail == 'http_401':
                if force_refresh_oauth(state.client, log):
                    kind, detail = ping_endpoint(state.client, endpoint, ColabRequestError, get_status_code)
            pings += 1
            if kind == 'ok':
                errors = 0
                log('keep_alive_ping n=%d result=%s beats=%d beat_fails=%d beat_via=%s endpoint=%s' % (
                    pings, detail, beats, beat_fails, beater.via(), endpoint))
            elif kind == 'http_4xx':
                errors += 1
                log('keep_alive_error n=%d consecutive_4xx=%d result=%s endpoint=%s' % (
                    pings, errors, detail, endpoint))
                if errors >= FOUR_XX_LIMIT:
                    exit_reason = 'consecutive_4xx'
                    log('keep_alive_stopped reason=consecutive_4xx pings=%d last=%s' % (pings, detail))
                    break
            else:
                log('keep_alive_error n=%d result=transient detail=%s' % (pings, detail))
            busy = None
            try:
                slots = slots_http.get(slots_url, headers=slots_hdr, timeout=10).json()
                if isinstance(slots, dict):
                    slots = slots.get('slots') or slots.get('data') or []
                active, prev_fp = inference_active(slots, prev_fp)
                server_seen = True
                busy = slots_fingerprint(slots)[0]
                if active:
                    last_active = time.monotonic()
            except Exception:
                pass
            if server_seen:
                idle = (time.monotonic() - last_active) / 60
            else:
                idle = 0.0
            save(endpoint=endpoint, time=time.strftime('%Y-%m-%d %H:%M:%S'), pings=pings,
                 last_result=detail, consecutive_4xx=errors, idle_min=round(idle, 1),
                 busy=busy, server_seen=server_seen, beats=beats, beat_fails=beat_fails)
            if server_seen and idle >= idle_min:
                exit_reason = 'idle'
                log('keep_alive_stopped reason=idle idle_min=%.0f' % idle)
                _stop_session(session, log)
                break
            if not server_seen and (time.monotonic() - last_active) / 60 >= idle_min:
                exit_reason = 'server_never_up'
                log('keep_alive_stopped reason=server_never_up boot_min=%.0f' % ((time.monotonic() - t0) / 60))
                _stop_session(session, log)
                break
            if time.monotonic() - t0 > MAX_HOURS * 3600:
                exit_reason = 'max_hours'
                log('keep_alive_stopped reason=max_hours')
                _stop_session(session, log)
                break
            time.sleep(max(0.0, BEAT_INTERVAL_S - (time.monotonic() - cycle_start)))
    except Exception as err:
        log('crashed: %r' % err)
        raise
    finally:
        slots_http.close()
        beat_file.unlink(missing_ok=True)
        if spawn == 'schtasks':
            delete_schtask(session)
        if exit_reason != 'starting':
            state_file.unlink(missing_ok=True)
    return 0


def _stop_session(session: str, log) -> None:
    log('idle guard running colab stop -s %s' % session)
    subprocess.run([find_colab(), 'stop', '-s', session], capture_output=True, timeout=180,
                   stdin=subprocess.DEVNULL, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except Exception:
        raise
