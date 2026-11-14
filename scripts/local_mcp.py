"""Lifecycle for the local MCP bridge: `python -X utf8 scripts/local_mcp.py start | status | stop`.

The bridge itself is scripts/local_mcp.mjs, a Streamable-HTTP MCP server that exposes this machine's
Pi-native read/write/edit/bash tools (see that file). This launcher is stdlib-only and owns everything
around it: the dedicated bearer token in ~/.config/colab-cli/local-mcp.json, the detached spawn, the
readiness check, the log and a pid-verified stop. It prints exactly one JSON line on stdout per run and
never prints the token.

Written for cloud models that reach this PC through the Colab WebUI's --ui-mcp-proxy tunnel. Configure
the bearer token in the client's private MCP settings; notebooks and public defaults never contain it.
LOCAL_MCP_CONFIG and LOCAL_MCP_PORT
redirect the files and the port so a test never touches the real bridge (default port 3426, default cwd
from $LOCAL_MCP_CWD, else ~/Documents).
"""
from __future__ import annotations

import errno
from contextlib import contextmanager
import http.client
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

SERVER = Path(__file__).with_name('local_mcp.mjs')
DEFAULT_PORT = 3426
READY_TIMEOUT = 8.0  # a launcher may not report success before the port answers
STOP_TIMEOUT = 5.0
TOKEN_BYTES = 32

_HOST = '127.0.0.1'
LOG_MAX_BYTES = 1024 * 1024  # rotate local-mcp.log before a start once it grows past this
LOG_KEEP_BYTES = 256 * 1024  # ... keeping only this much of its tail
PI_PACKAGE = ('@earendil-works', 'pi-coding-agent')
_NOT_AN_ADDRESS = ('127.0.0.1', '0.0.0.0')


def config_path() -> Path:
    """~/.config/colab-cli/local-mcp.json, or $LOCAL_MCP_CONFIG (then $LOCAL_MCP_CONFIG_DIR / local-mcp.json)."""
    env = os.environ.get('LOCAL_MCP_CONFIG')
    if env:
        return Path(env).expanduser()
    directory = os.environ.get('LOCAL_MCP_CONFIG_DIR')
    base = Path(directory).expanduser() if directory else Path.home() / '.config/colab-cli'
    return base / 'local-mcp.json'


def log_path() -> Path:
    """~/.config/colab-cli/local-mcp.log, or $LOCAL_MCP_LOG. Holds reasons, never the token."""
    env = os.environ.get('LOCAL_MCP_LOG')
    return Path(env).expanduser() if env else config_path().parent / 'local-mcp.log'


def port() -> int:
    raw = (os.environ.get('LOCAL_MCP_PORT') or '').strip()
    if not raw:
        return DEFAULT_PORT
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit('LOCAL_MCP_PORT must be an integer, got %r' % raw)
    if not 0 < value < 65536:
        raise SystemExit('LOCAL_MCP_PORT out of range: %d' % value)
    return value


def control_host() -> str:
    """Probe the explicitly bound interface; wildcard listeners are reached through loopback."""
    host = (os.environ.get('LOCAL_MCP_HOST') or '').strip()
    return {'': _HOST, '0.0.0.0': _HOST, '::': '::1'}.get(host, host)


def url() -> str:
    host = control_host()
    if ':' in host:
        host = '[' + host + ']'
    return 'http://%s:%d/mcp' % (host, port())


def default_cwd() -> str:
    """Initial cwd: $LOCAL_MCP_CWD, then ~/Documents, then ~.

    This is a default location; tools also accept absolute paths with the current user's full rights.
    """
    configured = (os.environ.get('LOCAL_MCP_CWD') or '').strip()
    if configured:
        return str(Path(configured).expanduser().resolve())
    documents = Path.home() / 'Documents'
    return str((documents if documents.is_dir() else Path.home()).resolve())


def tailscale_bind_ip() -> str:
    """local_tailscale_ip (.secrets.json, then LOCAL_TAILSCALE_IP) when it is a real IPv4 address, else ''.

    Unset, a <YOUR_...> placeholder, 127.0.0.1 and 0.0.0.0 all count as "not configured".
    """
    import ipaddress
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from secrets_config import optional_setting
        value = optional_setting('local_tailscale_ip', 'LOCAL_TAILSCALE_IP')
    except Exception:
        value = (os.environ.get('LOCAL_TAILSCALE_IP') or '').strip()
    try:
        address = str(ipaddress.IPv4Address(value))
    except ValueError:
        return ''
    return '' if address in _NOT_AN_ADDRESS else address


def listen_env() -> dict:
    """Bind settings handed to local_mcp.mjs.

    An explicit LOCAL_MCP_HOST (or `start --host`) is passed through unchanged. Otherwise the server binds
    127.0.0.1 and, when tailscale_bind_ip() is set, also <local_tailscale_ip> (LOCAL_MCP_EXTRA_HOST), so the
    bridge is never exposed on every interface by default.
    """
    explicit = (os.environ.get('LOCAL_MCP_HOST') or '').strip()
    if explicit:
        return {'LOCAL_MCP_HOST': explicit, 'LOCAL_MCP_EXTRA_HOST': ''}
    # An unset host lets Node apply its loopback default and add the configured Tailscale listener.
    return {'LOCAL_MCP_HOST': '', 'LOCAL_MCP_EXTRA_HOST': tailscale_bind_ip()}


def load_config() -> dict:
    """The stored config ({} when the file does not exist yet). Never creates or rewrites it."""
    try:
        data = json.loads(config_path().read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as err:
        raise SystemExit('%s cannot be read (%s)' % (config_path(), err))
    if not isinstance(data, dict):
        raise SystemExit('%s must hold a JSON object' % config_path())
    return data


def _write_config(data: dict) -> None:
    """Replace the config atomically, owner-only where the OS honours it."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def ensure_config() -> dict:
    """Config with a strong token and a working directory; generated on the first start only.

    Existing keys are kept, so a hand-written "cwd" or a user-chosen token survives.
    """
    data = load_config()
    changed = not data
    token = data.get('token')
    if not isinstance(token, str) or not token.strip():
        data['token'] = secrets.token_urlsafe(TOKEN_BYTES)
        changed = True
    cwd = data.get('cwd')
    if not isinstance(cwd, str) or not cwd.strip():
        data['cwd'] = default_cwd()
        changed = True
    if changed:
        _write_config(data)
    return data


def _redact(text: str) -> str:
    """Strip the configured token (and Bearer headers) from text that is about to be printed."""
    token = load_config().get('token')
    if isinstance(token, str) and len(token) >= 8:
        text = text.replace(token, '***')
    return text


def _request(method: str, path: str, body: dict | None = None, token: str = '', timeout: float = 2.0):
    conn = http.client.HTTPConnection(control_host(), port(), timeout=timeout)
    headers = {'Accept': 'application/json'}
    payload = None
    if body is not None:
        payload = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    if token:
        headers['Authorization'] = 'Bearer ' + token
    try:
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        return response.status, raw
    finally:
        conn.close()


def _json(raw: bytes):
    try:
        return json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        return None


def probe(token: str = '', timeout: float = 2.0) -> dict | None:
    """GET /health; {} when the port does not answer, else the decoded body (may be a foreign service)."""
    try:
        status, raw = _request('GET', '/health', token=token, timeout=timeout)
    except (OSError, http.client.HTTPException):
        return None
    if status != 200:
        return {'http_status': status}
    body = _json(raw)
    return body if isinstance(body, dict) else {'http_status': status}


def _holder(health: dict) -> str:
    """What answers on the port: ours, or something else."""
    if health.get('server') == 'local_mcp':
        return 'local_mcp'
    return 'another program'


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == 'nt':
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return ctypes.get_last_error() == 5  # exists, we just may not look inside
            kernel32.CloseHandle(handle)
            return True
        except Exception:
            pass
    try:
        os.kill(pid, 0)
    except OSError as err:
        return err.errno == errno.EPERM  # exists, owned by someone else
    return True


def _log_tail(limit: int = 800) -> str:
    try:
        return _redact(log_path().read_text(encoding='utf-8', errors='replace')[-limit:].strip())
    except OSError:
        return ''


def _node() -> str:
    explicit = (os.environ.get('LOCAL_MCP_NODE') or '').strip()
    found = explicit or shutil.which('node')
    if not found:
        raise SystemExit('node not found on PATH (install a Node.js version supported by Pi or set LOCAL_MCP_NODE)')
    return found


def _pi_dir(node: str) -> Path:
    """The global Pi coding agent package, searched like local_mcp.mjs does; SystemExit with a hint if absent."""
    candidates = []
    explicit = (os.environ.get('LOCAL_MCP_PI_ROOT') or '').strip()
    if explicit:
        candidates.append(Path(explicit))
    roots = []
    if os.name == 'nt':
        roots.append(Path(os.environ.get('APPDATA') or Path.home() / 'AppData/Roaming') / 'npm/node_modules')
        roots.append(Path(node).resolve().parent / 'node_modules')
    else:
        if os.environ.get('npm_config_prefix'):
            roots.append(Path(os.environ['npm_config_prefix']) / 'lib/node_modules')
        node_parents = Path(node).resolve().parents
        if len(node_parents) >= 2:
            roots.append(node_parents[1] / 'lib/node_modules')
        roots += [Path('/opt/homebrew/lib/node_modules'), Path('/usr/local/lib/node_modules'),
                  Path('/usr/lib/node_modules'), Path.home() / '.npm-global/lib/node_modules']
    roots += [Path(entry) for entry in (os.environ.get('NODE_PATH') or '').split(os.pathsep) if entry]
    candidates += [root.joinpath(*PI_PACKAGE) for root in roots]
    for directory in candidates:
        if (directory / 'package.json').is_file():
            return directory
    raise SystemExit('%s not found (install it with "npm i -g %s" or set LOCAL_MCP_PI_ROOT)'
                     % ('/'.join(PI_PACKAGE), '/'.join(PI_PACKAGE)))


def _rotate_log() -> None:
    """Keep the last LOG_KEEP_BYTES (from a line start) once the log passes LOG_MAX_BYTES."""
    path = log_path()
    try:
        if path.stat().st_size <= LOG_MAX_BYTES:
            return
        with path.open('rb') as handle:
            handle.seek(-LOG_KEEP_BYTES, os.SEEK_END)
            tail = handle.read()
        newline = tail.find(b'\n')
        path.write_bytes(tail[newline + 1:] if newline >= 0 else tail)
    except OSError:
        pass


@contextmanager
def _lifecycle_lock():
    """The button and keeper share one port and token; serialize config creation, start and stop.

    OS file locks are released if a launcher exits, so a crashed launcher leaves no stale lock.
    """
    path = config_path().with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'\0')
            handle.flush()
        deadline = time.monotonic() + READY_TIMEOUT + 2
        while True:
            try:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as err:
                if err.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise SystemExit('another local_mcp start or stop is still running; retry shortly')
                time.sleep(0.1)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)



def _post_initialize(token: str, timeout: float = 2.0) -> int:
    """HTTP status of an authenticated initialize on the running server; 0 when the port does not answer."""
    try:
        status, _ = _request(
            'POST', '/mcp',
            body={'jsonrpc': '2.0', 'id': 'probe', 'method': 'initialize', 'params': {
                'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': 'local_mcp.py', 'version': '1'}}},
            token=token, timeout=timeout)
        return status
    except (OSError, http.client.HTTPException):
        return 0


def start() -> dict:
    with _lifecycle_lock():
        return _start()


def _start() -> dict:
    """Start the bridge when it is not already serving; report a conflict instead of taking over a port."""
    health = probe()
    if health is not None:
        if health.get('server') != 'local_mcp':
            return {'ok': False, 'started': False, 'url': url(), 'token_file': str(config_path()),
                    'error': 'port %d is held by %s; not starting another server and not stopping that one'
                             % (port(), _holder(health))}
        config = load_config()
        token = config.get('token') if isinstance(config.get('token'), str) else ''
        status = _post_initialize(token)
        if status == 401:
            return {'ok': False, 'started': False, 'url': url(), 'token_file': str(config_path()),
                    'error': 'port %d already serves local_mcp, but with a different token than %s; stop that '
                             'instance first (python -X utf8 scripts/local_mcp.py stop)'
                             % (port(), config_path())}
        if status == 200:
            running = probe(token) or {}
            return {'ok': True, 'started': False, 'pid': running.get('pid'), 'url': url(),
                    'token_file': str(config_path())}
        return {'ok': False, 'started': False, 'url': url(), 'token_file': str(config_path()),
                'error': 'local_mcp answers on port %d but rejects an initialize request (HTTP %s); '
                         'see %s' % (port(), status or 'timeout', log_path())}

    # Prerequisites first, so a machine without Node or Pi gets a clear reason and no local-mcp.json.
    node = _node()
    if not SERVER.is_file():
        raise SystemExit('missing server: %s' % SERVER)
    _pi_dir(node)
    config = ensure_config()
    env = {**os.environ, 'LOCAL_MCP_CONFIG': str(config_path()), 'LOCAL_MCP_PORT': str(port()),
           'LOCAL_MCP_LOG': str(log_path()), **listen_env()}
    log_path().parent.mkdir(parents=True, exist_ok=True)
    _rotate_log()
    flags = 0
    if os.name == 'nt':
        flags = (getattr(subprocess, 'DETACHED_PROCESS', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
                 | 0x01000000)  # CREATE_BREAKAWAY_FROM_JOB: survives the launcher's job object
    with open(log_path(), 'ab') as log:
        proc = subprocess.Popen(
            [node, str(SERVER)], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            close_fds=True, creationflags=flags, start_new_session=(os.name != 'nt'))
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {'ok': False, 'started': False, 'url': url(), 'token_file': str(config_path()),
                    'error': 'server exited with code %s before it was ready: %s'
                             % (proc.returncode, _log_tail() or '(no log)')}
        health = probe(timeout=0.5)
        if health is not None:
            if health.get('server') != 'local_mcp':
                return {'ok': False, 'started': False, 'url': url(), 'token_file': str(config_path()),
                        'error': 'port %d was taken by %s while local_mcp started (pid %d); that process was '
                                 'left alone: %s' % (port(), _holder(health), proc.pid, _log_tail() or '(no log)')}
            running = probe(str(config.get('token') or ''), timeout=0.5) or {}
            if running.get('pid') == proc.pid:
                return {'ok': True, 'started': True, 'pid': proc.pid, 'url': url(),
                        'token_file': str(config_path())}
        time.sleep(0.2)
    try:
        proc.terminate()
    except OSError:
        pass
    return {'ok': False, 'started': False, 'url': url(), 'token_file': str(config_path()),
            'error': 'local_mcp did not answer on port %d within %.0fs (pid %d stopped): %s'
                     % (port(), READY_TIMEOUT, proc.pid, _log_tail() or '(no log)')}


def status() -> dict:
    """Read-only report: whether the bridge serves, where its config is, never the token."""
    result = {'ok': True, 'running': False, 'url': url(), 'token_file': str(config_path()),
              'config_exists': config_path().is_file()}
    health = probe()
    if health is None:
        return result
    if health.get('server') != 'local_mcp':
        result['port_holder'] = _holder(health)
        return result
    config = load_config()
    token = config.get('token') if isinstance(config.get('token'), str) else ''
    running = probe(token) if token else health
    result.update({'running': True, 'pid': (running or health).get('pid'),
                   'cwd': (running or {}).get('cwd')})
    return result


def stop() -> dict:
    with _lifecycle_lock():
        return _stop()


def _stop() -> dict:
    """Stop the bridge after proving the pid on the port is the one that answered, never a recycled pid."""
    config = load_config()
    token = config.get('token') if isinstance(config.get('token'), str) else ''
    health = probe()
    if health is None:
        return {'ok': True, 'stopped': False, 'url': url(), 'token_file': str(config_path()),
                'reason': 'not running'}
    if health.get('server') != 'local_mcp':
        return {'ok': False, 'stopped': False, 'url': url(), 'token_file': str(config_path()),
                'error': 'port %d is held by %s; refusing to stop a process this launcher does not own'
                         % (port(), _holder(health))}
    if not token:
        return {'ok': False, 'stopped': False, 'url': url(), 'token_file': str(config_path()),
                'error': 'no token in %s, so the bridge cannot confirm ownership of its process' % config_path()}
    running = probe(token)
    pid = int((running or {}).get('pid') or 0)
    if not pid:
        return {'ok': False, 'stopped': False, 'url': url(), 'token_file': str(config_path()),
                'error': 'local_mcp answers without a verified pid (wrong token in %s?)' % config_path()}
    # Two independent reads must name the same live pid before anything is killed.
    if (probe(token) or {}).get('pid') != pid or not _pid_alive(pid):
        return {'ok': False, 'stopped': False, 'url': url(), 'token_file': str(config_path()),
                'error': 'pid %d did not verify twice; not killing it' % pid}
    _terminate(pid)
    deadline = time.monotonic() + STOP_TIMEOUT
    while time.monotonic() < deadline:
        if probe(timeout=0.5) is None:
            return {'ok': True, 'stopped': True, 'pid': pid, 'url': url(), 'token_file': str(config_path())}
        time.sleep(0.2)
    return {'ok': False, 'stopped': False, 'pid': pid, 'url': url(), 'token_file': str(config_path()),
            'error': 'pid %d did not exit within %.0fs' % (pid, STOP_TIMEOUT)}


def _terminate(pid: int) -> None:
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], capture_output=True)
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass


_COMMANDS = {'start': start, 'status': status, 'stop': stop}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help'):
        sys.stderr.write('usage: python -X utf8 scripts/local_mcp.py {start|status|stop} [--host HOST]\n')
        return 2
    if '--host' in argv[1:]:
        index = argv.index('--host', 1)
        if index + 1 >= len(argv):
            _emit({'ok': False, 'error': '--host needs an address'})
            return 2
        os.environ['LOCAL_MCP_HOST'] = argv[index + 1]  # explicit bind: same as LOCAL_MCP_HOST
    command = _COMMANDS.get(argv[0])
    if command is None:
        _emit({'ok': False, 'error': 'unknown command %r (expected start, status or stop)' % argv[0]})
        return 2
    try:
        result = command()
    except SystemExit as err:  # raised by port()/load_config() with a one-line reason
        _emit({'ok': False, 'error': str(err)})
        return 1
    except Exception as err:  # an unexpected OSError etc. still owes the caller one JSON line
        _emit({'ok': False, 'error': '%s: %s' % (type(err).__name__, err)})
        return 1
    _emit(result)
    return 0 if result.get('ok') else 1


def _emit(result: dict) -> None:
    """Write the single result line. Under pythonw stdout is absent, so the caller gets no line at all."""
    text = json.dumps(result, ensure_ascii=False)
    if sys.stdout is not None:
        sys.stdout.write(text + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    raise SystemExit(main())