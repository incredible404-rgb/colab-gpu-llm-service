"""Shared configuration loader for the notebook generators.

Resolution order for every field: ``.secrets.json`` (repo root) -> environment variable -> placeholder.
``--sanitized`` skips the first two sources entirely, so a generated notebook can never carry a real
credential, tailnet name or Drive file id. ``--output-dir <dir>`` redirects the generated .ipynb files.
"""
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SECRETS_FILE = ROOT_DIR / '.secrets.json'

# field -> (environment variable, placeholder)
FIELDS = {
    'api_key': ('LLAMA_API_KEY', '<YOUR_LLAMA_API_KEY>'),
    'ts_client_id': ('TS_CLIENT_ID', '<YOUR_TS_OAUTH_CLIENT_ID>'),
    'ts_client_secret': ('TS_CLIENT_SECRET', '<YOUR_TS_OAUTH_CLIENT_SECRET>'),
    'tailnet_fqdn': ('TAILNET_FQDN', 'qwen.<YOUR_TAILNET>.ts.net'),
    # Tailscale IP of the Windows machine running local_novel_mcp.py (the WebUI file bridge dials it).
    'local_tailscale_ip': ('LOCAL_TAILSCALE_IP', '<YOUR_LOCAL_TAILSCALE_IP>'),
    'g4huihui_drive_id': ('G4HUIHUI_DRIVE_ID', '<YOUR_G4HUIHUI_NOTEBOOK_DRIVE_ID>'),
    'gemma31b_drive_id': ('GEMMA31B_DRIVE_ID', '<YOUR_GEMMA31B_NOTEBOOK_DRIVE_ID>'),
    'rvn27b_drive_id': ('RVN27B_DRIVE_ID', '<YOUR_RVN27B_NOTEBOOK_DRIVE_ID>'),
    'glm53_drive_id': ('GLM53_DRIVE_ID', '<YOUR_GLM53_NOTEBOOK_DRIVE_ID>'),
}


def _arg_value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return None


def _read_secrets(path):
    """Parse .secrets.json: the top level must be an object and every field a non-empty string."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except ValueError as e:
        raise SystemExit('%s is not valid JSON: %s' % (path.name, e))
    if not isinstance(data, dict):
        raise SystemExit('%s must be a JSON object at the top level, got %s' % (path.name, type(data).__name__))
    bad = [k for k, v in data.items() if not isinstance(v, str) or not v.strip()]
    if bad:
        raise SystemExit('%s: fields must be non-empty strings: %s' % (path.name, ', '.join(sorted(bad))))
    return data


def load_config(argv=None):
    """Return (config dict, sanitized bool, output_dir Path, sources dict field -> origin)."""
    argv = sys.argv[1:] if argv is None else argv
    sanitized = '--sanitized' in argv
    out_arg = _arg_value(argv, '--output-dir')
    output_dir = Path(out_arg).expanduser().resolve() if out_arg else ROOT_DIR / 'notebooks'
    output_dir.mkdir(parents=True, exist_ok=True)

    stored = {}
    if not sanitized and SECRETS_FILE.exists():
        stored = _read_secrets(SECRETS_FILE)
    config, sources = {}, {}
    for field, (env_name, placeholder) in FIELDS.items():
        if sanitized:
            config[field], sources[field] = placeholder, 'placeholder'
        elif stored.get(field):
            config[field], sources[field] = stored[field], '.secrets.json'
        elif os.environ.get(env_name):
            config[field], sources[field] = os.environ[env_name], 'env'
        else:
            config[field], sources[field] = placeholder, 'placeholder'
    return config, sanitized, output_dir, sources


def report(config, sanitized, sources):
    """One line on where the values came from; never prints a value."""
    if sanitized:
        print('Config: --sanitized, every sensitive field is a placeholder')
        return
    origin = sorted(set(sources.values()))
    print('Config sources: ' + ', '.join(origin))
    missing = [f for f, s in sources.items() if s == 'placeholder']
    if missing:
        print('WARNING: no value for %s; placeholders were written (create .secrets.json from '
              '.secrets.example.json)' % ', '.join(missing), file=sys.stderr)


def guard_sync(sanitized, sources, config=None):
    """Refuse to push a notebook to Colab when it holds placeholders or empty values."""
    if '--sync-colab' not in sys.argv:
        return
    if sanitized or any(src == 'placeholder' for src in sources.values()):
        raise SystemExit('--sync-colab needs real values from .secrets.json / environment; refusing to push '
                         'a notebook with placeholders')
    bad = sorted(f for f, v in (config or {}).items()
                 if not isinstance(v, str) or not v.strip() or '<YOUR_' in v)
    if bad:
        raise SystemExit('--sync-colab refused: empty or <YOUR_...> placeholder value in %s' % ', '.join(bad))
