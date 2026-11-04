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
    'lit_drive_id': ('LIT_DRIVE_ID', '<YOUR_LIT_NOTEBOOK_DRIVE_ID>'),
    'g4_drive_id': ('G4_DRIVE_ID', '<YOUR_G4_NOTEBOOK_DRIVE_ID>'),
}


def _arg_value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return None


def load_config(argv=None):
    """Return (config dict, sanitized bool, output_dir Path, sources dict field -> origin)."""
    argv = sys.argv[1:] if argv is None else argv
    sanitized = '--sanitized' in argv
    out_arg = _arg_value(argv, '--output-dir')
    output_dir = Path(out_arg).expanduser().resolve() if out_arg else ROOT_DIR / 'notebooks'
    output_dir.mkdir(parents=True, exist_ok=True)

    stored = {}
    if not sanitized and SECRETS_FILE.exists():
        stored = json.loads(SECRETS_FILE.read_text(encoding='utf-8'))
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


def guard_sync(sanitized, sources):
    """Refuse to push a notebook to Colab when it holds placeholders."""
    if '--sync-colab' not in sys.argv:
        return
    if sanitized or any(s == 'placeholder' for s in sources.values()):
        raise SystemExit('--sync-colab needs real values from .secrets.json / environment; refusing to push '
                         'a notebook with placeholders')
