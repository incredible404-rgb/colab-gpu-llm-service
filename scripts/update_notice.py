"""Tell the person at the terminal, at most once a day, that a newer release of this repository exists.

It only tells: nothing is pulled or changed, since a clone may hold local edits (a model added to MODELS) and
its scripts start paid GPUs. The line goes to stderr, so a command's stdout (often one JSON object an agent
parses) is untouched. COLAB_UPDATE_CHECK=0 turns it off; a copy that is not a git clone with tags stays silent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RELEASES = 'https://github.com/incredible404-rgb/colab-gpu-llm-service/releases'
LATEST_URL = RELEASES + '/latest'
STATE = Path.home() / '.config/colab-cli/update-check.json'
EVERY_S = 24 * 3600
TIMEOUT_S = 3


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def version_numbers(tag: str) -> tuple[int, ...] | None:
    """'v1.2.3' or '1.2.3' -> (1, 2, 3); None for anything else, a prerelease tag included."""
    parts = tag.strip().removeprefix('v').split('.')
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def newer_than(tag: str, current: str) -> bool:
    a, b = version_numbers(tag or ''), version_numbers(current or '')
    return a is not None and b is not None and a > b


def current_tag() -> str:
    """The newest release tag this clone contains; '' when it is not a git clone or has no tags."""
    try:
        done = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'], cwd=REPO, capture_output=True,
                              text=True, stdin=subprocess.DEVNULL, timeout=TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return ''
    return done.stdout.strip() if done.returncode == 0 else ''


def latest_tag() -> str:
    """The tag GitHub's /releases/latest redirects to (no API, so no rate limit); '' when it names none."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(urllib.request.Request(LATEST_URL, method='HEAD'), timeout=TIMEOUT_S) as res:
            location = res.headers.get('Location', '')
    except urllib.error.HTTPError as err:  # the unfollowed 302 arrives here
        with err:
            location = err.headers.get('Location', '') if err.headers else ''
    marker = '/releases/tag/'
    return location.rsplit(marker, 1)[1].strip() if marker in location else ''


def _read_state() -> dict:
    try:
        state = json.loads(STATE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def message(now: float | None = None) -> str:
    """The line to show, '' when there is nothing to say. Asks GitHub when the last answer is a day old."""
    if os.environ.get('COLAB_UPDATE_CHECK', '').strip().lower() in ('0', 'off', 'false', 'no'):
        return ''
    current = current_tag()
    if version_numbers(current) is None:
        return ''
    now = time.time() if now is None else now
    state = _read_state()
    latest = str(state.get('latest') or '')
    if now - float(state.get('checked_at') or 0) >= EVERY_S:
        try:
            latest = latest_tag() or latest
        except (OSError, ValueError):
            pass  # offline: keep the last answer, and do not ask again before tomorrow
        try:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps({'checked_at': now, 'latest': latest}) + '\n', encoding='utf-8')
        except OSError:
            pass
    if not newer_than(latest, current):
        return ''
    return ('[update] colab-gpu-llm-service %s is out (this clone has %s). Notes: %s/tag/%s\n'
            '[update] To update: git -C "%s" pull --ff-only, then python -X utf8 scripts/colab_kernel.py --all'
            % (latest, current, RELEASES, latest, REPO))


def show() -> None:
    """Print the line to stderr. Never raises: a notice must not turn a finished command into a failure."""
    try:
        line = message()
        if line:
            print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
