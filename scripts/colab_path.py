"""Locate the `colab` executable without relying on the caller's PATH.

`uv tool install` puts colab(.exe) in uv's tool bin directory and `uv tool update-shell` only edits the PATH of
terminals opened later. An agent cannot reopen its terminal, and a keep-alive daemon spawned by that agent
inherits the same PATH, so every script that runs `colab` resolves it here: PATH first, then the directories uv
uses (UV_TOOL_BIN_DIR, XDG_BIN_HOME, ~/.local/bin), then `uv tool dir --bin`.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path

INSTALL_HINT = ('python -m pip install uv && python -m uv tool install "google-colab-cli==0.7.5" '
                '(then `python -m uv tool dir --bin` prints the folder holding colab; no terminal restart needed)')


def _names() -> tuple:
    return ('colab.exe', 'colab.cmd', 'colab') if os.name == 'nt' else ('colab',)


def _in_dir(directory) -> str | None:
    for name in _names():
        cand = Path(directory) / name
        if cand.is_file():
            return str(cand)
    return None


def _uv_bin_dir() -> str | None:
    """`uv tool dir --bin`, via `uv` on PATH, in ~/.local/bin, or `python -m uv`. None when uv cannot be run."""
    uv = shutil.which('uv')
    if not uv:
        for name in ('uv.exe', 'uv'):
            cand = Path.home() / '.local' / 'bin' / name
            if cand.is_file():
                uv = str(cand)
                break
    cmds = [[uv, 'tool', 'dir', '--bin']] if uv else []
    cmds.append([sys.executable, '-m', 'uv', 'tool', 'dir', '--bin'])
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL,
                               creationflags=0x08000000 if os.name == 'nt' else 0)
        except (OSError, subprocess.SubprocessError):
            continue
        line = (r.stdout or '').strip().splitlines()
        if r.returncode == 0 and line:
            return line[-1].strip()
    return None


@functools.lru_cache(maxsize=1)
def find_colab() -> str:
    """Absolute path of the colab executable, or plain 'colab' when nothing is found (the caller reports it)."""
    found = shutil.which('colab')
    if found:
        return found
    dirs = [os.environ.get(v) for v in ('UV_TOOL_BIN_DIR', 'XDG_BIN_HOME')]
    dirs.append(str(Path.home() / '.local' / 'bin'))
    for directory in filter(None, dirs):
        found = _in_dir(directory)
        if found:
            return found
    uv_dir = _uv_bin_dir()
    if uv_dir:
        found = _in_dir(uv_dir)
        if found:
            return found
    return 'colab'
