"""Make the console streams UTF-8 for scripts whose output carries Chinese text or kernel logs.

On Windows the default stdout codec follows the console code page (cp1252, cp936, ...) or, redirected, the
ANSI page; echoing a kernel line then raises UnicodeEncodeError in the middle of a paid operation. `python -X utf8`
fixes the same thing from the command line; this makes the scripts safe even when it was forgotten.
"""
import sys


def force_utf8():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            try:
                reconfigure(encoding='utf-8', errors='replace')
            except (OSError, ValueError):
                pass
