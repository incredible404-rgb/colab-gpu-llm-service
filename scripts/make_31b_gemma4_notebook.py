"""Thin wrapper for Gemma-4-31B-Ortenzya-Heretic-Q8_0 (delegates to colab_kernel.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_kernel import run_single

if __name__ == '__main__':
    run_single('gemma4-31b')
