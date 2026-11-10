"""Thin wrapper for Qwen3.8-27B-RVN-Heretic-BF16 (delegates to colab_kernel.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_kernel import run_single

if __name__ == '__main__':
    run_single('qwen38-27b-rvn')
