"""Thin wrapper for Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS (delegates to colab_kernel.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_kernel import run_single

if __name__ == '__main__':
    run_single('qwen38-fn')
