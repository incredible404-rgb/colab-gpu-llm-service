"""Thin wrapper for Qwopus3.5-122B-A10B-Kimi-Q4_K_M (delegates to colab_kernel.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_kernel import run_single

if __name__ == '__main__':
    run_single('qwen35-122b')
