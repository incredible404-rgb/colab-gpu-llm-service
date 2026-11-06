"""Thin wrapper for Nemotron-3.5-Super-VL-120B-Heretic (delegates to colab_kernel.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_kernel import run_single

if __name__ == '__main__':
    run_single('nemotron-120b')
