"""Thin wrapper for Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL (delegates to colab_kernel.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_kernel import run_single

if __name__ == '__main__':
    run_single('qwen38-huihui-v2')
