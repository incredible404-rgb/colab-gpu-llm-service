"""Thin wrapper for GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4 (delegates to colab_kernel.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_kernel import run_single

if __name__ == '__main__':
    run_single('glm53-flash')
