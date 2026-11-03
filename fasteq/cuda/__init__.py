import os, glob
import torch

_this_dir = os.path.dirname(__file__)
cands = glob.glob(os.path.join(_this_dir, "_cuda*.so"))
# Main defaults to a Python-only installation. Triton and the native JIT
# compile their own kernels, so they do not require the optional static library.
# Explicit CUDA builds still register exactly the same torch.ops entry points.
if cands:
    so_path = max(cands, key=os.path.getmtime)
    torch.ops.load_library(so_path)
