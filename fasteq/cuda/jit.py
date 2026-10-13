"""FastEq-owned native CUDA compilation and launch support.

This supplements FastEq's existing specialized CUDA extension/JIT kernels for
operators with dynamic layouts. GPU implementations reside in FastEq, and no
EQX module is imported or called by this compiler.
"""

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock

_loaded = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(
    max_workers=max(1, min(8, int(os.environ.get("MAX_JOBS", "8"))))
)


@lru_cache(maxsize=1)
def runtime():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("FastEq CUDA JIT requires CUDA_HOME and a CUDA toolkit")
    directory = Path(__file__).parent / "src"
    toolkit = Path(CUDA_HOME)
    sources = [directory / "jit_runtime.cpp"]
    flags = ["-O3"]
    linear = directory / "indexed_linear_plan.cpp"
    if linear.exists():
        sources.append(linear)
        flags.append("-DFASTEQ_INDEXED_LINEAR")
    norm = directory / "norm_plan.cpp"
    if norm.exists():
        sources.append(norm)
        flags.append("-DFASTEQ_NORM")
    modulation = directory / "modulation_plan.cpp"
    if modulation.exists():
        sources.append(modulation)
        flags.append("-DFASTEQ_MODULATION")
    segmented = directory / "segmented_product_plan.cpp"
    if segmented.exists():
        sources.append(segmented)
        flags.append("-DFASTEQ_SEGMENTED_PRODUCT")
    with _lock:
        return load(
            name="fasteq_cuda_jit",
            sources=[str(p) for p in sources],
            extra_include_paths=[str(directory), str(toolkit / "include")],
            extra_cflags=flags,
            extra_ldflags=[
                f"-L{toolkit / 'lib64'}",
                f"-Wl,-rpath,{toolkit / 'lib64'}",
                f"-L{toolkit / 'lib64/stubs'}",
                "-lnvrtc",
                "-lcuda",
                "-lc10_cuda",
            ],
            with_cuda=False,
            verbose=False,
        )


def _binary(source, options):
    compiler = runtime()
    digest = hashlib.sha256(
        (source + repr(options) + repr(compiler.version())).encode()
    ).hexdigest()
    cache = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "fasteq/native_cuda"
    )
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (digest + ".cubin")
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            result = compiler.compile(source, options)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            try:
                temporary.write_bytes(result)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path.read_bytes()


def kernels(sources, device, entry="run"):
    """Return resident executables for source strings on one CUDA device."""
    device = torch.device(device)
    with torch.cuda.device(device):
        major, minor = torch.cuda.get_device_capability(device)
        options = (
            "--std=c++17",
            f"--gpu-architecture=sm_{major}{minor}",
            "--fmad=true",
        )
        keys = [(device, source, entry) for source in dict.fromkeys(sources)]
        missing = [key for key in keys if key not in _loaded]
        if missing and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm up FastEq CUDA operators and derivatives before graph capture"
            )
        # Build before entering worker threads to keep extension loading serial.
        compiler = runtime()
        pending = {key: _pool.submit(_binary, key[1], options) for key in missing}
        for key in missing:
            _loaded[key] = compiler.Kernel(pending[key].result(), entry)
        # CUDA Graphs may outlive Python plan caches: executables stay resident.
        return {source: _loaded[(device, source, entry)] for source in sources}
