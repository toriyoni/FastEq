"""Exclusive-GPU shared Linear GEMM versus a frozen v6 native runtime."""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType

import torch
from benchmark_linear_v5 import errors, force_loss, metadata
from benchmark_models import assert_exclusive_gpu, one_batch, stats

import fasteq
from fasteq.ops.indexed_linear import _plan


def load_v6(root):
    """Isolate the immutable v6 Python dispatcher and C++/CUDA execution."""
    directory = root / "src/FastEq/fasteq"
    prefix = "fasteq_v6_linear_bench"
    # Pybind's native type registry is process-global. Isolate only C++
    # namespace/type names in private copies; the original CUDA/math and
    # autograd code remain frozen. Never write the immutable source tree.
    cpp_directory = Path(os.environ["TORCH_EXTENSIONS_DIR"]).parent / "frozen_v6_cpp"
    cpp_directory.mkdir(parents=True, exist_ok=True)
    for filename in ("jit_runtime.h", "jit_runtime.cpp", "indexed_linear_plan.cpp", "norm_plan.cpp", "modulation_plan.cpp"):
        source = (directory / "cuda/src" / filename).read_text()
        source = source.replace("fasteq_jit", "fasteq_v6_linear_bench_jit")
        source = re.sub(r"\bLinearPlan\b", "V6BenchLinearPlan", source)
        source = re.sub(r"\bModulationPlan\b", "V6BenchModulationPlan", source)
        source = source.replace('"V6BenchModulationPlan"', '"ModulationPlan"')
        (cpp_directory / filename).write_text(source)
    for name, relative in ((prefix, "."), (prefix + ".cuda", "cuda"), (prefix + ".ops", "ops")):
        module = ModuleType(name)
        module.__path__, module.__package__ = [str(directory / relative)], name
        sys.modules[name] = module
    for suffix, relative in (("cuda.jit", "cuda/jit.py"), ("ops.indexed_linear", "ops/indexed_linear.py")):
        name, path = prefix + "." + suffix, directory / relative
        module = ModuleType(name)
        module.__package__, module.__file__ = name.rpartition(".")[0], str(path)
        source = path.read_text().replace("from fasteq.cuda.jit", f"from {prefix}.cuda.jit")
        source = source.replace('name="fasteq_cuda_jit"', 'name="fasteq_v6_linear_bench_cuda_jit"')
        source = source.replace('directory = Path(__file__).parent / "src"', f"directory = Path({str(cpp_directory)!r})")
        source = source.replace('"fasteq::indexed_linear"', f'"{prefix}::indexed_linear"')
        sys.modules[name] = module
        # Execute our local frozen-v6 module source with dispatcher/extension
        # names isolated; no network or user-supplied code is evaluated.
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return sys.modules[prefix + ".ops.indexed_linear"]._plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--nodes", nargs="+", type=int, default=[3072, 49152])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float64"), default=["float32", "float64"])
    parser.add_argument("--maps", nargs="+", choices=("up", "coefs"), default=["up", "coefs"])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    candidate = args.root / "src/FastEq_v7"
    assert Path(fasteq.__file__).resolve().is_relative_to(candidate)
    assert Path.cwd().resolve() == candidate
    # The isolated environment's sitecustomize establishes base cache paths
    # at Python startup; override after startup and before any JIT load.
    cache = args.cache_dir or args.root / "cache/v7_linear_work/torch_extensions"
    assert cache.is_relative_to(args.root / "cache/v7_linear_work")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)
    os.sched_setaffinity(0, {160})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["FASTEQ_EXPERIMENTAL_LINEAR_GEMM"] = "1"
    os.environ["FASTEQ_EXPERIMENTAL_LINEAR_SIMT"] = "1"
    _plan.cache_clear()
    assert_exclusive_gpu()
    baseline = load_v6(args.root)
    production = ["fasteq/ops/indexed_linear.py", "fasteq/cuda/src/indexed_linear.cu", "fasteq/cuda/src/indexed_linear_plan.cpp"]
    files = [root / relative for root in (args.root / "src/FastEq", candidate) for relative in production]
    files += [args.root / "src/FastEq/fasteq/cuda/src" / filename for filename in
              ("jit_runtime.h", "jit_runtime.cpp", "norm_plan.cpp", "modulation_plan.cpp")]

    def hashes():
        return {str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

    frozen = hashes()
    report = {"metadata": {"scope": "exclusive paired operator", "fasteq_file": fasteq.__file__,
        "cpu_affinity": [160], "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "source_sha256": frozen, "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tf32": False, "baseline": "independent frozen v6 dispatcher/CUDA/native C++ runtime",
        "candidate_flag": "FASTEQ_EXPERIMENTAL_LINEAR_GEMM=1", "baseline_simt_opt_in": True}, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for name in args.maps:
            din, dout, numel = (896, 1344, 319488) if name == "up" else (384, 576, 221184)
            for nodes in args.nodes:
                assert_exclusive_gpu()
                assert hashes() == frozen
                torch.manual_seed(763)
                x, y = [torch.randn(nodes, width, device="cuda", dtype=dtype) for width in (din, dout)]
                w = torch.randn(1, 1, numel, device="cuda", dtype=dtype)
                ids = torch.zeros(nodes, device="cuda", dtype=torch.int64)
                plans = [factory(metadata(name), dtype, x.device) for factory in (baseline, _plan)]
                for direction, role in (("forward", 2), ("dx", 0), ("dw", 1), ("force_loss", None)):
                    if role is None:
                        leaves = [(x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()) for _ in plans]
                        functions = [lambda p=p, a=a, b=b, ids=ids: force_loss(p, a, b, ids) for p, (a, b) in zip(plans, leaves, strict=True)]
                    else:
                        functions = [lambda p=p, role=role, ids=ids, x=x, w=w, y=y: p.contract(role, ids, x, w, y) for p in plans]
                    outputs = [fn() for fn in functions]
                    pairs = list(zip(outputs[1], outputs[0], strict=True)) if role is None else [(outputs[1], outputs[0])]
                    differences = [errors(a, b) for a, b in pairs]
                    # Fixed original tolerances. Record an explicit status:
                    # a changed reduction order is never silently excused.
                    failures = []
                    for a, b in pairs:
                        try:
                            torch.testing.assert_close(a, b, atol=3e-5 if dtype == torch.float32 else 2e-11,
                                                       rtol=5e-5 if dtype == torch.float32 else 2e-10)
                        except AssertionError as exception:
                            failures.append(str(exception))
                    for fn in functions:
                        for _ in range(5):
                            fn()
                    samples = [[], []]
                    for round_id in range(args.rounds):
                        assert_exclusive_gpu()
                        assert hashes() == frozen
                        for index in ((0, 1) if round_id % 2 == 0 else (1, 0)):
                            samples[index].append(one_batch(functions[index], 3 if role is None else args.iterations))
                        assert_exclusive_gpu()
                        assert hashes() == frozen
                    row = {"dtype": dtype_name, "map": name, "nodes": nodes, "direction": direction,
                        "precision": differences, "original_tolerances_passed": not failures,
                        "precision_failures": failures, "raw_samples": {"v6": samples[0], "v7_gemm": samples[1]},
                        "wall_ms": {label: stats([s["wall_ms"] for s in sample]) for label, sample in
                                    zip(("v6", "v7_gemm"), samples, strict=True)},
                        "paired_speedup": stats([a["wall_ms"] / b["wall_ms"] for a, b in zip(*samples, strict=True)])}
                    report["results"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(dtype_name, name, nodes, direction, "speedup", row["paired_speedup"]["median"],
                          "precision", not failures, differences, flush=True)
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
