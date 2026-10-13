"""Paired eager CUDA timing for the equivariant adapters (no profiler).

Compilation and correctness checks precede timing. CUDA events measure the
GPU timeline, including stream idle time caused by Python dispatch. Wall time
also includes dispatch and end-of-batch synchronization. Neither is a hardware
counter or a sum of standalone kernel durations.
"""

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("FASTEQ_UNIFORM1D_FWD_TUNE", "0")
os.environ.setdefault("FASTEQ_UNIFORM1D_BWD_TUNE", "0")
for flag in ("EQX", "OEQ", "CUE", "EQT"):
    os.environ["TACE_USE_" + flag] = "0"

import torch
from e3nn import o3
from e3nn.nn import Gate
from eqx.o2 import TensorProduct as O2TensorProduct
from prophet.layer_norm import RMSLayerNorm
from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm

from fasteq.adapters import adapt_model, adaptation_report

ROOT = Path("/public-data/zhouxibo/zxb/tace_prophet_fasteq")
OPERATORS = (
    "linear",
    "tace_norm",
    "prophet_norm",
    "gate",
    "sh",
    "o2_ece",
    "uvu",
    "uuu",
)


def build(name, n, dtype, channels=64, lmax=2):
    ir = o3.Irreps([(channels, (l, (-1) ** l)) for l in range(lmax + 1)])
    if name == "linear":
        module = o3.Linear(ir, ir)
        shapes = [(n, ir.dim)]
    elif name == "tace_norm":
        module = EquivariantMergeLayerNorm(list(range(lmax + 1)), channels)
        shapes = [(n, (lmax + 1) ** 2, channels)]
    elif name == "prophet_norm":
        module = RMSLayerNorm(ir, centering=False)
        shapes = [(n, ir.dim)]
    elif name == "gate":
        gated = o3.Irreps([(channels, (l, (-1) ** l)) for l in range(1, lmax + 1)])
        module = Gate(
            f"{channels}x0e",
            [torch.nn.SiLU()],
            f"{channels * lmax}x0e",
            [torch.nn.Sigmoid()],
            gated,
        )
        shapes = [(n, module.irreps_in.dim)]
    elif name == "sh":
        module = o3.SphericalHarmonics(
            list(range(lmax + 1)), normalize=True, normalization="component"
        )
        shapes = [(n, 3)]
    elif name == "o2_ece":
        module = O2TensorProduct(
            f"{channels}x1me",
            f"{channels}x1me",
            f"{channels}x0ee+{channels}x0oe+{channels}x2me",
            [(0, 0, i, "uuu", True) for i in range(3)],
        )
        shapes = [(n, channels * 2), (n, channels * 2)]
    else:
        ir2 = o3.Irreps(
            [
                (1 if name == "uvu" else channels, (l, (-1) ** l))
                for l in range(lmax + 1)
            ]
        )
        paths = [
            (i, j, k, name, True)
            for i, (_, a) in enumerate(ir)
            for j, (_, b) in enumerate(ir2)
            for k, (_, c) in enumerate(ir)
            if c in a * b
        ][:10]
        module = o3.TensorProduct(
            ir, ir2, ir, paths, internal_weights=False, shared_weights=False
        )
        shapes = [(n, ir.dim), (n, ir2.dim), (n, module.weight_numel)]
    module = module.to(device="cuda", dtype=dtype)
    inputs = [
        torch.randn(shape, device="cuda", dtype=dtype).requires_grad_()
        for shape in shapes
    ]
    return module, inputs


def operation(module, inputs, cotangent, mode):
    if mode == "forward":
        with torch.no_grad():
            return module(*inputs)
    output = module(*inputs)
    # A fixed scalar objective avoids gradient accumulation and supplies a
    # nonlinear term, so input Hessians are also exercised for linear maps.
    energy = (output * cotangent).sum() / output.numel() + 0.1 * output.square().mean()
    leaves = inputs + list(module.parameters())
    if mode == "backward":
        return torch.autograd.grad(energy, leaves, allow_unused=True)
    first = torch.autograd.grad(energy, inputs, create_graph=True, allow_unused=True)
    force_loss = sum(
        g.square().mean() for g in first if g is not None and g.requires_grad
    )
    second = torch.autograd.grad(force_loss, leaves, allow_unused=True)
    return first, second


def flatten(value):
    if isinstance(value, tuple):
        return [t for v in value for t in flatten(v)]
    return [value]


def compare(a, b, dtype):
    max_abs = 0.0
    for x, y in zip(flatten(a), flatten(b)):
        if x is None or y is None:
            z = y if x is None else x
            if z is not None:
                torch.testing.assert_close(
                    z,
                    torch.zeros_like(z),
                    atol=1e-6 if dtype == torch.float32 else 1e-10,
                    rtol=0,
                )
            continue
        torch.testing.assert_close(
            x,
            y,
            atol=3e-4 if dtype == torch.float32 else 3e-8,
            rtol=4e-4 if dtype == torch.float32 else 3e-7,
        )
        if x.numel():
            max_abs = max(max_abs, (x - y).abs().max().item())
    return max_abs


def summary(values):
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median": statistics.median(values),
        "q1": q[0],
        "q3": q[2],
        "iqr": q[2] - q[0],
        "samples": values,
    }


def measure(fn, iterations):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        result = fn()
        del result
    end.record()
    end.synchronize()
    wall = time.perf_counter() - wall_start
    return start.elapsed_time(end) * 1000 / iterations, wall * 1e6 / iterations


def peak(fn):
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    value = torch.cuda.max_memory_allocated() - base
    del result
    return value


def save(path, metadata, rows, errors):
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": rows, "errors": errors}, indent=2)
    )
    fields = [
        "operator",
        "dtype",
        "n",
        "channels",
        "lmax",
        "mode",
        "backend",
        "baseline_event_us",
        "adapted_event_us",
        "event_speedup",
        "baseline_wall_us",
        "adapted_wall_us",
        "wall_speedup",
        "baseline_peak_bytes",
        "adapted_peak_bytes",
        "baseline_rows_s",
        "adapted_rows_s",
        "event_speedup_iqr",
        "max_abs_error",
        "iterations",
        "batches",
    ]
    with (path / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operators", nargs="+", choices=OPERATORS, default=list(OPERATORS)
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument("--nodes", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--edges", nargs="+", type=int, default=[1024, 8192])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["forward", "backward", "force_loss"],
        default=["forward", "backward", "force_loss"],
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "performance/operators")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261007)
    gpu = torch.cuda.get_device_properties(0)
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu.name,
        "gpu_uuid": str(gpu.uuid),
        "total_memory_bytes": gpu.total_memory,
        "device_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32": False,
        "compile_and_warmup_excluded": True,
        "autotuning": False,
        "timing": "paired alternating eager batches; CUDA event GPU timeline includes host dispatch gaps; wall includes synchronization",
        "memory": "PyTorch CUDA peak allocated minus persistent allocation immediately before one operation, with result retained until sampled; excludes cached/reserved memory",
        "force_loss": "forward scalar objective; grad wrt all explicit differentiable inputs with create_graph=True; squared first gradients; grad wrt inputs and parameters",
        "verification": "forward, first and second gradients at 16 rows, identical parameters and inputs, before full-size warmup",
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "gpu_state_start": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                "--format=csv",
            ],
            text=True,
        ),
    }
    metadata["runs"] = [metadata["settings"]]
    rows, errors = [], []
    if args.resume and (args.output / "results.json").exists():
        previous = json.loads((args.output / "results.json").read_text())
        rows, errors = previous["results"], previous["errors"]
        metadata["runs"] = (
            previous["metadata"].get("runs", [previous["metadata"]["settings"]])
            + metadata["runs"]
        )
        metadata["gpu_state_start"] = previous["metadata"]["gpu_state_start"]
    done = {
        (r["operator"], r["dtype"], r["n"], r["channels"], r["lmax"], r["mode"])
        for r in rows
    }
    for name in args.operators:
        sizes = args.edges if name in ("sh", "o2_ece", "uvu", "uuu") else args.nodes
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for n in sizes:
                try:
                    print(f"SETUP {name} {dtype_name} n={n}", flush=True)
                    reference, inputs = build(name, n, dtype, args.channels, args.lmax)
                    adapted = adapt_model(reference)
                    adapted_inputs = [
                        x.detach().clone().requires_grad_() for x in inputs
                    ]
                    cotangent = torch.randn_like(reference(*inputs).detach())
                    small = [
                        [x[:16].detach().clone().requires_grad_() for x in inputs]
                        for _ in range(2)
                    ]
                    verify = {}
                    for mode in args.modes:
                        verify[mode] = compare(
                            operation(reference, small[0], cotangent[:16], mode),
                            operation(adapted, small[1], cotangent[:16], mode),
                            dtype,
                        )
                    functions = []
                    for mode in args.modes:
                        if (
                            name,
                            dtype_name,
                            n,
                            args.channels,
                            args.lmax,
                            mode,
                        ) in done:
                            continue
                        print(f"WARM {name} {dtype_name} n={n} {mode}", flush=True)
                        functions = [
                            lambda m=m, xs=xs, cotangent=cotangent, mode=mode: (
                                operation(m, xs, cotangent, mode)
                            )
                            for m, xs in (
                                (reference, inputs),
                                (adapted, adapted_inputs),
                            )
                        ]
                        for fn in functions:
                            for _ in range(args.warmup):
                                result = fn()
                                del result
                        torch.cuda.synchronize()
                        events, walls = [[], []], [[], []]
                        for batch in range(args.batches):
                            for which in (0, 1) if batch % 2 == 0 else (1, 0):
                                event, wall = measure(functions[which], args.iterations)
                                events[which].append(event)
                                walls[which].append(wall)
                        event_stats = [summary(values) for values in events]
                        wall_stats = [summary(values) for values in walls]
                        ratios = summary([a / b for a, b in zip(*events)])
                        peaks = [peak(fn) for fn in functions]
                        row = {
                            "operator": name,
                            "dtype": dtype_name,
                            "n": n,
                            "channels": args.channels,
                            "lmax": args.lmax,
                            "mode": mode,
                            "backend": adaptation_report(adapted)[0]["backend"],
                            "baseline_event_us": event_stats[0]["median"],
                            "adapted_event_us": event_stats[1]["median"],
                            "event_speedup": ratios["median"],
                            "baseline_wall_us": wall_stats[0]["median"],
                            "adapted_wall_us": wall_stats[1]["median"],
                            "wall_speedup": statistics.median(
                                [a / b for a, b in zip(*walls)]
                            ),
                            "baseline_peak_bytes": peaks[0],
                            "adapted_peak_bytes": peaks[1],
                            "baseline_rows_s": n * 1e6 / event_stats[0]["median"],
                            "adapted_rows_s": n * 1e6 / event_stats[1]["median"],
                            "event_speedup_iqr": ratios["iqr"],
                            "max_abs_error": verify[mode],
                            "iterations": args.iterations,
                            "batches": args.batches,
                            "event_statistics_us": event_stats,
                            "wall_statistics_us": wall_stats,
                            "paired_event_speedup": ratios,
                        }
                        rows.append(row)
                        save(args.output, metadata, rows, errors)
                        print(
                            f"RESULT {name} {dtype_name} n={n} {mode}: event {row['baseline_event_us']:.2f} -> {row['adapted_event_us']:.2f} us ({row['event_speedup']:.3f}x), wall {row['wall_speedup']:.3f}x",
                            flush=True,
                        )
                    del (
                        reference,
                        adapted,
                        inputs,
                        adapted_inputs,
                        cotangent,
                        small,
                        functions,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001 -- Record failed grid entries and continue.
                    traceback.print_exc()
                    errors.append(
                        {
                            "operator": name,
                            "dtype": dtype_name,
                            "n": n,
                            "error": repr(exc),
                        }
                    )
                    save(args.output, metadata, rows, errors)
    metadata["gpu_state_end"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv",
        ],
        text=True,
    )
    save(args.output, metadata, rows, errors)
    print(f"DONE {len(rows)} results; {len(errors)} errors; {args.output}", flush=True)


if __name__ == "__main__":
    main()
