"""Python/native modulation host comparison; shared GPUs allow precision only."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import pynvml
import torch
from benchmark_models import assert_exclusive_gpu, one_batch, stats
from e3nn import o3
from fasteq.ops.modulation import ModulatedResidualPlan


def nvml_state():
    pynvml.nvmlInit()
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "uuid": uuid,
        "memory_free": memory.free,
        "memory_used": memory.used,
        "compute_processes": [
            {"pid": p.pid, "memory_bytes": p.usedGpuMemory}
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"])
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument(
        "--modes", nargs="+", default=["energy", "forces", "force_loss"]
    )
    parser.add_argument("--precision-only", action="store_true")
    parser.add_argument("--allocator-fraction", type=float, default=0.025)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.precision_only:
        torch.cuda.set_per_process_memory_fraction(args.allocator_fraction)
    else:
        assert_exclusive_gpu()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "driver.py").write_bytes(Path(__file__).read_bytes())
    sources = [
        "src/FastEq/fasteq/ops/modulation.py",
        "src/FastEq/fasteq/ops/equivariant_norm.py",
        "src/FastEq/fasteq/triton/fused_equivariant_gate.py",
        "src/FastEq/fasteq/cuda/triton_launch.py",
        "src/FastEq/fasteq/cuda/jit.py",
        "src/FastEq/fasteq/cuda/src/jit_runtime.cpp",
        "src/FastEq/fasteq/cuda/src/jit_runtime.h",
        "src/FastEq/fasteq/cuda/src/modulation_plan.cpp",
        "src/FastEq/fasteq/cuda/src/indexed_linear_plan.cpp",
        "src/FastEq/fasteq/cuda/src/norm_plan.cpp",
        "src/FastEq/benchmarks/adapters/benchmark_models.py",
    ]

    def hashes():
        return {
            path: hashlib.sha256((args.root / path).read_bytes()).hexdigest()
            for path in sources
        }

    frozen = hashes()
    current = (args.root / sources[0]).read_text()
    baseline_path = (
        args.root / "performance/fasteq_optimization/v5_snapshot" / sources[0]
    )
    original = baseline_path.read_text()

    def mathematics(source):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name
            in {
                "_forward",
                "_backward",
                "_double",
                "_gate_reduce",
                "_Gradient",
                "_Modulation",
            }
        }

    assert mathematics(original) == mathematics(current), (
        "CUDA math/Python reference changed from v5"
    )
    report = {
        "metadata": {
            "kind": "shared_precision_only"
            if args.precision_only
            else "exclusive_paired_microchain",
            "source_sha256": frozen,
            "v5_modulation_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "cuda_math_and_python_ad_ast_match_v5": True,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "gpu": torch.cuda.get_device_name(),
            "protocol": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "baseline_note": "The microbenchmark isolates native versus flag-off Python with the current shared launcher. CUDA math and Python AD classes match frozen v5; this is not a fully isolated frozen-v5 runtime baseline. The separate model driver loads frozen v5 independently.",
            "scope": "Positions and all projection/gating/skip weights -> modulation -> nonlinear energy -> physical forces -> energy+force loss -> all parameter/position grads. Scalar MLP unmodified. Python flag-off and opt-in share verified identical v5 CUDA math.",
            "timing_note": "No timing/profiler collected on a shared GPU. Normal paired wall/event timing excludes warmup/compilation; event span includes host launch gaps and is not summed kernel work.",
        },
        "results": [],
    }
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        ir = o3.Irreps(f"{args.channels}x0e+{args.channels}x1o+{args.channels}x2e")
        previous = os.environ.get("FASTEQ_EXPERIMENTAL_NATIVE_MODULATION")
        plans = []
        for enabled in (False, True):
            os.environ["FASTEQ_EXPERIMENTAL_NATIVE_MODULATION"] = str(int(enabled))
            plans.append(ModulatedResidualPlan(ir))
        if previous is None:
            os.environ.pop("FASTEQ_EXPERIMENTAL_NATIVE_MODULATION")
        else:
            os.environ["FASTEQ_EXPERIMENTAL_NATIVE_MODULATION"] = previous
        torch.manual_seed(867)
        pos = torch.randn(args.nodes, 3, device="cuda", dtype=dtype, requires_grad=True)
        weights = [
            torch.randn(3, width, device="cuda", dtype=dtype, requires_grad=True)
            for width in (ir.dim, plans[0].channels, ir.dim)
        ]
        parameters = (pos, *weights)
        index = torch.tensor(plans[0]._index, device="cuda")
        seed = torch.randn(args.nodes, ir.dim, device="cuda", dtype=dtype)
        for mode in args.modes:
            assert hashes() == frozen
            before = nvml_state()

            def execute(
                plan,
                collect=False,
                *,
                mode=mode,
                parameters=parameters,
                pos=pos,
                weights=weights,
                index=index,
                seed=seed,
            ):
                for parameter in parameters:
                    parameter.grad = None
                with torch.set_grad_enabled(mode != "energy"):
                    x, gates = pos @ weights[0], (pos @ weights[1]).sin()
                    residual = pos.sin() @ weights[2] / 9
                    y = (
                        (residual + x * gates.index_select(-1, index))
                        if plan is None
                        else plan(x, gates, residual)
                    )
                    force = None
                    energy = (y.sin() * seed).sum() / pos.shape[0] + y.square().mean()
                    if mode != "energy":
                        force = -torch.autograd.grad(
                            energy, pos, create_graph=mode == "force_loss"
                        )[0]
                        if mode == "force_loss":
                            (energy.square() / 10 + force.square().sum()).backward()
                if not collect:
                    return y
                values = [y.detach().clone(), energy.detach().clone()]
                if force is not None:
                    values.append(force.detach().clone())
                values.extend(
                    p.grad.detach().clone() for p in parameters if p.grad is not None
                )
                return values, None if y.grad_fn is None else y.grad_fn.name()

            expected, _ = execute(None, True)
            validations = []
            for name, plan in zip(
                ("python_current", "native_candidate"), plans, strict=True
            ):
                actual, node_name = execute(plan, True)
                assert len(actual) == len(expected)
                dif = total = 0.0
                maximum = 0.0
                for a, b in zip(actual, expected, strict=True):
                    torch.testing.assert_close(
                        a,
                        b,
                        atol=4e-5 if dtype == torch.float32 else 2e-10,
                        rtol=4e-4 if dtype == torch.float32 else 2e-9,
                    )
                    delta = a.double() - b.double()
                    maximum = max(maximum, delta.abs().max().item())
                    dif += delta.square().sum().item()
                    total += b.double().square().sum().item()
                validations.append(
                    {
                        "backend": name,
                        "max_abs": maximum,
                        "relative_l2": (dif / max(total, 1e-300)) ** 0.5,
                        "compared_tensor_count": len(actual),
                        "forward_node": node_name,
                    }
                )
            native = plans[1]._native[(pos.device, dtype)]
            assert native.cached_executables > 0
            if mode != "energy":
                assert validations[1]["forward_node"] == "FastEqModulationBackward"
            samples = [[], []]
            if not args.precision_only:
                functions = [lambda p=p, run=execute: run(p) for p in plans]
                for function in functions:
                    for _ in range(args.warmup):
                        function()
                for round_index in range(args.rounds):
                    assert_exclusive_gpu()
                    assert hashes() == frozen
                    for k in ((round_index % 2), (1 - round_index % 2)):
                        samples[k].append(one_batch(functions[k], args.iterations))
                    assert_exclusive_gpu()
                    assert hashes() == frozen
            row = {
                "dtype": dtype_name,
                "nodes": args.nodes,
                "channels": args.channels,
                "mode": mode,
                "source_sha256": frozen,
                "validation": validations,
                "native_cached_executables": native.cached_executables,
                "nvml_before": before,
                "nvml_after": nvml_state(),
            }
            if not args.precision_only:
                row.update(
                    wall_ms={
                        name: stats([s["wall_ms"] for s in sample])
                        for name, sample in zip(
                            ("python_current", "native_candidate"), samples, strict=True
                        )
                    },
                    event_ms={
                        name: stats([s["cuda_event_ms"] for s in sample])
                        for name, sample in zip(
                            ("python_current", "native_candidate"), samples, strict=True
                        )
                    },
                    samples={
                        name: sample
                        for name, sample in zip(
                            ("python_current", "native_candidate"), samples, strict=True
                        )
                    },
                    paired_python_over_native=stats(
                        [
                            a["wall_ms"] / b["wall_ms"]
                            for a, b in zip(*samples, strict=True)
                        ]
                    ),
                )
            report["results"].append(row)
            assert hashes() == frozen
            (args.output / "results.json").write_text(json.dumps(report, indent=2))
            print(
                dtype_name,
                mode,
                validations,
                "nativecache",
                native.cached_executables,
                flush=True,
            )


if __name__ == "__main__":
    main()
