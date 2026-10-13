"""Independent frozen-v4 versus GateNorm candidate full training-chain benchmark."""

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import torch
from benchmark_models import assert_exclusive_gpu, one_batch, stats
from e3nn import o3
from fasteq.cuda.triton_launch import _RESIDENT_KERNELS
from fasteq.ops.gate import GatePlan
from fasteq.ops.gate_norm import GateNormPlan
from prophet.layer_norm import RMSLayerNorm
from prophet.model import Gate


def load_v4(root):
    archive = root / "performance/fasteq_optimization/v4_snapshot"
    hashes = {}
    for relative, name in (
        ("src/FastEq/fasteq/triton/gate_norm.py", "fasteq.triton._gate_norm_v4"),
        ("src/FastEq/fasteq/ops/gate_norm.py", "fasteq.ops._gate_norm_v4"),
    ):
        source = (archive / relative).read_text()
        hashes[relative] = hashlib.sha256(source.encode()).hexdigest()
        if "/ops/" in relative:
            source = source.replace(
                "from ..triton import gate_norm as kernels",
                "from ..triton import _gate_norm_v4 as kernels",
            )
        module = types.ModuleType(name)
        module.__file__ = str(archive / relative)
        module.__package__ = name.rsplit(".", 1)[0]
        sys.modules[name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module.GateNormPlan, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"])
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dtypes", nargs="+", default=["float64"])
    parser.add_argument("--warps", nargs="+", type=int, default=[0])
    parser.add_argument("--modes", nargs="+", default=["backward", "force_loss"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    old, old_hashes = load_v4(args.root)
    dependencies = [
        "src/FastEq/fasteq/ops/gate_norm.py",
        "src/FastEq/fasteq/triton/gate_norm.py",
        "src/FastEq/fasteq/ops/gate.py",
        "src/FastEq/fasteq/ops/equivariant_norm.py",
        "src/FastEq/fasteq/triton/fused_equivariant_gate.py",
        "src/FastEq/fasteq/triton/fused_equivariant_layer_norm.py",
        "src/FastEq/fasteq/cuda/triton_launch.py",
        "src/FastEq/fasteq/cuda/jit.py",
        "src/prophet/prophet/model.py",
        "src/prophet/prophet/layer_norm.py",
    ]

    def source_hashes():
        return {
            name: hashlib.sha256((args.root / name).read_bytes()).hexdigest()
            for name in dependencies
        }

    frozen = source_hashes()
    for name in dependencies[2:]:
        assert (
            frozen[name]
            == hashlib.sha256(
                (
                    args.root / "performance/fasteq_optimization/v4_snapshot" / name
                ).read_bytes()
            ).hexdigest()
        ), name
    assert_exclusive_gpu()
    report = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(),
            "uuid": os.getenv("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "source_sha256": frozen,
            "v4_gate_norm_sha256": old_hashes,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol": vars(args)
            | {"root": str(args.root), "output": str(args.output)},
            "scope": "Positions [N,3] and live [3,DIN] weights -> projected/skip -> GateNorm -> nonlinear energy -> first forces with create_graph -> energy+force loss -> all coordinates, weights and gamma gradients. No optimizer step.",
            "runtime_note": "Shared FastEq mathematical primitives and Triton launcher SHA verified against frozen v4; indexed-linear runtime additions do not enter this chain.",
        },
        "results": [],
    }
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        c, n = args.channels, args.nodes
        ir = o3.Irreps(f"{3 * c}x0e+{c}x1o+{c}x2e")
        out = o3.Irreps(f"{c}x0e+{c}x1o+{c}x2e")
        gate = GatePlan(Gate(ir), "prophet")
        norm = RMSLayerNorm(out, centering=False).cuda().to(dtype)
        plans = [("v4", old(gate, norm))]
        for warps in args.warps:
            plan = GateNormPlan(gate, norm)
            original_launch = plan.launch

            def launch(
                fn, arguments, x, *, original=original_launch, warps=warps, **extra
            ):
                if fn.__name__ != "forward":
                    extra["num_warps"] = warps
                return original(fn, arguments, x, **extra)

            if warps:
                plan.launch = launch
            plans.append((f"candidate_w{warps}" if warps else "production", plan))
        torch.manual_seed(751)
        positions = torch.randn(n, 3, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(3, ir.dim, device="cuda", dtype=dtype, requires_grad=True)
        skip_weight = torch.randn_like(weight, requires_grad=True)
        parameters = (positions, weight, skip_weight, *norm.parameters())
        seed = torch.randn(n, out.dim, device="cuda", dtype=dtype)
        for mode in args.modes:
            assert source_hashes() == frozen

            def execute(
                plan,
                *,
                mode=mode,
                positions=positions,
                weight=weight,
                skip_weight=skip_weight,
                parameters=parameters,
                norm=norm,
                seed=seed,
                n=n,
            ):
                for parameter in parameters:
                    parameter.grad = None
                with torch.set_grad_enabled(mode != "forward"):
                    projected = positions @ weight
                    skip = positions.sin() @ skip_weight / 9
                    y = plan(norm, projected, skip)
                    if mode != "forward":
                        energy = (y.sin() * seed).sum() / n + y.square().mean()
                        if mode == "backward":
                            energy.backward()
                        else:
                            force = -torch.autograd.grad(
                                energy, positions, create_graph=True
                            )[0]
                            (energy.square() / 10 + force.square().sum()).backward()
                return y

            functions = [
                lambda p=plan, execute=execute: execute(p) for _, plan in plans
            ]
            outputs = []
            for function in functions:
                y = function()
                outputs.append(
                    (
                        y.detach().clone(),
                        [
                            p.grad.detach().clone() if p.grad is not None else None
                            for p in parameters
                        ],
                    )
                )
            atol, rtol = (4e-5, 4e-4) if dtype == torch.float32 else (2e-10, 2e-9)
            errors = []
            for output, gradients in outputs:
                torch.testing.assert_close(output, outputs[0][0], atol=atol, rtol=rtol)
                diff = norm2 = 0.0
                for actual, reference in zip(gradients, outputs[0][1]):
                    if reference is None:
                        assert actual is None
                    else:
                        torch.testing.assert_close(
                            actual, reference, atol=atol, rtol=rtol
                        )
                        diff += (
                            (actual.double() - reference.double()).square().sum().item()
                        )
                        norm2 += reference.double().square().sum().item()
                errors.append(
                    {
                        "output_max_abs": (output - outputs[0][0]).abs().max().item(),
                        "gradient_relative_l2": (diff / max(norm2, 1e-300)) ** 0.5,
                    }
                )
            for function in functions:
                for _ in range(args.warmup):
                    function()
            torch.cuda.synchronize()
            samples = [[] for _ in plans]
            for block in range(args.rounds):
                assert_exclusive_gpu()
                assert source_hashes() == frozen
                for index in [(block + i) % len(plans) for i in range(len(plans))]:
                    samples[index].append(one_batch(functions[index], args.iterations))
                assert_exclusive_gpu()
                assert source_hashes() == frozen
            for index, (name, plan) in enumerate(plans):
                row = {
                    "dtype": dtype_name,
                    "nodes": n,
                    "channels": c,
                    "mode": mode,
                    "candidate": name,
                    "validation": errors[index],
                    "samples": samples[index],
                    "wall_ms": stats([r["wall_ms"] for r in samples[index]]),
                    "event_ms": stats([r["cuda_event_ms"] for r in samples[index]]),
                    "paired_v4_over_candidate": stats(
                        [
                            a["wall_ms"] / b["wall_ms"]
                            for a, b in zip(samples[0], samples[index])
                        ]
                    ),
                    "kernel_resources": [
                        {
                            "entry": key[0].__name__,
                            "kwargs": repr(key[3]),
                            "compiled_hash": next(
                                k[0]
                                for k, v in _RESIDENT_KERNELS.items()
                                if v is kernel
                            ),
                            "registers": kernel.registers,
                            "local_bytes": kernel.local_bytes,
                            "threads": threads,
                            "shared_bytes": shared,
                        }
                        for key, value in plan._launchers.items()
                        if value is not None
                        for kernel, threads, shared in (value,)
                    ],
                }
                if args.profile:
                    path = args.output / f"{dtype_name}_{mode}_{name}.trace.json"
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=False,
                    ) as prof:
                        functions[index]()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(path))
                    grouped = {}
                    for event in json.loads(path.read_text())["traceEvents"]:
                        if event.get("cat") == "kernel" and event["name"] in (
                            "forward",
                            "backward",
                            "double_backward",
                            "_tree_reduce_rows",
                            "backward_norm",
                            "double_backward_norm",
                            "gate_transpose_global",
                            "backward_direct",
                            "double_backward_direct",
                        ):
                            group = grouped.setdefault(
                                event["name"], {"count": 0, "duration_us": 0}
                            )
                            group["count"] += 1
                            group["duration_us"] += event["dur"]
                    row["profile_kernel_groups"] = grouped
                report["results"].append(row)
                print(
                    dtype_name,
                    mode,
                    name,
                    row["wall_ms"]["median"],
                    row["paired_v4_over_candidate"]["median"],
                    flush=True,
                )
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
