"""Three-way paired CUDA tensor-product adapter preparation comparisons."""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
from benchmark_operators import build, compare, measure, operation, peak, summary
from e3nn import o3
from v1_snapshot import load_v1

from fasteq.adapters import adapt_model


def make_case(name, n, dtype):
    if name != "uuu_unweighted":
        return build(name, n, dtype)
    ir = o3.Irreps([(64, (l, (-1) ** l)) for l in range(3)])
    paths = [
        (i, j, k, "uuu", False)
        for i, (_, a) in enumerate(ir)
        for j, (_, b) in enumerate(ir)
        for k, (_, c) in enumerate(ir)
        if c in a * b
    ][:10]
    m = o3.TensorProduct(
        ir, ir, ir, paths, internal_weights=False, shared_weights=False
    ).to(device="cuda", dtype=dtype)
    values = [
        torch.randn(n, ir.dim, dtype=dtype, device="cuda").requires_grad_()
        for _ in range(2)
    ]
    return m, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs="+", default=[1024, 8192])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["uvu", "uuu", "o2_ece", "uuu_unweighted"],
        choices=["uvu", "uuu", "o2_ece", "uuu_unweighted"],
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261007)
    root = Path(os.environ["TACE_PROPHET_ROOT"])
    v1_adapt, provenance = load_v1(root)
    result = {
        "metadata": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "batches": args.batches,
            "channels": 64,
            "lmax": 2,
            "tf32": False,
            "v1": provenance,
            "timing": "same process; rotating three-way original/v1/new batches; compilation+warmup excluded",
            "force_loss": "forward scalar objective; first gradients wrt all explicit feature/weight inputs; second wrt inputs/parameters",
        },
        "results": [],
        "layout_results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for name in args.operators:
        for dtype in (torch.float32, torch.float64):
            for n in args.nodes:
                original, values = make_case(name, n, dtype)
                modules = [original, v1_adapt(original), adapt_model(original)]
                cotangent = torch.randn_like(original(*values))
                for mode in ("forward", "backward", "force_loss"):
                    small = [
                        [v[:16].detach().clone().requires_grad_() for v in values]
                        for _ in modules
                    ]
                    original_result = operation(
                        modules[0], small[0], cotangent[:16], mode
                    )
                    errors = [
                        compare(
                            original_result,
                            operation(m, x, cotangent[:16], mode),
                            dtype,
                        )
                        for m, x in zip(modules[1:], small[1:])
                    ]
                    functions = [
                        lambda m=m, values=values, cotangent=cotangent, mode=mode: (
                            operation(m, values, cotangent, mode)
                        )
                        for m in modules
                    ]
                    for fn in functions:
                        for _ in range(args.warmup):
                            fn()
                    samples = [[], [], []]
                    for batch in range(args.batches):
                        order = [(batch + i) % 3 for i in range(3)]
                        if batch % 2:
                            order.reverse()
                        for index in order:
                            event, wall = measure(functions[index], args.iterations)
                            samples[index].append((event, wall))
                    stats = {
                        label: {
                            "wall_us": summary([s[1] for s in group]),
                            "event_us": summary([s[0] for s in group]),
                        }
                        for label, group in zip(
                            ("original", "v1", "optimized"), samples
                        )
                    }
                    ratios = {
                        "v1_over_optimized": summary(
                            [a[1] / b[1] for a, b in zip(samples[1], samples[2])]
                        ),
                        "original_over_optimized": summary(
                            [a[1] / b[1] for a, b in zip(samples[0], samples[2])]
                        ),
                    }
                    row = {
                        "operator": name,
                        "dtype": str(dtype).split(".")[-1],
                        "n": n,
                        "mode": mode,
                        "max_abs_error_v1": errors[0],
                        "max_abs_error_new": errors[1],
                        **stats,
                        "speedup": ratios,
                    }
                    result["results"].append(row)
                    save()
                    print(
                        f"{name} {dtype} N={n} {mode}: "
                        + " -> ".join(
                            f"{stats[label]['wall_us']['median']:.2f}"
                            for label in ("original", "v1", "optimized")
                        )
                        + f" us; v1/new {ratios['v1_over_optimized']['median']:.3f}x",
                        flush=True,
                    )
                if name == "uuu":
                    plans = [m._fasteq_plan for m in modules[1:]]
                    for mode in ("forward", "backward", "force_loss"):

                        def layout_op(plan, values=values, mode=mode):
                            if mode == "forward":
                                with torch.no_grad():
                                    return plan._layout(values[0], 0)
                            output = plan._layout(values[0], 0)
                            loss = output.square().mean()
                            grad = torch.autograd.grad(
                                loss, values[0], create_graph=mode == "force_loss"
                            )[0]
                            if mode == "backward":
                                return grad
                            return torch.autograd.grad(grad.square().mean(), values[0])[
                                0
                            ]

                        functions = [
                            lambda plan=plan, layout_op=layout_op: layout_op(plan)
                            for plan in plans
                        ]
                        compare(functions[0](), functions[1](), dtype)
                        for fn in functions:
                            for _ in range(args.warmup):
                                fn()
                        samples = [[], []]
                        for batch in range(args.batches):
                            for index in (0, 1) if batch % 2 == 0 else (1, 0):
                                samples[index].append(
                                    measure(functions[index], args.iterations)
                                )
                        wall = [
                            statistics.median(x[1] for x in group) for group in samples
                        ]
                        row = {
                            "dtype": str(dtype).split(".")[-1],
                            "n": n,
                            "mode": mode,
                            "v1_wall_us": wall[0],
                            "new_wall_us": wall[1],
                            "speedup": wall[0] / wall[1],
                            "v1_peak_bytes": peak(functions[0]),
                            "new_peak_bytes": peak(functions[1]),
                            "samples_wall_us": [
                                [x[1] for x in group] for group in samples
                            ],
                        }
                        result["layout_results"].append(row)
                        save()
                        print(
                            f"layout {dtype} N={n} {mode}: {wall[0]:.2f}->{wall[1]:.2f}us {wall[0] / wall[1]:.3f}x",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
