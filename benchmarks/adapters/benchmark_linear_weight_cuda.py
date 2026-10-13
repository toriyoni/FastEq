"""Isolate native shared-weight CUDA reduction from host-plan savings."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from benchmark_operators import compare, measure
from e3nn import o3
from eqx.kernels.cuda import kernels, runtime
from eqx.o3.cuda import shared_weight_source, source
from eqx.o3.linear import ElementLinear, MoEElementLinear
from eqx.utils import parse_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", type=int, default=[128, 1024])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/public-data/zhouxibo/zxb/tace_prophet_fasteq/performance/optimization/linear/shared_weight.json"
        ),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(841)
    # Establish the CUDA primary context before manually loading CUDA modules.
    _cuda_context_tensor = torch.empty(1, device="cuda")
    rows = []
    for name in ("float32", "float64"):
        dtype = getattr(torch, name)
        for n in args.nodes:
            for experts in (1, 3):
                desc = o3.Linear(
                    "64x0e+64x1o+64x2e", "64x0e+64x1o+64x2e", internal_weights=False
                )
                element = (
                    ElementLinear(desc)
                    if experts == 1
                    else MoEElementLinear(desc, experts)
                )
                decoded = parse_metadata(element.metadata)
                specs = [source(decoded, role, dtype) for role in range(3)]
                extra = shared_weight_source(decoded, dtype)
                compiled = kernels(
                    [s[0] for s in specs] + [extra], torch.device("cuda:0")
                )
                original = [compiled[s[0]] for s in specs]
                plan_args = (
                    [s[1] for s in specs],
                    [s[2] for s in specs],
                    [True, False, True],
                    decoded[0],
                    decoded[1],
                )
                plans = [
                    runtime().ElementLinearPlan(ks, *plan_args)
                    for ks in (original, original + [compiled[extra]])
                ]
                ids = torch.zeros(n, dtype=torch.long, device="cuda")
                x = torch.randn(n, decoded[0], device="cuda", dtype=dtype)
                w = torch.randn(1, experts, decoded[2], device="cuda", dtype=dtype)
                dy = torch.randn(n, decoded[1], device="cuda", dtype=dtype)
                functions = [
                    lambda p=p, ids=ids, x=x, w=w, dy=dy: p.contract(1, ids, x, w, dy)
                    for p in plans
                ]
                error = compare(*[fn() for fn in functions], dtype)
                for fn in functions:
                    for _ in range(6):
                        fn()
                event, wall = [[], []], [[], []]
                for batch in range(9):
                    for i in range(2) if batch % 2 == 0 else (1, 0):
                        a, b = measure(functions[i], 40)
                        event[i].append(a)
                        wall[i].append(b)
                row = {
                    "dtype": name,
                    "n": n,
                    "experts": experts,
                    "generic_event_us": statistics.median(event[0]),
                    "aggregated_event_us": statistics.median(event[1]),
                    "generic_wall_us": statistics.median(wall[0]),
                    "aggregated_wall_us": statistics.median(wall[1]),
                    "speedup": statistics.median(
                        [a / b for a, b in zip(event[0], event[1], strict=False)]
                    ),
                    "event_samples_us": event,
                    "wall_samples_us": wall,
                    "max_abs_error": error,
                }
                rows.append(row)
                print(row, flush=True)
    args.output.write_text(
        json.dumps(
            {
                "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
