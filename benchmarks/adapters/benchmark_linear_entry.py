"""Paired native-role contract versus dedicated forward without placeholder."""

import json
import statistics
from pathlib import Path

import torch
from benchmark_operators import compare, measure
from e3nn import o3
from eqx.o3.contraction import contract, contract_forward
from eqx.o3.linear import ElementLinear


def main():
    torch.set_num_threads(1)
    rows = []
    for name in ("float32", "float64"):
        dtype = getattr(torch, name)
        for n in (128, 1024):
            desc = o3.Linear(
                "64x0e+64x1o+64x2e", "64x0e+64x1o+64x2e", internal_weights=False
            )
            element = ElementLinear(desc)
            ids = torch.zeros(n, dtype=torch.long, device="cuda")
            x = torch.randn(n, element.input_dim, device="cuda", dtype=dtype)
            w = torch.randn(1, 1, desc.weight_numel, device="cuda", dtype=dtype)
            functions = [
                lambda element=element, ids=ids, x=x, w=w: contract(
                    element.metadata, 2, ids, x, w, x.new_empty((0, element.output_dim))
                ),
                lambda element=element, ids=ids, x=x, w=w: contract_forward(
                    element.metadata, ids, x, w
                ),
            ]
            error = compare(*[fn() for fn in functions], dtype)
            for fn in functions:
                for _ in range(5):
                    fn()
            event, wall = [[], []], [[], []]
            for batch in range(9):
                for i in (0, 1) if batch % 2 == 0 else (1, 0):
                    a, b = measure(functions[i], 100)
                    event[i].append(a)
                    wall[i].append(b)
            row = {
                "dtype": name,
                "n": n,
                "old_event_us": statistics.median(event[0]),
                "forward_event_us": statistics.median(event[1]),
                "old_wall_us": statistics.median(wall[0]),
                "forward_wall_us": statistics.median(wall[1]),
                "speedup": statistics.median(
                    [a / b for a, b in zip(wall[0], wall[1], strict=False)]
                ),
                "max_abs_error": error,
                "event_samples_us": event,
                "wall_samples_us": wall,
            }
            rows.append(row)
            print(row, flush=True)
    target = Path(
        "/public-data/zhouxibo/zxb/tace_prophet_fasteq/performance/optimization/linear/forward_entry.json"
    )
    target.write_text(
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
