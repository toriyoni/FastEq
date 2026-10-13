"""Check reuse of FastEq's existing FP64 Tensor-Core Linear forward."""

import json
import statistics
from pathlib import Path

import torch
from benchmark_operators import measure
from e3nn import o3

from fasteq.ops.indexed_linear import IndexedLinear


def main():
    torch.set_num_threads(1)
    rows = []
    for channels in (64, 128):
        ir = o3.Irreps([(channels, (degree, (-1) ** degree)) for degree in range(4)])
        desc = o3.Linear(ir, ir, internal_weights=False)
        m = IndexedLinear(desc, layout="ir_mul")
        scales = [float(i.path_weight) for i in desc.instructions]
        for nodes in (128, 1024):
            x = torch.randn(nodes, 16, channels, device="cuda", dtype=torch.float64)
            w = torch.randn(4, channels, channels, device="cuda", dtype=torch.float64)
            ids = torch.zeros(nodes, device="cuda", dtype=torch.int64)
            functions = [
                lambda x=x, w=w, scales=scales: torch.ops.equi_linear.forward(
                    x, w, [1, 3, 5, 7], scales
                ).flatten(1),
                lambda m=m, x=x, w=w, ids=ids: m(
                    x.flatten(1), w.reshape(1, 1, -1), ids
                ),
            ]
            old, new = (fn() for fn in functions)
            torch.testing.assert_close(old, new, atol=2e-11, rtol=2e-10)
            for fn in functions:
                for _ in range(5):
                    fn()
            times = [[], []]
            for batch in range(9):
                for i in (0, 1) if batch % 2 == 0 else (1, 0):
                    _, wall = measure(functions[i], 40)
                    times[i].append(wall)
            row = {
                "dtype": "float64",
                "channels": channels,
                "nodes": nodes,
                "legacy_wall_us": statistics.median(times[0]),
                "general_wall_us": statistics.median(times[1]),
                "legacy_advantage": statistics.median(
                    [b / a for a, b in zip(times[0], times[1], strict=True)]
                ),
                "samples_us": times,
                "max_abs_error": (old - new).abs().max().item(),
            }
            rows.append(row)
            print(row, flush=True)
    target = Path(
        "/public-data/zhouxibo/zxb/tace_prophet_fasteq/performance/native_fasteq/linear/legacy_candidate.json"
    )
    target.write_text(json.dumps({"results": rows}, indent=2))


if __name__ == "__main__":
    main()
