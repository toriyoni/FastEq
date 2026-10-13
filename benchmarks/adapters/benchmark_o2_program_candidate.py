"""Historical mixed-backend comparison of O(2) CUDA versus EQX expressions.

No production adapter is replaced. Shared and expanded weight programs expose
the tradeoff between compact forward weights and reduced parameter adjoints.
"""

import argparse
import copy
import json
import os
import statistics
from pathlib import Path

import torch
from benchmark_operators import build, compare, measure, operation, summary
from eqx.conv.edge import evaluate
from eqx.o2 import TensorProduct
from mixed_snapshot import load_mixed

adapt_model, _ = load_mixed(Path(os.environ["TACE_PROPHET_ROOT"]))
from fasteq_mixed.cache import indices
from fasteq_mixed.programs import Program, finish


class Candidate(torch.nn.Module):
    def __init__(self, module, shared):
        super().__init__()
        self.module = module
        self.shared = shared
        plan = module._fasteq_plan
        self.dims = plan.sizes
        self.weight_numel = plan.weight_numel
        p = Program()
        weight = p.input(0, "shared" if shared else "edge", plan.wsize)
        x = p.input(1, "edge", plan.sizes[0])
        y = p.input(2, "edge", plan.sizes[1])
        dummy = p.constant(plan.sizes[2], 0)
        out = p.product([weight, x, y, dummy], (plan.width, plan.dims, plan.rows), 3)
        self.metadata = finish(p, out)

    def forward(self, x, y, weight=None):
        weight = self.module._get_weights(weight)
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        n = x.shape[0]
        values = [
            weight.reshape(1, -1) if self.shared else weight.expand(n, -1),
            x.expand(*shape, self.dims[0]).reshape(n, self.dims[0]),
            y.expand(*shape, self.dims[1]).reshape(n, self.dims[1]),
        ]
        ids = indices(x, n)
        return evaluate(self.metadata, values, ids, ids, n)[0].reshape(
            *shape, self.dims[2]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-batched", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261007)
    report = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "scope": "bench-only, original production ECE vs vector Program.product; no production module/backend changed",
            "modes": "shared vector weight vs expanded vector weight; same parameters and CUDA data",
            "channels": 64,
            "precision_tolerance_unchanged": True,
            "timing": "paired rotating three-way wall/event eager batches, compile and warmup excluded",
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels = (
        ("current", "candidate_edge")
        if args.external_batched
        else ("current", "candidate_shared", "candidate_expanded")
    )
    for dtype in (torch.float32, torch.float64):
        for n in (1024, 8192):
            reference, inputs = build("o2_ece", n, dtype)
            if args.external_batched:
                reference = TensorProduct(
                    "64x1me",
                    "64x1me",
                    "64x0ee+64x0oe+64x2me",
                    [(0, 0, i, "uuu", True) for i in range(3)],
                    internal_weights=False,
                    shared_weights=False,
                ).to(device="cuda", dtype=dtype)
                inputs.append(
                    torch.randn(
                        n, reference.weight_numel, device="cuda", dtype=dtype
                    ).requires_grad_()
                )
            current = adapt_model(reference)
            modules = (
                [current, Candidate(copy.deepcopy(current), False)]
                if args.external_batched
                else [
                    current,
                    Candidate(copy.deepcopy(current), True),
                    Candidate(copy.deepcopy(current), False),
                ]
            )
            seed = torch.randn_like(reference(*inputs))
            for mode in ("forward", "backward", "force_loss"):
                xs = [
                    [x[:16].detach().clone().requires_grad_() for x in inputs]
                    for _ in range(1 + len(modules))
                ]
                expected = operation(reference, xs[0], seed[:16], mode)
                errors = [
                    compare(expected, operation(m, values, seed[:16], mode), dtype)
                    for m, values in zip(modules, xs[1:])
                ]
                functions = [
                    lambda m=m, inputs=inputs, seed=seed, mode=mode: operation(
                        m, inputs, seed, mode
                    )
                    for m in modules
                ]
                for fn in functions:
                    for _ in range(5):
                        fn()
                samples = [[] for _ in modules]
                for batch in range(args.batches):
                    order = [(batch + i) % len(modules) for i in range(len(modules))]
                    if batch % 2:
                        order.reverse()
                    for index in order:
                        samples[index].append(
                            measure(functions[index], args.iterations)
                        )
                stats = {
                    label: {
                        "wall_us": summary([x[1] for x in group]),
                        "event_us": summary([x[0] for x in group]),
                    }
                    for label, group in zip(labels, samples)
                }
                row = {
                    "dtype": str(dtype).split(".")[-1],
                    "n": n,
                    "mode": mode,
                    "errors_vs_original": dict(zip(labels, errors)),
                    **stats,
                    "speedup": {
                        label: statistics.median(
                            a[1] / b[1] for a, b in zip(samples[0], samples[i])
                        )
                        for i, label in enumerate(labels[1:], 1)
                    },
                }
                report["results"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"{dtype} N={n} {mode}: "
                    + " -> ".join(
                        f"{stats[label]['wall_us']['median']:.2f}us" for label in labels
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
