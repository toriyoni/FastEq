"""Paired original e3nn / first-adapter / native-host Linear benchmark.

The first adapter is loaded from the immutable optimization snapshot and only
renamed at the dispatcher to coexist in one process. CUDA compilation and all
correctness checks finish before alternating timing batches.
"""

import argparse
import json
import os
import statistics
import subprocess
import types
from pathlib import Path

import torch
from benchmark_operators import compare, measure, operation
from e3nn import o3
from eqx.o3.contraction import contract_forward
from eqx.o3.linear import ElementLinear

ROOT = Path("/public-data/zhouxibo/zxb/tace_prophet_fasteq")


def load_first_adapter():
    path = ROOT / "performance/optimization/v1_snapshot/src/tace/eqx/o3/contraction.py"
    module = types.ModuleType("eqx.o3.contraction_linear_v1")
    code = path.read_text().replace("eqx::element_linear", "eqx::element_linear_v1")
    exec(compile(code, str(path), "exec"), module.__dict__)  # noqa: S102 - replay the immutable local v1 source snapshot.
    return module.contract


class Map(torch.nn.Module):
    def __init__(self, linear, weight, indices, backend, first):
        super().__init__()
        self.linear = linear
        self.register_buffer("indices", indices, persistent=False)
        self.weight = torch.nn.Parameter(
            weight[0].clone() if linear.shared_weights else weight.clone()
        )
        self.backend = backend
        self.element = ElementLinear(linear)
        self.first = first

    def forward(self, features):
        if self.backend == "e3nn":
            return self.linear(
                features,
                self.weight
                if self.linear.shared_weights
                else self.weight[self.indices],
            )
        if self.backend == "native":
            return contract_forward(
                self.element.metadata,
                self.indices,
                features,
                self.weight.reshape(-1, 1, self.element.weight_numel),
            )
        return self.first(
            self.element.metadata,
            2,
            self.indices,
            features,
            self.weight.reshape(-1, 1, self.element.weight_numel),
            features.new_empty((0, self.element.output_dim)),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument("--nodes", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--elements", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261007)
    first = load_first_adapter()
    metadata = {
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "device_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "timing": "paired alternating eager wall and CUDA event batches; no profiler; compile/check/warmup excluded",
        "v1": "immutable contraction.py snapshot with renamed dispatcher, same CUDA source",
        "force_loss": "grad only feature input with create_graph=True; force loss then grad feature+weights",
        "gpu_state_start": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv",
            ],
            text=True,
        ),
    }
    rows = []
    for name in args.dtypes:
        dtype = getattr(torch, name)
        for n in args.nodes:
            for elements in args.elements:
                ir = o3.Irreps("64x0e+64x1o+64x2e")
                linear = o3.Linear(
                    ir, ir, internal_weights=False, shared_weights=elements == 1
                ).to("cuda", dtype)
                ids = torch.arange(n, device="cuda") % elements
                weight = torch.randn(
                    elements, linear.weight_numel, device="cuda", dtype=dtype
                )
                models = [
                    Map(linear, weight, ids, backend, first).cuda()
                    for backend in ("e3nn", "v1", "native")
                ]
                inputs = [
                    torch.randn(n, ir.dim, device="cuda", dtype=dtype).requires_grad_()
                    for _ in models
                ]
                for x in inputs[1:]:
                    with torch.no_grad():
                        x.copy_(inputs[0])
                cotangent = torch.randn(n, ir.dim, device="cuda", dtype=dtype)
                for mode in ("forward", "backward", "force_loss"):
                    funcs = [
                        lambda m=m, x=x, cotangent=cotangent, mode=mode: operation(
                            m, [x], cotangent, mode
                        )
                        for m, x in zip(models, inputs, strict=False)
                    ]
                    values = [fn() for fn in funcs]
                    error_v1 = compare(values[0], values[1], dtype)
                    error_native = compare(values[0], values[2], dtype)
                    del values
                    for fn in funcs:
                        for _ in range(5):
                            fn()
                    event, wall = [[] for _ in funcs], [[] for _ in funcs]
                    for batch in range(args.batches):
                        for i in (0, 1, 2) if batch % 2 == 0 else (2, 1, 0):
                            a, b = measure(funcs[i], args.iterations)
                            event[i].append(a)
                            wall[i].append(b)
                    row = {
                        "dtype": name,
                        "n": n,
                        "elements": elements,
                        "channels": 64,
                        "lmax": 2,
                        "mode": mode,
                        "event_us": dict(
                            zip(
                                ("e3nn", "v1", "native"),
                                map(statistics.median, event),
                                strict=False,
                            )
                        ),
                        "wall_us": dict(
                            zip(
                                ("e3nn", "v1", "native"),
                                map(statistics.median, wall),
                                strict=False,
                            )
                        ),
                        "event_samples_us": event,
                        "wall_samples_us": wall,
                        "v1_over_native": statistics.median(
                            [a / b for a, b in zip(wall[1], wall[2], strict=False)]
                        ),
                        "e3nn_over_native": statistics.median(
                            [a / b for a, b in zip(wall[0], wall[2], strict=False)]
                        ),
                        "max_abs_error_v1": error_v1,
                        "max_abs_error_native": error_native,
                    }
                    rows.append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps({"metadata": metadata, "results": rows}, indent=2)
                    )
                    print(
                        f"RESULT {name} N{n} types{elements} {mode}: {row['wall_us']}, v1/native {row['v1_over_native']:.3f}x",
                        flush=True,
                    )
    metadata["gpu_state_end"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv",
        ],
        text=True,
    )
    args.output.write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
