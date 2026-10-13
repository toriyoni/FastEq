"""Warm the real Prophet GateNorm path, then capture one named training adjoint.

Example: ncu --nvtx --nvtx-include 'gate_norm_target/' --kernel-name-base function
--kernel-name regex:'^double_backward$' --launch-count 1 --set full python
probe_gate_norm_v4.py --mode double --targets force_loss --output evidence.json
"""

import argparse
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_models import assert_exclusive_gpu
from e3nn import o3
from fasteq.adapters.fusion import ProphetPostprocess
from fasteq.cuda.triton_launch import _RESIDENT_KERNELS
from prophet.layer_norm import RMSLayerNorm
from prophet.model import Gate


def find_target_node(roots, name):
    """Locate the one probe operation without relying on view-node depth."""
    pending = [root.grad_fn for root in roots if root.grad_fn is not None]
    visited = {}
    selected = []
    while pending:
        node = pending.pop()
        if node is None or id(node) in visited:
            continue
        visited[id(node)] = node
        if node.name() == name:
            selected.append(node)
        pending.extend(child for child, _ in node.next_functions if child is not None)
    assert len(selected) == 1, (name, [node.name() for node in selected])
    return selected[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"])
    )
    parser.add_argument("--mode", choices=("backward", "double"), required=True)
    parser.add_argument(
        "--targets", choices=("training", "force_loss"), default="force_loss"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = [
        args.root / relative
        for relative in (
            "src/FastEq/fasteq/adapters/fusion.py",
            "src/FastEq/fasteq/ops/gate_norm.py",
            "src/FastEq/fasteq/triton/gate_norm.py",
            "src/FastEq/fasteq/ops/gate.py",
            "src/FastEq/fasteq/ops/equivariant_norm.py",
            "src/FastEq/fasteq/cuda/triton_launch.py",
            "src/FastEq/fasteq/cuda/jit.py",
            "src/FastEq/fasteq/cuda/src/jit_runtime.cpp",
        )
    ]

    def hashes():
        return {
            str(path.relative_to(args.root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        }

    frozen = hashes()
    assert_exclusive_gpu()
    dtype = getattr(torch, args.dtype)
    irreps = o3.Irreps("192x0e+64x1o+64x2e")
    out = o3.Irreps("64x0e+64x1o+64x2e")
    gate = Gate(irreps).cuda().to(dtype)
    norm = RMSLayerNorm(out, centering=False).cuda().to(dtype)
    conv = SimpleNamespace(
        gate=gate, layer_norm=None, linear_2=SimpleNamespace(irreps_out=irreps)
    )
    plan = ProphetPostprocess(conv, norm)
    assert plan.fused is not None and plan.dim == 704 and plan.outdim == 576
    torch.manual_seed(751)
    x = torch.randn(args.nodes, 704, device="cuda", dtype=dtype, requires_grad=True)
    skip = torch.randn_like(x, requires_grad=True)
    gamma = tuple(norm.parameters())
    parameters = (x, skip, *gamma)
    seed = torch.randn(args.nodes, 576, device="cuda", dtype=dtype)

    def prepare():
        y = plan(x, skip)
        energy = (y.sin() * seed).sum() / args.nodes + y.square().mean()
        return y, energy

    def first(energy):
        target = parameters if args.targets == "training" else (x,)
        return torch.autograd.grad(energy, target, create_graph=True, retain_graph=True)

    def second(energy, gradients):
        objective = sum(value.square().sum() for value in gradients)
        if args.targets == "force_loss":
            objective = objective + energy.square() / 10
        return torch.autograd.grad(objective, parameters)

    for _ in range(args.warmup):
        _, energy = prepare()
        gradient = first(energy)
        if args.mode == "double":
            second(energy, gradient)
    torch.cuda.synchronize()
    assert_exclusive_gpu()
    assert hashes() == frozen
    y, energy = prepare()
    gradient = first(energy) if args.mode == "double" else None
    target_node = find_target_node(
        gradient if args.mode == "double" else (y,),
        "_GateNormBackwardBackward" if args.mode == "double" else "_GateNormBackward",
    )
    push_threads, pop_threads = [], []

    def push_on_engine_thread(_gradients):
        push_threads.append(threading.get_native_id())
        torch.cuda.nvtx.range_push("gate_norm_target")

    def pop_on_engine_thread(_gradients, _seeds):
        torch.cuda.nvtx.range_pop()
        pop_threads.append(threading.get_native_id())

    handles = (
        target_node.register_prehook(push_on_engine_thread),
        target_node.register_hook(pop_on_engine_thread),
    )
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("gate_norm_step")
    try:
        result = first(energy) if args.mode == "backward" else second(energy, gradient)
    finally:
        torch.cuda.nvtx.range_pop()
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize()
    assert push_threads == pop_threads and len(push_threads) == 1
    assert hashes() == frozen
    kernel_resources = [
        {
            "entry": key[0].__name__,
            "compiled_hash": next(
                resident_key[0]
                for resident_key, resident_kernel in _RESIDENT_KERNELS.items()
                if resident_kernel is kernel
            ),
            "device": str(key[1]),
            "pointer_alignment_16": list(key[4]),
            "registers_per_thread": kernel.registers,
            "local_bytes": kernel.local_bytes,
            "threads": threads,
            "shared_bytes": shared,
            "compiler_kwargs": repr(key[3]),
        }
        for key, value in plan.fused._launchers.items()
        if value is not None
        for kernel, threads, shared in (value,)
    ]
    report = {
        "mode": args.mode,
        "targets": args.targets,
        "dtype": args.dtype,
        "nodes": args.nodes,
        "input_dim": 704,
        "output_dim": 576,
        "gamma_shapes": [list(p.shape) for p in gamma],
        "all_gamma_require_grad": all(p.requires_grad for p in gamma),
        "first_targets": "x,skip,all gamma"
        if args.targets == "training"
        else "x only, with create_graph for later full parameter gradients",
        "second_targets": "x,skip,all gamma; direct gamma seeds present"
        if args.targets == "training"
        else "x,skip,all gamma; force-loss target pattern with missing first gamma seeds",
        "target_nvtx": "gate_norm_target",
        "target_node": target_node.name(),
        "nvtx_engine_thread_ids": push_threads,
        "main_thread_id": threading.get_native_id(),
        "target_kernel": "backward" if args.mode == "backward" else "double_backward",
        "source_sha256": frozen,
        "benchmark_driver_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kernel_resources": kernel_resources,
        "resident_fallback_count": sum(
            value is None for value in plan.fused._launchers.values()
        ),
        "gradient_norms": [value.norm().item() for value in result],
        "note": "First backward includes affine reduction only for training targets; double target filtering must select double_backward, because force-loss also traverses the indirect energy output path.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
