"""Paired eager model latency for the exact equivariant CUDA adapters.

Neighbor construction and compilation are excluded. Geometry, model execution,
position cloning and (for force_loss) zero_grad/loss/backward are included.
Weights remain fixed: force_loss is a gradient step, without optimizer.step.
"""

import argparse
import csv
import json
import os
import statistics
import time
from copy import deepcopy
from pathlib import Path

for name in ("EQX", "OEQ", "CUE", "EQT"):
    os.environ["TACE_USE_" + name] = "0"
os.environ.setdefault("FASTEQ_UNIFORM1D_FWD_TUNE", "0")
os.environ.setdefault("FASTEQ_UNIFORM1D_BWD_TUNE", "0")

import numpy as np
import pynvml
import torch

from fasteq.adapters import adapt_model

_NVML_HANDLE = None


def assert_exclusive_gpu():
    """Outside timed regions, reject a GPU with another compute process."""
    global _NVML_HANDLE
    if _NVML_HANDLE is None:
        pynvml.nvmlInit()
        selected = str(torch.cuda.get_device_properties(0).uuid)
        selected = selected if selected.startswith("GPU-") else "GPU-" + selected
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByUUID(selected)
    others = [
        p.pid
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(_NVML_HANDLE)
        if p.pid != os.getpid()
    ]
    if others:
        raise RuntimeError(
            f"GPU became shared with other compute processes: {others}; discard this case"
        )


def build_model(name, channels, dtype, correlation):
    torch.manual_seed(1701)
    if name == "prophet":
        from prophet import Prophet

        return (
            Prophet(
                n_species=2,
                lmax=2,
                hidden_irreps=f"{channels}x0e+{channels}x1o+{channels}x2e",
                n_layers=2,
                radial_basis_size=8,
                radial_mlp_size=channels,
                radial_mlp_layers=2,
                cutoffs=[2.0, 4.0],
                kernel=False,
            )
            .cuda()
            .to(dtype)
        )
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    cfg = deepcopy(DEFAULT_MODEL_CONFIG)
    cfg.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=channels,
        Lmax=2,
        lmax=2,
        mmax=1,
        statistics=[
            {
                "atomic_numbers": [1, 8],
                "avg_num_neighbors": 12.0,
                "atomic_energy": {1: 0.0, 8: 0.0},
            }
        ],
        target_property=["energy", "forces"],
    )
    cfg["atomic_basis"].update(
        type=["cgtp", "so2"] if name == "tece" else "cgtp",
        num_head=4,
        use_radial_rotary_attention=name == "tece",
        use_asymmetric_contraction=True,
    )
    cfg["node_embedding"]["type"] = "linear"
    cfg["radial_basis"].update(
        hidden=[channels, channels], bias=True, apply_cutoff=True
    )
    cfg["readout_emlp"].update(hidden=[16], use_one_body_magmoms=False)
    cfg["scale_shift"]["enable"] = False
    cfg["product_basis"]["correlation"] = correlation
    cfg["layer_norm"].update(
        pre_norm_type="merge_layer_norm", final_norm_type="merge_rms_norm"
    )
    return TensorModel(e3nnTACE(**cfg)).cuda().to(dtype)


def build_data(nodes, neighbors, dtype):
    # One fixed nonperiodic geometry. Distance sorting is outside timed calls.
    g = torch.Generator().manual_seed(2701 + nodes)
    positions = torch.rand(nodes, 3, generator=g, dtype=torch.float64) * 5.0
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    receiver = torch.arange(nodes).repeat_interleave(neighbors)
    sender = distances.topk(neighbors, largest=False).indices.flatten()
    edge = torch.stack((sender, receiver)).cuda()
    species = torch.arange(nodes).remainder(2).cuda()
    positions = positions.cuda().to(dtype)
    batch = torch.zeros(nodes, dtype=torch.long, device="cuda")
    return {
        "positions": positions,
        "species": species,
        "edge_index": edge,
        "edge_attr": torch.zeros(edge.shape[1], 3, device="cuda", dtype=dtype),
        "node_attrs": torch.nn.functional.one_hot(species, 2).to(dtype),
        "edge_shifts": torch.zeros(edge.shape[1], 3, device="cuda", dtype=dtype),
        "lattice": (torch.eye(3, device="cuda", dtype=dtype) * 20.0).unsqueeze(0),
        "batch": batch,
        "ptr": torch.tensor([0, nodes], device="cuda"),
        "fidelity_idx": torch.zeros(1, dtype=torch.long, device="cuda"),
        "n_node": torch.tensor([nodes], device="cuda"),
        "n_edge": torch.tensor([edge.shape[1]], device="cuda"),
    }


def configure(model, name, mode):
    model.train(mode == "force_loss")
    model.zero_grad(set_to_none=True)
    if name != "prophet":
        model.reset_target_property(
            ["energy"] if mode == "energy" else ["energy", "forces"]
        )


def forward(model, name, data, mode):
    positions = data["positions"].detach().clone()
    if name == "prophet":
        if mode == "energy":
            edge = data["edge_index"]
            displacements = positions[edge[0]] - positions[edge[1]]
            energy = model.node_energies(
                displacements, data["species"], edge[0], edge[1]
            )
            return {"energy": energy, "forces": None}
        energy, force, _ = model(
            data["species"],
            positions,
            data["edge_attr"],
            data["edge_index"],
            None,
            data["n_node"],
            data["n_edge"],
            data["batch"],
        )
        return {"energy": energy, "forces": force}
    values = {
        k: data[k]
        for k in (
            "node_attrs",
            "edge_index",
            "edge_shifts",
            "lattice",
            "batch",
            "ptr",
            "fidelity_idx",
        )
    }
    values["positions"] = positions
    out = model(values)
    return {"energy": out["energy"], "forces": out["forces"]}


def execute(model, name, data, mode):
    if mode == "force_loss":
        model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(mode != "energy"):
        output = forward(model, name, data, mode)
        if mode == "force_loss":
            loss = output["energy"].square().mean() + output["forces"].square().mean()
            loss.backward()
    return output


def validate_pair(pair, name, data, mode, dtype):
    values = [execute(m, name, data, mode) for m in pair]
    atol, rtol = (2e-4, 2e-3) if dtype == torch.float32 else (2e-8, 2e-7)
    errors = {}
    for key in ("energy", "forces"):
        if values[0][key] is not None:
            torch.testing.assert_close(
                values[0][key], values[1][key], atol=atol, rtol=rtol
            )
            errors[key] = (values[0][key] - values[1][key]).abs().max().item()
    if mode == "force_loss":
        diff = norm = 0.0
        for (na, a), (nb, b) in zip(
            pair[0].named_parameters(), pair[1].named_parameters()
        ):
            assert na == nb
            if a.grad is None:
                assert b.grad is None or b.grad.count_nonzero() == 0
                continue
            assert b.grad is not None, na
            # Report an aggregate norm; entrywise tiny gradients are unstable
            # in FP32 reductions at larger graph sizes.
            diff += (a.grad.double() - b.grad.double()).square().sum().item()
            norm += a.grad.double().square().sum().item()
        errors["gradient_relative_l2"] = (diff / max(norm, 1e-300)) ** 0.5
        assert errors["gradient_relative_l2"] < (
            2e-4 if dtype == torch.float32 else 2e-7
        ), errors
    for m in pair:
        m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return errors


def one_batch(fn, iterations):
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    before = time.perf_counter()
    start.record()
    for _ in range(iterations):
        result = fn()
        del result
    end.record()
    end.synchronize()
    wall = (time.perf_counter() - before) * 1000 / iterations
    return {"wall_ms": wall, "cuda_event_ms": start.elapsed_time(end) / iterations}


def stats(values):
    return {
        "median": statistics.median(values),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "min": min(values),
        "max": max(values),
    }


def benchmark_pair(pair, name, data, mode, args):
    fns = [lambda m=m: execute(m, name, data, mode) for m in pair]
    cold = []
    for fn in fns:
        t = time.perf_counter()
        for _ in range(args.warmup):
            out = fn()
            del out
        torch.cuda.synchronize()
        cold.append(time.perf_counter() - t)
    samples = [[], []]
    for block in range(args.rounds):
        assert_exclusive_gpu()
        # Alternate backend order to reduce order/clock drift bias.
        for index in (0, 1) if block % 2 == 0 else (1, 0):
            samples[index].append(one_batch(fns[index], args.iterations))
        assert_exclusive_gpu()
    metrics = []
    for index in range(2):
        for m in pair:
            m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fns[index]()
        torch.cuda.synchronize()
        del out
        peak = torch.cuda.max_memory_allocated() - base
        metrics.append(
            {
                "wall_ms": stats([s["wall_ms"] for s in samples[index]]),
                "cuda_event_ms": stats([s["cuda_event_ms"] for s in samples[index]]),
                "peak_allocated_increment_bytes": peak,
                "warmup_seconds": cold[index],
                "samples": samples[index],
            }
        )
    paired = [
        samples[0][i]["wall_ms"] / samples[1][i]["wall_ms"] for i in range(args.rounds)
    ]
    return {
        "baseline": metrics[0],
        "adapted": metrics[1],
        "wall_speedup": metrics[0]["wall_ms"]["median"]
        / metrics[1]["wall_ms"]["median"],
        "paired_wall_speedup": stats(paired),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["tace", "tece", "prophet"])
    parser.add_argument("--nodes", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument(
        "--modes", nargs="+", default=["energy", "forces", "force_loss"]
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert args.warmup > 0 and args.rounds >= 3 and args.iterations > 0
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        raise RuntimeError("Unset CUDA_LAUNCH_BLOCKING before benchmarking")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_threads": torch.get_num_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "execution": "paired eager wall and CUDA-event timeline; no profiler or CUDA Graph",
        "gpu_sharing_guard": "NVML check before and after each paired batch, outside timed regions",
        "training_scope": "zero_grad + energy/forces + loss + backward; optimizer excluded; fixed weights",
        "graph": "prebuilt directed 12 nearest neighbors; geometry inside timed call; no stress",
        "model_config": {
            "layers": 2,
            "lmax": 2,
            "channels": args.channels,
            "correlation": args.correlation,
            "prophet_cutoffs": [2.0, 4.0],
            "tace_cutoff": 4.0,
            "tece_mmax": 1,
            "tece_heads": 4,
        },
        "protocol": {
            "warmup": args.warmup,
            "rounds": args.rounds,
            "iterations_per_round": args.iterations,
        },
    }
    report = {"metadata": metadata, "results": []}
    path = args.output / "models.json"
    if args.resume and path.exists():
        previous = json.loads(path.read_text())
        for key in ("visible_device", "model_config", "protocol", "torch", "cuda"):
            assert previous["metadata"][key] == metadata[key], (
                f"Resume metadata mismatch: {key}"
            )
        report["results"] = previous["results"]
        metadata["resume_note"] = (
            "Previously completed, validated paired cases retained. Only GPU process monitor changed from CLI to cached NVML, outside timed regions."
        )
        metadata["previous_guard"] = previous["metadata"]["gpu_sharing_guard"]
    done = {(r["model"], r["dtype"], r["nodes"], r["mode"]) for r in report["results"]}
    path.write_text(json.dumps(report, indent=2) + "\n")
    for name in args.models:
        for dtype_name in args.dtypes:
            if all(
                (name, dtype_name, n, mode) in done
                for n in args.nodes
                for mode in args.modes
            ):
                continue
            dtype = getattr(torch, dtype_name)
            baseline = build_model(name, args.channels, dtype, args.correlation)
            adapted = adapt_model(baseline)
            pair = [baseline, adapted]
            parameters = sum(p.numel() for p in baseline.parameters())
            for nodes in args.nodes:
                data = build_data(nodes, args.neighbors, dtype)
                for mode in args.modes:
                    if (name, dtype_name, nodes, mode) in done:
                        continue
                    for m in pair:
                        configure(m, name, mode)
                    print(
                        f"START {name} {dtype_name} N={nodes} E={data['edge_index'].shape[1]} {mode}",
                        flush=True,
                    )
                    # This first pass also warms lazy compilation; excluded.
                    assert_exclusive_gpu()
                    t = time.perf_counter()
                    validation = validate_pair(pair, name, data, mode, dtype)
                    validate_seconds = time.perf_counter() - t
                    edge = data["edge_index"]
                    distances = (
                        data["positions"][edge[0]] - data["positions"][edge[1]]
                    ).norm(dim=-1)
                    active_edges = {
                        str(c): int((distances < c).sum())
                        for c in ([2.0, 4.0] if name == "prophet" else [4.0])
                    }
                    row = dict(
                        model=name,
                        dtype=dtype_name,
                        nodes=nodes,
                        edges=data["edge_index"].shape[1],
                        active_edges_by_cutoff=active_edges,
                        channels=args.channels,
                        parameters=parameters,
                        mode=mode,
                        validation=validation,
                        validation_compile_seconds=validate_seconds,
                        **benchmark_pair(pair, name, data, mode, args),
                    )
                    for backend in ("baseline", "adapted"):
                        row[backend]["atoms_per_second"] = (
                            1000 * nodes / row[backend]["wall_ms"]["median"]
                        )
                    report["results"].append(row)
                    path.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        f"DONE {name} {dtype_name} N={nodes} {mode}: "
                        f"{row['baseline']['wall_ms']['median']:.3f} -> {row['adapted']['wall_ms']['median']:.3f} ms "
                        f"({row['wall_speedup']:.3f}x)",
                        flush=True,
                    )
            del pair, baseline, adapted, data
            torch.cuda.empty_cache()
    with (args.output / "models.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "dtype",
                "nodes",
                "edges",
                "channels",
                "mode",
                "baseline_ms",
                "adapted_ms",
                "speedup",
                "baseline_atoms_s",
                "adapted_atoms_s",
            ],
        )
        writer.writeheader()
        for r in report["results"]:
            writer.writerow(
                {
                    **{
                        k: r[k]
                        for k in (
                            "model",
                            "dtype",
                            "nodes",
                            "edges",
                            "channels",
                            "mode",
                        )
                    },
                    "baseline_ms": r["baseline"]["wall_ms"]["median"],
                    "adapted_ms": r["adapted"]["wall_ms"]["median"],
                    "speedup": r["wall_speedup"],
                    "baseline_atoms_s": r["baseline"]["atoms_per_second"],
                    "adapted_atoms_s": r["adapted"]["atoms_per_second"],
                }
            )


if __name__ == "__main__":
    main()
