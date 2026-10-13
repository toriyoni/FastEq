"""Same-math Prophet ablation of v4 GateNorm and EQFFN implementations.

All five variants share weights/data in one process. Component rollback keeps
the current Linear/TP/SH plans; only verified v3 postprocess/FFN plans change.
Normal eager timing excludes validation, compilation, profiler and optimizer.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
from benchmark_models import (
    assert_exclusive_gpu,
    build_data,
    build_model,
    configure,
    execute,
    one_batch,
    stats,
)
from fasteq.adapters import adapt_model
from v3_snapshot import load_v3

LABELS = ("v3", "v4", "v4_gn_v3", "v4_ffn_v3", "v4_gn_ffn_v3")
SOURCE_DIRS = (
    "src/FastEq/fasteq/adapters",
    "src/FastEq/fasteq/ops",
    "src/FastEq/fasteq/triton",
    "src/FastEq/fasteq/cuda",
    "src/tace/tace/models",
    "src/tace/eqx",
    "src/prophet/prophet",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root):
    return {
        str(path.relative_to(root)): _sha(path)
        for relative in SOURCE_DIRS
        for path in sorted((root / relative).rglob("*"))
        if path.is_file()
        and path.suffix in (".py", ".cpp", ".cu", ".h", ".cuh")
        and "__pycache__" not in path.parts
        and "generated" not in str(path)
    }


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _snapshot_hashes(provenance):
    archive = Path(provenance["archive"])
    return {row["path"]: _sha(archive / row["path"])
            for row in provenance["manifest"]["files"]}


def _load_spy(root):
    path = root / "src/FastEq/tests/adapters/eqx_cuda_spy.py"
    spec = importlib.util.spec_from_file_location("_ablation_eqx_cuda_spy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forbid_eqx_cuda, path


def _plan_inventory(model):
    inventory = {"postprocess": {}, "ffn": {}}
    for name, module in model.named_modules():
        post = getattr(module, "_fasteq_postprocess", None)
        if post is not None:
            inventory["postprocess"][name] = {
                "class": type(post).__module__ + "." + type(post).__qualname__,
                "input_dim": post.dim,
                "output_dim": post.outdim,
                "fused": getattr(post, "fused", None) is not None,
            }
        if type(module).__name__ == "EQFFN":
            plan = module._fasteq_plan
            inventory["ffn"][name] = {
                "class": type(plan).__module__ + "." + type(plan).__qualname__,
                "features": module.irreps.dim,
                "gates": module.num_ch,
            }
    assert len(inventory["postprocess"]) == len(inventory["ffn"]) == 4
    return inventory


def _rollback(model, old_postprocess, old_ffn, *, gate_norm, ffn):
    """Replace only non-Module execution plans, preserving parameter objects."""
    parameters = tuple((name, id(value)) for name, value in model.named_parameters())
    state_keys = tuple(model.state_dict())
    protected = {
        name: id(module._fasteq_plan)
        for name, module in model.named_modules()
        if hasattr(module, "_fasteq_plan")
        and type(module).__name__ != "EQFFN"
    }
    counts = {"gate_norm": 0, "ffn": 0}
    for module in model.modules():
        if type(module).__module__ != "prophet.model":
            continue
        if gate_norm and type(module).__name__ == "ProphetBlock":
            assert module._fasteq_fused_norms
            for conv, norm in zip(module.tpconv, module.norm, strict=True):
                conv._fasteq_postprocess = old_postprocess(conv, norm)
                counts["gate_norm"] += 1
        if ffn and type(module).__name__ == "EQFFN":
            module._fasteq_plan = old_ffn(module)
            counts["ffn"] += 1
    assert counts == {"gate_norm": 4 if gate_norm else 0, "ffn": 4 if ffn else 0}
    assert parameters == tuple((name, id(value)) for name, value in model.named_parameters())
    assert state_keys == tuple(model.state_dict())
    for name, module in model.named_modules():
        if name in protected:
            assert id(module._fasteq_plan) == protected[name], name
    assert model._fasteq_backend_version == "fasteq-optimized-v4"
    return counts


def _reference(model, data, mode):
    configure(model, "prophet", mode)
    values = execute(model, "prophet", data, mode)
    output = {key: value.detach().clone() if value is not None else None
              for key, value in values.items()}
    gradients = {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    } if mode == "force_loss" else None
    model.zero_grad(set_to_none=True)
    return output, gradients


def _validate(model, data, mode, dtype, reference, forbid_eqx_cuda):
    expected, gradients = reference
    configure(model, "prophet", mode)
    with forbid_eqx_cuda() as calls:
        values = execute(model, "prophet", data, mode)
    assert not calls
    atol, rtol = (2e-4, 2e-3) if dtype == torch.float32 else (2e-8, 2e-7)
    errors = {"eqx_cuda_spy": "passed"}
    for key, baseline in expected.items():
        if baseline is None:
            assert values[key] is None
        else:
            torch.testing.assert_close(baseline, values[key], atol=atol, rtol=rtol)
            errors[key + "_max_abs"] = (baseline - values[key]).abs().max().item()
    if gradients is not None:
        parameters = dict(model.named_parameters())
        assert tuple(parameters) == tuple(gradients)
        diff = norm = 0.0
        nonzero = 0
        for name, baseline in gradients.items():
            actual = parameters[name].grad
            if baseline is None:
                assert actual is None or actual.count_nonzero() == 0, name
            else:
                assert actual is not None, name
                assert actual.shape == baseline.shape, name
                assert torch.isfinite(actual).all(), name
                nonzero += bool(baseline.count_nonzero())
                diff += (baseline.double() - actual.double()).square().sum().item()
                norm += baseline.double().square().sum().item()
        errors["gradient_relative_l2"] = (diff / max(norm, 1e-300)) ** 0.5
        assert errors["gradient_relative_l2"] < (2e-4 if dtype == torch.float32 else 2e-7)
        errors["all_named_parameters_checked"] = len(parameters)
        errors["reference_nonzero_gradient_parameters"] = nonzero
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return errors


def paired_contributions(samples):
    """Same-round signed latency deltas; do not assume gains are additive."""
    wall = {label: [value["wall_ms"] for value in values]
            for label, values in samples.items()}
    ratios = {
        label + "_over_v4": stats([a / b for a, b in zip(values, wall["v4"], strict=True)])
        for label, values in wall.items() if label != "v4"
    }
    delta = {
        "gate_norm_rollback": [a - b for a, b in zip(wall["v4_gn_v3"], wall["v4"], strict=True)],
        "ffn_rollback": [a - b for a, b in zip(wall["v4_ffn_v3"], wall["v4"], strict=True)],
        "both_rollback": [a - b for a, b in zip(wall["v4_gn_ffn_v3"], wall["v4"], strict=True)],
    }
    delta["interaction_both_minus_individuals"] = [
        both - gn - ffn for both, gn, ffn in zip(
            delta["both_rollback"], delta["gate_norm_rollback"], delta["ffn_rollback"], strict=True
        )
    ]
    return ratios, {name: {"stats": stats(values), "samples_ms": values}
                    for name, values in delta.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cpu-core", type=int)
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float64"),
                        default=["float32", "float64"])
    parser.add_argument("--modes", nargs="+", choices=("energy", "forces", "force_loss"),
                        default=["energy", "forces", "force_loss"])
    args = parser.parse_args()
    assert args.warmup > 0 and args.rounds >= 7 and args.iterations > 0
    assert len(set(args.dtypes)) == len(args.dtypes)
    assert len(set(args.modes)) == len(args.modes)
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    if args.cpu_core is not None:
        os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert_exclusive_gpu()
    frozen = source_hashes(args.root)
    source_digest = _digest(frozen)
    model_evidence = args.root / "performance/fasteq_optimization/models_final/models.json"
    evidence = json.loads(model_evidence.read_text())
    assert evidence["metadata"]["fasteq_source_sha256"] == frozen, (
        "Current production differs from final36 evidence; choose an explicit new experiment"
    )
    v3_adapt, provenance = load_v3(args.root)
    snapshot_hashes = _snapshot_hashes(provenance)
    assert snapshot_hashes == {row["path"]: row["sha256"]
                               for row in provenance["manifest"]["files"]}
    old_postprocess = sys.modules["fasteq_v3.adapters.fusion"].ProphetPostprocess
    old_ffn = sys.modules["fasteq_v3.adapters.programs"].FFNPlan
    forbid_eqx_cuda, spy_path = _load_spy(args.root)
    drivers = {str(path): _sha(path) for path in (
        Path(__file__), Path(__file__).with_name("benchmark_models.py"),
        Path(__file__).with_name("v3_snapshot.py"), spy_path,
    )}

    def guard():
        assert_exclusive_gpu()
        assert source_hashes(args.root) == frozen, "Production sources changed; discard case"
        assert _snapshot_hashes(provenance) == snapshot_hashes, "v3 sources changed"
        assert {path: _sha(Path(path)) for path in drivers} == drivers, "Benchmark code changed"

    report = {
        "metadata": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "model_config": {"model": "prophet", "channels": 64, "nodes": 256,
                             "layers": 2, "lmax": 2, "neighbors": 12, "cutoffs": [2.0, 4.0]},
            "variants": {
                "v3": "complete verified v3 adapter",
                "v4": "complete current v4 adapter",
                "v4_gn_v3": "current v4; four postprocesses use v3 residual/Gate/Norm composition",
                "v4_ffn_v3": "current v4; four EQFFN plans use v3 modulation implementation",
                "v4_gn_ffn_v3": "current v4; both sets of four plans rolled back",
            },
            "component_rollback": "Parameter objects/state keys/current non-FFN plans preserved; complete v3 is a separate control",
            "protocol": {"warmup": args.warmup, "rounds": args.rounds,
                         "iterations": args.iterations,
                         "order": "five-way rotating order, reversed on odd rounds",
                         "timing": "normal eager wall + CUDA-event stream timeline; no profiler/capture"},
            "scope": "geometry + cloned positions + model; force_loss includes zero_grad/loss/backward; no optimizer/stress; prebuilt neighbors",
            "validation": "Every variant first validates physical force-loss and all named parameter gradients against original; per-mode outputs revalidated; strong EQX CUDA spy outside timing",
            "guard": "NVML/current-production SHA/v3 snapshot SHA/driver SHA before and after every round and case",
            "v3": provenance,
            "fasteq_source_sha256": frozen,
            "fasteq_source_sha256_digest": source_digest,
            "benchmark_driver_sha256": drivers,
            "final36_evidence": {"path": str(model_evidence), "sha256": _sha(model_evidence)},
        },
        "preflight_validation": {}, "plan_inventory": {}, "results": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "ablation.json"
    assert not output_path.exists(), "Use a fresh output directory; existing measurements are preserved"

    def save():
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output_path)

    guard()
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        original = build_model("prophet", 64, dtype, 2)
        models = {"v3": v3_adapt(original),
                  **{label: adapt_model(original) for label in LABELS if label != "v3"}}
        for label in LABELS[2:]:
            _rollback(models[label], old_postprocess, old_ffn,
                      gate_norm=label in ("v4_gn_v3", "v4_gn_ffn_v3"),
                      ffn=label in ("v4_ffn_v3", "v4_gn_ffn_v3"))
        state = original.state_dict()
        for label, model in models.items():
            assert tuple(model.state_dict()) == tuple(state)
            assert tuple(dict(model.named_parameters())) == tuple(dict(original.named_parameters()))
            for name, value in model.state_dict().items():
                assert torch.equal(value, state[name]), (label, name)
        data = build_data(256, 12, dtype)
        reference = _reference(original, data, "force_loss")
        report["preflight_validation"][dtype_name] = {
            label: _validate(model, data, "force_loss", dtype, reference, forbid_eqx_cuda)
            for label, model in models.items()
        }
        report["plan_inventory"][dtype_name] = {
            label: _plan_inventory(model) for label, model in models.items()
        }
        del reference
        guard()
        save()
        for mode in args.modes:
            print(f"START Prophet {dtype_name} C64 N256 {mode}", flush=True)
            guard()
            reference = _reference(original, data, mode)
            validation = {label: _validate(model, data, mode, dtype, reference, forbid_eqx_cuda)
                          for label, model in models.items()}
            del reference
            fns = {label: (lambda model=model, data=data, mode=mode: execute(
                model, "prophet", data, mode)) for label, model in models.items()}
            warmup = {}
            for label, fn in fns.items():
                before = time.perf_counter()
                for _ in range(args.warmup):
                    out = fn()
                    del out
                torch.cuda.synchronize()
                warmup[label] = time.perf_counter() - before
            guard()
            samples = {label: [] for label in LABELS}
            orders = []
            for block in range(args.rounds):
                guard()
                order = [LABELS[(block + offset) % len(LABELS)]
                         for offset in range(len(LABELS))]
                if block % 2:
                    order.reverse()
                for label in order:
                    samples[label].append(one_batch(fns[label], args.iterations))
                orders.append(order)
                guard()
            ratios, deltas = paired_contributions(samples)
            metrics = {label: {
                "wall_ms": stats([value["wall_ms"] for value in values]),
                "cuda_event_ms": stats([value["cuda_event_ms"] for value in values]),
                "warmup_seconds": warmup[label], "samples": values,
            } for label, values in samples.items()}
            guard()
            report["results"].append({
                "model": "prophet", "dtype": dtype_name, "nodes": 256, "mode": mode,
                "fasteq_source_sha256_digest": source_digest,
                "validation": validation, "round_orders": orders,
                "metrics": metrics, "paired_speedup": ratios,
                "paired_signed_latency_deltas_ms": deltas,
            })
            save()
            print("DONE " + " / ".join(
                f"{label} {metrics[label]['wall_ms']['median']:.3f}ms" for label in LABELS
            ), flush=True)
        del models, original, data, state, fns
        gc.collect()
        torch.cuda.empty_cache()
    guard()
    assert len(report["results"]) == len(args.dtypes) * len(args.modes)
    report["metadata"]["completed"] = True
    save()


if __name__ == "__main__":
    main()
