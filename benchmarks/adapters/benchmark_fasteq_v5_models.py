"""Interleaved original/frozen-v4/current-v5 exact equivariant model timing.

Geometry/position clones and force-loss zero_grad/loss/backward are included.
Compilation, neighbor construction, validation and optimizer steps are excluded.
The smoke protocol validates loading/gradients; its timing is not formal evidence.
"""

import argparse
import gc
import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from ablate_prophet_v4 import _load_spy, source_hashes
from benchmark_models import (
    assert_exclusive_gpu,
    build_data,
    build_model,
    configure,
    execute,
    one_batch,
    stats,
)
from v4_snapshot import load_v4

from fasteq.adapters import adapt_model

LABELS = ("original", "v4", "v5")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fused16_policy(model):
    """Same math and parameters: disable optimized adjoints, use old fusion16."""
    parameters = tuple((name, id(value)) for name, value in model.named_parameters())
    plans = {name: id(module._fasteq_plan) for name, module in model.named_modules()
             if hasattr(module, "_fasteq_plan")}
    count, large = 0, 0
    dtype = next(model.parameters()).dtype
    for module in model.modules():
        post = getattr(module, "_fasteq_postprocess", None)
        plan = getattr(post, "fused", None)
        if plan is None:
            continue
        original = plan.launch
        predicates = [name for name in ("split_adjoints", "rematerialize_adjoints")
                      if hasattr(plan, name)]
        assert predicates, "Optimized adjoint predicate changed; update explicit fused16 policy"
        for predicate in predicates:
            setattr(plan, predicate, lambda x: False)

        def launch(fn, arguments, x, _original=original, _dim=plan.dim, **extra):
            name = getattr(fn, "__name__", getattr(fn, "name", ""))
            if (name in ("backward", "double_backward") and x.dtype == torch.float64
                    and _dim >= 512 and x.shape[0] >= 32):
                extra["num_warps"] = 16
            return _original(fn, arguments, x, **extra)

        plan.launch = launch
        count += 1
        large += dtype == torch.float64 and plan.dim >= 512
    assert parameters == tuple((name, id(value)) for name, value in model.named_parameters())
    assert plans == {name: id(module._fasteq_plan) for name, module in model.named_modules()
                     if hasattr(module, "_fasteq_plan")}
    if type(model).__module__ == "prophet.model":
        assert count == 4
    return {"disabled_optimized_adjoints": count, "large_fp64_warp16_plans": large,
            "other_norms_warp4_plans": count - large,
            "selector": "FP64,input_dim>=512,N>=32; backward/double_backward only; forward4"}


def _reference(model, name, data, mode):
    configure(model, name, mode)
    output = execute(model, name, data, mode)
    values = {key: value.detach().clone() if value is not None else None
              for key, value in output.items()}
    gradients = {key: value.grad.detach().clone() if value.grad is not None else None
                 for key, value in model.named_parameters()} if mode == "force_loss" else None
    model.zero_grad(set_to_none=True)
    return values, gradients


def _validate(model, name, data, mode, dtype, reference, guard=None):
    """Keep established tolerance, checking every named gradient with exact keys."""
    expected, gradients = reference
    configure(model, name, mode)
    with guard() if guard is not None else nullcontext():
        actual = execute(model, name, data, mode)
    atol, rtol = (2e-4, 2e-3) if dtype == torch.float32 else (2e-8, 2e-7)
    errors = {}
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None, key
        else:
            torch.testing.assert_close(value, actual[key], atol=atol, rtol=rtol)
            errors[key] = (value - actual[key]).abs().max().item()
    if gradients is not None:
        parameters = dict(model.named_parameters())
        assert tuple(parameters) == tuple(gradients), "Named parameter key/order mismatch"
        assert len(parameters) == len(gradients)
        assert len(parameters) == 81 if name == "prophet" else len(parameters) > 0
        difference = magnitude = 0.0
        nonzero, reference_none, actual_none = [], [], []
        for key, parameter in parameters.items():
            baseline, value = gradients[key], parameter.grad
            assert torch.isfinite(parameter).all(), key
            if value is not None:
                assert torch.isfinite(value).all(), key
            if baseline is None:
                reference_none.append(key)
                assert value is None or value.count_nonzero() == 0, key
            else:
                assert torch.isfinite(baseline).all(), key
                assert value is not None and value.shape == baseline.shape, key
                difference += (value.double() - baseline.double()).square().sum().item()
                magnitude += baseline.double().square().sum().item()
                if baseline.count_nonzero():
                    nonzero.append(key)
            if value is None:
                actual_none.append(key)
        errors["gradient_relative_l2"] = (difference / max(magnitude, 1e-300)) ** 0.5
        assert errors["gradient_relative_l2"] < (2e-4 if dtype == torch.float32 else 2e-7), errors
        errors["named_parameters_checked"] = len(parameters)
        errors["reference_nonzero_gradient_parameters"] = len(nonzero)
        errors["reference_none_parameters"] = reference_none
        errors["actual_none_parameters"] = actual_none
        errors["all_parameter_values_and_present_gradients_finite"] = True
    if guard is not None:
        errors["eqx_cuda_spy"] = "passed"
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("tace", "tece", "prophet"),
                        default=["tece", "prophet", "tace"])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float64"),
                        default=["float32", "float64"])
    parser.add_argument("--modes", nargs="+", choices=("energy", "forces", "force_loss"),
                        default=["energy", "forces", "force_loss"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cpu-core", type=int, default=164)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fused16-extra", action="store_true",
                        help="Add same-math v5 old-fusion16 for large FP64 adjoints; other norms4; pilot only")
    parser.add_argument("--gate-norm-policy", choices=("fused16",),
                        help="Alias for --fused16-extra")
    args = parser.parse_args()
    args.fused16_extra = args.fused16_extra or args.gate_norm_policy == "fused16"
    labels = (*LABELS, "v5_fused16") if args.fused16_extra else LABELS
    if args.smoke:
        args.warmup, args.rounds, args.iterations = 2, 3, 1
    else:
        assert args.warmup >= 5 and args.rounds >= 9 and args.iterations >= 3
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    for values in (args.models, args.dtypes, args.modes):
        assert len(values) == len(set(values))
    os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert_exclusive_gpu()
    frozen = source_hashes(args.root)
    source_digest = _digest(frozen)
    v4_adapt, provenance = load_v4(args.root)
    forbid_eqx_cuda, spy_path = _load_spy(args.root)
    archive = Path(provenance["archive"])
    snapshot_hashes = provenance["manifest"]["files_sha256"]
    cpp_directory = Path(provenance["compatibility_sources"])
    drivers = {str(path): _sha(path) for path in (
        Path(__file__), Path(__file__).with_name("v4_snapshot.py"),
        Path(__file__).with_name("benchmark_models.py"),
        Path(__file__).with_name("ablate_prophet_v4.py"), spy_path,
    )}

    def guard():
        assert_exclusive_gpu()
        assert source_hashes(args.root) == frozen, "Current production changed; discard case"
        assert {name: _sha(archive / name) for name in snapshot_hashes} == snapshot_hashes, "v4 snapshot changed"
        assert {name: _sha(cpp_directory / name) for name in provenance["compatibility_sha256"]} == provenance["compatibility_sha256"], "v4 runtime compatibility copy changed"
        assert {name: _sha(Path(name)) for name in drivers} == drivers, "Benchmark driver changed"
        assert _sha(archive / "manifest.json") == provenance["manifest_sha256"]

    report = {"metadata": {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "status": "loader_validation_only" if args.smoke else "policy_pilot" if args.fused16_extra else "formal_paired_models",
        "comparison": "same weights/data/process: original, independent frozen FastEq v4, current FastEq v5",
        "model_config": {"channels": 64, "nodes": 256, "neighbors": 12, "layers": 2,
                         "lmax": 2, "correlation": 2},
        "protocol": {"warmup": args.warmup, "rounds": args.rounds,
                     "iterations": args.iterations,
                     "order": f"{len(labels)}-way rotating order; reverse odd rounds",
                     "timing": "normal eager wall and CUDA-event stream timeline, no profiler/capture"},
        "scope": "geometry + cloned positions + model; force_loss includes zero_grad/loss/backward; no optimizer/stress; prebuilt neighbors",
        "validation": "full physical force-loss first; all named parameter keys/counts/finite gradients; original reference; both normal native eager and strong EQX-spy paths; nonlinear force loss exercises saved-input/upstream indirect routes",
        "guards": "NVML/currentsource/v4snapshot/isolatedCPP/driver SHA before and after each case/round",
        "v4": provenance, "current_source_sha256": frozen,
        "current_source_sha256_digest": source_digest,
        "benchmark_driver_sha256": drivers,
        "labels": labels,
        "fused16_extra": "per-instance old backward/double_backward16warps only FP64,input_dim>=512,N>=32; other norms/forward4; optimized adjoint switchesFalse; other plans/parameters unchanged" if args.fused16_extra else None,
    }, "preflight_validation": {}, "parameter_names": {}, "results": []}
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / ("loader_validation.json" if args.smoke else "policy_pilot.json" if args.fused16_extra else "models.json")
    assert not output_path.exists(), "Use a fresh output directory; existing evidence is preserved"

    def save():
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output_path)

    guard()
    for name in args.models:
        for dtype_name in args.dtypes:
            print(f"BUILD {name} {dtype_name}", flush=True)
            dtype = getattr(torch, dtype_name)
            original = build_model(name, 64, dtype, 2)
            models = {"original": original, "v4": v4_adapt(original),
                      "v5": adapt_model(original)}
            if args.fused16_extra:
                models["v5_fused16"] = adapt_model(original)
                policy_count = _fused16_policy(models["v5_fused16"])
                report["metadata"].setdefault("fused16_plan_counts", {})[name + "/" + dtype_name] = policy_count
                assert models["v5_fused16"]._fasteq_backend_version == "fasteq-optimized-v5"
            assert models["v4"]._fasteq_backend_version == "fasteq-optimized-v4"
            assert models["v5"]._fasteq_backend_version == "fasteq-optimized-v5"
            so2_convolutions = sum(
                (type(module).__module__, type(module).__name__) == (
                    "tace.models._e3nn.tece_oam_rra", "Convolution"
                ) for module in models["v4"].modules()
            )
            expected_frozen_methods = 2 * so2_convolutions
            assert models["v4"]._fasteq_v4_frozen_method_count == expected_frozen_methods
            if name != "tece":
                assert expected_frozen_methods == 0
            parameters = dict(original.named_parameters())
            if name == "prophet":
                assert len(parameters) == 81
            for label, model in models.items():
                assert tuple(dict(model.named_parameters())) == tuple(parameters), label
                assert tuple(model.state_dict()) == tuple(original.state_dict()), label
                for key, value in model.state_dict().items():
                    assert torch.equal(value, original.state_dict()[key]), (label, key)
                for key, value in model.named_parameters():
                    assert torch.isfinite(value).all(), (label, key)
            data = build_data(256, 12, dtype)
            key = name + "/" + dtype_name
            report["parameter_names"][key] = list(parameters)
            reference = _reference(original, name, data, "force_loss")
            report["preflight_validation"][key] = {
                label: {
                    "native_eager_force_loss": _validate(model, name, data, "force_loss", dtype, reference),
                    "eqx_spy_force_loss": _validate(model, name, data, "force_loss", dtype, reference, forbid_eqx_cuda),
                } for label, model in models.items() if label != "original"
            }
            del reference
            guard()
            save()
            for mode in args.modes:
                print(f"START {name} {dtype_name} N256 {mode}", flush=True)
                guard()
                reference = _reference(original, name, data, mode)
                validation = {label: _validate(model, name, data, mode, dtype, reference)
                              for label, model in models.items() if label != "original"}
                del reference
                fns = {label: (lambda model=model, name=name, data=data, mode=mode: execute(
                    model, name, data, mode)) for label, model in models.items()}
                warmup = {}
                for label, fn in fns.items():
                    before = time.perf_counter()
                    for _ in range(args.warmup):
                        out = fn()
                        del out
                    torch.cuda.synchronize()
                    warmup[label] = time.perf_counter() - before
                guard()
                samples, orders = {label: [] for label in labels}, []
                for block in range(args.rounds):
                    guard()
                    order = [labels[(block + i) % len(labels)] for i in range(len(labels))]
                    if block % 2:
                        order.reverse()
                    for label in order:
                        samples[label].append(one_batch(fns[label], args.iterations))
                    orders.append(order)
                    guard()
                metrics = {label: {"wall_ms": stats([s["wall_ms"] for s in values]),
                                   "cuda_event_ms": stats([s["cuda_event_ms"] for s in values]),
                                   "samples": values, "warmup_seconds": warmup[label]}
                           for label, values in samples.items()}
                ratio_pairs = [("original", "v5"), ("v4", "v5"), ("original", "v4")]
                if args.fused16_extra:
                    ratio_pairs.extend((("v5_fused16", "v5"), ("v4", "v5_fused16"),
                                        ("original", "v5_fused16")))
                ratios = {left + "_over_" + right: stats([
                    x["wall_ms"] / y["wall_ms"] for x, y in zip(samples[left], samples[right], strict=True)
                ]) for left, right in ratio_pairs}
                guard()
                report["results"].append({
                    "model": name, "dtype": dtype_name, "nodes": 256, "mode": mode,
                    "current_source_sha256_digest": source_digest,
                    "backend_versions": {label: model._fasteq_backend_version
                                         for label, model in models.items() if label != "original"},
                    "frozen_v4_method_count": models["v4"]._fasteq_v4_frozen_method_count,
                    "expected_frozen_v4_method_count": expected_frozen_methods,
                    "so2_convolution_count": so2_convolutions,
                    "validation": validation, **metrics, "paired_speedup": ratios,
                    "round_orders": orders,
                })
                save()
                print("DONE " + " / ".join(
                    f"{label} {metrics[label]['wall_ms']['median']:.3f}ms" for label in labels
                ), flush=True)
            del models, original, data, parameters, fns
            gc.collect()
            torch.cuda.empty_cache()
    guard()
    assert len(report["results"]) == len(args.models) * len(args.dtypes) * len(args.modes)
    report["metadata"]["completed"] = True
    save()


if __name__ == "__main__":
    main()
