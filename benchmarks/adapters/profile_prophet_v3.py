"""Frozen v3 Prophet timing beside mixed, with separate diagnostic profiles."""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
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
    validate_pair,
)
from mixed_snapshot import load_mixed
from profile_models import hardware_snapshot, summarize
from profile_tece64 import event_rows, kernel_owners, shape
from v3_snapshot import load_v3


def scopes(model, label):
    """Add reversible instance and helper annotations only during profiling."""
    descriptors, restorations, classes = {}, [], set()

    def wrapper(fn, key):
        descriptors[key] = {"calls": 0, "first_args": None}

        def annotated(*args, **kwargs):
            row = descriptors[key]
            row["calls"] += 1
            if row["first_args"] is None:
                row["first_args"] = shape(args)
            with torch.profiler.record_function(key):
                return fn(*args, **kwargs)

        return annotated

    class Postprocess:
        def __init__(self, original, key):
            self.original = original
            self.annotated = wrapper(original, key)

        def __call__(self, *args, **kwargs):
            return self.annotated(*args, **kwargs)

    for path, module in model.named_modules():
        plan = getattr(module, "_fasteq_plan", None)
        if plan is not None or type(module).__name__ in (
            "ProphetBlock",
            "TPConvolution",
            "EQFFN",
            "RMSLayerNorm",
            "Gate",
        ):
            original = module.forward
            had_local = "forward" in module.__dict__
            old_local = module.__dict__.get("forward")
            key = "module::" + label + "::" + path
            module.forward = wrapper(original, key)
            descriptors[key]["class"] = type(module).__name__
            descriptors[key]["backend"] = getattr(plan, "backend", None)
            restorations.append((module, "forward", had_local, old_local))
        if plan is not None:
            classes.add(type(plan))
            for name in ("norm", "modulate", "element"):
                helper = getattr(plan, name, None)
                if helper is not None:
                    classes.add(type(helper))
        original = getattr(module, "_fasteq_postprocess", None)
        if original is not None:
            module._fasteq_postprocess = Postprocess(
                original, "detail::" + label + "::ResidualGateNorm::" + path
            )
            restorations.append((module, "_fasteq_postprocess", True, original))
            for name in ("pre", "gate", "post"):
                helper = getattr(original, name, None)
                if helper is not None:
                    classes.add(type(helper))

    # Find the actual loaded helper modules through the frozen plan classes.
    # Do not import live production modules, or affect timed calls.
    norm_modules = set()
    for cls in classes:
        if cls.__name__ == "NormPlan":
            norm_modules.add(sys.modules[cls.__module__])
        if "__call__" in cls.__dict__ and not issubclass(cls, torch.nn.Module):
            original = cls.__dict__["__call__"]
            cls.__call__ = wrapper(original, "detail::" + label + "::" + cls.__name__)
            restorations.append((cls, "__call__", True, original))
    for module in norm_modules:
        if hasattr(module, "_reduce_rows"):
            original = module._reduce_rows
            module._reduce_rows = wrapper(
                original, "detail::" + label + "::Norm_parameter_reduce_rows"
            )
            restorations.append((module, "_reduce_rows", True, original))
        for name in ("_Norm", "_NormBackward"):
            cls = getattr(module, name, None)
            if cls is None:
                continue
            for method in ("forward", "backward"):
                if method not in cls.__dict__:
                    continue
                original = cls.__dict__[method]
                fn = (
                    original.__func__
                    if isinstance(original, staticmethod)
                    else original
                )
                setattr(
                    cls,
                    method,
                    staticmethod(
                        wrapper(fn, "detail::" + label + "::" + name + "_" + method)
                    ),
                )
                restorations.append((cls, method, True, original))

    def undo():
        for obj, attribute, had_local, original in reversed(restorations):
            if had_local:
                setattr(obj, attribute, original)
            else:
                delattr(obj, attribute)

    return descriptors, undo


def kernel_families(kernels):
    groups = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    for row in kernels:
        name = row["name"]
        family = (
            "norm_parameter_reduce_rows"
            if "reduce_rows" in name
            else "gate_double"
            if "gate_double" in name
            else "gate_first"
            if "gate_backward" in name
            else "gate_forward"
            if "gate_forward" in name
            else "norm_double"
            if name == "_double_kernel"
            else "norm_first"
            if name == "_backward_kernel"
            else "norm_forward"
            if name == "_forward_kernel"
            else "other"
        )
        groups[family]["count"] += row["count"]
        groups[family]["duration_us"] += row["duration_us"]
    return dict(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"])
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    assert args.warmup >= 3 and args.rounds >= 3 and args.iterations > 0
    args.output.mkdir(exist_ok=True, parents=True)
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    v3, v3_provenance = load_v3(args.root)
    mixed, mixed_provenance = load_mixed(args.root)
    report = {
        "metadata": {
            "model": "prophet",
            "channels": 64,
            "nodes": 256,
            "neighbors": 12,
            "layers": 2,
            "lmax": 2,
            "cutoffs": [2.0, 4.0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hardware": hardware_snapshot(),
            "v3": v3_provenance,
            "mixed": mixed_provenance,
            "normal_protocol": {
                "rounds": args.rounds,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "order": "alternating per round",
            },
            "scope": "benchmark_models.execute: cloned positions/geometry/model; force_loss includes zero_grad/loss/backward; no optimizer or stress; neighbors prebuilt",
            "profile_note": "One instrumented step per backend/case, separate from normal timing; kernel duration sums are not normal latency; nested annotation GPU totals must not be added.",
        },
        "results": [],
    }
    labels = ("mixed", "v3")
    for dtype_name in ("float32", "float64"):
        dtype = getattr(torch, dtype_name)
        original = build_model("prophet", 64, dtype, 2)
        models = [mixed(original), v3(original)]
        data = build_data(256, 12, dtype)
        for mode in ("energy", "forces", "force_loss"):
            print("CASE", dtype_name, mode, flush=True)
            for model in models:
                configure(model, "prophet", mode)
            assert_exclusive_gpu()
            validation = validate_pair(models, "prophet", data, mode, dtype)
            warmup = []
            for model in models:
                before = time.perf_counter()
                for _ in range(args.warmup):
                    out = execute(model, "prophet", data, mode)
                    del out
                torch.cuda.synchronize()
                warmup.append(time.perf_counter() - before)
            samples = [[], []]
            for block in range(args.rounds):
                assert_exclusive_gpu()
                for i in (0, 1) if block % 2 == 0 else (1, 0):
                    samples[i].append(
                        one_batch(
                            lambda m=models[i], mode=mode, data=data: execute(
                                m, "prophet", data, mode
                            ),
                            args.iterations,
                        )
                    )
                assert_exclusive_gpu()
            timing = {
                label: {
                    "wall_ms": stats([s["wall_ms"] for s in ss]),
                    "cuda_event_ms": stats([s["cuda_event_ms"] for s in ss]),
                    "samples": ss,
                    "warmup_seconds": cold,
                }
                for label, ss, cold in zip(labels, samples, warmup)
            }
            row = {
                "dtype": dtype_name,
                "mode": mode,
                "validation": validation,
                "normal_timing": timing,
                "paired_mixed_over_v3": stats(
                    [
                        x["wall_ms"] / y["wall_ms"]
                        for x, y in zip(samples[0], samples[1])
                    ]
                ),
                "profiles": [],
            }
            print(
                "TIMING",
                dtype_name,
                mode,
                json.dumps(
                    {label: timing[label]["wall_ms"]["median"] for label in labels}
                ),
                flush=True,
            )
            for label, model in zip(labels, models):
                descriptors, undo = scopes(model, label)
                step = "profile_prophet_" + dtype_name + "_" + mode + "_" + label
                assert_exclusive_gpu()
                try:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                        profile_memory=False,
                        with_stack=False,
                    ) as profile:
                        with torch.profiler.record_function(step):
                            out = execute(model, "prophet", data, mode)
                            del out
                        torch.cuda.synchronize()
                finally:
                    undo()
                trace = args.output / (
                    dtype_name + "_" + mode + "_" + label + ".trace.json"
                )
                profile.export_chrome_trace(str(trace))
                summary = summarize(profile, trace, step)
                summary.update(
                    {
                        "backend": label,
                        "shape_groups": event_rows(profile),
                        "kernel_owners": kernel_owners(trace),
                        "module_and_helper_scopes": descriptors,
                        "kernel_families": kernel_families(
                            summary["cuda_kernel_groups"]
                        ),
                    }
                )
                row["profiles"].append(summary)
                print(
                    "PROFILE_DONE",
                    dtype_name,
                    mode,
                    label,
                    summary["cuda_kernel_launch_count"],
                    summary["sum_cuda_kernel_duration_us"],
                    flush=True,
                )
            report["results"].append(row)
            (args.output / "profile.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        del models, original, data
        gc.collect()
        torch.cuda.empty_cache()
    assert_exclusive_gpu()
    report["metadata"]["final_hardware"] = hardware_snapshot()
    (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
