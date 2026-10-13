"""Same-process original/frozen-v5/current-v6 physical model comparison.

All weights and geometry are identical; no optimizer, profiler or stress is
timed. Compile/validation are outside normal timing. Smoke timings are not a
formal performance result. A shared GPU is rejected by NVML.
"""

import argparse
import atexit
import gc
import json
import os
import time
from pathlib import Path

EXPERIMENTAL_FLAGS = ("FASTEQ_EXPERIMENTAL_LINEAR_SIMT", "FASTEQ_EXPERIMENTAL_NATIVE_MODULATION")
for experimental_flag in EXPERIMENTAL_FLAGS:
    os.environ.setdefault(experimental_flag, "1")
    assert os.environ[experimental_flag] == "1", experimental_flag + " must be1 before importing current plans"

import pynvml
import torch
from benchmark_models import (
    assert_exclusive_gpu,
    build_data,
    build_model,
    execute,
    one_batch,
    stats,
)
from candidate_execution_observers import CandidateExecutionObservers
from paired_model_common import (
    PARAMETER_COUNTS,
    digest,
    load_spy,
    reference,
    sha256,
    source_hashes,
    validate,
)
from v5_snapshot import load_v5

from fasteq.adapters import adapt_model

LABELS = ("original", "v5", "v6")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(PARAMETER_COUNTS), default=["tece", "prophet", "tace"])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float64"), default=["float32", "float64"])
    parser.add_argument("--modes", nargs="+", choices=("energy", "forces", "force_loss"), default=["energy", "forces", "force_loss"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cpu-core", type=int, default=164)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true", help="Validate models/gradients without any timing or warmup")
    parser.add_argument("--allow-shared-validation", action="store_true", help="Explicitly allow sharing only with --validation-only; PyTorch allocator capped at2.5%%")
    parser.add_argument("--execution-evidence", nargs="*", type=Path, default=[], help="Independent current-kernel execution evidence, recorded and SHA-guarded")
    args = parser.parse_args()
    args.root = args.root.resolve()
    assert not args.allow_shared_validation or args.validation_only, "Shared GPUs are allowed only for correctness; normal timing always requires exclusivity"
    assert not (args.validation_only and args.smoke), "Choose correctness-only or exclusive timed loader smoke"
    if args.validation_only:
        args.warmup, args.rounds, args.iterations = 0, 0, 0
    elif args.smoke:
        args.warmup, args.rounds, args.iterations = 2, 3, 1
    else:
        assert args.warmup >= 5 and args.rounds >= 9 and args.iterations >= 3
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    for values in (args.models, args.dtypes, args.modes):
        assert len(values) == len(set(values))
    os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sharing_checks = []
    allocator_limit = None
    if args.allow_shared_validation:
        allocator_limit = int(torch.cuda.get_device_properties(0).total_memory * 0.025)
        assert allocator_limit <= 2.1 * 1024**3, "Shared-validation allocator cap exceeds2.1GiB"
        torch.cuda.set_per_process_memory_fraction(0.025)

    def gpu_guard():
        if not args.allow_shared_validation:
            assert_exclusive_gpu()
            return
        pynvml.nvmlInit()
        selected = str(torch.cuda.get_device_properties(0).uuid)
        selected = selected if selected.startswith("GPU-") else "GPU-" + selected
        handle = pynvml.nvmlDeviceGetHandleByUUID(selected)
        others = sorted(process.pid for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        if process.pid != os.getpid())
        assert torch.cuda.memory_reserved() <= allocator_limit
        sharing_checks.append({"external_compute_pids": others,
                               "allocator_reserved_bytes": torch.cuda.memory_reserved(),
                               "allocator_peak_reserved_bytes": torch.cuda.max_memory_reserved()})

    gpu_guard()
    frozen = source_hashes(args.root)
    source_digest = digest(frozen)
    v5_adapt, provenance = load_v5(args.root)
    forbid_eqx_cuda, spy_path = load_spy(args.root)
    archive, cpp_directory = Path(provenance["archive"]), Path(provenance["compatibility_sources"])
    snapshot_hashes = provenance["manifest"]["files_sha256"]
    drivers = {str(path): sha256(path) for path in (
        Path(__file__), Path(__file__).with_name("v5_snapshot.py"),
        Path(__file__).with_name("paired_model_common.py"),
        Path(__file__).with_name("benchmark_models.py"), spy_path,
        Path(__file__).with_name("candidate_execution_observers.py"),
    )}
    evidence_hashes = {str(path.resolve()): sha256(path) for path in args.execution_evidence}
    observations = CandidateExecutionObservers() if args.validation_only else None
    if observations is not None:
        observations.install()
        atexit.register(observations.close)

    def guard():
        gpu_guard()
        assert source_hashes(args.root) == frozen, "Current production changed; discard case"
        assert {name: sha256(archive / name) for name in snapshot_hashes} == snapshot_hashes, "v5 snapshot changed"
        assert {name: sha256(cpp_directory / name) for name in provenance["compatibility_sha256"]} == provenance["compatibility_sha256"], "Isolated v5 C++ source changed"
        assert {name: sha256(Path(name)) for name in drivers} == drivers, "Benchmark code changed"
        assert sha256(archive / "manifest.json") == provenance["manifest_sha256"]
        assert {name: sha256(Path(name)) for name in evidence_hashes} == evidence_hashes, "Independent execution evidence changed"

    report = {"metadata": {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "status": ("shared_validation_only" if args.allow_shared_validation else "exclusive_validation_only")
                  if args.validation_only else "loader_validation_only" if args.smoke else "formal_paired_models",
        "comparison": "same weights/data/process: original, independent frozen FastEq v5, current FastEq v6",
        "labels": LABELS,
        "requested_cases": {"models": args.models, "dtypes": args.dtypes, "modes": args.modes},
        "model_config": {"channels": 64, "nodes": 256, "neighbors": 12, "layers": 2, "lmax": 2, "correlation": 2},
        "parameter_counts": PARAMETER_COUNTS,
        "protocol": {"warmup": args.warmup, "rounds": args.rounds, "iterations": args.iterations,
                     "order": "three-way rotating order; reverse odd rounds",
                     "timing": "none" if args.validation_only else "normal eager wall and CUDA-event stream timeline; no profiler/capture"},
        "scope": "geometry + cloned positions + model; force_loss includes zero_grad/loss/backward; no optimizer/stress; prebuilt neighbors",
        "validation": "full physical force-loss first; all44/57/81 named parameter keys/counts/finite gradients and reference-None checks; native eager and strong EQX-spy separately; nonlinear force loss covers saved-input/upstream indirect routes",
        "guards": "NVML/current sources/v5 snapshot/isolated CPP/driver SHA before and after each case/round",
        "v5": provenance, "current_source_sha256": frozen, "current_source_sha256_digest": source_digest,
        "benchmark_driver_sha256": drivers,
        "historical_timing_used": False,
        "experimental_flags": {flag: os.environ[flag] for flag in EXPERIMENTAL_FLAGS},
        "independent_execution_evidence_sha256": evidence_hashes,
        "timing_fields_empty": args.validation_only,
        "gpu_sharing_policy": "explicit correctness-only sharing" if args.allow_shared_validation else "NVML exclusive",
        "shared_allocator_fraction": 0.025 if args.allow_shared_validation else None,
        "shared_allocator_limit_bytes": allocator_limit, "sharing_checks": sharing_checks,
        "candidate_execution_observations": observations.report if observations is not None else None,
    }, "preflight_validation": {}, "parameter_names": {}, "results": []}
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / ("validation_only.json" if args.validation_only else "loader_validation.json" if args.smoke else "models.json")
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
            models = {"original": original, "v5": v5_adapt(original), "v6": adapt_model(original)}
            assert models["v5"]._fasteq_backend_version == "fasteq-optimized-v5"
            assert models["v6"]._fasteq_backend_version == "fasteq-optimized-v6"
            convolutions = sum((type(module).__module__, type(module).__name__) == (
                "tace.models._e3nn.tece_oam_rra", "Convolution") for module in models["v5"].modules())
            expected_methods = 2 * convolutions
            assert models["v5"]._fasteq_v5_frozen_method_count == expected_methods
            if name != "tece":
                assert expected_methods == 0
            parameters = dict(original.named_parameters())
            assert len(parameters) == PARAMETER_COUNTS[name]
            state = original.state_dict()
            for label, model in models.items():
                assert tuple(dict(model.named_parameters())) == tuple(parameters), label
                assert tuple(model.state_dict()) == tuple(state), label
                for key, value in model.state_dict().items():
                    assert torch.equal(value, state[key]), (label, key)
                for key, value in model.named_parameters():
                    assert torch.isfinite(value).all(), (label, key)
            data = build_data(256, 12, dtype)
            key = name + "/" + dtype_name
            report["parameter_names"][key] = list(parameters)
            expected = reference(original, name, data, "force_loss")
            report["preflight_validation"][key] = {label: {
                "native_eager_force_loss": validate(model, name, data, "force_loss", dtype, expected),
                "eqx_spy_force_loss": validate(model, name, data, "force_loss", dtype, expected, forbid_eqx_cuda),
            } for label, model in models.items() if label != "original"}
            del expected
            guard()
            save()
            for mode in args.modes:
                print(f"START {name} {dtype_name} N256 {mode}", flush=True)
                guard()
                expected = reference(original, name, data, mode)
                validation = {label: validate(model, name, data, mode, dtype, expected)
                              for label, model in models.items() if label != "original"}
                del expected
                if args.validation_only:
                    guard()
                    report["results"].append({
                        "model": name, "dtype": dtype_name, "nodes": 256, "mode": mode,
                        "current_source_sha256_digest": source_digest,
                        "backend_versions": {label: model._fasteq_backend_version
                                             for label, model in models.items() if label != "original"},
                        "frozen_v5_method_count": models["v5"]._fasteq_v5_frozen_method_count,
                        "expected_frozen_v5_method_count": expected_methods,
                        "so2_convolution_count": convolutions,
                        "validation": validation, "timing_fields_empty": True,
                    })
                    save()
                    print("VALIDATED without timing", flush=True)
                    continue
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
                samples, orders = {label: [] for label in LABELS}, []
                round_source_digests = []
                for block in range(args.rounds):
                    guard()
                    order = [LABELS[(block + i) % len(LABELS)] for i in range(len(LABELS))]
                    if block % 2:
                        order.reverse()
                    for label in order:
                        samples[label].append(one_batch(fns[label], args.iterations))
                    orders.append(order)
                    guard()
                    round_source_digests.append(source_digest)
                metrics = {label: {"wall_ms": stats([s["wall_ms"] for s in values]),
                                   "cuda_event_ms": stats([s["cuda_event_ms"] for s in values]),
                                   "samples": values, "warmup_seconds": warmup[label]}
                           for label, values in samples.items()}
                ratios = {left + "_over_" + right: stats([
                    x["wall_ms"] / y["wall_ms"] for x, y in zip(samples[left], samples[right], strict=True)
                ]) for left, right in (("original", "v6"), ("v5", "v6"), ("original", "v5"))}
                guard()
                report["results"].append({
                    "model": name, "dtype": dtype_name, "nodes": 256, "mode": mode,
                    "current_source_sha256_digest": source_digest,
                    "backend_versions": {label: model._fasteq_backend_version
                                         for label, model in models.items() if label != "original"},
                    "frozen_v5_method_count": models["v5"]._fasteq_v5_frozen_method_count,
                    "expected_frozen_v5_method_count": expected_methods,
                    "so2_convolution_count": convolutions,
                    "validation": validation, **metrics, "paired_speedup": ratios,
                    "round_orders": orders, "round_source_sha256_digests": round_source_digests,
                })
                save()
                print("DONE " + " / ".join(
                    f"{label} {metrics[label]['wall_ms']['median']:.3f}ms" for label in LABELS
                ), flush=True)
            del models, original, data, parameters, state
            if not args.validation_only:
                del fns
            gc.collect()
            torch.cuda.empty_cache()
    guard()
    assert len(report["results"]) == len(args.models) * len(args.dtypes) * len(args.modes)
    if observations is not None:
        observations.close()
    report["metadata"]["completed"] = True
    save()


if __name__ == "__main__":
    main()
