"""CPU-only provenance, raw-pair statistics and named-gradient record audit.

Never imports Torch, initializes CUDA, reruns models, or changes source/results.
"""

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from pathlib import Path

SOURCE_DIRS = ("src/FastEq/fasteq/adapters", "src/FastEq/fasteq/ops", "src/FastEq/fasteq/triton",
               "src/FastEq/fasteq/cuda", "src/tace/tace/models", "src/tace/eqx", "src/prophet/prophet")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] * (upper - position) + values[upper] * (position - lower) if lower != upper else values[lower]


def stats(values):
    return {"median": statistics.median(values), "q25": percentile(values, 0.25),
            "q75": percentile(values, 0.75), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    initial_input_sha = sha(args.input)
    report = json.loads(args.input.read_text())
    metadata = report["metadata"]
    if not metadata.get("completed"):
        raise SystemExit("Model run is still incomplete; no audit result written.")
    checks, failures = [], []

    def check(name, condition, **details):
        checks.append({"check": name, "passed": bool(condition), **details})
        if not condition:
            failures.append(name)

    check("formal_protocol", metadata["status"] == "formal_paired_models" and metadata["labels"] == ["original", "v4", "v5"]
          and metadata["protocol"]["rounds"] == 9 and metadata["protocol"]["warmup"] >= 5
          and metadata["protocol"]["iterations"] >= 3 and metadata["fused16_extra"] is None)
    current = {str(p.relative_to(root)): sha(p) for relative in SOURCE_DIRS for p in sorted((root / relative).rglob("*"))
               if p.is_file() and p.suffix in (".py", ".cpp", ".cu", ".h", ".cuh")
               and "__pycache__" not in p.parts and "generated" not in str(p)}
    check("current_source_actual_files", current == metadata["current_source_sha256"], count=len(current))
    check("current_source_digest", digest(current) == metadata["current_source_sha256_digest"])
    drivers = {name: sha(Path(name)) for name in metadata["benchmark_driver_sha256"]}
    check("actual_driver_files", drivers == metadata["benchmark_driver_sha256"], count=len(drivers))
    v4 = metadata["v4"]
    archive = Path(v4["archive"])
    manifest = json.loads((archive / "manifest.json").read_text())
    check("actual_v4_manifest", manifest == v4["manifest"] and sha(archive / "manifest.json") == v4["manifest_sha256"])
    archived = {name: sha(archive / name) for name in manifest["files_sha256"]}
    check("all_archived_source_files", archived == manifest["files_sha256"], count=len(archived))
    check("archived_benchmark_drivers", {name: sha(archive / "benchmark_driver" / name)
          for name in manifest["benchmark_driver_sha256"]} == manifest["benchmark_driver_sha256"])
    cpp_directory = Path(v4["compatibility_sources"])
    cpp_hashes = {}
    transformations = (("LinearPlan", "V4LinearPlan"), ("LinearDerivative", "V4LinearDerivative"),
                       ("bind_indexed_linear", "bind_v4_indexed_linear"), ("bind_norm", "bind_v4_norm"))
    for name in v4["compatibility_sha256"]:
        expected = (archive / "src/FastEq/fasteq/cuda/src" / name).read_text().replace("fasteq_jit", "fasteq_v4_jit")
        for before, after in transformations:
            expected = re.sub(r"\b" + before + r"\b", after, expected)
        actual = (cpp_directory / name).read_text()
        cpp_hashes[name] = sha(cpp_directory / name)
        check("isolated_cpp_namespace_only::" + name, actual == expected and hashlib.sha256(expected.encode()).hexdigest() == v4["compatibility_sha256"][name])
    check("all_isolated_cpp_actual_files", cpp_hashes == v4["compatibility_sha256"])
    check("loaded_snapshot_source_entries", all(row["sha256"] == archived[row["path"]] for row in v4["loaded"]), count=len(v4["loaded"]))
    check("live_shared_dependencies_unchanged", all(sha(root / row["path"]) == row["sha256"] for row in v4["shared_dependencies"]), count=len(v4["shared_dependencies"]))
    check("independent_frozen_runtime", v4["extension_name"] == "fasteq_v4_cuda_jit" and v4["namespace_changes_only"]
          and "independent frozen v4 C++" in v4["native_runtime"])
    check("minimal_model_method_binding", v4["bound_model_classes"] == ["tace.models._e3nn.tece_oam_rra.Convolution"]
          and v4["bound_model_methods"] == ["forward", "rotary_score"])
    keys = {(case["model"], case["dtype"], case["mode"]) for case in report["results"]}
    expected_keys = set(itertools.product(("tace", "tece", "prophet"), ("float32", "float64"), ("energy", "forces", "force_loss")))
    check("complete_18_case_grid", keys == expected_keys and len(report["results"]) == 18)
    named_counts = {"tace": 44, "tece": 57, "prophet": 81}
    named_records = []

    def validate_record(key, backend, context, value):
        model, dtype = key.split("/")
        names = report["parameter_names"][key]
        missing = value["reference_none_parameters"]
        check(f"named_parameter_record::{key}::{backend}::{context}",
              len(names) == len(set(names)) == named_counts[model] == value["named_parameters_checked"]
              and len(missing) == len(set(missing)) and set(missing).issubset(names)
              and value["reference_nonzero_gradient_parameters"] == len(names) - len(missing)
              and value["actual_none_parameters"] == missing
              and value["all_parameter_values_and_present_gradients_finite"] is True
              and math.isfinite(value["gradient_relative_l2"])
              and value["gradient_relative_l2"] < (2e-4 if dtype == "float32" else 2e-7)
              and (context != "eqx_spy_force_loss" or value["eqx_cuda_spy"] == "passed"))
        named_records.append({"model_dtype": key, "backend": backend, "context": context,
                              "parameter_count": len(names), "reference_nonzero_gradient_parameters": value["reference_nonzero_gradient_parameters"],
                              "reference_none_parameters": missing, "actual_none_parameters": value["actual_none_parameters"],
                              "gradient_relative_l2": value["gradient_relative_l2"]})

    for key, backends in report["preflight_validation"].items():
        check("preflight_backends::" + key, set(backends) == {"v4", "v5"})
        for backend, contexts in backends.items():
            check("preflight_contexts::" + key + "::" + backend, set(contexts) == {"native_eager_force_loss", "eqx_spy_force_loss"})
            for context, value in contexts.items():
                validate_record(key, backend, context, value)
    check("complete_preflight_grid", set(report["preflight_validation"]) == {name + "/" + dtype for name, dtype in itertools.product(named_counts, ("float32", "float64"))})
    recomputed = []
    for case in report["results"]:
        name = "/".join((case["model"], case["dtype"], case["mode"]))
        expected_methods = case["expected_frozen_v4_method_count"]
        check("case_versions::" + name, case["backend_versions"] == {"v4": "fasteq-optimized-v4", "v5": "fasteq-optimized-v5"}
              and case["current_source_sha256_digest"] == metadata["current_source_sha256_digest"]
              and case["frozen_v4_method_count"] == expected_methods
              and (expected_methods > 0 and expected_methods % 2 == 0 if case["model"] == "tece" else expected_methods == 0)
              and case["nodes"] == 256)
        for backend, value in case["validation"].items():
            check("finite_validation_outputs::" + name + "::" + backend, all(math.isfinite(value[field]) for field in ("energy", "forces") if field in value))
            if case["mode"] == "force_loss":
                validate_record(case["model"] + "/" + case["dtype"], backend, "timed_case_force_loss", value)
        for backend in metadata["labels"]:
            samples = case[backend]["samples"]
            check("nine_finite_positive_samples::" + name + "::" + backend, len(samples) == 9 and all(math.isfinite(row[field]) and row[field] > 0
                  for row in samples for field in ("wall_ms", "cuda_event_ms")))
            for field in ("wall_ms", "cuda_event_ms"):
                recompute = stats([row[field] for row in samples])
                check("raw_stats::" + name + "::" + backend + "::" + field,
                      all(math.isclose(value, case[backend][field][stat], rel_tol=1e-12, abs_tol=1e-12) for stat, value in recompute.items()))
        for index, order in enumerate(case["round_orders"]):
            expected = [metadata["labels"][(index + i) % 3] for i in range(3)]
            if index % 2:
                expected.reverse()
            check("actual_pairing_order::" + name + "::" + str(index), order == expected)
        check("nine_pairing_orders::" + name, len(case["round_orders"]) == 9)
        for left, right in (("original", "v5"), ("v4", "v5"), ("original", "v4")):
            values = [a["wall_ms"] / b["wall_ms"] for a, b in zip(case[left]["samples"], case[right]["samples"], strict=True)]
            recompute = stats(values)
            check("raw_paired_stats::" + name + "::" + left + "_over_" + right,
                  all(math.isclose(value, case["paired_speedup"][left + "_over_" + right][stat], rel_tol=1e-12, abs_tol=1e-12) for stat, value in recompute.items()))
            recomputed.append({"case": name, "ratio": left + "_over_" + right, "raw_nine_ratios": values, **recompute})
    check("input_unchanged_during_audit", sha(args.input) == initial_input_sha)
    audit = {"passed": not failures, "input": str(args.input), "input_sha256": initial_input_sha,
             "driver_sha256": sha(Path(__file__)), "check_count": len(checks), "failures": failures,
             "current_source_file_count": len(current), "archived_source_file_count": len(archived),
             "independent_cpp_file_count": len(cpp_hashes), "current_driver_file_count": len(drivers),
             "parameter_names": report["parameter_names"], "named_parameter_validation_records": named_records,
             "recomputed_paired_stats": recomputed, "checks": checks,
             "scope": "CPU-only actual file/hash comparison and stored raw sample/validation record consistency; no model/GPU rerun",
             "limitations": ["Raw parameter-gradient tensors were not archived. Named-key/order/shape/finiteness comparisons were enforced by the source-verified GPU driver; this audit checks all corresponding saved records.",
                             "GPU sharing history cannot be rechecked after the run; source-verified driver performs NVML guards before/after each round/case and completed=true records successful completion.",
                             "The run metadata records CUDA_VISIBLE_DEVICES and GPU model, but not physical GPU UUID. This audit does not infer an unrecorded UUID."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print("MODEL_DATA_AUDIT", audit["passed"], len(checks), len(named_records), len(recomputed))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
