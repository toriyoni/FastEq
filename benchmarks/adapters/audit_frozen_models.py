"""Audit new three-way frozen-baseline model records using CPU/stdlib only.

Recomputes paired statistics from raw rounds and verifies the actual source,
snapshot, driver and isolated C++ files. Does not execute models or import Torch.
"""

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from pathlib import Path

SOURCE_DIRS = (
    "src/FastEq/fasteq/adapters", "src/FastEq/fasteq/ops", "src/FastEq/fasteq/triton",
    "src/FastEq/fasteq/cuda", "src/tace/tace/models", "src/tace/eqx", "src/prophet/prophet",
)
SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".c", ".cu", ".h", ".hpp", ".cuh"}
PARAMETER_COUNTS = {"tace": 44, "tece": 57, "prophet": 81}
RUNTIME_SOURCES = {"jit_runtime.h", "jit_runtime.cpp", "indexed_linear_plan.cpp", "norm_plan.cpp"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] if lower == upper else (
        values[lower] * (upper - position) + values[upper] * (position - lower)
    )


def stats(values):
    return {"median": statistics.median(values), "q25": percentile(values, 0.25),
            "q75": percentile(values, 0.75), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="Verified archive of current production source, when live source has changed")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="v5")
    parser.add_argument("--current", default="v6")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--allow-validation-only", action="store_true")
    args = parser.parse_args()
    source_root = (args.source_root or args.root).resolve()
    input_sha = sha(args.input)
    report = json.loads(args.input.read_text())
    metadata = report["metadata"]
    if not metadata.get("completed"):
        raise SystemExit("Incomplete run: no audit written")
    baseline, current = args.baseline, args.current
    labels, adapted = ["original", baseline, current], {baseline, current}
    checks, failures, records, recomputed = [], [], [], []

    def check(name, condition, **details):
        checks.append({"check": name, "passed": bool(condition), **details})
        if not condition:
            failures.append(name)

    smoke = metadata["status"] == "loader_validation_only"
    validation_only = metadata["status"] in {"shared_validation_only", "exclusive_validation_only"}
    protocol = metadata["protocol"]
    rounds = protocol["rounds"]
    check("allowed_run_status", (smoke and args.allow_smoke) or (validation_only and args.allow_validation_only)
          or metadata["status"] == "formal_paired_models")
    check("labels_and_same_run_protocol", metadata["labels"] == labels and metadata["historical_timing_used"] is False
          and ((protocol["warmup"] == rounds == protocol["iterations"] == 0 and protocol["timing"] == "none") if validation_only
               else protocol["warmup"] >= (2 if smoke else 5) and rounds >= (3 if smoke else 9)
               and protocol["iterations"] >= (1 if smoke else 3)))
    check("current_candidate_optins", metadata["experimental_flags"] == {
        "FASTEQ_EXPERIMENTAL_LINEAR_SIMT": "1", "FASTEQ_EXPERIMENTAL_NATIVE_MODULATION": "1",
    })
    evidence = metadata["independent_execution_evidence_sha256"]
    check("independent_execution_evidence_files", {name: sha(Path(name)) for name in evidence} == evidence)
    if validation_only:
        check("correctness_only_has_no_timing", metadata["timing_fields_empty"] is True)
        if metadata["status"] == "shared_validation_only":
            check("shared_validation_allocator_limit", metadata["shared_allocator_fraction"] == 0.025
                  and metadata["shared_allocator_limit_bytes"] <= 2.1 * 1024**3
                  and bool(metadata["sharing_checks"]) and all(row["allocator_peak_reserved_bytes"] <= metadata["shared_allocator_limit_bytes"]
                                                              for row in metadata["sharing_checks"]))
    sources = {str(path.relative_to(source_root)): sha(path)
               for relative in SOURCE_DIRS for path in sorted((source_root / relative).rglob("*"))
               if path.is_file() and path.suffix in SOURCE_SUFFIXES
               and "__pycache__" not in path.parts and "generated" not in str(path)}
    check("actual_current_source_catalog", sources == metadata["current_source_sha256"], count=len(sources))
    source_digest = digest(sources)
    check("actual_current_source_digest", source_digest == metadata["current_source_sha256_digest"])
    drivers = {name: sha(Path(name)) for name in metadata["benchmark_driver_sha256"]}
    check("actual_driver_files", drivers == metadata["benchmark_driver_sha256"], count=len(drivers))
    frozen = metadata[baseline]
    archive = Path(frozen["archive"])
    manifest = json.loads((archive / "manifest.json").read_text())
    check("actual_baseline_manifest", manifest == frozen["manifest"] and manifest["version"] == baseline
          and sha(archive / "manifest.json") == frozen["manifest_sha256"])
    archived = {name: sha(archive / name) for name in manifest["files_sha256"]}
    check("actual_489_baseline_files", archived == manifest["files_sha256"] and len(archived) == 489)
    check("frozen_runtime_catalog", set(frozen["runtime_source_names"]) == RUNTIME_SOURCES
          and set(frozen["compatibility_sha256"]) == RUNTIME_SOURCES)
    cpp_directory, cpp_hashes = Path(frozen["compatibility_sources"]), {}
    number = baseline.removeprefix("v")
    transformations = (("LinearPlan", "V" + number + "LinearPlan"),
                       ("LinearDerivative", "V" + number + "LinearDerivative"),
                       ("bind_indexed_linear", "bind_" + baseline + "_indexed_linear"),
                       ("bind_norm", "bind_" + baseline + "_norm"))
    for name in frozen["compatibility_sha256"]:
        expected = (archive / "src/FastEq/fasteq/cuda/src" / name).read_text().replace(
            "fasteq_jit", "fasteq_" + baseline + "_jit"
        )
        for before, after in transformations:
            expected = re.sub(r"\b" + before + r"\b", after, expected)
        actual = (cpp_directory / name).read_text()
        cpp_hashes[name] = sha(cpp_directory / name)
        check("isolated_cpp_namespace_only::" + name, actual == expected
              and hashlib.sha256(expected.encode()).hexdigest() == frozen["compatibility_sha256"][name])
    check("actual_isolated_cpp_files", cpp_hashes == frozen["compatibility_sha256"])
    check("loaded_snapshot_files", all(row["sha256"] == archived[row["path"]] for row in frozen["loaded"]))
    check("shared_dependencies_source_verified", all(sha(source_root / row["path"]) == row["sha256"]
          for row in frozen["shared_dependencies"]))
    check("independent_frozen_runtime", frozen["extension_name"] == "fasteq_" + baseline + "_cuda_jit"
          and frozen["namespace_changes_only"] and "independent frozen " + baseline + " C++" in frozen["native_runtime"])
    check("minimal_tece_method_binding", frozen["bound_model_classes"] == ["tace.models._e3nn.tece_oam_rra.Convolution"]
          and frozen["bound_model_methods"] == ["forward", "rotary_score"])
    requested = metadata["requested_cases"]
    if validation_only:
        observations = metadata["candidate_execution_observations"]
        linear_calls, modulation_calls = observations["indexed_linear_calls"], observations["modulation_calls"]
        check("actual_candidate_observation_counts", observations["simt_forward_source_proven_calls"] == sum(
            row["simt_forward_source_proven"] for row in linear_calls
        ) and observations["native_modulation_autograd_forward_calls"] == sum(
            row["native_autograd_node_observed"] for row in modulation_calls
        ))
        if "tece" in requested["models"] and "float32" in requested["dtypes"]:
            check("actual_large_fp32_indexed_calls", observations["simt_forward_source_proven_calls"] > 0
                  and all(row["dtype"] == "torch.float32" and row["features"][0] >= 1024 and row["weight"][0] == 1
                          for row in linear_calls if row["simt_forward_source_proven"]))
        if "prophet" in requested["models"]:
            check("actual_native_modulation_forward_first_second", observations["native_modulation_autograd_forward_calls"] > 0
                  and observations["native_modulation_first_adjoint_invocations"] > 0
                  and observations["native_modulation_second_adjoint_invocations"] > 0
                  and any(count >= 3 for row in modulation_calls for count in row["native_cached_executables"])
                  and all(row["output_grad_fn"] == "FastEqModulationBackward" for row in modulation_calls
                          if row["native_autograd_node_observed"]))
    expected_keys = set(itertools.product(requested["models"], requested["dtypes"], requested["modes"]))
    keys = {(row["model"], row["dtype"], row["mode"]) for row in report["results"]}
    check("requested_unique_case_grid", keys == expected_keys and len(report["results"]) == len(expected_keys))
    if not smoke and not validation_only:
        full_grid = set(itertools.product(PARAMETER_COUNTS, ("float32", "float64"), ("energy", "forces", "force_loss")))
        check("formal_complete_18_case_grid", keys == full_grid and len(report["results"]) == 18)
    expected_preflight = {name + "/" + dtype for name, dtype in itertools.product(requested["models"], requested["dtypes"])}
    check("complete_preflight_grid", set(report["preflight_validation"]) == expected_preflight
          and set(report["parameter_names"]) == expected_preflight)

    def validation_record(key, backend, context, value, mode):
        model, dtype = key.split("/")
        outputs = {"energy"} if mode == "energy" else {"energy", "forces"}
        check("finite_output_validation::" + key + "::" + backend + "::" + context,
              outputs.issubset(value) and all(math.isfinite(value[field]) and value[field] >= 0 for field in outputs))
        if mode != "force_loss":
            return
        names = report["parameter_names"][key]
        nonzero, zero, absent = (value["reference_nonzero_gradient_parameter_names"],
                                value["reference_zero_gradient_parameters"], value["reference_none_parameters"])
        combined = nonzero + zero + absent
        check("full_named_gradient_validation::" + key + "::" + backend + "::" + context,
              len(names) == len(set(names)) == PARAMETER_COUNTS[model] == value["named_parameters_checked"]
              and len(combined) == len(set(combined)) == len(names) and set(combined) == set(names)
              and len(nonzero) == value["reference_nonzero_gradient_parameters"]
              and set(value["actual_none_parameters"]).issubset(absent)
              and value["all_parameter_values_and_present_gradients_finite"] is True
              and math.isfinite(value["gradient_relative_l2"])
              and 0 <= value["gradient_relative_l2"] < (2e-4 if dtype == "float32" else 2e-7)
              and (context != "eqx_spy_force_loss" or value["eqx_cuda_spy"] == "passed"))
        records.append({"model_dtype": key, "backend": backend, "context": context,
                        "named_parameters_checked": value["named_parameters_checked"],
                        "reference_nonzero_gradient_parameters": len(nonzero),
                        "reference_zero_gradient_parameters": zero, "reference_none_parameters": absent,
                        "gradient_relative_l2": value["gradient_relative_l2"]})

    for key, backends in report["preflight_validation"].items():
        check("preflight_backends::" + key, set(backends) == adapted)
        for backend, contexts in backends.items():
            check("preflight_contexts::" + key + "::" + backend,
                  set(contexts) == {"native_eager_force_loss", "eqx_spy_force_loss"})
            for context, value in contexts.items():
                validation_record(key, backend, context, value, "force_loss")
    for case in report["results"]:
        key = case["model"] + "/" + case["dtype"]
        name = key + "/" + case["mode"]
        expected_methods = 2 * case["so2_convolution_count"]
        check("case_versions_and_binding::" + name,
              case["backend_versions"] == {backend: "fasteq-optimized-" + backend for backend in adapted}
              and case["frozen_" + baseline + "_method_count"] == case["expected_frozen_" + baseline + "_method_count"] == expected_methods
              and (expected_methods > 0 if case["model"] == "tece" else expected_methods == 0)
              and case["nodes"] == 256)
        check("case_source_guards::" + name, case["current_source_sha256_digest"] == source_digest
              and (validation_only or case["round_source_sha256_digests"] == [source_digest] * rounds))
        check("case_validation_backends::" + name, set(case["validation"]) == adapted)
        for backend, value in case["validation"].items():
            validation_record(key, backend, case["mode"], value, case["mode"])
        if validation_only:
            check("case_has_no_timing_fields::" + name, case["timing_fields_empty"] is True
                  and not set(labels + ["paired_speedup", "round_orders", "round_source_sha256_digests"]).intersection(case))
            continue
        for backend in labels:
            samples = case[backend]["samples"]
            check("positive_raw_samples::" + name + "::" + backend, len(samples) == rounds
                  and all(math.isfinite(row[field]) and row[field] > 0 for row in samples
                          for field in ("wall_ms", "cuda_event_ms")))
            for field in ("wall_ms", "cuda_event_ms"):
                values = stats([row[field] for row in samples])
                check("recomputed_raw_stats::" + name + "::" + backend + "::" + field,
                      all(math.isclose(value, case[backend][field][stat], rel_tol=1e-12, abs_tol=1e-12)
                          for stat, value in values.items()))
        expected_orders = []
        for index in range(rounds):
            order = [labels[(index + i) % 3] for i in range(3)]
            expected_orders.append(list(reversed(order)) if index % 2 else order)
        check("exact_round_pairing_order::" + name, case["round_orders"] == expected_orders)
        for left, right in (("original", current), (baseline, current), ("original", baseline)):
            values = [a["wall_ms"] / b["wall_ms"] for a, b in zip(case[left]["samples"], case[right]["samples"], strict=True)]
            paired = stats(values)
            ratio = left + "_over_" + right
            check("recomputed_paired_stats::" + name + "::" + ratio,
                  all(math.isclose(value, case["paired_speedup"][ratio][stat], rel_tol=1e-12, abs_tol=1e-12)
                      for stat, value in paired.items()))
            recomputed.append({"case": name, "ratio": ratio, "raw_round_ratios": values, **paired})
    check("expected_full_gradient_record_count", len(records) == 4 * len(expected_preflight)
          + 2 * sum(case["mode"] == "force_loss" for case in report["results"]))
    check("input_unchanged_during_audit", sha(args.input) == input_sha)
    audit = {"passed": not failures, "input": str(args.input), "input_sha256": input_sha,
             "audit_driver_sha256": sha(Path(__file__)), "check_count": len(checks), "failures": failures,
             "current_source_file_count": len(sources), "archived_source_file_count": len(archived),
             "independent_cpp_file_count": len(cpp_hashes), "current_driver_file_count": len(drivers),
             "named_parameter_validation_records": records, "recomputed_paired_stats": recomputed,
             "checks": checks, "scope": "CPU-only hashes/stored validation/raw-pair consistency; no model or GPU rerun",
             "limitations": ["Parameter-gradient tensors are not archived; source-verified GPU driver enforces complete key/order/shape/finite comparisons. This audit checks all saved validation records.",
                             "NVML sharing history cannot be rechecked after the run. Guards are enforced before/after each round/case by the source-verified driver.",
                             "CUDA event timing measures a stream timeline, not the sum of individual kernel execution times."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print("FROZEN_MODEL_AUDIT", audit["passed"], len(checks), len(records), len(recomputed))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
