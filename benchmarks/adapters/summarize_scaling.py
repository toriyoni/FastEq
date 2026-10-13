"""CPU-only summaries of the current original/v6 scaling-to-OOM campaign.

Only source-linked ``status=ok`` point files supply performance data. Singles
never supply a paired gain; CUDA OOM requires an explicit allocation witness.
The module imports no torch, CUDA, NumPy or operator factory runtime.
"""

import argparse
import ast
import csv
import hashlib
import io
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path

LABELS = ("original", "v6")
DTYPES = ("float32", "float64")
OPERATOR_MODES = ("forward", "backward", "force_loss")
MODEL_MODES = ("energy", "forces", "force_loss")
MODELS = ("tace", "tece", "prophet")
ROUNDS = 9
MODEL_BASELINES = {
    "tace": "original PyTorch/e3nn cgtp; optional acceleration disabled",
    "tece": "original PyTorch/e3nn plus automatic EQX CUDA Wigner-D",
    "prophet": "original PyTorch/e3nn; kernel=False",
}
LIMITATIONS = [
    "Operator force_loss is a nonlinear feature-gradient higher-order microchain, not physical atomic forces. Models use the actual positions→energy→forces→loss→parameter-gradient chain; no optimizer.step.",
    "N is atom count. Node operators use N rows; edge operators use 12N independent rows. Model E=12N, C64, lmax2 and two layers remain fixed.",
    "Above N256 the geometry repeats translated disconnected 256-atom motifs in one graph. Construction stores O(N*12) data. This capacity workload is not a connected large-system MD simulation.",
    "Original implementations may already execute CUDA. EQX CUDA, extracted original expressions and complete modules are identified separately.",
    "Only same-point paired round ratios are speedups. Single-backend measurements after the other backend OOM have no paired ratio.",
    "Doubling establishes a last-success/first-OOM interval, not the precise maximum atom count. Setup, execution, warmup and validation OOM stages remain distinct.",
    "PyTorch allocator peaks do not include every CUDA context/library allocation; NVML snapshots are not exact whole-run memory peaks.",
    "Shared GPUs, CPU OOM, errors and timeouts cannot establish a true exclusive GPU OOM boundary or complete the campaign.",
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def operator_catalog(path=None):
    """Read literal OperatorSpec entries without importing torch-backed code."""
    path = Path(path or Path(__file__).with_name("scaling_operator_factories.py"))
    source = ast.parse(path.read_text())
    names = ("name", "family", "rows_factor", "baseline_kind", "baseline_source", "description")
    for node in source.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SPECS" for target in node.targets
        ):
            if not isinstance(node.value, (ast.Tuple, ast.List)):
                raise ValueError("Operator SPECS must remain a literal sequence")
            result = []
            for call in node.value.elts:
                if not isinstance(call, ast.Call) or len(call.args) != len(names):
                    raise ValueError("Unknown OperatorSpec schema")
                result.append(dict(zip(names, (ast.literal_eval(arg) for arg in call.args), strict=True)))
            if len({row["name"] for row in result}) != len(result):
                raise ValueError("Duplicate operator names")
            return result
    raise ValueError("Factory SPECS not found")


def _key(row):
    return tuple(row.get(field) for field in ("kind", "name", "dtype", "mode"))


def _name(key):
    return "/".join(str(value) for value in key)


def _expected(catalog):
    operators = (("operator", row["name"], dtype, mode)
                 for row, dtype, mode in itertools.product(catalog, DTYPES, OPERATOR_MODES))
    models = (("model", name, dtype, mode)
              for name, dtype, mode in itertools.product(MODELS, DTYPES, MODEL_MODES))
    return sorted([*operators, *models])


def _quantile(values, fraction):
    ordered = sorted(values)
    offset = (len(ordered) - 1) * fraction
    left, right = math.floor(offset), math.ceil(offset)
    return ordered[left] + (ordered[right] - ordered[left]) * (offset - left)


def sample_stats(values, *, positive=True):
    if not isinstance(values, list) or len(values) != ROUNDS:
        raise ValueError(f"Expected {ROUNDS} raw rounds")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or (value <= 0 if positive else value < 0)
           for value in values):
        raise ValueError("Nonfinite, negative or zero sample")
    return {"median": statistics.median(values), "q25": _quantile(values, 0.25),
            "q75": _quantile(values, 0.75), "min": min(values), "max": max(values)}


def _atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _write_csv(path, rows, columns):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in columns})
    _atomic(path, buffer.getvalue())


def _metadata(point):
    metadata = point.get("metadata") or {}
    if not isinstance(metadata, dict):
        return {}
    # Older in-progress workers may group factory metadata beneath case.
    return metadata.get("case", metadata)


def _baseline(point, catalog):
    metadata = _metadata(point)
    if point.get("kind") == "model":
        return {"baseline_kind": "original_model", "baseline_source": metadata.get(
            "original_baseline", MODEL_BASELINES.get(point.get("name"), "unknown")),
            "rows_factor": 1}
    spec = catalog.get(point.get("name"), {})
    return {field: metadata.get(field, spec.get(field))
            for field in ("baseline_kind", "baseline_source", "rows_factor")}


def _explicitly_shared(point):
    for name in ("gpu", "gpu_after", "gpu_failure"):
        gpu = point.get(name) or {}
        if not isinstance(gpu, dict):
            continue
        if gpu.get("exclusive") is False or gpu.get("shared") is True:
            return True
        if any(gpu.get(field) for field in ("external_compute_pids", "foreign_compute_pids", "foreign_pids")):
            return True
        for check in gpu.get("checks", []):
            if isinstance(check, dict) and any(check.get(field) for field in (
                "external_compute_pids", "foreign_compute_pids", "foreign_pids"
            )):
                return True
    return False


def _valid_measurements(point):
    if point.get("status") != "ok":
        raise ValueError("Only completed ok points can supply timing")
    if _explicitly_shared(point):
        raise ValueError("Point explicitly records GPU sharing")
    if not point.get("source_guard", {}).get("manifest_sha256"):
        raise ValueError("Missing frozen source-guard evidence")
    if point.get("runtime", {}).get("allocator_memory_fraction", 1.0) != 1.0:
        raise ValueError("An allocator cap cannot supply full-GPU performance/capacity")
    protocol, variant = point.get("protocol"), point.get("variant")
    if protocol == "paired":
        labels = LABELS
        if variant is not None:
            raise ValueError("Paired point has a single variant")
        if point.get("validation", {}).get("status") != "passed":
            raise ValueError("Paired performance lacks successful current-point precision validation")
    elif protocol == "single" and variant in LABELS:
        labels = (variant,)
    else:
        raise ValueError("Unknown timing protocol/variant")
    measurements = point.get("measurements", {})
    if set(measurements) != set(labels):
        raise ValueError("Measurement backend keys differ from protocol")
    result = {}
    for label in labels:
        measurement = measurements[label]
        wall = sample_stats(measurement.get("wall_ms"))
        event = sample_stats(measurement.get("event_ms"), positive=False)
        iterations = measurement.get("iterations")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("Missing positive iteration count")
        result[label] = {"wall": wall, "event": event}
    ratios = {}
    if protocol == "paired":
        if measurements["original"]["iterations"] != measurements["v6"]["iterations"]:
            raise ValueError("Paired variants use different iteration counts")
        for field, label in (("wall_ms", "wall"), ("event_ms", "event")):
            left, right = (measurements[name][field] for name in LABELS)
            if any(value <= 0 for value in right):
                # A zero CUDA event can occur for a view-only forward; no ratio.
                ratios[label] = None
                if point.get("ratios", {}).get(label) not in (None, []):
                    raise ValueError("Stored ratio with zero denominator")
                continue
            values = [x / y for x, y in zip(left, right, strict=True)]
            stored = point.get("ratios", {}).get(label)
            if not isinstance(stored, list) or len(stored) != ROUNDS or any(
                not isinstance(actual, (int, float)) or not math.isfinite(actual)
                or not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12)
                for expected, actual in zip(values, stored, strict=True)
            ):
                raise ValueError("Stored ratios do not match same-round original/v6 samples")
            ratios[label] = sample_stats(values, positive=label != "event")
    elif point.get("ratios") not in (None, {}):
        raise ValueError("A single-backend point cannot have paired ratios")
    return result, ratios


def _point_paths(campaign, state):
    requested = list(state.get("points", []))
    for row in state.get("series", {}).values():
        if isinstance(row, dict):
            requested.extend(row.get("points", []))
    # Also capture completed files between their atomic save and controller save.
    if (campaign / "points").is_dir():
        requested.extend(str(path.relative_to(campaign)) for path in (campaign / "points").rglob("*.json")
                         if not path.name.endswith(".partial.json"))
    paths, failures = [], []
    for value in sorted(set(requested)):
        path = (campaign / value).resolve()
        if not path.is_relative_to(campaign):
            failures.append({"path": value, "reason": "Point outside campaign"})
        elif not path.is_file():
            failures.append({"path": value, "reason": "Missing point file"})
        else:
            paths.append(path)
    return paths, failures


def summarize(campaign, *, output=None, catalog=None):
    """Recompute partial summaries and never mutate point/controller evidence."""
    campaign = Path(campaign).resolve()
    output = Path(output).resolve() if output else campaign / "summary"
    output.mkdir(parents=True, exist_ok=True)
    state_path = campaign / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"series": {}}
    catalog_rows = catalog if catalog is not None else operator_catalog()
    catalog_by_name = {row["name"]: row for row in catalog_rows}
    expected = _expected(catalog_rows)
    expected_set = set(expected)
    series = {}
    issues = []
    for state_key, row in state.get("series", {}).items():
        key = _key(row)
        if key not in expected_set:
            issues.append({"state_key": state_key, "reason": "Unexpected series identity"})
        elif key in series:
            issues.append({"state_key": state_key, "reason": "Duplicate series identity"})
        else:
            series[key] = row
    paths, failures = _point_paths(campaign, state)
    issues.extend(failures)
    points, ids, point_hashes = [], set(), {}
    attempts, measurements, pairs, excluded = [], [], [], []
    successful = {key: {label: [] for label in LABELS} for key in expected}
    single_successful = {key: {label: [] for label in LABELS} for key in expected}
    oom_witnesses = {key: {label: [] for label in LABELS} for key in expected}
    for path in paths:
        relative = str(path.relative_to(campaign))
        try:
            point = json.loads(path.read_text())
            identity = point.get("id")
            if not isinstance(identity, str) or not identity:
                raise ValueError("Missing point id")
            if identity in ids:
                raise ValueError("Duplicate point id")
            ids.add(identity)
            key = _key(point)
            if key not in expected_set:
                raise ValueError("Unknown point series")
            if isinstance(point.get("atoms"), bool) or not isinstance(point.get("atoms"), int) or point["atoms"] < 1:
                raise ValueError("Invalid atom count")
        except (ValueError, TypeError, AttributeError) as error:
            issues.append({"path": relative, "reason": str(error)})
            continue
        point_hashes[relative] = _sha(path)
        points.append(point)
        base = {"id": identity, "point_file": relative, **dict(zip(
            ("kind", "name", "dtype", "mode"), key, strict=True)),
            "atoms": point["atoms"], "protocol": point.get("protocol"),
            "variant": point.get("variant"), **_baseline(point, catalog_by_name)}
        attempts.append({**base, "status": point.get("status"), "stage": point.get("stage"),
                         "failure_variant": point.get("failure_variant"), "elapsed_s": point.get("elapsed_s"),
                         "error_kind": point.get("error_kind"),
                         "is_cuda_oom": (point.get("oom") or {}).get("is_cuda_oom", False)})
        manifest_sha = state.get("manifest_sha256")
        if manifest_sha and point.get("status") in ("ok", "oom") and (
            point.get("source_guard", {}).get("manifest_sha256") != manifest_sha
            or point.get("source_guard_end", {}).get("manifest_sha256", manifest_sha) != manifest_sha
        ):
            issues.append({"path": relative, "reason": "Point source guard does not match controller frozen manifest"})
            continue
        if point.get("status") == "oom":
            if point.get("protocol") == "paired":
                # A pair's setup/input OOM can have no attributable backend.
                # It is an honest attempt, never an individual capacity witness.
                continue
            witness = point.get("oom") or {}
            label = point.get("failure_variant")
            if witness.get("is_cuda_oom") is True and label in LABELS and not _explicitly_shared(point):
                oom_witnesses[key][label].append(point)
            else:
                issues.append({"path": relative, "reason": "OOM lacks exclusive CUDA allocation witness/backend"})
            continue
        if point.get("status") != "ok":
            continue
        try:
            stats, ratios = _valid_measurements(point)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            excluded.append({"path": relative, "reason": str(error)})
            continue
        for label, values in stats.items():
            measurement = point["measurements"][label]
            successful[key][label].append(point["atoms"])
            if point["protocol"] == "single":
                single_successful[key][label].append(point["atoms"])
            row = {**base, "backend": label, "rounds": ROUNDS,
                   "iterations": measurement["iterations"], "warmup": measurement.get("warmup")}
            for field in ("wall", "event"):
                row.update({f"{field}_{stat}_ms": value for stat, value in values[field].items()})
            for field in ("peak_allocated_bytes", "peak_reserved_bytes", "incremental_peak_allocated_bytes"):
                row[field] = measurement.get(field)
            measurements.append(row)
        if point["protocol"] == "paired":
            row = {**base, "rounds": ROUNDS, "iterations": point["measurements"]["original"]["iterations"]}
            for label, values in stats.items():
                for field in ("wall", "event"):
                    row.update({f"{label}_{field}_{stat}_ms": value for stat, value in values[field].items()})
            for field, values in ratios.items():
                row.update({f"{field}_speedup_{stat}": value for stat, value in (values or {}).items()})
            pairs.append(row)
    boundaries = []
    for key in expected:
        source = series.get(key, {})
        start_atoms = source.get("start_atoms", state.get("config", {}).get("start_atoms", state.get("start_atoms", 64)))
        for label in LABELS:
            boundary = source.get("boundaries", {}).get(label, {})
            first_oom = boundary.get("first_oom_atoms")
            witness = [point for point in oom_witnesses[key][label]
                       if point["atoms"] == first_oom and point.get("protocol") == "single"
                       and point.get("variant") == label]
            complete = (isinstance(first_oom, int) and not isinstance(first_oom, bool)
                        and first_oom > 0 and bool(witness)
                        and boundary.get("status") in ("oom", "confirmed_oom")
                        and boundary.get("completed", True) is True)
            declared_last = boundary.get("last_success_atoms")
            if complete and declared_last is not None:
                complete = (declared_last in single_successful[key][label]
                            and declared_last * 2 == first_oom
                            and boundary.get("last_success_confirmed", True) is True)
            elif complete:
                complete = first_oom == start_atoms
            if boundary.get("status") in ("oom", "confirmed_oom") and not complete:
                issues.append({"series": _name(key), "backend": label,
                               "reason": "Controller OOM boundary lacks single-backend CUDA OOM/last-success witnesses"})
            observed_last = max(successful[key][label], default=None)
            if complete and observed_last is not None and observed_last >= first_oom:
                complete = False
                issues.append({"series": _name(key), "backend": label,
                               "reason": "Successful measurement reaches or exceeds declared first OOM"})
            status = "oom" if complete else boundary.get("status") or ("pending" if source else "missing_series")
            row = {**dict(zip(("kind", "name", "dtype", "mode"), key, strict=True)),
                   "backend": label, "status": status, "oom_verified": complete,
                   "last_success_atoms": declared_last,
                   "last_timed_success_atoms": observed_last,
                   "last_single_success_atoms": max(single_successful[key][label], default=None),
                   "first_oom_atoms": first_oom if complete else None,
                   "oom_stage": boundary.get("oom_stage") if complete else None,
                   "oom_point_files": ";".join(str(point.get("id")) for point in witness),
                   "phase": source.get("phase"), "next_atoms": source.get("next_atoms"),
                   "series_completed": source.get("completed", False)}
            boundaries.append(row)
    verified = sum(row["oom_verified"] for row in boundaries)
    performance_completed = (verified == len(expected) * len(LABELS)
                             and len(series) == len(expected)
                             and all(row.get("completed") is True for row in series.values())
                             and not issues and not excluded)
    if state.get("performance_completed") is True and not performance_completed:
        issues.append({"reason": "Controller claims completion but independent summary does not verify every OOM boundary"})
    summary = {
        "status": "completed" if performance_completed else "partial",
        "performance_completed": performance_completed,
        "expected_series": len(expected), "observed_series": len(series),
        "missing_series": [_name(key) for key in expected if key not in series],
        "expected_backend_oom_boundaries": len(expected) * len(LABELS),
        "verified_backend_oom_boundaries": verified,
        "missing_backend_oom_boundaries": [f"{_name(_key(row))}/{row['backend']}"
                                           for row in boundaries if not row["oom_verified"]],
        "paired_ok_points": len(pairs), "single_ok_points": sum(
            row["protocol"] == "single" for row in measurements),
        "valid_timed_backend_rows": len(measurements),
        "attempt_status_counts": dict(Counter(row["status"] for row in attempts)),
        "attempt_error_kind_counts": dict(Counter(row["error_kind"] for row in attempts if row["error_kind"])),
        "accuracy_gaps": {_name(key): row["accuracy_gaps"] for key, row in series.items()
                          if row.get("accuracy_gaps")},
        "excluded_performance_points": excluded,
        "integrity_issues": issues,
        "source_campaign": str(campaign), "state_sha256": _sha(state_path) if state_path.exists() else None,
        "point_files_sha256": point_hashes, "summarizer_sha256": _sha(__file__),
        "limitations": LIMITATIONS, "catalog": catalog_rows,
        "statistic": "Linear-interpolated percentiles of nine raw rounds; paired speedup is median/IQR of same-round original/v6 ratios",
        "raw_samples_remain_in_point_files": True,
    }
    identity_columns = ["id", "point_file", "kind", "name", "dtype", "mode", "atoms", "protocol", "variant",
                        "baseline_kind", "baseline_source", "rows_factor"]
    time_columns = [f"{field}_{stat}_ms" for field in ("wall", "event")
                    for stat in ("median", "q25", "q75", "min", "max")]
    _write_csv(output / "measurements.csv", measurements, identity_columns + ["backend", "rounds", "iterations", "warmup"]
               + time_columns + ["peak_allocated_bytes", "peak_reserved_bytes", "incremental_peak_allocated_bytes"])
    pair_columns = [f"{label}_{field}_{stat}_ms" for label in LABELS for field in ("wall", "event")
                    for stat in ("median", "q25", "q75", "min", "max")]
    ratio_columns = [f"{field}_speedup_{stat}" for field in ("wall", "event")
                     for stat in ("median", "q25", "q75", "min", "max")]
    _write_csv(output / "paired.csv", pairs, identity_columns + ["rounds", "iterations"] + pair_columns + ratio_columns)
    _write_csv(output / "attempts.csv", attempts, identity_columns + ["status", "stage", "failure_variant", "error_kind", "elapsed_s", "is_cuda_oom"])
    boundary_columns = ["kind", "name", "dtype", "mode", "backend", "status", "oom_verified", "last_success_atoms",
                        "last_timed_success_atoms", "last_single_success_atoms", "first_oom_atoms", "oom_stage", "oom_point_files", "phase", "next_atoms", "series_completed"]
    _write_csv(output / "boundaries.csv", boundaries, boundary_columns)
    _atomic(output / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    markdown = ["# 等变算子与模型规模扫描", "",
                "**已完成全部独占 GPU OOM 扫描。**" if performance_completed else "**部分结果：尚未完成全部独占 GPU OOM 扫描。**", "",
                f"期望 {len(expected)} 条系列，当前 {len(series)} 条；真实 GPU OOM 边界 {verified}/{len(boundaries)}。",
                f"合法配对规模点 {len(pairs)} 个，单后端独立规模点 {summary['single_ok_points']} 个。旧 v6 全量测试没有重跑。", "",
                "时间与加速比来自当前 point 原始九轮样本；单后端结果只有独立时间，未拼接成配对加速比。",
                "未出现真实 OOM 的后端仍显示 pending；共享、错误、CPU OOM、超时均不计完成。", "",
                f"排除性能点 {len(excluded)} 个，证据问题 {len(issues)} 个。详见 [summary.json](summary.json)。", "",
                f"存在精度对照缺口的系列 {len(summary['accuracy_gaps'])} 条；这些缺口不会被单后端时间替代。", "",
                "- [各后端时间与显存 CSV](measurements.csv)",
                "- [共同规模配对时间、IQR 与加速比 CSV](paired.csv)",
                "- [每后端最后成功/首 OOM 边界 CSV](boundaries.csv)",
                "- [全部尝试与失败阶段 CSV](attempts.csv)", "", "## 显存边界", "",
                "| 类型/对象/精度/模式 | 后端 | 状态 | 最后成功 N | 首 GPU OOM N | OOM 阶段 |",
                "|---|---|---|---:|---:|---|"]
    for row in boundaries:
        markdown.append(f"| {_name(_key(row))} | {row['backend']} | {row['status']} | "
                        f"{row['last_success_atoms'] or '—'} | {row['first_oom_atoms'] or '待测'} | {row['oom_stage'] or '—'} |")
    markdown.extend(["", "## 测量范围与限制", ""])
    markdown.extend(f"- {limitation}" for limitation in LIMITATIONS)
    _atomic(output / "summary.md", "\n".join(markdown) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: CAMPAIGN/summary")
    args = parser.parse_args()
    summary = summarize(args.campaign, output=args.output)
    print(json.dumps({key: summary[key] for key in (
        "status", "performance_completed", "expected_series", "paired_ok_points",
        "single_ok_points", "verified_backend_oom_boundaries")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
