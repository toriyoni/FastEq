"""CPU regressions for accepting/rejecting scaling evidence, with no GPU imports."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "benchmarks/adapters/summarize_scaling.py"
SPEC = importlib.util.spec_from_file_location("_scaling_summary_cpu_test", PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
CATALOG = [{"name": "linear", "family": "Linear", "rows_factor": 1,
            "baseline_kind": "original_module", "baseline_source": "e3nn.o3.Linear",
            "description": "test catalog"}]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _point(identity, *, protocol="paired", variant=None, status="ok", atoms=64):
    values = {"wall_ms": list(range(1, 10)), "event_ms": [value / 2 for value in range(1, 10)],
              "iterations": 1, "warmup": 5, "peak_allocated_bytes": 1024,
              "peak_reserved_bytes": 2048, "incremental_peak_allocated_bytes": 512}
    labels = ("original", "v6") if protocol == "paired" else (variant,)
    measurements = {label: dict(values) for label in labels}
    if "original" in measurements:
        measurements["original"]["wall_ms"] = [value * 2 for value in values["wall_ms"]]
        measurements["original"]["event_ms"] = [value * 2 for value in values["event_ms"]]
    return {"id": identity, "kind": "operator", "name": "linear", "dtype": "float32",
            "mode": "forward", "atoms": atoms, "protocol": protocol, "variant": variant,
            "status": status, "failure_variant": variant, "stage": "complete",
            "measurements": measurements if status == "ok" else {},
            "ratios": {"wall": [2.0] * 9, "event": [2.0] * 9} if protocol == "paired" and status == "ok" else None,
            "gpu": {"uuid": "GPU-test", "foreign_pids": []},
            "source_guard": {"manifest_sha256": "test-source"},
            "validation": {"status": "passed" if protocol == "paired" else "not_paired"}}


def _campaign(path, points, *, boundaries=None, completed=False):
    refs = []
    for point in points:
        relative = "points/" + point["id"] + ".json"
        refs.append(relative)
        _write(path / relative, point)
    row = {"kind": "operator", "name": "linear", "dtype": "float32", "mode": "forward",
           "points": refs, "boundaries": boundaries or {}, "completed": completed}
    _write(path / "state.json", {"series": {"series": row}, "start_atoms": 64})


def test_paired_and_single_never_splice_into_speedup(tmp_path):
    _campaign(tmp_path, [_point("pair"), _point("single", protocol="single", variant="v6", atoms=128)])
    result = SUMMARY.summarize(tmp_path, catalog=CATALOG)
    assert result["paired_ok_points"] == result["single_ok_points"] == 1
    assert result["performance_completed"] is False
    with (tmp_path / "summary/paired.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["wall_speedup_median"]) == 2
    assert float(rows[0]["original_wall_median_ms"]) == 10
    assert float(rows[0]["v6_wall_median_ms"]) == 5
    assert result["expected_series"] == 24
    assert len(result["missing_series"]) == 23


@pytest.mark.parametrize("corruption", ["ratio", "shared", "source", "rounds", "iterations", "validation"])
def test_invalid_ok_point_excluded(tmp_path, corruption):
    point = _point("bad")
    if corruption == "ratio":
        point["ratios"]["wall"][4] = 3
    elif corruption == "shared":
        point["gpu_after"] = {"foreign_pids": [123]}
    elif corruption == "source":
        point["source_guard"] = {}
    elif corruption == "rounds":
        point["measurements"]["v6"]["wall_ms"].pop()
    elif corruption == "iterations":
        point["measurements"]["original"]["iterations"] = 3
    else:
        point["validation"] = {"status": "not_paired"}
    _campaign(tmp_path, [point])
    result = SUMMARY.summarize(tmp_path, catalog=CATALOG)
    assert result["paired_ok_points"] == result["valid_timed_backend_rows"] == 0
    assert len(result["excluded_performance_points"]) == 1


def test_pair_oom_requires_independent_capacity_witness(tmp_path):
    oom = _point("coupled_oom", status="oom", atoms=128)
    oom["failure_variant"] = "original"
    oom["oom"] = {"is_cuda_oom": True, "stage": "warmup"}
    boundaries = {"original": {"first_oom_atoms": 128, "last_success_atoms": 64,
                               "status": "confirmed_oom", "oom_stage": "warmup"}}
    _campaign(tmp_path, [_point("pair"), oom], boundaries=boundaries)
    result = SUMMARY.summarize(tmp_path, catalog=CATALOG)
    assert result["verified_backend_oom_boundaries"] == 0
    assert result["integrity_issues"]


def test_unattributed_pair_setup_oom_is_honest_attempt_not_bad_boundary(tmp_path):
    oom = _point("pair_setup_oom", status="oom", atoms=128)
    oom["failure_variant"] = "unknown"
    oom["oom"] = {"is_cuda_oom": True, "stage": "input_allocation"}
    _campaign(tmp_path, [oom])
    result = SUMMARY.summarize(tmp_path, catalog=CATALOG)
    assert result["attempt_status_counts"] == {"oom": 1}
    assert result["verified_backend_oom_boundaries"] == 0
    assert not result["integrity_issues"]


def test_single_oom_and_single_last_success_confirm_one_backend_only(tmp_path):
    success = _point("single64", protocol="single", variant="original")
    oom = _point("single128", protocol="single", variant="original", status="oom", atoms=128)
    oom["oom"] = {"is_cuda_oom": True, "stage": "timing_round_0"}
    boundaries = {"original": {"first_oom_atoms": 128, "last_success_atoms": 64,
                               "status": "confirmed_oom", "oom_stage": "timing_round_0"}}
    _campaign(tmp_path, [success, oom], boundaries=boundaries)
    result = SUMMARY.summarize(tmp_path, catalog=CATALOG)
    assert result["verified_backend_oom_boundaries"] == 1
    assert result["performance_completed"] is False
    assert not result["integrity_issues"]


def test_cpu_oom_timeout_or_missing_witness_never_confirm_gpu_boundary(tmp_path):
    oom = _point("bad_oom", protocol="single", variant="v6", status="oom")
    oom["oom"] = {"is_cuda_oom": False, "stage": "setup"}
    timeout = _point("timeout", protocol="single", variant="v6", status="timeout", atoms=128)
    boundaries = {"v6": {"first_oom_atoms": 64, "last_success_atoms": None,
                         "status": "confirmed_oom", "oom_stage": "setup"}}
    _campaign(tmp_path, [oom, timeout], boundaries=boundaries)
    result = SUMMARY.summarize(tmp_path, catalog=CATALOG)
    assert result["verified_backend_oom_boundaries"] == 0
    assert result["attempt_status_counts"]["timeout"] == 1


def test_actual_catalog_requires_144_series_and_imports_no_torch(tmp_path):
    before = "torch" in sys.modules
    catalog = SUMMARY.operator_catalog()
    assert len(catalog) == 21
    assert len(SUMMARY._expected(catalog)) == 144
    result = SUMMARY.summarize(tmp_path)
    assert result["expected_backend_oom_boundaries"] == 288
    assert result["paired_ok_points"] == 0
    assert result["status"] == "partial"
    assert ("torch" in sys.modules) == before
