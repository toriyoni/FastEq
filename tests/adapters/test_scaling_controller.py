"""CPU-only tests of capacity-boundary classification and resume semantics."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

CONTROLLER = Path(__file__).resolve().parents[2] / "benchmarks/adapters/run_scaling_until_oom.py"
spec = importlib.util.spec_from_file_location("_scaling_controller_test", CONTROLLER)
controller = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controller
spec.loader.exec_module(controller)
KEY = "model/tece/float32/force_loss"


def _row():
    return controller.series_catalog([KEY])[KEY]


def _apply(row, status, path, **extra):
    job = controller.next_job(KEY, row)
    result = {"status": status, **extra}
    if status == "oom" and "oom" not in result:
        result["oom"] = {"is_cuda_oom": True, "stage": "timing_round_2"}
    action = controller.apply_result(row, job, result, path)
    return job, action


def test_full_catalog_and_small_subset():
    assert len(controller.series_catalog()) == 144
    assert len(controller.series_catalog(["model/*/*/*"])) == 18
    assert len(controller.series_catalog(["operator/*/*/*"])) == 126
    assert list(controller.series_catalog([KEY])) == [KEY]


def test_paired_oom_never_claims_backend_capacity():
    row = _row()
    _apply(row, "ok", "pair64.json")
    _apply(row, "oom", "pair128.json")
    assert row["phase"] == "single" and row["next_atoms"] == 128
    assert all(boundary["first_oom_atoms"] is None for boundary in row["boundaries"].values())
    assert all(boundary["last_success_atoms"] is None for boundary in row["boundaries"].values())
    assert not row["completed"]


def test_confirms_last_success_and_continues_survivor():
    row = _row()
    _apply(row, "ok", "pair64.json")
    _apply(row, "oom", "pair128.json")
    job, _ = _apply(row, "oom", "original128.json")
    assert job["variant"] == "original"
    assert not row["boundaries"]["original"]["completed"]
    _apply(row, "ok", "v6128.json")
    job, _ = _apply(row, "ok", "original64.json")
    assert job["role"] == "boundary_confirmation" and job["atoms"] == 64
    assert row["boundaries"]["original"]["last_success_point"] == "original64.json"
    assert row["active_variants"] == ["v6"]
    _apply(row, "ok", "v6256.json")
    _apply(row, "oom", "v6512.json")
    assert row["completed"] and row["next_atoms"] is None
    assert row["boundaries"]["original"]["first_oom_atoms"] == 128
    assert row["boundaries"]["v6"]["first_oom_atoms"] == 512
    assert row["boundaries"]["v6"]["last_success_atoms"] == 256
    assert all(boundary["last_success_confirmed"] for boundary in row["boundaries"].values())


def test_first_size_oom_has_no_invented_last_success():
    row = _row()
    _apply(row, "oom", "pair64.json")
    _apply(row, "oom", "original64.json")
    _apply(row, "oom", "v664.json")
    assert row["completed"]
    assert all(boundary["first_oom_atoms"] == 64 for boundary in row["boundaries"].values())
    assert all(boundary["last_success_atoms"] is None for boundary in row["boundaries"].values())


@pytest.mark.parametrize("result", [
    {"status": "timeout"}, {"status": "error", "error_kind": "cpu_oom"},
    {"status": "error", "error_kind": "index_limit"},
    {"status": "error", "error_kind": "validation"},
    {"status": "oom", "oom": {"is_cuda_oom": False}},
])
def test_paired_non_cuda_failure_switches_to_independent_probes(result):
    row = _row()
    job = controller.next_job(KEY, row)
    controller.apply_result(row, job, result, "failure.json")
    assert row["phase"] == "single" and row["status"] == "running"
    assert not row["completed"]
    assert controller.next_job(KEY, row)["variant"] == "original"
    assert all(boundary["first_oom_atoms"] is None for boundary in row["boundaries"].values())


def test_original_index_limit_does_not_stop_v6_capacity_scan():
    row = _row()
    _apply(row, "oom", "pair64.json")
    _apply(row, "error", "original64.json", error_kind="index_limit")
    assert row["boundaries"]["original"]["status"] == "incomplete_error"
    assert row["active_variants"] == ["v6"]
    _apply(row, "ok", "v664.json")
    _apply(row, "ok", "v6128.json")
    _apply(row, "oom", "v6256.json")
    assert row["finalized"] and not row["completed"]
    assert row["status"] == "incomplete_error" and not row["active_variants"]
    assert row["boundaries"]["original"]["first_oom_atoms"] is None
    assert row["boundaries"]["v6"]["first_oom_atoms"] == 256
    assert row["boundaries"]["v6"]["last_success_atoms"] == 128
    assert row["boundaries"]["v6"]["completed"]


def test_source_drift_requests_global_failstop():
    row = _row()
    _, action = _apply(row, "error", "drift.json", error_kind="source_drift")
    assert action == "source_drift_failstop"
    assert row["phase"] == "paired" and row["status"] == "incomplete_error"


def test_validation_gap_is_recorded_but_capacity_continues():
    row = _row()
    _apply(row, "error", "validation.json", error_kind="validation")
    assert row["phase"] == "single"
    assert row["paired_timing_allowed"] is False
    assert row["accuracy_gaps"][0]["point"] == "validation.json"
    assert not row["completed"]


@pytest.mark.parametrize("status", ["shared_gpu", "interrupted"])
def test_transient_failure_retries_identical_job(status):
    row = _row()
    before = controller.next_job(KEY, row)
    _apply(row, status, "invalid.json")
    assert controller.next_job(KEY, row) == before
    assert not row["completed"] and not row["errors"]


def test_confirmation_oom_tightens_boundary():
    row = _row()
    _apply(row, "ok", "pair64.json")
    _apply(row, "ok", "pair128.json")
    _apply(row, "oom", "pair256.json")
    _apply(row, "oom", "original256.json")
    _apply(row, "oom", "v6256.json")
    _apply(row, "oom", "original128.json")
    _apply(row, "ok", "v6128.json")
    _apply(row, "ok", "original64.json")
    assert row["completed"]
    assert row["boundaries"]["original"]["first_oom_atoms"] == 128
    assert row["boundaries"]["original"]["last_success_atoms"] == 64


def test_only_idle_allowed_gpu_is_selected():
    inventory = [{"index": 0, "uuid": "GPU-busy", "compute_pids": [7], "free_mib": 80000},
                 {"index": 1, "uuid": "GPU-idle", "compute_pids": [], "free_mib": 79000}]
    assert controller.available_gpu(inventory)["index"] == 1
    assert controller.available_gpu(inventory, [0]) is None


def _manifest(tmp_path, worker):
    frozen = tmp_path / "v6_manifest.json"
    frozen.write_text('{}\n')
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "status": "frozen_ready", "source_manifest_v6_path": str(frozen),
        "source_manifest_v6_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "new_files_sha256": {str(worker): hashlib.sha256(worker.read_bytes()).hexdigest()},
    }))
    return path


def test_manifest_rejects_driver_drift(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text('print("test")\n')
    path = _manifest(tmp_path, worker)
    controller.read_manifest(path, tmp_path)
    worker.write_text('print("changed")\n')
    with pytest.raises(RuntimeError, match="source SHA mismatch"):
        controller.read_manifest(path, tmp_path)


def test_resume_preserves_results_without_gpu_work(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text('''import json,sys
from pathlib import Path
args=sys.argv[1:]
def value(key):return args[args.index(key)+1]
Path(value("--output")).write_text(json.dumps({"status":"ok","atoms":int(value("--atoms"))}))
''')
    manifest = _manifest(tmp_path, worker)
    monkeypatch.setattr(controller, "nvidia_inventory", lambda: [{
        "index": 3, "uuid": "GPU-test", "free_mib": 80000, "compute_pids": []}])
    command = ["--campaign", str(tmp_path), "--root", str(tmp_path), "--manifest", str(manifest),
               "--worker", str(worker), "--summarizer", str(tmp_path / "not-ready.py"),
               "--only", KEY, "--max-points", "1", "--poll-seconds", "0.01"]
    controller.main(command)
    first = json.loads((tmp_path / "state.json").read_text())
    first_path = tmp_path / first["series"][KEY]["points"][0]
    original_bytes = first_path.read_bytes()
    controller.main(command + ["--resume"])
    resumed = json.loads((tmp_path / "state.json").read_text())
    assert resumed["points_total"] == 2
    assert resumed["series"][KEY]["next_atoms"] == 256
    assert first_path.read_bytes() == original_bytes
    assert not resumed["performance_completed"]
    assert resumed["status"] == "partial_development_limit"


def test_list_and_dry_run_do_not_need_manifest(tmp_path, capsys):
    controller.main(["--campaign", str(tmp_path / "absent"), "--only", KEY, "--list"])
    assert capsys.readouterr().out.strip() == KEY
    controller.main(["--campaign", str(tmp_path / "absent"), "--only", KEY, "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["series_count"] == 1
    assert not (tmp_path / "absent").exists()
