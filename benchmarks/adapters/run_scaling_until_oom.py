"""Resume-safe, CPU-only controller for original/v6 scaling to CUDA OOM.

Every point runs in a fresh worker process. Paired OOM never establishes an
individual backend capacity: both variants are then probed separately, and an
OOM boundary needs a successful fresh single-variant N/2 witness. No finite
atom-count cap is imposed. Shared GPUs, timeouts and non-CUDA failures are not
reported as capacity boundaries. This module never imports Torch or NVML.
"""

import argparse
import csv
import fcntl
import fnmatch
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OPERATOR_NAMES = (
    "linear", "gate_o3", "gate_prophet", "gate_o2", "sh", "tace_norm", "prophet_norm",
    "tp_uvu", "tp_uuu", "tp_o2_uuu", "tp_o2_u1u", "tp_uvw", "tp_nonuniform",
    "wigner", "rotate", "rotary", "local_split", "local_product", "invariant",
    "modulated_residual", "residual_gate_norm",
)
MODEL_NAMES = ("tace", "tece", "prophet")
VARIANTS = ("original", "v6")
DEFAULT_ROOT = Path("/public-data/zhouxibo/zxb/tace_prophet_fasteq")
SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def series_catalog(patterns=()):
    result = {}
    for kind, names, modes in (
        ("operator", OPERATOR_NAMES, ("forward", "backward", "force_loss")),
        ("model", MODEL_NAMES, ("energy", "forces", "force_loss")),
    ):
        for name in names:
            for dtype in ("float32", "float64"):
                for mode in modes:
                    key = f"{kind}/{name}/{dtype}/{mode}"
                    if patterns and not any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns):
                        continue
                    result[key] = {
                        "kind": kind, "name": name, "dtype": dtype, "mode": mode,
                        "start_atoms": 64, "next_atoms": 64, "phase": "paired", "active_variants": list(VARIANTS),
                        "boundaries": {label: {
                            "first_oom_atoms": None, "oom_stage": None,
                            "last_success_atoms": None, "last_success_confirmed": False,
                            "status": "pending", "completed": False,
                            "oom_point": None, "last_success_point": None,
                        } for label in VARIANTS},
                        "completed": False, "status": "pending", "points": [],
                        "paired_last_success_atoms": None,
                        "single_next_atoms": {}, "single_success_points": {label: {} for label in VARIANTS},
                        "confirmation_atoms": {}, "point_roles": {}, "errors": [],
                    }
    if not result:
        raise ValueError("--only patterns matched no series")
    return result


def _next_single(row):
    active = row["active_variants"]
    if active:
        label = active[0]
        row["next_variant"] = label
        row["next_atoms"] = row["confirmation_atoms"].get(label, row["single_next_atoms"][label])
    else:
        row["completed"] = all(boundary["completed"] and boundary["status"] == "confirmed_oom"
                               for boundary in row["boundaries"].values())
        row["status"] = "completed" if row["completed"] else "incomplete_error"
        row["finalized"] = True
        row["next_variant"] = None
        row["next_atoms"] = None


def _activate_single(row, atoms):
    row["phase"] = "single"
    row["single_next_atoms"] = {label: atoms for label in VARIANTS}
    for boundary in row["boundaries"].values():
        boundary["status"] = "scanning"
    _next_single(row)


def next_job(key, row):
    if row["completed"] or row["status"] == "incomplete_error":
        return None
    if row["phase"] == "single":
        _next_single(row)
    label = row.get("next_variant") if row["phase"] == "single" else None
    return {
        "series_key": key,
        "kind": row["kind"], "name": row["name"], "dtype": row["dtype"], "mode": row["mode"],
        "atoms": row["next_atoms"], "protocol": row["phase"], "variant": label,
        "role": "boundary_confirmation" if label in row["confirmation_atoms"] else "scaling",
    }


def apply_result(row, job, result, relative_path):
    """Pure state transition used by the controller and CPU regression tests."""
    row["points"].append(relative_path)
    row["point_roles"][relative_path] = {"role": job["role"], "variant": job["variant"]}
    status = result.get("status")
    is_cuda_oom = status == "oom" and result.get("oom", {}).get("is_cuda_oom") is True
    if status == "shared_gpu":
        row["status"] = "waiting_gpu_retry"
        return "retry"
    if status == "interrupted":
        row["status"] = "pending_resume"
        return "retry"
    if result.get("error_kind") == "exclusive_unverified":
        row["status"] = "waiting_gpu_guard_retry"
        return "retry"
    if status not in ("ok", "success") and not is_cuda_oom:
        error = {"point": relative_path, "status": status,
                 "error_kind": result.get("error_kind"),
                 "variant": job["variant"],
                 "message": result.get("error", result.get("message"))}
        row["errors"].append(error)
        if result.get("error_kind") == "source_drift":
            row["status"] = "incomplete_error"
            return "source_drift_failstop"
        if result.get("error_kind") == "validation":
            row.setdefault("accuracy_gaps", []).append(error)
            row["paired_timing_allowed"] = False
        row["status"] = "running"
        if job["protocol"] == "paired":
            _activate_single(row, job["atoms"])
            return "paired_error_requires_independent_capacity_probes"
        boundary = row["boundaries"][job["variant"]]
        boundary["status"] = "incomplete_error"
        boundary["completed"] = False
        boundary["error_point"] = relative_path
        row["confirmation_atoms"].pop(job["variant"], None)
        row["active_variants"].remove(job["variant"])
        _next_single(row)
        return "single_error_continue_other_variant"
    row["status"] = "running"
    atoms, variant = job["atoms"], job["variant"]
    if job["protocol"] == "paired":
        if not is_cuda_oom:
            row["paired_last_success_atoms"] = atoms
            row["next_atoms"] = atoms * 2
            return "paired_success"
        _activate_single(row, atoms)
        return "paired_oom_requires_independent_confirmation"
    boundary = row["boundaries"][variant]
    if not is_cuda_oom:
        row["single_success_points"][variant][str(atoms)] = relative_path
        boundary["last_success_atoms"] = atoms
        boundary["last_success_point"] = relative_path
        boundary["last_success_confirmed"] = True
        if variant in row["confirmation_atoms"]:
            del row["confirmation_atoms"][variant]
            boundary["status"] = "confirmed_oom"
            boundary["completed"] = True
            row["active_variants"].remove(variant)
        else:
            boundary["status"] = "scanning"
            row["single_next_atoms"][variant] = atoms * 2
            row["active_variants"].append(row["active_variants"].pop(0))
        _next_single(row)
        return "single_success"
    # An OOM during N/2 confirmation tightens the observed boundary. Keep
    # descending until a real fresh single success or the initial N64 OOM.
    boundary["first_oom_atoms"] = atoms
    boundary["oom_stage"] = result["oom"].get("stage")
    boundary["oom_point"] = relative_path
    previous = atoms // 2
    success = row["single_success_points"][variant].get(str(previous))
    if atoms == row["start_atoms"] or success is not None:
        boundary["last_success_atoms"] = previous if success is not None else None
        boundary["last_success_point"] = success
        boundary["last_success_confirmed"] = success is not None
        boundary["status"] = "confirmed_oom"
        boundary["completed"] = True
        row["confirmation_atoms"].pop(variant, None)
        row["active_variants"].remove(variant)
    else:
        boundary["last_success_atoms"] = None
        boundary["last_success_point"] = None
        boundary["last_success_confirmed"] = False
        boundary["status"] = "awaiting_last_success_confirmation"
        row["confirmation_atoms"][variant] = previous
        row["active_variants"].append(row["active_variants"].pop(0))
    _next_single(row)
    return "single_cuda_oom"


def nvidia_inventory():
    """Read process ownership without importing CUDA libraries or Torch."""
    gpu = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True, timeout=20)
    processes = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True, timeout=20)
    busy = {}
    for fields in csv.reader(io.StringIO(processes.stdout)):
        if len(fields) >= 2 and fields[0].strip().startswith("GPU-"):
            busy.setdefault(fields[0].strip(), []).append(int(fields[1].strip()))
    inventory = []
    for fields in csv.reader(io.StringIO(gpu.stdout)):
        if len(fields) != 4:
            raise RuntimeError("Unexpected nvidia-smi GPU inventory")
        index, uuid, total, free = (value.strip() for value in fields)
        inventory.append({"index": int(index), "uuid": uuid,
                          "total_mib": int(total), "free_mib": int(free),
                          "compute_pids": sorted(busy.get(uuid, []))})
    return inventory


def available_gpu(inventory, allowed=None):
    candidates = [gpu for gpu in inventory if not gpu["compute_pids"]
                  and (allowed is None or gpu["index"] in allowed)]
    return max(candidates, key=lambda gpu: (gpu["free_mib"], -gpu["index"]), default=None)


def read_manifest(path, root=None):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("status") != "frozen_ready":
        raise RuntimeError("Campaign manifest must be frozen_ready before GPU workers start")
    if not manifest.get("source_manifest_v6_path") or not manifest.get("source_manifest_v6_sha256"):
        raise RuntimeError("Campaign manifest is missing frozen-v6 provenance")
    if not isinstance(manifest.get("new_files_sha256"), dict) or not manifest["new_files_sha256"]:
        raise RuntimeError("Campaign manifest is missing new driver SHA guards")
    base = Path(root or manifest.get("root", DEFAULT_ROOT))
    frozen_path = Path(manifest["source_manifest_v6_path"])
    if not frozen_path.is_absolute():
        frozen_path = base / frozen_path
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != manifest["source_manifest_v6_sha256"]:
        raise RuntimeError("Frozen-v6 source manifest SHA mismatch")
    for name, expected in manifest["new_files_sha256"].items():
        source = Path(name)
        if not source.is_absolute():
            source = base / source
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError("New campaign source SHA mismatch: " + name)
    return manifest, hashlib.sha256(raw).hexdigest()


def worker_command(args, job, gpu, path):
    command = [sys.executable, str(args.worker),
               "--kind", job["kind"], "--name", job["name"], "--dtype", job["dtype"],
               "--mode", job["mode"], "--atoms", str(job["atoms"]), "--protocol", job["protocol"],
               "--gpu", str(gpu["index"]), "--gpu-uuid", gpu["uuid"],
               "--output", str(path), "--rounds", str(args.rounds), "--warmup", str(args.warmup),
               "--manifest", str(args.manifest)]
    if job["variant"] is not None:
        command.extend(("--variant", job["variant"]))
    return command


def _process_start(pid):
    try:
        # /proc comm may contain spaces; parse fields after its closing ')'.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"state": fields[0], "start_ticks": fields[19]}
    except (OSError, IndexError):
        return None


class Controller:
    def __init__(self, args, state):
        self.args, self.state = args, state
        self.stop_requested = False
        self.points_this_run = 0
        self.last_summary_phase = None
        self.state_path = args.campaign / "state.json"

    def save(self):
        self.state["updated_at"] = utc_now()
        atomic_json(self.state_path, self.state)
        atomic_json(self.args.campaign / "status.json", {
            key: self.state.get(key) for key in (
                "status", "updated_at", "performance_completed", "selected_completed",
                "points_total", "running_job", "gpu_inventory", "manifest_sha256",
            )
        })

    def summarize(self, force=False):
        phase = (self.state["status"], self.state.get("selected_completed"))
        if not force and self.points_this_run % 5 and phase == self.last_summary_phase:
            return
        self.last_summary_phase = phase
        script = self.args.summarizer
        if not script.is_file():
            self.state["summary_status"] = "summarizer_not_ready"
            return
        log = self.args.campaign / "summarizer.log"
        try:
            with log.open("a") as stream:
                completed = subprocess.run([sys.executable, str(script), "--campaign", str(self.args.campaign)],
                                           stdout=stream, stderr=subprocess.STDOUT, timeout=60, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            self.state["summary_status"] = "error"
            self.state.setdefault("summarizer_errors", []).append({"at": utc_now(), "error": str(error)})
            return
        self.state["summary_status"] = "ok" if completed.returncode == 0 else "error"
        if completed.returncode:
            self.state.setdefault("summarizer_errors", []).append({"at": utc_now(), "exit_code": completed.returncode})

    def _terminate_owned_worker(self, process):
        # Only the process group created by this controller is targeted.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)

    def _point_path(self, job):
        stem = job["series_key"].replace("/", "__")
        label = job["variant"] or "pair"
        attempt = len(self.state["series"][job["series_key"]]["points"]) + 1
        return self.args.campaign / "points" / f"{stem}__N{job['atoms']}__{label}__a{attempt:03d}.json"

    def _synthetic_result(self, job, status, error_kind, message):
        return {key: value for key, value in {
            **job, "status": status, "error_kind": error_kind,
            "error": message, "timing_fields_empty": True,
            "failure_variant": job["variant"] or "unknown",
            "controller_recorded_failure": True,
        }.items() if key != "series_key"}

    def consume(self, running, returncode=None, fallback=None):
        job = running["job"]
        path = self.args.campaign / running["point"]
        if fallback is not None:
            fallback["id"] = path.stem
            # Preserve any partial worker output rather than overwriting it.
            if path.exists():
                path.replace(path.with_suffix(".partial.json"))
            atomic_json(path, fallback)
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            result = self._synthetic_result(job, "error", "worker_no_result", str(error))
            result["id"] = path.stem
            atomic_json(path, result)
        if returncode not in (None, 0) and result.get("status") in ("ok", "success", "oom"):
            result = self._synthetic_result(job, "error", "worker_exit_nonzero",
                                            f"Worker exited {returncode}; original status is not accepted")
            result["id"] = path.stem
            path.replace(path.with_suffix(".partial.json"))
            atomic_json(path, result)
        action = apply_result(self.state["series"][job["series_key"]], job, result, running["point"])
        self.state["running_job"] = None
        self.state["points_total"] += 1
        self.points_this_run += 1
        self.state["last_result"] = {"point": running["point"], "status": result.get("status"),
                                     "action": action, "worker_exit_code": returncode}
        self.save()
        self.summarize()
        return action

    def recover(self):
        running = self.state.get("running_job")
        if running is None:
            return
        previous = _process_start(running["pid"])
        fallback = None
        while previous is not None and previous["start_ticks"] == running.get("pid_start_ticks") and previous["state"] != "Z":
            self.state["status"] = "waiting_previous_worker"
            self.save()
            if self.stop_requested:
                return
            if time.time() - running["started_epoch"] > self.args.timeout:
                # A recovered process is not our current Popen child. Verify
                # both process identity and its exact owned output argument
                # before terminating the process group recorded at launch.
                fields = Path(f"/proc/{running['pid']}/cmdline").read_bytes().split(b"\0")
                worker = str(self.args.worker).encode()
                output = str(self.args.campaign / running["point"]).encode()
                if worker not in fields or b"--output" not in fields or output not in fields or os.getpgid(running["pid"]) != running["pid"]:
                    raise RuntimeError("Recovered worker identity is unverified; no process was terminated")
                os.killpg(running["pid"], signal.SIGTERM)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    current = _process_start(running["pid"])
                    if current is None or current["state"] == "Z" or current["start_ticks"] != running["pid_start_ticks"]:
                        break
                    time.sleep(0.1)
                else:
                    os.killpg(running["pid"], signal.SIGKILL)
                fallback = self._synthetic_result(running["job"], "timeout", "recovered_worker_timeout",
                                                  f"Owned recovered worker exceeded {self.args.timeout}s; this is not CUDA OOM")
                break
            time.sleep(self.args.poll_seconds)
            previous = _process_start(running["pid"])
        path = self.args.campaign / running["point"]
        if fallback is None and not path.exists():
            fallback = self._synthetic_result(
                running["job"], "interrupted", "controller_interrupted", "Previous worker ended without a complete result; point will be retried")
        self.consume(running, fallback=fallback)

    def execute(self, job, gpu):
        path = self._point_path(job)
        if path.exists():
            raise RuntimeError("Point path already exists; prior evidence must never be overwritten")
        command = worker_command(self.args, job, gpu, path)
        job = {**job, "id": path.stem}
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], FASTEQ_EXPERIMENTAL_LINEAR_SIMT="1",
                           FASTEQ_EXPERIMENTAL_NATIVE_MODULATION="1")
        with path.with_suffix(".log").open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            process_info = _process_start(process.pid)
            running = {"job": job, "point": str(path.relative_to(self.args.campaign)),
                       "pid": process.pid, "pid_start_ticks": process_info["start_ticks"] if process_info else None,
                       "started_at": utc_now(), "started_epoch": time.time(), "gpu": gpu, "command": command}
            self.state["running_job"] = running
            self.state["status"] = "running_worker"
            self.save()
            self.summarize()
            start = heartbeat = time.monotonic()
            fallback = None
            while process.poll() is None:
                if self.stop_requested:
                    self._terminate_owned_worker(process)
                    fallback = self._synthetic_result(job, "interrupted", "controller_stopped", "Owned worker stopped; resume retries this point")
                    break
                if time.monotonic() - start > self.args.timeout:
                    self._terminate_owned_worker(process)
                    fallback = self._synthetic_result(job, "timeout", "worker_timeout", f"Worker exceeded {self.args.timeout}s; this is not CUDA OOM")
                    break
                if time.monotonic() - heartbeat >= 30:
                    self.save()
                    heartbeat = time.monotonic()
                time.sleep(1)
            return self.consume(running, returncode=process.returncode, fallback=fallback)

    def run(self):
        self.recover()
        keys = self.state["requested_series_keys"]
        while not self.stop_requested:
            if self.args.max_points is not None and self.points_this_run >= self.args.max_points:
                self.state["status"] = "partial_development_limit"
                break
            pending = [key for key in keys if not self.state["series"][key]["completed"]
                       and self.state["series"][key]["status"] != "incomplete_error"]
            if not pending:
                self.state["selected_completed"] = all(self.state["series"][key]["completed"] for key in keys)
                self.state["performance_completed"] = (self.state["selected_completed"] and len(keys) == 144
                                                        and self.args.max_points is None and not self.args.only
                                                        and not any(row.get("accuracy_gaps") for row in self.state["series"].values()))
                if self.state["performance_completed"]:
                    self.state["status"] = "completed"
                elif any(row.get("accuracy_gaps") for row in self.state["series"].values()):
                    self.state["status"] = "incomplete_accuracy"
                else:
                    self.state["status"] = "partial_subset_completed" if self.state["selected_completed"] else "incomplete_errors"
                break
            _, current_manifest_sha = read_manifest(self.args.manifest, self.args.root)
            if current_manifest_sha != self.state["manifest_sha256"]:
                raise RuntimeError("Campaign manifest changed while running; no further workers may start")
            try:
                inventory = nvidia_inventory()
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.state["status"] = "waiting_gpu_inventory_error"
                self.state["gpu_inventory_error"] = str(error)
                self.save()
                self.summarize()
                time.sleep(self.args.poll_seconds)
                continue
            self.state["gpu_inventory"] = inventory
            gpu = available_gpu(inventory, self.args.gpu)
            if gpu is None:
                self.state["status"] = "waiting_exclusive_gpu"
                self.save()
                self.summarize()
                time.sleep(self.args.poll_seconds)
                continue
            cursor = self.state.get("series_cursor", 0) % len(keys)
            key = next(keys[(cursor + offset) % len(keys)] for offset in range(len(keys))
                       if keys[(cursor + offset) % len(keys)] in pending)
            self.state["series_cursor"] = (keys.index(key) + 1) % len(keys)
            action = self.execute(next_job(key, self.state["series"][key]), gpu)
            if action == "source_drift_failstop":
                self.state["status"] = "source_drift_failstop"
                break
            if action == "retry":
                time.sleep(self.args.poll_seconds)
        if self.stop_requested:
            self.state["status"] = "stopped_resume_available"
        self.state["selected_completed"] = all(self.state["series"][key]["completed"] for key in keys)
        self.save()
        self.summarize(force=True)
        self.save()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("TACE_PROPHET_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("benchmark_scaling_case.py"))
    parser.add_argument("--summarizer", type=Path, default=Path(__file__).with_name("summarize_scaling.py"))
    parser.add_argument("--gpu", nargs="+", type=int, help="Physical GPU indices allowed; otherwise inspect all GPUs")
    parser.add_argument("--only", nargs="+", default=[], help="fnmatch patterns kind/name/dtype/mode, e.g. 'model/tece/*/force_loss'")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-points", type=int, help="Development limit only; never marks full performance completed")
    parser.add_argument("--resume", action="store_true", help="Require existing state; matching existing state also resumes automatically")
    parser.add_argument("--list", action="store_true", help="Print selected series and exit without GPU initialization")
    parser.add_argument("--dry-run", action="store_true", help="Print protocol/plan and exit without mutations or GPU work")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.campaign = (args.campaign or args.root / "performance/fasteq_optimization/v6/scaling_20261013").resolve()
    args.manifest = (args.manifest or args.campaign / "manifest.json").resolve()
    args.worker, args.summarizer = args.worker.resolve(), args.summarizer.resolve()
    if args.rounds < 9 or args.warmup < 5:
        parser.error("Normal performance requires rounds >=9 and warmup >=5")
    if not 0 < args.poll_seconds <= 60 or args.timeout <= 0:
        parser.error("poll-seconds must be in (0,60], and timeout must be positive")
    if args.max_points is not None and args.max_points <= 0:
        parser.error("max-points must be positive")
    if args.gpu is not None and (len(args.gpu) != len(set(args.gpu)) or any(index < 0 for index in args.gpu)):
        parser.error("gpu indices must be unique and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    selected = series_catalog(args.only)
    if args.list:
        print("\n".join(selected))
        return
    if args.dry_run:
        print(json.dumps({"campaign": str(args.campaign), "series_count": len(selected),
                          "series_keys": list(selected), "atoms": "64,128,256,... until individual CUDA OOM; no max",
                          "paired_oom_policy": "fresh single variants, then successful single N/2 confirmation",
                          "rounds": args.rounds, "warmup": args.warmup,
                          "worker": str(args.worker), "manifest": str(args.manifest),
                          "development_max_points": args.max_points}, indent=2))
        return
    if not args.worker.is_file():
        raise RuntimeError("Worker is not ready; create and freeze it before starting")
    _, manifest_sha = read_manifest(args.manifest, args.root)
    args.campaign.mkdir(parents=True, exist_ok=True)
    (args.campaign / "points").mkdir(exist_ok=True)
    with (args.campaign / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another controller already owns this campaign") from error
        state_path = args.campaign / "state.json"
        request = {"series_keys": list(selected), "rounds": args.rounds, "warmup": args.warmup,
                   "worker": str(args.worker), "manifest": str(args.manifest)}
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("schema_version") != SCHEMA_VERSION or state.get("request") != request:
                raise RuntimeError("Resume request differs from existing campaign; preserve it and use a new campaign")
            if state.get("manifest_sha256") != manifest_sha:
                raise RuntimeError("Frozen manifest differs from existing campaign")
        else:
            if args.resume:
                raise RuntimeError("--resume requires an existing state.json")
            state = {"schema_version": SCHEMA_VERSION, "request": request,
                     "manifest_sha256": manifest_sha, "started_at": utc_now(),
                     "status": "pending", "series": selected, "requested_series_keys": list(selected),
                     "requested_series_count": len(selected), "full_series_count": 144,
                     "performance_completed": False, "selected_completed": False,
                     "points_total": 0, "running_job": None, "series_cursor": 0}
        state["controller"] = {"pid": os.getpid(), "started_at": utc_now(), "gpu_allowlist": args.gpu,
                               "poll_seconds": args.poll_seconds, "timeout": args.timeout,
                               "development_max_points": args.max_points}
        controller = Controller(args, state)

        def stop(signum, frame):
            controller.stop_requested = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        controller.save()
        try:
            controller.run()
        except Exception as error:
            controller.state["status"] = "controller_error"
            controller.state["fatal_error"] = {"type": type(error).__name__, "message": str(error)}
            controller.state["performance_completed"] = False
            controller.save()
            controller.summarize(force=True)
            controller.save()
            raise


if __name__ == "__main__":
    main()
