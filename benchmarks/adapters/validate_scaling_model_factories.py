"""Small correctness checks of new model factories; never performance/OOM data."""

import argparse
import gc
import json
import os
from pathlib import Path

from scaling_model_case import build_case

# isort: split
# The factory sets both v6 flags before importing the original model driver.
import pynvml
import torch
from candidate_execution_observers import CandidateExecutionObservers
from paired_model_common import load_spy, reference, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-core", type=int, default=170)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve earlier validation; use a fresh output path"
    os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.025)
    pynvml.nvmlInit()
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
    foreign = sorted(p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid != os.getpid())
    forbid_eqx_cuda, _ = load_spy(Path(os.environ["TACE_PROPHET_ROOT"]))
    observations = CandidateExecutionObservers()
    report = {"scope": "new scaling model factory correctness only; no timings or OOM stress",
              "shared_gpu_correctness_only": bool(foreign), "gpu_uuid": uuid,
              "foreign_compute_pids": foreign, "allocator_fraction": 0.025,
              "timing_fields_empty": True, "performance_completed": False,
              "results": [], "completed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    observations.install()
    try:
        for name in ("tace", "tece", "prophet"):
            for dtype in (torch.float32, torch.float64):
                # TECE N128 exercises the FP32 SIMT predicate (1536 edges).
                atoms = 128 if name == "tece" else 64
                stages = []
                case = build_case(name, atoms, dtype, "cuda:0", "force_loss", stage_callback=stages.append)
                errors = {}
                for mode in ("energy", "forces", "force_loss"):
                    expected = reference(case.models["original"], name, case.data, mode)
                    errors[mode] = {
                        "native_eager": validate(case.models["v6"], name, case.data, mode, dtype, expected),
                        "eqx_spy": validate(case.models["v6"], name, case.data, mode, dtype, expected, forbid_eqx_cuda),
                    }
                    del expected
                report["results"].append({"metadata": case.metadata, "factory_stages": stages, "validation": errors})
                save()
                print(f"VALIDATED {name} {dtype} N{atoms} energy/forces/force_loss", flush=True)
                case.clear_gradients()
                del case
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        observations.close()
        report["candidate_execution_observations"] = observations.report
        report["allocator_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        save()
    report["completed"] = True
    save()


if __name__ == "__main__":
    main()
