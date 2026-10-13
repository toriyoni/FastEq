"""Original/current-v6 model cases for fresh-process atom-count OOM sweeps.

Only new orchestration imports this file. Existing benchmark drivers and
production code are unchanged. Each measured call retains the exact previous
energy, forces and force-loss computation; no optimizer step is added.
"""

import hashlib
import os
from dataclasses import dataclass, field
from functools import partial

for _flag in ("FASTEQ_EXPERIMENTAL_LINEAR_SIMT", "FASTEQ_EXPERIMENTAL_NATIVE_MODULATION"):
    os.environ.setdefault(_flag, "1")
    if os.environ[_flag] != "1":
        raise RuntimeError(_flag + " must be 1 for the current-v6 scaling comparison")

import torch
from benchmark_models import build_model, configure, execute
from paired_model_common import PARAMETER_COUNTS
from scaling_geometry import build_scaling_data

from fasteq.adapters import adapt_model, adaptation_report

EXPERIMENTAL_FLAGS = ("FASTEQ_EXPERIMENTAL_LINEAR_SIMT", "FASTEQ_EXPERIMENTAL_NATIVE_MODULATION")
MODES = ("energy", "forces", "force_loss")
LABELS = ("original", "v6")
MODEL_CONFIG = {"channels": 64, "neighbors": 12, "layers": 2, "lmax": 2, "correlation": 2}


def _state_sha256(model):
    """Small model weights only; no atom-scaled tensors are copied to CPU."""
    sha = hashlib.sha256()
    for name, value in model.state_dict().items():
        sha.update(name.encode())
        sha.update(str(value.dtype).encode())
        sha.update(str(tuple(value.shape)).encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


@dataclass
class ScalingModelCase:
    models: dict
    data: dict
    metadata: dict
    callables: dict = field(default_factory=dict)

    def run(self, label):
        return self.callables[label]()

    def clear_gradients(self):
        for model in self.models.values():
            model.zero_grad(set_to_none=True)


def build_case(model_name, atoms, dtype, device, mode, *, variant=None, stage_callback=None):
    """Build one independent variant, or both variants for small validation.

    The OOM worker should pass variant='original' or variant='v6' in a fresh
    process. Returning both is convenient for small correctness comparisons
    but must not be used to establish a variant's individual memory boundary.
    Optional stage_callback receives model_setup/adaptation/geometry_setup/
    configuration, allowing setup OOM to remain distinct from execution OOM.
    The caller owns exclusivity checks, timing and process isolation.
    """
    if model_name not in PARAMETER_COUNTS:
        raise ValueError("unknown model: " + str(model_name))
    if mode not in MODES:
        raise ValueError("unknown execution mode: " + str(mode))
    if variant is not None and variant not in LABELS:
        raise ValueError("unknown variant: " + str(variant))
    if isinstance(dtype, str):
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        dtype = getattr(torch, dtype)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("model scaling cases require CUDA; geometry supports CPU validation")
    flags = {name: os.environ.get(name) for name in EXPERIMENTAL_FLAGS}
    if any(value != "1" for value in flags.values()):
        raise RuntimeError("Both current-v6 experimental flags must remain enabled")
    stage = stage_callback or (lambda value: None)
    with torch.cuda.device(device):
        stage("model_setup")
        original = build_model(model_name, MODEL_CONFIG["channels"], dtype, MODEL_CONFIG["correlation"])
        state_hash = _state_sha256(original)
        parameter_names = list(dict(original.named_parameters()))
        if len(parameter_names) != PARAMETER_COUNTS[model_name]:
            raise AssertionError("unexpected named parameter count")
        models = {}
        if variant in (None, "original"):
            models["original"] = original
        if variant in (None, "v6"):
            stage("adaptation")
            candidate = adapt_model(original)
            if candidate._fasteq_backend_version != "fasteq-optimized-v6":
                raise AssertionError("current-v6 backend marker mismatch")
            if list(dict(candidate.named_parameters())) != parameter_names:
                raise AssertionError("adaptation changed named parameter keys/order")
            if _state_sha256(candidate) != state_hash:
                raise AssertionError("adaptation changed model weights")
            models["v6"] = candidate
            del candidate
        del original
        stage("geometry_setup")
        data, geometry = build_scaling_data(atoms, MODEL_CONFIG["neighbors"], dtype, device)
        stage("configuration")
        for model in models.values():
            configure(model, model_name, mode)
        callables = {label: partial(execute, model, model_name, data, mode)
                     for label, model in models.items()}
        reports = {label: adaptation_report(model) for label, model in models.items() if label == "v6"}
    metadata = {
        "kind": "model",
        "model": model_name,
        "mode": mode,
        "dtype": str(dtype).removeprefix("torch."),
        "labels": list(models),
        "weight_seed": 1701,
        "state_sha256": state_hash,
        "named_parameter_count": len(parameter_names),
        "named_parameter_names": parameter_names,
        "model_config": MODEL_CONFIG,
        "geometry": geometry,
        "experimental_flags": flags,
        "adaptation_reports": reports,
        "backend_versions": {label: getattr(model, "_fasteq_backend_version", "original")
                             for label, model in models.items()},
        "scope": "prebuilt neighbors; geometry calculations and cloned positions + model; force_loss adds zero_grad and mean(E^2)+mean(F^2).backward(); no optimizer or stress",
        "force_loss_create_graph": mode == "force_loss",
        "energy_grad_enabled": mode != "energy",
        "model_train": mode == "force_loss",
        "original_baseline": {
            "tace": "original PyTorch/e3nn cgtp, optional acceleration disabled",
            "tece": "original PyTorch/e3nn plus automatic EQX CUDA Wigner-D path",
            "prophet": "original PyTorch/e3nn with kernel=False",
        }[model_name],
        "geometry_setup_and_neighbor_construction_timed": False,
        "pair_built_together_for_individual_oom": False if variant is not None else None,
    }
    return ScalingModelCase(models=models, data=data, metadata=metadata, callables=callables)
