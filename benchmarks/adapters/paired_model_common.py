"""Shared validation/provenance helpers for new frozen-baseline experiments."""

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from benchmark_models import configure, execute

PARAMETER_COUNTS = {"tace": 44, "tece": 57, "prophet": 81}
SOURCE_DIRS = (
    "src/FastEq/fasteq/adapters", "src/FastEq/fasteq/ops",
    "src/FastEq/fasteq/triton", "src/FastEq/fasteq/cuda",
    "src/tace/tace/models", "src/tace/eqx", "src/prophet/prophet",
)
SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".c", ".cu", ".h", ".hpp", ".cuh"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes(root):
    return {str(path.relative_to(root)): sha256(path)
            for relative in SOURCE_DIRS
            for path in sorted((root / relative).rglob("*"))
            if path.is_file() and path.suffix in SOURCE_SUFFIXES
            and "__pycache__" not in path.parts and "generated" not in str(path)}


def load_spy(root):
    path = root / "src/FastEq/tests/adapters/eqx_cuda_spy.py"
    spec = importlib.util.spec_from_file_location("_paired_model_eqx_cuda_spy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forbid_eqx_cuda, path


def reference(model, name, data, mode):
    configure(model, name, mode)
    output = execute(model, name, data, mode)
    values = {key: value.detach().clone() if value is not None else None
              for key, value in output.items()}
    gradients = {key: value.grad.detach().clone() if value.grad is not None else None
                 for key, value in model.named_parameters()} if mode == "force_loss" else None
    model.zero_grad(set_to_none=True)
    return values, gradients


def validate(model, name, data, mode, dtype, expected, guard=None):
    """Check every named parameter, using the established physical-model tolerances."""
    outputs, gradients = expected
    configure(model, name, mode)
    with guard() if guard is not None else nullcontext():
        actual = execute(model, name, data, mode)
    atol, rtol = (2e-4, 2e-3) if dtype == torch.float32 else (2e-8, 2e-7)
    errors = {}
    assert tuple(actual) == tuple(outputs), "Model output key mismatch"
    for key, baseline in outputs.items():
        if baseline is None:
            assert actual[key] is None, key
        else:
            assert torch.isfinite(baseline).all() and torch.isfinite(actual[key]).all(), key
            torch.testing.assert_close(baseline, actual[key], atol=atol, rtol=rtol)
            errors[key] = (baseline - actual[key]).abs().max().item()
    if gradients is not None:
        parameters = dict(model.named_parameters())
        assert tuple(parameters) == tuple(gradients), "Named parameter key/order mismatch"
        assert len(parameters) == len(gradients) == PARAMETER_COUNTS[name]
        difference = magnitude = 0.0
        nonzero, zero, reference_none, actual_none = [], [], [], []
        for key, parameter in parameters.items():
            baseline, value = gradients[key], parameter.grad
            assert torch.isfinite(parameter).all(), key
            if value is not None:
                assert torch.isfinite(value).all(), key
            if baseline is None:
                reference_none.append(key)
                assert value is None or value.count_nonzero() == 0, key
            else:
                assert torch.isfinite(baseline).all(), key
                assert value is not None and value.shape == baseline.shape, key
                difference += (value.double() - baseline.double()).square().sum().item()
                magnitude += baseline.double().square().sum().item()
                if baseline.count_nonzero():
                    nonzero.append(key)
                else:
                    zero.append(key)
            if value is None:
                actual_none.append(key)
        errors["gradient_relative_l2"] = (difference / max(magnitude, 1e-300)) ** 0.5
        assert errors["gradient_relative_l2"] < (2e-4 if dtype == torch.float32 else 2e-7), errors
        errors["named_parameters_checked"] = len(parameters)
        errors["reference_nonzero_gradient_parameters"] = len(nonzero)
        errors["reference_nonzero_gradient_parameter_names"] = nonzero
        errors["reference_zero_gradient_parameters"] = zero
        errors["reference_none_parameters"] = reference_none
        errors["actual_none_parameters"] = actual_none
        errors["all_parameter_values_and_present_gradients_finite"] = True
    if guard is not None:
        errors["eqx_cuda_spy"] = "passed"
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return errors
