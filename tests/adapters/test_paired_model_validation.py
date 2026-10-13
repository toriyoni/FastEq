"""CPU regressions for complete force-loss benchmark validation."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/adapters"))
import paired_model_common as validation


class NamedParameters(torch.nn.Module):
    def __init__(self, count=44):
        super().__init__()
        self.values = torch.nn.ParameterDict({f"p{i:02d}": torch.nn.Parameter(
            torch.tensor([0.5], dtype=torch.float64)
        ) for i in range(count)})


@pytest.fixture
def fixture(monkeypatch):
    model = NamedParameters()
    outputs = {"energy": torch.tensor([1.0], dtype=torch.float64),
               "forces": torch.zeros(1, 3, dtype=torch.float64)}
    gradients = {name: torch.tensor([0.2], dtype=torch.float64)
                 for name, _ in model.named_parameters()}
    actual_gradients = {name: value.clone() for name, value in gradients.items()}

    def execute(model, name, data, mode):
        for key, parameter in model.named_parameters():
            parameter.grad = actual_gradients[key]
        return outputs

    monkeypatch.setattr(validation, "execute", execute)
    monkeypatch.setattr(validation, "configure", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return model, outputs, gradients, actual_gradients


def test_all_parameter_keys_checked_including_last(fixture):
    model, outputs, gradients, _ = fixture
    gradients.pop(next(reversed(gradients)))
    with pytest.raises(AssertionError, match="Named parameter key/order mismatch"):
        validation.validate(model, "tace", None, "force_loss", torch.float64, (outputs, gradients))


def test_last_nonfinite_gradient_rejected(fixture):
    model, outputs, gradients, actual = fixture
    actual[next(reversed(actual))] = torch.tensor([float("nan")], dtype=torch.float64)
    with pytest.raises(AssertionError):
        validation.validate(model, "tace", None, "force_loss", torch.float64, (outputs, gradients))


def test_reference_none_cannot_become_nonzero(fixture):
    model, outputs, gradients, _ = fixture
    gradients[next(reversed(gradients))] = None
    with pytest.raises(AssertionError):
        validation.validate(model, "tace", None, "force_loss", torch.float64, (outputs, gradients))


def test_reference_none_allows_exact_zero_and_checks_all44(fixture):
    model, outputs, gradients, actual = fixture
    last = next(reversed(gradients))
    gradients[last], actual[last] = None, torch.zeros(1, dtype=torch.float64)
    result = validation.validate(model, "tace", None, "force_loss", torch.float64, (outputs, gradients))
    assert result["named_parameters_checked"] == 44
    assert result["reference_none_parameters"] == [last]
    assert result["reference_nonzero_gradient_parameters"] == 43
    assert result["gradient_relative_l2"] == 0
    assert result["all_parameter_values_and_present_gradients_finite"]


def test_equal_infinite_outputs_rejected(fixture):
    model, outputs, gradients, _ = fixture
    outputs["energy"] = torch.tensor([float("inf")], dtype=torch.float64)
    with pytest.raises(AssertionError):
        validation.validate(model, "tace", None, "force_loss", torch.float64, (outputs, gradients))
