"""Prophet energy/forces/stress and force-loss through FastEq-only operators."""

import json
import os
from pathlib import Path

import pytest
import torch
from eqx_cuda_spy import forbid_eqx_cuda

from fasteq.adapters import adapt_model, adaptation_report

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_prophet_all_properties_without_eqx_backend(dtype):
    from prophet import Prophet
    torch.manual_seed(643)
    original = Prophet(n_species=2, lmax=2, hidden_irreps="4x0e+4x1o+4x2e", n_layers=2,
                       radial_basis_size=4, radial_mlp_size=8, radial_mlp_layers=2,
                       cutoffs=[2.0, 4.0], kernel=False).cuda().to(dtype).train()
    adapted = adapt_model(original)
    assert original.state_dict().keys() == adapted.state_dict().keys()
    positions = torch.tensor([[0., 0., 0.], [1., .3, .2], [.4, 1.1, -.2]], dtype=dtype, device="cuda")
    edges = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device="cuda")
    def evaluate(model):
        energy, forces, stress = model(
            torch.tensor([0, 1, 0], device="cuda"), positions.clone(),
            torch.zeros(6, 3, device="cuda", dtype=dtype), edges,
            (torch.eye(3, device="cuda", dtype=dtype) * 8).unsqueeze(0),
            torch.tensor([3], device="cuda"), torch.tensor([6], device="cuda"),
            torch.zeros(3, device="cuda", dtype=torch.long),
        )
        assert stress is not None
        (energy.square().sum() + forces.square().sum() + stress.square().sum()).backward()
        return energy, forces, stress

    # Reference execution may legitimately use its original EQX implementation.
    reference = evaluate(original)
    with forbid_eqx_cuda() as calls:
        fast = evaluate(adapted)
    assert not calls
    values = (reference, fast)
    atol, rtol = (4e-5, 4e-4) if dtype == torch.float32 else (2e-8, 2e-7)
    errors = {}
    for name, left, right in zip(("energy", "forces", "stress"), *values):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
        errors[name] = (left-right).abs().max().item()
    diff = norm = 0.
    for (name, left), (other, right) in zip(original.named_parameters(), adapted.named_parameters()):
        assert name == other
        if left.grad is None:
            assert right.grad is None or not torch.count_nonzero(right.grad)
        else:
            assert right.grad is not None, name
            torch.testing.assert_close(left.grad, right.grad, atol=atol*10, rtol=rtol*10,
                                       msg=lambda msg, name=name: name + "\n" + msg)
            diff += (left.grad.double()-right.grad.double()).square().sum().item()
            norm += left.grad.double().square().sum().item()
    errors["parameter_gradient_relative_l2"] = (diff / max(norm, 1e-300)) ** .5
    report = {"dtype": str(dtype), "properties": errors,
              "eqx_runtime_spy": "compiler/launcher aliases, resident plans, dispatcher ops forbidden; passed",
              "adaptation": adaptation_report(adapted)}
    if os.environ.get("FASTEQ_VALIDATION_DIR"):
        path = Path(os.environ["FASTEQ_VALIDATION_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        (path / ("prophet_owned_" + str(dtype).split(".")[-1] + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "adaptation"}))
