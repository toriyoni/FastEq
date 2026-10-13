"""Energy/force/stress and force-loss parameter-gradient integration checks."""

import json
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from fasteq.adapters import adapt_model, adaptation_report

for name in ("EQX", "OEQ", "CUE", "EQT"):
    os.environ["TACE_USE_" + name] = "0"
os.environ.setdefault("FASTEQ_UNIFORM1D_FWD_TUNE", "0")
os.environ.setdefault("FASTEQ_UNIFORM1D_BWD_TUNE", "0")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def compare(reference, call, dtype, label):
    model = adapt_model(reference)
    model.load_state_dict(reference.state_dict(), strict=True)
    assert model.state_dict().keys() == reference.state_dict().keys()
    refs = dict(reference.named_parameters())
    assert refs.keys() == dict(model.named_parameters()).keys()
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
            assert not hasattr(module, "_fasteq_plan")
    calls = Counter()
    handles = []
    for row in adaptation_report(model):
        if row["module"] in dict(model.named_modules()):
            m = dict(model.named_modules())[row["module"]]
            handles.append(
                m.register_forward_hook(
                    lambda m, i, o, n=row["module"]: calls.update([n])
                )
            )
    outputs = []
    for m in (reference, model):
        output = call(m)
        outputs.append(output)
        sum(v.square().sum() for v in output.values() if v is not None).backward()
    atol, rtol = (3e-5, 3e-4) if dtype == torch.float32 else (2e-8, 2e-7)
    errors = {}
    for key, value in outputs[0].items():
        if value is not None:
            torch.testing.assert_close(value, outputs[1][key], atol=atol, rtol=rtol)
            errors[key] = (value - outputs[1][key]).abs().max().item()
    maxgrad = 0.0
    grad_error_squared = grad_squared = 0.0
    for name, p in model.named_parameters():
        expected = refs[name].grad
        if expected is None:
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, name
        else:
            assert p.grad is not None, name
            torch.testing.assert_close(
                expected,
                p.grad,
                atol=atol * 10,
                rtol=rtol * 10,
                msg=lambda msg, name=name: name + "\n" + msg,
            )
            maxgrad = max(maxgrad, (expected - p.grad).abs().max().item())
            grad_error_squared += (
                (expected.double() - p.grad.double()).square().sum().item()
            )
            grad_squared += expected.double().square().sum().item()
    for handle in handles:
        handle.remove()
    report = {
        "model": label,
        "dtype": str(dtype),
        "max_abs_errors": errors,
        "max_parameter_gradient_abs_error": maxgrad,
        "parameter_gradient_relative_l2_error": (
            grad_error_squared / max(grad_squared, 1e-300)
        )
        ** 0.5,
        "adapted": adaptation_report(model),
        "forward_calls": dict(calls),
    }
    if os.environ.get("FASTEQ_VALIDATION_DIR"):
        path = Path(os.environ["FASTEQ_VALIDATION_DIR"])
        path.mkdir(exist_ok=True, parents=True)
        (path / (label + "_" + str(dtype).split(".")[-1] + ".json")).write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("adapted", "forward_calls")}
        )
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_prophet(dtype):
    from prophet import Prophet

    torch.manual_seed(32)
    model = (
        Prophet(
            n_species=2,
            lmax=2,
            hidden_irreps="4x0e+4x1o+4x2e",
            n_layers=2,
            radial_basis_size=4,
            radial_mlp_size=8,
            radial_mlp_layers=2,
            cutoffs=[2.0, 4.0],
            kernel=False,
        )
        .cuda()
        .to(dtype)
        .train()
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]], device="cuda", dtype=dtype
    )
    edges = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device="cuda")

    def call(m):
        e, f, s = m(
            torch.tensor([0, 1, 0], device="cuda"),
            positions.clone(),
            torch.zeros(6, 3, device="cuda", dtype=dtype),
            edges,
            None,
            torch.tensor([3], device="cuda"),
            torch.tensor([6], device="cuda"),
            torch.zeros(3, dtype=torch.long, device="cuda"),
        )
        return {"energy": e, "forces": f, "stress": s}

    compare(model, call, dtype, "prophet")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tece", [False, True])
def test_tace_tece(dtype, tece):
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    torch.manual_seed(33)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=4,
        Lmax=2,
        lmax=2,
        mmax=1,
        statistics=[
            {"atomic_numbers": [1], "avg_num_neighbors": 2.0, "atomic_energy": {1: 0.0}}
        ],
        target_property=["energy", "forces", "stress", "virials"],
    )
    config["atomic_basis"].update(
        type=["cgtp", "so2"] if tece else "cgtp",
        num_head=2,
        use_radial_rotary_attention=tece,
        use_asymmetric_contraction=True,
    )
    config["node_embedding"]["type"] = "linear"
    config["radial_basis"].update(hidden=[4], bias=True, apply_cutoff=False)
    config["readout_emlp"].update(hidden=[3], use_one_body_magmoms=False)
    config["scale_shift"]["enable"] = False
    config["product_basis"]["correlation"] = 3
    config["layer_norm"].update(
        pre_norm_type="merge_layer_norm", final_norm_type="merge_rms_norm"
    )
    model = TensorModel(e3nnTACE(**config)).cuda().to(dtype).train()
    data = {
        "positions": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]],
            device="cuda",
            dtype=dtype,
        ),
        "node_attrs": torch.ones(3, 1, device="cuda", dtype=dtype),
        "edge_index": torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], device="cuda"
        ),
        "edge_shifts": torch.zeros(6, 3, device="cuda", dtype=dtype),
        "lattice": torch.eye(3, device="cuda", dtype=dtype).unsqueeze(0) * 8,
        "batch": torch.zeros(3, dtype=torch.long, device="cuda"),
        "ptr": torch.tensor([0, 3], device="cuda"),
        "fidelity_idx": torch.zeros(1, dtype=torch.long, device="cuda"),
    }

    def call(m):
        out = m({k: v.clone() for k, v in data.items()})
        return {k: out[k] for k in ("energy", "forces", "stress", "virials")}

    compare(model, call, dtype, "tece" if tece else "tace")
