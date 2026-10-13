"""Element-indexed CUDA maps preserve LoRA and expert parameter gradients."""

import pytest
import torch
from tace.models.linear import (
    e3nnElementLinear,
    e3nnMoEElementLinear,
    enable_lora,
)
from test_equivariant_cuda import check

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("matrix_weight", ["0", "1"])
@pytest.mark.parametrize("lora", [False, True])
def test_element_linear_native(moe, matrix_weight, lora):
    torch.manual_seed(410)
    cls = e3nnMoEElementLinear if moe else e3nnElementLinear
    m = cls(
        "6x0e+6x1o",
        "9x0e+9x1o",
        bias=True,
        num_elements=3,
        use_matrix_weight=matrix_weight,
        **({"num_experts": 3} if moe else {}),
    ).double()
    if lora:
        enable_lora(m, r=2, freeze_base=False, train_bias=True)
        with torch.no_grad():
            for b in m.lora_B:
                b.normal_(std=0.2)
    with torch.no_grad():
        m.bias.normal_(std=0.2)
    x = torch.randn(4, m.irreps_in.dim, device="cuda", dtype=torch.float64)
    # Soft element attributes exercise the ordinary map's existing bias mixing.
    attrs = torch.tensor(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1]],
        device="cuda",
        dtype=torch.float64,
    )
    calls = []
    m.eqx_linear.register_forward_hook(lambda *_: calls.append(True))
    check(m, [x, attrs])
    # EQX supplies reference interfaces; neither copy uses its CUDA executor.
    assert len(calls) == 0
