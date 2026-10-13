import pytest
import torch
from e3nn import o3
from test_equivariant_cuda import check

from fasteq.adapters import adapt_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_large_uvw():
    # dW contains 64**3 entries: exceeds the old local_product grid.y limit.
    m = o3.TensorProduct("64x0e", "64x0e", "64x0e", [(0, 0, 0, "uvw", True)]).double()
    check(m, [torch.randn(2, 64, device="cuda", dtype=torch.float64) for _ in range(2)])


def test_o2_u1u():
    from eqx.o2 import TensorProduct

    m = TensorProduct("3x1me", "1x0ee", "3x1me", [(0, 0, 0, "u1u", True)]).double()
    check(m, [torch.randn(3, n, device="cuda", dtype=torch.float64) for n in (6, 1)])


def test_stored_cg_precision():
    # Unequal multiplicities select the generic CUDA contraction, keeping this
    # regression independent of the Uniform1D specialization cache.
    m = o3.TensorProduct("2x2e", "3x2e", "4x2e", [(0, 0, 0, "uvw", True)]).double()
    check(
        m,
        [torch.randn(2, n, device="cuda", dtype=torch.float64) for n in (10, 15)],
        atol=1e-11,
        rtol=1e-10,
    )


def test_prophet_embedded_backend_rejected():
    from prophet.model import TPConvolution

    # A loaded OEQ convolution must fail explicitly, before inplace conversion.
    m = TPConvolution.__new__(TPConvolution)
    torch.nn.Module.__init__(m)
    m.kernel = True
    with pytest.raises(NotImplementedError, match="kernel=False"):
        adapt_model(m, inplace=True)
    assert not hasattr(m, "_fasteq_equivariant")


def test_cg_plan_after_state_load_and_dtype_change():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        reference = o3.TensorProduct(
            "2x2e", "3x2e", "4x2e", [(0, 0, 0, "uvw", True)]
        ).cuda()
    finally:
        torch.set_default_dtype(previous)
    native = adapt_model(reference)
    # Round the stored reference table, then load it after plan construction.
    reference.float().double()
    native.load_state_dict(reference.state_dict())
    x, y = [torch.randn(2, n, device="cuda", dtype=torch.float64) for n in (10, 15)]
    torch.testing.assert_close(native(x, y), reference(x, y), atol=1e-11, rtol=1e-10)
    # Changing the dtype after a kernel was cached must also invalidate it.
    native.float()
    reference.float()
    torch.testing.assert_close(
        native(x.float(), y.float()),
        reference(x.float(), y.float()),
        atol=3e-6,
        rtol=3e-5,
    )
