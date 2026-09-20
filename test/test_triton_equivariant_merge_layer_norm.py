"""Native EQv3 MergeLayerNorm factory/inference and option compatibility."""
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")
from fasteq.triton import from_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP GPU required")


@pytest.fixture(scope="module")
def native():
    value = os.environ.get("FASTEQ_EQUIFORMER_V3_LAYER_NORM")
    if not value:
        pytest.skip("set FASTEQ_EQUIFORMER_V3_LAYER_NORM to the original source")
    path = Path(value)
    spec = importlib.util.spec_from_file_location("_native_merge_norm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def check(source, layout, path, record_property):
    source = source.cuda().eval()
    torch.manual_seed(20260920)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn_like(parameter))
    x = torch.randn(17, (source.lmax + 1) ** 2, source.num_channels, device="cuda")
    if layout == "KNC":
        x = x.transpose(0, 1).contiguous().transpose(0, 1)
    adapter = from_reference(source).eval()
    assert adapter.spec.num_stats == 1
    assert adapter.spec.center_scalar == source.centering
    assert {id(p) for p in adapter.parameters()} == {id(p) for p in source.parameters()}
    record_property("source/sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    record_property("reference", "original EQv3 FP32 inference and first-order autograd")
    record_property("shape", list(x.shape))
    record_property("layout", layout)

    def compare(name, actual, expected):
        error = (actual.double() - expected.double()).abs()
        ratio = error / (5e-5 + 5e-4 * expected.double().abs())
        record_property(name + "/max_abs", error.max().item())
        record_property(name + "/max_tolerance_ratio", ratio.max().item())
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-4)

    with torch.inference_mode():
        compare("inference", adapter(x), source(x))
    x.requires_grad_()
    upstream = torch.randn_like(x)
    named = dict(source.named_parameters())
    actual, expected = adapter(x), source(x)
    compare("forward", actual, expected)
    actual_grads = torch.autograd.grad(actual, (x, *named.values()), upstream)
    expected_grads = torch.autograd.grad(expected, (x, *named.values()), upstream)
    for name, a, b in zip(("dX", *named), actual_grads, expected_grads):
        compare(name, a, b)


@pytest.mark.parametrize("norm_type", (
    "merge_layer_norm", "merge_layer_norm_attn_rms_norm", "merge_rms_norm"))
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
@pytest.mark.parametrize("lmax,channels", ((3, 128), (6, 128)))
def test_native_model_factory(native, norm_type, layout, lmax, channels, record_property):
    module, path = native
    source = module.get_normalization_layer(norm_type, lmax, channels)
    assert type(source).__name__ == "EquivariantMergeLayerNorm"
    record_property("norm_type", norm_type)
    check(source, layout, path, record_property)


@pytest.mark.parametrize("normalization,balanced", (
    ("component", True), ("component", False), ("norm", True)))
@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("affine", (True, False))
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
@pytest.mark.parametrize("lmax,channels", ((0, 1), (2, 7)))
def test_native_options(native, normalization, balanced, center, affine,
                        layout, lmax, channels, record_property):
    module, path = native
    source = module.EquivariantMergeLayerNorm(
        lmax, channels, normalization=normalization,
        std_balance_degrees=balanced, centering=center, affine=affine)
    check(source, layout, path, record_property)
