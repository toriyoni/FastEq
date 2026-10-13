"""Opt-in CUDA adaptation of TACE/TECE and Prophet equivariant operators.

EQX model classes may supply interface metadata; execution uses FastEq. No global backend flags,
scalar MLP, graph softmax, neighbor construction or checkpoints are changed.
"""

from copy import deepcopy
from types import MethodType

import torch
from e3nn import o3
from e3nn.nn import Gate as E3Gate
from eqx.o2 import Gate as O2Gate
from eqx.o2 import Linear as O2Linear
from eqx.o2 import TensorProduct as O2TP

from fasteq.ops.gate import GatePlan

from .harmonics import spherical_harmonics
from .programs import FFNPlan, LinearPlan, NormPlan
from .tensor_product import TensorProductPlan


def _linear(self, x, weight=None, bias=None):
    return self._fasteq_plan(self, x, weight, bias)


def _tp(self, x, y, weight=None):
    return self._fasteq_plan(self, x, y, weight)


def _product_wrapper(self, x, y, weight=None):
    return self.tp(x, y, weight)


def _scatter_wrapper(self, x, y, weight, edge_index):
    message = self.tp(x[edge_index[0]], y, weight)
    return message.new_zeros((x.shape[0], self.irreps_out.dim)).index_add(
        0, edge_index[1], message
    )


def _gate(self, x):
    return self._fasteq_plan(x)


def _norm(self, x):
    return self._fasteq_plan(self, x)


def _ffn(self, scalars_feat, irreps_feat):
    return self._fasteq_plan(self, scalars_feat, irreps_feat)


def _sh(self, x):
    return spherical_harmonics(self._ls_list, x, self.normalize, self.normalization)


def _wigner_packed(self, vectors, *, method=None):
    from fasteq.ops.rotation import wigner_packed

    return wigner_packed(vectors, self.lmax)


def _wigner_blocks(self, vectors, *, method=None):
    packed = _wigner_packed(self, vectors, method=method)
    result, offset = [], 0
    for degree in range(self.lmax + 1):
        size = 2 * degree + 1
        result.append(
            packed[:, offset : offset + size * size].reshape(
                vectors.shape[0], size, size
            )
        )
        offset += size * size
    return result


def _validate_model(model):
    # Fail before mutating an inplace model; these optional families do not
    # execute the O(3)/TECE operators translated by this adapter.
    unsupported_interactions = {
        "O3CartesianIctpIctcInteraction",
        "O2CgtpInteraction",
        "UvO2Interaction",
        "UuO2Interaction",
        "O2MagneticInteraction",
    }
    unsupported_o2 = {
        "UuLinear",
        "LocalFrame",
        "O3TensorProduct",
        "AsymmetricContraction",
        "CircularHarmonics",
    }
    for name, m in model.named_modules():
        cls, module = type(m).__name__, type(m).__module__
        if module == "prophet.model" and cls == "TPConvolution" and m.kernel:
            raise NotImplementedError(
                "Load/build Prophet with kernel=False before FastEq adaptation; "
                "kernel=True embeds an OpenEquivariance convolution."
            )
        if (
            (module == "tace.models._e3nn.inter" and cls in unsupported_interactions)
            or (module.startswith("eqx.o2.") and cls in unsupported_o2)
            or module.startswith(("eqx.co3.", "eqx.conv.co3."))
        ):
            raise NotImplementedError(
                f"{name}: {module}.{cls} is outside the cgtp/so2 adapter coverage"
            )


def adapt_model(model, *, inplace=False):
    """Convert a model instance, preserving parameter names and state_dict.

    CUDA FP32/FP64 only. The default copies the model; inplace=True preserves
    parameter identity for optimizers created before conversion. Forward,
    backward and force-loss double backward use native CUDA. FastEq Uniform1D
    handles uniform uvu/uuu/u1u products including O(2); FastEq supplemental
    kernels handle other coupling layouts, normalization and indexed Linear.
    Unsupported activations or product modes raise during conversion.
    Use ``adaptation_report(model)`` to inspect the concrete backend per module.
    """
    # ScriptFunctions are immutable, but e3nn's SH callable cannot be pickled.
    memo = {
        id(v): v
        for m in model.modules()
        for v in vars(m).values()
        if isinstance(v, torch.jit.ScriptFunction)
    }
    if getattr(model, "_fasteq_backend_version", None) == "fasteq-optimized-v6":
        return model if inplace else deepcopy(model, memo)
    _validate_model(model)
    result = model if inplace else deepcopy(model, memo)
    # Re-conversion of a mixed-backend instance must replace its old plans.
    for m in result.modules():
        for attribute in ("_fasteq_postprocess", "_fasteq_fused_norms"):
            if hasattr(m, attribute):
                delattr(m, attribute)
    report, skip = [], []
    for name, m in list(result.named_modules()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in skip):
            continue
        cls, module = type(m).__name__, type(m).__module__
        if module.startswith("tace."):
            m._fasteq_equivariant = True
        plan = forward = None
        backend = "FastEq CUDA/Triton"
        if isinstance(m, (E3Gate, O2Gate)) or (
            cls == "Gate" and module == "prophet.model"
        ):
            kind = (
                "o2"
                if isinstance(m, O2Gate)
                else "prophet"
                if module == "prophet.model"
                else "o3"
            )
            plan, forward = GatePlan(m, kind), _gate
            skip.extend((name + "." if name else "") + n for n, _ in m.named_children())
        elif isinstance(m, (o3.Linear, O2Linear)):
            plan, forward = LinearPlan(m, isinstance(m, O2Linear)), _linear
        elif isinstance(m, (o3.TensorProduct, O2TP)):
            plan, forward = TensorProductPlan(m, isinstance(m, O2TP)), _tp
            backend = plan.backend
        elif module == "tace.models.linear" and cls in (
            "e3nnElementLinear",
            "e3nnMoEElementLinear",
        ):
            from fasteq.ops.indexed_linear import IndexedLinear

            m._fasteq_linear = IndexedLinear(
                m.linear, num_experts=getattr(m, "num_experts", 1)
            )
            backend = "FastEq indexed/MoE CUDA Linear"
            skip.append((name + "." if name else "") + "linear")
        elif cls == "EquivariantMergeLayerNorm" and module.startswith("tace."):
            plan, forward = NormPlan(m, True), _norm
            backend = "FastEq equivariant norm (extended precision/second derivatives)"
        elif cls == "RMSLayerNorm" and module == "prophet.layer_norm":
            plan, forward = NormPlan(m), _norm
            backend = "FastEq equivariant norm (source statistics/affine parameters)"
        elif cls == "EQFFN" and module == "prophet.model":
            plan, forward = FFNPlan(m), _ffn
        elif isinstance(m, o3.SphericalHarmonics):
            forward = _sh
            backend = (
                "FastEq spherical-harmonic kernels reused from existing FastEq branch"
            )
        elif cls == "WignerD" and module.startswith("eqx."):
            m.forward_packed = MethodType(_wigner_packed, m)
            m.matrix_blocks = MethodType(_wigner_blocks, m)
            backend = "FastEq native CUDA Wigner/alignment"
        elif cls == "Convolution" and module.endswith("tece_oam_rra"):
            if m._eqx_rotation_in is None or m._eqx_rotation_out is None:
                raise NotImplementedError(
                    "TECE rotation requires matching degree metadata"
                )
            m._fasteq_equivariant = True
            backend = "FastEq native CUDA frame/rotary operations"
        else:
            if cls in ("UuuTensorProduct", "UvuTensorProduct") and module.startswith(
                "tace."
            ):
                m.forward = MethodType(_product_wrapper, m)
            if cls == "O3ScatterTensorProduct" and module.startswith("tace."):
                m.forward = MethodType(_scatter_wrapper, m)
            if cls == "Prophet" and module == "prophet.model":
                m._fasteq_spherical_harmonics = spherical_harmonics
                report.append(
                    {
                        "module": (name + "." if name else "") + "spherical_harmonics",
                        "operator": "SphericalHarmonics",
                        "backend": "FastEq existing spherical-harmonic kernels",
                    }
                )
            continue
        if plan is not None:
            m._fasteq_plan = plan
            backend = getattr(plan, "backend", backend)
        if forward is not None:
            m.forward = MethodType(forward, m)
        report.append({"module": name, "operator": cls, "backend": backend})
    from .fusion import install_prophet_fusions

    report.extend(install_prophet_fusions(result))
    result._fasteq_report = report
    result._fasteq_backend_version = "fasteq-optimized-v6"
    return result


def adaptation_report(model):
    """Return conversion provenance, without claiming kernels were executed."""
    return deepcopy(getattr(model, "_fasteq_report", []))


__all__ = ["adapt_model", "adaptation_report", "spherical_harmonics"]
