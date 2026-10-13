"""Prophet residual/Gate/RMS fusion with FastEq composition for other norms."""

from fasteq.ops.gate import GatePlan
from fasteq.ops.gate_norm import GateNormPlan

from .programs import NormPlan


class ProphetPostprocess:
    def __init__(self, convolution, post_norm):
        self.pre_norm = convolution.layer_norm
        self.post_norm = post_norm
        self.dim = convolution.linear_2.irreps_out.dim
        self.outdim = post_norm.irreps.dim
        self.pre = NormPlan(self.pre_norm) if self.pre_norm is not None else None
        self.gate = GatePlan(convolution.gate, "prophet")
        self.post = NormPlan(post_norm)
        self.fused = (GateNormPlan(self.gate, post_norm)
                      if self.pre is None and GateNormPlan.supports(post_norm) else None)

    @property
    def parameters(self):
        values = []
        for norm in (self.pre_norm, self.post_norm):
            if norm is None:
                continue
            for name in ("affine_weight", "affine_bias"):
                container = getattr(norm, name, None)
                if container is not None:
                    values.extend(value for value in container if value is not None)
        return values

    def __call__(self, projected, skip):
        if self.fused is not None and GateNormPlan.supports(self.post_norm):
            return self.fused(self.post_norm, projected, skip)
        value = projected + skip
        if self.pre is not None:
            value = self.pre(self.pre_norm, value)
        return self.post(self.post_norm, self.gate(value))


def install_prophet_fusions(model):
    report = []
    for name, block in model.named_modules():
        if (
            type(block).__module__ != "prophet.model"
            or type(block).__name__ != "ProphetBlock"
        ):
            continue
        for i, (conv, norm) in enumerate(zip(block.tpconv, block.norm)):
            conv._fasteq_postprocess = ProphetPostprocess(conv, norm)
            report.append(
                {
                    "module": (name + "." if name else "") + f"tpconv.{i}",
                    "operator": "ResidualGateRMSNorm",
                    "backend": (conv._fasteq_postprocess.fused.backend
                                if conv._fasteq_postprocess.fused is not None
                                else "FastEq Gate/norm kernel composition (no external executor)"),
                }
            )
        block._fasteq_fused_norms = True
    return report
