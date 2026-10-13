"""Source-preserving Gate metadata for the existing FastEq Triton kernels.

The execution kernels are reused from FastEq d37eaca. e3nn, Prophet and EQX O(2)
modules only supply their interface descriptions, activation constants, and
permutations; no EQX executor, compiler, launcher or autograd backend is called.
"""

import inspect
import math

import torch

from ..triton.fused_equivariant_gate import FastGateFunction, act_code


def _activation(act):
    """Return the raw opcode and exact normalization of the existing source."""
    if act is None:
        return 0, 1.0
    name = type(act).__name__
    if name == "normalize2mom":
        if hasattr(act, "_is_id"):
            code, scale = _activation(act.f)
            return code, scale * (1.0 if act._is_id else float(act.cst))
        f = act.f
        if inspect.isfunction(f):
            closure = inspect.getclosurevars(f).nonlocals
            if set(closure) == {"c", "f"}:
                code, scale = _activation(closure["f"])
                return code, scale / float(closure["c"])
        return _activation(f)
    if name == "ScaledActivation":
        code, scale = _activation(act.activation)
        return code, scale * float(act.scale_factor)
    return act_code(act), 1.0


def _activated(indices, module):
    """Describe activated scalar channels in the source module's own order."""
    result = []
    slices = getattr(module, "_slices", None)
    if slices is None:
        slices = module.irreps_in.slices()
    for sl, act in zip(slices, module.acts):
        code, scale = _activation(act)
        result.extend((indices[i], code, scale) for i in range(sl.start, sl.stop))
    return result


class GatePlan:
    """e3nn/O(2)/Prophet Gate using FastEq's fused first/second-order kernels."""

    backend = "FastEq Triton Gate (reused d37eaca, source constants/layout)"

    def __init__(self, module, kind):
        if kind not in ("o3", "o2", "prophet"):
            raise ValueError("Gate kind must be o3, o2, or prophet")
        self.dim = module.irreps_out.dim if kind == "prophet" else module.irreps_in.dim
        self._cache = {}
        if kind == "o2":
            slices = module.irreps_in.slices()
            extracted = [
                [
                    i
                    for entry in group
                    for i in range(slices[entry].start, slices[entry].stop)
                ]
                for group in (
                    module._scalar_locations,
                    module._gate_locations,
                    module._gated_locations,
                )
            ]
            scalars = _activated(extracted[0], module.act_scalars)
            gates = _activated(extracted[1], module.act_gates)
            sources = [extracted[2][i] for i in module._gated_indices.tolist()]
            gated = [
                (source, gates[g][0], gates[g][1], sign * gates[g][2])
                for source, g, sign in zip(
                    sources,
                    module._expanded_indices.tolist(),
                    module._gated_signs.tolist(),
                )
            ]
        else:
            if kind == "prophet":
                extracted = [
                    list(range(sl.start, sl.stop))
                    for sl in (
                        module.slice_scalars_extra,
                        module.slice_scalars_gates,
                        module.slice_vectors,
                    )
                ]
                scalar_act, gate_act = module.scalars_extra, module.scalars_gates
                tp, gate_first = module.scalars_gates_vectors_elemwise, True
            else:
                slices = module.sc.cut.irreps_in.slices()
                extracted = [
                    [
                        i
                        for entry in group
                        for i in range(slices[entry].start, slices[entry].stop)
                    ]
                    for group in module.sc.cut.instructions
                ]
                scalar_act, gate_act = module.act_scalars, module.act_gates
                tp, gate_first = module.mul, False
            scalars = _activated(extracted[0], scalar_act)
            gates = _activated(extracted[1], gate_act)
            gated = []
            for instruction in tp.instructions:
                vi, gi = (
                    (instruction.i_in2, instruction.i_in1)
                    if gate_first
                    else (instruction.i_in1, instruction.i_in2)
                )
                vr, gr = (
                    (tp.irreps_in2, tp.irreps_in1)
                    if gate_first
                    else (tp.irreps_in1, tp.irreps_in2)
                )
                mul, ir = vr[vi]
                vs, gs = vr.slices()[vi], gr.slices()[gi]
                path_scale = instruction.path_weight / math.sqrt(ir.dim)
                for u in range(mul):
                    gate, code, scale = gates[gs.start + u]
                    for component in range(ir.dim):
                        source = extracted[2][vs.start + u * ir.dim + component]
                        gated.append((source, gate, code, scale * path_scale))
        rows = [(source, -1, code, scale) for source, code, scale in scalars] + gated
        self.outdim = len(rows)
        if self.outdim != module.irreps_out.dim and kind != "prophet":
            raise ValueError("Gate output metadata does not match source irreps")
        self._install_rows(rows)

    @classmethod
    def from_rows(cls, dim, rows):
        """Reuse FastEq Gate kernels for exact user-supplied scalar maps."""
        plan = cls.__new__(cls)
        plan.dim, plan.outdim = dim, len(rows)
        plan._cache = {}
        plan._install_rows(rows)
        return plan

    def _install_rows(self, rows):
        contributions = [[] for _ in range(self.dim)]
        for output, (source, gate, _code, _scale) in enumerate(rows):
            if gate < 0:
                contributions[source].append((output, 0))
            else:
                contributions[source].append((output, 1))
                contributions[gate].append((output, 2))
        max_fan = max(1, max(map(len, contributions), default=0))
        in_out = torch.full((self.dim, max_fan), -1, dtype=torch.int32)
        in_role = torch.full((self.dim, max_fan), -1, dtype=torch.int8)
        for i, values in enumerate(contributions):
            for j, (output, role) in enumerate(values):
                in_out[i, j], in_role[i, j] = output, role
        self.metadata = (
            torch.tensor([row[0] for row in rows], dtype=torch.int32),
            torch.tensor([row[1] for row in rows], dtype=torch.int32),
            torch.tensor([row[2] for row in rows], dtype=torch.int8),
            torch.tensor([row[3] for row in rows], dtype=torch.float64),
            in_out,
            in_role,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __call__(self, x):
        if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
            raise TypeError("FastEq Gate requires GPU float32/float64")
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"Gate input must have trailing dimension {self.dim}")
        key = x.device, x.dtype
        if key not in self._cache:
            with torch.inference_mode(False), torch.no_grad():
                self._cache[key] = tuple(
                    value.to(device=x.device, dtype=x.dtype if i == 3 else value.dtype)
                    for i, value in enumerate(self.metadata)
                )
        return FastGateFunction.apply(x.contiguous(), *self._cache[key])


__all__ = ["GatePlan"]
