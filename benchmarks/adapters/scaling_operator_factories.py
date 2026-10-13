"""Original-interface/FastEq-v6 pairs for independent scaling experiments.

No timing or OOM probing happens in this module. ``atoms`` is a synthetic graph
size; edge operators use exactly 12 * atoms independent rows, without neighbor
construction. This is an operator workload mapping, not a model prediction.
Candidate and original own equal initial state; inputs can be reused with
``autograd.grad`` because the functions do not accumulate leaf gradients.
"""

import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from types import SimpleNamespace

os.environ.setdefault("FASTEQ_EXPERIMENTAL_LINEAR_SIMT", "1")
os.environ.setdefault("FASTEQ_EXPERIMENTAL_NATIVE_MODULATION", "1")

import torch
from e3nn import o3
from e3nn.nn import Gate as O3Gate

from fasteq.adapters import adapt_model, adaptation_report


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    family: str
    rows_factor: int
    baseline_kind: str
    baseline_source: str
    description: str


SPECS = (
    OperatorSpec("linear", "Linear", 1, "original_module", "e3nn.o3.Linear",
                 "O(3) shared-weight channel mixing, C64 at degrees 0, 1, 2"),
    OperatorSpec("gate_o3", "Gate", 1, "original_module", "e3nn.nn.Gate",
                 "O(3) scalar SiLU and sigmoid vector gates"),
    OperatorSpec("gate_prophet", "Gate", 1, "original_module", "prophet.model.Gate",
                 "Prophet SiLU scalar/gate activations, input 3C scalar + C vector/degree"),
    OperatorSpec("gate_o2", "Gate", 12, "original_module", "eqx.o2.Gate",
                 "Actual TECE nonlinearity: preactivated 192 scalars, 192 gates, 192 order-one channels"),
    OperatorSpec("sh", "SphericalHarmonics", 12, "original_module", "e3nn.o3.SphericalHarmonics",
                 "Normalized component spherical harmonics for degrees 0, 1, 2"),
    OperatorSpec("tace_norm", "EquivariantNorm", 1, "original_module",
                 "tace.models._e3nn.layer_norm.EquivariantMergeLayerNorm",
                 "TACE degree-wise merge normalization"),
    OperatorSpec("prophet_norm", "EquivariantNorm", 1, "original_module",
                 "prophet.layer_norm.RMSLayerNorm", "Prophet balanced noncentering RMS"),
    OperatorSpec("tp_uvu", "Uniform1D", 12, "original_module", "e3nn.o3.TensorProduct",
                 "Ten admissible O(3) uvu paths with external per-edge weights"),
    OperatorSpec("tp_uuu", "Uniform1D", 1, "original_module", "e3nn.o3.TensorProduct",
                 "Ten admissible O(3) uuu paths with external per-node weights"),
    OperatorSpec("tp_o2_uuu", "Uniform1D", 12, "original_module", "eqx.o2.TensorProduct",
                 "Actual TECE ECE: 192x0ee+192x1me inputs/output, external per-edge coefficients"),
    OperatorSpec("tp_o2_u1u", "Uniform1D", 12, "original_module", "eqx.o2.TensorProduct",
                 "O(2) order-one x one invariant channel, shared trainable weights"),
    OperatorSpec("tp_uvw", "SegmentedProduct", 12, "original_module", "e3nn.o3.TensorProduct",
                 "Generic 64 x 64 to 64 scalar uvw, 64^3 shared trainable weights"),
    OperatorSpec("tp_nonuniform", "SegmentedProduct", 1, "original_module", "e3nn.o3.TensorProduct",
                 "Nonuniform 64 scalar / 32 vector channels; scalar uuu plus vector-vector scalar uvw"),
    OperatorSpec("wigner", "WignerD", 12, "original_module", "eqx.o2.WignerD.forward_packed",
                 "Original auto dispatch uses EQX CUDA on CUDA; full degree matrices, lmax2"),
    OperatorSpec("rotate", "FrameRotation", 12, "original_model_expression",
                 "tace.models._e3nn.tece_oam_rra.Convolution.forward: torch.bmm(wigner,message)",
                 "Actual unfused TECE dense global-to-local rotation, 9 to 7 angular entries and paired 2C channels"),
    OperatorSpec("rotary", "RotaryScore", 12, "original_model_method",
                 "tace.models._e3nn.tece_oam_rra.Convolution.rotary_score",
                 "Original fused=False Torch path vs FastEq rotary; same radial projection, bias and temperature"),
    OperatorSpec("local_split", "LocalSplit", 12, "original_model_expression",
                 "tace.models._e3nn.tece_oam_rra.Convolution.forward",
                 "Original fused=False narrow tuple vs FastEq shared-adjoint split"),
    OperatorSpec("local_product", "LocalProduct", 12, "original_module",
                 "tace.models._e3nn.tece_oam_rra.Convolution.radial_product (eqx.o2.TensorProduct)",
                 "Actual unfused TECE radial multiplication, paired 2C channels and five coefficient groups"),
    OperatorSpec("invariant", "Invariant", 1, "extracted_original_expression",
                 "prophet.model.EQFFN.forward: per-irrep mean(x**2)",
                 "Original EQFFN invariant extraction only; no standalone original module exists"),
    OperatorSpec("modulated_residual", "ModulatedResidual", 1, "extracted_original_expression",
                 "prophet.model.EQFFN.forward: per-irrep modulation + residual",
                 "Original EQFFN loop/cat expression only, excluding unchanged scalar MLP"),
    OperatorSpec("residual_gate_norm", "ResidualGateRMSNorm", 1, "original_module_composition",
                 "prophet.model.Gate + prophet.layer_norm.RMSLayerNorm",
                 "Original residual addition, Gate, RMS vs FastEq fused Prophet postprocessing"),
)
SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


def catalog():
    """Serializable workload catalog; independent of CUDA availability."""
    return [asdict(spec) for spec in SPECS]


def _irreps(channels=64):
    return o3.Irreps([(channels, (degree, (-1) ** degree)) for degree in range(3)])


def _tece_template():
    """CPU-only source module matching C64/lmax2/mmax1/asymmetric TECE."""
    from tace.models._e3nn.tece_oam_rra import Convolution
    return Convolution(
        mmax=1, lmax=2, num_channel=64, num_head=4, edge_ace_hidden=64,
        num_radial_basis=8, gate_m0=False, use_asymmetric_contraction=True,
        use_radial_rotary_attention=True,
        reshape_in=SimpleNamespace(irreps=_irreps()),
        reshape_out=SimpleNamespace(irreps=_irreps()),
        scalar_act=torch.nn.SiLU(), tensor_act=torch.nn.SiLU())


def _state_equal(original, candidate):
    left, right = original.state_dict(), candidate.state_dict()
    if left.keys() != right.keys():
        raise AssertionError("Adaptation changed parameter/buffer names")
    for name in left:
        if not torch.equal(left[name], right[name]):
            raise AssertionError(f"Initial state differs at {name}")


def _rand(shape, device, dtype):
    # Do not transiently duplicate large inputs while finding capacity limits.
    return torch.randn(shape, device=device, dtype=dtype).mul_(0.2).requires_grad_()


def _mark(metadata, stage):
    callback = metadata.get("_stage_callback")
    if callback is not None:
        callback(stage)


def _module_pair(module, shapes, device, dtype, metadata):
    _mark(metadata, "module_setup")
    selection = metadata.get("_backend")
    if selection is not None:
        # Build source/adapter metadata on CPU so capacity probing never
        # transiently owns the other backend's CUDA parameters or buffers.
        original = module.to(device="cpu", dtype=dtype)
        if selection == "baseline":
            _mark(metadata, "module_cuda_allocation")
            original.to(device=device)
            return _pair(original, None, shapes, device, dtype, metadata,
                         tuple(original.parameters()), ())
        candidate = adapt_model(original)
        _state_equal(original, candidate)
        metadata["adaptation_report"] = adaptation_report(candidate)
        _mark(metadata, "module_cuda_allocation")
        candidate.to(device=device)
        return _pair(None, candidate, shapes, device, dtype, metadata,
                     (), tuple(candidate.parameters()))
    _mark(metadata, "module_cuda_allocation")
    original = module.to(device=device, dtype=dtype)
    candidate = adapt_model(original)
    _state_equal(original, candidate)
    metadata["adaptation_report"] = adaptation_report(candidate)
    return _pair(original, candidate, shapes, device, dtype, metadata,
                 tuple(original.parameters()), tuple(candidate.parameters()))


def _pair(original, candidate, shapes, device, dtype, metadata,
          original_parameters=(), candidate_parameters=()):
    selection = metadata.pop("_backend", None)
    if selection == "baseline":
        candidate, candidate_parameters = None, ()
    elif selection == "candidate":
        original, original_parameters = None, ()
    metadata["input_shapes"] = [list(shape) for shape in shapes]
    metadata["parameter_shapes"] = [list(value.shape) for value in (
        original_parameters if original_parameters else candidate_parameters)]
    metadata["parameter_initial_state_equal"] = True
    metadata["single_backend_allocation"] = selection is not None
    _mark(metadata, "input_allocation")
    inputs = tuple(_rand(shape, device, dtype) for shape in shapes)
    _mark(metadata, "input_allocation_complete")
    metadata.pop("_stage_callback", None)
    return {
        "baseline": original,
        "candidate": candidate,
        "inputs": inputs,
        "baseline_parameters": original_parameters,
        "candidate_parameters": candidate_parameters,
        "metadata": metadata,
    }


class _RotaryScore(torch.nn.Module):
    """Call the actual TECE method with its unchanged radial and temperature maps."""

    def __init__(self):
        super().__init__()
        from eqx.o2 import Linear
        self.lmax, self.mmax, self.num_head, self.num_channel_per_head = 2, 1, 4, 16
        self.radial_proj = Linear("8x0ee", "8x0ee", biases=True)
        self.attention_scale = (16 * 7) ** -0.5
        self.temperature_min, self.temperature_max = 0.25, 4.0
        self.temperature_logit = torch.nn.Parameter(torch.full((4,), math.log(0.75 / 3)))

    def forward(self, query, key, radial_basis):
        from tace.models._e3nn.tece_oam_rra import Convolution
        return Convolution.rotary_score(self, query, key, radial_basis, fused=False)


def build_case(name, atoms, dtype, device="cuda", *, backend=None, variant=None,
               stage_callback=None):
    """Build one pair; allocations occur only here, never during catalog import.

    The input tuple contains differentiable floating tensors. Parameter tuples
    list the corresponding trainable leaves in identical state_dict order.
    Multiple tensor outputs (LocalSplit) are intentional. Run each operation
    on fresh graphs and use ``autograd.grad``, not accumulating ``backward``.
    """
    if name not in SPEC_BY_NAME:
        raise ValueError(f"Unknown operator {name!r}")
    if variant not in (None, "original", "v6"):
        raise ValueError("variant must be original, v6 or None")
    if variant is not None:
        mapped = {"original": "baseline", "v6": "candidate"}[variant]
        if backend is not None and backend != mapped:
            raise ValueError("backend and variant disagree")
        backend = mapped
    if backend not in (None, "baseline", "candidate"):
        raise ValueError("backend must be baseline, candidate or None")
    if atoms < 1:
        raise ValueError("atoms must be positive")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("Only FP32 and FP64 are in this benchmark")
    flags = {key: os.environ.get(key) for key in (
        "FASTEQ_EXPERIMENTAL_LINEAR_SIMT", "FASTEQ_EXPERIMENTAL_NATIVE_MODULATION")}
    if any(value != "1" for value in flags.values()):
        raise RuntimeError("Both v6 experimental flags must be 1 before constructing candidates")
    spec, channels = SPEC_BY_NAME[name], 64
    rows = atoms * spec.rows_factor
    ir = _irreps(channels)
    metadata = {**asdict(spec), "atoms": atoms, "rows": rows, "channels": channels,
                "lmax": 2, "mmax": 1, "heads": 4, "edge_factor": 12,
                "dtype": str(dtype).removeprefix("torch."), "candidate_flags": flags,
                "workload_mapping": "independent rows=N or 12N; no neighbor graph or model energy"}
    if backend is not None:
        metadata["_backend"] = backend
    if stage_callback is not None:
        metadata["_stage_callback"] = stage_callback
    _mark(metadata, "factory_setup")
    if name == "linear":
        return _module_pair(o3.Linear(ir, ir), [(rows, ir.dim)], device, dtype, metadata)
    if name == "gate_o3":
        module = O3Gate("64x0e", [torch.nn.SiLU()], "128x0e", [torch.nn.Sigmoid()],
                        "64x1o+64x2e")
        return _module_pair(module, [(rows, module.irreps_in.dim)], device, dtype, metadata)
    if name == "gate_prophet":
        from prophet.model import Gate
        module = Gate("192x0e+64x1o+64x2e")
        return _module_pair(module, [(rows, module.irreps_out.dim)], device, dtype, metadata)
    if name == "gate_o2":
        module = _tece_template().nonlinearity
        metadata["activation_scope"] = "Gate inputs already activated by actual TECE scalar_act/tensor_act; these outer activations are excluded"
        return _module_pair(module, [(rows, module.irreps_in.dim)], device, dtype, metadata)
    if name == "sh":
        module = o3.SphericalHarmonics([0, 1, 2], normalize=True, normalization="component")
        return _module_pair(module, [(rows, 3)], device, dtype, metadata)
    if name == "tace_norm":
        from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm
        return _module_pair(EquivariantMergeLayerNorm([0, 1, 2], channels),
                            [(rows, 9, channels)], device, dtype, metadata)
    if name == "prophet_norm":
        from prophet.layer_norm import RMSLayerNorm
        return _module_pair(RMSLayerNorm(ir, centering=False), [(rows, ir.dim)], device, dtype, metadata)
    if name in ("tp_uvu", "tp_uuu"):
        mode = name.removeprefix("tp_")
        ir2 = _irreps(1 if mode == "uvu" else channels)
        paths = [(i, j, k, mode, True) for i, (_, a) in enumerate(ir)
                 for j, (_, b) in enumerate(ir2) for k, (_, c) in enumerate(ir) if c in a * b][:10]
        module = o3.TensorProduct(ir, ir2, ir, paths, internal_weights=False, shared_weights=False)
        return _module_pair(module, [(rows, ir.dim), (rows, ir2.dim), (rows, module.weight_numel)],
                            device, dtype, metadata)
    if name in ("tp_o2_uuu", "tp_o2_u1u"):
        from eqx.o2 import TensorProduct
        if name == "tp_o2_uuu":
            module = _tece_template().ece
        else:
            module = TensorProduct("64x1me", "1x0ee", "64x1me", [(0, 0, 0, "u1u", True)])
        shapes = [(rows, module.irreps_in1.dim), (rows, module.irreps_in2.dim)]
        if name == "tp_o2_uuu":
            shapes.append((rows, module.weight_numel))
            metadata["external_per_edge_weights"] = True
        return _module_pair(module, shapes, device, dtype, metadata)
    if name == "tp_uvw":
        module = o3.TensorProduct("64x0e", "64x0e", "64x0e", [(0, 0, 0, "uvw", True)])
        return _module_pair(module, [(rows, 64), (rows, 64)], device, dtype, metadata)
    if name == "tp_nonuniform":
        module = o3.TensorProduct("64x0e+32x1o", "64x0e+32x1o", "64x0e+32x1o",
                                   [(0, 0, 0, "uuu", True), (1, 1, 0, "uvw", True)])
        # Keep the unequal multiplicity supplement representative of actual
        # angular couplings, with one generic vector-vector -> scalar path.
        metadata["description"] = "Nonuniform 64 scalar / 32 vector channels; weighted scalar uuu and vector-vector scalar uvw"
        return _module_pair(module, [(rows, 160), (rows, 160)], device, dtype, metadata)
    if name == "wigner":
        from eqx.o2 import WignerD
        original = WignerD(mmax=1, lmax=2, method="auto").to(dtype=dtype)
        candidate = None if backend == "baseline" else adapt_model(original)
        if candidate is not None:
            _state_equal(original, candidate)
            metadata["adaptation_report"] = adaptation_report(candidate)
            candidate.to(device=device)
        if backend != "candidate":
            original.to(device=device)
        return _pair(original.forward_packed, None if candidate is None else candidate.forward_packed, [(rows, 3)],
                     device, dtype, metadata)
    if name == "rotate":
        from fasteq.ops.rotation import rotate as candidate
        angular = repr(((0, 1, 1, 1, 2, 2, 2, 2, 2), (0, 1, 2, 1, 2, 1, 2)))
        metadata["angular_metadata"] = angular
        metadata["differentiable_matrix_domain"] = "matching-degree entries, as actual Wigner materialization"
        metadata["mask_included_in_both"] = True
        metadata["mask_scope"] = "A shared constant degree mask is evaluated in both callables to project matrix-leaf gradients onto the actual Wigner subspace; includes its cost in both timings"
        _mark(metadata, "constant_cuda_allocation")
        mask = torch.tensor([[a == b for b in (0, 1, 1, 1, 2, 2, 2, 2, 2)]
                             for a in (0, 1, 2, 1, 2, 1, 2)], device=device, dtype=dtype)
        return _pair(lambda x, m: torch.bmm(m * mask, x),
                     lambda x, m: candidate(x, m * mask, angular),
                     [(rows, 9, 128), (rows, 7, 9)], device, dtype, metadata)
    if name == "rotary":
        original = _RotaryScore().to(dtype=dtype)
        candidate = deepcopy(original)
        candidate._fasteq_equivariant = True
        _state_equal(original, candidate)
        if backend != "candidate":
            original.to(device=device)
        if backend != "baseline":
            candidate.to(device=device)
        return _pair(original, candidate, [(rows, 7, 64), (rows, 7, 64), (rows, 8)], device, dtype,
                     metadata, tuple(original.parameters()), tuple(candidate.parameters()))
    if name == "local_split":
        from fasteq.ops.local_product import LocalSplit
        template = _tece_template()
        ranges = template.projection_slices
        split = template._eqx_split_metadata
        dim = template.linear_up.irreps_out.dim
        metadata["ranges"], metadata["kernel_metadata"] = ranges, split
        return _pair(lambda x: tuple(x.narrow(1, start, width) for start, width in ranges),
                     lambda x: LocalSplit.apply(x, ranges, split), [(rows, dim)], device, dtype, metadata)
    if name == "local_product":
        from fasteq.ops.local_product import local_product as candidate
        template = _tece_template()
        original = template.radial_product.to(dtype=dtype)
        if backend != "candidate":
            _mark(metadata, "module_cuda_allocation")
            original.to(device=device)
        packed = template._eqx_weight_metadata
        metadata["unfused_original"] = True
        metadata["candidate_scope"] = "FastEq local_product math corresponding to TECE radial_product; adapter's actual unfused radial_product may use Uniform1D, measured separately from this supplement"
        metadata["kernel_metadata"] = packed
        return _pair(original, lambda x, y: candidate(packed, x, y),
                     [(rows, original.irreps_in1.dim), (rows, original.irreps_in2.dim)], device, dtype, metadata)
    if name == "invariant":
        from fasteq.triton.irreps_invariants import InvariantPlan
        plan = InvariantPlan(ir)

        def original(x):
            return torch.cat([x[:, sl].reshape(rows, mul, angular.dim).square().mean(-1)
                              for (mul, angular), sl in zip(ir, ir.slices())], -1)

        return _pair(original, plan, [(rows, ir.dim)], device, dtype, metadata)
    if name == "modulated_residual":
        from fasteq.ops.modulation import ModulatedResidualPlan
        plan = ModulatedResidualPlan(ir)

        def original(x, gates, residual):
            output, offset = [], 0
            for (mul, angular), sl in zip(ir, ir.slices()):
                weights = gates[:, offset:offset + mul].unsqueeze(-1)
                output.append((x[:, sl].reshape(rows, mul, angular.dim) * weights).reshape(rows, -1))
                offset += mul
            return residual + torch.cat(output, -1)

        return _pair(original, plan, [(rows, ir.dim), (rows, plan.channels), (rows, ir.dim)],
                     device, dtype, metadata)
    if name == "residual_gate_norm":
        from prophet.layer_norm import RMSLayerNorm
        from prophet.model import Gate

        from fasteq.adapters.fusion import ProphetPostprocess
        input_ir = o3.Irreps("192x0e+64x1o+64x2e")
        original_gate = Gate(input_ir).to(dtype=dtype)
        original_norm = RMSLayerNorm(ir, centering=False).to(dtype=dtype)
        gate, norm = deepcopy(original_gate), deepcopy(original_norm)
        _state_equal(original_gate, gate)
        _state_equal(original_norm, norm)
        if backend != "candidate":
            original_gate.to(device=device)
            original_norm.to(device=device)
        if backend != "baseline":
            gate.to(device=device)
            norm.to(device=device)
        convolution = SimpleNamespace(gate=gate, layer_norm=None,
                                      linear_2=SimpleNamespace(irreps_out=input_ir))
        candidate = ProphetPostprocess(convolution, norm)
        if candidate.fused is None:
            raise AssertionError("Expected actual residual/Gate/RMS fusion")
        return _pair(lambda x, skip: original_norm(original_gate(x + skip)), candidate,
                     [(rows, input_ir.dim)] * 2, device, dtype, metadata,
                     tuple(original_norm.parameters()), tuple(norm.parameters()))
    raise AssertionError(f"Missing factory for {name}")


def build_backend(name, atoms, dtype, device="cuda", *, backend):
    """One-backend capacity worker API; no unselected CUDA model allocation."""
    case = build_case(name, atoms, dtype, device, backend=backend)
    return {"function": case[backend], "inputs": case["inputs"],
            "parameters": case[f"{backend}_parameters"], "metadata": case["metadata"]}


__all__ = ["SPECS", "SPEC_BY_NAME", "OperatorSpec", "build_backend", "build_case", "catalog"]
