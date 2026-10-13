"""Exact e3nn/O(2) path translation to existing native CUDA products."""

import math
import os

import torch
from e3nn import o3

from .cache import TensorCache, indices


class TensorProductPlan:
    @staticmethod
    def _cg_state(module):
        return tuple(
            (name, id(value), value._version, value.dtype, value.device)
            for name, value in module.named_buffers()
            if "_w3j_" in name
        )

    def __init__(self, module, o2=False):
        self.o2 = o2
        self.cg_state = self._cg_state(module) if not o2 else ()
        irreps = (module.irreps_in1, module.irreps_in2, module.irreps_out)
        self.entries = tuple(tuple((a.mul, a.ir.dim) for a in ir) for ir in irreps)
        self.sizes = tuple(ir.dim for ir in irreps)
        modes = {i.connection_mode for i in module.instructions}
        if not modes <= {"uvu", "uuu", "uvw", "u1u"}:
            raise NotImplementedError(f"Tensor product connection modes {modes}")
        common = len({m for m, _ in self.entries[0] + self.entries[2]}) == 1
        c = self.entries[0][0][0] if common else 1
        y_muls = {m for m, _ in self.entries[1]}
        scalar_y = modes <= {"uvu", "u1u"} and y_muls == {1}
        uniform = common and (scalar_y or (modes <= {"uuu"} and y_muls == {c}))
        self.width = c if uniform else 1
        self.ywidth = (1 if scalar_y else c) if uniform else 1
        self.backend = (
            "fasteq.uniform1d CUDA" if uniform else "fasteq.segmented_product CUDA"
        )
        self.weight_numel = module.weight_numel
        self.dummy = (
            0
            if not uniform
            or (module.weight_numel and all(i.has_weight for i in module.instructions))
            else max(1, self.width)
        )
        self.wsize = module.weight_numel + self.dummy
        self.cache = {}
        self.constants = TensorCache()
        self.permutations = {}
        self.vector = not uniform
        offsets = [list(ir.slices()) for ir in irreps]
        rows, woff = [], 0
        terms, coefficients = [], []
        for ins in module.instructions:
            m1, d1 = self.entries[0][ins.i_in1]
            m2, d2 = self.entries[1][ins.i_in2]
            mo, do = self.entries[2][ins.i_out]
            if o2:
                from eqx.o2._clebsch_gordan import clebsch_gordan_product

                a = torch.eye(d1, dtype=torch.float64).reshape(d1, 1, d1, 1)
                b = torch.eye(d2, dtype=torch.float64).reshape(1, d2, d2, 1)
                cg = clebsch_gordan_product(
                    a,
                    irreps[0][ins.i_in1].ir,
                    b,
                    irreps[1][ins.i_in2].ir,
                    irreps[2][ins.i_out].ir,
                    elementwise=True,
                )[..., 0]
            else:
                degrees = (
                    irreps[0][ins.i_in1].ir.l,
                    irreps[1][ins.i_in2].ir.l,
                    irreps[2][ins.i_out].ir.l,
                )
                # e3nn can construct tables in FP32 then cast the module to
                # FP64. Preserve those stored coefficients, including their
                # rounding, instead of silently changing the trained model.
                stored = getattr(
                    module._compiled_main_left_right,
                    "_w3j_" + "_".join(map(str, degrees)),
                    None,
                )
                cg = (
                    stored.detach().cpu().double()
                    if stored is not None
                    else o3.wigner_3j(*degrees, dtype=torch.float64)
                )
            for ai, bi, oi in cg.nonzero().tolist():
                coefficient = float(cg[ai, bi, oi]) * ins.path_weight
                if uniform:
                    ix = offsets[0][ins.i_in1].start // c + ai
                    iy = offsets[1][ins.i_in2].start // self.ywidth + bi
                    iz = offsets[2][ins.i_out].start // c + oi
                    iw = woff // c if ins.has_weight else module.weight_numel // c
                    rows.append(((iw, ix, iy, iz), coefficient))
                # Keep an alternate sparse descriptor in the module's native
                # layout. Small scalar-Y products can reuse FastEq's complete
                # shared-adjoint CUDA implementation without layout copies or
                # differentiable expansion of a shared parameter.
                terms.append(
                    (
                        {"uuu": 0, "u1u": 1, "uvu": 2, "uvw": 3}[
                            ins.connection_mode
                        ],
                        woff,
                        offsets[0][ins.i_in1].start + (ai * m1 if o2 else ai),
                        1 if o2 else d1,
                        offsets[1][ins.i_in2].start + (bi * m2 if o2 else bi),
                        1 if o2 else d2,
                        offsets[2][ins.i_out].start + (oi * mo if o2 else oi),
                        1 if o2 else do,
                        m1,
                        m2,
                        mo,
                        int(ins.has_weight),
                    )
                )
                coefficients.append(coefficient)
            if ins.has_weight:
                woff += math.prod(ins.path_shape)
        self.rows = tuple(rows)
        self.dims = (
            self.wsize // self.width,
            self.sizes[0] // self.width,
            self.sizes[1] // self.ywidth,
            self.sizes[2] // self.width,
        )
        self.segmented_metadata = repr(
            ((self.weight_numel, *self.sizes), tuple(terms), tuple(coefficients))
        )
        self.scalar_y_small = (
            uniform and scalar_y and 0 < len(terms) <= 16
            and all(term[0] in (1, 2) and term[9] == 1 for term in terms)
        )
        self.metadata = (
            self.segmented_metadata if self.vector
            else repr((self.width, self.dims, self.rows))
        )

    def _layout(self, x, which, inverse=False):
        if self.o2 or self.width == 1:
            return x
        key = (which, inverse)
        if key not in self.permutations:
            permutation, start = [], 0
            for mul, dim in self.entries[which]:
                if inverse:
                    permutation.extend(
                        start + a * mul + u for u in range(mul) for a in range(dim)
                    )
                else:
                    permutation.extend(
                        start + u * dim + a for a in range(dim) for u in range(mul)
                    )
                start += mul * dim
            self.permutations[key] = (
                None if permutation == list(range(start)) else permutation
            )
        permutation = self.permutations[key]
        if permutation is None:
            return x
        ids = self.constants.get(
            (x.device, key),
            lambda: torch.tensor(permutation, dtype=torch.long, device=x.device),
        )
        return x.index_select(1, ids)

    def __call__(self, module, x, y, weight=None):
        if not self.o2 and self.cg_state != self._cg_state(module):
            # .to(), deepcopy and load_state_dict can replace/cast CG buffers.
            # Rebuild only when their identity or version actually changes.
            module._fasteq_plan = TensorProductPlan(module, self.o2)
            return module._fasteq_plan(module, x, y, weight)
        if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
            raise TypeError("Adapted tensor products require CUDA FP32/FP64")
        weight = module._get_weights(weight)
        if any(v.dtype != x.dtype or v.device != x.device for v in (y, weight)):
            raise ValueError(
                "Tensor product features and weights must share dtype and device"
            )
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        n = math.prod(shape)
        x = x.expand(*shape, self.sizes[0]).reshape(n, self.sizes[0])
        y = y.expand(*shape, self.sizes[1]).reshape(n, self.sizes[1])
        shared = math.prod(weight.shape[:-1]) == 1
        # The O(3) nine-term proposal regressed against existing Uniform1D in
        # both dtypes. Keep its established kernel; enable the measured O(2)
        # scalar-Y specialization only.
        scalar_y_sparse = (
            self.scalar_y_small and self.o2
            and os.environ.get("FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y") == "1"
        )
        if (
            not self.vector and not scalar_y_sparse
            and torch.is_grad_enabled() and weight.requires_grad
        ):
            # Uniform1D's forward accepts WB=1, but its existing grad_w stores
            # are owned by each edge and do not reduce a shared destination.
            # Expand only differentiable weights: autograd's CUDA broadcast
            # transpose performs the shared reduction and remains second-order.
            shared = False
        rows = 1 if shared else n
        weight = (
            weight.reshape(1, self.weight_numel)
            if shared
            else weight.expand(*shape, self.weight_numel).reshape(n, self.weight_numel)
        )
        if scalar_y_sparse:
            from fasteq.ops.segmented_product import segmented_product

            result = segmented_product(self.segmented_metadata, weight, x, y)
            return result.reshape(*shape, self.sizes[2])
        if self.dummy:
            ones = self.constants.get(
                (x.device, x.dtype, rows, self.dummy),
                lambda: x.new_ones(rows, self.dummy),
            )
            w = torch.cat((weight, ones), -1) if self.weight_numel else ones
        else:
            w = weight
        if not self.vector and n:
            from fasteq.ops.uniform1d_jit import fast_uniform1d_jit

            key = (x.device, x.dtype)
            if key not in self.cache:
                rows = sorted(self.rows, key=lambda r: r[0][3])
                with torch.inference_mode(False), torch.no_grad():
                    ids = torch.tensor(
                        [r[0] for r in rows], dtype=torch.int32, device=x.device
                    ).reshape(-1, 4)
                    meta = {
                        name: ids[:, i].contiguous()
                        for i, name in enumerate(
                            ("i_list", "j_list", "k_list", "v_list")
                        )
                    }
                    meta["coeff_list"] = x.new_tensor([r[1] for r in rows])
                meta.update(
                    w_seg_num=self.dims[0],
                    x_seg_num=self.dims[1],
                    y_seg_num=self.dims[2],
                    out_seg_num=self.dims[3],
                    u_dim=self.width,
                    size_list=[self.wsize, *self.sizes],
                )
                self.cache[key] = meta
            ix = indices(x, n, sequence=True)
            result = fast_uniform1d_jit(
                w.contiguous(),
                self._layout(x, 0).contiguous(),
                self._layout(y, 1).contiguous(),
                {1: ix},
                {},
                self.cache[key],
                False,
            )
            result = self._layout(result, 2, True)
        elif not self.vector:
            # Empty edge lists never launch a kernel, but preserve gradient connectivity.
            result = x.new_zeros(n, self.sizes[2]) + (x.sum() + y.sum() + w.sum()) * 0
        else:
            from fasteq.ops.segmented_product import segmented_product

            result = segmented_product(self.metadata, weight, x, y)
        return result.reshape(*shape, self.sizes[2])
