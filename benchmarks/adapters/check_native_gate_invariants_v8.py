"""Affected native Gate/invariant QA, before adapter integration."""
from contextlib import contextmanager
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND = ROOT/'src/FastEq_v8'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def native_only():
    import triton.runtime.jit
    sys.path.insert(0, str(CAND/'tests/adapters'))
    from eqx_cuda_spy import forbid_eqx_cuda
    old = triton.runtime.jit.JITFunction.__getitem__
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Candidate executed a Triton JIT kernel')
    triton.runtime.jit.JITFunction.__getitem__ = forbidden
    try:
        with forbid_eqx_cuda() as calls:
            yield calls
        assert calls == []
    finally:
        triton.runtime.jit.JITFunction.__getitem__ = old


def close(a, b, dtype, level=0):
    import torch
    atol, rtol = (3e-5, 3e-5) if dtype == torch.float32 else (2e-10, 2e-9)
    factor = (1, 10, 30)[min(level, 2)]
    if a is None or b is None:
        value = b if a is None else a
        assert value is None or not torch.count_nonzero(value)
        return {'zero_or_none': True}
    torch.testing.assert_close(a, b, atol=atol*factor, rtol=rtol*factor)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta = a.detach().double()-b.detach().double()
    return {'max_abs': delta.abs().max().item() if delta.numel() else 0.,
            'relative_l2': delta.norm().item()/max(b.detach().double().norm().item(), 1e-30),
            'full_elements': a.numel(), 'atol': atol*factor, 'rtol': rtol*factor}


def derivative_check(candidate, original, data, dtype):
    import torch
    x = data.detach().requires_grad_()
    xr = data.detach().clone().requires_grad_()
    expected = original(xr)
    with native_only():
        actual = candidate(x)
    errors = [close(actual, expected, dtype)]
    seed = torch.randn_like(actual).requires_grad_()
    seedr = seed.detach().clone().requires_grad_()
    with native_only():
        first = torch.autograd.grad((actual*seed).sum()+actual.square().sum()/7, x,
                                    create_graph=True)[0]
    first_r = torch.autograd.grad((expected*seedr).sum()+expected.square().sum()/7, xr,
                                  create_graph=True)[0]
    errors.append(close(first, first_r, dtype, 1))
    direction = torch.randn_like(first)
    with native_only():
        second = torch.autograd.grad(first, (x, seed), direction, allow_unused=True)
    second_r = torch.autograd.grad(first_r, (xr, seedr), direction, allow_unused=True)
    errors += [close(a, b, dtype, 2) for a, b in zip(second, second_r, strict=True)]
    return errors


def gate_tables(plan, dtype):
    import torch
    with torch.inference_mode(False), torch.no_grad():
        return tuple(value.to(device='cuda', dtype=dtype if i == 3 else value.dtype)
                     for i, value in enumerate(plan.metadata))


def gate_original(plan, x):
    import torch
    cols = []
    src, gate, code, scale = plan.metadata[:4]
    for s, g, c, k in zip(src.tolist(), gate.tolist(), code.tolist(), scale.tolist(), strict=True):
        z = x[..., s if g < 0 else g]
        value = z
        if c == 1: value = torch.nn.functional.silu(z)
        elif c == 2: value = torch.sigmoid(z)
        elif c == 3: value = torch.tanh(z)
        value = value*k
        cols.append(value if g < 0 else x[..., s]*value)
    return torch.stack(cols, -1) if cols else x[..., :0]


def gate_cases(guard):
    import torch
    from e3nn.nn import Gate
    from fasteq.ops.gate import GatePlan
    from fasteq.ops.native_gate import FastGateFunction, FastGateBackwardFunction
    records = []
    for dtype in (torch.float32, torch.float64):
        from eqx.o2 import Gate as O2Gate
        from prophet.model import Gate as ProphetGate
        modules = {
            'o3': Gate('2x0e', [torch.nn.SiLU()], '3x0e', [torch.nn.Sigmoid()], '3x1o'),
            'o3_scalar': Gate('2x0e', [torch.nn.Tanh()], '3x0e', [None], '3x0e'),
            'prophet': ProphetGate('8x0e+3x1o'),
            'o2': O2Gate('2x0ee', [torch.nn.SiLU()], '3x0ee', [torch.nn.Sigmoid()], '3x1me'),
            'o2_sign': O2Gate('2x0oe+3x0ee', [torch.nn.Tanh(), torch.nn.SiLU()],
                             '3x0oe+2x0ee', [torch.nn.Tanh(), torch.nn.Sigmoid()], '5x1me'),
        }
        for name, module in modules.items():
            guard.check()
            module = module.cuda().to(dtype)
            kind = 'o3' if name.startswith('o3') else 'o2' if name.startswith('o2') else 'prophet'
            plan = GatePlan(module, kind)
            tables = gate_tables(plan, dtype)
            candidate = lambda x: FastGateFunction.apply(x.contiguous(), *tables)
            original = lambda x: module(x.reshape(-1, plan.dim)).reshape(*x.shape[:-1], plan.outdim)
            data = torch.randn(plan.dim, 2, 3, device='cuda', dtype=dtype).permute(1, 2, 0)
            errors = derivative_check(candidate, original, data, dtype)
            records.append({'case': 'gate_'+name, 'dtype': str(dtype), 'noncontiguous_multiple_leading_axes': True,
                            'full_forward_first_double': True, 'errors': errors, 'status': 'pass'})
        plan = GatePlan.from_rows(11, [(0,-1,0,.3),(1,-1,1,.7),(2,-1,2,-.4),(3,-1,3,1.5),
                                     (4,7,1,.75),(5,7,2,-.25),(6,8,3,.3),(8,8,2,.55),(4,7,1,.75)])
        tables = gate_tables(plan, dtype)
        candidate = lambda x: FastGateFunction.apply(x.contiguous(), *tables)
        original = lambda x: gate_original(plan, x)
        for rows, mode in ((0,'empty'), (1,'single'), (129,'tail'), (33,'saturated')):
            data = torch.randn(rows*11+1, device='cuda', dtype=dtype)[1:].view(rows, 11)
            if mode == 'saturated' and rows:
                data.copy_(torch.linspace(-30, 30, data.numel(), device='cuda', dtype=dtype).reshape_as(data))
            errors = derivative_check(candidate, original, data, dtype)
            records.append({'case': 'gate_metadata_'+mode, 'dtype': str(dtype), 'odd_storage_offset': True,
                            'self_gate': True, 'all_4_activations': True, 'errors': errors, 'status': 'pass'})
        # Structural second-adjoint pruning; direct first-backward interface.
        for requested in ((0,), (1,), (0,1)):
            data = torch.randn(7,11,device='cuda',dtype=dtype)
            x = data.detach().requires_grad_(0 in requested)
            gy = torch.randn(7,plan.outdim,device='cuda',dtype=dtype).requires_grad_(1 in requested)
            xr = data.detach().clone().requires_grad_()
            gyr = gy.detach().clone().requires_grad_(1 in requested)
            dxr = torch.autograd.grad((original(xr)*gyr).sum(), xr, create_graph=True)[0]
            with native_only():
                dx = FastGateBackwardFunction.apply(x, gy, *tables)
                grads = torch.autograd.grad(dx.sin().sum(), tuple((x,gy)[i] for i in requested))
            gradsr = torch.autograd.grad(dxr.sin().sum(), tuple((xr,gyr)[i] for i in requested))
            errors = [close(dx,dxr,dtype,1)] + [close(a,b,dtype,2) for a,b in zip(grads,gradsr,strict=True)]
            records.append({'case': 'gate_selected_second', 'dtype': str(dtype), 'requested': requested,
                            'errors': errors, 'status': 'pass'})
        torch.cuda.synchronize(); guard.check()
    return records


def invariant_original(irreps, x):
    import torch
    outputs = [x[:, sl].reshape(x.shape[0], mul, ir.dim).square().mean(-1)
               for (mul, ir), sl in zip(irreps, irreps.slices(), strict=True)]
    return torch.cat(outputs, -1) if outputs else x[:, :0]


def invariant_cases(guard):
    import torch
    from e3nn import o3
    from fasteq.ops.native_invariants import InvariantPlan, _First
    records = []
    for dtype in (torch.float32, torch.float64):
        for name, spec, rows, layout in (
            ('scalar_only','65x0e',129,'contiguous'),
            ('mixed','2x0e+3x1o+1x2e',129,'transpose'),
            ('tail','3x1o+5x2e',33,'offset'),
            ('wide_angular','1x20e',17,'transpose'),
            ('empty_rows','2x0e+3x1o+1x2e',0,'contiguous'),
            ('empty_irreps','',7,'contiguous'),
        ):
            guard.check()
            irreps = o3.Irreps(spec); plan = InvariantPlan(irreps)
            if layout=='transpose': data=torch.randn(plan.dim,rows,device='cuda',dtype=dtype).T
            elif layout=='offset': data=torch.randn(rows*plan.dim+1,device='cuda',dtype=dtype)[1:].view(rows,plan.dim)
            else: data=torch.randn(rows,plan.dim,device='cuda',dtype=dtype)
            with torch.inference_mode(), native_only(): plan(torch.ones_like(data))
            errors = derivative_check(plan,lambda x:invariant_original(irreps,x),data,dtype)
            records.append({'case': 'invariant_'+name, 'dtype': str(dtype), 'rows': rows,
                            'angular_width': max(plan._data[1],default=0),
                            'definition': 'sumsq / angular_width, not sqrt', 'errors': errors,'status':'pass'})
        irreps=o3.Irreps('2x0e+3x1o+1x2e');plan=InvariantPlan(irreps)
        data=torch.randn(7,plan.dim,device='cuda',dtype=dtype)
        with native_only(): plan(data)
        tables=plan._cache[data.device]
        for requested in ((0,),(1,),(0,1)):
            x=data.detach().requires_grad_(0 in requested)
            g=torch.randn(7,plan.channels,device='cuda',dtype=dtype).requires_grad_(1 in requested)
            xr=data.detach().clone().requires_grad_();gr=g.detach().clone().requires_grad_(1 in requested)
            dxr=torch.autograd.grad((invariant_original(irreps,xr)*gr).sum(),xr,create_graph=True)[0]
            with native_only():
                dx=_First.apply(x,g,plan,tables)
                grads=torch.autograd.grad(dx.sin().sum(),tuple((x,g)[i]for i in requested))
            gradsr=torch.autograd.grad(dxr.sin().sum(),tuple((xr,gr)[i]for i in requested))
            errors=[close(dx,dxr,dtype,1)]+[close(a,b,dtype,2)for a,b in zip(grads,gradsr,strict=True)]
            records.append({'case':'invariant_selected_second','dtype':str(dtype),'requested':requested,
                            'errors':errors,'status':'pass'})
        torch.cuda.synchronize();guard.check()
    return records


def physical_graph_cases(guard):
    import torch
    from e3nn import o3
    from fasteq.ops.gate import GatePlan
    from fasteq.ops.native_gate import FastGateFunction
    from fasteq.ops.native_invariants import InvariantPlan
    records=[]
    for dtype in (torch.float32,torch.float64):
        gate=GatePlan.from_rows(18,[(i,-1,1,.7)for i in range(2)]+
                               [(i,2+(i-6)//3,2,.5)for i in range(6,18)])
        tables=gate_tables(gate,dtype)
        invariant=InvariantPlan(o3.Irreps('2x0e+4x1o'))
        coords=torch.randn(33,3,device='cuda',dtype=dtype).mul_(.2).requires_grad_()
        projection=torch.randn(3,18,device='cuda',dtype=dtype).mul_(.04).requires_grad_()
        readout=torch.randn(invariant.channels,device='cuda',dtype=dtype).mul_(.03).requires_grad_()
        leaves=(coords,projection,readout)
        def run(native, selected=leaves):
            coords, projection, readout = selected
            x=(coords[:,:,None]*projection).sum(1).sin()
            if native:
                gated=FastGateFunction.apply(x,*tables);inv=invariant(gated)
            else:
                gated=gate_original(gate,x);inv=invariant_original(o3.Irreps('2x0e+4x1o'),gated)
            energy=(inv*readout).sin().sum()
            forces=-torch.autograd.grad(energy,coords,create_graph=True)[0]
            grads=torch.autograd.grad((forces-.13).square().mean(),selected)
            return energy,forces,*grads
        expected=run(False)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),native_only():actual=run(True)
        torch.cuda.current_stream().wait_stream(stream)
        errors=[close(a,b,dtype,2)for a,b in zip(actual,expected,strict=True)]
        # Independent static leaves avoid inheriting Autograd stream edges
        # from the earlier default-stream reference/validation execution.
        with torch.cuda.stream(stream),native_only():
            graph_leaves=tuple(value.detach().clone().requires_grad_()for value in leaves)
            for _ in range(3):run(True,graph_leaves)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with native_only():
            with torch.cuda.graph(graph,stream=stream):captured=run(True,graph_leaves)
            with torch.cuda.stream(stream),torch.no_grad():
                graph_leaves[1].add_(.017);graph_leaves[0].mul_(.71);graph_leaves[2].add_(.01)
                graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        replay_errors=[close(a,b,dtype,2)for a,b in zip(captured,run(False,graph_leaves),strict=True)]
        limit=2e-4 if dtype==torch.float32 else 2e-7
        assert max(e['relative_l2']for e in errors[3:]+replay_errors[3:]) <= limit
        torch.cuda.synchronize();guard.check()
        records.append({'case':'native_gate_invariants_physical_stream_live_graph','dtype':str(dtype),
                        'full_outputs':True,'all_parameter_names':['projection','readout'],
                        'coords_gradient':True,'errors':errors,'replay_errors':replay_errors,'status':'pass'})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--groups',nargs='+',choices=['gate','invariant','physical'],default=['gate','invariant','physical'])
    args=parser.parse_args()
    import torch
    from benchmark_scaling_case import ExclusiveGuard
    os.sched_setaffinity(0,{172});torch.set_num_threads(1);torch.manual_seed(78214)
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu_uuid
    torch.backends.cuda.matmul.allow_tf32=False
    names=['fasteq/ops/native_gate.py','fasteq/cuda/src/native_gate.cuh',
           'fasteq/ops/native_invariants.py','fasteq/cuda/src/native_invariants.cuh',
           'fasteq/cuda/src/equi_linear_kernels/device_cuda_helper.cuh',
           'fasteq/ops/gate.py','fasteq/cuda/jit.py','fasteq/cuda/src/jit_runtime.cpp','fasteq/cuda/src/jit_runtime.h']
    hashes={str(CAND/name):digest(CAND/name)for name in names}
    hashes[str(Path(__file__).resolve())]=digest(__file__)
    report={'status':'running','groups':args.groups,'source_sha256':hashes,'records':[],
            'scope':'Only native Gate/Invariant, complete fixture outputs/gradients; no normal latency or model benchmark.'}
    guard=ExclusiveGuard(args.gpu_uuid);report['before_guard']=guard.check()
    from native_uvw_v8_common import write_json
    try:
        groups={'gate':gate_cases,'invariant':invariant_cases,'physical':physical_graph_cases}
        for name in args.groups:
            report['records'].extend(groups[name](guard));write_json(args.output,report)
        from fasteq.cuda.jit import runtime
        report['runtime_binary']={'path':str(runtime().__file__),'sha256':digest(runtime().__file__)}
        assert all(digest(p)==v for p,v in hashes.items()),'Source drift during Gate/invariant QA'
        report.update(status='pass',passed_cases=len(report['records']),after_guard=guard.check())
    except Exception as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:write_json(args.output,report)
    print(json.dumps({'status':report['status'],'passed_cases':report['passed_cases']}),flush=True)


if __name__=='__main__':main()
