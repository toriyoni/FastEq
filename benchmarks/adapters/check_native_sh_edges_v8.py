"""Affected SH clamp, physical force-loss, and live-graph checks."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from e3nn.o3 import SphericalHarmonics as Reference
from fasteq.ops.spherical_harmonics import SphericalHarmonics as Native

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
sys.path.insert(0,str(ROOT/'performance/fasteq_optimization/v7/driver'))
sys.path.insert(0,str(ROOT/'src/FastEq_v8/benchmarks/adapters'))
from pair_common import ExclusiveGuard
from check_native_gate_invariants_v8 import native_only


def check(actual,expected,dtype):
    tolerance=(dict(atol=3e-5,rtol=5e-5) if dtype==torch.float32
               else dict(atol=2e-11,rtol=2e-10))
    errors=[]
    for a,b in zip(actual,expected,strict=True):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a,b,**tolerance)
        delta=(a-b).double()
        errors.append({'elements':a.numel(),'max_abs':delta.abs().max().item() if a.numel() else 0.,
                       'relative_l2':(delta.norm()/b.double().norm().clamp_min(1e-30)).item()})
    return errors


def physical(module,values):
    coords,scale,shift,readout=values
    output=module(coords*scale+shift)
    energy=((output*readout).sin()+.01*output.square()).sum()
    force=-torch.autograd.grad(energy,coords,create_graph=True)[0]
    gradients=torch.autograd.grad((force-.07).square().mean(),values)
    return (output,energy,force,*gradients)


def main():
    guard=ExclusiveGuard('GPU-af44a7bc-24c0-45be-df6e-03f4c445c78d');guard.check()
    torch.manual_seed(93282);records=[]
    native=Native([0,1,2,3],True,'component')
    reference=Reference([0,1,2,3],True,'component')
    for dtype in (torch.float32,torch.float64):
        # Use the mathematical clamped radius explicitly: PyTorch norm's
        # Hessian at exactly zero is undefined even in its inactive branch.
        x=torch.tensor([[0.,0.,0.],[.2e-12,-.1e-12,.3e-12],
                        [.6e-12,.72e-12,.4e-12],[.001,-.002,.003]],device='cuda',dtype=dtype,requires_grad=True)
        seed=torch.randn(4,16,device='cuda',dtype=dtype,requires_grad=True)
        def derive(fn):
            y=fn(x)
            first=torch.autograd.grad(y,x,seed,create_graph=True)[0]
            second=torch.autograd.grad(first,(x,seed),torch.ones_like(first))
            return y,first,*second
        def piecewise(x):
            radius=x.square().sum(-1,keepdim=True)
            radius=torch.where(radius<1e-24,torch.full_like(radius,1e-24),radius).sqrt()
            return Reference([0,1,2,3],False,'component')(x/radius)
        expected=derive(piecewise)
        with native_only():actual=derive(native)
        records.append({'case':'clamped_zero_and_near_boundary','dtype':str(dtype),
                        'errors':check(actual,expected,dtype)})
        base=[torch.randn(7,3,device='cuda',dtype=dtype)*.2,
              torch.ones(3,device='cuda',dtype=dtype)*.7,
              torch.randn(3,device='cuda',dtype=dtype)*.01,
              torch.randn(16,device='cuda',dtype=dtype)*.03]
        leaves=[v.detach().requires_grad_() for v in base]
        expected=physical(reference,leaves)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),native_only():actual=physical(native,leaves)
        torch.cuda.current_stream().wait_stream(stream)
        records.append({'case':'physical_full_parameters_nondefault_stream','dtype':str(dtype),
                        'parameters':['scale','shift','readout'],'coordinates_gradient':True,
                        'errors':check(actual,expected,dtype)})
        # New leaves receive no autograd work on the default stream before capture.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),native_only():
            static=[v.detach().clone().requires_grad_() for v in base]
            for _ in range(3):physical(native,static)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with native_only(),torch.cuda.graph(graph,stream=stream):result=physical(native,static)
        with torch.no_grad():static[1].mul_(.83);static[3].add_(.01)
        with native_only():graph.replay()
        torch.cuda.synchronize()
        records.append({'case':'live_parameters_physical_CUDA_graph','dtype':str(dtype),
                        'errors':check(result,physical(reference,static),dtype)})
        guard.check()
    files=[Path(__file__),ROOT/'src/FastEq_v8/fasteq/ops/spherical_harmonics.py',
           ROOT/'src/FastEq_v8/fasteq/ops/scalar_expression.py']
    out=ROOT/'performance/fasteq_optimization/v8/sh/edges_QA.json'
    out.write_text(json.dumps({'status':'passed','cases':records,'guard':guard.check(),
                              'known_radial_precision_limitation':str(ROOT/'performance/fasteq_optimization/v8/sh/edge_axis_limitation.json'),
                              'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
    print('6 native SH edge/physical/Graph conditions passed',flush=True)


if __name__=='__main__':main()
