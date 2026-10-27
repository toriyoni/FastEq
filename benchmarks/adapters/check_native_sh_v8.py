"""Full-output native SH adjoints against original e3nn and historical FastEq."""
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
sys.path.insert(0,str(ROOT/'performance/fasteq_optimization/v7/driver'))
from pair_common import ExclusiveGuard
from fasteq.ops.spherical_harmonics import SphericalHarmonics
from fasteq.triton.spherical_harmonics import SphericalHarmonics as Old
from e3nn.o3 import SphericalHarmonics as Reference


def evaluate(Module,degrees,normalize,normalization,source,seed):
    x=source.detach().requires_grad_();go=seed.detach().requires_grad_()
    y=Module(list(degrees),normalize,normalization)(x)
    if y.requires_grad:
        first=torch.autograd.grad(y,x,go,create_graph=True)[0]
        if first.requires_grad:
            second=torch.autograd.grad(first.square().mean(),(x,go),allow_unused=True)
        else:second=(torch.zeros_like(x),torch.zeros_like(go))
    else:
        # e3nn's l=0 is a constant with no autograd edge. Validate the native
        # exact-zero adjoints instead of asking autograd for a missing edge.
        first=torch.zeros_like(x);second=(torch.zeros_like(x),torch.zeros_like(go))
    return y,first,*second


def main():
    guard=ExclusiveGuard('GPU-af44a7bc-24c0-45be-df6e-03f4c445c78d');guard.check()
    torch.manual_seed(20261014)
    records=[]
    output=ROOT/'performance/fasteq_optimization/v8/sh/GPU_QA.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    for dtype in (torch.float32,torch.float64):
        for degrees in ((0,1,2,3),(3,1,0,3),(0,),(4,)):
            for normalize in (False,True):
                for normalization in ('component','integral','norm'):
                    source=torch.randn(3,7,device='cuda',dtype=dtype).t()
                    seed=torch.randn(7,sum(2*l+1 for l in degrees),device='cuda',dtype=dtype)
                    all_results=[evaluate(M,degrees,normalize,normalization,source,seed)
                                 for M in (Reference,Old,SphericalHarmonics)]
                    tolerance=(dict(atol=3e-5,rtol=5e-5) if dtype==torch.float32
                               else dict(atol=2e-11,rtol=2e-10))
                    for reference in all_results[:2]:
                        for actual,expected in zip(all_results[2],reference):
                            if actual is None or expected is None:
                                value=actual if expected is None else expected
                                assert value is None or not value.count_nonzero()
                            else:torch.testing.assert_close(actual,expected,**tolerance)
                    guard.check()
                    records.append({'dtype':str(dtype),'degrees':degrees,'normalize':normalize,
                                    'normalization':normalization,'full_forward_first_second_passed':True})
                    output.write_text(json.dumps({'status':'running','cases':records},indent=2)+'\n')
    output.write_text(json.dumps({'status':'passed','cases':records,'guard':guard.check(),
        'scope':'48 native CUDA SH full output/first/mixed-second conditions; no model/performance claim',
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/FastEq_v8/fasteq/ops/spherical_harmonics.py',
            ROOT/'src/FastEq_v8/fasteq/ops/scalar_expression.py')}},indent=2)+'\n')
    print('48 native SH forward/first/mixed-second checks passed',flush=True)


if __name__=='__main__':main()
