"""Affected SH lightweight dispatcher QA: full mixed AD, masks, and physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Preserve previous whole attempt')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    import torch
    from e3nn.o3 import SphericalHarmonics as Reference
    from benchmark_scaling_case import ExclusiveGuard
    from native_sh_dispatch_spy_v8 import native_only
    from check_native_sh_edges_v8 import physical,check
    from fasteq.ops.spherical_harmonics import SphericalHarmonics as Native,_SHExpression,_program
    from fasteq.ops.scalar_expression import _adjoint,scalar_expression
    os.sched_setaffinity(0,{172});torch.set_num_threads(1);torch.manual_seed(98142)
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu_uuid
    guard=ExclusiveGuard(args.gpu_uuid);before=guard.check()
    root=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
    files=[Path(__file__),root/'src/FastEq_v8/fasteq/ops/spherical_harmonics.py',
           root/'src/FastEq_v8/fasteq/ops/scalar_expression.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
    report={'status':'running','source_sha256':hashes,'before_guard':before,'records':[],
            'scope':'Affected lightweight SH eager dispatch, original CUDA emitter/AD unchanged. Existing 48 ordinary QA reused via AST bridge; these meaningful mixed-third, selected/None and physical cases are additional.'}
    def full(a,b,dtype):
        if a is None or b is None:
            value=b if a is None else a
            assert value is None or not torch.count_nonzero(value)
            return {'zero_or_none':True}
        return check((a,),(b,),dtype)[0]
    try:
        for dtype in (torch.float32,torch.float64):
            for degrees,normalize,normalization,rows,layout in (
                ((0,1,2),True,'component',33,'offset'),
                ((3,1,0,3),False,'norm',17,'transpose'),
                ((4,),True,'integral',2,'leading')):
                width=sum(2*l+1 for l in degrees)
                if layout=='offset':source=torch.randn(rows*3+1,device='cuda',dtype=dtype)[1:].view(rows,3)
                elif layout=='transpose':source=torch.randn(3,rows,device='cuda',dtype=dtype).T
                else:source=torch.randn(2,2,3,device='cuda',dtype=dtype)
                # Avoid an ill-conditioned axis/radius oracle in ordinary AD.
                source.add_(.7)
                seed=torch.randn(*source.shape[:-1],width,device='cuda',dtype=dtype)*.1
                d2=torch.randn_like(source)*.1;d3x=torch.randn_like(source)*.1;d3g=torch.randn_like(seed)*.1
                def mixed(module):
                    x=source.detach().requires_grad_();g=seed.detach().clone().requires_grad_()
                    y=module(x);first=torch.autograd.grad(y,x,g,create_graph=True)[0]
                    second=torch.autograd.grad(first,(x,g),d2,create_graph=True)
                    third=torch.autograd.grad((second[0]*d3x).sum()+(second[1]*d3g).sum(),(x,g))
                    return y,first,*second,*third
                expected=mixed(Reference(list(degrees),normalize,normalization))
                with native_only():actual=mixed(Native(list(degrees),normalize,normalization))
                errors=[full(a,b,dtype)for a,b in zip(actual,expected,strict=True)]
                report['records'].append({'case':'full_mixed_second_third','dtype':str(dtype),
                    'degrees':degrees,'normalize':normalize,'normalization':normalization,'layout':layout,
                    'full_tensors':True,'errors':errors})
            # First-adjoint node: select x, output seed, or both, then exercise
            # None seeds in its multiple-output second-adjoint node.
            metadata=_program((0,1,2),True,'component');adjoint=_adjoint(metadata,(0,))
            source=torch.randn(7,3,device='cuda',dtype=dtype)+.7
            seed=torch.randn(7,9,device='cuda',dtype=dtype)*.1
            for requested in ((0,),(1,),(0,1)):
                leaves=[source.detach().clone().requires_grad_(0 in requested),
                        seed.detach().clone().requires_grad_(1 in requested)]
                refs=[v.detach().clone().requires_grad_(v.requires_grad)for v in leaves]
                with native_only():first=_SHExpression.apply(adjoint,*leaves)[0]
                expected=scalar_expression(adjoint,refs)[0]
                direction=torch.randn_like(first)*.1
                with native_only():second=torch.autograd.grad(first,tuple(leaves[i]for i in requested),direction,create_graph=True)
                second_ref=torch.autograd.grad(expected,tuple(refs[i]for i in requested),direction,create_graph=True)
                # Only the first second-adjoint output is seeded. When both
                # leaves are required the other output's gradient is None.
                with native_only():third=torch.autograd.grad(second[0].sin().sum(),tuple(leaves[i]for i in requested),allow_unused=True)
                third_ref=torch.autograd.grad(second_ref[0].sin().sum(),tuple(refs[i]for i in requested),allow_unused=True)
                report['records'].append({'case':'selected_active_and_None_second_seed','dtype':str(dtype),
                    'requested':requested,'errors':[full(first,expected,dtype)]+[full(a,b,dtype)for a,b in zip((*second,*third),(*second_ref,*third_ref),strict=True)]})
            # Constant outputs retain exact-zero gradients in the recursive AD.
            x=torch.randn(3,3,device='cuda',dtype=dtype,requires_grad=True)
            with native_only():
                y=Native([0],True,'component')(x)
                g=torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0]
                h=torch.autograd.grad(g,x,torch.ones_like(g),create_graph=True)[0]
                t=torch.autograd.grad(h,x,torch.ones_like(h))[0]
            assert not g.count_nonzero()and not h.count_nonzero()and not t.count_nonzero()
            report['records'].append({'case':'constant_exact_zero_first_second_third','dtype':str(dtype)})
            native=Native([0,1,2,3],True,'component');reference=Reference([0,1,2,3],True,'component')
            base=[torch.randn(7,3,device='cuda',dtype=dtype)*.2,
                  torch.ones(3,device='cuda',dtype=dtype)*.7,
                  torch.randn(3,device='cuda',dtype=dtype)*.01,
                  torch.randn(16,device='cuda',dtype=dtype)*.03]
            leaves=[v.detach().requires_grad_()for v in base]
            expected=physical(reference,leaves)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),native_only():actual=physical(native,leaves)
            torch.cuda.current_stream().wait_stream(stream)
            errors=check(actual,expected,dtype)
            report['records'].append({'case':'physical_all_parameters_nondefault_stream','dtype':str(dtype),
                'parameters':['scale','shift','readout'],'coordinate_gradient':True,'errors':errors})
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),native_only():
                static=[v.detach().clone().requires_grad_()for v in base]
                for _ in range(3):physical(native,static)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with native_only(),torch.cuda.graph(graph,stream=stream):result=physical(native,static)
            with torch.cuda.stream(stream),torch.no_grad():
                static[1].mul_(.83);static[3].add_(.01);graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            errors=check(result,physical(reference,static),dtype)
            report['records'].append({'case':'physical_live_parameters_CUDA_graph','dtype':str(dtype),'errors':errors})
            # Preserve the established piecewise clamp oracle away from the
            # known near-epsilon axis cancellation limitation.
            x=torch.tensor([[0.,0.,0.],[.2e-12,-.1e-12,.3e-12],
                [.6e-12,.72e-12,.4e-12],[.001,-.002,.003]],device='cuda',dtype=dtype,requires_grad=True)
            seed=torch.randn(4,16,device='cuda',dtype=dtype,requires_grad=True)
            def derive(fn):
                y=fn(x);first=torch.autograd.grad(y,x,seed,create_graph=True)[0]
                return y,first,*torch.autograd.grad(first,(x,seed),torch.ones_like(first))
            def piecewise(x):
                radius=x.square().sum(-1,keepdim=True)
                radius=torch.where(radius<1e-24,torch.full_like(radius,1e-24),radius).sqrt()
                return Reference([0,1,2,3],False,'component')(x/radius)
            expected=derive(piecewise)
            with native_only():actual=derive(native)
            report['records'].append({'case':'clamped_zero_nonaxis_near_boundary','dtype':str(dtype),
                                      'errors':check(actual,expected,dtype)})
            torch.cuda.synchronize();guard.check()
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in hashes.items())
        report.update(status='passed',after_guard=guard.check())
    except Exception as error:
        report.update(status='error',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f"{len(report['records'])} affected lightweight SH cases passed",flush=True)


if __name__=='__main__':main()
