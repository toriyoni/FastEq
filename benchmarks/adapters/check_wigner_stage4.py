"""Targeted store-tile correctness; no scale/OOM or broad suite rerun."""
import json
import os
import sys
import torch
from benchmark_wigner_stage4 import configure,RoundGuard,FLAG,CAND,compare_values,source_records
from fasteq.ops.scalar_expression import ScalarProgram,scalar_expression,_parse,_store_tile_configuration,_source
from fasteq.ops.rotation import wigner_packed,_wigner_metadata
from eqx.o2 import WignerD
sys.path.insert(0,str(CAND/'tests/adapters'))
from eqx_cuda_spy import forbid_eqx_cuda


def chain(fn,inputs):
    z=fn(*inputs);groups=(z,)if isinstance(z,torch.Tensor)else tuple(z)
    objective=sum(v.sin().sum()+.1*v.square().sum()for v in groups)
    first=(torch.autograd.grad(objective,inputs,create_graph=True,allow_unused=True)
           if objective.requires_grad else tuple(None for _ in inputs))
    terms=[g.square().sum()for g in first if g is not None and g.requires_grad]
    second=torch.autograd.grad(sum(terms),inputs,allow_unused=True)if terms else tuple(None for _ in inputs)
    return(z,first,second)


def native(x,degree,flag):
    os.environ[FLAG]=str(flag);return wigner_packed(x,degree)


def checks():
    passed=[]
    for dtype in (torch.float32,torch.float64):
        tolerance=dict(atol=3e-4,rtol=3e-4)if dtype==torch.float32 else dict(atol=3e-8,rtol=3e-7)
        for degree in range(4):
            previous=torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
                frame=WignerD(mmax=min(1,degree),lmax=degree,method='auto').to(device='cuda',dtype=dtype)
            finally:torch.set_default_dtype(previous)
            for rows in (0,1,31,33,63,65,129):
                x=(torch.randn(3,rows+2,device='cuda',dtype=dtype).mul_(.2).T[1:rows+1]
                   +torch.tensor([.7,.4,-.3],device='cuda',dtype=dtype))
                # Preserve a genuinely noncontiguous input and nonzero offset.
                base=torch.empty(3,rows+2,device='cuda',dtype=dtype);base.T[1:rows+1].copy_(x)
                x=base.T[1:rows+1].requires_grad_()
                legacy=chain(lambda v:native(v,degree,0),(x,))
                tiled=chain(lambda v:native(v,degree,1),(x,))
                compare_values(tiled,legacy,dict(atol=0.,rtol=0.))
                if rows:
                    compare_values(tiled,chain(frame.forward_packed,(x,)),tolerance)
            passed.append(f'degree{degree}/{dtype}/tails0,1,31,33,63,65,129/noncontig/first+higher')
        # Multi-output IR includes an empty group, plus higher adjoints that
        # return gradients for two input slots and output seed slots.
        p=ScalarProgram();a=[p.add('input',0,i)for i in range(3)];b=[p.add('input',1,i)for i in range(3)]
        wide=tuple(p.add('add',p.add('mul',a[i%3],b[(i+1)%3]),p.constant(i/17))for i in range(9))
        groups=((),wide,tuple(a));metadata=repr((tuple(p.nodes),groups,(3,3)))
        x,y=[torch.randn(33,3,device='cuda',dtype=dtype).mul_(.2).requires_grad_()for _ in range(2)]
        def expression(flag):
            os.environ[FLAG]=str(flag);return chain(lambda a,b:scalar_expression(metadata,[a,b]),(x,y))
        compare_values(expression(1),expression(0),dict(atol=0.,rtol=0.))
        # Use only the wide output: None seeds for the other groups must work.
        def subset(flag):
            os.environ[FLAG]=str(flag);z=scalar_expression(metadata,[x,y])[1]
            first=torch.autograd.grad(z.sin().sum(),(x,),create_graph=True)[0]
            return(z,first,torch.autograd.grad(first.square().sum(),(x,y),allow_unused=True))
        compare_values(subset(1),subset(0),dict(atol=0.,rtol=0.))
        passed.append(f'multigroup0,9,3/twoinputs/None-seeds/subset/{dtype}')
        coords=torch.randn(33,3,device='cuda',dtype=dtype).mul_(.2).requires_grad_()
        matrix=torch.randn(3,3,device='cuda',dtype=dtype).mul_(.03).requires_grad_()
        bias=torch.tensor([.7,.4,-.3],device='cuda',dtype=dtype).requires_grad_()
        def physical(flag):
            os.environ[FLAG]=str(flag);z=wigner_packed(coords@matrix+bias,2)
            energy=z.sin().sum()+.1*z.square().sum()
            force=-torch.autograd.grad(energy,coords,create_graph=True)[0]
            final=torch.autograd.grad((force-.13).square().mean(),(coords,matrix,bias))
            return(z,force,final)
        compare_values(physical(1),physical(0),dict(atol=0.,rtol=0.))
        passed.append(f'coordinate-loss/full-trainable-matrix+bias/{dtype}')
    # Warm each degree/dtype/transpose strategy already above; capture uses
    # fresh static leaves independent from the eager no-EQX instrumentation.
    dtype=torch.float64;x=torch.randn(65,3,device='cuda',dtype=dtype).mul_(.2).add_(.5).requires_grad_()
    os.environ[FLAG]='1'
    with forbid_eqx_cuda()as calls:expected=chain(lambda v:wigner_packed(v,2),(x,))
    assert not calls
    static=x.detach().clone().requires_grad_();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):actual=chain(lambda v:wigner_packed(v,2),(static,))
    torch.cuda.current_stream().wait_stream(stream)
    compare_values(actual,expected,dict(atol=0.,rtol=0.))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):recorded=chain(lambda v:wigner_packed(v,2),(static,))
    with torch.no_grad():static.add_(.01)
    graph.replay();compare_values(recorded,chain(lambda v:wigner_packed(v,2),(static,)),dict(atol=0.,rtol=0.))
    passed.append('degree2/f64/fresh-static/nondefaultstream/Graph-live-input/no-EQX')
    return passed


def main():
    configure();guard=RoundGuard();hashes=source_records();attempt=0
    while True:
        attempt+=1;before=guard.wait('before_qa',attempt=attempt);torch.manual_seed(73129)
        passed=checks();torch.cuda.synchronize()
        try:after=guard.guard.check();break
        except Exception as error:
            from benchmark_scaling_case import SharedGPUError
            if not isinstance(error,SharedGPUError):raise
            guard.reject('after_qa',error,attempt=attempt)
    assert source_records()==hashes
    print(json.dumps({'status':'pass','unique_cases':len(passed),'passed':passed,'attempts':attempt,
                      'before_guard':before,'after_guard':after,'rejections':guard.rejections,'source_sha256':hashes}),flush=True)


if __name__=='__main__':main()
