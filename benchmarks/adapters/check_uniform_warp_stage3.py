"""Only the new scalar-Y warp domain, including shared adjoints and tails."""
import json
import os
from pathlib import Path
import sys
import torch
from benchmark_uniform_stage3 import configure,UUID,CAND
from benchmark_scaling_case import ExclusiveGuard
from fasteq.ops.segmented_product import segmented_product

sys.path.insert(0,str(CAND/'tests/adapters'))
from test_uniform_stage3_v7 import make_module,close,chain
from fasteq.adapters import adapt_model


def main():
    configure();torch.manual_seed(19331);os.environ['FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y']='1'
    os.environ['FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y_WARP']='1'
    guard=ExclusiveGuard(UUID);before=guard.check();passed=[]
    for dtype in (torch.float32,torch.float64):
        for kind,external,mixed,unweighted,channels,rows in (
            ('o2',False,False,False,33,257),
            ('o3',True,True,False,64,17),
            ('o3',False,False,True,33,17),
        ):
            guard.check();original=make_module(kind,dtype,external=external,mixed=mixed,
                                               unweighted=unweighted,channels=channels)
            candidate=adapt_model(original)
            args=[torch.randn(width,rows,device='cuda',dtype=dtype).mul_(.2).T.requires_grad_()
                  for width in (original.irreps_in1.dim,original.irreps_in2.dim)]
            if external:args.append(torch.randn(original.weight_numel,rows,device='cuda',dtype=dtype).mul_(.2).T.requires_grad_())
            os.environ['FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y_WARP']='0'
            reference=chain(original,args);serial=chain(candidate,args)
            close(serial,reference,dtype)
            os.environ['FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y_WARP']='1'
            actual=chain(candidate,args)
            close(actual,reference,dtype);close(actual,serial,dtype)
            close(chain(candidate,args,subset=(0,)),chain(original,args,subset=(0,)),dtype)
            torch.cuda.synchronize();guard.check()
            passed.append(f'{kind}/{dtype}/{channels}/{rows}/external{external}/mixed{mixed}/unweighted{unweighted}')
        # A shared Y leaf forces role2's bounded row partial/reduction path.
        # The unused Y coordinates force compact indexing; shared W exercises
        # required-bit-zero when weighted and unweighted paths coexist.
        dims=(33,66,32,66)
        terms=((1,0,0,2,7,1,0,2,33,1,33,1),
               (2,0,1,2,11,1,1,2,33,1,33,0))
        metadata=repr((dims,terms,(.7,-.3)))
        args=(torch.randn(1,33,device='cuda',dtype=dtype).mul_(.2).requires_grad_(),
              torch.randn(1025,66,device='cuda',dtype=dtype).mul_(.2).requires_grad_(),
              torch.randn(1,32,device='cuda',dtype=dtype).mul_(.2).requires_grad_())
        def run(warp):
            os.environ['FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y_WARP']=str(warp)
            z=segmented_product(metadata,*args)
            first=torch.autograd.grad((z.sin()+.1*z.square()).sum(),args,create_graph=True)
            second=torch.autograd.grad(sum(g.square().sum()for g in first),args)
            return(z,*first,*second)
        close(run(1),run(0),dtype);torch.cuda.synchronize();guard.check()
        passed.append(f'shared-Y/compact/chunked/mixed/{dtype}')
        os.environ['FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y_WARP']='1'
    report={'status':'pass','unique_cases':len(passed),'passed':passed,'before_guard':before,'after_guard':guard.check()}
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
