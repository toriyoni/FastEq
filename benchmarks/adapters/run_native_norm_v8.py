"""Affected native Norm QA; no normal timings. Every fixture is checkpointed."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
import torch
from e3nn import o3
from fasteq.ops.equivariant_norm import NormPlan
from fasteq.ops import native_equivariant_norm as native

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CANDIDATE=ROOT/'src/FastEq_v8'
UUID='GPU-e7d1b5bc-5939-b204-406e-a57dd38b7341'


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def hashes():
    paths=[Path(__file__),*[CANDIDATE/rel for rel in ('fasteq/ops/equivariant_norm.py',
        'fasteq/ops/native_equivariant_norm.py','fasteq/cuda/src/native_equivariant_norm.cuh',
        'fasteq/cuda/src/cuda_utils.hpp','fasteq/cuda/jit.py','fasteq/cuda/src/jit_runtime.cpp','fasteq/cuda/src/norm_plan.cpp')]]
    return{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def close(actual,expected,dtype):
    records=[]
    for i,(a,b) in enumerate(zip(actual,expected,strict=True)):
        if a is None or b is None:
            value=b if a is None else a
            assert value is None or not torch.count_nonzero(value)
            records.append({'slot':i,'none_or_zero':True});continue
        torch.testing.assert_close(a,b,atol=8e-5 if dtype==torch.float32 else 2e-9,
                                   rtol=2e-4 if dtype==torch.float32 else 2e-8)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta=(a.detach().double()-b.detach().double());norm=b.detach().double().norm().item()
        records.append({'slot':i,'elements':a.numel(),'full_comparison':True,
                        'max_abs':delta.abs().max().item() if a.numel() else 0,
                        'relative_l2':delta.norm().item()/max(norm,1e-30)})
    return records


def modules(dtype):
    from prophet.layer_norm import RMSLayerNorm
    from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm
    entries=[
        ('tace_balanced',EquivariantMergeLayerNorm([0,1,2],7,centering=True),True),
        ('tace_duplicate_unbalanced',EquivariantMergeLayerNorm([0,0,1,2],3,std_balance_degrees=False,centering=True),True),
        ('tace_norm_noaffine',EquivariantMergeLayerNorm([0,1,2],5,normalization='norm',centering=False,affine=False),True),
        ('prophet_centered',RMSLayerNorm(o3.Irreps('5x0e+3x1o+2x2e'),centering=True),False),
        ('prophet_rms_multiscalar',RMSLayerNorm(o3.Irreps('3x0e+2x0o+3x1o+2x2e'),centering=False,eps=1e-5),False),
        ('prophet_ragged_zero',RMSLayerNorm(o3.Irreps('0x0e+3x0e+0x1o+2x1o+1x2e'),centering=True,eps=1e-7),False),
        ('prophet_noaffine',RMSLayerNorm(o3.Irreps('5x0e+7x1o+2x2e'),centering=False,affine=False,eps=1e-5),False)]
    result=[]
    for name,module,tace in entries:
        module.to(device='cuda',dtype=dtype)
        with torch.no_grad():
            for parameter in module.parameters():parameter.copy_(.7+torch.rand_like(parameter)*.5)
        result.append((name,module,NormPlan(module,tace=tace)))
    return result


def reference(module,x,plan):
    if all(channels>0 for _,channels,*_ in plan.parts):return module(x.reshape(-1,*plan.shape)).reshape(x.shape)
    # Upstream Prophet uses ambiguous reshape(-1,0,d) for zero multiplicity.
    # Keep its exact formulas with explicit N for this extension fixture.
    rows=x.reshape(-1,plan.dim).shape[0];flat=x.reshape(rows,plan.dim);pieces=[];norms=[]
    for index,(mul,ir) in enumerate(module.irreps):
        value=flat[:,module.input_slices[index]].reshape(rows,mul,ir.dim)
        if mul and module.centering and ir.l==0:value=value-value.mean(1,keepdim=True)
        pieces.append(value)
        norms.append((value.square()/(ir.dim*len(module.irreps))).sum(1,keepdim=True).mean(2,keepdim=True))
    inverse=(torch.stack(norms).sum(0)+module.eps).rsqrt();outputs=[]
    for index,((mul,ir),value) in enumerate(zip(module.irreps,pieces)):
        output=value*module.affine_weight[index][:,None]*inverse if module.affine else value*inverse
        if module.affine_bias is not None and ir.l==0:output=output+module.affine_bias[index][:,None]
        outputs.append(output.reshape(rows,mul*ir.dim))
    return torch.cat(outputs,1).reshape(x.shape)


def feature(rows,plan,dtype,layout='plain'):
    if layout=='stride2':value=torch.randn(rows,plan.dim*2,device='cuda',dtype=dtype)[:,::2]
    elif layout=='offset':value=torch.randn(1+rows*plan.dim,device='cuda',dtype=dtype)[1:].reshape(rows,plan.dim)
    else:value=torch.randn(rows,plan.dim,device='cuda',dtype=dtype)
    return value.mul_(.2).reshape(rows,*plan.shape).requires_grad_()


def chain(fn,x,parameters,seed,requested=None,directions='all',state=None):
    leaves=(x,*parameters);requested=tuple(range(len(leaves))) if requested is None else requested
    if state is not None:state['phase']='forward'
    output=fn(x)
    if state is not None:state['phase']='first'
    first=torch.autograd.grad(output,[leaves[i] for i in requested],seed,create_graph=True,allow_unused=True)
    terms=[]
    for slot,value in zip(requested,first):
        if value is not None and value.requires_grad and (directions=='all' or directions=='u_only' and slot==0 or directions=='parameter_only' and slot>0):
            terms.append(value.square().sum()/max(value.numel(),1))
    if state is not None:state['phase']='higher'
    higher=torch.autograd.grad(sum(terms),(*leaves,seed),allow_unused=True)
    return(output,*first,*higher)


@contextmanager
def dispatch_probe():
    previous=native.launch;calls=[]
    def launch(plan,fn,arguments,x,kwargs):
        calls.append({'entry':getattr(fn,'__name__',None),'kwargs':kwargs})
        return previous(plan,fn,arguments,x,kwargs)
    native.launch=launch
    try:yield calls
    finally:native.launch=previous


def ordinary(report,path,guard):
    for dtype in (torch.float32,torch.float64):
        for index,(name,module,plan) in enumerate(modules(dtype)):
            rows=33 if index%2 else 17;x=feature(rows,plan,dtype,'stride2' if index%2 else 'offset')
            xr=x.detach().clone().requires_grad_();parameters=tuple(module.parameters());seed=torch.randn_like(x).requires_grad_()
            oracle=lambda value:reference(module,value,plan);candidate=lambda value:plan(module,value)
            cases=[('all',(0,*range(1,len(parameters)+1)),'all'),('u_only',(0,),'u_only')]
            if parameters:cases.append(('parameter_seeds',(0,*range(1,len(parameters)+1)),'parameter_only'))
            selected=next((i+1 for i,p in enumerate(parameters) if p.numel()),None)
            if selected is not None:cases.append(('one_parameter',(selected,),'all'))
            for label,requested,directions in cases:
                guard.check();expected=chain(oracle,xr,parameters,seed,requested,directions)
                with dispatch_probe() as calls:actual=chain(candidate,x,parameters,seed,requested,directions)
                precision=close(actual,expected,dtype)
                first=next(c['kwargs'] for c in calls if c['entry']=='_backward_kernel')
                if requested==(0,):assert first['WIDTH']==0 and not any(first['GGRADFLAGS']+first['BGRADFLAGS'])
                if selected is not None and requested==(selected,):assert not first['DXFLAG'] and first['WIDTH']==parameters[selected-1].numel()
                torch.cuda.synchronize();guard.check()
                report['ordinary'].append({'case':name,'dtype':str(dtype),'label':label,'requested':requested,'directions':directions,
                    'status':'pass','precision':precision,'dispatch_flags':calls,'reference':'upstream_module' if all(c>0 for _,c,*_ in plan.parts) else 'same upstream formulas with explicit zero-multiplicity N'})
                write_json(path,report)


def physical(report,path,guard):
    for dtype in (torch.float32,torch.float64):
        for name,module,plan in [modules(dtype)[0],modules(dtype)[3]]:
            coords=(torch.randn(19,3,device='cuda',dtype=dtype)*.2).requires_grad_()
            projection=(torch.randn(3,plan.dim,device='cuda',dtype=dtype)*.1).requires_grad_()
            bias=(torch.randn(plan.dim,device='cuda',dtype=dtype)*.1).requires_grad_()
            readout=(torch.randn(*plan.shape,device='cuda',dtype=dtype)*.1).requires_grad_()
            parameters=(projection,bias,readout,*module.parameters())
            def execute(fn):
                feature=(coords@projection+bias).sin().reshape(19,*plan.shape)
                output=fn(feature);energy=((output*readout).sin()+.07*output.square()).sum()
                force=-torch.autograd.grad(energy,coords,create_graph=True)[0]
                loss=(force-.017).square().mean()
                gradients=torch.autograd.grad(loss,(coords,*parameters),allow_unused=True)
                return(output,energy,force,loss,*gradients)
            guard.check();expected=execute(lambda x:reference(module,x,plan))
            with dispatch_probe() as calls:actual=execute(lambda x:plan(module,x))
            precision=close(actual,expected,dtype)
            first=next(c['kwargs'] for c in calls if c['entry']=='_backward_kernel');assert first['WIDTH']==0
            assert any(any(c['kwargs'].get('GGRADFLAGS',())) for c in calls if c['entry']=='_double_kernel')
            torch.cuda.synchronize();guard.check()
            report['physical'].append({'case':name,'dtype':str(dtype),'status':'pass','all_parameter_count':len(parameters),
                'precision':precision,'dispatch_flags':calls,'scope':'coordinates -> E -> physical F(create_graph) -> force loss -> every norm/projection/bias/readout parameter'})
            write_json(path,report)


def graph_and_live(report,path,guard):
    for name,module,plan in [modules(torch.float64)[0],modules(torch.float64)[3]]:
        dtype=torch.float64;x=feature(257,plan,dtype);parameters=tuple(module.parameters());seed=torch.randn_like(x).requires_grad_()
        operation=lambda value:plan(module,value);oracle=lambda value:reference(module,value,plan)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):chain(operation,x,parameters,seed)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=chain(operation,x,parameters,seed)
        errors=[]
        for _ in range(2):
            with torch.no_grad():
                x.copy_(torch.randn_like(x)*.2);seed.copy_(torch.randn_like(seed))
                for p in parameters:p.copy_(.7+torch.rand_like(p)*.5)
            graph.replay();errors.append(close(captured,chain(oracle,x,parameters,seed),dtype))
        torch.cuda.synchronize();guard.check()
        # Ordinary parameter replacement exercises fresh pointers after hot cache.
        old=tuple(module.parameters());module.load_state_dict({k:(.7+torch.rand_like(v)*.5 if v.is_floating_point() else v) for k,v in module.state_dict().items()},assign=True)
        assert all(a is not b for a,b in zip(old,module.parameters()))
        live=close((operation(x),),(oracle(x),),dtype)
        report['graphs'].append({'case':name,'status':'pass','nondefault_stream':True,'independent_leaf_capture':True,
            'live_input_and_parameter_replays':errors,'parameter_pointer_replacement':live});write_json(path,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']==UUID;assert not torch.cuda.is_initialized()
    assert 'v8_norm' in os.environ['TORCH_EXTENSIONS_DIR']
    from benchmark_scaling_case import ExclusiveGuard
    guard=ExclusiveGuard(UUID);before=guard.check();os.sched_setaffinity(0,{174});torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    torch.manual_seed(6443);args.output.parent.mkdir(parents=True,exist_ok=True)
    stop=args.output.with_suffix('.observer.stop');stop.unlink(missing_ok=True);obs=args.output.with_suffix('.observer.json')
    observer=subprocess.Popen(['taskset','-c','175',sys.executable,str(ROOT/'performance/fasteq_optimization/v8/native_matrix/observer.py'),str(os.getpid()),str(obs),str(stop)])
    start_hashes=hashes();report={'status':'running','ordinary':[],'physical':[],'graphs':[],
        'source_sha256':start_hashes,'guard_before':before,'normal_performance_measured':False,'tf32':False}
    try:
        ordinary(report,args.output,guard);physical(report,args.output,guard);graph_and_live(report,args.output,guard)
        for path,expected in start_hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,'Source drift: '+path
        report['guard_after']=guard.check();report['status']='pass'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:
        stop.touch();observer.wait(timeout=10);report['observer']=json.loads(obs.read_text())
        if report['observer']['foreign']:report['status']='foreign_gpu_discarded'
        write_json(args.output,report)
    assert report['status']=='pass';print(json.dumps({'status':report['status'],'ordinary':len(report['ordinary']),
        'physical':len(report['physical']),'graphs':len(report['graphs'])}))


if __name__=='__main__':main()
