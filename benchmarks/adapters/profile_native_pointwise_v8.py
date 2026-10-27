"""Isolated handwritten modulation/fused GateNorm dispatch and native kernels."""
import argparse
from contextlib import contextmanager
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
from benchmark_scaling_case import ExclusiveGuard
from check_native_norm_v8 import write_json
from run_native_pointwise_v8 import ROOT,V8,UUID,source_hashes
from test_native_pointwise_v8_extra import compare


@contextmanager
def source_probe(operator,directory):
    if operator=='modulation':
        from fasteq.ops import native_modulation as native
        modules=[native]
    else:
        from fasteq.ops import native_gate_norm as native, native_equivariant_norm as norm
        modules=[native,norm]
    records=[];state={'phase':'forward','region':'binding_warmup'};directory.mkdir(parents=True,exist_ok=True);previous=[]
    for module in modules:
        old=module.source_for;previous.append((module,old))
        def probe(role,kwargs,dtype,_module=module,_old=old):
            source=_old(role,kwargs,dtype);digest=hashlib.sha256(source.encode()).hexdigest();path=directory/(digest+'.cu')
            if not path.exists():path.write_text(source)
            records.append({'phase':state['phase'],'observed_region':state['region'],
                'owner_path':str(Path(_module.__file__).resolve()),'owner_function':'_source',
                'emitted_source_path':str(path),'emitted_source_sha256':digest,'entry':_module._ENTRIES[role],
                'role':role,'kwargs':kwargs,'kernel_count_claim':False})
            return source
        module.source_for=probe
    try:yield records,state
    finally:
        for module,old in previous:module.source_for=old;assert module.source_for is old


def chain(operation,leaves,seed,state):
    state['phase']='forward';y=operation(*leaves[:3])
    state['phase']='first';first=torch.autograd.grad(y,leaves,seed,create_graph=True,allow_unused=True)
    state['phase']='higher';loss=sum(g.square().sum()/max(g.numel(),1) for g in first if g is not None and g.requires_grad)
    second=torch.autograd.grad(loss,(*leaves,seed),allow_unused=True)
    return(y,*first,*second)


def fixture(operator,dtype,resident,rows):
    if operator=='modulation':
        from fasteq.ops.modulation import ModulatedResidualPlan
        os.environ['FASTEQ_EXPERIMENTAL_NATIVE_MODULATION']='1' if resident else '0'
        plan=ModulatedResidualPlan(o3.Irreps('64x0e+64x1o+64x2e'))
        leaves=[torch.randn(rows,width,device='cuda',dtype=dtype).mul_(.2).requires_grad_() for width in (plan.dim,plan.channels,plan.dim)]
        seed=torch.randn_like(leaves[0]).requires_grad_();index=torch.tensor(plan._index,device='cuda')
        operation=plan;oracle=lambda x,g,r:r+x*g.index_select(-1,index)
    else:
        # Use the real Prophet postprocess Gate/Norm metadata, without model mm.
        sys.path.insert(0,str(V8/'tests/adapters'))
        from test_fasteq_gate_norm_cuda import modules
        gate,norm,_,post=modules(dtype,64);plan=post.fused
        leaves=[torch.randn(rows,plan.dim,device='cuda',dtype=dtype).mul_(.2).requires_grad_() for _ in range(2)]
        leaves.append(torch.empty(0,device='cuda',dtype=dtype))
        leaves.extend(norm.parameters());seed=torch.randn(rows,plan.outdim,device='cuda',dtype=dtype).requires_grad_()
        operation=lambda projected,skip,unused:plan(norm,projected,skip)
        oracle=lambda projected,skip,unused:norm(gate(projected+skip))
        # The placeholder is only the uniform callable ABI; do not request AD.
    return plan,leaves,seed,operation,oracle


def execute(operator,operation,leaves,seed,state):
    if operator=='modulation':return chain(operation,leaves,seed,state)
    state['phase']='forward';y=operation(*leaves[:3]);active=(*leaves[:2],*leaves[3:])
    state['phase']='first';first=torch.autograd.grad(y,active,seed,create_graph=True,allow_unused=True)
    state['phase']='higher';second=torch.autograd.grad(sum(g.square().sum()/max(g.numel(),1) for g in first if g is not None),(*active,seed),allow_unused=True)
    return(y,*first,*second)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--operator',choices=['modulation','gate_norm'],required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUID and not torch.cuda.is_initialized()
    guard=ExclusiveGuard(UUID);guard.check();os.sched_setaffinity(0,{174});torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    args.output.parent.mkdir(parents=True,exist_ok=True);stop=args.output.with_suffix('.observer.stop');stop.unlink(missing_ok=True);obs=args.output.with_suffix('.observer.json')
    observer=subprocess.Popen(['taskset','-c','175',sys.executable,str(ROOT/'performance/fasteq_optimization/v8/native_matrix/observer.py'),str(os.getpid()),str(obs),str(stop)])
    pinned=source_hashes(args.operator);report={'status':'running','operator':args.operator,'profiles':[],
        'source_sha256':pinned,'normal_performance_measured':False}
    try:
        configurations=[(torch.float32,False),(torch.float64,False)]
        if args.operator=='modulation':configurations+=[(torch.float32,True),(torch.float64,True)]
        for dtype,resident in configurations:
            label=str(dtype).split('.')[-1]+('_cpp' if resident else '_python');stem=args.output.parent/('profile_'+label)
            plan,leaves,seed,operation,oracle=fixture(args.operator,dtype,resident,4096)
            with source_probe(args.operator,stem.with_name(stem.name+'_sources')) as (dispatch,state):
                # Cold preparation records the actual source bound into the
                # resident C++ cache. It is outside the diagnostic trace.
                result=execute(args.operator,operation,leaves,seed,state)
                expected=execute(args.operator,oracle,leaves,seed,{'phase':None});compare(result,expected,dtype)
                for _ in range(2):execute(args.operator,operation,leaves,seed,state)
                torch.cuda.synchronize();guard.check();state['region']='isolated_profiler'
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    execute(args.operator,operation,leaves,seed,state);torch.cuda.synchronize()
            guard.check();trace=stem.with_suffix('.trace.json');profiler.export_chrome_trace(str(trace))
            manifest={'candidate_root':str(V8),'isolated_operator':args.operator+'_'+label,'scope':'operator_and_adjoints',
                'phases':['forward','first','higher'],'source_files':[{'path':p,'sha256':d} for p,d in pinned.items()],
                'dispatch':dispatch,'hooks_restored':True,'timing_claim':False,
                'resident_binding_note':'C++ warmed executable binding is observed before profile; actual subsequent kernels are retained in trace.' if resident else None}
            bindings=stem.with_suffix('.bindings.json');write_json(bindings,manifest)
            checker=ROOT/'performance/fasteq_optimization/v8/check_native_evidence.py';spec=importlib.util.spec_from_file_location('pointwise_native_evidence',checker)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            evidence=module.inspect_evidence(json.loads(trace.read_text()),manifest);write_json(stem.with_suffix('.evidence.json'),evidence)
            assert evidence['status']=='pass',evidence
            report['profiles'].append({'label':label,'rows':4096,'status':'pass','trace':str(trace),'bindings':str(bindings),'kernel_evidence':evidence,
                'precision':'Full F/first/double errors checked before trace; no sampling'});write_json(args.output,report)
        for p,digest in pinned.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,'Source drift: '+p
        report['status']='pass'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:
        stop.touch();observer.wait(timeout=10);report['observer']=json.loads(obs.read_text())
        if report['observer']['foreign']:report['status']='foreign_gpu_discarded'
        write_json(args.output,report)
    assert report['status']=='pass';print(json.dumps({'status':report['status'],'profiles':len(report['profiles'])}))


if __name__=='__main__':main()
