"""Small normal latency comparison: original Torch, frozen V7 Triton, native V8."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode=True
import torch
from e3nn import o3
from benchmark_scaling_case import ExclusiveGuard
from check_native_norm_v8 import write_json
from run_native_pointwise_v8 import ROOT,V8,UUID,source_hashes
from test_native_pointwise_v8_extra import compare


def legacy_module(name,relative,rewrite=None):
    path=ROOT/'src/FastEq_v7'/relative;text=path.read_text();original=text
    if rewrite:text=text.replace(*rewrite)
    spec=importlib.util.spec_from_loader(name,loader=None);module=importlib.util.module_from_spec(spec);module.__file__=str(path);sys.modules[name]=module
    exec(compile(text,str(path),'exec'),vars(module))
    return module,{'path':str(path),'sha256':hashlib.sha256(original.encode()).hexdigest(),
        'executed_reference_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'import_alias_rewrite_only':rewrite}


def build(operator,dtype,rows):
    legacy_norm,norm_origin=legacy_module('fasteq.ops._pointwise_v7_equivariant_norm','fasteq/ops/equivariant_norm.py')
    origins=[norm_origin]
    if operator=='modulation':
        from fasteq.ops.modulation import ModulatedResidualPlan
        module,origin=legacy_module('fasteq.ops._pointwise_v7_modulation','fasteq/ops/modulation.py',
            ('from .equivariant_norm import ','from ._pointwise_v7_equivariant_norm import '));origins.append(origin)
        ir=o3.Irreps('64x0e+64x1o+64x2e');native=ModulatedResidualPlan(ir);legacy=module.ModulatedResidualPlan(ir);legacy._native_enabled=False
        leaves=[torch.randn(rows,width,device='cuda',dtype=dtype).mul_(.2).requires_grad_() for width in (native.dim,native.channels,native.dim)]
        index=torch.tensor(native._index,device='cuda');functions={'original':lambda:leaves[2]+leaves[0]*leaves[1].index_select(-1,index),
            'legacy_triton':lambda:legacy(*leaves),'native_cuda':lambda:native(*leaves)};parameters=()
        metadata={'D':native.dim,'channels':native.channels,'native_cpp_enabled':native._native_enabled,
            'legacy_cpp_disabled_reason':'V7 C++ launches old Triton pointer ABI; use its exact Python/TR arithmetic with resident launcher.'}
    elif operator.startswith('norm_'):
        from fasteq.ops.equivariant_norm import NormPlan
        if operator=='norm_tace':
            from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm
            norm=EquivariantMergeLayerNorm([0,1,2],64,centering=True)
        else:
            from prophet.layer_norm import RMSLayerNorm
            norm=RMSLayerNorm(o3.Irreps('64x0e+64x1o+64x2e'),centering=False,eps=1e-5)
        norm=norm.cuda().to(dtype);tace=operator=='norm_tace';native=NormPlan(norm,tace=tace);legacy=legacy_norm.NormPlan(norm,tace=tace)
        x=torch.randn(rows,*native.shape,device='cuda',dtype=dtype).mul_(.2).requires_grad_();leaves=[x];parameters=tuple(norm.parameters())
        functions={'original':lambda:norm(x),'legacy_triton':lambda:legacy(norm,x),'native_cuda':lambda:native(norm,x)}
        metadata={'D':native.dim,'shape':native.shape,'centering':tace}
    else:
        from fasteq.ops.gate_norm import GateNormPlan
        sys.path.insert(0,str(V8/'tests/adapters'));from test_fasteq_gate_norm_cuda import modules
        gate,norm,_,post=modules(dtype,64);native=post.fused
        module,origin=legacy_module('fasteq.ops._pointwise_v7_gate_norm','fasteq/ops/gate_norm.py',
            ('from .equivariant_norm import (','from ._pointwise_v7_equivariant_norm import ('));origins.append(origin)
        legacy=module.GateNormPlan(native.gate,norm);leaves=[torch.randn(rows,native.dim,device='cuda',dtype=dtype).mul_(.2).requires_grad_() for _ in range(2)]
        parameters=tuple(norm.parameters());functions={'original':lambda:norm(gate(leaves[0]+leaves[1])),
            'legacy_triton':lambda:legacy(norm,*leaves),'native_cuda':lambda:native(norm,*leaves)}
        metadata={'D_in':native.dim,'D_out':native.outdim,'fused_native_kernels':'one per role, plus ordered parameter row transpose'}
    return functions,leaves,parameters,metadata,origins


def chain(fn,leaves,parameters,mode):
    with torch.set_grad_enabled(mode!='forward'):output=fn()
    if mode=='forward':return(output,)
    objective=(output.sin()+.1*output.square()).mean()
    if mode=='backward':return(output,*torch.autograd.grad(objective,(*leaves,*parameters),allow_unused=True))
    first=torch.autograd.grad(objective,leaves,create_graph=True,allow_unused=True)
    higher=torch.autograd.grad(sum(g.square().mean() for g in first if g is not None and g.requires_grad),(*leaves,*parameters),allow_unused=True)
    return(output,*first,*higher)


@contextmanager
def forbid_triton():
    from triton.runtime.jit import JITFunction
    a,b=JITFunction.__getitem__,JITFunction.warmup
    def rejected(*args,**kwargs):raise AssertionError('Production pointwise path executed Triton')
    JITFunction.__getitem__=rejected;JITFunction.warmup=rejected
    try:yield
    finally:JITFunction.__getitem__=a;JITFunction.warmup=b


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--operator',choices=['modulation','gate_norm','norm_tace','norm_prophet'],required=True)
    parser.add_argument('--dtype',choices=['float32','float64'],required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rows',type=int,default=4096);parser.add_argument('--rounds',type=int,default=7);args=parser.parse_args()
    assert not args.output.exists() and os.environ['CUDA_VISIBLE_DEVICES']==UUID and not torch.cuda.is_initialized()
    args.output.parent.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,{174});torch.set_num_threads(1);torch.manual_seed(5449)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    guard=ExclusiveGuard(UUID);guard.check();dtype=getattr(torch,args.dtype)
    family=args.operator if args.operator in ('modulation','gate_norm') else 'gate_norm';pinned=source_hashes(family)
    functions,leaves,parameters,metadata,origins=build(args.operator,dtype,args.rows)
    report={'status':'running','operator':args.operator,'dtype':args.dtype,'rows':args.rows,'metadata':metadata,
        'scope':'Operator rows; feature_higher differentiates feature gradients and every live operator parameter, not physical forces',
        'measurement_kind':'normal_eager_no_profiler_or_hooks','source_sha256':pinned,'legacy_reference_origins':origins,'modes':{},
        'protocol':{'rounds':args.rounds,'iterations':3,'warmup':3,'paired_order':'rotate/reverse all3 variants','worker_cpu':174,'observer_cpu':175}}
    stop=args.output.with_suffix('.observer.stop');stop.unlink(missing_ok=True);obs=args.output.with_suffix('.observer.json')
    observer=subprocess.Popen(['taskset','-c','175',sys.executable,str(ROOT/'performance/fasteq_optimization/v8/native_matrix/observer.py'),str(os.getpid()),str(obs),str(stop)])
    try:
        from fasteq.cuda.jit import runtime
        for mode in ('forward','backward','feature_higher'):
            expected=chain(functions['original'],leaves,parameters,mode)
            with forbid_triton():actual=chain(functions['native_cuda'],leaves,parameters,mode)
            compare(actual,expected,dtype);compare(actual,chain(functions['legacy_triton'],leaves,parameters,mode),dtype)
            for fn in functions.values():
                for _ in range(3):chain(fn,leaves,parameters,mode)
            torch.cuda.synchronize();guard.check();del actual,expected
            samples={label:{'wall_ms':[],'event_ms':[]} for label in functions};orders=[];labels=list(functions)
            for round_index in range(args.rounds):
                offset=round_index%3;order=labels[offset:]+labels[:offset]
                if round_index%2:order.reverse()
                orders.append(order)
                for label in order:
                    guard.check();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start=time.perf_counter();begin.record()
                    for _ in range(3):chain(functions[label],leaves,parameters,mode)
                    end.record();end.synchronize();samples[label]['wall_ms'].append((time.perf_counter()-start)*1000/3)
                    samples[label]['event_ms'].append(begin.elapsed_time(end)/3);guard.check()
            report['modes'][mode]={'samples':samples,'orders':orders,'medians':{label:{metric:statistics.median(values) for metric,values in measured.items()} for label,measured in samples.items()},
                'paired_speedups_to_native':{label:{metric:[a/b for a,b in zip(samples[label][metric],samples['native_cuda'][metric],strict=True)] for metric in ('wall_ms','event_ms')} for label in labels if label!='native_cuda'}}
            write_json(args.output,report)
        module=runtime();report['runtime_binary']={'path':module.__file__,'sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
        for path,digest in pinned.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Source drift: '+path
        report['status']='pass'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:
        stop.touch();observer.wait(timeout=10);report['observer']=json.loads(obs.read_text())
        if report['observer']['foreign']:report['status']='foreign_gpu_discarded'
        write_json(args.output,report)
    assert report['status']=='pass';print(json.dumps({'status':report['status'],'operator':args.operator,'dtype':args.dtype,'modes':3}))


if __name__=='__main__':main()
