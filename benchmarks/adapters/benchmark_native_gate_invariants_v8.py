"""Normal Gate/Invariant/SH latency and separate native-only kernel profiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from native_uvw_v8_common import ROOT, CANDIDATE, distribution, parse_trace, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(operator, rows, dtype, include_library_sh_reference=False):
    import torch
    from e3nn import o3
    from fasteq.ops.gate import GatePlan
    from fasteq.ops.native_gate import FastGateFunction
    from fasteq.triton.fused_equivariant_gate import FastGateFunction as LegacyGate
    if operator.startswith('gate_'):
        if operator=='gate_o3':
            from e3nn.nn import Gate
            module=Gate('64x0e',[torch.nn.SiLU()],'128x0e',[torch.nn.Sigmoid()],'64x1o+64x2e');kind='o3'
        elif operator=='gate_prophet':
            from prophet.model import Gate
            module=Gate('192x0e+64x1o+64x2e');kind='prophet'
        else:
            from scaling_operator_factories import _tece_template
            module=_tece_template().nonlinearity;kind='o2'
        module=module.cuda().to(dtype);native=GatePlan(module,kind)
        x=torch.randn(rows,native.dim,device='cuda',dtype=dtype).requires_grad_()
        with torch.no_grad():native(x)
        tables=native._cache[x.device,x.dtype]
        functions={'original':module,'legacy_triton':lambda x:LegacyGate.apply(x.contiguous(),*tables),'native_cuda':native}
        metadata={'input_dim':native.dim,'output_dim':native.outdim,'gate_kind':kind,
                  'actual_candidate':'GatePlan.__call__ -> native_gate.FastGateFunction',
                  'reference':'Original module, identical interface/normalization/maps.'}
    elif operator=='invariant':
        from fasteq.adapters.programs import FFNPlan
        from fasteq.triton.irreps_invariants import InvariantPlan as LegacyInvariant
        from types import SimpleNamespace
        ir=o3.Irreps('64x0e+64x1o+64x2e')
        native=FFNPlan(SimpleNamespace(irreps=ir)).norm
        legacy=LegacyInvariant(ir)
        def original(x):
            return torch.cat([x[:,sl].reshape(rows,mul,irrep.dim).square().mean(-1)
                              for (mul,irrep),sl in zip(ir,ir.slices(),strict=True)],-1)
        x=torch.randn(rows,ir.dim,device='cuda',dtype=dtype).requires_grad_()
        functions={'original':original,'legacy_triton':legacy,'native_cuda':native}
        metadata={'input_dim':ir.dim,'output_dim':native.channels,'definition':'per-copy sumsq/angularwidth',
                  'actual_candidate':'FFNPlan.norm -> native_invariants.InvariantPlan',
                  'reference':'Original EQFFN extracted Torch loop/cat; not a standalone original module.'}
    else:
        from fasteq.ops.spherical_harmonics import spherical_harmonics as native
        from fasteq.triton.spherical_harmonics import spherical_harmonics as legacy
        x=torch.randn(rows,3,device='cuda',dtype=dtype).requires_grad_()
        functions={'original':lambda x:o3.spherical_harmonics([0,1,2],x,True,'component'),
                   'legacy_triton':lambda x:legacy([0,1,2],x,True,'component'),
                   'native_cuda':lambda x:native([0,1,2],x,True,'component')}
        if include_library_sh_reference:
            import importlib.util
            archived=ROOT/'performance/fasteq_optimization/v8/sh/measured_component_source/fasteq/ops/spherical_harmonics.py'
            spec=importlib.util.spec_from_file_location('fasteq.ops._sh_library_v8_measured_reference',archived)
            reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
            functions['native_library']=lambda x:reference.spherical_harmonics([0,1,2],x,True,'component')
        metadata={'input_dim':3,'output_dim':9,'degrees':[0,1,2],'normalize':True,'normalization':'component',
                  'reference':'Original e3nn normalized SH; native keeps original FastEq recurrence.'}
        if include_library_sh_reference:
            metadata['native_library_reference']='Archived original V8 SH wrapper, existing torch.library custom_op and AD; raw launcher body shared unchanged after helper extraction.'
    return x,functions,metadata


def chain(fn,x,mode):
    import torch
    with torch.set_grad_enabled(mode!='forward'):y=fn(x)
    if mode=='forward':return(y,)
    objective=(y.sin()+.1*y.square()).mean()
    if mode=='backward':return y,torch.autograd.grad(objective,x)[0]
    first=torch.autograd.grad(objective,x,create_graph=True)[0]
    return y,first,torch.autograd.grad(first.square().mean(),x)[0]


def precision(actual,expected,dtype,operator):
    import torch
    if operator=='invariant':
        atol,rtol=(2e-5,2e-4)if dtype==torch.float32 else(2e-11,2e-10);factors=(1,5,20)
    else:
        atol,rtol=(3e-5,3e-5)if dtype==torch.float32 else(2e-10,2e-9);factors=(1,10,30)
    errors=[]
    for index,(a,b)in enumerate(zip(actual,expected,strict=True)):
        torch.testing.assert_close(a,b,atol=atol*factors[index],rtol=rtol*factors[index])
        assert torch.isfinite(a).all()and torch.isfinite(b).all()
        delta=a.detach().double()-b.detach().double()
        errors.append({'slot':index,'full_elements':a.numel(),'max_abs':delta.abs().max().item(),
                       'relative_l2':delta.norm().item()/max(b.detach().double().norm().item(),1e-30)})
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',choices=['gate_o3','gate_prophet','gate_o2','invariant','sh'],required=True)
    p.add_argument('--rows',type=int,default=4096)
    p.add_argument('--dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--mode',choices=['forward','backward','feature_higher'],default='feature_higher')
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=9);p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--iterations',type=int,default=3);p.add_argument('--profile',action='store_true')
    p.add_argument('--include-library-sh-reference',action='store_true')
    args=p.parse_args()
    if args.include_library_sh_reference and args.operator!='sh':p.error('Library dispatch comparison is SH-only')
    if args.output.exists():raise RuntimeError('Preserve existing whole attempt output; use a new destination')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    import torch
    from benchmark_scaling_case import ExclusiveGuard
    from check_native_gate_invariants_v8 import native_only
    if args.operator=='sh':
        from native_sh_dispatch_spy_v8 import native_only
    os.sched_setaffinity(0,{172});torch.set_num_threads(1);torch.manual_seed(594217)
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu_uuid
    torch.backends.cuda.matmul.allow_tf32=False
    guard=ExclusiveGuard(args.gpu_uuid);before=guard.check()
    if args.operator=='sh':
        names=['fasteq/ops/spherical_harmonics.py','fasteq/ops/scalar_expression.py','fasteq/triton/spherical_harmonics.py',
               'benchmarks/adapters/native_sh_dispatch_spy_v8.py']
    else:
        names=['fasteq/ops/native_gate.py','fasteq/cuda/src/native_gate.cuh','fasteq/ops/native_invariants.py',
               'fasteq/cuda/src/native_invariants.cuh','fasteq/cuda/src/equi_linear_kernels/device_cuda_helper.cuh',
               'fasteq/ops/gate.py','fasteq/adapters/programs.py','fasteq/triton/fused_equivariant_gate.py',
               'fasteq/triton/irreps_invariants.py']
    names+=['fasteq/cuda/jit.py','fasteq/cuda/src/jit_runtime.cpp','fasteq/cuda/src/jit_runtime.h']
    hashes={str(CANDIDATE/name):sha(CANDIDATE/name)for name in names}
    hashes[str(Path(__file__).resolve())]=sha(__file__)
    if args.include_library_sh_reference:
        archive=ROOT/'performance/fasteq_optimization/v8/sh/measured_component_source/fasteq/ops/spherical_harmonics.py'
        hashes[str(archive)]=sha(archive)
    report={'status':'running','operator':args.operator,'dtype':args.dtype,'rows':args.rows,'mode':args.mode,
            'atoms_scope':'independent_operator_rows; no atom graph','source_sha256':hashes,
            'fasteq_environment':{key:value for key,value in sorted(os.environ.items())
                                  if key.startswith('FASTEQ_')},
            'before_guard':before,'scope':'feature_higher differentiates feature gradients, not physical coordinates/model loss',
            'protocol':{'rounds':args.rounds,'warmup':args.warmup,'iterations':args.iterations,
                        'ordering':'Same process, all variants rotated/reversed by round; do not mix with fresh-process matrix timings.',
                        'workers_cpu':172,'observer_cpu':171,'observer_interval_s':.02,
                        'observer_overhead':'Not independently measured; polling can miss shorter foreign jobs.'}}
    monitor=None;observer_dir=args.output.with_suffix('.observer')
    try:
        dtype=getattr(torch,args.dtype);x,fns,metadata=build(args.operator,args.rows,dtype,args.include_library_sh_reference)
        report['metadata']=metadata
        # Precision and spies are outside all normal timed blocks.
        original=chain(fns['original'],x,args.mode)
        with native_only():actual=chain(fns['native_cuda'],x,args.mode)
        report['precision_original']=precision(actual,original,dtype,args.operator)
        report['precision_legacy']=precision(actual,chain(fns['legacy_triton'],x,args.mode),dtype,args.operator)
        if 'native_library' in fns:
            report['precision_native_library']=precision(actual,chain(fns['native_library'],x,args.mode),dtype,args.operator)
        for fn in fns.values():
            for _ in range(args.warmup):chain(fn,x,args.mode)
        torch.cuda.synchronize();guard.check()
        from fasteq.cuda.jit import runtime
        report['runtime_binary']={'path':runtime().__file__,'sha256':sha(runtime().__file__)}
        if args.profile:
            with native_only(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                 torch.profiler.ProfilerActivity.CUDA])as prof:
                chain(fns['native_cuda'],x,args.mode);torch.cuda.synchronize()
            trace=args.output.with_suffix('.trace.json');prof.export_chrome_trace(str(trace))
            profile=parse_trace(trace)
            assert not profile['forbidden_library_kernel_names']and profile['torch_matmul_cpu_ops']==0
            assert not any('triton' in row['name'].lower()for row in profile['kernels'])
            if args.operator=='sh':
                trace_events=json.loads(trace.read_text())['traceEvents']
                assert not any('eqx::' in event.get('name','')for event in trace_events if event.get('cat')=='cpu_op')
                profile['scalar_custom_op_cpu_events']=sum(event.get('name')=='fasteq::scalar_expression'
                                                           for event in trace_events if event.get('cat')=='cpu_op')
                assert profile['scalar_custom_op_cpu_events']==0,'Candidate profile selected library fallback, not eager SH'
            report.update(measurement_kind='instrumented_native_profile_not_normal_latency',profile=profile,
                          trace=str(trace),trace_sha256=sha(trace),native_only=True)
        else:
            observer=ROOT/'performance/fasteq_optimization/v7/driver/gpu_foreign_watch.py'
            command=[sys.executable,str(observer),'--uuid',args.gpu_uuid,'--owner-pid',str(os.getpid()),
                     '--directory',str(observer_dir),'--cpu','171','--interval','.02']
            monitor=subprocess.Popen(command,stdout=args.output.with_suffix('.observer.log').open('w'),stderr=subprocess.STDOUT)
            started=time.monotonic()
            while not(observer_dir/'ready').exists():
                if monitor.poll()is not None or time.monotonic()-started>20:raise RuntimeError('Observer not ready')
                time.sleep(.05)
            labels=list(fns);samples={name:{'wall_ms':[],'event_ms':[]}for name in labels};orders=[]
            for number in range(args.rounds):
                shift=number%len(labels);order=labels[shift:]+labels[:shift]
                if number%2:order.reverse()
                orders.append(order)
                for label in order:
                    guard.check();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin=time.perf_counter();start.record()
                    for _ in range(args.iterations):chain(fns[label],x,args.mode)
                    end.record();end.synchronize();wall=(time.perf_counter()-begin)*1000/args.iterations
                    samples[label]['wall_ms'].append(wall);samples[label]['event_ms'].append(start.elapsed_time(end)/args.iterations)
                    guard.check()
            report.update(measurement_kind='normal_eager_no_profiler_or_spies_in_timing',samples=samples,orders=orders,
                          medians={label:{key:statistics.median(values)for key,values in pair.items()}for label,pair in samples.items()},
                          paired_ratios_to_native={label:{key:distribution([a/b for a,b in zip(samples[label][key],samples['native_cuda'][key],strict=True)])
                              for key in ('wall_ms','event_ms')}for label in labels if label!='native_cuda'})
        assert all(sha(path)==value for path,value in hashes.items()),'Component source drift'
        report.update(status='ok',after_guard=guard.check())
    except Exception as error:
        report.update(status='error',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        if monitor is not None:
            (observer_dir/'stop').touch();monitor.wait(timeout=15)
            observation=json.loads((observer_dir/'observer.json').read_text());report['observer']=observation
            if observation.get('status')!='complete'or observation.get('foreign_observations'):
                report['status']='rejected_whole_attempt'
        write_json(args.output,report)
    if report['status']!='ok':raise RuntimeError('Rejected complete attempt; preserve all samples')
    print(json.dumps({'status':report['status'],'operator':args.operator,'dtype':args.dtype,
                      'medians':report.get('medians'),'profile':report.get('profile',{}).get('kernel_only_total_us')}),flush=True)


if __name__=='__main__':main()
