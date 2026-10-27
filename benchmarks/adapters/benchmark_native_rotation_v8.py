"""Actual TECE Frame/Rotary wrappers: normal V7/native comparisons, no profiler."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True
ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND=ROOT/'src/FastEq_v8'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain(function,inputs,parameters):
    import torch
    y=function(*inputs)
    loss=(y.sin()+.1*y.square()).mean()
    first=torch.autograd.grad(loss,inputs,create_graph=True,allow_unused=True)
    terms=[g.square().mean()for g in first if g is not None and g.requires_grad]
    higher=torch.autograd.grad(sum(terms),(*inputs,*parameters),allow_unused=True)
    return y,*first,*higher


def comparison(actual,expected,dtype):
    import torch
    from check_native_rotation_v8 import tolerance
    errors=[]
    for index,(a,b)in enumerate(zip(actual,expected,strict=True)):
        if a is None or b is None:
            value=b if a is None else a
            assert value is None or not torch.count_nonzero(value)
            errors.append({'slot':index,'zero_or_none':True});continue
        torch.testing.assert_close(a,b,**tolerance(dtype))
        assert torch.isfinite(a).all()and torch.isfinite(b).all()
        delta=a.detach().double()-b.detach().double()
        errors.append({'slot':index,'full_elements':a.numel(),'max_abs':delta.abs().max().item(),
            'relative_l2':delta.norm().item()/max(b.detach().double().norm().item(),1e-30)})
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator',choices=('rotate','rotary'),required=True)
    parser.add_argument('--dtype',choices=('float32','float64'),required=True)
    parser.add_argument('--atoms',type=int,default=4096)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=9)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=3)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Preserve whole attempt; choose new output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    import torch
    import fasteq.ops.rotation as native
    from benchmark_scaling_case import ExclusiveGuard
    from scaling_operator_factories import build_case
    from native_uvw_v8_common import distribution,write_json
    os.sched_setaffinity(0,{172});torch.set_num_threads(1);torch.manual_seed(831424)
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu_uuid
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    guard=ExclusiveGuard(args.gpu_uuid);before=guard.check()
    reference_path=ROOT/'src/FastEq_v7/fasteq/ops/rotation.py'
    spec=importlib.util.spec_from_file_location('fasteq.ops._v7_rotation_perf_reference',reference_path)
    naive=importlib.util.module_from_spec(spec);sys.modules[spec.name]=naive;spec.loader.exec_module(naive)
    source_files=[Path(__file__),reference_path]+[CAND/name for name in (
        'benchmarks/adapters/scaling_operator_factories.py','benchmarks/adapters/check_native_rotation_v8.py',
        'fasteq/ops/rotation.py','fasteq/cuda/src/fctp_rotation.cuh','fasteq/cuda/src/fctp_shared.cuh',
        'fasteq/cuda/src/cuda_utils.hpp','fasteq/cuda/jit.py','fasteq/cuda/src/jit_runtime.cpp','fasteq/cuda/src/jit_runtime.h')]
    source_files+=[ROOT/'src/tace/tace/models/_e3nn/tece_oam_rra.py',ROOT/'src/tace/eqx/o2/linear.py']
    pending=[path for path in source_files if path.suffix in ('.h','.cuh','.hpp')]
    closure=set()
    while pending:
        path=pending.pop().resolve()
        if path in closure:continue
        closure.add(path)
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"',path.read_text(),flags=re.MULTILINE):
            candidates=[path.parent/include,CAND/'fasteq/cuda/src'/include]
            target=next((p.resolve()for p in candidates if p.is_file()),None)
            if target is not None and target.is_relative_to(CAND):pending.append(target)
    source_files+=sorted(closure)
    hashes={str(p):sha(p)for p in source_files}
    report={'status':'running','operator':args.operator,'dtype':args.dtype,'atoms_equivalent':args.atoms,
        'rows':args.atoms*12,'mode':'feature_higher','before_guard':before,'source_sha256':hashes,
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'caches':{key:os.environ.get(key)for key in ('TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME','TRITON_CACHE_DIR')},
        'fasteq_environment':{key:value for key,value in sorted(os.environ.items())if key.startswith('FASTEQ_')},
        'scope':'Actual TECE wrapper; nonlinear output loss -> all feature inputs(create_graph) -> feature-gradient loss -> feature inputs plus all wrapper parameters. No physical coords or graph; prior physical QA reused.',
        'protocol':{'rounds':args.rounds,'warmup':args.warmup,'iterations':args.iterations,
            'ordering':'Same process, three variants rotated/reversed; no profiler/spies in normal blocks.',
            'worker_cpu':172,'observer_cpu':171,'observer_interval_s':.02,
            'observer_limit':'Observer overhead not separately measured; polling may miss shorter foreign processes.'},
        'references':{'original':'Actual scaling factory original fused=False Torch rotation/rotary expression. TECE original o2 Linear/TensorProduct/Wigner are EQX interfaces; common rotary radial_proj is EQX o2.Linear, implemented with ordinary Torch matmul/bias and included unchanged in all variants. EQX Wigner auto can execute its CUDA; Wigner generation is excluded here.',
            'v7_naive':'Frozen V7 rotation.py CUDA expression implementation (source originally adapted from EQX), benchmark-only reference. Imported under distinct module name; same current FastEq compiler/launcher executes its unchanged old CUDA sources.',
            'v8_native':'Frozen V8 hand-written CUDA from original FastEq FCTP shared transpose/vector helpers.'},
        'routing':'For rotary only, choose native or V7 raw rotary_product alias before each block; calls inside timer use unchanged Convolution.rotary_score wrapper. Restore original identity in finally. Routing assignments are outside timers.'}
    monitor=None;original_rotary=native.rotary_product
    try:
        dtype=getattr(torch,args.dtype);case=build_case(args.operator,args.atoms,dtype)
        inputs=case['inputs'];report['factory_metadata']=case['metadata']
        if args.operator=='rotate':
            angular=case['metadata']['angular_metadata']
            degrees_in,degrees_out=__import__('ast').literal_eval(angular)
            mask=inputs[1].new_tensor([[a==b for b in degrees_in]for a in degrees_out])
            functions={'original':case['baseline'],'v7_naive':lambda x,m:naive.rotate(x,m*mask,angular),
                       'v8_native':case['candidate']}
            parameters={label:()for label in functions}
            report['actual_shape']='TECE C64 asymmetric paired 2C: rows×9×128 -> rows×7×128, matrix rows×7×9; constant degree mask included in every variant.'
        else:
            functions={'original':case['baseline'],'v7_naive':case['candidate'],'v8_native':case['candidate']}
            parameters={'original':case['baseline_parameters'],'v7_naive':case['candidate_parameters'],
                        'v8_native':case['candidate_parameters']}
            report['actual_shape']='TECE C64: query/key rows×7×64, radial rows×8, heads4; radial projection, trig, bias and temperature included unchanged.'
        def select(label):
            if args.operator=='rotary':native.rotary_product=naive.rotary_product if label=='v7_naive'else original_rotary
            return lambda:chain(functions[label],inputs,parameters[label])
        actual=select('v8_native')()
        report['precision']={label:comparison(actual,select(label)(),dtype)for label in ('original','v7_naive')}
        del actual
        for label in functions:
            execute=select(label)
            for _ in range(args.warmup):execute()
        torch.cuda.synchronize();guard.check()
        from fasteq.cuda.jit import runtime
        report['runtime_binary']={'path':runtime().__file__,'sha256':sha(runtime().__file__)}
        observer=ROOT/'performance/fasteq_optimization/v7/driver/gpu_foreign_watch.py'
        directory=args.output.with_suffix('.observer')
        command=[sys.executable,str(observer),'--uuid',args.gpu_uuid,'--owner-pid',str(os.getpid()),
            '--directory',str(directory),'--cpu','171','--interval','.02']
        monitor=subprocess.Popen(command,stdout=args.output.with_suffix('.observer.log').open('w'),stderr=subprocess.STDOUT)
        started=time.monotonic()
        while not(directory/'ready').exists():
            if monitor.poll()is not None or time.monotonic()-started>20:raise RuntimeError('Observer failed readiness')
            time.sleep(.05)
        labels=list(functions);samples={label:{'wall_ms':[],'event_ms':[]}for label in labels};orders=[]
        for number in range(args.rounds):
            shift=number%len(labels);order=labels[shift:]+labels[:shift]
            if number%2:order.reverse()
            orders.append(order)
            for label in order:
                execute=select(label);guard.check()
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin=time.perf_counter();start.record()
                for _ in range(args.iterations):execute()
                end.record();end.synchronize()
                samples[label]['wall_ms'].append((time.perf_counter()-begin)*1000/args.iterations)
                samples[label]['event_ms'].append(start.elapsed_time(end)/args.iterations)
                guard.check()
        report.update(samples=samples,orders=orders,
            medians={label:{key:statistics.median(values)for key,values in pair.items()}for label,pair in samples.items()},
            paired_ratios_to_native={label:{key:distribution([a/b for a,b in zip(samples[label][key],samples['v8_native'][key],strict=True)])
                for key in ('wall_ms','event_ms')}for label in ('original','v7_naive')},
            measurement_kind='normal_eager_no_profiler_or_spies_in_timing')
        assert all(sha(path)==value for path,value in hashes.items()),'Measured component source drift'
        report.update(status='ok',after_guard=guard.check())
    except Exception as error:
        report.update(status='error',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        native.rotary_product=original_rotary
        assert native.rotary_product is original_rotary
        if monitor is not None:
            (directory/'stop').touch();monitor.wait(timeout=15)
            observation=json.loads((directory/'observer.json').read_text());report['observer']=observation
            if observation.get('status')!='complete'or observation.get('foreign_observations'):report['status']='rejected_whole_attempt'
        write_json(args.output,report)
    if report['status']!='ok':raise RuntimeError('Rejected whole attempt; preserve its evidence')
    print(json.dumps({'status':report['status'],'operator':args.operator,'dtype':args.dtype,'medians':report['medians']}),flush=True)


if __name__=='__main__':main()
