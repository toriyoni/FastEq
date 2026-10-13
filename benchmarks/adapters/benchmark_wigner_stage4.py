"""Guarded original EQX CUDA / FastEq direct / tile Wigner comparison."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import fasteq
from benchmark_scaling_case import ExclusiveGuard,SharedGPUError,execute_operator,flatten_tensors
from scaling_operator_factories import build_case

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND=ROOT/'src/FastEq_v7'
OUT=ROOT/'performance/fasteq_optimization/v7/wigner'
FLAG='FASTEQ_EXPERIMENTAL_WIGNER_STORE_TILE'
UUID=os.environ.get('CUDA_VISIBLE_DEVICES','')
LABELS=('original','legacy','tile')


def configure():
    assert UUID in ('GPU-13222c8f-3c05-b93a-e9d6-1192c18a00ce','GPU-af44a7bc-24c0-45be-df6e-03f4c445c78d')
    assert Path(fasteq.__file__).resolve().is_relative_to(CAND)
    assert Path.cwd().resolve()==CAND
    os.sched_setaffinity(0,{173});torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    for flag in ('FASTEQ_EXPERIMENTAL_LINEAR_SIMT','FASTEQ_EXPERIMENTAL_LINEAR_GEMM',
                 'FASTEQ_EXPERIMENTAL_NATIVE_MODULATION','FASTEQ_EXPERIMENTAL_SEGMENTED_TASK_MASK',
                 'FASTEQ_EXPERIMENTAL_SEGMENTED_REDUCE','FASTEQ_EXPERIMENTAL_SEGMENTED_GEMM',
                 'FASTEQ_EXPERIMENTAL_SEGMENTED_PAIR','FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y',
                 'FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y_WARP'):
        os.environ[flag]='1'
    os.environ['FASTEQ_UNIFORM1D_FWD_TUNE']='0';os.environ['FASTEQ_UNIFORM1D_BWD_TUNE']='0'


def make_case(atoms,dtype):
    torch.manual_seed(19331)
    case=build_case('wigner',atoms,dtype,'cuda')
    case['metadata'].update({'atoms':atoms,'rows':12*atoms,'degree':2,'packed_width':35,
                             'original_execution':'unchanged EQX CUDA, WignerD method auto',
                             'legacy_execution':'FastEq scalar_expression direct stores, tile flag0',
                             'tile_execution':'FastEq same ScalarProgram with shared output tile, flag1'})
    return case


def execute(case,label,mode):
    os.environ[FLAG]='1'if label=='tile'else'0'
    return execute_operator(case,'original'if label=='original'else'v6',mode,torch)


def source_records():
    files=['fasteq/ops/scalar_expression.py','fasteq/ops/rotation.py','fasteq/adapters/__init__.py',
           'fasteq/cuda/jit.py','fasteq/cuda/src/jit_runtime.cpp','fasteq/cuda/src/jit_runtime.h',
           'benchmarks/adapters/scaling_operator_factories.py','benchmarks/adapters/benchmark_scaling_case.py',
           'benchmarks/adapters/benchmark_wigner_stage4.py']
    return {name:hashlib.sha256((CAND/name).read_bytes()).hexdigest()for name in files if (CAND/name).exists()}


def ratio(samples):
    q=statistics.quantiles(samples,n=4,method='inclusive')
    return {'rounds':samples,'median':statistics.median(samples),'q25':q[0],'q75':q[2]}


def compare_values(actual,reference,tolerance):
    actual=flatten_tensors(actual);reference=flatten_tensors(reference);errors={}
    assert actual.keys()==reference.keys()
    for name,a in actual.items():
        b=reference[name]
        if a is None or b is None:
            value=b if a is None else a
            assert value is None or torch.count_nonzero(value)==0
            errors[name]=None
        else:
            torch.testing.assert_close(a,b,**tolerance,msg=name)
            errors[name]={'max_abs':(a-b).abs().max().item()if a.numel()else 0.,'bitwise_equal':torch.equal(a,b)}
    return errors


class RoundGuard:
    def __init__(self):
        self.guard=ExclusiveGuard(UUID);self.rejections=[]
    def reject(self,stage,error,**context):
        record={'stage':stage,'error':str(error),'time_unix':time.time(),**context}
        self.rejections.append(record);print(json.dumps({'rejected':record}),flush=True)
    def wait(self,stage,**context):
        while True:
            try:return self.guard.check()
            except SharedGPUError as error:self.reject(stage,error,**context);time.sleep(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--atoms',type=int,required=True)
    p.add_argument('--dtype',choices=['float32','float64'],required=True)
    p.add_argument('--mode',choices=['forward','backward','force_loss'],required=True)
    p.add_argument('--rounds',type=int,default=9);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure();guard=RoundGuard();before=guard.wait('before_setup');hashes=source_records()
    case=make_case(args.atoms,getattr(torch,args.dtype))
    tolerance=dict(atol=2e-4,rtol=2e-3)if args.dtype=='float32'else dict(atol=2e-8,rtol=2e-7)
    precision_attempt=0
    while True:
        precision_attempt+=1;guard.wait('before_precision',attempt=precision_attempt)
        values={label:execute(case,label,args.mode)for label in LABELS}
        precision={label:compare_values(values[label],values['original'],tolerance)for label in ('legacy','tile')}
        precision['tile_vs_legacy']=compare_values(values['tile'],values['legacy'],dict(atol=0.,rtol=0.))
        del values;gc.collect();torch.cuda.synchronize()
        try:precision_after=guard.guard.check();break
        except SharedGPUError as error:guard.reject('after_precision',error,attempt=precision_attempt)
    for label in LABELS:
        for _ in range(2):execute(case,label,args.mode)
    torch.cuda.synchronize()
    timers={label:(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))for label in LABELS}
    for a,b in timers.values():a.record();b.record()
    torch.cuda.synchronize();wall={label:[]for label in LABELS};events={label:[]for label in LABELS};r=0;attempt=0
    while r<args.rounds:
        attempt+=1;shift=r%3;order=LABELS[shift:]+LABELS[:shift]
        if r%2:order=tuple(reversed(order))
        guard.wait('before_round',accepted_round=r,attempt=attempt,order=order)
        local_wall={};local_events={};stage='warmup'
        try:
            for label in LABELS:execute(case,label,args.mode)
            torch.cuda.synchronize();guard.guard.check()
            for label in order:
                stage='before_'+label;guard.guard.check();a,b=timers[label]
                start=time.perf_counter();a.record();execute(case,label,args.mode);b.record();torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                stage='after_'+label;guard.guard.check();local_wall[label]=elapsed;local_events[label]=a.elapsed_time(b)
        except SharedGPUError as error:
            guard.reject(stage,error,accepted_round=r,attempt=attempt,order=order);continue
        for label in LABELS:wall[label].append(local_wall[label]);events[label].append(local_events[label])
        r+=1
    report={'status':'ok','mode':args.mode,'chain_kind':'feature_higher without physical coordinates'if args.mode=='force_loss'else args.mode,
            'dtype':args.dtype,'metadata':case['metadata'],'precision':precision,'tolerance_vs_original':tolerance,
            'tile_vs_legacy_tolerance':{'atol':0.,'rtol':0.},'wall_ms':wall,'cuda_event_ms':events,
            'wall_medians_ms':{k:statistics.median(v)for k,v in wall.items()},
            'event_medians_ms':{k:statistics.median(v)for k,v in events.items()},
            'paired_wall_ratios':{k+'/tile':ratio([a/b for a,b in zip(wall[k],wall['tile'],strict=True)])for k in ('original','legacy')},
            'paired_event_ratios':{k+'/tile':ratio([a/b for a,b in zip(events[k],events['tile'],strict=True)])for k in ('original','legacy')},
            'source_sha256':hashes,'round_attempts':attempt,'precision_attempts':precision_attempt,'rejections':guard.rejections,
            'precision_after_guard':precision_after,'provenance':{'gpu_uuid':UUID,'before_guard':before,'after_guard':guard.wait('after_measurement'),
            'actual_import':fasteq.__file__,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'tf32':False,
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),
            'torch_extensions_dir':os.environ.get('TORCH_EXTENSIONS_DIR'),'native_cache':os.environ.get('XDG_CACHE_HOME'),
            'flags':{k:v for k,v in os.environ.items()if k.startswith('FASTEQ_')},'route_flags':{'original':0,'legacy':0,'tile':1},
            'timer':'precreated dummy-recorded events reused; full eager wall includes enqueue/synchronization',
            'retry':'reject whole three-backend round on foreign; warm all backends before retry same accepted round order; same process/device/source/inputs/weights'}}
    assert source_records()==hashes,'Source drift'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'file':str(args.output),'wall':report['wall_medians_ms'],'event':report['event_medians_ms'],
                      'paired_wall':report['paired_wall_ratios'],'paired_event':report['paired_event_ratios']}),flush=True)


if __name__=='__main__':main()
