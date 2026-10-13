"""Targeted scalarY role/phase profiling, separate from normal timings."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from benchmark_scaling_case import ExclusiveGuard
from benchmark_uniform_stage3 import build_case,configure,execute,source_records,UUID,LABELS
from fasteq.ops import segmented_product as implementation


def main():
    p=argparse.ArgumentParser();p.add_argument('--operator',choices=['o2_u1u','o3_first'],required=True)
    p.add_argument('--rows',type=int,required=True);p.add_argument('--dtype',choices=['float32','float64'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure();guard=ExclusiveGuard(UUID);before_guard=guard.check()
    hashes=source_records();case=build_case(args.operator,args.rows,getattr(torch,args.dtype))
    report={'measurement_kind':'instrumented single Torch CPU/CUDA profiler execution per backend; not normal latency',
            'metadata':case['metadata'],'dtype':args.dtype,'mode':'feature_higher','source_sha256':hashes,'backends':{},
            'profile_driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'before_guard':before_guard,'gpu_uuid':UUID}
    for label in LABELS:
        for _ in range(2):execute(case,label)
        torch.cuda.synchronize();guard.check();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        intervals=[];native_execute=implementation._execute;original_grad=torch.autograd.grad
        def annotated_execute(metadata,role,required,rows,values):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            with torch.profiler.record_function(f'ScalarYRole{role}_required{required}'):
                a.record();result=native_execute(metadata,role,required,rows,values);b.record()
            intervals.append({'kind':'role','role':role,'required':required,'start':a,'end':b})
            return result
        def annotated_grad(*a,**kwargs):
            phase='first'if kwargs.get('create_graph')else'higher'
            s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
            with torch.profiler.record_function('ScalarYPhase_'+phase):
                s.record();result=original_grad(*a,**kwargs);e.record()
            intervals.append({'kind':'phase','phase':phase,'start':s,'end':e});return result
        implementation._execute=annotated_execute;torch.autograd.grad=annotated_grad
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
                torch.cuda._sleep(1000);execute(case,label);torch.cuda.synchronize()
        finally:
            implementation._execute=native_execute;torch.autograd.grad=original_grad
        guard.check();kernels=[];cpu=[]
        for event in prof.key_averages():
            row={'name':event.key,'count':event.count,'self_cpu_us':event.self_cpu_time_total,
                 'self_device_us':event.self_device_time_total,'inclusive_device_us':event.device_time_total}
            if str(event.device_type).endswith('CUDA')and not event.is_user_annotation:
                if 'spin_kernel'not in event.key and 'sleep'not in event.key.lower():kernels.append(row)
            else:cpu.append(row)
        measured=[]
        for item in intervals:
            measured.append({k:v for k,v in item.items()if k not in ('start','end')}|
                            {'instrumented_cuda_ms':item['start'].elapsed_time(item['end'])})
        report['backends'][label]={'intervals':measured,'kernels':sorted(kernels,key=lambda r:r['self_device_us'],reverse=True),
                                   'top_cpu':sorted(cpu,key=lambda r:r['self_cpu_us'],reverse=True)[:25],
                                   'summed_kernel_device_us':sum(r['self_device_us']for r in kernels),
                                   'profile_memory':{'before_allocated_bytes':before,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                                     'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'scope':'instrumented process holds original and adapted modules'}}
    report['after_guard']=guard.check()
    assert source_records()==hashes,'Source drift during profile'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
