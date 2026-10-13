"""Instrumented role intervals; never interpreted as normal paired latency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from benchmark_scaling_case import ExclusiveGuard
from benchmark_segmented_stage2 import chain
from scaling_operator_factories import build_case
from fasteq.ops import segmented_product as native


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rows',type=int,default=65536);p.add_argument('--dtype',default='float32')
    p.add_argument('--operator',choices=['tp_nonuniform','tp_uvw'],default='tp_nonuniform')
    args=p.parse_args();os.sched_setaffinity(0,{173});torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    for key in ('FASTEQ_EXPERIMENTAL_LINEAR_SIMT','FASTEQ_EXPERIMENTAL_NATIVE_MODULATION',
                'FASTEQ_EXPERIMENTAL_LINEAR_GEMM','FASTEQ_EXPERIMENTAL_SEGMENTED_TASK_MASK',
                'FASTEQ_EXPERIMENTAL_SEGMENTED_REDUCE','FASTEQ_EXPERIMENTAL_SEGMENTED_GEMM',
                'FASTEQ_EXPERIMENTAL_SEGMENTED_PAIR'):
        os.environ[key]='1'
    uuid='GPU-13222c8f-3c05-b93a-e9d6-1192c18a00ce'
    assert os.environ['CUDA_VISIBLE_DEVICES']==uuid
    guard=ExclusiveGuard(uuid);guard.check();torch.manual_seed(7419)
    if args.operator=='tp_uvw': assert args.rows%12==0
    case=build_case(args.operator,args.rows//12 if args.operator=='tp_uvw' else args.rows,getattr(torch,args.dtype),'cuda')
    metadata=case['candidate']._fasteq_plan.metadata
    leaves=[case['candidate'].weight.detach().reshape(1,-1).requires_grad_(),*case['inputs']]
    original=lambda metadata,w,x,y:case['baseline'](x,y,w.reshape(-1))
    report={'measurement_kind':'instrumented Torch profiler and CUDA event role intervals',
            'rows':args.rows,'dtype':args.dtype,'operator':args.operator,
            'source_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            'cpp_sha256':hashlib.sha256((Path(native.__file__).parents[1]/'cuda/src/segmented_product_plan.cpp').read_bytes()).hexdigest()}
    for label,fn in [('candidate',native.segmented_product),('original_torch',original)]:
        for _ in range(2):chain(fn,metadata,leaves,'force_loss')
        torch.cuda.synchronize();guard.check()
        before_allocated=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        roles=[];execute=native._execute;pair=native._FeaturePair.forward
        def record(metadata,role,required,rows,values):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            with torch.profiler.record_function(f'SegmentedRole{role}_required{required}'):
                start.record();out=execute(metadata,role,required,rows,values);end.record()
            roles.append((role,required,start,end));return out
        native._execute=record
        def record_pair(ctx,metadata,required,rows,*values):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            with torch.profiler.record_function(f'SegmentedFeaturePair_required{required}'):
                start.record();out=pair(ctx,metadata,required,rows,*values);end.record()
            roles.append(('feature_pair',required,start,end));return out
        native._FeaturePair.forward=staticmethod(record_pair)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                torch.cuda._sleep(1000)
                chain(fn,metadata,leaves,'force_loss')
                torch.cuda.synchronize()
        finally:
            native._execute=execute
            native._FeaturePair.forward=staticmethod(pair)
        guard.check()
        ops=[];kernels=[]
        for event in prof.key_averages():
            item={'name':event.key,'count':event.count,'self_cpu_us':event.self_cpu_time_total,
                  'self_device_us':event.self_device_time_total,'inclusive_device_us':event.device_time_total}
            if str(event.device_type).endswith('CUDA') and not event.is_user_annotation:
                if 'spin_kernel' not in event.key and 'sleep' not in event.key.lower():kernels.append(item)
            else:ops.append(item)
        report[label]={'roles':[{'role':r,'required':req,'instrumented_cuda_ms':s.elapsed_time(e)} for r,req,s,e in roles],
                       'kernels':sorted(kernels,key=lambda x:x['self_device_us'],reverse=True),
                       'top_cpu':sorted(ops,key=lambda x:x['self_cpu_us'],reverse=True)[:25],
                       'total_kernel_device_us':sum(x['self_device_us'] for x in kernels),
                       'profile_memory':{'before_allocated_bytes':before_allocated,
                                         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                         'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                                         'scope':'instrumented profiler execution; cached factory holds both modules'}}
        print(label,json.dumps(report[label]),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
