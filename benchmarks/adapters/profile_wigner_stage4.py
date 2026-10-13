"""One guarded instrumented Wigner execution per role, not normal latency."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from benchmark_wigner_stage4 import configure,RoundGuard,make_case,execute,source_records,LABELS,UUID
from benchmark_scaling_case import SharedGPUError
from fasteq.ops import scalar_expression as implementation


def main():
    p=argparse.ArgumentParser();p.add_argument('--atoms',type=int,required=True)
    p.add_argument('--dtype',choices=['float32','float64'],required=True)
    p.add_argument('--mode',choices=['forward','force_loss'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    configure();guard=RoundGuard();before=guard.wait('before_setup');hashes=source_records();case=make_case(args.atoms,getattr(torch,args.dtype))
    report={'kind':'one CPU+CUDA profiler execution per backend; instrumentation, not normal latency','mode':args.mode,
            'dtype':args.dtype,'metadata':case['metadata'],'source_sha256':hashes,'backends':{},'gpu_uuid':UUID,
            'before_guard':before,'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for label in LABELS:
        attempt=0
        while True:
            attempt+=1;guard.wait('before_profile',label=label,attempt=attempt)
            for _ in range(2):execute(case,label,args.mode)
            torch.cuda.synchronize()
            try:guard.guard.check()
            except SharedGPUError as error:guard.reject('after_warm',error,label=label,attempt=attempt);continue
            intervals=[];configs={};native_source=implementation._source;native_kernels=implementation.kernels;native_grad=torch.autograd.grad
            def source(metadata,dtype,strategy='direct',block=128):
                code=native_source(metadata,dtype,strategy,block)
                configs[code]={'output_groups':tuple(map(len,implementation._parse(metadata)[1])),
                               'strategy':strategy,'block':block,'source_sha256':hashlib.sha256(code.encode()).hexdigest()}
                return code
            class AnnotatedKernel:
                def __init__(self,kernel,code):self.kernel,self.code=kernel,code
                def launch(self,args,gridx,gridy,block,stream):
                    config=configs[self.code];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    with torch.profiler.record_function('ScalarStore_'+config['strategy']+'_groups'+str(config['output_groups'])):
                        a.record();result=self.kernel.launch(args,gridx,gridy,block,stream);b.record()
                    intervals.append({'kind':'scalar_launch',**config,'rows':args[-1],'grid_x':gridx,'start':a,'end':b})
                    return result
            def kernels(sources,device,entry='run'):
                result=native_kernels(sources,device,entry)
                return {code:AnnotatedKernel(kernel,code)for code,kernel in result.items()}
            def grad(*a,**kwargs):
                phase='first'if kwargs.get('create_graph')else'higher';s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                with torch.profiler.record_function('WignerPhase_'+phase):
                    s.record();result=native_grad(*a,**kwargs);e.record()
                intervals.append({'kind':'phase','phase':phase,'start':s,'end':e});return result
            allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            implementation._source=source;implementation.kernels=kernels;torch.autograd.grad=grad
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
                    torch.cuda._sleep(1000);execute(case,label,args.mode);torch.cuda.synchronize()
            finally:implementation._source=native_source;implementation.kernels=native_kernels;torch.autograd.grad=native_grad
            try:guard.guard.check()
            except SharedGPUError as error:guard.reject('after_profile',error,label=label,attempt=attempt);continue
            break
        gpu=[];cpu=[]
        for event in prof.key_averages():
            row={'name':event.key,'count':event.count,'self_cpu_us':event.self_cpu_time_total,
                 'self_device_us':event.self_device_time_total,'inclusive_device_us':event.device_time_total}
            if str(event.device_type).endswith('CUDA')and not event.is_user_annotation:
                if 'spin_kernel'not in event.key and 'sleep'not in event.key.lower():gpu.append(row)
            else:cpu.append(row)
        measured=[{k:v for k,v in item.items()if k not in ('start','end')}|
                  {'instrumented_cuda_ms':item['start'].elapsed_time(item['end'])}for item in intervals]
        report['backends'][label]={'attempts':attempt,'intervals':measured,
                                  'kernels':sorted(gpu,key=lambda r:r['self_device_us'],reverse=True),
                                  'summed_kernel_device_us':sum(r['self_device_us']for r in gpu),
                                  'top_cpu':sorted(cpu,key=lambda r:r['self_cpu_us'],reverse=True)[:25],
                                  'memory':{'before_allocated_bytes':allocated,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'scope':'profile process holds original and adapted factory modules'}}
    report['after_guard']=guard.wait('after_measurement');report['rejections']=guard.rejections
    assert source_records()==hashes
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
