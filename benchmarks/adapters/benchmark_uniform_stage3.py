"""Exclusive scalarY Uniform comparison: original / frozen route / opt-in route.

The O3 first-layer case has external per-edge weights, so its feature_higher
objective differentiates all three input leaves before the higher derivative.
The O2 u1u case has an internal shared parameter; first inputs are X and Y.
"""
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
from fasteq.adapters import adapt_model
from benchmark_scaling_case import ExclusiveGuard,SharedGPUError

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND=ROOT/'src/FastEq_v7'
UUID=os.environ.get('CUDA_VISIBLE_DEVICES','GPU-13222c8f-3c05-b93a-e9d6-1192c18a00ce')
ALLOWED_UUIDS={'GPU-13222c8f-3c05-b93a-e9d6-1192c18a00ce',
               'GPU-3a6d868b-775e-3236-988f-2772678876f2'}
FLAG='FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y'
LABELS=('original','legacy','new')


def configure():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    assert UUID in ALLOWED_UUIDS
    assert Path(fasteq.__file__).resolve().is_relative_to(CAND)
    os.sched_setaffinity(0,{173});torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    for flag in ('FASTEQ_EXPERIMENTAL_LINEAR_SIMT','FASTEQ_EXPERIMENTAL_LINEAR_GEMM',
                 'FASTEQ_EXPERIMENTAL_NATIVE_MODULATION','FASTEQ_EXPERIMENTAL_SEGMENTED_TASK_MASK',
                 'FASTEQ_EXPERIMENTAL_SEGMENTED_REDUCE','FASTEQ_EXPERIMENTAL_SEGMENTED_GEMM',
                 'FASTEQ_EXPERIMENTAL_SEGMENTED_PAIR'):
        os.environ[flag]='1'
    os.environ['FASTEQ_UNIFORM1D_FWD_TUNE']='0'
    os.environ['FASTEQ_UNIFORM1D_BWD_TUNE']='0'
    os.environ.setdefault('FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y_WARP','1')


def build_case(operator,rows,dtype,device='cuda'):
    torch.manual_seed(19331)
    if operator=='o2_u1u':
        from eqx.o2 import TensorProduct
        original=TensorProduct('64x1me','1x0ee','64x1me',[(0,0,0,'u1u',True)])
        baseline_source='eqx.o2.TensorProduct.forward; original Torch expressions'
        expected_terms=2
    elif operator=='o3_first':
        from e3nn import o3
        original=o3.TensorProduct('64x0e','1x0e+1x1o+1x2e','64x0e+64x1o+64x2e',
                                  [(0,j,j,'uvu',True)for j in range(3)],
                                  internal_weights=False,shared_weights=False)
        baseline_source='e3nn.o3.TensorProduct; actual TACE/TECE/Prophet first-layer shapes'
        expected_terms=9
    else:raise ValueError(operator)
    original=original.to(device='cpu',dtype=dtype)
    adapted=adapt_model(original)
    assert original.state_dict().keys()==adapted.state_dict().keys()
    for name,value in original.state_dict().items():assert torch.equal(value,adapted.state_dict()[name]),name
    assert adapted._fasteq_plan.scalar_y_small
    from fasteq.ops.segmented_product import _parse
    dimensions,terms,coefficients=_parse(adapted._fasteq_plan.segmented_metadata)
    assert len(terms)==expected_terms
    original.to(device=device);adapted.to(device=device)
    shapes=[(rows,original.irreps_in1.dim),(rows,original.irreps_in2.dim)]
    if operator=='o3_first':shapes.append((rows,original.weight_numel))
    inputs=tuple(torch.randn(shape,device=device,dtype=dtype).mul_(.2).requires_grad_()for shape in shapes)
    return {'original':original,'adapted':adapted,'inputs':inputs,
            'metadata':{'operator':operator,'rows':rows,'edge_factor':12,'atoms_equivalent':rows/12,
                        'channels':64,'input_shapes':shapes,'output_dim':original.irreps_out.dim,
                        'weight_numel':original.weight_numel,'shared_weights':original.shared_weights,
                        'internal_weights':original.internal_weights,'parameter_initial_state_equal':True,
                        'original_source':baseline_source,'legacy_source':'FastEq Uniform1D, flag=0',
                        'candidate_source':'FastEq segmented scalarY, original layout, flag=1',
                        'dimensions':dimensions,'terms':terms,'coefficients':coefficients}}


def execute(case,label,mode='force_loss'):
    os.environ[FLAG]='1'if label=='new'else'0'
    module=case['original']if label=='original'else case['adapted']
    inputs=case['inputs'];parameters=tuple(module.parameters());leaves=inputs+parameters
    with torch.set_grad_enabled(mode!='forward'):
        z=module(*inputs)
        if mode=='forward':return(z,)
        objective=z.sin().mean()+.1*z.square().mean()
        if mode=='backward':return(z,*torch.autograd.grad(objective,leaves,allow_unused=True))
        first=torch.autograd.grad(objective,inputs,create_graph=True,allow_unused=True)
        loss=sum(g.square().mean()for g in first if g is not None and g.requires_grad)
        higher=torch.autograd.grad(loss,leaves,allow_unused=True)
        return(z,*first,*higher)


def ratio_summary(samples):
    quartiles=statistics.quantiles(samples,n=4,method='inclusive')
    return {'rounds':samples,'median':statistics.median(samples),'q25':quartiles[0],'q75':quartiles[2]}


def source_records():
    files=['fasteq/adapters/tensor_product.py','fasteq/ops/segmented_product.py',
           'fasteq/ops/uniform1d_jit.py','fasteq/cuda/src/segmented_product_plan.cpp',
           'fasteq/cuda/src/indexed_linear_plan.cpp','fasteq/cuda/src/jit_runtime.h',
           'fasteq/cuda/src/jit_runtime.cpp','fasteq/cuda/jit.py',
           'benchmarks/adapters/benchmark_uniform_stage3.py']
    return {name:hashlib.sha256((CAND/name).read_bytes()).hexdigest()for name in files}


def main():
    p=argparse.ArgumentParser();p.add_argument('--operator',choices=['o2_u1u','o3_first'],required=True)
    p.add_argument('--rows',type=int,required=True);p.add_argument('--dtype',choices=['float32','float64'],required=True)
    p.add_argument('--mode',choices=['forward','backward','force_loss'],default='force_loss')
    p.add_argument('--rounds',type=int,default=9);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure();guard=ExclusiveGuard(UUID);rejections=[]
    def reject(stage,error,**context):
        record={'stage':stage,'error':str(error),'time_unix':time.time(),**context}
        rejections.append(record);print(json.dumps({'rejected':record}),flush=True)
    def wait_exclusive(stage,**context):
        while True:
            try:return guard.check()
            except SharedGPUError as error:
                reject(stage,error,**context);time.sleep(1)
    before=wait_exclusive('before_setup')
    hashes=source_records();case=build_case(args.operator,args.rows,getattr(torch,args.dtype))
    tolerance=dict(atol=3e-5,rtol=5e-5)if args.dtype=='float32'else dict(atol=2e-10,rtol=2e-9)
    precision_attempt=0
    while True:
        precision_attempt+=1;wait_exclusive('before_precision',attempt=precision_attempt)
        values={label:execute(case,label,args.mode)for label in LABELS}
        precision={}
        for label in ('legacy','new'):
            errors=[]
            for a,b in zip(values[label],values['original'],strict=True):
                if a is None or b is None:
                    assert a is b or torch.count_nonzero(b if a is None else a)==0
                    errors.append(None)
                else:
                    torch.testing.assert_close(a,b,**tolerance);errors.append((a-b).abs().max().item())
            precision[label]={'status':'pass','full_tensor_max_abs_errors':errors,'tolerance':tolerance}
        del values;gc.collect();torch.cuda.synchronize()
        try:precision_after=guard.check();break
        except SharedGPUError as error:reject('after_precision',error,attempt=precision_attempt)
    for label in LABELS:
        for _ in range(2):execute(case,label,args.mode)
    torch.cuda.synchronize()
    timers={label:(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))for label in LABELS}
    for start_event,end_event in timers.values():
        start_event.record();end_event.record()
    torch.cuda.synchronize()
    wall={label:[]for label in LABELS};events={label:[]for label in LABELS}
    r=0;attempt=0
    while r<args.rounds:
        attempt+=1
        shift=r%3;order=LABELS[shift:]+LABELS[:shift]
        if r%2:order=tuple(reversed(order))
        wait_exclusive('before_round',accepted_round=r,attempt=attempt,order=order)
        local_wall={};local_events={};stage='warmup'
        try:
            for label in LABELS:execute(case,label,args.mode)
            torch.cuda.synchronize();guard.check()
            for label in order:
                stage='before_'+label;guard.check();start_event,end_event=timers[label]
                start=time.perf_counter();start_event.record();execute(case,label,args.mode);end_event.record()
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                stage='after_'+label;guard.check()
                local_wall[label]=elapsed;local_events[label]=start_event.elapsed_time(end_event)
        except SharedGPUError as error:
            reject(stage,error,accepted_round=r,attempt=attempt,order=order)
            continue
        for label in LABELS:
            wall[label].append(local_wall[label]);events[label].append(local_events[label])
        r+=1
    report={'status':'ok','measurement_kind':'normal exclusive interleaved three-backend eager timing',
            'mode':args.mode,'chain_kind':'feature_higher; first includes all external input leaves; no physical coordinates',
            'timer_protocol':'reused precreated and dummy-recorded CUDA events; wall includes enqueue and synchronization',
            'retry_protocol':'one process, immutable inputs/weights/source/device; reject complete three-backend round on any foreign guard; retry accepted-index order after untimed warmup of all backends',
            'rejected_attempts':rejections,'total_round_attempts':attempt,'precision_attempts':precision_attempt,
            'precision_after_guard':precision_after,
            'dtype':args.dtype,'metadata':case['metadata'],'wall_ms':wall,'cuda_event_ms':events,
            'wall_medians_ms':{k:statistics.median(v)for k,v in wall.items()},
            'event_medians_ms':{k:statistics.median(v)for k,v in events.items()},'precision':precision,
            'paired_round_ratios':{f'{k}/new':ratio_summary([a/b for a,b in zip(wall[k],wall['new'],strict=True)])for k in ('original','legacy')},
            'source_sha256':hashes,'stage2_snapshot_manifest':str(ROOT/'performance/fasteq_optimization/v7/stages/tp/manifest.json'),
            'stage2_manifest_sha256':hashlib.sha256((ROOT/'performance/fasteq_optimization/v7/stages/tp/manifest.json').read_bytes()).hexdigest(),
            'provenance':{'actual_import':fasteq.__file__,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
                          'cpu_affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),
                          'torch_extensions_dir':os.environ.get('TORCH_EXTENSIONS_DIR'),'native_cache':os.environ.get('XDG_CACHE_HOME'),
                          'before_guard':before,'after_guard':wait_exclusive('after_measurement'),'tf32':False,
                          'common_flags':{k:v for k,v in os.environ.items()if k.startswith('FASTEQ_')},
                          'route_flags':{'original':0,'legacy':0,'new':1}}}
    assert source_records()==hashes,'Source drift during benchmark'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'file':str(args.output),'median_wall_ms':report['wall_medians_ms'],
                      'paired_ratios':report['paired_round_ratios']}),flush=True)


if __name__=='__main__':main()
