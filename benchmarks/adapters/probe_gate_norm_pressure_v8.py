"""Retain actual failed F32 pressure fixture and compare original TT arithmetic."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
import torch
from benchmark_scaling_case import ExclusiveGuard
from check_native_norm_v8 import write_json
from run_native_pointwise_v8 import ROOT,V8,UUID
import test_native_pointwise_v8_extra as extra


def metrics(actual,expected):
    values=[]
    for slot,(a,b) in enumerate(zip(actual,expected,strict=True)):
        if a is None or b is None:values.append({'slot':slot,'none_or_zero':True});continue
        delta=a.detach().double()-b.detach().double();norm=b.detach().double().norm().item()
        accepted=torch.isclose(a,b,atol=8e-5,rtol=2e-4)
        worst=delta.abs().reshape(-1).argmax().item() if a.numel() else None
        values.append({'slot':slot,'elements':a.numel(),'max_abs':delta.abs().max().item() if a.numel() else 0,
            'relative_l2':delta.norm().item()/max(norm,1e-30),'mismatched_elements':int((~accepted).sum()),
            'greatest_abs_flat_index':worst,'atol':8e-5,'rtol':2e-4})
    return values


def chain(operation,p,s,seed,parameters):
    output=operation(p,s);first=torch.autograd.grad(output,(p,s,*parameters),seed,create_graph=True)
    higher=torch.autograd.grad(sum(g.square().sum() for g in first),(p,s,seed,*parameters),allow_unused=True)
    return(output,*first,*higher)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUID and not torch.cuda.is_initialized();guard=ExclusiveGuard(UUID);guard.check();os.sched_setaffinity(0,{174});torch.set_num_threads(1)
    out=ROOT/'performance/fasteq_optimization/v8/gate_norm';path=out/'pressure_replay.json';stop=path.with_suffix('.observer.stop');stop.unlink(missing_ok=True);obs=path.with_suffix('.observer.json')
    observer=subprocess.Popen(['taskset','-c','175',sys.executable,str(ROOT/'performance/fasteq_optimization/v8/native_matrix/observer.py'),str(os.getpid()),str(obs),str(stop)])
    report={'status':'running','scope':'Diagnostic pressure replay, original Triton is reference only; no timing claim',
        'original_failed_case':'test_gate_norm_ragged_mixed_roles[257-dtype0]','raw_failure':str(out/'qa_initial.json'),
        'replayed_predecessors':[],'comparisons':[]}
    capture={}
    try:
        sys.path.insert(0,str(V8/'tests/adapters'))
        from test_gate_norm_v5_cuda import test_direct_offset_stream_and_full_double_cuda_graph,test_short_and_fp32_keep_original_fusion
        # The final predecessor explicitly reset the CUDA generator to 839.
        # Replay its exact random draws and subsequent four ragged fixtures.
        test_direct_offset_stream_and_full_double_cuda_graph();report['replayed_predecessors'].append('final existing wide Graph fixture, manual_seed839')
        for dtype,channels,nodes in ((torch.float32,64,64),(torch.float64,3,64),(torch.float64,64,6)):
            test_short_and_fp32_keep_original_fusion(dtype,channels,nodes)
            report['replayed_predecessors'].append({'fixture':'short_and_fp32_keep_original_fusion','dtype':str(dtype),'channels':channels,'nodes':nodes})
        for rows in (0,17):
            for dtype in (torch.float32,torch.float64):
                extra.test_gate_norm_ragged_mixed_roles(dtype,rows);report['replayed_predecessors'].append({'rows':rows,'dtype':str(dtype)})
        report['rng_before_failed_case_sha256']=hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
        def capture_inputs(norm,plan,p,s,seed):
            capture.update(norm=norm,plan=plan,p=p.detach().clone(),s=s.detach().clone(),seed=seed.detach().clone())
            torch.save({'projected':p.detach().cpu(),'skip':s.detach().cpu(),'seed':seed.detach().cpu(),
                'norm_state':{k:v.detach().cpu() for k,v in norm.state_dict().items()},
                'metadata':[v.cpu() for v in plan._metadata],'cuda_rng_before':torch.cuda.get_rng_state().cpu()},out/'pressure_inputs.pt')
        original_compare=extra.compare
        def retain_compare(actual,expected,dtype):
            report['comparisons'].append({'route':'native_vs_torch_replayed','metrics':metrics(actual,expected)})
            torch.save({'native':[v.detach().cpu() if v is not None else None for v in actual],
                'torch':[v.detach().cpu() if v is not None else None for v in expected]},out/'pressure_native_torch_outputs.pt')
        extra._CAPTURE_PRESSURE_INPUTS=capture_inputs;extra.compare=retain_compare
        try:extra.test_gate_norm_ragged_mixed_roles(torch.float32,257)
        finally:extra.compare=original_compare;del extra._CAPTURE_PRESSURE_INPUTS
        failed=next(x for x in report['comparisons'][0]['metrics'] if x['slot']==10)
        report['old_failure_reproduced']=failed['mismatched_elements']==1 and failed['max_abs']>=.0003662109375
        # Load frozen V7's actual GateNorm arithmetic under a reference-only
        # alias. Its native row reduction hook has the original ordered tree.
        oldpath=ROOT/'src/FastEq_v7/fasteq/ops/gate_norm.py';spec=importlib.util.spec_from_file_location('fasteq.ops._pressure_gate_norm_v7',oldpath)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        norm,plan=capture['norm'],capture['plan'];oldplan=module.GateNormPlan(plan.gate,norm)
        report['original_TT_plan_source_sha256']=hashlib.sha256(oldpath.read_bytes()).hexdigest()
        oldmath=ROOT/'src/FastEq_v7/fasteq/triton/gate_norm.py';live=V8/'fasteq/triton/gate_norm.py';assert oldmath.read_bytes()==live.read_bytes()
        report['original_TT_arithmetic_source_sha256']=hashlib.sha256(oldmath.read_bytes()).hexdigest()
        values=[]
        for route in ('native','torch','original_fasteq_triton'):
            p,s,seed=[capture[n].detach().clone().requires_grad_() for n in ('p','s','seed')]
            operation=(lambda p,s:plan(norm,p,s)) if route=='native' else (lambda p,s:oldplan(norm,p,s)) if route=='original_fasteq_triton' else (lambda p,s:extra.norm_reference(norm,extra.gate_reference(plan,p+s),plan.norm))
            result=chain(operation,p,s,seed,tuple(norm.parameters()));values.append((route,result))
        reference=values[1][1]
        for route,result in values:
            report['comparisons'].append({'route':route+'_vs_torch_saved_inputs','metrics':metrics(result,reference)})
            torch.save([v.detach().cpu() if v is not None else None for v in result],out/('pressure_'+route+'_outputs.pt'))
        report['native_vs_original_TT']=metrics(values[0][1],values[2][1]);torch.cuda.synchronize();guard.check()
        report['status']='diagnostic_complete'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:
        stop.touch();observer.wait(timeout=10);report['observer']=json.loads(obs.read_text())
        if report['observer']['foreign']:report['status']='foreign_gpu_discarded'
        write_json(path,report)
    print(json.dumps({'status':report['status'],'old_failure_reproduced':report['old_failure_reproduced'],'inputs':str(out/'pressure_inputs.pt')}))


if __name__=='__main__':main()
