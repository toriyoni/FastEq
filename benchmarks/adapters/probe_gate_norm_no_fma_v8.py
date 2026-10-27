"""OUT-only FMA policy trial on retained pressure inputs; no production edits."""
from contextlib import contextmanager
import hashlib
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
from probe_gate_norm_pressure_v8 import chain,metrics
import test_native_pointwise_v8_extra as extra


@contextmanager
def no_fma_kernels(records):
    from fasteq.cuda import jit
    previous=jit.kernels;resident={}
    def kernels(sources,device,entry='run'):
        if not entry.startswith('fasteq_gate_norm_'):return previous(sources,device,entry)
        compiler=jit.runtime();major,minor=torch.cuda.get_device_capability(device)
        options=('--std=c++17',f'--gpu-architecture=sm_{major}{minor}','--fmad=false')
        result={}
        for source in sources:
            key=(source,entry)
            if key not in resident:
                binary=jit._binary(source,options);resident[key]=compiler.Kernel(binary,entry)
                records.append({'entry':entry,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                    'cubin_sha256':hashlib.sha256(binary).hexdigest(),'options':options,'registers':resident[key].registers,
                    'local_bytes':resident[key].local_bytes})
            result[source]=resident[key]
        return result
    jit.kernels=kernels
    try:yield
    finally:jit.kernels=previous


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUID and not torch.cuda.is_initialized();guard=ExclusiveGuard(UUID);guard.check();os.sched_setaffinity(0,{174});torch.set_num_threads(1)
    out=ROOT/'performance/fasteq_optimization/v8/gate_norm';path=out/'pressure_no_fma_trial.json';stop=path.with_suffix('.observer.stop');stop.unlink(missing_ok=True);obs=path.with_suffix('.observer.json')
    observer=subprocess.Popen(['taskset','-c','175',sys.executable,str(ROOT/'performance/fasteq_optimization/v8/native_matrix/observer.py'),str(os.getpid()),str(obs),str(stop)])
    report={'status':'running','scope':'Diagnostic original unchanged source compiled --fmad=false only for GateNorm; no timing/production change claim',
        'inputs_sha256':hashlib.sha256((out/'pressure_inputs.pt').read_bytes()).hexdigest(),'comparisons':[],'compile_records':[]}
    try:
        payload=torch.load(out/'pressure_inputs.pt',weights_only=False);saved=torch.load(out/'pressure_torch_outputs.pt',weights_only=False)
        expected=[x.cuda() if x is not None else None for x in saved]
        for flag in ('default_fma','no_fma'):
            norm,plan=extra.fused_fixture(torch.float32);norm.load_state_dict(payload['norm_state'])
            p,s,seed=[payload[n].cuda().requires_grad_() for n in ('projected','skip','seed')]
            if flag=='no_fma':
                with no_fma_kernels(report['compile_records']):result=chain(lambda p,s:plan(norm,p,s),p,s,seed,tuple(norm.parameters()))
            else:result=chain(lambda p,s:plan(norm,p,s),p,s,seed,tuple(norm.parameters()))
            report['comparisons'].append({'route':flag,'metrics':metrics(result,expected)})
            torch.save([x.detach().cpu() if x is not None else None for x in result],out/('pressure_'+flag+'_outputs.pt'))
        torch.cuda.synchronize();guard.check();report['status']='diagnostic_complete'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:
        stop.touch();observer.wait(timeout=10);report['observer']=json.loads(obs.read_text())
        if report['observer']['foreign']:report['status']='foreign_gpu_discarded'
        write_json(path,report)
    print(json.dumps({'status':report['status'],'DDY':[c['metrics'][10] for c in report['comparisons']]}))


if __name__=='__main__':main()
