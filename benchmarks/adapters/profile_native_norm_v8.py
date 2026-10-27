"""Isolated native Norm high-order profile; diagnostic only, no normal timing."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
import torch
from e3nn import o3
from fasteq.ops import native_equivariant_norm as native
from fasteq.ops.equivariant_norm import NormPlan
from run_native_norm_v8 import ROOT,CANDIDATE,UUID,write_json,hashes,chain,close,reference,feature


@contextmanager
def source_probe(directory):
    previous=native.source_for;records=[];state={'phase':'forward'};directory.mkdir(parents=True,exist_ok=True)
    def source_for(role,kwargs,dtype):
        source=previous(role,kwargs,dtype);digest=hashlib.sha256(source.encode()).hexdigest();path=directory/(digest+'.cu')
        if not path.exists():path.write_text(source)
        records.append({'phase':state['phase'],'owner_path':str(CANDIDATE/'fasteq/ops/native_equivariant_norm.py'),
            'owner_function':'_source','emitted_source_path':str(path),'emitted_source_sha256':digest,
            'entry':native._ENTRIES[role],'role':role,'kwargs':kwargs,'kernel_count_claim':False})
        return source
    native.source_for=source_for
    try:yield records,state
    finally:native.source_for=previous;assert native.source_for is previous


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUID;assert not torch.cuda.is_initialized()
    from benchmark_scaling_case import ExclusiveGuard
    guard=ExclusiveGuard(UUID);guard.check();os.sched_setaffinity(0,{174});torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    from prophet.layer_norm import RMSLayerNorm
    from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm
    args.output.parent.mkdir(parents=True,exist_ok=True);stop=args.output.with_suffix('.observer.stop');stop.unlink(missing_ok=True);obs=args.output.with_suffix('.observer.json')
    observer=subprocess.Popen(['taskset','-c','175',sys.executable,str(ROOT/'performance/fasteq_optimization/v8/native_matrix/observer.py'),str(os.getpid()),str(obs),str(stop)])
    source_hashes=hashes();source_hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report={'status':'running','profiles':[],'normal_performance_measured':False,'source_sha256':source_hashes}
    try:
        for kind,dtype in [('tace',torch.float64),('prophet',torch.float32)]:
            module=(EquivariantMergeLayerNorm([0,1,2],64,centering=True) if kind=='tace' else RMSLayerNorm(o3.Irreps('64x0e+64x1o+64x2e'),centering=True,eps=1e-5)).to(device='cuda',dtype=dtype)
            plan=NormPlan(module,tace=kind=='tace');parameters=tuple(module.parameters());operation=lambda x:plan(module,x)
            precision_x=feature(33,plan,dtype);precision_seed=torch.randn_like(precision_x).requires_grad_()
            precision=close(chain(operation,precision_x,parameters,precision_seed),
                            chain(lambda x:reference(module,x,plan),precision_x,parameters,precision_seed),dtype)
            x=feature(4096,plan,dtype);seed=torch.randn_like(x).requires_grad_()
            for _ in range(3):chain(operation,x,parameters,seed)
            torch.cuda.synchronize();guard.check();stem=args.output.parent/('profile_'+kind)
            with source_probe(args.output.parent/('profile_'+kind+'_sources')) as (dispatch,state):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    chain(operation,x,parameters,seed,state=state);torch.cuda.synchronize()
            guard.check();trace=stem.with_suffix('.trace.json');profiler.export_chrome_trace(str(trace))
            manifest={'candidate_root':str(CANDIDATE),'isolated_operator':kind+'_norm','scope':'operator_and_adjoints',
                'phases':['forward','first','higher'],'source_files':[{'path':p,'sha256':digest} for p,digest in source_hashes.items()],
                'dispatch':dispatch,'hooks_restored':True,'timing_claim':False}
            bindings=stem.with_suffix('.bindings.json');write_json(bindings,manifest)
            checker=ROOT/'performance/fasteq_optimization/v8/check_native_evidence.py';spec=importlib.util.spec_from_file_location('norm_native_evidence',checker);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            evidence=mod.inspect_evidence(json.loads(trace.read_text()),manifest);write_json(stem.with_suffix('.evidence.json'),evidence)
            assert evidence['status']=='pass',evidence
            report['profiles'].append({'kind':kind,'dtype':str(dtype),'rows':4096,'D':plan.dim,'status':'pass','precision_full_N33':precision,
                'trace':str(trace),'bindings':str(bindings),'evidence':str(stem.with_suffix('.evidence.json'))});write_json(args.output,report)
        for p,expected in source_hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==expected,'Source drift: '+p
        report['status']='pass'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:
        stop.touch();observer.wait(timeout=10);report['observer']=json.loads(obs.read_text())
        if report['observer']['foreign']:report['status']='foreign_gpu_discarded'
        write_json(args.output,report)
    assert report['status']=='pass';print(json.dumps({'status':report['status'],'profiles':len(report['profiles'])}))


if __name__=='__main__':main()
