"""Affected Modulation / fused GateNorm QA with exclusive GPU and full errors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
import pytest
import torch
from benchmark_scaling_case import ExclusiveGuard
from check_native_norm_v8 import write_json

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq');V8=ROOT/'src/FastEq_v8'
UUID='GPU-e7d1b5bc-5939-b204-406e-a57dd38b7341'


def source_hashes(operator):
    paths=['fasteq/ops/native_equivariant_norm.py','fasteq/ops/equivariant_norm.py']
    if operator=='modulation':paths+=['fasteq/ops/native_modulation.py','fasteq/ops/modulation.py','fasteq/cuda/src/native_modulation.cuh','fasteq/cuda/src/modulation_plan.cpp']
    else:paths+=['fasteq/ops/native_gate_norm.py','fasteq/ops/gate_norm.py','fasteq/cuda/src/native_gate_norm.cuh','fasteq/cuda/src/native_gate.cuh','fasteq/cuda/src/native_equivariant_norm.cuh','fasteq/cuda/src/cuda_utils.hpp']
    return {str(V8/p):hashlib.sha256((V8/p).read_bytes()).hexdigest() for p in paths}


class Audit:
    def __init__(self,report,path,guard):self.report,self.path,self.guard=report,path,guard;self.current=None
    def pytest_runtest_setup(self,item):
        self.guard.check();self.current=item.nodeid
        self.report['cases'].append({'nodeid':item.nodeid,'status':'running','full_errors':[]});write_json(self.path,self.report)
    def pytest_runtest_logreport(self,report):
        case=next((c for c in reversed(self.report['cases']) if c['nodeid']==report.nodeid),None)
        if case is None:return
        if report.failed:case.update(status='failed',phase=report.when,error=str(report.longrepr))
        elif report.when=='call':case['status']='pass' if report.passed else 'skipped'
        write_json(self.path,self.report)
    def compare(self,actual,expected,**kwargs):
        original_assert_close(actual,expected,**kwargs)
        if self.report['cases'] and isinstance(actual,torch.Tensor) and isinstance(expected,torch.Tensor):
            delta=actual.detach().double()-expected.detach().double();norm=expected.detach().double().norm().item()
            self.report['cases'][-1]['full_errors'].append({'elements':actual.numel(),'dtype':str(actual.dtype),
                'atol':kwargs.get('atol'),'rtol':kwargs.get('rtol'),'full_comparison':True,
                'max_abs':delta.abs().max().item() if delta.numel() else 0,'relative_l2':delta.norm().item()/max(norm,1e-30)})


original_assert_close=torch.testing.assert_close


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--operator',choices=['modulation','gate_norm'],required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUID and not torch.cuda.is_initialized()
    assert 'v8_'+args.operator in os.environ['TORCH_EXTENSIONS_DIR']
    guard=ExclusiveGuard(UUID);before=guard.check();os.sched_setaffinity(0,{174});torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    stop=args.output.with_suffix('.observer.stop');stop.unlink(missing_ok=True);obs=args.output.with_suffix('.observer.json')
    observer=subprocess.Popen(['taskset','-c','175',sys.executable,str(ROOT/'performance/fasteq_optimization/v8/native_matrix/observer.py'),str(os.getpid()),str(obs),str(stop)])
    pinned=source_hashes(args.operator)
    report={'status':'running','operator':args.operator,'scope':'affected existing full precision/adjoint/mask/lifetime/Graph QA',
        'source_sha256':pinned,'guard_before':before,'cases':[],'normal_performance_measured':False,'forbidden_triton_execution':True}
    audit=Audit(report,args.output,guard);torch.testing.assert_close=audit.compare
    from triton.runtime.jit import JITFunction
    old_getitem,old_warmup=JITFunction.__getitem__,JITFunction.warmup
    def forbidden(*unused,**kwargs):raise AssertionError('Triton arithmetic/compilation forbidden in native pointwise QA')
    JITFunction.__getitem__=forbidden;JITFunction.warmup=forbidden
    tests=(['tests/adapters/test_native_modulation_cuda.py','tests/adapters/test_modulated_residual_cuda.py','tests/adapters/test_modulation_review_cuda.py',
            'benchmarks/adapters/test_native_pointwise_v8_extra.py','-k','(modulation or modulated_residual) and not ffn_all_parameter'])
    if args.operator=='gate_norm':tests=['tests/adapters/test_fasteq_gate_norm_cuda.py','tests/adapters/test_gate_norm_v5_cuda.py','benchmarks/adapters/test_native_pointwise_v8_extra.py','-k','gate_norm']
    try:
        status=pytest.main(['-q','--tb=short',*tests],plugins=[audit]);torch.cuda.synchronize();assert status==0,'pytest exit '+str(status)
        for path,digest in pinned.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Source drift: '+path
        report['guard_after']=guard.check();report['status']='pass'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:
        torch.testing.assert_close=original_assert_close;JITFunction.__getitem__=old_getitem;JITFunction.warmup=old_warmup
        stop.touch();observer.wait(timeout=10);report['observer']=json.loads(obs.read_text())
        if report['observer']['foreign']:report['status']='foreign_gpu_discarded'
        write_json(args.output,report)
    assert report['status']=='pass';print(json.dumps({'status':report['status'],'cases':len(report['cases'])}))


if __name__=='__main__':main()
