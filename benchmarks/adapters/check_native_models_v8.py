"""One affected full-parameter physical case for each native model/dtype."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import fasteq
from scaling_model_case import build_case
from paired_model_common import reference, validate
from native_sh_dispatch_spy_v8 import native_only

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND=ROOT/'src/FastEq_v8'
sys.path.insert(0,str(ROOT/'performance/fasteq_optimization/v7/driver'))
from pair_common import ExclusiveGuard, source_hashes, atomic_json


def model_source_hashes():
    # Freeze all production implementations and the actual model/QA imports.
    # Unrelated benchmark drivers may be prepared concurrently without changing
    # any arithmetic or execution path of these physical checks.
    hashes={str(CAND/name):digest for name,digest in source_hashes(CAND).items()
            if name.startswith('fasteq/')}
    drivers=('check_native_models_v8.py','scaling_model_case.py','scaling_geometry.py',
             'benchmark_models.py','paired_model_common.py','native_sh_dispatch_spy_v8.py')
    paths=[CAND/'benchmarks/adapters'/name for name in drivers]
    paths.append(CAND/'tests/adapters/eqx_cuda_spy.py')
    for directory in (ROOT/'src/tace/tace',ROOT/'src/tace/eqx',ROOT/'src/prophet/prophet'):
        paths.extend(p for p in directory.rglob('*.py') if '__pycache__' not in p.parts)
    hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return hashes


def main():
    uuid=os.environ['CUDA_VISIBLE_DEVICES']
    assert Path(fasteq.__file__).resolve().is_relative_to(CAND)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    guard=ExclusiveGuard(uuid);guard.check()
    hashes=model_source_hashes()
    output=ROOT/'performance/fasteq_optimization/v8/models/physical_QA.json'
    report={'status':'running','source_root':str(CAND),'actual_import':fasteq.__file__,
            'sources':hashes,'records':[],'flags':{k:v for k,v in os.environ.items() if k.startswith('FASTEQ_')},
            'scope':'Affected native integration, N129 C64 two-layer physical energy+forces loss; full outputs and all named parameter gradients. No OOM or full-suite repeat.'}
    started=time.monotonic()
    try:
        for name in ('tace','tece','prophet'):
            for dtype in (torch.float32,torch.float64):
                case=build_case(name,129,dtype,'cuda','force_loss')
                expected=reference(case.models['original'],name,case.data,'force_loss')
                errors=validate(case.models['v6'],name,case.data,'force_loss',dtype,expected,guard=native_only)
                guard.check()
                report['records'].append({'model':name,'dtype':str(dtype),'atoms':129,'edges':129*12,
                    'errors':errors,'configuration':case.metadata,'legacy_factory_candidate_label':'v6 selects current imported FastEq_v8',
                    'candidate_Triton_and_EQX_execution':'forbidden during complete physical loss.backward'})
                atomic_json(output,report)
                print(json.dumps({'passed':name,'dtype':str(dtype),'parameter_gradient_relative_l2':errors['gradient_relative_l2']}),flush=True)
                del expected,case
                torch.cuda.empty_cache()
        assert model_source_hashes()==hashes,'Native source drift during model QA'
        report.update(status='passed',guard=guard.check())
    except Exception as error:
        report.update(status='failed',error={'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        report['elapsed_s']=time.monotonic()-started;atomic_json(output,report)


if __name__=='__main__':main()
