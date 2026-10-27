"""CPU CUBIN bridge for an unused FCTP helper addition and Wigner host rewrite."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
import torch
from check_native_norm_v8 import nvrtc_compile,write_json
ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq');V8=ROOT/'src/FastEq_v8';OUT=ROOT/'performance/fasteq_optimization/v8'


def main():
    assert not torch.cuda.is_initialized()
    current_path=V8/'fasteq/ops/rotation.py';previous_path=OUT/'rotation/source/rotation.py'
    def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=nodes(current_path),nodes(previous_path)
    names=['_frame_source','_rotary_source','_Frame','_Rotary','_alignment_quaternion','_wigner_metadata','wigner_packed']
    assert all(a[n]==b[n] for n in names)
    header=(V8/'fasteq/cuda/src/fctp_shared.cuh').read_text()
    marker='// Contiguity alone does not imply vector alignment (e.g. storage_offset=1).\n'
    unused=header.split(marker,1)[1].split('}  // namespace fasteq_fctp',1)[0]
    base_header=header.replace(marker+unused,'')
    output=OUT/'rotation/source_bridge_after_wigner_cpu.json';directory=OUT/'rotation/shared_header_bridge_sources';directory.mkdir(exist_ok=True)
    report={'status':'running','scope':'CPU-only exact NVRTC CUBIN bridge; completed GPU QA is retained',
        'previous_rotation_sha256':hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        'current_rotation_sha256':hashlib.sha256(current_path.read_bytes()).hexdigest(),
        'unchanged_AST_symbols':names,'current_fctp_shared_sha256':hashlib.sha256(header.encode()).hexdigest(),
        'entries':[],'cuda_initialized_before':False,'gpu_repeat_needed':None}
    try:
        for folder in ('profile_frame_sources','profile_rotary_sources'):
            for path in sorted((OUT/'rotation'/folder).glob('*.cu')):
                previous=path.read_text();start=previous.index('#pragma once\n\n// Extracted from FastEq')
                end=previous.index('}  // namespace fasteq_fctp',start)+len('}  // namespace fasteq_fctp\n')
                previous_header=previous[start:end]
                assert re.sub(r'\s+','',previous_header)==re.sub(r'\s+','',base_header)
                report['previous_fctp_shared_sha256_from_emitted_source']=hashlib.sha256(previous_header.encode()).hexdigest()
                current=previous[:start]+header+previous[end:];digest=hashlib.sha256(current.encode()).hexdigest();artifact=directory/(digest+'.cu');artifact.write_text(current)
                oldbinary,newbinary=nvrtc_compile(previous),nvrtc_compile(current)
                assert oldbinary['cubin_sha256']==newbinary['cubin_sha256']
                report['entries'].append({'previous_emitted_source_path':str(path),'previous_emitted_source_sha256':hashlib.sha256(previous.encode()).hexdigest(),
                    'current_emitted_source_path':str(artifact),'current_emitted_source_sha256':digest,
                    'cubin_byte_identical':True,'cubin_sha256':newbinary['cubin_sha256'],'cubin_bytes':newbinary['cubin_bytes']})
                write_json(output,report)
        assert not torch.cuda.is_initialized();report['cuda_initialized_after']=False;report['gpu_repeat_needed']=False;report['status']='pass'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:write_json(output,report)
    print(json.dumps({'status':report['status'],'CUBIN_byte_exact_entries':len(report['entries']),'gpu_repeat_needed':False}))


if __name__=='__main__':main()
