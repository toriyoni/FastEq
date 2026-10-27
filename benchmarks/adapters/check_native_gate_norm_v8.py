"""CPU-only compile preparation of fused FastEq CUDA GateNorm."""
import argparse
import ast
import hashlib
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import torch
from check_native_norm_v8 import nvrtc_compile,write_json
from check_native_modulation_v8 import node_dump,ROOT,V8
from fasteq.ops.native_gate_norm import source_for


def prepare(path):
    assert not torch.cuda.is_initialized();directory=path.parent/'prepared_sources';directory.mkdir(parents=True,exist_ok=True)
    report={'status':'running','scope':'CPU NVRTC compile only; no GPU precision/performance',
            'cuda_initialized_before':False,'specializations':[],'autograd_nodes_unchanged':[]}
    configurations=[(22,16,((0,2,1,0,0,-1,0),(2,3,3,1,0,-1,0),(11,1,5,2,0,-1,0)),(2,3,1)),
        (768,576,((0,64,1,0,0,-1,0),(64,64,3,1,0,-1,0),(256,64,5,2,0,-1,0)),(64,64,64)),
        (2304,2112,((0,0,1,0,0,-1,0),(0,64,1,1,0,-1,0),(64,64,32,2,0,-1,0)),(0,64,64)),
        (8064,8000,((0,32,1,0,0,-1,0),(32,32,249,1,0,-1,0)),(32,32))]
    try:
        for name in ('_GateNorm','_GateNormBackward'):
            assert node_dump(V8/'fasteq/ops/gate_norm.py',name)==node_dump(ROOT/'src/FastEq_v7/fasteq/ops/gate_norm.py',name)
            report['autograd_nodes_unchanged'].append(name)
        for dtype in (torch.float32,torch.float64):
            for case,(din,dout,parts,widths) in enumerate(configurations):
                base={'DIN':din,'DOUT':dout,'PARTS':parts,'GFLAGS':(True,)*len(widths),'EPS':1e-5,'MAX_FAN':6}
                for role,label,flags in [(0,'forward',{}),
                    (1,'all',{'DZFLAG':True,'GGRADFLAGS':(True,)*len(widths)}),
                    (1,'parameter_only',{'DZFLAG':False,'GGRADFLAGS':(True,)*len(widths)}),
                    (2,'all',{'DDZFLAG':True,'DDYFLAG':True,'GGRADFLAGS':(True,)*len(widths),'HFLAGS':(True,)*len(widths),'HAS_U':True}),
                    (2,'u_only',{'DDZFLAG':True,'DDYFLAG':False,'GGRADFLAGS':(False,)*len(widths),'HFLAGS':(False,)*len(widths),'HAS_U':True}),
                    (2,'h_only',{'DDZFLAG':True,'DDYFLAG':True,'GGRADFLAGS':(False,)*len(widths),'HFLAGS':(True,)*len(widths),'HAS_U':False})]:
                    if role:flags={**flags,'GOFFSETS':tuple(sum(widths[:i]) for i in range(len(widths))),'WIDTH':sum(widths) if any(flags['GGRADFLAGS']) else 0}
                    kwargs={**base,**flags};source=source_for(role,kwargs,dtype)
                    artifact=directory/f'case{case}_{str(dtype).split(".")[-1]}_role{role}_{label}.cu';artifact.write_text(source)
                    report['specializations'].append({'case':case,'dtype':str(dtype),'role':role,'label':label,'flags':flags,
                        'source_path':str(artifact),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**nvrtc_compile(source)})
                    write_json(path,report)
        assert not torch.cuda.is_initialized();report['cuda_initialized_after']=False;report['status']='pass'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:write_json(path,report)
    print({'status':report['status'],'specializations':len(report['specializations']),'cuda_initialized':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
