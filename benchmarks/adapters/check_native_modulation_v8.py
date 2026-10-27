"""CPU-only NVRTC preparation of FastEq hand-written modulation roles."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import torch
from check_native_norm_v8 import nvrtc_compile, write_json
from fasteq.ops.native_modulation import source_for

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
V8=ROOT/'src/FastEq_v8'


def node_dump(path,name):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    return ast.dump(node,include_attributes=False)


def prepare(path):
    assert not torch.cuda.is_initialized()
    report={'status':'running','scope':'CPU NVRTC compile only; no GPU precision/performance',
            'cuda_initialized_before':False,'specializations':[],'autograd_nodes_unchanged':[]}
    directory=path.parent/'prepared_sources';directory.mkdir(parents=True,exist_ok=True)
    configurations=[(16,6,((0,2,1,0),(2,3,3,2),(11,1,5,5))),
                    (576,192,((0,64,1,0),(64,64,3,64),(256,64,5,128))),
                    (103,11,((0,0,1,0),(0,2,1,0),(2,0,3,2),(2,7,13,2),(93,2,5,9)))]
    try:
        for name in ('_Modulation','_Gradient'):
            assert node_dump(V8/'fasteq/ops/modulation.py',name)==node_dump(ROOT/'src/FastEq_v7/fasteq/ops/modulation.py',name)
            report['autograd_nodes_unchanged'].append(name)
        for dtype in (torch.float32,torch.float64):
            for case,(dimension,channels,parts) in enumerate(configurations):
                base={'D':dimension,'C':channels,'PARTS':parts}
                for role,flags in [(0,{}),*( (1,{'NEED_X':bool(mask&1),'NEED_G':bool(mask&2)}) for mask in (1,2,3)),
                    *((2,{'NEED_X':bool(mask&1),'NEED_G':bool(mask&2),'NEED_DY':bool(mask&4),'HAS_U':bool(seed&1),'HAS_V':bool(seed&2)})
                      for mask,seed in ((1,2),(2,1),(4,1),(4,2),(7,3)))]:
                    kwargs={**base,**flags};source=source_for(role,kwargs,dtype)
                    artifact=directory/f'case{case}_{str(dtype).split(".")[-1]}_role{role}_{len(report["specializations"])}.cu'
                    artifact.write_text(source)
                    report['specializations'].append({'case':case,'dtype':str(dtype),'role':role,'flags':flags,
                        'source_path':str(artifact),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**nvrtc_compile(source)})
                    write_json(path,report)
        assert not torch.cuda.is_initialized();report['cuda_initialized_after']=False;report['status']='pass'
    except BaseException as error:
        report['status']='failed';report['error']=repr(error);raise
    finally:write_json(path,report)
    print(json.dumps({'status':report['status'],'specializations':len(report['specializations']),'cuda_initialized':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
