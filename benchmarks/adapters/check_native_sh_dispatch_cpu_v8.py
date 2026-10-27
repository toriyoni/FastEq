"""CPU AST/metadata, fake and compile fallback bridge for SH dispatch changes."""
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND=ROOT/'src/FastEq_v8'
OUT=ROOT/'performance/fasteq_optimization/v8/sh'


def main():
    os.sched_setaffinity(0,{172})
    import torch
    torch.set_num_threads(1)
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._python_dispatch import TorchDispatchMode
    import fasteq.ops.spherical_harmonics as sh
    import fasteq.ops.scalar_expression as scalar
    archive=OUT/'measured_component_source'
    def node(base,path,name):
        tree=ast.parse((base/path).read_text())
        return next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name)
    def dump(n):return ast.dump(n,include_attributes=False)
    unchanged=['ScalarProgram','_parse','_store_tile_configuration','_source','_adjoint','_fake','_setup','_backward']
    for name in unchanged:
        assert dump(node(archive,'fasteq/ops/scalar_expression.py',name))==dump(node(CAND,'fasteq/ops/scalar_expression.py',name))
    assert [dump(n)for n in node(archive,'fasteq/ops/scalar_expression.py','scalar_expression').body]==[
        dump(n)for n in node(CAND,'fasteq/ops/scalar_expression.py','_execute_raw').body]
    for name in ['_program','_operator','spherical_harmonics']:
        assert dump(node(archive,'fasteq/ops/spherical_harmonics.py',name))==dump(node(CAND,'fasteq/ops/spherical_harmonics.py',name))
    records=[]
    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):return func(*args,**(kwargs or {}))
    original_raw=scalar._execute_raw
    def forbid_raw(*_args,**_kwargs):raise AssertionError('Fake/compiled-fake smoke attempted raw CUDA execution')
    scalar._execute_raw=forbid_raw
    try:
        for dtype in (torch.float32,torch.float64):
            with FakeTensorMode():
                x=torch.empty(7,3,device='cuda',dtype=dtype)
                y=sh.spherical_harmonics([0,1,2],x,True,'component')
                assert y.shape==(7,9)and y.device==x.device and y.dtype==dtype
                records.append({'case':'CUDA_fake_forward','dtype':str(dtype),'shape':list(y.shape)})
            with FakeTensorMode(),Mode():
                x=torch.empty(7,3,device='cuda',dtype=dtype)
                y=sh.spherical_harmonics([0,1,2],x,True,'component')
                assert y.shape==(7,9)
                records.append({'case':'fake_with_dispatch_mode','dtype':str(dtype),'shape':list(y.shape)})
        metadata=sh._program((0,1,2),True,'component')
        with FakeTensorMode():
            x=torch.empty(7,3,device='cuda')
            compiled=torch.compile(lambda z:scalar.scalar_expression(metadata,[z])[0],backend='eager',fullgraph=True)
            y=compiled(x);assert y.shape==(7,9)
            records.append({'case':'preserved_custom_op_fullgraph_compiled_fake','shape':list(y.shape)})
        oldpath=archive/'fasteq/ops/spherical_harmonics.py'
        spec=importlib.util.spec_from_file_location('fasteq.ops._cpu_sh_library_reference',oldpath)
        old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        outcomes=[]
        for module in (old,sh):
            torch._dynamo.reset()
            with FakeTensorMode():
                x=torch.empty(7,3,device='cuda')
                try:
                    torch.compile(lambda z:module.spherical_harmonics([0,1,2],z,True,'component'),backend='eager',fullgraph=True)(x)
                    outcomes.append({'status':'passed'})
                except Exception as error:
                    outcomes.append({'status':'unsupported','type':type(error).__name__,
                                     'first_line':str(error).splitlines()[0]})
        print(json.dumps({'outer_API_compile_outcomes':outcomes}),flush=True)
        assert outcomes[0]['status']==outcomes[1]['status']
        if outcomes[0]['status']=='unsupported':assert outcomes[0]['type']==outcomes[1]['type']
        records.append({'case':'outer_public_API_fullgraph_old_new_parity','old':outcomes[0],'new':outcomes[1],
                        'scope':'Existing cached module/AST program construction fullgraph limitation unchanged; no additional compiled API support is claimed.'})
    finally:scalar._execute_raw=original_raw
    paths=[CAND/'fasteq/ops/spherical_harmonics.py',CAND/'fasteq/ops/scalar_expression.py',
           CAND/'benchmarks/adapters/native_sh_dispatch_spy_v8.py',Path(__file__)]
    report={'status':'passed','math_AST_identical':unchanged+['_program','_operator','spherical_harmonics'],
            'old_execute_body_equals_raw_helper':True,'records':records,
            'native_raw_execution_forbidden_in_CPU_smoke':True,
            'torch_cuda_context_initialized_by_Dynamo_guards':torch.cuda.is_initialized(),
            'functorch_scope':'Active transforms retain the existing custom-op fallback. No new transform rule/support is claimed.',
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}}
    (OUT/'dispatch_CPU_AST_bridge.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(records)} CPU fallback checks and strict mathematical AST bridge passed; no raw CUDA execution')


if __name__=='__main__':main()
