"""CPU exact coefficients/IR equality for independently rederived Wigner builder."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
import torch
from check_native_norm_v8 import write_json
from fasteq.ops import rotation

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq');OUT=ROOT/'performance/fasteq_optimization/v8/wigner'
REFERENCE=OUT/'reference_quaternion_coefficients_v7.py'


def traced_coefficients(path):
    """CPU test-only AST instrumentation records integers before normalization."""
    source=path.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_quaternion_coefficients')
    node=copy.deepcopy(node);node.decorator_list=[]
    class Trace(ast.NodeTransformer):
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='coefficients' and n.value.func.attr=='append':
                return [ast.parse('integer_values.append(value)').body[0],n]
            return self.generic_visit(n)
        def visit_Return(self,n):
            n=self.generic_visit(n)
            if isinstance(n.value,ast.Tuple) and len(n.value.elts)==3:n.value.elts.append(ast.Name(id='integer_values',ctx=ast.Load()))
            return n
    node=Trace().visit(node);node.body.insert(1,ast.parse('integer_values=[]').body[0]);ast.fix_missing_locations(node)
    environment=vars(rotation).copy();exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::cpu_integer_trace','exec'),environment)
    return environment['_quaternion_coefficients']


def tensor_sha(value):return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not torch.cuda.is_initialized();torch.set_num_threads(1);args.output.parent.mkdir(parents=True,exist_ok=True)
    candidate=Path(rotation.__file__);pinned=hashlib.sha256(candidate.read_bytes()).hexdigest()
    report={'status':'running','scope':'CPU-only exact integer / tensor-byte / final ScalarProgram metadata comparisons',
        'candidate':str(candidate),'candidate_sha256':pinned,'reference':str(REFERENCE),
        'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),'degrees':[],'metadata':[],
        'gpu_rerun_needed':None,'cuda_initialized_before':False,'integer_trace':'AST instrumentation only appends pre-normalization integers; production public 3-tensor interface unchanged'}
    try:
        start=time.time();current=traced_coefficients(candidate)(12);report['native_build_seconds']=time.time()-start;write_json(args.output,report)
        start=time.time();old=traced_coefficients(REFERENCE)(12);report['reference_build_seconds']=time.time()-start
        for a,b in zip(current[:3],old[:3],strict=True):assert torch.equal(a,b)
        assert current[3]==old[3]
        report['maximum_degree_tables']=[{'dtype':str(t.dtype),'shape':list(t.shape),'sha256':tensor_sha(t)} for t in current[:3]]
        # Packed coefficient construction emits every degree in ascending order;
        # compare all 13 degree boundaries and their exact underlying integers.
        for degree in range(13):
            width=sum((2*l+1)**2 for l in range(degree+1));stop=int(current[0][width])
            assert torch.equal(current[0][:width+1],old[0][:width+1])
            assert torch.equal(current[1][:stop],old[1][:stop]);assert torch.equal(current[2][:stop],old[2][:stop]);assert current[3][:stop]==old[3][:stop]
            report['degrees'].append({'degree':degree,'packed_outputs':width,'terms':stop,'integer_coefficients_exact':True,
                'pointer_exponent_coefficient_bytes_exact':True});write_json(args.output,report)
        # Evaluate the unchanged public metadata builder with each actual degree
        # prefix from both CPU implementations; no CUDA compiler/context used.
        metadata_node=next(n for n in ast.parse(candidate.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_wigner_metadata')
        metadata_node=copy.deepcopy(metadata_node);metadata_node.decorator_list=[]
        def evaluator(tables):
            environment=vars(rotation).copy()
            def prefix(degree):
                width=sum((2*l+1)**2 for l in range(degree+1));stop=int(tables[0][width])
                return tables[0][:width+1],tables[1][:stop],tables[2][:stop]
            environment['_quaternion_coefficients']=prefix
            exec(compile(ast.Module(body=[metadata_node],type_ignores=[]),'CPU metadata equality','exec'),environment)
            return environment['_wigner_metadata']
        own_metadata,old_metadata=evaluator(current),evaluator(old)
        for degree in range(13):
            for dtype in ('float','double'):
                current_ir=own_metadata(degree,dtype);old_ir=old_metadata(degree,dtype)
                assert current_ir==old_ir
                report['metadata'].append({'degree':degree,'dtype':dtype,'metadata_bytes':len(current_ir.encode()),
                    'metadata_sha256':hashlib.sha256(current_ir.encode()).hexdigest(),'byte_exact':True});write_json(args.output,report)
                del current_ir,old_ir
        # Coefficient changes leave the Frame/Rotary emitters and recursive AD
        # nodes untouched. Their completed GPU QA is not repeated.
        archive=OUT.parent/'rotation/source/rotation.py'
        def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=nodes(candidate),nodes(archive)
        names=['_frame_source','_rotary_source','_Frame','_Rotary','_alignment_quaternion','_wigner_metadata','wigner_packed']
        assert all(a[n]==b[n] for n in names);report['unchanged_cuda_emitters_and_nodes']=names
        assert hashlib.sha256(candidate.read_bytes()).hexdigest()==pinned and not torch.cuda.is_initialized()
        report['cuda_initialized_after']=False;report['gpu_rerun_needed']=False;report['status']='pass'
    except BaseException as error:report['status']='failed';report['error']=repr(error);raise
    finally:write_json(args.output,report)
    print(json.dumps({'status':report['status'],'degree_fixtures':len(report['degrees']),'metadata_fixtures':len(report['metadata']),'gpu_rerun_needed':False}))


if __name__=='__main__':main()
