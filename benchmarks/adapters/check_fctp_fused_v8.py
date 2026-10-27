"""Only the newly fused scalar64 FCTP domain, including recursive adjoints."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from fasteq.ops import segmented_product as native
from native_uvw_v8_common import (ROOT, chain, compare, dense_metadata, random_leaves,
                                  reference, source_hashes, verify_hashes, write_json)
from check_native_gate_invariants_v8 import native_only
sys.path.insert(0,str(ROOT/'performance/fasteq_optimization/v7/driver'))
from pair_common import ExclusiveGuard


def main():
    assert os.environ.get('FASTEQ_EXPERIMENTAL_FCTP_FUSED')=='1'
    assert os.environ.get('FASTEQ_EXPERIMENTAL_FCTP_F32_F64MMA')=='1'
    uuid=os.environ.get('FASTEQ_FUSED_QA_UUID','GPU-af44a7bc-24c0-45be-df6e-03f4c445c78d')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==uuid
    guard=ExclusiveGuard(uuid);guard.check()
    torch.manual_seed(49196)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    hashes=source_hashes();hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dtypes=tuple(getattr(torch,name) for name in os.environ.get('FASTEQ_FUSED_QA_DTYPES','float32,float64').split(','))
    tag=os.environ.get('FASTEQ_FUSED_QA_TAG','fused')
    output=ROOT/('performance/fasteq_optimization/v8/uvw/'+tag+'_QA.json')
    report={'status':'running','source_sha256':hashes,'scope':'Only scalar64 continuous-Y FCTP fused dispatch; no timing/OOM rerun',
            'flags':{k:v for k,v in os.environ.items() if k.startswith('FASTEQ_')},'records':[]}
    records=report['records'];metadata=dense_metadata(64)
    started=time.monotonic()
    def record(value):
        torch.cuda.synchronize();guard.check();records.append(value);write_json(output,report)
    try:
        for dtype in dtypes:
            for rows,layout in ((9,'offset'),(129,'transpose')):
                leaves=random_leaves(metadata,rows,dtype,layout)
                expected=chain(reference,metadata,leaves,third=True)
                with native_only():actual=chain(native.segmented_product,metadata,leaves,third=True)
                record({'case':'full_first_mixed_second_third','dtype':str(dtype),'rows':rows,'layout':layout,
                        'errors':compare(actual,expected,dtype)})
            leaves=random_leaves(metadata,17,dtype,'offset')
            for requested in ((0,),(1,),(2,),(0,2),(1,2)):
                expected=chain(reference,metadata,leaves,requested=requested)
                with native_only():actual=chain(native.segmented_product,metadata,leaves,requested=requested)
                record({'case':'selected_leaves','dtype':str(dtype),'requested':requested,
                        'errors':compare(actual,expected,dtype)})
            for chosen in (0,1):
                def omitted(fn):
                    out=fn(metadata,*leaves)
                    first=torch.autograd.grad(out.sin().mean(),leaves[1:],create_graph=True)
                    higher=torch.autograd.grad(first[chosen].square().mean(),leaves)
                    return out,*first,*higher
                expected=omitted(reference)
                with native_only():actual=omitted(native.segmented_product)
                record({'case':'paired_None_seed','dtype':str(dtype),'chosen':chosen,
                        'errors':compare(actual,expected,dtype)})
            values=(*random_leaves(metadata,9,dtype,'offset'),
                    torch.randn(9,64,device='cuda',dtype=dtype).mul_(.2).requires_grad_())
            for role in range(4):
                def evaluate(candidate):
                    if candidate:
                        out=native._Product.apply(metadata,role,0,9,*values)
                    else:
                        out=reference(metadata,*values[:3])
                        if role!=3:out=torch.autograd.grad((out*values[3]).sum(),values[role],create_graph=True)[0]
                    first=torch.autograd.grad((out.sin()+.1*out.square()).mean(),
                                              [values[i] for i in range(4) if i!=role],create_graph=True)
                    second=torch.autograd.grad(sum(g.square().mean() for g in first),values,allow_unused=True)
                    return out,*first,*second
                expected=evaluate(False)
                with native_only():actual=evaluate(True)
                record({'case':'direct_transpose_role','role':role,'dtype':str(dtype),
                        'errors':compare(actual,expected,dtype)})
        # Reuse the already reviewed physical fixture; all five parameters and
        # coordinates are compared, including a live update/nondefault stream.
        from check_native_uvw_v8 import physical
        with native_only():physical_records=physical(native,guard,torch,dtypes=dtypes)
        for value in physical_records:record(value)
        # Fresh stream-owned leaves keep default-stream reference autograd out
        # of capture while testing all live inputs and full dW.
        for dtype in dtypes:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),native_only():
                leaves=random_leaves(metadata,17,dtype,'offset')
                for _ in range(3):chain(native.segmented_product,metadata,leaves)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with native_only(),torch.cuda.graph(graph,stream=stream):actual=chain(native.segmented_product,metadata,leaves)
            with torch.no_grad():leaves[0].add_(.013);leaves[1].mul_(.81)
            with native_only():graph.replay()
            torch.cuda.synchronize()
            expected=chain(reference,metadata,leaves)
            record({'case':'full_dW_live_inputs_CUDA_graph','dtype':str(dtype),
                    'errors':compare(actual,expected,dtype)})
        verify_hashes(hashes)
        report.update(status='passed',passed_cases=len(records),guard=guard.check())
    except Exception as error:
        report.update(status='failed',error={'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        report['elapsed_s']=time.monotonic()-started;write_json(output,report)
    print(f'{len(records)} fused FCTP recursive/physical/Graph conditions passed',flush=True)


if __name__=='__main__':main()
