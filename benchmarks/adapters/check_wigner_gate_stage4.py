"""Only row-gate boundary dispatch and tile capture after the final gate."""
import json
import os
import torch
from benchmark_wigner_stage4 import configure,RoundGuard,FLAG,compare_values,source_records
from check_wigner_stage4 import chain,native
from fasteq.ops import scalar_expression as implementation
from fasteq.ops.rotation import wigner_packed


def checks():
    passed=[]
    for dtype in (torch.float32,torch.float64):
        for rows in (3072,16385):
            x=torch.randn(rows,3,device='cuda',dtype=dtype).mul_(.2).add_(.5).requires_grad_()
            original_source=implementation._source;observed=[]
            def source(metadata,dtype,strategy='direct',block=128):
                observed.append((tuple(map(len,implementation._parse(metadata)[1])),strategy,block))
                return original_source(metadata,dtype,strategy,block)
            implementation._source=source
            try:
                legacy=chain(lambda v:native(v,2,0),(x,));legacy_observed=tuple(observed);observed.clear()
                tile=chain(lambda v:native(v,2,1),(x,));tile_observed=tuple(observed)
            finally:implementation._source=original_source
            assert all(row[1]=='direct'for row in legacy_observed)
            if rows==3072:assert tile_observed==legacy_observed
            else:assert [row[1]for row in tile_observed]==['tile','direct','tile','direct']
            compare_values(tile,legacy,dict(atol=0.,rtol=0.))
            passed.append({'dtype':str(dtype),'rows':rows,'legacy_dispatch':legacy_observed,'tile_dispatch':tile_observed})
    x=torch.randn(16385,3,device='cuda',dtype=torch.float64).mul_(.2).add_(.5).requires_grad_()
    os.environ[FLAG]='1';stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):chain(lambda v:wigner_packed(v,2),(x,))
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):recorded=chain(lambda v:wigner_packed(v,2),(x,))
    with torch.no_grad():x.add_(.01)
    graph.replay();compare_values(recorded,chain(lambda v:wigner_packed(v,2),(x,)),dict(atol=0.,rtol=0.))
    passed.append({'rows':16385,'dtype':'float64','case':'static-leaf nondefaultstream Tile Graph liveinput'})
    return passed


def main():
    configure();guard=RoundGuard();hashes=source_records();attempt=0
    while True:
        attempt+=1;before=guard.wait('before_gate_qa',attempt=attempt);torch.manual_seed(73129)
        passed=checks();torch.cuda.synchronize()
        try:after=guard.guard.check();break
        except Exception as error:
            from benchmark_scaling_case import SharedGPUError
            if not isinstance(error,SharedGPUError):raise
            guard.reject('after_gate_qa',error,attempt=attempt)
    assert source_records()==hashes
    print(json.dumps({'status':'pass','unique_cases':len(passed),'passed':passed,'attempts':attempt,
                      'before_guard':before,'after_guard':after,'rejections':guard.rejections,'source_sha256':hashes}),flush=True)


if __name__=='__main__':main()
