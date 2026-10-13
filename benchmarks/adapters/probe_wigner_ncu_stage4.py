"""One replayed forward kernel; NCU counter evidence, never normal timing."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from benchmark_wigner_stage4 import configure,RoundGuard,make_case,execute,source_records,UUID
from benchmark_scaling_case import SharedGPUError


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['legacy','tile'],required=True)
    p.add_argument('--atoms',type=int,default=131072);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure();guard=RoundGuard();before=guard.wait('before_setup')
    hashes=source_records();case=make_case(args.atoms,torch.float32)
    while True:
        guard.wait('before_warm')
        for _ in range(3):execute(case,args.backend,'forward')
        torch.cuda._sleep(1000);torch.cuda.synchronize()
        try:range_before=guard.guard.check();break
        except SharedGPUError as error:guard.reject('after_warm',error)
    torch.cuda.nvtx.range_push('WIGNER_STORES')
    try:
        value=execute(case,args.backend,'forward');torch.cuda.synchronize()
    finally:torch.cuda.nvtx.range_pop()
    # Do not retry a contaminated counter launch: --launch-count 1 would
    # otherwise retain the invalid earlier launch's report.
    after=guard.guard.check()
    assert source_records()==hashes
    report={'status':'clean counter launch; counters are separate CSV','measurement_kind':'NCU single-kernel replay, not normal timing',
            'backend':args.backend,'metadata':case['metadata'],'gpu_uuid':UUID,'before_guard':before,
            'range_before_guard':range_before,'after_guard':after,'source_sha256':hashes,
            'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rejections_before_target':guard.rejections}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output,flush=True)


if __name__=='__main__':main()
