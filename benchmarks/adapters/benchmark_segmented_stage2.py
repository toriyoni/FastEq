"""Exclusive targeted TP stage2 checks and paired timing against stage1 CUDA."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch
import fasteq
from benchmark_scaling_case import ExclusiveGuard
from fasteq.ops import segmented_product as candidate

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
BASE = ROOT / 'performance/fasteq_optimization/v7/stages/linear/source/FastEq/fasteq/ops/segmented_product.py'


def chain(fn, metadata, leaves, mode):
    w, x, y = leaves
    with torch.set_grad_enabled(mode != 'forward'):
        z = fn(metadata, w, x, y)
    if mode == 'forward':
        return (z,)
    objective = (z.sin() + .1 * z.square()).mean()
    if mode == 'backward':
        return (z, *torch.autograd.grad(objective, leaves))
    first = torch.autograd.grad(objective, (x, y), create_graph=True)
    loss = sum(g.square().mean() for g in first)
    return (z, *first, *torch.autograd.grad(loss, leaves))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--rows', type=int, default=4096)
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--stage', choices=['2a','2b','2c'], default='2b')
    p.add_argument('--operator', choices=['synthetic_uvw','tp_nonuniform'], default='synthetic_uvw')
    p.add_argument('--disable-pair',action='store_true')
    args = p.parse_args()
    uuid = 'GPU-13222c8f-3c05-b93a-e9d6-1192c18a00ce'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == uuid
    assert str(Path(fasteq.__file__).resolve()).startswith(str(ROOT / 'src/FastEq_v7/'))
    os.sched_setaffinity(0, {173})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    os.environ['FASTEQ_EXPERIMENTAL_SEGMENTED_TASK_MASK'] = '1'
    os.environ['FASTEQ_EXPERIMENTAL_SEGMENTED_REDUCE'] = str(int(args.stage != '2a'))
    os.environ['FASTEQ_EXPERIMENTAL_SEGMENTED_GEMM'] = str(int(args.stage == '2c'))
    os.environ['FASTEQ_EXPERIMENTAL_SEGMENTED_PAIR'] = str(int(args.stage=='2c' and not args.disable_pair))
    guard = ExclusiveGuard(uuid)
    before_guard=guard.check()
    spec = importlib.util.spec_from_file_location('stage1_segmented_reference', BASE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(7419)
    c, rows = args.channels, args.rows
    metadata = repr(((c**3,c,c,c), ((3,0,0,1,0,1,0,1,c,c,c,1),), (c**-1,)))
    leaves = [torch.randn(shape, device='cuda', dtype=dtype).requires_grad_()
              for shape in ((1,c**3),(rows,c),(rows,c))]
    from e3nn import o3
    torch_module=o3.TensorProduct(f'{c}x0e',f'{c}x0e',f'{c}x0e',[(0,0,0,'uvw',True)]).to(device='cuda',dtype=dtype)
    torch_reference=lambda metadata,w,x,y:torch_module(x,y,w.reshape(-1))
    if args.operator == 'tp_nonuniform':
        os.environ['FASTEQ_EXPERIMENTAL_LINEAR_SIMT']='1'
        os.environ['FASTEQ_EXPERIMENTAL_NATIVE_MODULATION']='1'
        from scaling_operator_factories import build_case
        case=build_case('tp_nonuniform',rows,dtype,'cuda')
        metadata=case['candidate']._fasteq_plan.metadata
        leaves=[case['candidate'].weight.detach().reshape(1,-1).requires_grad_(),*case['inputs']]
        torch_reference=lambda metadata,w,x,y:case['baseline'](x,y,w.reshape(-1))
    report = {'stage': args.stage, 'operator':args.operator,'dtype':args.dtype,'channels':c,'rows':rows,
              'reference_source':str(BASE),'candidate_source':candidate.__file__, 'modes':{}}
    source_root=Path(fasteq.__file__).parents[1]
    files=[source_root/name for name in ('fasteq/ops/segmented_product.py',
            'fasteq/cuda/src/segmented_product_plan.cpp','fasteq/cuda/src/jit_runtime.h',
            'fasteq/cuda/src/jit_runtime.cpp','fasteq/cuda/jit.py','fasteq/cuda/src/indexed_linear_plan.cpp')]
    files += [Path(__file__),BASE]
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report['effective_flags']={name:os.environ.get(name) for name in (
        'FASTEQ_EXPERIMENTAL_LINEAR_GEMM','FASTEQ_EXPERIMENTAL_LINEAR_SIMT',
        'FASTEQ_EXPERIMENTAL_NATIVE_MODULATION','FASTEQ_EXPERIMENTAL_SEGMENTED_TASK_MASK',
        'FASTEQ_EXPERIMENTAL_SEGMENTED_REDUCE','FASTEQ_EXPERIMENTAL_SEGMENTED_GEMM',
        'FASTEQ_EXPERIMENTAL_SEGMENTED_PAIR')}
    report['provenance']={'actual_import':fasteq.__file__,'torch_version':torch.__version__,
                          'cuda_version':torch.version.cuda,'cpu_affinity':sorted(os.sched_getaffinity(0)),
                          'threads':torch.get_num_threads(),'torch_extensions_dir':os.environ['TORCH_EXTENSIONS_DIR'],
                          'native_cuda_cache':os.environ.get('XDG_CACHE_HOME'),
                          'before_guard':before_guard,'tf32':False}
    report['rows_factor']=12 if args.operator=='synthetic_uvw' else 1
    report['atoms_equivalent']=rows/report['rows_factor']
    report['chain_kind']='feature_higher; no physical coordinate graph'
    report['dimensions'],report['terms'],report['coefficients']=candidate._parse(metadata)
    # The original scalar loop and GEMM accumulate 4096 channels in different
    # orders. Use the adapter's established FP32 reference tolerance.
    tolerance = dict(atol=3e-5,rtol=5e-5) if dtype == torch.float32 else dict(atol=2e-11,rtol=2e-10)
    for mode in ('forward','backward','force_loss'):
        reference = chain(base.segmented_product, metadata, leaves, mode)
        actual = chain(candidate.segmented_product, metadata, leaves, mode)
        for a,b in zip(actual,reference,strict=True):
            torch.testing.assert_close(a,b,**tolerance)
        if torch_reference is not None:
            original=chain(torch_reference,metadata,leaves,mode)
            for a,b in zip(actual,original,strict=True):
                torch.testing.assert_close(a,b,**tolerance)
        for fn in (base.segmented_product,candidate.segmented_product,torch_reference):
            for _ in range(2):
                chain(fn,metadata,leaves,mode)
        torch.cuda.synchronize()
        times={'previous_v7':[], 'candidate':[], 'original_torch':[]}
        for r in range(args.rounds):
            order = [('previous_v7',base.segmented_product),('candidate',candidate.segmented_product),('original_torch',torch_reference)]
            shift=r%3;order=order[shift:]+order[:shift]
            if r % 2: order.reverse()
            for label,fn in order:
                guard.check()
                start=time.perf_counter()
                chain(fn,metadata,leaves,mode)
                torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                guard.check()
                times[label].append(elapsed)
        med={key:statistics.median(value) for key,value in times.items()}
        ratios={key:[a/b for a,b in zip(times[key],times['candidate'],strict=True)]
                for key in ('previous_v7','original_torch')}
        def summary(values):
            # Inclusive quartiles on the paired nine normal rounds.
            quartiles=statistics.quantiles(values,n=4,method='inclusive')
            return {'median':statistics.median(values),'q25':quartiles[0],'q75':quartiles[2]}
        report['modes'][mode]={'wall_ms':times,'median_ms':med,
                              'speedup':med['previous_v7']/med['candidate'], 'precision':'pass',
                              'speedup_original_torch':med['original_torch']/med['candidate'],
                              'max_abs_errors_previous':[(a-b).abs().max().item() for a,b in zip(actual,reference,strict=True)]}
        report['modes'][mode]['paired_round_ratios']={key:{'rounds':values,**summary(values)} for key,values in ratios.items()}
        report['modes'][mode]['speedup_kind']='ratio of wall-time medians; paired_round_ratios separately reported'
        if torch_reference is not None:
            report['modes'][mode]['max_abs_errors_original_torch']=[(a-b).abs().max().item() for a,b in zip(actual,original,strict=True)]
        print(mode, report['modes'][mode], flush=True)
    report['provenance']['after_guard']=guard.check()
    for file,digest in report['source_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,'Source drift during benchmark: '+file
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__': main()
