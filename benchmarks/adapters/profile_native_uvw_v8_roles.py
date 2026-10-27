"""Isolated instrumented UVW role intervals with restored function identities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from native_uvw_v8_common import CANDIDATE, UUID, write_json
sys.path.insert(0, str(CANDIDATE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', default=UUID)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--operator', choices=['tp_uvw', 'tp_nonuniform'], default='tp_uvw')
    parser.add_argument('--atoms', type=int, default=4096)
    parser.add_argument('--unfused', action='store_true')
    args = parser.parse_args()
    import torch
    from fasteq.ops import segmented_product as native
    from benchmark_native_uvw_v8 import worker
    assert hasattr(torch.autograd, '_profiler_enabled')
    old_execute = native._execute
    old_pair = native._FeaturePair.forward
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    intervals = []
    def role_call(label, fn, values):
        if not torch.autograd._profiler_enabled():
            return fn()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        with torch.profiler.record_function(label):
            start.record(); result = fn(); end.record()
        intervals.append((label, start, end, [list(v.shape) for v in values]))
        return result
    def execute(metadata, role, required, rows, values):
        return role_call(f'UVW_role{role}_required{required}',
                         lambda: old_execute(metadata, role, required, rows, values), values)
    def pair(ctx, metadata, required, rows, *values):
        return role_call(f'UVW_feature_pair_required{required}',
                         lambda: old_pair(ctx, metadata, required, rows, *values), values)
    native._execute = execute
    native._FeaturePair.forward = staticmethod(pair)
    try:
        worker(argparse.Namespace(command='profile', label='v8_unfused' if args.unfused else 'v8_native',
               gpu_uuid=args.gpu_uuid, dtype=args.dtype, operator=args.operator, atoms=args.atoms,
               warmup=5, iterations=3, snapshot=False, mode='feature_higher', output=args.output))
    finally:
        native._execute = old_execute
        native._FeaturePair.forward = staticmethod(old_pair)
        assert native._execute is old_execute and native._FeaturePair.forward is old_pair
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == digest
    records = [{'label': label, 'instrumented_event_ms': start.elapsed_time(end), 'operand_shapes': shapes}
               for label, start, end, shapes in intervals]
    assert records
    grouped = {}
    for row in records:
        bucket = grouped.setdefault(row['label'], {'count': 0, 'instrumented_event_ms': 0.})
        bucket['count'] += 1
        bucket['instrumented_event_ms'] += row['instrumented_event_ms']
    report = {'status': 'ok', 'measurement_kind': 'instrumented_role_intervals_not_normal_latency',
              'wrapper_sha256': digest, 'kernel_report': str(args.output),
              'kernel_report_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'function_identities_restored': True, 'roles': records, 'grouped': grouped,
              'scope': 'CUDA events bracket actual FastEq role dispatch only during profiling. Kernel-only math sums come from separate Chrome cat=kernel classification.'}
    write_json(args.output.with_suffix('.roles.json'), report)
    print(json.dumps({'roles': grouped, 'restored': True}), flush=True)


if __name__ == '__main__':
    main()
