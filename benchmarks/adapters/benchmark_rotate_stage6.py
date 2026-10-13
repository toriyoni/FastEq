"""Prepared normal/profiler driver; no benchmark runs before stage6 begins."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND = ROOT / 'src/FastEq_v7'
sys.path.insert(0, str(CAND / 'benchmarks/adapters'))
from benchmark_scaling_case import ExclusiveGuard, SharedGPUError
from scaling_operator_factories import build_case
import torch
import fasteq

FLAG = 'FASTEQ_EXPERIMENTAL_ROTATE_VECTOR'
LABELS = ('original', 'legacy', 'new')


def statistics_for(samples):
    q = statistics.quantiles(samples, n=4, method='inclusive')
    return {'samples': samples, 'median': statistics.median(samples),
            'q25': q[0], 'q75': q[2]}


def run(case, label, mode):
    os.environ[FLAG] = '1' if label == 'new' else '0'
    inputs = case['inputs']
    operation = case['baseline' if label == 'original' else 'candidate']
    with torch.set_grad_enabled(mode != 'forward'):
        output = operation(*inputs)
        if mode == 'forward':
            return (output,)
        loss = output.sin().mean() + .1 * output.square().mean()
        first = torch.autograd.grad(loss, inputs, create_graph=mode == 'feature_higher')
        if mode == 'backward':
            return (output, *first)
        second = torch.autograd.grad(sum(g.square().mean() for g in first), inputs)
        return (output, *first, *second)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--atoms', type=int, required=True)
    parser.add_argument('--mode', choices=['forward', 'backward', 'feature_higher'], required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--cpu', type=int, default=174)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == args.gpu_uuid
    assert Path(fasteq.__file__).resolve().is_relative_to(CAND)
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    guard = ExclusiveGuard(args.gpu_uuid)
    rejections = []

    def rejected(stage, error, **context):
        item = {'stage': stage, 'error': str(error), 'time_unix': time.time(), **context}
        rejections.append(item)
        print(json.dumps({'rejected': item}), flush=True)

    def idle(stage, **context):
        while True:
            try:
                return guard.check()
            except SharedGPUError as error:
                rejected(stage, error, **context)
                time.sleep(1)

    before = idle('before_setup')
    source = CAND / 'fasteq/ops/rotation.py'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    driver_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.manual_seed(23383)
    # Full small-shape QA keeps three complete derivative tuples out of the
    # large-workload peak. Large timing is separate and is not full-output QA.
    precision_atoms = min(args.atoms, 257)
    precision_case = build_case('rotate', precision_atoms, 'float64', 'cuda')
    precision_attempts = 0
    while True:
        precision_attempts += 1
        idle('before_precision', attempt=precision_attempts)
        results = {label: run(precision_case, label, args.mode) for label in LABELS}
        torch.cuda.synchronize()
        precision = {}
        for label in ('legacy', 'new'):
            errors = []
            for a, b in zip(results[label], results['original'], strict=True):
                torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-7)
                errors.append((a - b).abs().max().item())
            precision[label] = {'status': 'passed', 'full_tensor_max_absolute': errors}
        for a, b in zip(results['legacy'], results['new'], strict=True):
            torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-7)
        del results
        gc.collect()
        torch.cuda.synchronize()
        try:
            precision_guard = guard.check()
            break
        except SharedGPUError as error:
            rejected('after_precision', error, attempt=precision_attempts)
    del precision_case
    gc.collect()
    torch.cuda.empty_cache()
    idle('before_timing_setup')
    case = build_case('rotate', args.atoms, 'float64', 'cuda')
    if args.profile:
        profiles = {}
        for label in LABELS:
            attempt = 0
            while True:
                attempt += 1
                idle('before_profile', label=label, attempt=attempt)
                for _ in range(3):
                    run(case, label, args.mode)
                torch.cuda.synchronize()
                try:
                    guard.check()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        run(case, label, args.mode)
                        torch.cuda.synchronize()
                    guard.check()
                except SharedGPUError as error:
                    rejected('profile', error, label=label, attempt=attempt)
                    continue
                trace = args.output.with_name(args.output.stem + '_' + label + '_trace.json')
                prof.export_chrome_trace(str(trace))
                events = [{'name': e.name, 'device_time_us': e.device_time_total}
                          for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                profiles[label] = {'trace': str(trace), 'kernels': events,
                                   'summed_kernel_device_ms': sum(e['device_time_us'] for e in events) / 1000}
                break
        measurement = {'kind': 'instrumented profile; kernel sums are not normal wall latency',
                       'profiles': profiles}
    else:
        timers = {label: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for label in LABELS}
        for start, end in timers.values():
            start.record()
            end.record()
        torch.cuda.synchronize()
        wall, device = ({label: [] for label in LABELS} for _ in range(2))
        accepted, attempts = 0, 0
        while accepted < args.rounds:
            attempts += 1
            shift = accepted % 3
            order = LABELS[shift:] + LABELS[:shift]
            if accepted % 2:
                order = tuple(reversed(order))
            idle('before_round', accepted=accepted, attempt=attempts, order=order)
            current_wall, current_device = {}, {}
            stage = 'warmup'
            try:
                for label in LABELS:
                    for _ in range(2):
                        run(case, label, args.mode)
                torch.cuda.synchronize()
                guard.check()
                for label in order:
                    stage = 'before_' + label
                    guard.check()
                    start, end = timers[label]
                    start_time = time.perf_counter()
                    start.record()
                    for _ in range(args.iterations):
                        run(case, label, args.mode)
                    end.record()
                    torch.cuda.synchronize()
                    current_wall[label] = (time.perf_counter() - start_time) * 1000 / args.iterations
                    current_device[label] = start.elapsed_time(end) / args.iterations
                    stage = 'after_' + label
                    guard.check()
            except SharedGPUError as error:
                rejected(stage, error, accepted=accepted, attempt=attempts, order=order)
                continue
            for label in LABELS:
                wall[label].append(current_wall[label])
                device[label].append(current_device[label])
            accepted += 1
        measurement = {'kind': 'normal same-process interleaved; complete-round foreign rejection',
                       'round_attempts': attempts, 'wall_ms': wall, 'cuda_event_ms': device,
                       'wall_median_ms': {k: statistics.median(v) for k, v in wall.items()},
                       'event_median_ms': {k: statistics.median(v) for k, v in device.items()},
                       'paired_wall': {k + '/new': statistics_for([a / b for a, b in zip(wall[k], wall['new'], strict=True)])
                                       for k in ('original', 'legacy')},
                       'paired_event': {k + '/new': statistics_for([a / b for a, b in zip(device[k], device['new'], strict=True)])
                                        for k in ('original', 'legacy')}}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == driver_sha
    report = {'status': 'ok', 'args': vars(args) | {'output': str(args.output)}, 'source_sha256': source_sha,
              'driver_sha256': driver_sha,
              'metadata': case['metadata'], 'mode_scope': 'operator feature-gradient chain, not physical force loss',
              'dtype': 'float64', 'tolerance': {'atol': 3e-8, 'rtol': 3e-7},
              'measurement': measurement, 'precision': precision, 'precision_attempts': precision_attempts,
              'precision_scope': {'full_output_atoms': precision_atoms, 'large_output_validation': False,
                                  'large_input_setup_and_QA_excluded_from_timing': True},
              'precision_after_guard': precision_guard, 'rejected_attempts': rejections,
              'provenance': {'before_guard': before, 'after_guard': idle('after_measurement'),
                             'actual_import': fasteq.__file__, 'torch': torch.__version__,
                             'cuda': torch.version.cuda, 'tf32': False, 'cpu': sorted(os.sched_getaffinity(0)),
                             'torch_extensions_dir': os.environ.get('TORCH_EXTENSIONS_DIR'),
                             'native_cache': os.environ.get('XDG_CACHE_HOME'),
                             'common_flags': {k: v for k, v in os.environ.items() if k.startswith('FASTEQ_')}}}
    report['provenance']['per_backend_route_flags'] = {'original': '0', 'legacy': '0', 'new': '1'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'status': 'ok',
                      'measurement': {k: v for k, v in measurement.items() if k in ('wall_median_ms', 'paired_wall', 'paired_event')}}), flush=True)


if __name__ == '__main__':
    main()
