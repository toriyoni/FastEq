"""Fresh-process UVW comparisons and isolated kernel evidence.

Normal latency and instrumented profiler output are deliberately separate.
The frozen V7 cuBLAS implementation is a naive reference only; the V8 worker
never imports its FastEq package. Use controller after production-source freeze.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from native_uvw_v8_common import (ROOT, CANDIDATE, NAIVE, UUID, chain, configure,
    distribution, operator_case, parse_trace, source_hashes, tolerance,
    verify_hashes, write_json)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logical_snapshot(values, parameter_slot, torch):
    result = []
    for index, value in enumerate(values):
        if value is None:
            result.append(None); continue
        value = value.detach()
        finite = bool(torch.isfinite(value).all())
        assert finite
        if index == parameter_slot or value.numel() <= 256:
            sample = value.cpu().reshape(-1)
            scope = 'full_parameter_gradient' if index == parameter_slot else 'full_small_output'
        else:
            indices = torch.linspace(0, value.numel()-1, 256, device=value.device).long()
            sample = value.reshape(-1).index_select(0, indices).cpu()
            scope = 'logical_sample256_with_full_finite'
        result.append({'data': sample, 'shape': list(value.shape), 'scope': scope,
                       'finite': finite, 'elements': value.numel()})
    return result


def worker(args):
    # The script belongs to V8; this explicit import root isolates a V7 child.
    source = NAIVE if args.label == 'v7_naive' else CANDIDATE
    sys.path.insert(0, str(source))
    import torch
    import fasteq
    from benchmark_scaling_case import ExclusiveGuard
    from fasteq.ops import segmented_product as native
    configure(source, fusion=args.label != 'v8_unfused', cpu=args.worker_cpu, uuid=args.gpu_uuid)
    fused = args.label in ('v8_native', 'v8_native_simt')
    os.environ['FASTEQ_EXPERIMENTAL_FCTP_FUSED'] = str(int(fused))
    os.environ['FASTEQ_EXPERIMENTAL_FCTP_F32_F64MMA'] = str(int(fused))
    os.environ['FASTEQ_EXPERIMENTAL_MATRIX_F32_F64MMA'] = str(int(
        args.label.startswith('v8_') and args.label != 'v8_native_simt'))
    torch.manual_seed(838471)
    guard = ExclusiveGuard(args.gpu_uuid)
    before = guard.check()
    hashes = source_hashes(source)
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    report = {'status': 'running', 'label': args.label, 'operator': args.operator,
              'mode': args.mode, 'dtype': args.dtype, 'atoms_equivalent': args.atoms,
              'rows': args.atoms*12, 'rows_factor': 12, 'atoms_scope': 'equivalent_only_no_atom_graph',
              'chain_scope': 'feature_higher uses dX/dY with create_graph then loss to all W/X/Y; no physical coordinates',
              'source_root': str(source), 'actual_import': fasteq.__file__, 'source_sha256': hashes,
              'driver_sha256': sha(__file__), 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'CPU_affinity': sorted(os.sched_getaffinity(0)), 'torch_allow_tf32': False,
              'candidate_matrix_tf32x3_optin': (args.label.startswith('v8_') and
                  os.environ.get('FASTEQ_EXPERIMENTAL_MATRIX_TF32X3') == '1'),
              'candidate_matrix_f32_on_f64_mma': (args.label.startswith('v8_') and
                  os.environ.get('FASTEQ_EXPERIMENTAL_MATRIX_F32_F64MMA') == '1'),
              'matrix_compute_note': 'PyTorch TF32 remains disabled. The H100 candidate can promote FP32 inputs exactly to double registers and reuse original FastEq FP64 MMA; outputs remain FP32. FP64 retains native FP64 MMA.',
              'caches': {key: os.environ.get(key) for key in ('TORCH_EXTENSIONS_DIR', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR')},
              'flags': {k: v for k, v in os.environ.items() if k.startswith('FASTEQ_')},
              'before_guard': before}
    try:
        dtype = getattr(torch, args.dtype)
        metadata, leaves, original = operator_case(args.operator, args.atoms*12, dtype)
        fn = original if args.label == 'original_e3nn' else native.segmented_product
        execute = lambda: chain(fn, metadata, leaves, args.mode, requested=(1, 2))
        for _ in range(args.warmup):
            execute()
        torch.cuda.synchronize(); guard.check()
        from fasteq.cuda.jit import runtime
        binary = Path(runtime().__file__).resolve()
        report['loaded_runtime_binary'] = {'path': str(binary), 'sha256': sha(binary)}
        if args.snapshot:
            parameter_slot = 1 if args.mode == 'backward' else 3 if args.mode == 'feature_higher' else -1
            torch.save({'values': logical_snapshot(execute(), parameter_slot, torch),
                        'parameter_slot': parameter_slot}, args.output.with_suffix('.pt'))
            report['snapshot_path'] = str(args.output.with_suffix('.pt'))
        if args.command == 'profile':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                execute()
                torch.cuda.synchronize()
            trace = args.output.with_suffix('.trace.json')
            profile.export_chrome_trace(str(trace))
            report.update(measurement_kind='instrumented_profile_not_normal_latency',
                          trace=str(trace), trace_sha256=sha(trace), profile=parse_trace(trace))
            if args.label.startswith('v8_'):
                assert not report['profile']['forbidden_library_kernel_names'], report['profile']
                assert report['profile']['torch_matmul_cpu_ops'] == 0, report['profile']
                assert report['profile']['kernel_count'] > 0
                report['candidate_no_library_matmul_evidence'] = True
        else:
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            before_allocated = torch.cuda.memory_allocated()
            guard.check()
            begin = time.perf_counter(); start.record()
            for _ in range(args.iterations):
                execute()
            end.record(); end.synchronize()
            wall = (time.perf_counter() - begin)*1000/args.iterations
            event = start.elapsed_time(end)/args.iterations
            guard.check()
            report.update(measurement_kind='normal_eager_no_profiler', wall_ms=wall, event_ms=event,
                          iterations=args.iterations,
                          memory={'before_allocated_bytes': before_allocated,
                                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        verify_hashes(hashes)
        report.update(status='ok', after_guard=guard.check(), metadata=metadata)
    except Exception as error:
        report.update(status='error', error={'type': type(error).__name__, 'message': str(error),
                                           'traceback': traceback.format_exc()})
        raise
    finally:
        write_json(args.output, report)


def launch_worker(args, label, mode, round_index, directory):
    source = NAIVE if label == 'v7_naive' else CANDIDATE
    env = dict(os.environ)
    env['PYTHONPATH'] = str(source)
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    cache = ROOT / 'cache/v8_uvw_validation' / ('v7_naive' if label == 'v7_naive' else 'v8_native')
    env['TORCH_EXTENSIONS_DIR'] = str(cache / 'torch_extensions')
    env['XDG_CACHE_HOME'] = str(cache)
    env['TRITON_CACHE_DIR'] = str(cache / 'triton')
    path = directory / f'round{round_index+1:02d}_{label}.json'
    command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--label', label,
               '--operator', args.operator, '--mode', mode, '--dtype', args.dtype,
               '--gpu-uuid', args.gpu_uuid,
               '--worker-cpu', str(args.worker_cpu),
               '--atoms', str(args.atoms), '--warmup', str(args.warmup), '--iterations', str(args.iterations),
               '--output', str(path)]
    if round_index == 0:
        command.append('--snapshot')
    with path.with_suffix('.log').open('w') as log:
        child = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=args.timeout, check=False)
    result = json.loads(path.read_text()) if path.exists() else {}
    if child.returncode or result.get('status') != 'ok':
        raise RuntimeError(f'{label} worker failed ({child.returncode}): {path}; ' + str(result.get('error')))
    return path, result


def precision(first, torch, dtype):
    candidate = torch.load(first['v8_native'].with_suffix('.pt'), weights_only=True)
    records = {}
    for label, path in first.items():
        if label == 'v8_native':
            continue
        other = torch.load(path.with_suffix('.pt'), weights_only=True)
        assert other['parameter_slot'] == candidate['parameter_slot']
        errors = []
        for index, (a, b) in enumerate(zip(candidate['values'], other['values'], strict=True)):
            if a is None or b is None:
                value = b if a is None else a
                assert value is None or not torch.count_nonzero(value['data'])
                continue
            assert a['shape'] == b['shape'] and a['scope'] == b['scope'] and a['finite'] and b['finite']
            difference = a['data'].double() - b['data'].double()
            l2 = difference.norm().item()/max(b['data'].double().norm().item(), 1e-30)
            if index == candidate['parameter_slot']:
                assert l2 <= (2e-4 if dtype == torch.float32 else 2e-7), (label, index, l2)
            else:
                torch.testing.assert_close(a['data'], b['data'], **tolerance(dtype))
            errors.append({'slot': index, 'scope': a['scope'], 'full_finite': True,
                           'max_abs': difference.abs().max().item() if difference.numel() else 0., 'relative_l2': l2})
        records[label] = {'status': 'pass', 'errors': errors}
    return records


def controller(args):
    os.sched_setaffinity(0, {170})
    if args.output.exists():
        raise RuntimeError('Preserve existing output; choose a new whole-attempt directory')
    args.output.mkdir(parents=True)
    observer = ROOT / 'performance/fasteq_optimization/v7/driver/gpu_foreign_watch.py'
    observer_dir = args.output / 'observer'
    command = [sys.executable, str(observer), '--uuid', args.gpu_uuid, '--owner-pid', str(os.getpid()),
               '--directory', str(observer_dir), '--cpu', str(args.observer_cpu), '--interval', '.02']
    report = {'status': 'running', 'operator': args.operator, 'dtype': args.dtype, 'atoms_equivalent': args.atoms,
              'rows': args.atoms*12, 'atoms_scope': 'equivalent_only_no_atom_graph', 'children': [], 'modes': {},
              'protocol': {'rounds': args.rounds, 'warmup': args.warmup, 'iterations': args.iterations,
                           'order': 'Rotating variant order, reversed on odd rounds; fresh process each block.',
                           'timing': 'Normal eager host and CUDA events; excludes import/build/warmup/validation.',
                           'external_observer': {'cpu': args.observer_cpu, 'interval_s': .02, 'source': str(observer), 'sha256': sha(observer)},
                           'observer_overhead': 'Not independently quantified; same active observer throughout.',
                           'detection_limit': '20ms polling plus worker boundaries can miss shorter foreign processes.',
                           'workers_cpu': args.worker_cpu, 'controller_cpu': args.controller_cpu, 'torch_allow_tf32': False,
                           'candidate_matrix_tf32x3_optin': os.environ.get('FASTEQ_EXPERIMENTAL_MATRIX_TF32X3') == '1',
                           'source_isolation': 'Frozen V7 naive in its own child; V8 native in separate private cache.'}}
    labels = ['original_e3nn', 'v7_naive', 'v8_native']
    if args.ablation:
        labels.insert(2, 'v8_unfused')
    if args.fctp_ablation:
        labels.insert(2, 'v8_dense')
    if args.matrix_ablation:
        labels.insert(2, 'v8_native_simt')
    write_json(args.output / 'status.json', report)
    with (args.output / 'observer.log').open('w') as log:
        monitor = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            while not (observer_dir / 'ready').exists():
                if monitor.poll() is not None or time.monotonic()-started > 20:
                    raise RuntimeError('Observer initialization failed')
                time.sleep(.05)
            for mode in args.modes:
                directory = args.output / mode
                directory.mkdir()
                samples = {label: {'wall_ms': [], 'event_ms': []} for label in labels}
                orders, first = [], {}
                for round_index in range(args.rounds):
                    shift = round_index % len(labels)
                    order = labels[shift:] + labels[:shift]
                    if round_index % 2:
                        order = list(reversed(order))
                    orders.append(order)
                    for label in order:
                        path, value = launch_worker(args, label, mode, round_index, directory)
                        if not round_index:
                            first[label] = path
                        samples[label]['wall_ms'].append(value['wall_ms'])
                        samples[label]['event_ms'].append(value['event_ms'])
                        report['children'].append({'mode': mode, 'round': round_index+1, 'label': label,
                                                   'path': str(path), 'sha256': sha(path)})
                        print(json.dumps({'finished': report['children'][-1], 'wall_ms': value['wall_ms']}), flush=True)
                        write_json(args.output / 'status.json', report)
                import torch
                p = precision(first, torch, getattr(torch, args.dtype))
                ratios = {label: {metric: distribution([a/b for a, b in zip(samples[label][metric],
                            samples['v8_native'][metric], strict=True)]) for metric in ('wall_ms', 'event_ms')}
                          for label in labels if label != 'v8_native'}
                report['modes'][mode] = {'samples': samples, 'execution_orders': orders, 'precision': p,
                                        'paired_ratios_to_v8': ratios,
                                        'medians': {label: {k: statistics.median(v) for k, v in values.items()}
                                                    for label, values in samples.items()}}
            report['status'] = 'complete'
        except Exception as error:
            report.update(status='error', error={'type': type(error).__name__, 'message': str(error),
                                               'traceback': traceback.format_exc()})
            raise
        finally:
            (observer_dir / 'stop').touch()
            monitor.wait(timeout=15)
            observation = json.loads((observer_dir / 'observer.json').read_text())
            report['observer'] = observation
            if observation.get('status') != 'complete' or observation.get('foreign_observations'):
                report['status'] = 'rejected_whole_attempt'
            write_json(args.output / 'result.json', report)
            write_json(args.output / 'status.json', report)
    if report['status'] != 'complete':
        raise RuntimeError('Whole normal attempt rejected; preserve raw samples and restart in a new directory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'controller', 'profile'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', default=UUID)
    parser.add_argument('--operator', choices=['tp_uvw', 'tp_nonuniform'], default='tp_uvw')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--atoms', type=int, default=4096)
    parser.add_argument('--mode', choices=['forward', 'backward', 'feature_higher'], default='feature_higher')
    parser.add_argument('--modes', nargs='+', choices=['forward', 'backward', 'feature_higher'],
                        default=['forward', 'backward', 'feature_higher'])
    parser.add_argument('--label', choices=['original_e3nn', 'v7_naive', 'v8_unfused', 'v8_dense', 'v8_native_simt', 'v8_native'], default='v8_native')
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--fctp-ablation', action='store_true')
    parser.add_argument('--matrix-ablation', action='store_true')
    parser.add_argument('--worker-cpu', type=int, default=172)
    parser.add_argument('--observer-cpu', type=int, default=171)
    parser.add_argument('--controller-cpu', type=int, default=170)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    if args.command == 'controller':
        os.sched_setaffinity(0, {args.controller_cpu})
        controller(args)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        worker(args)


if __name__ == '__main__':
    main()
