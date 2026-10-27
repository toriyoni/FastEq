"""Actual eager physical model diagnostics, separate from normal timings.

No per-launch source instrumentation or TorchDispatchMode is installed. Python
external-launch guards are used only in warmup; the profile independently checks
CPU dispatcher events, including calls initiated from native C++ plans.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=('tace', 'tece', 'prophet'),
                        default=['tace', 'tece', 'prophet'])
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--atoms', type=int, default=4096)
    parser.add_argument('--worker-cpu', type=int, default=173)
    parser.add_argument('--observer-cpu', type=int, default=172)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve earlier diagnostics; choose a new output directory')
    if args.warmup < 3:
        raise ValueError('Warm every complete physical chain at least three times')
    os.sched_setaffinity(0, {args.worker_cpu})
    args.output.mkdir(parents=True)
    root = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
    observer_directory = args.output / 'observer'
    observer_driver = root / 'performance/fasteq_optimization/v7/driver/gpu_foreign_watch.py'
    uuid = os.environ['CUDA_VISIBLE_DEVICES']
    observer = subprocess.Popen([sys.executable, str(observer_driver), '--uuid', uuid,
        '--owner-pid', str(os.getpid()), '--directory', str(observer_directory),
        '--cpu', str(args.observer_cpu), '--interval', '.02'])
    report = {'status': 'running', 'driver_sha256': sha(__file__), 'records': [],
              'purpose': 'instrumented diagnostics only; not normal latency or speedup',
              'scope': 'prebuilt neighbors; complete coordinate E/F loss.backward, all parameters',
              'guard_scope': 'untimed eager Python alias guards plus independent profiler C++ dispatcher events',
              'matrix_library_scope': 'ordinary scalar MLP ATen/cuBLAS is permitted; native equivariant source ownership is checked separately',
              'flags': {k: v for k, v in os.environ.items() if k.startswith('FASTEQ_')},
              'uuid': uuid, 'worker_cpu': args.worker_cpu}
    started = time.monotonic()
    try:
        deadline = time.monotonic() + 30
        while not (observer_directory / 'ready').exists():
            if observer.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Independent NVML observer failed to start')
            time.sleep(.02)
        import torch
        import fasteq
        from check_native_models_v8 import model_source_hashes
        from native_sh_dispatch_spy_v8 import native_only
        from scaling_model_case import build_case
        from profile_models import summarize
        from native_models_v8_common import ExclusiveGuard, atomic_json
        from benchmark_native_models_v8 import imported_provenance
        candidate = root / 'src/FastEq_v8'
        assert Path(fasteq.__file__).resolve().is_relative_to(candidate)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        guard = ExclusiveGuard(uuid)
        guard.check()
        frozen = model_source_hashes()
        frozen[str(Path(__file__).resolve())] = sha(__file__)
        for filename in ('profile_models.py', 'benchmark_native_models_v8.py',
                         'native_models_v8_common.py', 'check_native_gate_invariants_v8.py'):
            dependency = Path(__file__).with_name(filename)
            frozen[str(dependency.resolve())] = sha(dependency)
        report.update(source_root=str(candidate), actual_import=fasteq.__file__,
                      pre_run_source_sha256=frozen)
        for filename, digest in frozen.items():
            source = Path(filename)
            destination = args.output / 'source_archive' / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            assert sha(destination) == digest
        atomic_json(args.output / 'profile.json', report)
        jit = importlib.import_module('fasteq.cuda.jit')
        for name in args.models:
            case = build_case(name, args.atoms, args.dtype, 'cuda', 'force_loss', variant='v6')
            model = case.models['v6']
            compiled = {}
            old_binary = jit._binary
            def observed_binary(code, options):
                binary = old_binary(code, options)
                digest = hashlib.sha256(code.encode()).hexdigest()
                compiled[digest] = {'source_sha256': digest, 'options': list(options),
                                   'cubin_sha256': hashlib.sha256(binary).hexdigest()}
                artifact = args.output / 'actual_NVRTC'
                artifact.mkdir(exist_ok=True)
                (artifact / (digest + '.cu')).write_text(code)
                (artifact / (digest + '.cubin')).write_bytes(binary)
                return binary
            jit._binary = observed_binary
            try:
                with native_only():
                    for _ in range(args.warmup):
                        output = case.run('v6')
                        assert all(value is None or torch.isfinite(value).all()
                                   for value in output.values())
                        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                                   for parameter in model.parameters())
                        del output
                        case.clear_gradients()
                        torch.cuda.synchronize()
                        guard.check()
            finally:
                jit._binary = old_binary
            resident = set(jit._loaded)
            provenance = imported_provenance(candidate, Path(os.environ['XDG_CACHE_HOME']).resolve())
            marker = f'native_model::{name}::{args.dtype}::physical_force_loss'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                    profile_memory=False, with_stack=False) as profile:
                with torch.profiler.record_function(marker):
                    output = case.run('v6')
                    del output
                torch.cuda.synchronize()
            assert resident == set(jit._loaded), 'Cold NVRTC kernel appeared during profile'
            assert provenance == imported_provenance(candidate, Path(os.environ['XDG_CACHE_HOME']).resolve()), 'Loaded native binary/source changed during profile'
            trace = args.output / f'{name}_{args.dtype}.trace.json'
            profile.export_chrome_trace(str(trace))
            summary = summarize(profile, trace, marker)
            forbidden = [row for row in summary['all_cpu_ops']
                         if row['name'].split('::', 1)[0].lower().startswith('eqx')
                         or 'triton' in row['name'].lower()]
            assert not forbidden, 'External equivariant dispatcher event: ' + repr(forbidden)
            report['records'].append({'model': name, 'dtype': args.dtype, 'atoms': args.atoms,
                'configuration': case.metadata, 'actual_warmup_NVRTC_cubins': compiled,
                'actual_native_binary_and_generated_source_provenance': provenance,
                'forbidden_EQX_Triton_CPU_events': forbidden, 'trace_sha256': sha(trace),
                'resident_NVRTC_before_after_equal': True, **summary})
            atomic_json(args.output / 'profile.json', report)
            print(json.dumps({'profiled': name, 'dtype': args.dtype,
                              'kernel_ms_diagnostic': summary['sum_cuda_kernel_duration_us'] / 1000}), flush=True)
            case.clear_gradients()
            del model, case
            torch.cuda.empty_cache()
            guard.check()
        actual = model_source_hashes()
        actual[str(Path(__file__).resolve())] = sha(__file__)
        for filename in ('profile_models.py', 'benchmark_native_models_v8.py',
                         'native_models_v8_common.py', 'check_native_gate_invariants_v8.py'):
            dependency = Path(__file__).with_name(filename)
            actual[str(dependency.resolve())] = sha(dependency)
        assert actual == frozen, 'Production source changed during diagnostic'
        report.update(status='passed', final_guard=guard.check())
    except Exception as error:
        report.update(status='failed', error={'message': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        observer_directory.mkdir(parents=True, exist_ok=True)
        (observer_directory / 'stop').touch()
        observer.wait(timeout=30)
        observed = json.loads((observer_directory / 'observer.json').read_text())
        report['observer'] = observed
        if observed['status'] != 'complete' or observed['foreign_observations']:
            report.update(status='rejected_foreign_or_observer_failure')
        report['elapsed_s'] = time.monotonic() - started
        temporary = args.output / 'profile.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output / 'profile.json')
    if report['status'] != 'passed':
        raise RuntimeError('Native model diagnostic was not accepted')


if __name__ == '__main__':
    main()
