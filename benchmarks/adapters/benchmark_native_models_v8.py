"""Fresh-process native V8 physical model comparisons, independent of V6/V7 drivers.

Prepare/freeze/plan are CPU-only. Start controller after complete native model QA
and a full source freeze; no old scaling-to-OOM campaign is repeated here.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from native_models_v8_common import (BASELINE_NOTES, CANDIDATE, DTYPES, LABELS, MODELS, MODES,
    PARAMETER_COUNTS, REFERENCE, ROOT, V7_MANIFEST, ControllerGuard, ExclusiveGuard, atomic_json,
    child_environment, compare, cpu_snapshot, freeze, order, representative_points,
    register_compiled_binaries, sha256, stats, verify_manifest)


def imported_provenance(source, cache):
    modules = {}
    eqx_root = (ROOT / 'src/tace/eqx').resolve()
    for name, module in sys.modules.items():
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if (name.startswith(('fasteq', 'eqx', 'uniform1d_')) or path.is_relative_to(eqx_root)
                or (path.suffix == '.so' and
                    (path.is_relative_to(cache) or path.is_relative_to(source)))):
            modules[name] = str(path)
    for name, path in modules.items():
        if (name == 'fasteq' or name.startswith('fasteq.')) and not Path(path).is_relative_to(source):
            raise RuntimeError('FastEq import leaked from another variant: ' + name + ':' + path)
        if name.startswith('fasteq_') and not (Path(path).is_relative_to(cache) or Path(path).is_relative_to(source)):
            raise RuntimeError('FastEq extension leaked from another stage/label cache: ' + path)
        if name.startswith('uniform1d_') and not (Path(path).is_relative_to(source) or Path(path).is_relative_to(cache)):
            raise RuntimeError('Uniform1D extension leaked from another variant: ' + path)
    binary = {path: sha256(path) for path in modules.values() if Path(path).suffix == '.so'}
    # cpp_extension.load can return an anonymous module without sys.modules
    # registration. The process maps identify its actual loaded shared object.
    mapped = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[5].startswith('/'):
            continue
        filename = parts[5]
        deleted = filename.endswith(' (deleted)')
        path = Path(filename.removesuffix(' (deleted)')).resolve()
        if '.so' not in path.name or not (path.is_relative_to(source)
                or path.is_relative_to(cache) or path.is_relative_to(eqx_root)):
            continue
        if deleted or not path.is_file():
            raise RuntimeError('A loaded native library was replaced/deleted: ' + filename)
        binary[str(path)] = sha256(path)
        mapped.append(str(path))
    generated_sources = {}
    for filename in binary:
        path = Path(filename)
        if 'uniform1d_jit_codegen' in path.parts:
            generated_sources.update({str(p.resolve()): sha256(p)
                                      for p in path.parent.glob('*.cu')})
    # FastEq's legacy CUDA library is loaded with torch.ops rather than as an
    # importable extension module. Record actual loaded native libraries too.
    import torch
    for path in torch.ops.loaded_libraries:
        path = Path(path).resolve()
        if path.is_relative_to(source) or 'eqx' in str(path):
            binary[str(path)] = sha256(path)
    return {'module_files': modules, 'loaded_library_sha256': binary,
            'actual_mapped_library_paths': sorted(set(mapped)),
            'actual_Uniform1D_generated_source_sha256': generated_sources,
            'cache_scope': 'native runtime/NVRTC use stage-label private caches; original Uniform1D uses its selected source root/codegen directory, isolated by variant source and checked within/across children'}


def worker(args):
    report = {'status': 'running', 'stage': 'startup', 'model': args.model, 'dtype': args.dtype,
              'atoms': args.atoms, 'edges': args.atoms * 12, 'mode': args.mode, 'label': args.label,
              'manifest': str(args.manifest.resolve()), 'manifest_sha256': sha256(args.manifest),
              'worker_driver_sha256': sha256(__file__), 'measurement_kind': 'normal_eager_no_profiler'}
    started = time.monotonic()
    try:
        manifest = verify_manifest(args.manifest)
        selected = manifest['variants'][args.label]
        desired = child_environment(manifest, args.label, args.gpu_uuid)
        for name in list(os.environ):
            if name.startswith('FASTEQ_') or name in ('TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR'):
                os.environ.pop(name)
        os.environ.update(desired)
        os.sched_setaffinity(0, {args.worker_cpu})
        guard = ExclusiveGuard(args.gpu_uuid)
        report['before_guard'] = guard.check()
        source = Path(selected['source_root']).resolve()
        if 'fasteq' in sys.modules:
            raise RuntimeError('FastEq imported before variant source selection')
        sys.path[:0] = [str(source), manifest['driver_directory'], str(ROOT / 'src/tace'), str(ROOT / 'src/prophet')]
        import torch
        import fasteq
        assert Path(fasteq.__file__).resolve() == source / 'fasteq/__init__.py'
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        uuid = uuid if uuid.startswith('GPU-') else 'GPU-' + uuid
        assert uuid == args.gpu_uuid
        report.update(source_root=str(source), actual_import=fasteq.__file__,
          baseline_note=BASELINE_NOTES[args.model], runtime={'torch': torch.__version__, 'cuda': torch.version.cuda,
          'device': torch.cuda.get_device_name(0), 'pid': os.getpid(), 'CPU_affinity': sorted(os.sched_getaffinity(0)),
          'torch_allow_tf32': False, 'float32_matmul_precision': 'highest'},
          flags={k: v for k, v in os.environ.items() if k.startswith('FASTEQ_')}, caches=selected['cache'])
        from scaling_model_case import build_case
        def stage(value):
            report['stage'] = value
        case = build_case(args.model, args.atoms, args.dtype, 'cuda', args.mode,
                          variant=selected['legacy_factory_label'], stage_callback=stage)
        alias = selected['legacy_factory_label']
        model = case.models[alias]
        named = list(model.named_parameters())
        assert len(named) == PARAMETER_COUNTS[args.model]
        report['configuration'] = case.metadata
        report['configuration']['measured_label'] = args.label
        report['configuration']['legacy_factory_variant_label'] = alias
        invoke = lambda: case.run(alias)
        clear = lambda: model.zero_grad(set_to_none=True)
        # Capture actual NVRTC binary hashes during untimed setup only. The
        # wrapper is removed before measurement and adds no timed thread/probe.
        cuda_jit = importlib.import_module('fasteq.cuda.jit')
        old_binary = cuda_jit._binary
        compiled = {}
        def observed_binary(source_text, options):
            binary = old_binary(source_text, options)
            key = hashlib.sha256(source_text.encode()).hexdigest()
            compiled[key] = {'source_sha256': key, 'options': list(options),
                             'cubin_sha256': hashlib.sha256(binary).hexdigest()}
            return binary
        cuda_jit._binary = observed_binary
        if args.label == 'v8_native':
            from native_sh_dispatch_spy_v8 import native_only
            scope = native_only
        else:
            scope = nullcontext
        try:
            report['stage'] = 'warmup'
            with scope():
                for _ in range(args.warmup):
                    guard.check()
                    output = invoke()
                    del output
                    clear()
                    torch.cuda.synchronize()
                    guard.check()
                if args.snapshot:
                    report['stage'] = 'full_validation_snapshot'
                    output = invoke()
                    parameters = named if args.mode == 'force_loss' else []
                    snapshot = cpu_snapshot(output, parameters, True, torch)
                    snapshot_path = args.output.with_suffix('.pt')
                    torch.save(snapshot, snapshot_path)
                    report['snapshot'] = {'path': str(snapshot_path), 'sha256': sha256(snapshot_path),
                          'full_energy_forces': True, 'full_named_parameter_gradients': args.mode == 'force_loss',
                          'named_parameter_count': len(parameters), 'complete_finiteness_checked': True}
                    del output, snapshot
                    clear()
        finally:
            cuda_jit._binary = old_binary
        report['candidate_no_EQX_Triton_evidence'] = ('complete untimed real eager physical execution under compiler/launcher/alias guards without TorchDispatchMode; spies restored before timing; C++-initiated EQX dispatcher calls require separate profiler evidence'
                                                     if args.label == 'v8_native' else 'not required for reference/naive')
        report['actual_loaded_NVRTC_cubins'] = compiled
        before_kernels = sorted((str(key[0]), hashlib.sha256(key[1].encode()).hexdigest(), key[2])
                                for key in cuda_jit._loaded)
        report['resident_NVRTC_before_timing'] = before_kernels
        before_binary = imported_provenance(source, Path(selected['cache']['XDG_CACHE_HOME']).resolve())
        report['binary_and_import_provenance_before_timing'] = before_binary
        verify_manifest(args.manifest)
        gc.collect()
        clear()
        torch.cuda.synchronize()
        guard.check()
        report['stage'] = 'timing'
        base_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        first = torch.cuda.Event(enable_timing=True)
        last = torch.cuda.Event(enable_timing=True)
        # Torch creates CUDA events lazily on first record. Initialize them
        # outside the host timer so every fresh worker excludes this setup.
        first.record()
        last.record()
        last.synchronize()
        guard.check()
        begin = time.perf_counter()
        first.record()
        for _ in range(args.iterations):
            output = invoke()
            del output
        last.record()
        last.synchronize()
        report['measurement'] = {'wall_ms': (time.perf_counter() - begin) * 1000 / args.iterations,
                    'event_ms': first.elapsed_time(last) / args.iterations, 'iterations': args.iterations,
                    'warmup': args.warmup, 'CUDA_event_initialization_excluded': True,
                    'memory': {'before_allocated_bytes': base_allocated,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                      'incremental_peak_allocated_bytes': torch.cuda.max_memory_allocated() - base_allocated}}
        clear()
        guard.check()
        after_binary = imported_provenance(source, Path(selected['cache']['XDG_CACHE_HOME']).resolve())
        assert before_binary == after_binary, 'Loaded native binary/import set changed during timed block'
        after_kernels = sorted((str(key[0]), hashlib.sha256(key[1].encode()).hexdigest(), key[2])
                               for key in cuda_jit._loaded)
        assert before_kernels == after_kernels, 'New NVRTC kernels were prepared inside the timed block'
        report['resident_NVRTC_before_after_equal'] = True
        report['untimed_hooks_restored_before_timing'] = cuda_jit._binary is old_binary
        verify_manifest(args.manifest)
        report.update(status='ok', stage='complete', after_guard=guard.check())
    except Exception as error:
        report.update(status='error', error={'type': type(error).__name__, 'message': str(error),
                                            'traceback': traceback.format_exc()})
    report['process_elapsed_s'] = time.monotonic() - started
    atomic_json(args.output, report)
    print(json.dumps({k: report.get(k) for k in ('status', 'stage', 'model', 'dtype', 'mode', 'process_elapsed_s')}), flush=True)
    return 0 if report['status'] == 'ok' else 2


def launch(args, manifest, label, mode, rid, directory, guard):
    output = directory / f'round{rid+1:02d}_{label}.json'
    command = [sys.executable, str(Path(manifest['driver_directory']) / Path(__file__).name), 'worker',
               '--manifest', str(args.manifest.resolve()), '--model', args.model, '--atoms', str(args.atoms),
               '--dtype', args.dtype, '--label', label, '--mode', mode, '--gpu-uuid', args.gpu_uuid,
               '--worker-cpu', str(args.worker_cpu), '--warmup', str(args.warmup),
               '--iterations', str(args.iterations), '--output', str(output)]
    if rid == 0:
        command.append('--snapshot')
    environment = child_environment(manifest, label, args.gpu_uuid)
    with output.with_suffix('.log').open('w') as log:
        with subprocess.Popen(command, cwd=manifest['variants'][label]['source_root'], env=environment,
                              stdout=log, stderr=subprocess.STDOUT) as child:
            try:
                returncode = child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                guard.register_owned_exit(child.pid)
                raise
            guard.register_owned_exit(child.pid)
    value = json.loads(output.read_text()) if output.exists() else {}
    if returncode or value.get('status') != 'ok':
        raise RuntimeError(f'Model worker failed: {output}, returncode={returncode}, error={value.get("error")}')
    if value['manifest_sha256'] != sha256(args.manifest):
        raise RuntimeError('Worker manifest SHA mismatch')
    return output, value


def precision(first, dtype, mode):
    import torch
    original_label = 'original' if 'original' in first else 'v7_naive'
    original = torch.load(first[original_label].with_suffix('.pt'), map_location='cpu', weights_only=True)
    records = {}
    for label, path in first.items():
        if label == original_label:
            continue
        candidate = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        records[label] = compare(original, candidate, dtype, torch)
        assert records[label]['full_output_comparison']
        if mode == 'force_loss':
            assert len(records[label]['all_parameter_gradient_names']) == PARAMETER_COUNTS[json.loads(path.read_text())['model']]
        else:
            records[label]['parameter_gradient_relative_l2'] = None
            records[label]['parameter_gradient_scope'] = 'not computed by this energy/forces mode'
    return {'reference_label': original_label, 'comparisons': records,
            'snapshot_sha256': {label: sha256(path.with_suffix('.pt')) for label, path in first.items()}}


def controller(args):
    os.sched_setaffinity(0, {args.controller_cpu})
    manifest = verify_manifest(args.manifest)
    if args.output.exists():
        raise RuntimeError('Preserve prior whole-attempt output; choose a new directory for retry')
    args.output.mkdir(parents=True)
    observer_dir = args.output / 'observer'
    monitor_command = [sys.executable, manifest['observer'], '--uuid', args.gpu_uuid,
                       '--owner-pid', str(os.getpid()), '--directory', str(observer_dir),
                       '--cpu', str(args.observer_cpu), '--interval', '.02']
    report = {'status': 'running', 'model': args.model, 'dtype': args.dtype, 'atoms': args.atoms,
      'edges': args.atoms * 12, 'baseline_note': BASELINE_NOTES[args.model],
      'manifest': str(args.manifest.resolve()), 'manifest_sha256': sha256(args.manifest),
      'driver_sha256': manifest['driver_sha256'], 'children': [], 'modes': {},
      'compiled_binary_registry': {},
      'protocol': {**manifest['protocol'], 'rounds': args.rounds, 'warmup': args.warmup,
          'iterations': args.iterations, 'labels': args.labels, 'workers_cpu': args.worker_cpu,
          'observer_cpu': args.observer_cpu, 'controller_cpu': args.controller_cpu,
          'order': 'AB/BA for two labels (ABBA across adjacent rounds); rotating positions and reverse alternate rounds for three labels',
          'snapshots': 'full outputs + all physical-loss parameter gradients in round1, outside timing',
          'source_isolation': 'one variant per fresh child; stage/label-private native runtime/NVRTC caches and variant-source-isolated original Uniform1D codegen; no V8 FastEq imports in V7 child'}}
    atomic_json(args.output / 'status.json', report)
    monitor = None
    try:
        with (args.output / 'observer.log').open('w') as log:
            monitor = subprocess.Popen(monitor_command, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while not (observer_dir / 'ready').exists():
            if monitor.poll() is not None or time.monotonic() - started > 20:
                raise RuntimeError('External 20ms observer did not initialize')
            time.sleep(.05)
        guard = ControllerGuard(args.gpu_uuid)
        guard.check()
        for mode in args.modes:
            directory = args.output / mode
            directory.mkdir()
            samples = {label: {'wall_ms': [], 'event_ms': []} for label in args.labels}
            first = {}
            orders = []
            identities = {}
            for rid in range(args.rounds):
                chosen_order = order(args.labels, rid)
                orders.append(chosen_order)
                for label in chosen_order:
                    guard.check()
                    verify_manifest(args.manifest)
                    path, value = launch(args, manifest, label, mode, rid, directory, guard)
                    register_compiled_binaries(report['compiled_binary_registry'], value, path)
                    guard.check()
                    verify_manifest(args.manifest)
                    identity = {k: value['configuration'][k] for k in ('state_sha256', 'named_parameter_names', 'geometry', 'model_config')}
                    if identities:
                        assert identity == next(iter(identities.values())), 'Variants did not receive identical weights/geometry/configuration'
                    identities[label] = identity
                    if rid == 0:
                        first[label] = path
                    for metric in ('wall_ms', 'event_ms'):
                        samples[label][metric].append(value['measurement'][metric])
                    child = {'mode': mode, 'round': rid+1, 'label': label, 'path': str(path), 'sha256': sha256(path),
                             'wall_ms': value['measurement']['wall_ms'], 'event_ms': value['measurement']['event_ms']}
                    report['children'].append(child)
                    print(json.dumps({'finished': child}), flush=True)
                    atomic_json(args.output / 'status.json', report)
                if rid == 0:
                    p = precision(first, args.dtype, mode)
                    report['modes'][mode] = {'precision': p}
                    atomic_json(args.output / 'status.json', report)
            ratios = {label: {metric: stats([a / b for a, b in zip(samples[label][metric], samples['v8_native'][metric], strict=True)])
                       for metric in ('wall_ms', 'event_ms')} for label in args.labels if label != 'v8_native'}
            report['modes'][mode] = {'samples': samples, 'orders': orders, 'precision': p,
                        'paired_ratios_to_v8': ratios,
                        'latency': {label: {metric: stats(values) for metric, values in measured.items()}
                                    for label, measured in samples.items()}}
            atomic_json(args.output / 'status.json', report)
        verify_manifest(args.manifest)
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='error', error={'type': type(error).__name__, 'message': str(error),
                                            'traceback': traceback.format_exc()})
    finally:
        if monitor is not None:
            (observer_dir / 'stop').touch()
            monitor.wait(timeout=15)
        observer_path = observer_dir / 'observer.json'
        observation = json.loads(observer_path.read_text()) if observer_path.exists() else {}
        report['observer'] = observation
        report['observer_report_sha256'] = sha256(observer_path) if observer_path.exists() else None
        if (observation.get('status') != 'complete' or observation.get('foreign_observations')
                or observation.get('observer_driver_sha256', {}).get(manifest['observer']) != manifest['driver_sha256'][manifest['observer']]):
            report['status'] = 'rejected_whole_attempt'
        atomic_json(args.output / 'result.json', report)
        atomic_json(args.output / 'status.json', report)
    if report['status'] != 'complete':
        print(json.dumps({'status': report['status'], 'error': report.get('error'), 'preserve_attempt': str(args.output)}), flush=True)
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'plan', 'controller', 'worker'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--reference-source', type=Path, default=REFERENCE)
    parser.add_argument('--v7-manifest', type=Path, default=V7_MANIFEST)
    parser.add_argument('--model', choices=MODELS, default='tece')
    parser.add_argument('--dtype', choices=DTYPES, default='float32')
    parser.add_argument('--atoms', type=int, choices=(1024, 4096), default=4096)
    parser.add_argument('--mode', choices=MODES, default='force_loss')
    parser.add_argument('--modes', choices=MODES, nargs='+', default=list(MODES))
    parser.add_argument('--label', choices=LABELS, default='v8_native')
    parser.add_argument('--labels', choices=LABELS, nargs='+', default=list(LABELS))
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--gpu-uuid', default='GPU-af44a7bc-24c0-45be-df6e-03f4c445c78d')
    parser.add_argument('--worker-cpu', type=int, default=177)
    parser.add_argument('--observer-cpu', type=int, default=178)
    parser.add_argument('--controller-cpu', type=int, default=179)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    if args.command == 'plan':
        print(json.dumps({'GPU_work_started': False, 'representative_points': representative_points(),
              'full_supported_grid_count': len(MODELS) * len(DTYPES) * len(MODES) * 2,
              'supported_atoms': [1024, 4096], 'supported_labels': list(LABELS),
              'baseline_notes': BASELINE_NOTES,
              'no_new_OOM_grid': True}, indent=2))
        return 0
    if args.output is None:
        parser.error('--output is required')
    if args.command == 'freeze':
        freeze(args)
        return 0
    if args.manifest is None:
        parser.error('--manifest is required')
    if min(args.rounds, args.warmup, args.iterations) < 1:
        parser.error('rounds/warmup/iterations must be positive')
    if args.command == 'controller':
        if len(args.labels) != len(set(args.labels)) or 'v8_native' not in args.labels or len(args.labels) < 2:
            parser.error('controller needs unique labels including v8_native and a baseline')
        if len({args.worker_cpu, args.observer_cpu, args.controller_cpu}) != 3:
            parser.error('worker/observer/controller CPU cores must be distinct')
        return controller(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return worker(args)


if __name__ == '__main__':
    raise SystemExit(main())
