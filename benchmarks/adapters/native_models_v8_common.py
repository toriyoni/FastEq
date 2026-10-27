"""CPU-only source/protocol helpers for fresh native V8 model comparisons.

The preserved V7 validation/observer helpers are imported read-only. No model,
FastEq, Torch, or CUDA context is imported by this module.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CANDIDATE = ROOT / 'src/FastEq_v8'
REFERENCE = ROOT / 'src/FastEq'
V7_MANIFEST = ROOT / 'performance/fasteq_optimization/v7/stages/rotate/manifest.json'
V7_DRIVER = ROOT / 'performance/fasteq_optimization/v7/driver'
sys.path.insert(0, str(V7_DRIVER))
from pair_common import (ExclusiveGuard, atomic_json, compare, cpu_snapshot,
                         frozen_v6_guard, sha256, source_hashes)

LABELS = ('original', 'v7_naive', 'v8_native')
MODELS = ('tace', 'tece', 'prophet')
MODES = ('energy', 'forces', 'force_loss')
DTYPES = ('float32', 'float64')
PARAMETER_COUNTS = {'tace': 44, 'tece': 57, 'prophet': 81}
BASE_FLAGS = {
    'FASTEQ_BACKEND': 'cuda',
    'FASTEQ_EXPERIMENTAL_LINEAR_SIMT': '1',
    'FASTEQ_EXPERIMENTAL_NATIVE_MODULATION': '1',
    'FASTEQ_UNIFORM1D_FWD_TUNE': '0',
    'FASTEQ_UNIFORM1D_BWD_TUNE': '0',
}
V8_FLAGS = {
    **BASE_FLAGS,
    **{'FASTEQ_EXPERIMENTAL_' + name: '1' for name in (
        'LINEAR_GEMM', 'SEGMENTED_TASK_MASK', 'SEGMENTED_REDUCE',
        'SEGMENTED_GEMM', 'SEGMENTED_PAIR', 'SEGMENTED_NATIVE_FUSION',
        'FCTP_FUSED', 'FCTP_F32_F64MMA', 'MATRIX_F32_F64MMA',
        'UNIFORM_SCALAR_Y', 'UNIFORM_SCALAR_Y_WARP', 'WIGNER_STORE_TILE',
        'LOCAL_SPLIT_VIEW')},
    'FASTEQ_EXPERIMENTAL_MATRIX_TF32X3': '0',
}
BASELINE_NOTES = {
    'tace': 'Original PyTorch/e3nn cgtp, optional TACE accelerators disabled.',
    'tece': 'Original PyTorch/e3nn plus its automatic EQX CUDA Wigner-D route; this baseline is not pure PyTorch.',
    'prophet': 'Original PyTorch/e3nn Prophet with kernel=False.',
}


class ControllerGuard(ExclusiveGuard):
    """Allow only a reaped child PID's short-lived NVML cache entry.

    Register Popen's real PID after wait. An existing /proc entry (including PID
    reuse) is always foreign; unknown/external exited PIDs are not exempt.
    """
    def __init__(self, uuid):
        super().__init__(uuid)
        self.completed_owned = set()

    def register_owned_exit(self, pid):
        if Path(f'/proc/{pid}').exists():
            raise RuntimeError('A supposedly completed worker still has a /proc entry')
        self.completed_owned.add(int(pid))

    def check(self):
        self.checks += 1
        foreign, ignored = [], []
        for process in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
            pid = int(process.pid)
            if pid == os.getpid():
                continue
            if pid in self.completed_owned and not Path(f'/proc/{pid}').exists():
                ignored.append(pid)
            else:
                foreign.append(pid)
        if foreign:
            raise RuntimeError('GPU is shared with foreign compute processes: ' + str(foreign))
        info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {'uuid': self.uuid, 'foreign_pids': foreign, 'checks': self.checks,
                'used_bytes': int(info.used), 'total_bytes': int(info.total),
                'ignored_cached_exited_owned_pids': ignored}


def stats(samples):
    q = statistics.quantiles(samples, n=4, method='inclusive') if len(samples) > 1 else [samples[0]] * 3
    return {'samples': samples, 'median': statistics.median(samples), 'q25': q[0], 'q75': q[2],
            'min': min(samples), 'max': max(samples), 'mean': statistics.mean(samples),
            'uncertainty_scope': 'empirical paired-round spread; not a confidence interval'}


def register_compiled_binaries(registry, worker, child_path):
    """Append new setup artifacts; reject changes to previously observed ones."""
    observed = dict(worker['binary_and_import_provenance_before_timing']['loaded_library_sha256'])
    observed.update({'generated:' + path: digest for path, digest in
                    worker['binary_and_import_provenance_before_timing'].get(
                        'actual_Uniform1D_generated_source_sha256', {}).items()})
    for source, record in worker['actual_loaded_NVRTC_cubins'].items():
        key = 'nvrtc:' + json.dumps([worker['label'], source, record['options']], separators=(',', ':'))
        observed[key] = record['cubin_sha256']
    for identity, digest in observed.items():
        previous = registry.get(identity)
        if previous is not None and previous['sha256'] != digest:
            raise RuntimeError('Previously observed compiled binary changed across children: ' + identity)
        if previous is None:
            registry[identity] = {'sha256': digest, 'first_child': str(child_path)}


def order(labels, index):
    labels = list(labels)
    # Two labels give AB then BA: ABBA across consecutive rounds. Three labels
    # rotate positions and reverse alternate rounds, matching the UVW protocol.
    shift = index % len(labels) if len(labels) > 2 else 0
    result = labels[shift:] + labels[:shift]
    return result if index % 2 == 0 else list(reversed(result))


def child_environment(manifest, label, gpu_uuid):
    selected = manifest['variants'][label]
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith('FASTEQ_') or name in ('TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR'):
            environment.pop(name)
    environment.update(selected['flags'])
    environment['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    environment['PYTHONNOUSERSITE'] = '1'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONPATH'] = os.pathsep.join((selected['source_root'], manifest['driver_directory'],
                                               str(ROOT / 'src/tace'), str(ROOT / 'src/prophet')))
    for key, value in selected['cache'].items():
        environment[key] = value
    environment['MAX_JOBS'] = '4'
    for name in ('EQX', 'OEQ', 'CUE', 'EQT'):
        environment['TACE_USE_' + name] = '0'
    return environment


def shared_hashes():
    result = {}
    for relative in ('src/tace/tace/models', 'src/tace/eqx', 'src/prophet/prophet'):
        path = ROOT / relative
        result.update({str(path / name): digest for name, digest in source_hashes(path).items()})
    return result


def verify_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest['status'] != 'frozen_ready':
        raise RuntimeError('The model source manifest is not frozen_ready')
    for source, expected in manifest['sources'].items():
        if source_hashes(source) != expected:
            raise RuntimeError('Model source/binary drift: ' + source)
    for filename, expected in {**manifest['shared_dependencies_sha256'], **manifest['driver_sha256']}.items():
        if sha256(filename) != expected:
            raise RuntimeError('Model shared dependency / driver drift: ' + filename)
    previous = manifest['v7_reference_manifest']
    if sha256(previous['path']) != previous['sha256']:
        raise RuntimeError('Preserved V7 reference manifest drift')
    if frozen_v6_guard() != manifest['frozen_v6_guard']:
        raise RuntimeError('Preserved V6 guard drift')
    return manifest


def freeze(args):
    directory = args.output.resolve()
    if directory.exists():
        raise RuntimeError('Preserve the existing frozen stage; choose a new directory')
    v7_path = args.v7_manifest.resolve()
    v7 = json.loads(v7_path.read_text())
    naive_source = Path(v7['candidate_source']).resolve()
    if v7['status'] != 'frozen_ready' or source_hashes(naive_source) != v7['sources'][str(naive_source)]:
        raise RuntimeError('V7 reference source differs from its immutable stage')
    naive_flags = {**BASE_FLAGS, **v7['source_runtime'][str(naive_source)]['flags']}
    roots = {'original': args.reference_source.resolve(), 'v7_naive': naive_source,
             'v8_native': args.candidate.resolve()}
    sources = {str(path): source_hashes(path) for path in roots.values()}
    shared = shared_hashes()
    frozen = frozen_v6_guard()
    directory.mkdir(parents=True)
    # Preserve the full catalog (including legacy binary) as pre-run evidence;
    # live selected roots must stay hash-identical throughout every whole attempt.
    archives = {}
    for label, source in roots.items():
        destination = directory / 'source_archive' / label
        for relative in sources[str(source)]:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        if source_hashes(destination) != sources[str(source)]:
            raise RuntimeError('Model source archive copy differs: ' + label)
        archives[label] = str(destination)
    driver_root = directory / 'driver'
    drivers = driver_root / 'benchmarks/adapters'
    drivers.mkdir(parents=True)
    names = ('native_models_v8_common.py', 'benchmark_native_models_v8.py',
             'scaling_model_case.py', 'scaling_geometry.py', 'benchmark_models.py',
             'paired_model_common.py', 'check_native_gate_invariants_v8.py',
             'native_sh_dispatch_spy_v8.py')
    for name in names:
        shutil.copy2(Path(__file__).parent / name, drivers / name)
    spy_target = driver_root / 'tests/adapters/eqx_cuda_spy.py'
    spy_target.parent.mkdir(parents=True)
    shutil.copy2(args.candidate / 'tests/adapters/eqx_cuda_spy.py', spy_target)
    observer = V7_DRIVER / 'gpu_foreign_watch.py'
    helper = V7_DRIVER / 'pair_common.py'
    driver_hashes = {str(p): sha256(p) for p in sorted(driver_root.rglob('*.py'))}
    driver_hashes.update({str(observer): sha256(observer), str(helper): sha256(helper)})
    cache_key = directory.name + '-' + hashlib.sha256(str(directory).encode()).hexdigest()[:10]
    stage_cache = ROOT / 'cache/v8_model_normal' / cache_key
    variants = {}
    for label, source in roots.items():
        cache = stage_cache / label
        variants[label] = {'source_root': str(source), 'flags': (V8_FLAGS if label == 'v8_native'
                         else naive_flags if label == 'v7_naive' else BASE_FLAGS),
            'legacy_factory_label': 'original' if label == 'original' else 'v6',
            'cache': {'TORCH_EXTENSIONS_DIR': str(cache / 'torch_extensions'),
                      'XDG_CACHE_HOME': str(cache), 'TRITON_CACHE_DIR': str(cache / 'triton'),
                      'FASTEQ_JIT_CACHE_DIR': str(cache / 'fasteq'),
                      'FASTEQ_STC_JIT_CACHE_DIR': str(cache / 'fasteq_stc')},
            'role': ('original model implementation' if label == 'original' else
                     'preserved V7 including cuBLAS matrix calls; naive comparison only' if label == 'v7_naive' else
                     'FastEq-owned native CUDA equivariant implementation')}
    manifest = {'status': 'frozen_ready', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'sources': sources, 'source_archives': archives, 'variants': variants,
                'shared_dependencies_sha256': shared, 'driver_directory': str(drivers),
                'driver_sha256': driver_hashes, 'observer': str(observer), 'frozen_v6_guard': frozen,
                'v7_reference_manifest': {'path': str(v7_path), 'sha256': sha256(v7_path)},
                'baseline_notes': BASELINE_NOTES,
                'protocol': {'timing': 'fresh process for every variant/round; eager host + CUDA events',
                  'geometry': 'prebuilt twelve-neighbor disconnected 256-atom motifs in one graph, matching old capacity fixture; not a large-system MD trajectory',
                  'included': ['position clone', 'model geometry computations', 'energy forward',
                      'coordinate force AD when selected', 'force_loss zero_grad and mean(E**2)+mean(F**2).backward()'],
                  'excluded': ['startup/import', 'model construction', 'neighbor/geometry data setup',
                      'compilation', 'warmup', 'snapshots', 'source/foreign-PID checks'],
                  'optimizer_step': False, 'full_allparams_snapshot': 'first round, physical force_loss; all names and complete gradients',
                  'full_outputs_snapshot': 'energy and forces complete, never sample256',
                  'gradient_relative_l2_limits': {'float32': 2e-4, 'float64': 2e-7},
                  'output_tolerances': {'float32': {'atol': 2e-4, 'rtol': 2e-3},
                                        'float64': {'atol': 2e-8, 'rtol': 2e-7}},
                  'torch_allow_tf32': False, 'observer_interval_s': .02,
                  'rejection_rule': 'reject a whole attempt on any foreign PID, source/binary drift, missing child or precision failure; never splice rounds',
                  'observer_limit': '20ms polling plus worker boundary checks can miss shorter interference; overhead not independently quantified',
                  'compiled_binary_scope': 'repository binaries frozen; every actual imported SO/NVRTC cubin and available Uniform1D generated CUDA source recorded and checked within/across children; new role artifacts may append during untimed setup',
                  'cache_scope': 'native runtime/NVRTC and env-aware caches are stage-label private; unchanged original Uniform1D codegen uses its variant-isolated source root directory',
                  'capacity_scope': 'two representative atom counts; no new OOM endpoints claimed'}}
    path = directory / 'manifest.json'
    atomic_json(path, manifest)
    verify_manifest(path)
    print(path, flush=True)
    return path


def representative_points():
    # All modes at the larger FP32 size show how gains propagate through the
    # physical AD chain. A smaller FP64 loss checks dtype/scale without redoing
    # the already completed doubling-to-OOM campaign.
    return ([{'model': m, 'atoms': 4096, 'dtype': 'float32', 'mode': mode}
             for m in MODELS for mode in MODES] +
            [{'model': m, 'atoms': 1024, 'dtype': 'float64', 'mode': 'force_loss'} for m in MODELS])
