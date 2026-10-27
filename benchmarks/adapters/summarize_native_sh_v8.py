"""Summarize preserved native SH runs; this is CPU-only artifact validation."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND = ROOT / 'src/FastEq_v8'
OUT = ROOT / 'performance/fasteq_optimization/v8/sh'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    normals, profiles = [], []
    component = {}
    for dtype in ('float32', 'float64'):
        path = OUT / f'normal_sh_{dtype}_R49152.json'
        row = json.loads(path.read_text())
        assert row['status'] == 'ok' and not row['observer']['foreign_observations']
        assert row['observer']['status'] == 'complete'
        assert row['fasteq_environment']['FASTEQ_EXPERIMENTAL_WIGNER_STORE_TILE'] == '1'
        for absolute, expected in row['source_sha256'].items():
            archive=OUT/'measured_component_source'/Path(absolute).relative_to(CAND)
            assert digest(archive if archive.exists() else absolute) == expected
            assert component.get(absolute, expected) == expected
            component[absolute] = expected
        normals.append({key: row[key] for key in ('dtype', 'rows', 'mode', 'medians',
                        'paired_ratios_to_native', 'metadata', 'fasteq_environment',
                        'runtime_binary', 'precision_original', 'precision_legacy')})
        normals[-1].update(artifact=str(path), artifact_sha256=digest(path),
                          ratio_of_wall_medians={label: row['medians'][label]['wall_ms']/row['medians']['native_cuda']['wall_ms']
                                                 for label in ('original', 'legacy_triton')})
        path = OUT / f'profile_sh_{dtype}_R49152.json'
        row = json.loads(path.read_text())
        assert row['status'] == 'ok' and row['native_only']
        assert row['source_sha256'] == component
        assert digest(row['trace']) == row['trace_sha256']
        profile = row['profile']
        assert not profile['forbidden_library_kernel_names'] and profile['torch_matmul_cpu_ops'] == 0
        assert all('triton' not in k['name'].lower() for k in profile['kernels'])
        own = [k for k in profile['kernels'] if k['name'] == 'run']
        assert len(own) == 1 and own[0]['count'] == 4
        spans = collections.defaultdict(list)
        for event in json.loads(Path(row['trace']).read_text())['traceEvents']:
            if event.get('cat') == 'cpu_op' and ('fasteq' in event['name'] or 'GeneratedBackward' in event['name']):
                spans[event['name']].append(event.get('dur', 0))
        profiles.append({'dtype': dtype, 'artifact': str(path), 'artifact_sha256': digest(path),
                         'kernel_only_total_us': profile['kernel_only_total_us'],
                         'native_scalar_kernels': own,
                         'native_scalar_fraction': own[0]['total_us']/profile['kernel_only_total_us'],
                         'memory_operation_total_us': profile['memory_operation_total_us'],
                         'trace': row['trace'], 'trace_sha256': row['trace_sha256'],
                         'cpu_custom_op_inclusive_spans_us': dict(spans),
                         'cpu_span_scope': 'Instrumented, nested inclusive CPU events; do not add them or infer normal latency from these durations.'})
    archive = OUT / 'measured_component_source'
    for absolute, expected in component.items():
        destination = archive / Path(absolute).relative_to(CAND)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert digest(destination) == expected
        else:
            destination.write_bytes(Path(absolute).read_bytes())
    manifest = {'scope': 'Actual SH recurrence, ScalarProgram CUDA emitter/AD, launcher and measured driver; runtime .so SHA separately recorded. Other runtime operator CUDA units are not SH mathematics dependencies.',
                'files': {str(Path(path).relative_to(CAND)): value for path, value in component.items()}}
    (OUT / 'measured_component_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    qa = json.loads((OUT / 'GPU_QA.json').read_text())
    edges = json.loads((OUT / 'edges_QA.json').read_text())
    limitation = json.loads((OUT / 'edge_axis_limitation.json').read_text())
    summary = {'status': 'verified', 'normal': normals, 'isolated_profiles': profiles,
               'component_manifest': str(OUT / 'measured_component_manifest.json'),
               'existing_qa_reused_without_repeat': {'ordinary': str(OUT / 'GPU_QA.json'),
                   'ordinary_status': qa.get('status'), 'edge_physical_graph': str(OUT / 'edges_QA.json'),
                   'edge_status': edges.get('status'), 'axis_limitation': limitation},
               'scope': '49152 independent rows, SH degrees 0/1/2, normalized component; forward plus feature first and second derivatives. Physical force loss has separate root QA and is not timed.',
               'protocol': '9 same-process rotated/reversed 3-way rounds; 5 warmups, 3 iterations. No profiler/spies in normal blocks, external CPU171 NVML20ms observer, CPU172 worker, GPU1 UUID explicit. Profiles separate; IQR is dispersion, not CI.',
               'interpretation': 'Native CUDA beats original e3nn around 2x, but regresses old FastEq Triton. Only 4 native CUDA run kernels, 13/16us total, versus normal wall around 956us; prioritize ScalarProgram torch.library/recursive AD dispatch and per-launch Python metadata overhead. This is a profile-supported hypothesis, not causal proof or permission to substitute profiler kernel time for normal latency.'}
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['SH native CUDA preserves the original FastEq polynomial recurrence and executes through the existing ScalarProgram CUDA emitter and recursive analytic transpose. Degree [0,1,2], normalized component, 49152 rows, FP32/FP64.', '',
             '|dtype|Original e3nn wall ms|Old FastEq Triton wall ms|Native CUDA wall ms|Paired original/native [IQR]|Paired old/native [IQR]|Ratio of old/native wall medians|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in normals:
        m=row['medians'];a=row['paired_ratios_to_native']['original']['wall_ms'];b=row['paired_ratios_to_native']['legacy_triton']['wall_ms']
        lines.append(f"|{row['dtype']}|{m['original']['wall_ms']:.4f}|{m['legacy_triton']['wall_ms']:.4f}|{m['native_cuda']['wall_ms']:.4f}|{a['median']:.3f} [{a['q25']:.3f}, {a['q75']:.3f}]|{b['median']:.3f} [{b['q25']:.3f}, {b['q75']:.3f}]|{row['ratio_of_wall_medians']['legacy_triton']:.3f}|")
    lines += ['', 'This is a feature-gradient higher microchain, not physical model force-loss timing. Nine rotated/reversed rounds, five warmups and three iterations; all normal points have full original/legacy accuracy checks and zero foreign observations. WIGNER_STORE_TILE=1 is explicit; shape-dependent selection retains direct stores for narrow adjoints. Raw wall/event samples and orders are preserved. IQR is dispersion, not confidence.', '',
              'Relative to original e3nn, paired ratios are approximately 1.96/1.97. Relative to old FastEq Triton, native normal wall medians are slower by 35.3%/22.7%. FP64 old/native paired IQR crosses one, so precise improvement/regression intervals are not established.', '',
              '|dtype|All CUDA kernels ms|ScalarProgram run kernels ms|run count|ScalarProgram kernel share|',
              '|---|---:|---:|---:|---:|']
    for row in profiles:
        own=row['native_scalar_kernels'][0]
        lines.append(f"|{row['dtype']}|{row['kernel_only_total_us']/1000:.5f}|{own['total_us']/1000:.5f}|{own['count']}|{100*row['native_scalar_fraction']:.1f}%|")
    lines += ['', 'Separate native-only traces show only four ScalarProgram CUDA run kernels, totaling 12.768us/15.934us; no Triton, cuBLAS, CUTLASS, ATen matrix multiplication or external EQX CUDA execution. Remaining CUDA kernels implement the common Torch nonlinear loss and reductions. Memory/annotations are separate. These instrumented kernel sums are not normal latency.', '',
              'The current bottleneck hypothesis is host-side torch.library custom-op/recursive AD dispatch and per-launch Python metadata work. Native CUDA math is only 15% of the instrumented GPU total, and normal wall is about 0.956ms. Trace contains nested custom-op/GeneratedBackward CPU events; their inclusive durations must not be added or used as normal timing. A dispatch-only controlled comparison is needed before claiming causality.', '',
              'Reuse existing root ordinary48 and edge6 QA without a full rerun. Ordinary/physical/stream/liveGraph QA passed. The existing FP32 normalized axis [1.01e-12,0,0] radial-derivative cancellation has no stable native/naive-clampedTorch oracle; edge_axis_limitation.json and failed source are retained. No accuracy tolerance or CUDA math was changed to hide that limitation.', '',
              'All SH normal/profile component source SHA hashes and trace SHA hashes were verified and measured sources archived. Runtime binary SHA is recorded per execution. Existing root SH QA has its own execution source guards.', '', f'Raw evidence: {OUT}', '']
    (OUT / 'SUMMARY.md').write_text('\n'.join(lines))
    print('SH 2 normal, 2 profiles, all source and trace hashes verified; no GPU tests repeated.')


if __name__ == '__main__':
    main()
