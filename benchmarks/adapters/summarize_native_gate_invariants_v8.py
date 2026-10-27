"""Reconcile stored Gate/Invariant normal/profile evidence without CUDA work."""
import hashlib
import json
from pathlib import Path


ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
OUT = ROOT / 'performance/fasteq_optimization/v8/gate_invariants'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = json.loads((OUT / 'measured_component_manifest.json').read_text())
    archived = OUT / 'measured_component_source'
    for relative, expected in manifest['files'].items():
        assert digest(archived / relative) == expected
    normal = []
    for path in sorted(OUT.glob('normal_*.json')):
        result = json.loads(path.read_text())
        assert result['status'] == 'ok'
        assert result['observer']['status'] == 'complete'
        assert not result['observer']['foreign_observations']
        for absolute, expected in result['source_sha256'].items():
            relative = str(Path(absolute).relative_to(ROOT / 'src/FastEq_v8'))
            assert manifest['files'][relative] == expected
        item = {key: result[key] for key in ('operator', 'dtype', 'rows', 'mode', 'medians',
                                           'paired_ratios_to_native', 'metadata', 'runtime_binary')}
        item['artifact'] = str(path)
        item['artifact_sha256'] = digest(path)
        item['ratio_of_wall_medians'] = {
            label: result['medians'][label]['wall_ms'] / result['medians']['native_cuda']['wall_ms']
            for label in ('original', 'legacy_triton')}
        normal.append(item)
    assert len(normal) == 8
    profiles = []
    for path in sorted(OUT.glob('profile_*.json')):
        if '.trace.' in path.name:
            continue
        result = json.loads(path.read_text())
        assert result['status'] == 'ok' and result['native_only']
        profile = result['profile']
        assert not profile['forbidden_library_kernel_names']
        assert profile['torch_matmul_cpu_ops'] == 0
        assert all('triton' not in row['name'].lower() for row in profile['kernels'])
        assert digest(result['trace']) == result['trace_sha256']
        for absolute, expected in result['source_sha256'].items():
            relative = str(Path(absolute).relative_to(ROOT / 'src/FastEq_v8'))
            assert manifest['files'][relative] == expected
        native = [row for row in profile['kernels'] if row['name'].startswith('native_')]
        total = sum(row['total_us'] for row in native)
        profiles.append({'artifact': str(path), 'artifact_sha256': digest(path),
                         'operator': result['operator'], 'dtype': result['dtype'],
                         'rows': result['rows'], 'kernel_only_total_us': profile['kernel_only_total_us'],
                         'native_operator_us': total,
                         'native_operator_fraction': total / profile['kernel_only_total_us'],
                         'native_kernel_names': native,
                         'memory_operation_total_us': profile['memory_operation_total_us'],
                         'trace': result['trace'], 'trace_sha256': result['trace_sha256']})
    assert len(profiles) == 4
    qa = json.loads((OUT / 'qa_reconciled_summary.json').read_text())
    smoke = json.loads((OUT / 'integration_smoke.json').read_text())
    assert qa['status'] == 'pass' and qa['unique_cases'] == 44
    assert smoke['status'] == 'pass' and len(smoke['records']) == 4
    summary = {'status': 'verified', 'normal': normal, 'isolated_profiles': profiles,
               'qa_unique_cases': 44, 'integration_smoke_cases': 4,
               'qa_reused_ordinary_cases': 42, 'qa_physical_graph_retest': 2,
               'qa_raw_initial_failure_preserved': qa['raw_initial_status'],
               'qa_graph_failure_root_cause': qa['graph_failure_root_cause'],
               'invariant_original_threshold_cpu_postcheck': qa['original_invariant_threshold_postcheck'],
               'component_manifest': str(OUT / 'measured_component_manifest.json'),
               'scope': 'Feature-gradient higher microchain. Physical coordinates/all-parameter force loss only in QA, not timed. No end-to-end/OOM rerun.',
               'protocol': 'Each point uses one process, 9 rotated/reversed 3-way rounds, 5 warmups, 3 iterations; no profiler/spies in timed blocks. External CPU171 NVML20ms observer. Interquartile ranges describe dispersion, not confidence.',
               'limitations': ['Prophet FP64 native wall median 1.488ms vs legacy 1.451ms; paired legacy/native IQR spans one. No robust gain over legacy.',
                               'Invariant FP64 paired legacy/native IQR also spans one. Do not imply consistent legacy speedup.',
                               'Shared runtime binary hashes recorded per execution; component snapshot scopes operator math, metadata and launcher, not unrelated CUDA modules.']}
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['Native Gate and Invariant CUDA are integrated into actual GatePlan / FFNPlan.norm. All 44 unique QA cases and 4 integration smoke cases passed. The initial CUDA Graph harness failure and the corrected physical/Graph-only retest remain in the raw artifacts; 42 ordinary cases were reused without repeating.', '',
             'Original FastEq activation and derivative formulas, deterministic source/gate fan-in, and per-copy sumsq/angular-width invariants are unchanged. Execution uses the existing FastEq NVRTC launcher and original subgroup warp reduction. Candidate QA/profile forbids Triton dispatch and external EQX execution.', '',
             'All timings below cover forward plus feature first and second derivatives using a common nonlinear loss. They are operator microchains, not model energy/forces/force-loss timings. Physical coordinates and all trainable parameters were validated separately. Independent operator rows: O3/Prophet/Invariant 4096, O2 49152.', '',
             '|Operator|dtype|rows|Original wall ms|Old FastEq Triton wall ms|Native wall ms|Paired original/native|Paired old/native [IQR]|Ratio of old/native wall medians|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in normal:
        m = row['medians']; a = row['paired_ratios_to_native']['original']['wall_ms']
        b = row['paired_ratios_to_native']['legacy_triton']['wall_ms']
        lines.append(f"|{row['operator']}|{row['dtype']}|{row['rows']}|{m['original']['wall_ms']:.4f}|{m['legacy_triton']['wall_ms']:.4f}|{m['native_cuda']['wall_ms']:.4f}|{a['median']:.3f}|{b['median']:.3f} [{b['q25']:.3f}, {b['q75']:.3f}]|{row['ratio_of_wall_medians']['legacy_triton']:.3f}|")
    lines += ['', 'Nine rotated/reversed rounds per point, five warmups and three iterations, same-process protocol. Timed blocks contain no profiler/hooks/spies; CPU172 worker and separate CPU171 NVML20ms observer saw zero foreign observations at all 8 points. Full tensor accuracy against original and legacy is checked outside timing. Event samples and exact orders are retained in each JSON. IQR is dispersion, not confidence. This protocol differs from the earlier fresh-process UVW test.', '',
              'Prophet FP64 has no established improvement over old FastEq: native wall median is 1.4882ms vs old 1.4515ms, despite paired ratio median 1.090. Paired IQR [0.780, 1.414] spans one. Invariant FP64 old/native paired IQR also spans one. These results must not be described as uniform wins over legacy.', '',
              '|Native-only profile|dtype|rows|All CUDA kernels ms|Native operator kernels ms|Native operator share|',
              '|---|---|---:|---:|---:|---:|']
    for row in profiles:
        lines.append(f"|{row['operator']}|{row['dtype']}|{row['rows']}|{row['kernel_only_total_us']/1000:.4f}|{row['native_operator_us']/1000:.4f}|{100*row['native_operator_fraction']:.1f}%|")
    lines += ['', 'Isolated profiles show native_gate_forward/backward/double or native_invariant_forward/backward/double; no Triton, cuBLAS, CUTLASS, ATen matrix multiplication, or external core execution. Chrome cat=kernel time excludes CUDA memory and annotations, and is not normal latency. The common Torch loss/reductions account for 60–75% of these profile totals, limiting this microchain’s visible operator speedup. No occupancy/bandwidth conclusion is claimed without counters.', '',
              'All 14 measured component files are archived with SHA256; all 8 normal and 4 profile source guards match the archived files. Each execution also records the loaded runtime .so SHA256. QA math SHA hashes are unchanged through integration. Stricter original Invariant threshold was postchecked from all 62 stored invariant comparison slots without a GPU repeat.', '',
              f'Raw evidence: {OUT}', '']
    (OUT / 'SUMMARY.md').write_text('\n'.join(lines))
    print('Verified 8 normal points, 4 profiles, 14 archived component files, 44 unique QA and 4 integration smoke cases.')


if __name__ == '__main__':
    main()
