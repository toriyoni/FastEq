"""CPU-only summary of the two accepted V8 native UVW points."""
import hashlib
import json
from pathlib import Path

from native_uvw_v8_common import ROOT, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ROOT / 'performance/fasteq_optimization/v8/uvw'
    manifest_path = out / 'measured_component_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    frozen = Path(manifest['snapshot_root'])
    for name, digest in manifest['files_sha256'].items():
        assert sha(frozen / name) == digest
    report = {'status': 'complete', 'measurement_scope': 'Only generic scalar C64 UVW, equivalent N4096 / 49152 independent rows, feature higher chain.',
              'feature_chain': 'output → dX/dY with create_graph → gradient loss → all W/X/Y. Not physical coordinates or model force loss.',
              'precision_scope': 'Full shared W gradients; large features sample256 after all-elements finite checks. Separate full fixture QA and physical synthetic loss.',
              'matrix_backend': 'Handwritten FastEq FP32 SIMT / FP64 MMA, TF32x3 prototype disabled; V7 cuBLAS is naive reference only.',
              'component_manifest': str(manifest_path), 'component_manifest_sha256': sha(manifest_path),
              'normal_points': [], 'isolated_profiles': [], 'qa': [],
              'all_children': 0, 'limitations': ['No model end-to-end timing or OOM sweep in this campaign.',
                  'IQR is sample dispersion, not confidence interval.',
                  '20 ms external NVML polling can miss shorter jobs; its overhead was not independently measured.',
                  'Unrelated operators are not frozen by this component manifest; entire checkout release is separate.']}
    for dtype in ('float32', 'float64'):
        path = out / f'normal_uvw_{"f32" if dtype == "float32" else "f64"}_N4096_attempt1/result.json'
        value = json.loads(path.read_text())
        assert value['status'] == 'complete' and value['observer']['foreign_observations'] == 0
        assert value['protocol']['rounds'] == 9
        mode = value['modes']['feature_higher']
        assert all(p['status'] == 'pass' for p in mode['precision'].values())
        candidate_children = []
        for child in value['children']:
            cp = Path(child['path'])
            assert sha(cp) == child['sha256']
            worker = json.loads(cp.read_text())
            assert worker['status'] == 'ok'
            assert worker['candidate_matrix_tf32x3_optin'] is False
            if child['label'].startswith('v8_'):
                source = Path(worker['source_root'])
                for filename, digest in worker['source_sha256'].items():
                    relative = str(Path(filename).relative_to(source))
                    assert manifest['files_sha256'][relative] == digest
                candidate_children.append({'path': str(cp), 'runtime_binary': worker['loaded_runtime_binary']})
        report['normal_points'].append({'dtype': dtype, 'result': str(path), 'sha256': sha(path),
            'rounds': 9, 'medians': mode['medians'], 'paired_ratios_to_v8': mode['paired_ratios_to_v8'],
            'precision': mode['precision'], 'candidate_child_binaries': candidate_children,
            'observer_foreign_observations': 0})
        report['all_children'] += len(value['children'])
        pp = out / f'profile_final_uvw_{"f32" if dtype == "float32" else "f64"}_N4096.json'
        profile = json.loads(pp.read_text())
        rp = pp.with_suffix('.roles.json')
        roles = json.loads(rp.read_text())
        assert profile['status'] == 'ok' and profile['candidate_no_library_matmul_evidence']
        assert not profile['profile']['forbidden_library_kernel_names']
        assert profile['profile']['torch_matmul_cpu_ops'] == 0 and roles['function_identities_restored']
        assert sha(profile['trace']) == profile['trace_sha256']
        report['isolated_profiles'].append({'dtype': dtype, 'result': str(pp), 'sha256': sha(pp),
            'roles_result': str(rp), 'roles_sha256': sha(rp),
            'scope': 'Instrumented diagnostic, separate from normal latency.',
            'kernel_only_total_us': profile['profile']['kernel_only_total_us'],
            'memory_operations_us': profile['profile']['memory_operation_total_us'],
            'kernels': profile['profile']['kernels'], 'roles': roles['grouped'],
            'no_cublas_cutlass_matmul_evidence': True})
    for name in ('qa_initial.json', 'qa_final_native.json'):
        path = out / name
        value = json.loads(path.read_text())
        assert value['status'] == 'pass'
        report['qa'].append({'path': str(path), 'sha256': sha(path), 'passed_cases': value['passed_cases'],
                             'groups': value['groups'], 'scope': value['scope']})
    write_json(out / 'SUMMARY.json', report)
    lines = ['# V8 native UVW validation and performance', '',
             'The handwritten FastEq matrix replacement is accurate, but it remains slower than both the original e3nn and the V7 cuBLAS naive reference in generic C64 UVW. Packing/direct-view and paired shared-memory contraction improve the native FP32 chain by 1.099×.', '',
             'Normal measurements: equivalent N4096 / 49152 independent rows; nine rounds, fresh process per variant block, five warmups, three iterations, separate 20 ms NVML observer. Both complete attempts passed with no observed foreign processes. This feature higher chain is not a physical model force loss.', '',
             '| dtype | Original e3nn ms | V7 naive ms | Native unfused ms | Native fused ms | Original/native paired | Naive/native paired |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for point in report['normal_points']:
        m, ratios = point['medians'], point['paired_ratios_to_v8']
        unfused = f"{m['v8_unfused']['wall_ms']:.4f}" if 'v8_unfused' in m else 'not measured'
        lines.append(f"| {point['dtype']} | {m['original_e3nn']['wall_ms']:.4f} | {m['v7_naive']['wall_ms']:.4f} | {unfused} | {m['v8_native']['wall_ms']:.4f} | {ratios['original_e3nn']['wall_ms']['median']:.6f} | {ratios['v7_naive']['wall_ms']['median']:.6f} |")
    lines += ['', 'Ratios are reference/native: values below 1 mean the native candidate is slower. FP32 fusion-off/native paired ratio is 1.099067 (IQR 1.097296–1.100279). IQR describes dispersion, not a confidence interval.', '',
              'Both isolated candidate profiler traces contain only FastEq matrix/contraction kernels and ordinary PyTorch elementwise/reduction loss operations; no cuBLAS/CUTLASS kernels or aten mm/bmm/addmm/matmul operations execute inside the candidate trace. Kernel-only sums exclude user annotations and separate memory operations. The profile wrapper restores all function identities; normal workers never install these hooks.', '',
              '| Instrumented kernel diagnostic | FP32 | FP64 |', '|---|---:|---:|']
    for key, title in (('kernel_only_total_us', 'Kernel-only sum ms'), ('memory_operations_us', 'Memory operation sum ms')):
        vals = [p[key]/1000 for p in report['isolated_profiles']]
        lines.append(f'| {title} | {vals[0]:.4f} | {vals[1]:.4f} |')
    lines += ['', 'The main remaining costs are the matrix kernels and materialized angular outer products. FP32 matrices account for approximately 71% of kernel time, outer products 19%, scatter contractions 9%; FP64 outer/scatter together account for approximately 40%. These are kernel timings, not NCU bandwidth or occupancy measurements.', '',
              'Precision: full shared-weight gradient comparisons passed. Complete fixture outputs/gradients passed the initial 44 checks and 19 affected retests, with overlap between those runs; these are not 63 unique cases. Final retests include full mixed second/third derivatives, real nonuniform e3nn paths, storage offsets/tails, synthetic physical forces and all five parameter tensors, nondefault stream, live-weight Graph replay, reciprocal input amplitudes, and paired weight cancellation. Existing tolerances were retained.', '',
              'TF32x3 remains disabled in all accepted results. The measured 17-file UVW dependency/driver component is archived with exact hashes in measured_component_manifest.json. The complete checkout release is handled separately. The first concurrently modified profile is retained as an explicitly rejected diagnostic and excluded from this summary.', '',
              '[Protocol and precision details](VALIDATION_PROTOCOL.md) · [Raw summary](SUMMARY.json) · [Measured component manifest](measured_component_manifest.json)', '']
    (out / 'SUMMARY.md').write_text('\n'.join(lines))
    print(json.dumps({'status': report['status'], 'normal_points': len(report['normal_points']),
                      'fresh_children': report['all_children'], 'summary': str(out / 'SUMMARY.md')}))


if __name__ == '__main__':
    main()
