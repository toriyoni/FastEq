"""CPU-only final normal Frame/Rotary performance evidence reconciliation."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
OUT=ROOT/'performance/fasteq_optimization/v8/rotation'
ARCHIVE=OUT/'normal_measured_component_source'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows=[];component={}
    for operator in ('rotate','rotary'):
        for dtype in ('float32','float64'):
            path=OUT/f'normal_{operator}_{dtype}_N4096.json'
            row=json.loads(path.read_text())
            assert row['status']=='ok'
            assert row['observer']['status']=='complete'and not row['observer']['foreign_observations']
            assert row['rows']==49152 and row['mode']=='feature_higher'
            for source,expected in row['source_sha256'].items():
                destination=ARCHIVE/Path(source).relative_to(ROOT)
                assert digest(destination if destination.exists()else source)==expected
                assert component.get(source,expected)==expected;component[source]=expected
            record={key:row[key]for key in ('operator','dtype','rows','mode','medians','paired_ratios_to_native',
                'factory_metadata','actual_shape','runtime_binary','precision','fasteq_environment')}
            record.update(artifact=str(path),artifact_sha256=digest(path),
                ratio_of_wall_medians={label:row['medians'][label]['wall_ms']/row['medians']['v8_native']['wall_ms']for label in ('original','v7_naive')})
            rows.append(record)
    for source,expected in component.items():
        destination=ARCHIVE/Path(source).relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            assert digest(source)==expected;destination.write_bytes(Path(source).read_bytes())
        assert digest(destination)==expected
    historical=json.loads((OUT/'summary.json').read_text())
    assert historical['qa']['operator_cases']==90 and historical['qa']['physical_chains']==4 and historical['qa']['graphs']==2
    summary={'status':'verified','normal_points':rows,'component_source_sha256':component,
        'component_archive':str(ARCHIVE),'normal_performance_measured':True,
        'prior_operator_QA_90_physical_4_graph_2_reused_without_repeat':str(OUT/'summary.json'),
        'prior_native_isolated_profiles_reused':historical['profiles'],
        'scope':'Actual scaling factories. Frame asymmetric TECE C64 means paired C128 channels. Rotary C64 includes common original radial projection/trig/bias/temperature plus feature higher and all wrapper-parameter gradients. No graph/model energy, physical force-loss latency, OOM or full QA rerun.',
        'protocol':'Each point: one process, 9 rotated/reversed three-way rounds, 5 warmups, 3 iterations. CPU172 worker, independent CPU171 NVML20ms observer, GPU explicit UUID, normal wall/event timing with no hooks/spies/profiler. IQR is dispersion, not confidence.',
        'V7_scope':'Only frozen V7 rotation.py mathematical CUDA/AD source is imported under a separate module name and compiled/launched with the same current FastEq runtime. This controls rotation arithmetic and excludes unrelated V7 backends/runtime build differences. V7 code adapted from EQX is reference-only, never production fallback.',
        'original_scope':'Frame bmm and Rotary Convolution.rotary_score fused=False Torch expressions. Original TECE o2 modules are EQX interfaces; common radial_proj uses EQX o2.Linear implemented through Torch matmul/bias. Wigner auto may use EQX CUDA but its generation is excluded here. No claim of wholly pure-PyTorch TECE model.',
        'limitations':['Shared runtime .so execution SHA is recorded separately; archived component guards include actual rotation header include closure, factory and wrapper dependency code.',
                       'Normal independent rows do not substitute for real full-model atom graph timings.',
                       'Observer overhead is not separately measured and shorter foreign GPU jobs may escape its 20ms polling.']}
    (OUT/'NORMAL_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['Frame improves both original and V7 comparisons in FP32/FP64. Rotary improves the original Torch wrapper but regresses against V7: native wall medians are 17.2% slower in FP32 and 7.4% slower in FP64, with both paired V7/native IQRs below one. The existing 90 operator QA, 4 physical all-parameter force-loss cases, 2 live Graph cases and isolated native profiles are reused without repeating.', '',
        '|Operator|dtype|Original wall ms|V7 naive wall ms|V8 native wall ms|Paired original/native [IQR]|Paired V7/native [IQR]|Ratio of V7/native wall medians|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['medians'];a=row['paired_ratios_to_native']['original']['wall_ms'];b=row['paired_ratios_to_native']['v7_naive']['wall_ms']
        lines.append(f"|{row['operator']}|{row['dtype']}|{m['original']['wall_ms']:.4f}|{m['v7_naive']['wall_ms']:.4f}|{m['v8_native']['wall_ms']:.4f}|{a['median']:.3f} [{a['q25']:.3f}, {a['q75']:.3f}]|{b['median']:.3f} [{b['q25']:.3f}, {b['q75']:.3f}]|{row['ratio_of_wall_medians']['v7_naive']:.3f}|")
    lines += ['', 'All 4 points use 49152 independent edge rows (N4096 equivalent ×12, no real atom graph). Frame uses actual asymmetric TECE C64 paired branches: [E,9,128] and [E,7,9] -> [E,7,128]. The same degree mask cost is included in all variants, projecting matrix-leaf gradients onto the Wigner subspace. Rotary uses [E,7,64] query/key, [E,8] radial basis and four heads; the real Convolution.rotary_score wrapper includes common radial projection, phase/trig, bias and temperature. Wrapper parameters [4], [64], [8] are all included in final higher gradients.', '',
        'Normal timing includes forward, a common nonlinear output loss, all feature-input first derivatives with create_graph, and the resulting feature-gradient loss differentiated to feature inputs and all wrapper parameters. This feature microchain is not model energy, physical forces or physical force-loss latency. The earlier physical QA covers those semantics separately.', '',
        'Original references fix fused=False and execute the actual Torch bmm/rotary expressions. Original TECE Linear/TensorProduct/Wigner modules are EQX interfaces: the included radial EQX o2.Linear uses Torch matmul/bias; Wigner auto may execute EQX CUDA, but Wigner generation is excluded here. We do not label the whole original TECE model pure PyTorch.', '',
        'The V7 reference imports only frozen rotation.py in a distinct module namespace and uses the current FastEq runtime to compile/launch its unchanged CUDA sources. Its historical EQX-adapted CUDA is a benchmark-only naive reference; V8 production uses original FastEq FCTP transpose, vector-load and block/warp-reduction CUDA helpers. For Rotary the raw function routing assignment occurs outside timers; the wrapper and common parameters are identical and the original module identity is restored in finally. This isolates operator arithmetic rather than comparing unrelated runtime/whole-V7 differences.', '',
        'Each point has 9 same-process rotated/reversed rounds, 5 warmups and 3 iterations. Timed blocks have no spies/hooks/profiler. CPU172 worker and independent CPU171 NVML20ms observer see zero foreign observations; whole attempt rejection is automatic on observed contamination. Full forward/feature-gradient/all-parameter-gradient tensor precision checks occur outside timing against original and V7. Raw wall/event samples and exact orders are retained. IQR is dispersion, not CI; ratios below one indicate a regression.', '',
        'Code/archive SHA checks cover all measured sources and owned recursive header includes, actual TECE wrapper, EQX radial Linear implementation and V7 reference. Each loaded runtime .so SHA is recorded. Existing isolated profile evidence retains native kernel names/dispatch bindings and no external matrix/Triton/EQX arithmetic for the isolated core; its instrumented durations are not normal speedups.', '',f'Raw evidence and measured sources: {OUT}', '']
    (OUT/'NORMAL_SUMMARY.md').write_text('\n'.join(lines))
    print(f'Verified 4 normal actual-wrapper points and {len(component)} archived source files; prior QA/profiles reused.')


if __name__=='__main__':main()
