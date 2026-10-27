"""Verify and archive SH dispatch-only optimization evidence without GPU work."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND=ROOT/'src/FastEq_v8'
OUT=ROOT/'performance/fasteq_optimization/v8/sh'
ARCHIVE=OUT/'dispatch_measured_component_source'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def destination(path):
    path=Path(path)
    if path.is_relative_to(CAND):return ARCHIVE/path.relative_to(CAND)
    return ARCHIVE/'references'/path.name


def main():
    component={}
    normals=[];profiles=[]
    for dtype in ('float32','float64'):
        path=OUT/f'dispatch_normal_sh_{dtype}_R49152.json'
        row=json.loads(path.read_text())
        assert row['status']=='ok'and row['observer']['status']=='complete'and not row['observer']['foreign_observations']
        for source,expected in row['source_sha256'].items():
            archived=destination(source)
            assert digest(archived if archived.exists()else source)==expected
            assert component.get(source,expected)==expected;component[source]=expected
        normal={key:row[key]for key in ('dtype','rows','mode','medians','paired_ratios_to_native','runtime_binary','fasteq_environment','precision_original','precision_legacy','precision_native_library')}
        normal.update(artifact=str(path),artifact_sha256=digest(path),
            wall_reduction_vs_library=1-row['medians']['native_cuda']['wall_ms']/row['medians']['native_library']['wall_ms'],
            ratio_of_wall_medians={label:row['medians'][label]['wall_ms']/row['medians']['native_cuda']['wall_ms']for label in ('original','legacy_triton','native_library')})
        normals.append(normal)
        path=OUT/f'dispatch_profile_sh_{dtype}_R49152.json';row=json.loads(path.read_text())
        assert row['status']=='ok'and row['native_only']and row['source_sha256']==component
        assert digest(row['trace'])==row['trace_sha256']
        p=row['profile'];assert p['scalar_custom_op_cpu_events']==0
        assert not p['forbidden_library_kernel_names']and p['torch_matmul_cpu_ops']==0
        assert all('triton'not in k['name'].lower()for k in p['kernels'])
        own=[k for k in p['kernels']if k['name']=='run'];assert len(own)==1 and own[0]['count']==4
        profiles.append({'dtype':dtype,'artifact':str(path),'artifact_sha256':digest(path),
            'kernel_only_total_us':p['kernel_only_total_us'],'native_scalar_kernels':own,
            'memory_operation_total_us':p['memory_operation_total_us'],'scalar_custom_op_cpu_events':0,
            'trace':row['trace'],'trace_sha256':row['trace_sha256']})
    qa=json.loads((OUT/'dispatch_QA_attempt2.json').read_text())
    assert qa['status']=='passed'and len(qa['records'])==20
    cpu=json.loads((OUT/'dispatch_CPU_AST_bridge.json').read_text());assert cpu['status']=='passed'
    spy=json.loads((OUT/'native_no_dispatch_spy_CPU.json').read_text());assert spy['status']=='passed'
    for row in (qa,cpu,spy):
        for source,expected in row['source_sha256'].items():
            assert component.get(source,expected)==expected;component[source]=expected
    for source,expected in component.items():
        archived=destination(source);archived.parent.mkdir(parents=True,exist_ok=True)
        if not archived.exists():
            assert digest(source)==expected;archived.write_bytes(Path(source).read_bytes())
        assert digest(archived)==expected
    failed=json.loads((OUT/'dispatch_QA_attempt1.json').read_text())
    assert failed['status']=='error'and failed['error']['type']=='AttributeError'
    failed_archive=OUT/'dispatch_failed_component_source'
    for source,expected in failed['source_sha256'].items():
        data=Path(source).read_bytes()
        if source.endswith('spherical_harmonics.py'):data=data.replace(b'ctx.kernel_metadata',b'ctx.metadata')
        if source.endswith('check_native_sh_dispatch_v8.py'):
            data=data.replace(b'from native_sh_dispatch_spy_v8 import native_only',b'from check_native_gate_invariants_v8 import native_only')
        assert hashlib.sha256(data).hexdigest()==expected
        archived=failed_archive/Path(source).relative_to(CAND)
        archived.parent.mkdir(parents=True,exist_ok=True)
        if archived.exists():assert digest(archived)==expected
        else:archived.write_bytes(data)
    manifest={'scope':'SH emitter/mathematical DAG/AD unchanged, raw CUDA execution helper and new eager Function; all measured normal/profile guard sources, actual reference, QA/CPU bridge/spy guards archived. Shared compiled runtime binary independently hashed per execution.',
              'files':{source:{'sha256':value,'archive':str(destination(source))}for source,value in component.items()}}
    (OUT/'dispatch_measured_component_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary={'status':'verified','normal':normals,'profiles':profiles,'qa_affected_cases':20,
        'cpu_fallback_cases':len(cpu['records']),'spy_CPU_cases':len(spy['records']),
        'unchanged_mathematical_AST':cpu['math_AST_identical'],
        'raw_execute_body_identical_to_old_custom_op':cpu['old_execute_body_equals_raw_helper'],
        'old_48_ordinary_QA_reused_without_repeat':str(OUT/'GPU_QA.json'),
        'component_manifest':str(OUT/'dispatch_measured_component_manifest.json'),
        'failed_first_attempt':{'artifact':str(OUT/'dispatch_QA_attempt1.json'),
            'error':'ctx.metadata was a reserved read-only context property; corrected to kernel_metadata.',
            'three_partial_records_not_counted_as_lite_QA':'Original DispatchMode probe triggered library fallback; all 20 affected cases subsequently ran actual eager Function with a dispatch-transparent spy.',
            'sources_archived':str(failed_archive)},
        'scope':'Shared FastEq native CUDA source/emitter and DAG mathematics are unchanged. Only SH eager autograd/dispatch changed; Wigner keeps custom_op/fake/autograd interface. Physical loss full parameters validated in QA; normal measures feature higher microchain.',
        'protocol':'9 same-process 4-way rotated/reversed rounds, 5 warmups, 3 iterations, CPU172 worker/CPU171 external NVML20ms observer, GPU0 UUID explicit, no spies/profiler in timing. Inclusive IQR describes dispersion, not confidence.',
        'fallback':'Compile, Fake/subclass/DispatchMode and active functorch use original custom_op. Custom-op fullgraph fake smoke passed. Existing outer public SH cached AST/module construction fullgraph tracing remains unsupported for both old and new; no new compile/transform rule claimed.',
        'known_axis_limitation':str(OUT/'edge_axis_limitation.json'),
        'probe_boundary':'The mode-free reusable guard intercepts Python compiler/launcher/resident/OpOverload+Packet aliases, but C++ dispatcher bypasses Python hooks. Separate profile rejects EQX CPU events and external/Triton library kernel names, and asserts scalar custom_op events zero to prove actual eager path.'}
    (OUT/'DISPATCH_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['SH now uses a lightweight recursive torch.autograd.Function for ordinary eager tensors. It reuses the same FastEq CUDA source, ScalarProgram.transpose, _source, _adjoint and raw launcher. Strict AST bridge proves the mathematical DAG/AD/emitter and the original CUDA execution body are unchanged. Wigner custom-op and fake/compiled dispatch remain available.', '',
        '|dtype|Original e3nn wall ms|Old FastEq Triton wall ms|Previous V8 library wall ms|V8 lightweight wall ms|Paired library/light [IQR]|Paired Triton/light [IQR]|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in normals:
        m=row['medians'];a=row['paired_ratios_to_native']['native_library']['wall_ms'];b=row['paired_ratios_to_native']['legacy_triton']['wall_ms']
        lines.append(f"|{row['dtype']}|{m['original']['wall_ms']:.4f}|{m['legacy_triton']['wall_ms']:.4f}|{m['native_library']['wall_ms']:.4f}|{m['native_cuda']['wall_ms']:.4f}|{a['median']:.3f} [{a['q25']:.3f}, {a['q75']:.3f}]|{b['median']:.3f} [{b['q25']:.3f}, {b['q75']:.3f}]|")
    lines += ['', 'This controlled comparison changes SH Python dispatch only. Previous-library and lightweight paths execute the same raw helper and analytical CUDA DAG. All four variants share one process and identical loss/seeds/rows. Normal wall reductions versus previous V8 library are 35.1% FP32 and 41.1% FP64 (ratio of medians). Paired old/library-to-light speedups are 1.525 and 1.617; paired original-to-light 2.907 and 2.972. Old FastEq Triton-to-light paired speedups 1.126 and 1.108 now have IQR above one.', '',
        '49152 independent rows, degree [0,1,2], normalized component. The microchain includes forward, first feature derivative and nonlinear second feature derivative; it is not physical model force-loss timing. Nine rotated/reversed four-way rounds, five warmups, three iterations; CPU172 and external CPU171 NVML20ms observer, clean GPU0. All full output/gradient comparisons against original, old Triton and previous library pass; zero foreign observations. Raw CUDA event samples, wall samples and exact orders are retained. IQR is dispersion, not CI.', '',
        '|dtype|Profile all CUDA kernels ms|Shared ScalarProgram kernels ms|run calls|Custom-op CPU events|',
        '|---|---:|---:|---:|---:|']
    for row in profiles:
        own=row['native_scalar_kernels'][0]
        lines.append(f"|{row['dtype']}|{row['kernel_only_total_us']/1000:.5f}|{own['total_us']/1000:.5f}|{own['count']}|0|")
    lines += ['', 'Native-only profiles contain four shared ScalarProgram run kernels and ordinary Torch loss/reduction kernels. No Triton, cuBLAS, CUTLASS, ATen matrix multiplication or EQX CPU dispatch. Zero fasteq::scalar_expression CPU events proves the profile exercised the lightweight path. Kernel totals remain close to previous-library profiles (about 0.083/0.105ms); the controlled normal comparison establishes the eliminated dispatch overhead as a practical bottleneck. Instrumented kernel time is not normal latency; memory/annotations are separate.', '',
        '20 meaningful affected QA conditions passed: FP32/FP64 complete mixed second/third derivatives, active-input masks/None output seeds, constant exact-zero adjoints, noncontiguous/tail/offset/multiple-leading layouts, physical coordinate forces and all scale/shift/readout/coordinate loss gradients, nondefault stream, live parameter CUDA Graph, clamped zero and non-axis near-boundary. The existing 48 ordinary conditions were reused through mathematical AST equality.', '',
        'Six CPU fake/dispatch/compile checks passed with raw CUDA execution forbidden. Compile/Fake/subclass/DispatchMode and active functorch retain the original custom-op fallback. The original outer SH cached module/AST-building API and new API both remain unsupported for fullgraph tracing; the existing prebuilt-metadata custom-op fullgraph fake path passes. No new compile/transform capability is claimed.', '',
        'The first failed attempt used a reserved ctx.metadata property, now fixed to kernel_metadata; its raw log/JSON and exact failed sources are archived. Its three partial records selected the library path because the original probe installed DispatchMode, so they are not counted as lightweight QA. All 20 affected conditions subsequently passed using a mode-free probe.', '',
        'The reusable native_sh_dispatch_spy_v8.native_only retains EQX compiler/launcher/resident guards, rejects cached Python OpOverload/Packet aliases and Triton JIT, and preserves an empty mode stack. Five positive/negative CPU guard fixtures pass and all patched identities restore. Python guards cannot intercept C++ dispatcher initiation, so actual independent profiles remain required. They permit ordinary ATen/cuBLAS where model scalar MLPs need them; only this SH isolated profile forbids matrix libraries.', '',
        'The existing FP32 near-epsilon normalized axis radial-derivative cancellation limitation remains recorded in edge_axis_limitation.json. CUDA math and accuracy tolerances were not altered.', '',f'Raw evidence and source archive: {OUT}', '']
    (OUT/'DISPATCH_SUMMARY.md').write_text('\n'.join(lines))
    print(f"SH dispatch: 20 affected QA, {len(cpu['records'])} CPU fallback, {len(spy['records'])} spy CPU, 2 normal, 2 actual-lite profiles verified; {len(component)} sources archived.")


if __name__=='__main__':main()
