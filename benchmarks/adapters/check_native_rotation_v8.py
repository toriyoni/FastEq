"""Native FastEq rotation QA and isolated profiler evidence; never normal timing."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CANDIDATE = ROOT / 'src/FastEq_v8'
UUID = 'GPU-af44a7bc-24c0-45be-df6e-03f4c445c78d'
DEGREES = ((0, 1, 1, 1, 2, 2, 2, 2, 2), (0, 1, 2, 1, 2, 1, 2))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hashes():
    paths = [Path(__file__).resolve()] + [CANDIDATE / rel for rel in (
        'fasteq/ops/rotation.py', 'fasteq/cuda/src/fctp_rotation.cuh',
        'fasteq/cuda/src/fctp_shared.cuh', 'fasteq/cuda/src/cuda_utils.hpp',
        'fasteq/cuda/jit.py', 'fasteq/cuda/src/jit_runtime.cpp')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class OnlyNativeArithmetic(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        assert name not in ('aten::mm', 'aten::bmm', 'aten::addmm', 'aten::baddbmm',
                            'aten::matmul', 'aten::einsum'), 'ATen matrix arithmetic: ' + name
        assert name.split('::')[0] not in ('eqx', 'cueq', 'cuequivariance', 'openequivariance'), name
        return func(*args, **(kwargs or {}))


def tolerance(dtype):
    return (dict(atol=3e-5, rtol=5e-5) if dtype == torch.float32
            else dict(atol=2e-11, rtol=2e-10))


def compare(actual, expected, dtype):
    result = []
    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if a is None or b is None:
            value = b if a is None else a
            assert value is None or not torch.count_nonzero(value)
            result.append({'slot': i, 'none_or_zero': True})
            continue
        torch.testing.assert_close(a, b, **tolerance(dtype))
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta = a.detach().double() - b.detach().double()
        result.append({'slot': i, 'elements': a.numel(), 'full_comparison': True,
                       'max_abs': delta.abs().max().item() if a.numel() else 0,
                       'relative_l2': delta.norm().item() / max(b.detach().double().norm().item(), 1e-30)})
    return result


def leaf(shape, dtype, layout):
    if layout == 'transpose' and len(shape) == 3:
        value = torch.randn(shape[0], shape[2], shape[1], dtype=dtype, device='cuda').transpose(1, 2)
    elif layout == 'stride2':
        value = torch.randn(*shape[:-1], shape[-1] * 2, dtype=dtype, device='cuda')[..., ::2]
    elif layout == 'offset':
        size = 1
        for width in shape: size *= width
        value = torch.randn(size + 1, dtype=dtype, device='cuda')[1:].reshape(shape)
    else:
        value = torch.randn(shape, dtype=dtype, device='cuda')
    value.mul_(.2)
    return value.requires_grad_()


def frame_reference(x, matrix, degrees=DEGREES):
    mask = matrix.new_tensor([[a == b for b in degrees[0]] for a in degrees[1]])
    return torch.bmm(matrix * mask, x)


def rotary_reference(query, key, phase, lmax, mmax, heads):
    # Independent tensor expression in mathematical real/imaginary pairs.
    rows, _, channels = query.shape
    per_head = channels // heads
    cursor, terms = 0, []
    for order in range(mmax + 1):
        width = lmax - order + 1
        qr = query[:, cursor:cursor + width].reshape(rows, width, heads, per_head)
        kr = key[:, cursor:cursor + width].reshape(rows, width, heads, per_head)
        real = (qr * kr).sum((1, 3))
        imaginary = real * 0
        if order:
            qi = query[:, cursor + width:cursor + 2 * width].reshape(rows, width, heads, per_head)
            ki = key[:, cursor + width:cursor + 2 * width].reshape(rows, width, heads, per_head)
            real = real + (qi * ki).sum((1, 3))
            imaginary = (qi * kr - qr * ki).sum((1, 3))
        terms.append(phase[:, order, :, 0] * real + phase[:, order, :, 1] * imaginary)
        cursor += width * (2 if order else 1)
    return torch.stack(terms).sum(0)


def chain(fn, inputs, requested=None, probe=None):
    requested = tuple(range(len(inputs))) if requested is None else requested
    if probe is not None: probe['phase'] = 'forward'
    output = fn(*inputs)
    denominator = max(output.numel(), 1)
    loss = (output.sin().sum() + .1 * output.square().sum()) / denominator
    if probe is not None: probe['phase'] = 'first'
    first = torch.autograd.grad(loss, [inputs[i] for i in requested], create_graph=True, allow_unused=True)
    terms = [g.square().sum() / max(g.numel(), 1) for g in first if g is not None and g.requires_grad]
    if probe is not None: probe['phase'] = 'higher'
    higher = torch.autograd.grad(sum(terms), inputs, allow_unused=True)
    return output, *first, *higher


def operator_cases(implementation, guard):
    result = []
    frame_conditions = [(17, 64, 'plain'), (1, 128, 'plain'), (0, 17, 'plain'),
                        (33, 127, 'offset'), (33, 129, 'transpose'), (7, 33, 'stride2'),
                        (5, 513, 'plain')]
    rotary_conditions = [(17, 2, 1, 64, 4, 'plain'), (1, 3, 3, 128, 4, 'plain'),
                         (0, 2, 1, 32, 2, 'plain'), (7, 2, 2, 66, 3, 'offset'),
                         (5, 3, 1, 96, 3, 'transpose'), (3, 0, 0, 33, 1, 'stride2')]
    for dtype in (torch.float32, torch.float64):
        for rows, channels, layout in frame_conditions:
            guard.check()
            x = leaf((rows, len(DEGREES[0]), channels), dtype, layout)
            matrix = leaf((rows, len(DEGREES[1]), len(DEGREES[0])), dtype, layout)
            reference = lambda x, m: frame_reference(x, m)
            candidate = lambda x, m: implementation.rotate(x, m, repr(DEGREES))
            for requested in ((0,), (1,), (0, 1)):
                expected = chain(reference, (x, matrix), requested)
                with OnlyNativeArithmetic(): actual = chain(candidate, (x, matrix), requested)
                precision = compare(actual, expected, dtype)
                result.append({'operator': 'frame', 'dtype': str(dtype), 'rows': rows,
                               'channels': channels, 'layout': layout, 'requested': requested,
                               'status': 'pass', 'precision': precision})
            torch.cuda.synchronize()
            guard.check()
        for rows, lmax, mmax, channels, heads, layout in rotary_conditions:
            guard.check()
            angular = sum((lmax - m + 1) * (2 if m else 1) for m in range(mmax + 1))
            inputs = (leaf((rows, angular, channels), dtype, layout),
                      leaf((rows, angular, channels), dtype, layout),
                      leaf((rows, mmax + 1, heads, 2), dtype, layout))
            reference = lambda q, k, p: rotary_reference(q, k, p, lmax, mmax, heads)
            candidate = lambda q, k, p: implementation.rotary_product(q, k, p, lmax, mmax, heads)
            for requested in ((0,), (1,), (2,), (0, 1, 2)):
                expected = chain(reference, inputs, requested)
                with OnlyNativeArithmetic(): actual = chain(candidate, inputs, requested)
                precision = compare(actual, expected, dtype)
                result.append({'operator': 'rotary', 'dtype': str(dtype), 'rows': rows,
                               'lmax': lmax, 'mmax': mmax, 'channels': channels, 'heads': heads,
                               'layout': layout, 'requested': requested, 'status': 'pass',
                               'precision': precision})
            torch.cuda.synchronize()
            guard.check()
    return result


def physical_cases(implementation, guard):
    result = []
    for dtype in (torch.float32, torch.float64):
        for kind in ('frame', 'rotary'):
            rows, channels, heads = 13, 64, 4
            coords = leaf((rows, 3), dtype, 'plain')
            if kind == 'frame':
                widths = (len(DEGREES[0]) * channels, len(DEGREES[1]) * len(DEGREES[0]))
                reference = lambda x, m: frame_reference(x, m)
                candidate = lambda x, m: implementation.rotate(x, m, repr(DEGREES))
                reshape = lambda values: (values[0].reshape(rows, len(DEGREES[0]), channels),
                                          values[1].reshape(rows, len(DEGREES[1]), len(DEGREES[0])))
            else:
                widths = (7 * channels, 7 * channels, 2 * heads * 2)
                reference = lambda q, k, p: rotary_reference(q, k, p, 2, 1, heads)
                candidate = lambda q, k, p: implementation.rotary_product(q, k, p, 2, 1, heads)
                reshape = lambda values: (values[0].reshape(rows, 7, channels), values[1].reshape(rows, 7, channels),
                                          values[2].reshape(rows, 2, heads, 2))
            parameters = []
            for width in widths:
                parameters += [leaf((3, width), dtype, 'plain'), leaf((width,), dtype, 'plain')]
            output_shape = (len(DEGREES[1]), channels) if kind == 'frame' else (heads,)
            readout = leaf(output_shape, dtype, 'plain')
            parameters += [readout]
            def execute(fn):
                values = [(coords @ parameters[2*i] + parameters[2*i+1]).sin() for i in range(len(widths))]
                # Projection/readout are ordinary model layers outside the equivariant kernel.
                output = fn(*reshape(values))
                energy = ((output * readout).sin() + .13 * output.square()).sum()
                force = -torch.autograd.grad(energy, coords, create_graph=True)[0]
                loss = (force - .017).square().mean()
                gradients = torch.autograd.grad(loss, (coords, *parameters))
                return output, energy, force, loss, *gradients
            guard.check()
            expected = execute(reference)
            actual = execute(candidate)
            result.append({'operator': kind, 'dtype': str(dtype), 'status': 'pass',
                           'all_parameter_count': len(parameters), 'precision': compare(actual, expected, dtype),
                           'scope': 'coords -> features -> E -> physical F(create_graph) -> loss -> every parameter plus coords'})
            torch.cuda.synchronize()
            guard.check()
    return result


def graph_cases(implementation, guard):
    result = []
    for kind in ('frame', 'rotary'):
        dtype = torch.float64
        if kind == 'frame':
            inputs = (leaf((65, 9, 128), dtype, 'plain'), leaf((65, 7, 9), dtype, 'plain'))
            original = lambda x, m: frame_reference(x, m)
            candidate = lambda x, m: implementation.rotate(x, m, repr(DEGREES))
        else:
            inputs = (leaf((65, 7, 64), dtype, 'plain'), leaf((65, 7, 64), dtype, 'plain'),
                      leaf((65, 2, 4, 2), dtype, 'plain'))
            original = lambda q, k, p: rotary_reference(q, k, p, 2, 1, 4)
            candidate = lambda q, k, p: implementation.rotary_product(q, k, p, 2, 1, 4)
        guard.check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): chain(candidate, inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream): captured = chain(candidate, inputs)
        replays = []
        for _ in range(2):
            with torch.no_grad():
                for value in inputs: value.copy_(torch.randn_like(value) * .2)
            graph.replay()
            replays.append(compare(captured, chain(original, inputs), dtype))
        torch.cuda.synchronize()
        guard.check()
        result.append({'operator': kind, 'status': 'pass', 'nondefault_stream': True,
                       'independent_static_leaves': True, 'live_input_replays': replays})
    return result


@contextmanager
def emitted_source_probe(implementation, directory):
    original = implementation._launch
    records, state = [], {'phase': 'warmup'}
    directory.mkdir(parents=True, exist_ok=True)
    def launch(source, values, output, groups, *, entry='run'):
        digest = hashlib.sha256(source.encode()).hexdigest()
        path = directory / (digest + '.cu')
        if not path.exists(): path.write_text(source)
        records.append({'phase': state['phase'], 'owner_path': str(CANDIDATE / 'fasteq/ops/rotation.py'),
                        'owner_function': '_frame_source' if entry.startswith('fasteq_frame_') else '_rotary_source',
                        'emitted_source_path': str(path), 'emitted_source_sha256': digest, 'entry': entry,
                        'rows': values[0].shape[0], 'groups': groups, 'kernel_count_claim': False})
        return original(source, values, output, groups, entry=entry)
    implementation._launch = launch
    try: yield records, state
    finally:
        implementation._launch = original
        assert implementation._launch is original


def profile_cases(implementation, guard, path, kinds=('frame', 'rotary')):
    records = []
    for kind in kinds:
        dtype, rows = torch.float64, 4096
        if kind == 'frame':
            inputs = (leaf((rows, 9, 128), dtype, 'plain'), leaf((rows, 7, 9), dtype, 'plain'))
            operation = lambda x, m: implementation.rotate(x, m, repr(DEGREES))
        else:
            inputs = (leaf((rows, 7, 64), dtype, 'plain'), leaf((rows, 7, 64), dtype, 'plain'),
                      leaf((rows, 2, 4, 2), dtype, 'plain'))
            operation = lambda q, k, p: implementation.rotary_product(q, k, p, 2, 1, 4)
        for _ in range(3): chain(operation, inputs)
        torch.cuda.synchronize()
        guard.check()
        stem = path.parent / ('profile_' + kind)
        with emitted_source_probe(implementation, path.parent / ('profile_' + kind + '_sources')) as (dispatch, state):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with OnlyNativeArithmetic(): chain(operation, inputs, probe=state)
                torch.cuda.synchronize()
        guard.check()
        trace_path = stem.with_suffix('.trace.json')
        profiler.export_chrome_trace(str(trace_path))
        manifest = {'candidate_root': str(CANDIDATE), 'isolated_operator': kind,
                    'scope': 'operator_and_adjoints', 'phases': ['forward', 'first', 'higher'],
                    'source_files': [{'path': p, 'sha256': digest} for p, digest in source_hashes().items()],
                    'dispatch': dispatch, 'hooks_restored': True, 'timing_claim': False}
        bindings_path = stem.with_suffix('.bindings.json')
        write_json(bindings_path, manifest)
        checker = ROOT / 'performance/fasteq_optimization/v8/check_native_evidence.py'
        spec = importlib.util.spec_from_file_location('rotation_evidence_checker', checker)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        evidence = module.inspect_evidence(json.loads(trace_path.read_text()), manifest)
        write_json(stem.with_suffix('.evidence.json'), evidence)
        assert evidence['status'] == 'pass', evidence
        records.append({'operator': kind, 'status': 'pass', 'trace': str(trace_path),
                        'bindings': str(bindings_path), 'evidence': str(stem.with_suffix('.evidence.json'))})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=174)
    parser.add_argument('--observer-cpu', type=int, default=175)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--profile-operator', choices=('frame', 'rotary'))
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == UUID
    assert not torch.cuda.is_initialized()
    assert 'v8_rotation' in os.environ['TORCH_EXTENSIONS_DIR']
    from benchmark_scaling_case import ExclusiveGuard
    guard = ExclusiveGuard(UUID)
    before = guard.check()
    os.sched_setaffinity(0, {args.cpu}); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = args.output.with_suffix('.observer.stop'); stop.unlink(missing_ok=True)
    observer_report = args.output.with_suffix('.observer.json')
    observer_script = ROOT / 'performance/fasteq_optimization/v8/native_matrix/observer.py'
    observer = subprocess.Popen(['taskset', '-c', str(args.observer_cpu), sys.executable,
                                 str(observer_script), str(os.getpid()), str(observer_report), str(stop)])
    hashes = source_hashes()
    import fasteq
    assert Path(fasteq.__file__).resolve().is_relative_to(CANDIDATE)
    from fasteq.ops import rotation as implementation
    torch.manual_seed(51423)
    torch.cuda.set_device(0)
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') == UUID.removeprefix('GPU-')
    report = {'status': 'running', 'source_sha256': hashes, 'guard_before': before,
              'gpu_uuid': UUID, 'cpu': args.cpu, 'observer_cpu': args.observer_cpu,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'tf32': False,
              'normal_performance_measured': False, 'private_cache': os.environ['TORCH_EXTENSIONS_DIR']}
    try:
        if args.profile_only:
            kinds = (args.profile_operator,) if args.profile_operator else ('frame', 'rotary')
            report['profiles'] = profile_cases(implementation, guard, args.output, kinds=kinds)
        else:
            report['operator_cases'] = operator_cases(implementation, guard)
            report['physical_cases'] = physical_cases(implementation, guard)
            report['graph_cases'] = graph_cases(implementation, guard)
        for name, expected in hashes.items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, 'Relevant source drift: ' + name
        report['guard_after'] = guard.check()
        report['status'] = 'pass'
    except BaseException as error:
        report['status'] = 'failed_discarded'; report['error'] = repr(error)
        raise
    finally:
        stop.touch(); observer.wait(timeout=10)
        observer_data = json.loads(observer_report.read_text())
        report['observer'] = observer_data
        if observer_data['foreign']:
            report['status'] = 'foreign_gpu_discarded'
        write_json(args.output, report)
    assert report['status'] == 'pass', report['status']
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'operator_cases': len(report.get('operator_cases', [])),
                      'physical_cases': len(report.get('physical_cases', [])),
                      'graph_cases': len(report.get('graph_cases', [])),
                      'profiles': len(report.get('profiles', []))}))


if __name__ == '__main__':
    main()
