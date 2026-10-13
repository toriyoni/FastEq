"""Stage6 targeted GPU QA preparation; never run before GPU handoff.

Checks the public unmasked-matrix API, higher gradients, physical force loss,
and CUDA graph replay with independent static leaf tensors. No timing metrics.
"""
from contextlib import contextmanager
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import torch

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND = ROOT / 'src/FastEq_v7'
sys.path.insert(0, str(CAND / 'benchmarks/adapters'))
sys.path.insert(0, str(CAND / 'tests/adapters'))
FLAG = 'FASTEQ_EXPERIMENTAL_ROTATE_VECTOR'
DEGREES_IN = (0, 1, 1, 1, 2, 2, 2, 2, 2)
DEGREES_OUT = (0, 1, 2, 1, 2, 1, 2)
METADATA = repr((DEGREES_IN, DEGREES_OUT))


@contextmanager
def enabled(value=True):
    previous = os.environ.get(FLAG)
    os.environ[FLAG] = '1' if value else '0'
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = previous


def mask(dtype):
    return torch.tensor([[a == b for b in DEGREES_IN] for a in DEGREES_OUT],
                        dtype=dtype, device='cuda')


def tolerance(dtype):
    return dict(atol=3e-8, rtol=3e-7) if dtype == torch.float64 else dict(atol=3e-5, rtol=5e-5)


def compare(actual, expected, dtype):
    records = []
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if a is None or b is None:
            nonempty = b if a is None else a
            assert nonempty is None or torch.count_nonzero(nonempty) == 0
            records.append({'slot': index, 'zero_or_none': True})
            continue
        torch.testing.assert_close(a, b, **tolerance(dtype))
        absolute = (a - b).abs().max().item() if a.numel() else 0.
        denominator = b.square().sum().sqrt().item()
        relative_l2 = (a - b).square().sum().sqrt().item() / max(denominator, 1e-30)
        records.append({'slot': index, 'max_absolute': absolute, 'relative_l2': relative_l2})
    return records


@contextmanager
def sources_selected(implementation):
    previous = implementation._launch
    records = []
    def launch(code, values, output, groups):
        records.append({'vector_source': 'double2' in code, 'output_shape': list(output.shape),
                        'groups': groups, 'note': 'source selection, not measured kernel count'})
        return previous(code, values, output, groups)
    implementation._launch = launch
    try:
        yield records
    finally:
        implementation._launch = previous


def leaves(rows, channels, dtype, offset=0):
    size = rows * 9 * channels
    storage = torch.randn(size + offset, device='cuda', dtype=dtype) * .2
    x = storage[offset:].reshape(rows, 9, channels).requires_grad_()
    m = (.2 + .01 * torch.randn(rows, 7, 9, device='cuda', dtype=dtype)).requires_grad_()
    assert not m.numel() or torch.count_nonzero(m) == m.numel()
    return x, m


def chain(fn, x, m, requested=(0, 1)):
    output = fn(x, m)
    objective = output.sin().mean() + .1 * output.square().mean()
    inputs = (x, m)
    first = torch.autograd.grad(objective, tuple(inputs[i] for i in requested), create_graph=True)
    loss = sum(value.square().mean() for value in first)
    higher = torch.autograd.grad(loss, inputs, allow_unused=True)
    return output, *first, *higher


def ordinary_cases(implementation, guard):
    from eqx_cuda_spy import forbid_eqx_cuda
    records = []
    conditions = [('aligned_f64', torch.float64, 128, 0, True),
                  ('odd_X_storage_offset', torch.float64, 128, 1, False),
                  ('C127_fallback', torch.float64, 127, 0, False),
                  ('FP32_fallback', torch.float32, 128, 0, False)]
    for name, dtype, channels, offset, vector_expected in conditions:
        guard.check()
        rows = 16385 if name in ('aligned_f64', 'odd_X_storage_offset') else 17
        x, m = leaves(rows, channels, dtype, offset)
        degree_mask = mask(dtype)
        original = lambda a, b: torch.bmm(b * degree_mask, a)
        candidate = lambda a, b: implementation.rotate(a, b, METADATA)
        for requested in ((0,), (1,), (0, 1)):
            expected = chain(original, x, m, requested)
            with enabled(False):
                previous = chain(candidate, x, m, requested)
            with enabled(), sources_selected(implementation) as selected, forbid_eqx_cuda() as calls:
                actual = chain(candidate, x, m, requested)
            assert calls == []
            assert selected and selected[0]['output_shape'] == [rows, 7, channels]
            assert selected[0]['vector_source'] == vector_expected, (name, selected)
            if dtype != torch.float64 or channels != 128:
                assert not any(value['vector_source'] for value in selected)
            errors = compare(actual, expected, dtype)
            compare(actual, previous, dtype)
            if requested == (1,):
                assert torch.count_nonzero(actual[1][:, degree_mask == 0]) == 0
            assert torch.count_nonzero(actual[-1][:, degree_mask == 0]) == 0
            torch.cuda.synchronize()
            guard.check()
            records.append({'case': name, 'requested': requested, 'status': 'pass',
                            'X_storage_offset': x.storage_offset(), 'X_pointer_mod16': x.data_ptr() % 16,
                            'matrix_all_entries_nonzero': True, 'source_selection': selected,
                            'precision': errors})
    # Verify structural needs_input_grad=False as well as selected grad targets.
    degree_mask = mask(torch.float64)
    for only in (0, 1):
        x, m = leaves(17, 128, torch.float64)
        values = (x, m)
        values[1 - only].requires_grad_(False)
        def first(fn):
            output = fn(*values)
            gradient = torch.autograd.grad(output.sin().sum(), values[only])[0]
            return output, gradient
        expected = first(lambda a, b: torch.bmm(b * degree_mask, a))
        with enabled():
            actual = first(lambda a, b: implementation.rotate(a, b, METADATA))
        compare(actual, expected, torch.float64)
        records.append({'case': 'structural_only_one_leaf', 'only': only, 'status': 'pass'})
    x, m = leaves(0, 128, torch.float64)
    with enabled():
        output = implementation.rotate(x, m, METADATA)
        gradients = torch.autograd.grad(output.sum(), (x, m))
    assert output.shape == (0, 7, 128)
    assert all(value.numel() == 0 for value in gradients)
    return records


def gate_cases(implementation, guard):
    records = []
    for rows, expected_vector in ((3072, False), (16385, True)):
        guard.check()
        x, m = leaves(rows, 128, torch.float64)
        with enabled(False):
            scalar = chain(lambda a, b: implementation.rotate(a, b, METADATA), x, m)
        with enabled(), sources_selected(implementation) as selected:
            actual = chain(lambda a, b: implementation.rotate(a, b, METADATA), x, m)
        compare(actual, scalar, torch.float64)
        assert selected[0]['vector_source'] == expected_vector
        if not expected_vector:
            assert not any(record['vector_source'] for record in selected)
        torch.cuda.synchronize()
        guard.check()
        records.append({'rows': rows, 'status': 'pass', 'source_selection': selected})
    return records


def transpose_cases(implementation, guard):
    records = []
    for dtype in (torch.float32, torch.float64):
        guard.check()
        x = (torch.randn(4, 128, 9, device='cuda', dtype=dtype) * .2).transpose(1, 2).requires_grad_()
        m = (.2 + .01 * torch.randn(4, 9, 7, device='cuda', dtype=dtype)).transpose(1, 2).requires_grad_()
        assert not x.is_contiguous() and not m.is_contiguous()
        expected = chain(lambda a, b: torch.bmm(b * mask(dtype), a), x, m)
        with enabled(False):
            scalar = chain(lambda a, b: implementation.rotate(a, b, METADATA), x, m)
        with enabled(), sources_selected(implementation) as selected:
            vector = chain(lambda a, b: implementation.rotate(a, b, METADATA), x, m)
        errors = compare(vector, expected, dtype)
        compare(scalar, expected, dtype)
        compare(vector, scalar, dtype)
        assert vector[1].is_contiguous() and vector[2].is_contiguous()
        assert torch.count_nonzero(vector[2][:, mask(dtype) == 0]) == 0
        torch.cuda.synchronize()
        guard.check()
        records.append({'dtype': str(dtype), 'status': 'pass', 'X_stride': list(x.stride()),
                        'M_stride': list(m.stride()), 'source_selection': selected, 'precision': errors})
    return records


def physical(implementation, guard):
    from eqx_cuda_spy import forbid_eqx_cuda
    dtype = torch.float64
    coords = (torch.randn(17, 3, device='cuda', dtype=dtype) * .2).requires_grad_()
    parameters = {
        'projection_X': (torch.randn(3, 9 * 128, device='cuda', dtype=dtype) * .03).requires_grad_(),
        'bias_X': (torch.randn(9 * 128, device='cuda', dtype=dtype) * .05).requires_grad_(),
        'projection_M': (torch.randn(3, 7 * 9, device='cuda', dtype=dtype) * .03).requires_grad_(),
        'bias_M': (torch.randn(7 * 9, device='cuda', dtype=dtype) * .05).requires_grad_(),
        'readout': (torch.randn(7, 128, device='cuda', dtype=dtype) * .03).requires_grad_(),
    }
    degree_mask = mask(dtype)
    def execute(fn):
        x = (coords @ parameters['projection_X'] + parameters['bias_X']).sin().reshape(17, 9, 128)
        m = (coords @ parameters['projection_M'] + parameters['bias_M']).cos().reshape(17, 7, 9)
        assert torch.count_nonzero(m) == m.numel()
        output = fn(x, m)
        energy = ((output * parameters['readout']).sin() + .13 * output.square()).sum()
        force = -torch.autograd.grad(energy, coords, create_graph=True)[0]
        loss = (force - .017).square().mean()
        gradients = torch.autograd.grad(loss, (coords, *parameters.values()))
        return output, energy, force, loss, *gradients
    guard.check()
    expected = execute(lambda x, m: torch.bmm(m * degree_mask, x))
    with enabled(), forbid_eqx_cuda() as calls:
        actual = execute(lambda x, m: implementation.rotate(x, m, METADATA))
    assert calls == []
    errors = compare(actual, expected, dtype)
    invalid = degree_mask.flatten() == 0
    # slots 4=coords, 5=projection_X, 6=bias_X, 7=projection_M, 8=bias_M.
    assert torch.count_nonzero(actual[7][:, invalid]) == 0
    assert torch.count_nonzero(actual[8][invalid]) == 0
    torch.cuda.synchronize()
    guard.check()
    return {'status': 'pass', 'parameter_names': list(parameters),
            'scope': 'coords -> X/M -> nonlinear energy -> physical F(create_graph) -> loss -> all 5 parameters plus coords',
            'precision': errors, 'EQX_cuda_calls': []}


def stream_and_graph(implementation, guard):
    degree_mask = mask(torch.float64)
    candidate = lambda x, m: implementation.rotate(x, m, METADATA)
    original = lambda x, m: torch.bmm(m * degree_mask, x)
    x, m = leaves(17, 128, torch.float64)
    expected = chain(original, x, m)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with enabled(), torch.cuda.stream(stream):
        actual = chain(candidate, x, m)
    torch.cuda.current_stream().wait_stream(stream)
    compare(actual, expected, torch.float64)
    # These are independent static leaves; no reused coordinate-generated
    # nonleaf graph is captured, and every derivative source is warmed first.
    static_x, static_m = leaves(16385, 128, torch.float64)
    stream.wait_stream(torch.cuda.current_stream())
    with enabled(), torch.cuda.stream(stream):
        for _ in range(3):
            chain(candidate, static_x, static_m)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with enabled(), torch.cuda.graph(graph, stream=stream):
        captured = chain(candidate, static_x, static_m)
    errors = []
    for _ in range(2):
        with torch.no_grad():
            static_x.copy_(torch.randn_like(static_x) * .2)
            static_m.copy_(.2 + torch.randn_like(static_m) * .01)
        graph.replay()
        expected = chain(original, static_x, static_m)
        errors.append(compare(captured, expected, torch.float64))
    torch.cuda.synchronize()
    guard.check()
    return {'status': 'pass', 'nondefault_stream': 'pass',
            'cuda_graph_static_inputs_are_independent_leaves': True,
            'live_input_update_replays': errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--cpu', type=int, default=174)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--affected-only', action='store_true')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == args.gpu_uuid
    assert Path(os.environ['TORCH_EXTENSIONS_DIR']).resolve().is_relative_to(ROOT / 'cache')
    assert 'rotate' in os.environ['TORCH_EXTENSIONS_DIR']
    assert not torch.cuda.is_initialized(), 'Guard must precede creating CUDA context'
    from benchmark_scaling_case import ExclusiveGuard
    guard = ExclusiveGuard(args.gpu_uuid)
    before = guard.check()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.manual_seed(15453)
    torch.cuda.set_device(0)
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    assert actual_uuid.removeprefix('GPU-') == args.gpu_uuid.removeprefix('GPU-')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    source = CAND / 'fasteq/ops/rotation.py'
    paths = (Path(__file__).resolve(), source)
    sha = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    spec = importlib.util.spec_from_file_location('stage6_rotation_qa', source)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    report = {'status': 'running', 'source_sha256': sha,
              'source': str(source), 'dtype_main': 'float64', 'metadata': METADATA,
              'guard_before': before, 'cpu': sorted(os.sched_getaffinity(0)),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'tf32': False,
              'torch_extensions_dir': os.environ['TORCH_EXTENSIONS_DIR'],
              'performance_measured': False}
    try:
        report['ordinary_cases'] = [] if args.affected_only else ordinary_cases(implementation, guard)
        report['gate_cases'] = gate_cases(implementation, guard)
        report['transpose_cases'] = transpose_cases(implementation, guard)
        report['physical_force_loss'] = physical(implementation, guard)
        report['stream_and_graph'] = stream_and_graph(implementation, guard)
        for path, expected in sha.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, 'Source drift: ' + path
        report['guard_after'] = guard.check()
        report['status'] = 'pass'
    except BaseException as error:
        report['status'] = 'failed_discarded'
        report['error'] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'ordinary_cases': len(report['ordinary_cases']),
                      'physical_parameters': len(report['physical_force_loss']['parameter_names'])}))


if __name__ == '__main__':
    main()
