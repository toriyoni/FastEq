"""External LocalSplit inference probe and targeted host/gradient verification.

This file owns no production implementation. GPU runs require an explicit UUID;
CPU QA never initializes CUDA. Zero-kernel forward timings use host wall only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
import torch

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CAND = ROOT / 'src/FastEq_v7'
FLAG = 'FASTEQ_EXPERIMENTAL_LOCAL_SPLIT_VIEW'
sys.path.insert(0, str(CAND / 'benchmarks/adapters'))
sys.path.insert(0, str(CAND / 'tests/adapters'))


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def probe_class(legacy):
    class LocalSplitViewProbe(legacy):
        @classmethod
        def apply(cls, features, ranges, metadata):
            if os.environ.get(FLAG) == '1' and not torch.is_grad_enabled():
                return tuple(features.narrow(1, start, size) for start, size in ranges)
            return super().apply(features, ranges, metadata)
    return LocalSplitViewProbe


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


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_signature(tensor, source):
    return {
        'shape': list(tensor.shape), 'stride': list(tensor.stride()),
        'storage_offset': tensor.storage_offset(), 'dtype': str(tensor.dtype),
        'device': str(tensor.device), 'requires_grad': tensor.requires_grad,
        'has_grad_fn': tensor.grad_fn is not None,
        'is_inference': tensor.is_inference(),
        'storage_alias': tensor.untyped_storage().data_ptr() == source.untyped_storage().data_ptr(),
    }


def feature(kind, rows, device, requires_grad=True):
    base = torch.arange(max(1, rows * 25), dtype=torch.float64, device=device)[:rows * 25]
    base = base.reshape(rows, 25).clone().requires_grad_(requires_grad)
    if kind == 'leaf':
        return base[:, 1:13].detach().requires_grad_(requires_grad)
    if kind == 'view':
        return base[:, 1:25:2]
    return (base * .03).sin()[:, 1:13]


def simple_metadata():
    ranges = ((1, 3), (8, 2))
    entries = tuple(((index, index - 1, -1), 1.) for index in range(1, 4))
    entries += tuple(((index, -1, index - 8), 1.) for index in range(8, 10))
    return ranges, repr((1, (12, 3, 2), entries))


def cpu_semantics(legacy, candidate):
    ranges, metadata = simple_metadata()
    records = []
    contexts = {'grad': torch.enable_grad, 'no_grad': torch.no_grad,
                'inference': torch.inference_mode}
    for rows in (0, 4):
        for kind in ('leaf', 'view', 'operation'):
            for require in (False, True):
                for context, factory in contexts.items():
                    signatures = []
                    outputs = []
                    for cls in (legacy, candidate):
                        x = feature(kind, rows, 'cpu', require)
                        with enabled(), factory():
                            y = cls.apply(x, ranges, metadata)
                        outputs.append(y)
                        signatures.append([tensor_signature(value, x) for value in y])
                    assert signatures[0] == signatures[1], (rows, kind, require, context, signatures)
                    for a, b in zip(*outputs, strict=True):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    assert all(value['storage_alias'] for value in signatures[1])
                    records.append({'rows': rows, 'kind': kind, 'requires_grad': require,
                                    'context': context, 'outputs': signatures[1]})
    x = feature('leaf', 4, 'cpu')
    for mode in (True, False, True, False):
        with enabled(), torch.set_grad_enabled(mode):
            y = candidate.apply(x, ranges, metadata)
        assert all((value.grad_fn is not None) == mode for value in y)
    for value in (False, True):
        with enabled(value), torch.no_grad():
            y = candidate.apply(x, ((0, 0), (4, 2)), metadata)
        assert tuple(y[0].shape) == (4, 0)
        assert y[1].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert not torch.cuda.is_initialized()
    return {'status': 'pass', 'semantic_cases': records,
            'dynamic_grad_and_empty_width': 'pass', 'cuda_initialized': False}


def chain(fn, source, ranges, metadata, omitted):
    x = source.detach().requires_grad_()
    outputs = fn(x, ranges, metadata)
    active = outputs[:1] if omitted else outputs
    objective = sum((value.sin() + .13 * value.square()).sum() for value in active)
    first = torch.autograd.grad(objective, x, create_graph=True)[0]
    higher = torch.autograd.grad(first.square().sum(), x)[0]
    return outputs, first, higher


def compare_chain(actual, expected, dtype):
    flat_actual = (*actual[0], *actual[1:])
    flat_expected = (*expected[0], *expected[1:])
    tolerance = dict(atol=3e-5, rtol=5e-5) if dtype == torch.float32 else dict(atol=2e-11, rtol=2e-10)
    for a, b in zip(flat_actual, flat_expected, strict=True):
        torch.testing.assert_close(a, b, **tolerance)


def gpu_qa(legacy, candidate, guard):
    from eqx_cuda_spy import forbid_eqx_cuda
    ranges, metadata = simple_metadata()
    original = lambda x, ranges, metadata: tuple(x.narrow(1, start, size) for start, size in ranges)
    records = []
    for dtype in (torch.float32, torch.float64):
        for omitted in (False, True):
            for noncontiguous in (False, True):
                guard.check()
                base = torch.randn(17, 25, device='cuda', dtype=dtype)
                x = base[:, 1:25:2] if noncontiguous else base[:, 1:13].contiguous()
                expected = chain(original, x, ranges, metadata, omitted)
                previous = chain(lambda *args: legacy.apply(*args), x, ranges, metadata, omitted)
                with enabled(), forbid_eqx_cuda() as calls:
                    actual = chain(lambda *args: candidate.apply(*args), x, ranges, metadata, omitted)
                assert calls == []
                compare_chain(actual, expected, dtype)
                compare_chain(actual, previous, dtype)
                torch.cuda.synchronize()
                guard.check()
                records.append({'dtype': str(dtype), 'omitted_output': omitted,
                                'noncontiguous_offset': noncontiguous, 'status': 'pass'})
        empty = torch.empty(0, 12, device='cuda', dtype=dtype, requires_grad=True)
        with enabled():
            y = candidate.apply(empty, ranges, metadata)
            gradients = torch.autograd.grad(sum(value.sum() for value in y), empty)[0]
        assert gradients.shape == empty.shape
        assert gradients.numel() == 0
    # The actual TECE split also verifies that the 960 omitted packed columns
    # stay zero through first and higher derivatives.
    from scaling_operator_factories import _tece_template
    template = _tece_template()
    actual_ranges = template.projection_slices
    actual_metadata = template._eqx_split_metadata
    actual_width = template.linear_up.irreps_out.dim
    actual_input = torch.randn(17, actual_width, device='cuda', dtype=torch.float64)
    for omitted in (False, True):
        expected = chain(original, actual_input, actual_ranges, actual_metadata, omitted)
        with enabled(), forbid_eqx_cuda() as calls:
            actual = chain(lambda *args: candidate.apply(*args), actual_input,
                           actual_ranges, actual_metadata, omitted)
        assert calls == []
        compare_chain(actual, expected, torch.float64)
        assert torch.count_nonzero(actual[1][:, 384:]) == 0
        assert torch.count_nonzero(actual[2][:, 384:]) == 0
        records.append({'dtype': 'torch.float64', 'omitted_output': omitted,
                        'actual_TECE_packed_width': actual_width, 'status': 'pass'})
    # Inference followed by training reuses one input while the routing mode changes.
    x = torch.randn(17, 12, device='cuda', dtype=torch.float64, requires_grad=True)
    with enabled(), torch.inference_mode():
        candidate.apply(x, ranges, metadata)
    compare_chain(chain(lambda *args: candidate.apply(*args), x, ranges, metadata, True),
                  chain(original, x, ranges, metadata, True), torch.float64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with enabled(), torch.cuda.stream(stream):
        actual = chain(lambda *args: candidate.apply(*args), x, ranges, metadata, False)
    torch.cuda.current_stream().wait_stream(stream)
    compare_chain(actual, chain(original, x, ranges, metadata, False), torch.float64)
    # The split itself has no kernel; capture a real downstream CUDA computation.
    static = torch.randn(17, 12, device='cuda', dtype=torch.float32)
    def downstream():
        outputs = candidate.apply(static, ranges, metadata)
        return sum(value.sin().square().sum() for value in outputs)
    stream.wait_stream(torch.cuda.current_stream())
    with enabled(), torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            downstream()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with enabled(), torch.no_grad(), torch.cuda.graph(graph, stream=stream):
        captured = downstream()
    for _ in range(2):
        static.copy_(torch.randn_like(static))
        graph.replay()
        with enabled(), torch.no_grad():
            expected = downstream()
        torch.testing.assert_close(captured, expected)
    torch.cuda.synchronize()
    guard.check()
    return {'status': 'pass', 'training_cases': records, 'empty_rows': 'pass',
            'inference_then_training': 'pass', 'nondefault_stream': 'pass',
            'downstream_cuda_graph_live_input': 'pass', 'EQX_cuda_calls': []}


def quantiles(values):
    q = statistics.quantiles(values, n=4, method='inclusive')
    return {'median': statistics.median(values), 'q25': q[0], 'q75': q[2]}


def host_benchmark(legacy, candidate, guard, atoms, iterations, rounds):
    from scaling_operator_factories import _tece_template
    template = _tece_template()
    ranges, metadata = template.projection_slices, template._eqx_split_metadata
    dim = template.linear_up.irreps_out.dim
    results = []
    functions = {
        'original': lambda x: tuple(x.narrow(1, start, size) for start, size in ranges),
        'legacy': lambda x: legacy.apply(x, ranges, metadata),
        'new': lambda x: candidate.apply(x, ranges, metadata),
    }
    for n in atoms:
        guard.check()
        rows = n * 12
        x = torch.randn(rows, dim, device='cuda', dtype=torch.float32)
        with enabled(), torch.no_grad():
            expected = functions['original'](x)
            for fn in functions.values():
                actual = fn(x)
                for a, b in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    assert a.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
                for _ in range(200):
                    fn(x)
            samples = {name: [] for name in functions}
            order_records = []
            for r in range(rounds):
                order = list(functions)
                shift = r % len(order)
                order = order[shift:] + order[:shift]
                if r % 2:
                    order.reverse()
                order_records.append(order)
                for name in order:
                    guard.check()
                    torch.cuda.synchronize()
                    fn = functions[name]
                    begin = time.perf_counter_ns()
                    for _ in range(iterations):
                        last = fn(x)
                    elapsed_ns = time.perf_counter_ns() - begin
                    torch.cuda.synchronize()
                    guard.check()
                    samples[name].append(elapsed_ns / iterations / 1000)
                    del last
        ratios = {name: [a / b for a, b in zip(samples[name], samples['new'], strict=True)]
                  for name in ('original', 'legacy')}
        results.append({'atoms': n, 'actual_rows': rows, 'dtype': 'float32',
                        'packed_width': dim, 'ranges': ranges, 'iterations_per_burst': iterations,
                        'round_orders': order_records, 'host_us_per_call': samples,
                        'host_us_summary': {name: quantiles(value) for name, value in samples.items()},
                        'paired_ratios': {name: {'samples': value, **quantiles(value)}
                                          for name, value in ratios.items()},
                        'precision': 'pass', 'math_cuda_kernel_count': 0})
        del x
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('qa', 'benchmark'))
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--legacy-source', type=Path, default=CAND / 'fasteq/ops/local_product.py')
    parser.add_argument('--candidate-kind', choices=('probe', 'production'), default='probe')
    parser.add_argument('--candidate-source', type=Path, default=CAND / 'fasteq/ops/local_product.py')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--cpu', type=int, default=174)
    parser.add_argument('--atoms', type=int, nargs='+', default=(256, 65536))
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cpu_only:
        assert args.action == 'qa'
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    else:
        assert args.gpu_uuid and os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid
        assert Path(os.environ['TORCH_EXTENSIONS_DIR']).resolve().is_relative_to(ROOT / 'cache')
        assert 'local_split' in os.environ['TORCH_EXTENSIONS_DIR']
    if args.candidate_kind == 'production':
        assert args.legacy_source.resolve() != args.candidate_source.resolve()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.manual_seed(31491)
    tracked = [Path(__file__).resolve(), args.legacy_source.resolve()]
    if args.candidate_kind == 'production':
        tracked.append(args.candidate_source.resolve())
    sources = {str(path): digest(path) for path in dict.fromkeys(tracked)}
    report = {'status': 'running', 'candidate_kind': args.candidate_kind,
              'action': args.action, 'source_sha256': sources, 'cpu_only': args.cpu_only,
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'torch': torch.__version__,
              'protocol': f'{args.rounds}-round rotated/reversed three-way same-process host bursts; no CUDA event gain',
              'IQR_is_confidence_interval': False,
              'timing_excludes': ['allocation', 'compile', 'validation', 'warmup', 'NVML guards', 'CUDA synchronize'],
              'timing_includes': ['Python call', 'view creation', 'ordinary output reference replacement'],
              'performance_scope': 'host forward only; burst means differ from full-case wall protocol'}
    try:
        guard = None
        if not args.cpu_only:
            from benchmark_scaling_case import ExclusiveGuard
            guard = ExclusiveGuard(args.gpu_uuid)
            report['guard_before'] = guard.check()
            torch.cuda.set_device(0)
            actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
            assert actual_uuid.removeprefix('GPU-') == args.gpu_uuid.removeprefix('GPU-')
            report['torch_extensions_dir'] = os.environ['TORCH_EXTENSIONS_DIR']
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            report['tf32'] = False
        legacy = module_at(args.legacy_source, 'local_split_legacy_probe').LocalSplit
        candidate = (probe_class(legacy) if args.candidate_kind == 'probe' else
                     module_at(args.candidate_source, 'local_split_production_probe').LocalSplit)
        if args.candidate_kind == 'production':
            assert isinstance(candidate.__dict__.get('apply'), classmethod)
        if args.action == 'qa':
            report['cpu_semantics'] = cpu_semantics(legacy, candidate) if args.cpu_only else 'run separately without CUDA'
            if not args.cpu_only:
                report['gpu_qa'] = gpu_qa(legacy, candidate, guard)
        else:
            report['results'] = host_benchmark(legacy, candidate, guard, args.atoms, args.iterations, args.rounds)
        for path, sha in sources.items():
            assert digest(path) == sha, 'Source changed during probe: ' + path
        if guard is not None:
            report['guard_after'] = guard.check()
        report['status'] = 'pass'
        atomic_json(args.output, report)
        print(json.dumps({'status': report['status'], 'output': str(args.output),
                          'cpu_semantic_cases': len(report.get('cpu_semantics', {}).get('semantic_cases', []))
                          if isinstance(report.get('cpu_semantics'), dict) else None,
                          'benchmark_conditions': len(report.get('results', []))}))
    except BaseException as error:
        report.pop('results', None)
        report['status'] = 'failed_discarded'
        report['error'] = repr(error)
        atomic_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
