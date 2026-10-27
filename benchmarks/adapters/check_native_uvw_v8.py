"""Targeted V8 UVW CUDA correctness: never an OOM or complete-suite rerun."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
from native_uvw_v8_common import (CANDIDATE, UUID, chain, compare, configure,
    dense_metadata, description, operator_case, random_leaves, reference,
    source_hashes, verify_hashes, write_json)


def ordinary(native, guard, torch):
    records = []
    conditions = [('sparse_mixed', description(), 33, 'contiguous', False),
                  ('sparse_transpose', description(), 257, 'transpose', False),
                  ('sparse_per_row_weights', description(), 17, 'stride2', True),
                  ('scalar64_tail_storage_offset', dense_metadata(), 4099, 'offset', False)]
    for dtype in (torch.float32, torch.float64):
        for name, metadata, rows, layout, per_row in conditions:
            guard.check()
            leaves = random_leaves(metadata, rows, dtype, layout, per_row)
            expected = chain(reference, metadata, leaves, third=True)
            actual = chain(native.segmented_product, metadata, leaves, third=True)
            errors = compare(actual, expected, dtype)
            torch.cuda.synchronize(); guard.check()
            records.append({'case': name, 'dtype': str(dtype), 'rows': rows, 'layout': layout,
                            'weight_per_row': per_row, 'full_mixed_second_third': True,
                            'precision': errors, 'status': 'pass'})
            print(json.dumps({'qa_case': name, 'dtype': str(dtype), 'status': 'pass'}), flush=True)
        for operator in ('tp_uvw', 'tp_nonuniform'):
            metadata, leaves, original = operator_case(operator, 396, dtype)
            errors = compare(chain(native.segmented_product, metadata, leaves, third=True),
                             chain(original, metadata, leaves, third=True), dtype)
            torch.cuda.synchronize(); guard.check()
            records.append({'case': 'actual_e3nn_' + operator, 'dtype': str(dtype),
                            'rows': 396, 'precision': errors, 'status': 'pass'})
    return records


def all_roles(native, guard, torch):
    records = []
    for dtype in (torch.float32, torch.float64):
        metadata = dense_metadata(8)
        w, x, y = random_leaves(metadata, 17, dtype, 'offset')
        z = torch.randn(17, 8, device='cuda', dtype=dtype).mul_(.2).requires_grad_()
        values = (w, x, y, z)
        for role in range(4):
            guard.check()
            def original():
                out = reference(metadata, *values[:3])
                if role == 3:
                    return out
                return torch.autograd.grad((out * z).sum(), values[role], create_graph=True)[0]
            def candidate():
                return native._Product.apply(metadata, role, 0, 17, *values)
            def three(fn):
                out = fn()
                requested = [values[i] for i in range(4) if i != role]
                first = torch.autograd.grad((out.sin() + .1*out.square()).mean(), requested,
                                            create_graph=True, allow_unused=True)
                second = torch.autograd.grad(sum(g.square().mean() for g in first if g is not None),
                                             values, create_graph=True, allow_unused=True)
                third = torch.autograd.grad(sum(g.sin().mean() for g in second if g is not None),
                                            values, allow_unused=True)
                return out, *first, *second, *third
            errors = compare(three(candidate), three(original), dtype)
            torch.cuda.synchronize(); guard.check()
            records.append({'case': 'direct_role', 'role': role, 'dtype': str(dtype),
                            'full_second_third': True, 'precision': errors, 'status': 'pass'})
    return records


def omitted_and_empty(native, guard, torch):
    records = []
    for dtype in (torch.float32, torch.float64):
        metadata = description()
        leaves = random_leaves(metadata, 17, dtype)
        for requested in ((0,), (1,), (2,), (0, 2)):
            errors = compare(chain(native.segmented_product, metadata, leaves, requested=requested),
                             chain(reference, metadata, leaves, requested=requested), dtype)
            records.append({'case': 'selected_leaves', 'requested': requested,
                            'dtype': str(dtype), 'precision': errors, 'status': 'pass'})
        for chosen in (0, 1):
            def omitted(fn):
                out = fn(metadata, *leaves)
                first = torch.autograd.grad(out.sin().mean(), leaves[1:], create_graph=True)
                higher = torch.autograd.grad(first[chosen].square().mean(), leaves)
                return out, *first, *higher
            errors = compare(omitted(native.segmented_product), omitted(reference), dtype)
            records.append({'case': 'paired_None_output_seed', 'chosen': chosen,
                            'dtype': str(dtype), 'precision': errors, 'status': 'pass'})
        for only in range(3):
            leaves = random_leaves(metadata, 17, dtype)
            for index, value in enumerate(leaves):
                value.requires_grad_(index == only)
            def structural(fn):
                out = fn(metadata, *leaves)
                return out, torch.autograd.grad(out.sin().mean(), leaves[only])[0]
            errors = compare(structural(native.segmented_product), structural(reference), dtype)
            records.append({'case': 'structural_needs_input_grad', 'only': only,
                            'dtype': str(dtype), 'precision': errors, 'status': 'pass'})
        leaves = random_leaves(metadata, 0, dtype)
        out = native.segmented_product(metadata, *leaves)
        gradients = torch.autograd.grad(out.sum(), leaves)
        assert out.shape == (0, 27) and all(not torch.count_nonzero(g) for g in gradients)
        records.append({'case': 'empty', 'dtype': str(dtype), 'status': 'pass'})
        torch.cuda.synchronize(); guard.check()
    return records


def physical(native, guard, torch, dtypes=None):
    records = []
    # Elementwise projections avoid inserting cuBLAS into the candidate chain.
    for dtype in (dtypes if dtypes is not None else (torch.float32, torch.float64)):
        metadata = dense_metadata(64)
        rows, c = 17, 64
        coords = torch.randn(rows, 3, device='cuda', dtype=dtype).mul_(.2).requires_grad_()
        parameters = {
            'projection_x': torch.randn(3, c, device='cuda', dtype=dtype).mul_(.07).requires_grad_(),
            'projection_y': torch.randn(3, c, device='cuda', dtype=dtype).mul_(.07).requires_grad_(),
            'shared_weight': torch.randn(1, c**3, device='cuda', dtype=dtype).mul_(.03).requires_grad_(),
            'coordinate_weight': torch.randn(3, c**3, device='cuda', dtype=dtype).mul_(.002).requires_grad_(),
            'readout': torch.randn(c, device='cuda', dtype=dtype).mul_(.04).requires_grad_(),
        }
        leaves = (coords, *parameters.values())
        def run(fn):
            x = (coords[:, :, None] * parameters['projection_x']).sum(1).sin()
            y = (coords[:, :, None] * parameters['projection_y']).sum(1).cos()
            w = parameters['shared_weight'] + (coords.mean(0)[:, None] * parameters['coordinate_weight']).sum(0).sin()
            out = fn(metadata, w, x, y)
            energy = ((out * parameters['readout']).sin() + .1*out.square()).sum()
            force = -torch.autograd.grad(energy, coords, create_graph=True)[0]
            gradients = torch.autograd.grad((force - .13).square().mean(), leaves)
            return energy, force, *gradients
        expected = run(reference)
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = run(native.segmented_product)
        torch.cuda.current_stream().wait_stream(stream)
        errors = compare(actual, expected, dtype)
        with torch.no_grad():
            parameters['shared_weight'].add_(.011)
            parameters['projection_x'].mul_(.71)
        changed = compare(run(native.segmented_product), run(reference), dtype)
        torch.cuda.synchronize(); guard.check()
        records.append({'case': 'coordinate_physical_force_loss', 'dtype': str(dtype), 'rows': rows,
                        'parameter_names': list(parameters), 'all_parameter_gradients_full': True,
                        'coordinates_gradient_full': True, 'nondefault_stream': True,
                        'precision': errors, 'live_update_precision': changed, 'status': 'pass'})
    return records


def graph(native, guard, torch):
    records = []
    from eqx_cuda_spy import forbid_eqx_cuda
    for dtype in (torch.float32, torch.float64):
        metadata = description()
        leaves = random_leaves(metadata, 33, dtype, 'offset')
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                chain(native.segmented_product, metadata, leaves)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with forbid_eqx_cuda() as calls:
            with torch.cuda.graph(graph, stream=stream):
                actual = chain(native.segmented_product, metadata, leaves)
            with torch.no_grad():
                leaves[0].add_(.013)
                leaves[1].mul_(.71)
            graph.replay()
        assert calls == []
        errors = compare(actual, chain(reference, metadata, leaves), dtype)
        torch.cuda.synchronize(); guard.check()
        records.append({'case': 'live_weights_CUDA_graph_no_EQX', 'dtype': str(dtype),
                        'precision': errors, 'status': 'pass'})
    return records


def stress(native, guard, torch):
    records = []
    dtype = torch.float32
    metadata = dense_metadata(64)
    for name, amplitude in (('reciprocal_amplitude_small_x', .001),
                            ('reciprocal_amplitude_large_x', 1000.),
                            ('paired_weight_cancellation', 1.)):
        guard.check()
        leaves = random_leaves(metadata, 129, dtype)
        with torch.no_grad():
            leaves[1].mul_(amplitude)
            leaves[2].div_(amplitude)
            if name == 'paired_weight_cancellation':
                weight = leaves[0].view(64, 64, 64)
                weight[1::2].copy_(-weight[::2])
                weight[1::2].add_(1e-3)
                leaves[1][:, 1::2].copy_(leaves[1][:, ::2])
        actual = chain(native.segmented_product, metadata, leaves, third=True)
        expected = chain(reference, metadata, leaves, third=True)
        errors = compare(actual, expected, dtype)
        full_gradient_l2 = [record['relative_l2'] for record in errors[1:]
                            if not record.get('zero_or_none')]
        # The same established FP32 gradient norm bound as normal snapshots.
        assert max(full_gradient_l2, default=0.) <= 2e-4, full_gradient_l2
        torch.cuda.synchronize(); guard.check()
        records.append({'case': name, 'dtype': str(dtype), 'rows': 129,
                        'full_mixed_second_third': True, 'precision': errors,
                        'all_full_gradient_l2_limit': 2e-4, 'status': 'pass'})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', default=UUID)
    parser.add_argument('--groups', nargs='+', choices=['ordinary', 'roles', 'omitted', 'physical', 'graph', 'stress'],
                        default=['ordinary', 'roles', 'omitted', 'physical', 'graph'])
    args = parser.parse_args()
    import torch
    import fasteq
    from benchmark_scaling_case import ExclusiveGuard
    from fasteq.ops import segmented_product as native
    sys.path.insert(0, str(CANDIDATE / 'tests/adapters'))
    configure(uuid=args.gpu_uuid); torch.manual_seed(83417)
    hashes = source_hashes()
    import hashlib
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {'status': 'running', 'groups': args.groups, 'source_sha256': hashes,
              'scope': 'Only affected native UVW; full fixture outputs and gradients. No timing or OOM grid.',
              'actual_import': fasteq.__file__, 'torch_version': torch.__version__,
              'cuda_version': torch.version.cuda, 'CPU': sorted(os.sched_getaffinity(0)),
              'cache': {key: os.environ.get(key) for key in ('TORCH_EXTENSIONS_DIR', 'XDG_CACHE_HOME')},
              'flags': {k: v for k, v in os.environ.items() if k.startswith('FASTEQ_')}, 'records': []}
    guard = ExclusiveGuard(args.gpu_uuid)
    report['before_guard'] = guard.check()
    started = time.monotonic()
    try:
        groups = {'ordinary': ordinary, 'roles': all_roles, 'omitted': omitted_and_empty,
                  'physical': physical, 'graph': graph, 'stress': stress}
        for group in args.groups:
            report['records'].extend(groups[group](native, guard, torch))
            write_json(args.output, report)
        verify_hashes(hashes)
        report.update(status='pass', passed_cases=len(report['records']), after_guard=guard.check())
    except Exception as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error),
                                             'traceback': traceback.format_exc()})
        raise
    finally:
        report['elapsed_s'] = time.monotonic() - started
        write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'passed_cases': report['passed_cases']}), flush=True)


if __name__ == '__main__':
    main()
