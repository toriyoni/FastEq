"""Owned validation support for the V8 native FastEq UVW campaign.

This module does not initialize CUDA at import. References use PyTorch/e3nn;
the candidate execution is always the FastEq segmented-product API.
"""
import hashlib
import ast
import json
import os
from pathlib import Path
import re
import statistics

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CANDIDATE = ROOT / 'src/FastEq_v8'
NAIVE = ROOT / 'src/FastEq_v7'
UUID = 'GPU-3a6d868b-775e-3236-988f-2772678876f2'
FLAGS = ('LINEAR_SIMT', 'NATIVE_MODULATION', 'LINEAR_GEMM',
         'SEGMENTED_TASK_MASK', 'SEGMENTED_REDUCE', 'SEGMENTED_GEMM',
         'SEGMENTED_PAIR')


def configure(source=CANDIDATE, fusion=True, cpu=172, uuid=UUID):
    import torch
    import fasteq
    assert Path(fasteq.__file__).resolve().is_relative_to(Path(source).resolve())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == uuid
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    for suffix in FLAGS:
        os.environ['FASTEQ_EXPERIMENTAL_' + suffix] = '1'
    os.environ['FASTEQ_EXPERIMENTAL_SEGMENTED_NATIVE_FUSION'] = str(int(fusion))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hashes(source=CANDIDATE, all_python=False):
    # Other V8 operators can be developed in parallel. The UVW dependency
    # guard covers its recursive AD, JIT/runtime, matrix/CUDA sources/headers;
    # final release freezes the entire checkout separately.
    if all_python:
        files = sorted(Path(source).glob('fasteq/**/*.py'))
    else:
        files = [Path(source) / name for name in
                 ('fasteq/ops/segmented_product.py', 'fasteq/ops/equivariant_norm.py',
                  'fasteq/cuda/jit.py')]
        # The paired VJP combines its explicit contributions with the shared
        # FastEq elementwise family, including Norm's owned ordered transpose.
        if (Path(source) / 'fasteq/ops/native_bias.py').is_file():
            files += [Path(source) / name for name in (
                'fasteq/ops/native_bias.py', 'fasteq/cuda/src/native_elementwise.cuh',
                'fasteq/ops/native_equivariant_norm.py',
                'fasteq/cuda/src/native_equivariant_norm.cuh',
                'fasteq/cuda/src/cuda_utils.hpp')]
    cuda = Path(source) / 'fasteq/cuda/src'
    jit = Path(source) / 'fasteq/cuda/jit.py'
    runtime = next(node for node in ast.parse(jit.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == 'runtime')
    # Read the actual runtime load-source declarations, including norm and
    # modulation sources which share the same loaded extension binary.
    compile_files = [cuda / node.right.value for node in ast.walk(runtime)
                     if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                     and isinstance(node.left, ast.Name) and node.left.id == 'directory'
                     and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                     and node.right.value.endswith(('.cpp', '.cu'))]
    # The continuous-Y FCTP loader declares its three sources with a literal
    # list comprehension. Include those actual runtime inputs as well.
    for node in ast.walk(runtime):
        if (isinstance(node, ast.ListComp) and len(node.generators) == 1
                and isinstance(node.elt, ast.BinOp) and isinstance(node.elt.op, ast.Div)
                and isinstance(node.elt.left, ast.Name) and node.elt.left.id == 'directory'
                and isinstance(node.elt.right, ast.Name)
                and isinstance(node.generators[0].target, ast.Name)
                and node.elt.right.id == node.generators[0].target.id
                and isinstance(node.generators[0].iter, (ast.Tuple, ast.List))):
            compile_files.extend(cuda / value.value for value in node.generators[0].iter.elts
                                 if isinstance(value, ast.Constant) and isinstance(value.value, str)
                                 and value.value.endswith(('.cpp', '.cu')))
    pending = compile_files + [cuda / 'jit_runtime.h', cuda / 'fctp_shared.cuh',
                              cuda / 'equi_linear_kernels/ptx_inst.cuh',
                              cuda / 'equi_linear_kernels/matrix_tf32.cuh']
    closure = set()
    while pending:
        path = pending.pop().resolve()
        if path in closure or not path.is_file():
            continue
        closure.add(path)
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', path.read_text(), flags=re.MULTILINE):
            options = [path.parent / include, cuda / include]
            target = next((p for p in options if p.is_file()), None)
            if target is not None and target.resolve().is_relative_to(Path(source).resolve()):
                pending.append(target)
    files += sorted(closure)
    files += [Path(__file__)]
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def verify_hashes(hashes):
    for name, digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, 'Source drift: ' + name


def tolerance(dtype):
    import torch
    return (dict(atol=3e-5, rtol=5e-5) if dtype == torch.float32
            else dict(atol=2e-11, rtol=2e-10))


def compare(actual, expected, dtype):
    import torch
    records = []
    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if a is None or b is None:
            value = b if a is None else a
            assert value is None or not torch.count_nonzero(value)
            records.append({'slot': i, 'zero_or_none': True})
            continue
        torch.testing.assert_close(a, b, **tolerance(dtype))
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta = (a.detach().double() - b.detach().double())
        norm = b.detach().double().norm().item()
        records.append({'slot': i, 'elements': a.numel(), 'full_comparison': True,
                        'max_abs': delta.abs().max().item() if a.numel() else 0.,
                        'relative_l2': delta.norm().item() / max(norm, 1e-30)})
    return records


def chain(fn, metadata, leaves, mode='feature_higher', requested=(0, 1, 2), third=False):
    import torch
    with torch.set_grad_enabled(mode != 'forward'):
        z = fn(metadata, *leaves)
    if mode == 'forward':
        return (z,)
    objective = (z.sin() + .1 * z.square()).mean()
    if mode == 'backward':
        return (z, *torch.autograd.grad(objective, leaves, allow_unused=True))
    first = torch.autograd.grad(objective, [leaves[i] for i in requested], create_graph=True,
                                allow_unused=True)
    terms = [g.square().mean() for g in first if g is not None and g.requires_grad]
    second = torch.autograd.grad(sum(terms), leaves, allow_unused=True, create_graph=third)
    if not third:
        return (z, *first, *second)
    terms = [g.sin().mean() for g in second if g is not None and g.requires_grad]
    last = torch.autograd.grad(sum(terms), leaves, allow_unused=True)
    return (z, *first, *second, *last)


def reference(metadata, w, x, y):
    import torch
    from fasteq.ops.segmented_product import _parse
    dims, terms, coefficients = _parse(metadata)
    rows = x.shape[0]
    out = x.new_zeros(rows, dims[3])
    for term, coefficient in zip(terms, coefficients, strict=True):
        mode, wo, xb, xs, yb, ys, zb, zs, c1, c2, co, weighted = term
        a = x[:, xb + torch.arange(c1, device=x.device) * xs]
        b = y[:, yb + torch.arange(c2, device=x.device) * ys]
        width = c1 * c2 * co if mode == 3 else c1 * c2 if mode == 2 else c1
        weight = w[:, wo:wo + width] if weighted else x.new_ones(1, width)
        if mode == 3:
            value = torch.einsum('eu,ev,euvw->ew', a, b, weight.reshape(-1, c1, c2, co))
        elif mode == 2:
            value = torch.einsum('eu,ev,euv->eu', a, b, weight.reshape(-1, c1, c2))
        else:
            value = a * b * weight
        out.index_add_(1, zb + torch.arange(co, device=x.device) * zs, value * coefficient)
    return out


def dense_metadata(channels=64):
    c = channels
    return repr(((c**3, c, c, c), ((3, 0, 0, 1, 0, 1, 0, 1, c, c, c, 1),), (1/c,)))


def description(mixed=True):
    dims = (8*9*11 + 8, 21, 24, 27)
    terms = [(3, 0, 1, 2, 2, 2, 3, 2, 8, 9, 11, 1),
             (3, 0, 0, 2, 1, 2, 3, 2, 8, 9, 11, 1)]
    coefficients = [.11329173, -.0813439]
    if mixed:
        terms += [(0, 8*9*11, 1, 2, 2, 2, 3, 2, 8, 8, 8, 1),
                  (1, 0, 1, 2, 0, 1, 3, 2, 8, 1, 8, 0)]
        coefficients += [.0391311, .023912]
    return repr((dims, tuple(terms), tuple(coefficients)))


def random_leaves(metadata, rows, dtype, layout='contiguous', per_row_weight=False):
    import torch
    from fasteq.ops.segmented_product import _parse
    dims = _parse(metadata)[0]
    leaves = []
    for n, d in ((rows if per_row_weight else 1, dims[0]), (rows, dims[1]), (rows, dims[2])):
        if layout == 'transpose':
            value = torch.randn(d, n, device='cuda', dtype=dtype).T.mul_(.2)
        elif layout == 'offset':
            value = torch.randn(n*d+1, device='cuda', dtype=dtype)[1:].view(n, d).mul_(.2)
        elif layout == 'stride2':
            value = torch.randn(n, d*2, device='cuda', dtype=dtype)[:, ::2].mul_(.2)
        else:
            value = torch.randn(n, d, device='cuda', dtype=dtype).mul_(.2)
        leaves.append(value.requires_grad_())
    return leaves


def operator_case(operator, rows, dtype):
    from scaling_operator_factories import build_case
    case = build_case(operator, rows // 12 if operator == 'tp_uvw' else rows, dtype, 'cuda')
    candidate = case['candidate']
    # Warmup constructs the descriptor from the actual adapter, before timing.
    candidate(*case['inputs'])
    metadata = candidate._fasteq_plan.metadata
    leaves = (candidate.weight.detach().reshape(1, -1).requires_grad_(), *case['inputs'])
    original = lambda metadata, w, x, y: case['baseline'](x, y, w.reshape(-1))
    return metadata, leaves, original


def distribution(values):
    q = statistics.quantiles(values, n=4, method='inclusive')
    return {'samples': values, 'median': statistics.median(values), 'q25': q[0], 'q75': q[2],
            'dispersion_kind': 'inclusive IQR, not confidence interval'}


def parse_trace(path):
    events = json.loads(Path(path).read_text())['traceEvents']
    kernel = [e for e in events if e.get('cat') == 'kernel']
    memory = [e for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset')]
    forbidden = [e.get('name', '') for e in kernel
                 if any(x in e.get('name', '').lower() for x in
                        ('cublas', 'cutlass', 'ampere_sgemm', 'volta_sgemm', 'sm80_xmma'))]
    mm = [e for e in events if e.get('cat') == 'cpu_op' and
          e.get('name') in ('aten::mm', 'aten::bmm', 'aten::addmm', 'aten::matmul')]
    grouped = {}
    for e in kernel:
        name = e['name']
        row = grouped.setdefault(name, {'name': name, 'count': 0, 'total_us': 0.})
        row['count'] += 1
        row['total_us'] += e.get('dur', 0.)
    return {'kernels': sorted(grouped.values(), key=lambda r: r['total_us'], reverse=True),
            'kernel_count': len(kernel), 'kernel_only_total_us': sum(e.get('dur', 0.) for e in kernel),
            'memory_operation_total_us': sum(e.get('dur', 0.) for e in memory),
            'forbidden_library_kernel_names': sorted(set(forbidden)), 'torch_matmul_cpu_ops': len(mm),
            'scope': 'Instrumented Chrome cat=kernel only; memory/annotations separate; not normal latency.'}
