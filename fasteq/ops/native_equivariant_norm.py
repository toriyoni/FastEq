"""Native CUDA arithmetic behind the existing FastEq Norm autograd interface.

This module does not replace Norm's ctx / selective-gradient / second-order
nodes. Integration is two hooks in equivariant_norm.py: NormPlan.launch ->
launch(), and _parameter_gradients -> parameter_gradients(). Preparation can
bind those hooks per instance/in a test context without editing the old file.
All GPU row math and the ordered parameter transpose are FastEq-owned CUDA.
"""
import ast
import re
from functools import lru_cache
from pathlib import Path

import torch

_ROLES = {'_forward_kernel': 0, '_backward_kernel': 1, '_double_kernel': 2}
_ENTRIES = ('fasteq_norm_forward', 'fasteq_norm_backward', 'fasteq_norm_double',
            'fasteq_norm_reduce_rows')


@lru_cache(maxsize=1)
def _warp_primitive():
    # Original FastEq CUDA/HIP warp primitive, specialized to the CUDA branch.
    utility = (Path(__file__).parents[1] / 'cuda/src/cuda_utils.hpp').read_text()
    def extract(signature):
        start = utility.index(signature)
        return utility[start:utility.index('\ntemplate <typename T>', start + 5)]
    return ('#define GPU_BACKEND_HIP 0\n#define FULL_MASK 0xffffffffu\n'
            + extract('template <typename T>\n__device__ __forceinline__ T gpu_shfl_down')
            + extract('template <typename T>\n__device__ __forceinline__ T warp_reduce_sum(T v)'))


def _parameter_function(name, scalar, prefix, parts, flags, ir_mul, bias=False, default=0):
    pointers = [f'const {scalar}* {prefix}{i}' for i in range(len(flags))]
    signature = ','.join(['int coordinate', *pointers])
    lines = [f'__device__ __forceinline__ {scalar} {name}({signature}) {{',
             f'{scalar} result={scalar}({default});']
    for start, channels, dimension, gamma_slot, gamma_base, beta_slot, beta_base in parts:
        slot, base = (beta_slot, beta_base) if bias else (gamma_slot, gamma_base)
        if channels <= 0 or slot < 0 or slot >= len(flags) or not flags[slot]:
            continue
        channel = (f'(coordinate-{start})%{channels}' if ir_mul
                   else f'(coordinate-{start})/{dimension}')
        lines.append(f'if(coordinate>={start} && coordinate<{start+channels*dimension}) '
                     f'result={prefix}{slot}[{base}+{channel}];')
    lines += ['return result;', '}']
    call = name + '(i' + (',' if pointers else '') + ','.join(f'{prefix}{i}' for i in range(len(flags))) + ')'
    return '\n'.join(lines) + '\n', call


def _partials_source(scalar, role, kw):
    lines = [f'__device__ __forceinline__ void fasteq_norm_parameter_partials('
             f'const {scalar}* value,const {scalar}* dy,{scalar}* partials,long long row) {{']
    width, dim = kw.get('WIDTH', 0), kw['D']
    gamma_flags = kw.get('GGRADFLAGS', ())
    beta_flags = kw.get('BGRADFLAGS', ()) if role == 1 else ()
    for start, channels, dimension, gamma_slot, gamma_base, beta_slot, beta_base in kw['PARTS']:
        if channels and gamma_slot >= 0 and gamma_flags and gamma_flags[gamma_slot]:
            offset = kw['GOFFSETS'][gamma_slot] + gamma_base
            padded = 1 << (dimension - 1).bit_length()
            coordinate = (f'{start}+angle*{channels}+channel' if kw['IR_MUL']
                          else f'{start}+channel*{dimension}+angle')
            lines += [f'for(int channel=threadIdx.x;channel<{channels};channel+=blockDim.x) {{',
                      f'{scalar} component[{padded}];', '#pragma unroll',
                      f'for(int angle=0;angle<{padded};++angle) '
                      f'component[angle]=angle<{dimension}?value[{coordinate}]:{scalar}(0);',
                      '#pragma unroll', f'for(int distance={padded//2};distance;distance>>=1)',
                      '  for(int angle=0;angle<distance;++angle)',
                      '    component[angle]=norm_add(component[angle],component[angle+distance]);',
                      f'partials[row*{width}+{offset}+channel]=component[0];', '}']
        if channels and beta_slot >= 0 and beta_flags and beta_flags[beta_slot]:
            offset = kw['BOFFSETS'][beta_slot] + beta_base
            # Preserve the existing scalar bias transpose exactly; NormPlan
            # exposes Prophet bias only for scalar irreps, and TACE centering.
            lines += [f'for(int channel=threadIdx.x;channel<{channels};channel+=blockDim.x)',
                      f'partials[row*{width}+{offset}+channel]=dy[row*{dim}+{start}+channel];']
    lines += ['}']
    return '\n'.join(lines) + '\n'


def _argument_names(role, kw):
    gamma = [f'g{i}' for i in range(len(kw.get('GFLAGS', ())))]
    beta = [f'k{i}' for i in range(len(kw.get('BFLAGS', ()) if role == 0 else kw.get('KFLAGS', ())))]
    if role == 0:
        return ['x', 'coefficient', *gamma, *beta, 'y'], {'y'}
    if role == 1:
        return ['x', 'dy', 'coefficient', *gamma, 'dx', 'partials'], {'dx', 'partials'}
    h = [f'h{i}' for i in range(len(kw.get('HFLAGS', ())))]
    return ['x', 'dy', 'coefficient', *gamma, 'u', *h, *beta, 'dx', 'ddy', 'partials'], {'dx', 'ddy', 'partials'}


@lru_cache(maxsize=256)
def _source(metadata, dtype):
    role, entries = ast.literal_eval(metadata)
    kw = dict(entries)
    scalar = 'double' if dtype == torch.float64 else 'float'
    defines = {'FASTEQ_NORM_TYPE': scalar, 'FASTEQ_NORM_FP64': int(dtype == torch.float64),
               'FASTEQ_NORM_ROLE': role, 'FASTEQ_NORM_ENTRY': _ENTRIES[role]}
    prototype, parameter_functions, suffix = '', '', ''
    if role < 3:
        threads = 32 * int(kw.get('num_warps', 4))
        if threads not in (128, 256):
            raise ValueError('Native Norm uses the original four/eight warp schedule')
        block = max(threads, 1 << (int(kw['D']) - 1).bit_length())
        defines.update({'FASTEQ_NORM_DIM': kw['D'], 'FASTEQ_NORM_THREADS': threads,
                        'FASTEQ_NORM_BLOCK': block, 'FASTEQ_NORM_CENTERS': len(kw['CENTERS']),
                        'FASTEQ_NORM_EPS': format(kw['EPS'], '.17g'),
                        'FASTEQ_NORM_HAS_U': int(kw.get('HAS_U', False)),
                        'FASTEQ_NORM_DXFLAG': int(kw.get('DXFLAG', False)),
                        'FASTEQ_NORM_DDXFLAG': int(kw.get('DDXFLAG', False)),
                        'FASTEQ_NORM_DDYFLAG': int(kw.get('DDYFLAG', False)),
                        'FASTEQ_NORM_GPARTIAL': int(any(kw.get('GGRADFLAGS', ())))})
        names, outputs = _argument_names(role, kw)
        defines['FASTEQ_NORM_SIGNATURE'] = ','.join(
            ('' if name in outputs else 'const ') + scalar + '* ' + name for name in names)
        groups = (('GAMMA', 'g', kw.get('GFLAGS', ()), False, 1),
                  ('BETA', 'k', kw.get('BFLAGS', ()) if role == 0 else (), True, 0),
                  ('H', 'h', kw.get('HFLAGS', ()), False, 0),
                  ('K', 'k', kw.get('KFLAGS', ()), True, 0))
        for tag, prefix, flags, bias, default in groups:
            function, call = _parameter_function('fasteq_norm_' + tag.lower(), scalar, prefix,
                                                 kw['PARTS'], flags, kw['IR_MUL'], bias, default)
            parameter_functions += function
            defines['FASTEQ_NORM_' + tag + '(i)'] = call
        for table, values in [('start', [a for a, _ in kw['CENTERS']]),
                              ('width', [b for _, b in kw['CENTERS']])]:
            payload = ','.join(map(str, values)) if values else '0'
            parameter_functions += f'__device__ __constant__ int fasteq_norm_center_{table}[]={{{payload}}};\n'
        prototype = (f'__device__ __forceinline__ void fasteq_norm_parameter_partials('
                     f'const {scalar}*,const {scalar}*,{scalar}*,long long);\n')
        suffix = _partials_source(scalar, role, kw)
    prefix = ''.join(f'#define {name} {value}\n' for name, value in defines.items())
    header = (Path(__file__).parents[1] / 'cuda/src/native_equivariant_norm.cuh').read_text()
    return prefix + _warp_primitive() + parameter_functions + prototype + header + suffix


def source_for(role, kwargs, dtype):
    """CPU-only specialization entry, also used by compile/provenance checks."""
    metadata = repr((role, tuple(sorted(kwargs.items()))))
    return _source(metadata, dtype)



@lru_cache(maxsize=64)
def _warp_forward_source(metadata, dtype):
    """Reuse NativeNorm's own specialization and original FastEq warp helper."""
    kwargs = dict(ast.literal_eval(metadata))
    source = source_for(0, kwargs, dtype)
    directory = Path(__file__).parents[1] / 'cuda/src'
    previous = (directory / 'native_equivariant_norm.cuh').read_text()
    assert source.count(previous) == 1
    source = source.replace(previous, (directory / 'native_equivariant_norm_warp.cuh').read_text())
    source = re.sub(r'(#define FASTEQ_NORM_THREADS) \d+', r'\1 32', source)
    return source.replace('#define FASTEQ_NORM_ENTRY fasteq_norm_forward',
                          '#define FASTEQ_NORM_ENTRY fasteq_norm_warp4_forward')


def _use_warp_forward(x, kwargs):
    # The throughput schedule is enabled only in the two measured C64 layouts.
    # Small-row graphs and FP64 retain the existing block reduction unchanged.
    if x.dtype != torch.float32 or x.shape[0] < 1024 or kwargs['D'] != 576:
        return False
    if kwargs['EPS'] != 1e-5 or kwargs.get('num_warps', 4) != 4:
        return False
    if kwargs['IR_MUL']:
        return (kwargs['PARTS'] == ((0,64,1,0,0,0,0),(64,64,3,0,64,-1,0),(256,64,5,0,128,-1,0))
                and kwargs['CENTERS'] == ((0,64),) and kwargs['GFLAGS'] == (True,)
                and kwargs['BFLAGS'] == (True,))
    return (kwargs['PARTS'] == ((0,64,1,0,0,-1,0),(64,64,3,1,0,-1,0),(256,64,5,2,0,-1,0))
            and kwargs['CENTERS'] == () and kwargs['GFLAGS'] == (True,True,True)
            and kwargs['BFLAGS'] == ())

def _flatten_tensors(arguments):
    result = []
    for value in arguments:
        if isinstance(value, torch.Tensor): result.append(value)
        elif isinstance(value, (tuple, list)): result.extend(_flatten_tensors(value))
        else: raise TypeError('Native Norm arguments must be tensors or tensor tuples')
    return result


def launch(plan, fn, arguments, x, kwargs):
    """Drop-in NormPlan.launch arithmetic hook; never calls a Triton JIT function."""
    name = getattr(fn, '__name__', None) or getattr(getattr(fn, 'fn', None), '__name__', None)
    if name not in _ROLES:
        raise ValueError('Unsupported native Norm arithmetic entry: ' + str(name))
    role = _ROLES[name]
    values = _flatten_tensors(arguments)
    if any(value.device != x.device or value.dtype != x.dtype for value in values):
        raise ValueError('Native Norm tensor arguments must share dtype/device')
    if not x.shape[0]: return
    # Cache only the resident executable and static launch metadata. Tensor
    # values/addresses remain live below, including replaced parameters.
    warp = role == 0 and _use_warp_forward(x, kwargs)
    key = ('native_norm', role, x.device, x.dtype, tuple(kwargs.items()), warp)
    prepared = plan._launchers.get(key)
    if prepared is None:
        source = (_warp_forward_source(repr(tuple(sorted(kwargs.items()))), x.dtype)
                  if warp else source_for(role, kwargs, x.dtype))
        entry = 'fasteq_norm_warp4_forward' if warp else _ENTRIES[role]
        from fasteq.cuda.jit import kernels
        kernel = kernels([source], x.device, entry=entry)[source]
        prepared = (kernel, 128 if warp else 32 * kwargs.get('num_warps', 4), 4 if warp else 1)
        plan._launchers[key] = prepared
    kernel, threads, rows_per_block = prepared
    pointers = [value.data_ptr() for value in values]
    if rows_per_block == 4: pointers.append(x.shape[0])
    kernel.launch(pointers, (x.shape[0] + rows_per_block - 1) // rows_per_block, 1,
                  threads, torch.cuda.current_stream(x.device).cuda_stream)


def reduce_rows(value, width, accumulation=None):
    """FastEq's fixed 256-row ordered tree in native CUDA, with live pointers."""
    if value.dtype not in (torch.float32, torch.float64) or not value.is_cuda:
        raise TypeError('Native row reduction requires CUDA FP32/FP64')
    if not width: return value.new_empty((0,))
    if not value.shape[0]: return value.new_zeros((width,))
    if value.shape[0] == 1: return value.reshape(width)
    if accumulation not in (None, 'fp32' if value.dtype == torch.float32 else 'fp64'):
        raise ValueError('Norm preserves its input precision for ordered reduction')
    from fasteq.cuda.jit import kernels
    source = source_for(3, {}, value.dtype)
    kernel = kernels([source], value.device, entry=_ENTRIES[3])[source]
    rows = value.shape[0]
    while rows > 1:
        groups = (rows + 255) // 256
        output = value.new_empty((groups, width))
        kernel.launch([value.data_ptr(), output.data_ptr(), rows, width], groups,
                      (width + 31) // 32, 128, torch.cuda.current_stream(value.device).cuda_stream)
        value, rows = output, groups
    return value.reshape(width)


def parameter_gradients(partials, parameters, requested, widths, width):
    """Drop-in _parameter_gradients hook, retaining None/shape/active ordering."""
    if not width: return [None] * len(parameters)
    combined = reduce_rows(partials, width)
    pieces = iter(combined.split(widths))
    return [next(pieces).reshape(parameter.shape) if needed else None
            for parameter, needed in zip(parameters, requested, strict=True)]


__all__ = ['launch', 'reduce_rows', 'parameter_gradients', 'source_for']
