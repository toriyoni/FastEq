"""Direct checks for the continuous-Y FCTP adjoint, before AD integration."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import load

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
SOURCE = ROOT / 'src/FastEq_v8/fasteq/cuda/src'
OUT = ROOT / 'performance/fasteq_optimization/v8/uvw/fused_features'
sys.path.insert(0, str(ROOT / 'performance/fasteq_optimization/v7/driver'))
from pair_common import ExclusiveGuard


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    guard = ExclusiveGuard('GPU-af44a7bc-24c0-45be-df6e-03f4c445c78d')
    guard.check()
    binder = OUT / 'feature_binding.cpp'
    binder.write_text('''#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
namespace fasteq_jit {
void fctp_features_into(at::Tensor&,at::Tensor&,const at::Tensor&,const at::Tensor&,
                        const at::Tensor&,const at::Tensor&,double,int);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&fasteq_jit::fctp_features_into);}
''')
    feature_source = Path(os.environ.get('FASTEQ_FCTP_FEATURE_SOURCE', SOURCE / 'fctp_general_features.cu'))
    tag = os.environ.get('FASTEQ_FCTP_FEATURE_QA_TAG', 'direct')
    sources = [feature_source, SOURCE / 'fctp_shared.cuh', SOURCE / 'equi_linear_kernels/ptx_inst.cuh',
               SOURCE / 'equi_linear_kernels/matrix_tf32_primitives.cuh']
    sources += [Path(__file__), binder]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    module = load('fasteq_fctp_features_check', [str(binder), str(sources[0])],
                  extra_include_paths=[str(SOURCE)], extra_cflags=['-O3'],
                  extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '--fmad=true'],
                  with_cuda=True, verbose=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(6129)
    records = []
    dtypes = tuple(getattr(torch, name) for name in os.environ.get(
        'FASTEQ_FCTP_FEATURE_QA_DTYPES', 'float32,float64').split(','))
    for dtype in dtypes:
        tolerance = (dict(atol=3e-5, rtol=5e-5) if dtype == torch.float32
                     else dict(atol=2e-11, rtol=2e-10))
        for rows, offset in ((1, 0), (9, 1), (17, 1), (33, 0)):
            def packed(shape, scale):
                return torch.randn(offset + int(torch.tensor(shape).prod()),
                                   dtype=dtype, device='cuda')[offset:].view(shape).mul_(scale)
            w = packed((4096, 64), .07)
            x, y, z = [packed((rows, 64), .2) for _ in range(3)]
            for coefficient in (.371, -.73, 0.):
                matrix = (z @ w.t()).view(rows, 64, 64)
                reference = ((matrix * y[:, None, :]).sum(2) * coefficient,
                             (matrix * x[:, :, None]).sum(1) * coefficient)
                for mask in (1, 2, 3):
                    outputs = [packed((rows, 64), 1.) if mask & (1 << i)
                               else torch.empty(0, device='cuda', dtype=dtype) for i in range(2)]
                    module.run(*outputs, w, x, y, z, coefficient, mask)
                    errors = []
                    for i in range(2):
                        if mask & (1 << i):
                            torch.testing.assert_close(outputs[i], reference[i], **tolerance)
                            delta = outputs[i].double() - reference[i].double()
                            errors.append({'role': i + 1, 'max_abs': delta.abs().max().item(),
                                           'relative_l2': (delta.norm() / reference[i].double().norm().clamp_min(1e-30)).item()})
                    guard.check()
                    records.append({'dtype': str(dtype), 'rows': rows, 'offset': offset,
                                    'coefficient': coefficient, 'mask': mask, 'errors': errors})
                    (OUT / (tag + '_QA.json')).write_text(json.dumps({'status': 'running', 'cases': records}, indent=2) + '\n')
    for name, digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    (OUT / (tag + '_QA.json')).write_text(json.dumps({'status': 'passed', 'cases': records,
        'sources': hashes, 'guard': guard.check(), 'cache': os.environ.get('TORCH_EXTENSIONS_DIR'),
        'scope': f'{len(records)} direct masked FCTP adjoint cases, no AD/model/performance claim'}, indent=2) + '\n')
    print(f'{len(records)} direct FCTP feature checks passed', flush=True)


if __name__ == '__main__':
    main()
