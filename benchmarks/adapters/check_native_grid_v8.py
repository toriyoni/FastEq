"""Only the new host view partition at the old CUDA grid.y boundary."""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import torch
from fasteq.cuda.jit import runtime
from native_uvw_v8_common import ROOT, source_hashes, verify_hashes, write_json
sys.path.insert(0, str(ROOT / 'performance/fasteq_optimization/v7/driver'))
from pair_common import ExclusiveGuard


def main():
    output = Path(os.environ['FASTEQ_GRID_QA_OUTPUT'])
    if output.exists():
        raise RuntimeError('Preserve prior attempts')
    uuid = os.environ['CUDA_VISIBLE_DEVICES']
    guard = ExclusiveGuard(uuid)
    guard.check()
    torch.set_num_threads(1)
    torch.manual_seed(73420)
    torch.backends.cuda.matmul.allow_tf32 = False
    hashes = source_hashes()
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {'status': 'running', 'source_sha256': hashes, 'records': [],
              'scope': 'new large row/batch host partition only; no full matrix suite repeat',
              'TF32': False}
    old_promote = os.environ.get('FASTEQ_EXPERIMENTAL_MATRIX_F32_F64MMA')
    try:
        for dtype, promote in ((torch.float32, '1'), (torch.float32, '0'), (torch.float64, '1')):
            os.environ['FASTEQ_EXPERIMENTAL_MATRIX_F32_F64MMA'] = promote
            tile = 64 if dtype == torch.float64 or promote == '1' else 128
            # Row tail crosses the exact old y limit with a small K/N. Batch
            # broadcast covers zero batch stride and a nonzero storage offset.
            for kind in ('rows', 'batches', 'combined'):
                if kind == 'rows':
                    m, k, n = 65535 * tile + 3, 1, 2
                    a = torch.randn(m * 2 + 1, device='cuda', dtype=dtype)[1::2].reshape(m, k)
                    b = torch.randn(k, n * 2 + 1, device='cuda', dtype=dtype)[:, 1::2]
                    c = torch.randn(m + 1, n * 2 + 1, device='cuda', dtype=dtype)[1:, 1::2]
                else:
                    batch, m = (65537, 1) if kind == 'batches' else (32769, tile + 1)
                    k, n = 1, 2
                    a = torch.randn(batch + 1, m, k, device='cuda', dtype=dtype)[1:]
                    b = torch.randn(1, k, n, device='cuda', dtype=dtype).expand(batch, k, n)
                    c = torch.randn(batch + 1, m, n, device='cuda', dtype=dtype)[1:]
                expected = .7 * (a @ b) + .2 * c
                runtime().native_gemm_into(c, a, b, .7, .2)
                torch.testing.assert_close(c, expected, atol=3e-5 if dtype == torch.float32 else 2e-11,
                                           rtol=5e-5 if dtype == torch.float32 else 2e-10)
                assert torch.isfinite(c).all()
                report['records'].append({'dtype': str(dtype), 'promote': promote,
                    'case': kind, 'shape': list(c.shape), 'full_outputs': True,
                    'maximum_abs': (c - expected).abs().max().item()})
                torch.cuda.synchronize()
                guard.check()
                write_json(output, report)
                del a, b, c, expected
                torch.cuda.empty_cache()
        verify_hashes(hashes)
        binary = Path(runtime().__file__)
        report.update(status='passed', passed_cases=len(report['records']), final_guard=guard.check(),
                      actual_runtime_binary={'path': str(binary), 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()})
    except Exception as error:
        report.update(status='failed', error={'message': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        if old_promote is None:
            os.environ.pop('FASTEQ_EXPERIMENTAL_MATRIX_F32_F64MMA', None)
        else:
            os.environ['FASTEQ_EXPERIMENTAL_MATRIX_F32_F64MMA'] = old_promote
        write_json(output, report)
    print(json.dumps({'status': report['status'], 'cases': len(report['records'])}))


if __name__ == '__main__':
    main()
