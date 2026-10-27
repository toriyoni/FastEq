"""Isolated check of the FCTP shared transpose reused by the UVW feature pair."""
import importlib.util
import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path("/public-data/zhouxibo/zxb/tace_prophet_fasteq")
sys.path.insert(0, str(ROOT / "performance/fasteq_optimization/v7/driver"))
from pair_common import ExclusiveGuard
from fasteq.cuda.jit import kernels


def main():
    spec = importlib.util.spec_from_file_location(
        "pair_source", ROOT / "src/FastEq_v8/fasteq/ops/segmented_product.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    guard = ExclusiveGuard("GPU-af44a7bc-24c0-45be-df6e-03f4c445c78d")
    guard.check()
    torch.empty(0, device="cuda")  # Establish the context before standalone JIT.
    records = []
    for dtype in (torch.float32, torch.float64):
        source = module._paired_source(dtype)
        kernel = kernels([source], "cuda", entry="paired_scatter")[source]
        for c1, c2 in ((64, 64), (33, 35), (32, 32)):
            start, n, total = 3, 7, 13
            xb, yb, xs, ys = 3, 5, 2, 3
            dxw, dyw = xb + c1 * xs + 4, yb + c2 * ys + 4
            matrix = torch.randn(n, c1 * c2, device="cuda", dtype=dtype)
            x = torch.randn(total, dxw, device="cuda", dtype=dtype)
            y = torch.randn(total, dyw, device="cuda", dtype=dtype)
            ox, oy = torch.randn_like(x), torch.randn_like(y)
            refx, refy = ox.clone(), oy.clone()
            scale = .2
            mx = matrix.view(n, c1, c2)
            refx[start:start+n, xb:xb+c1*xs:xs] += scale * (
                mx * y[start:start+n, yb:yb+c2*ys:ys].unsqueeze(1)).sum(2)
            refy[start:start+n, yb:yb+c2*ys:ys] += scale * (
                mx * x[start:start+n, xb:xb+c1*xs:xs].unsqueeze(2)).sum(1)
            bits = (struct.unpack("Q", struct.pack("d", scale))[0]
                    if dtype == torch.float64 else
                    struct.unpack("I", struct.pack("f", scale))[0])
            args = [matrix.data_ptr(), x.data_ptr(), y.data_ptr(), ox.data_ptr(),
                    oy.data_ptr(), start, n, dxw, dyw, xb, xs, yb, ys, c1, c2,
                    dxw, dyw, bits]
            kernel.launch(args, n, 1, 128, torch.cuda.current_stream().cuda_stream,
                          (c2 * (c1 + 1) + c1 + c2) * matrix.element_size())
            torch.cuda.synchronize()
            tolerance = 3e-5 if dtype == torch.float32 else 1e-12
            torch.testing.assert_close(ox, refx, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(oy, refy, atol=tolerance, rtol=tolerance)
            guard.check()
            records.append({"dtype": str(dtype), "channels": [c1, c2], "passed": True})
    output = ROOT / "performance/fasteq_optimization/v8/uvw/paired_tile_qa.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "kind": "Isolated CUDA FCTP transpose/tail/stride checks; not model precision or performance",
        "records": records, "guard": guard.check()}, indent=2) + "\n")
    print("paired FCTP tile: 6 dtype/stride/tail checks passed", flush=True)


if __name__ == "__main__":
    main()
