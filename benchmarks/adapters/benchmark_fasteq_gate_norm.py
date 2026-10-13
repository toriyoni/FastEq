"""Paired v3/new FastEq residual/Gate/RMS chain latency and true force losses."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_models import assert_exclusive_gpu, one_batch, stats
from e3nn import o3
from fasteq.adapters.fusion import ProphetPostprocess
from fasteq_v3_snapshot import load_v3_postprocess
from prophet.layer_norm import RMSLayerNorm
from prophet.model import Gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    dependencies = (
        "src/FastEq/fasteq/adapters/fusion.py", "src/FastEq/fasteq/ops/gate_norm.py",
        "src/FastEq/fasteq/triton/gate_norm.py", "src/FastEq/fasteq/ops/gate.py",
        "src/FastEq/fasteq/ops/equivariant_norm.py",
        "src/FastEq/fasteq/triton/fused_equivariant_gate.py",
        "src/FastEq/fasteq/triton/fused_equivariant_layer_norm.py",
        "src/FastEq/fasteq/cuda/triton_launch.py", "src/FastEq/fasteq/cuda/jit.py",
        "src/FastEq/fasteq/cuda/src/jit_runtime.cpp", "src/FastEq/fasteq/cuda/src/jit_runtime.h",
        "src/FastEq/fasteq/cuda/src/norm_plan.cpp",
        "src/prophet/prophet/model.py", "src/prophet/prophet/layer_norm.py",
    )

    def dependency_hashes():
        return {relative:hashlib.sha256((args.root/relative).read_bytes()).hexdigest()
                for relative in dependencies}

    frozen = dependency_hashes()
    digest = hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    sys_path = args.root / "src/FastEq/tests/adapters"
    import sys
    sys.path.insert(0, str(sys_path))
    from eqx_cuda_spy import forbid_eqx_cuda

    old_class, provenance = load_v3_postprocess(args.root)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    assert_exclusive_gpu()
    report = {"metadata": {"gpu": torch.cuda.get_device_name(),
        "uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "comparison": "verified FastEq v3 composition vs fused FastEq",
        "snapshot": provenance, "channels": args.channels,
        "current_dependency_sha256": frozen,
        "dependency_guard": "Before/after each case and alternating round, outside timed regions",
        "benchmark_driver_sha256": {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in (Path(__file__).name,"fasteq_v3_snapshot.py","benchmark_models.py")},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "protocol": {"warmup": args.warmup, "rounds": args.rounds, "iterations": args.iterations,
            "order": "alternating paired", "gpu_guard": "NVML before/after each round",
            "eqx_guard": "strong compiler/launcher/dispatcher prohibition during pair validation"},
        "scope": "chain inputs and live norm parameters; loss/zero_grad/backward included, no optimizer",
        "force_loss": "nonlinear energy; grad wrt first input create_graph; energy+force-square loss backward (including indirect output path)"},
        "results": []}
    channels = args.channels
    for dtype in (torch.float32, torch.float64):
        ir = o3.Irreps(f"{3*channels}x0e+{channels}x1o+{channels}x2e")
        out = o3.Irreps(f"{channels}x0e+{channels}x1o+{channels}x2e")
        gate = Gate(ir).cuda().to(dtype)
        norm = RMSLayerNorm(out, centering=False).cuda().to(dtype)
        conv = SimpleNamespace(gate=gate,layer_norm=None,linear_2=SimpleNamespace(irreps_out=ir))
        plans = (old_class(conv,norm),ProphetPostprocess(conv,norm))
        for nodes in args.nodes:
            torch.manual_seed(751)
            x = torch.randn(nodes,ir.dim,device="cuda",dtype=dtype,requires_grad=True)
            skip = torch.randn_like(x,requires_grad=True)
            params = (x,skip,*norm.parameters())
            seed = torch.randn(nodes,out.dim,device="cuda",dtype=dtype)
            for mode in ("forward", "backward", "force_loss"):
                assert dependency_hashes() == frozen, "Dependencies changed; discard this case"
                def execute(plan, *, mode=mode, params=params, seed=seed, x=x, skip=skip, nodes=nodes):
                    for p in params:
                        p.grad = None
                    with torch.set_grad_enabled(mode != "forward"):
                        y = plan(x,skip)
                        if mode != "forward":
                            energy = (y.sin()*seed).sum()/nodes + y.square().mean()
                            if mode == "backward":
                                energy.backward()
                            elif mode == "force_loss":
                                force = torch.autograd.grad(energy,x,create_graph=True)[0]
                                (energy.square()/10 + force.square().sum()).backward()
                    return y

                fns = [lambda p=p,execute=execute:execute(p) for p in plans]
                checked=[]
                for fn in fns:
                    result=fn()
                    checked.append((result.detach().clone(),[p.grad.detach().clone() if p.grad is not None else None for p in params]))
                atol,rtol=(4e-5,4e-4) if dtype==torch.float32 else (2e-10,2e-9)
                torch.testing.assert_close(checked[0][0],checked[1][0],atol=atol,rtol=rtol)
                diff=norm2=0.
                for a,b in zip(checked[0][1],checked[1][1]):
                    if a is None:
                        assert b is None
                    else:
                        torch.testing.assert_close(a,b,atol=atol*30,rtol=rtol*30)
                        diff+=(a.double()-b.double()).square().sum().item()
                        norm2+=a.double().square().sum().item()
                with forbid_eqx_cuda():
                    guarded=fns[1]()
                    torch.testing.assert_close(guarded,checked[1][0],atol=atol,rtol=rtol)
                    for p,b in zip(params,checked[1][1]):
                        if b is not None:
                            torch.testing.assert_close(p.grad,b,atol=atol*30,rtol=rtol*30)
                validation={"output_max_abs":(checked[0][0]-checked[1][0]).abs().max().item(),
                            "gradient_relative_l2":(diff/max(norm2,1e-300))**.5,"eqx_guard":"passed"}
                for fn in fns:
                    for _ in range(args.warmup):
                        fn()
                torch.cuda.synchronize()
                samples=[[],[]]
                for block in range(args.rounds):
                    assert_exclusive_gpu()
                    assert dependency_hashes() == frozen, "Dependencies changed; discard this case"
                    for i in (0,1) if block%2==0 else (1,0):
                        samples[i].append(one_batch(fns[i],args.iterations))
                    assert_exclusive_gpu()
                    assert dependency_hashes() == frozen, "Dependencies changed; discard this case"
                metrics=[{"wall_ms":stats([s["wall_ms"] for s in values]),
                          "cuda_event_ms":stats([s["cuda_event_ms"] for s in values]),"samples":values} for values in samples]
                row={"dtype":str(dtype),"nodes":nodes,"mode":mode,"validation":validation,
                     "current_dependency_sha256_digest":digest,
                     "baseline_v3":metrics[0],"fused":metrics[1],
                     "native_resident_launchers":len(plans[1].fused._launchers),
                     "native_abi_fallback_count":sum(value is None for value in plans[1].fused._launchers.values()),
                     "paired_wall_speedup":stats([a["wall_ms"]/b["wall_ms"] for a,b in zip(*samples)])}
                report["results"].append(row)
                assert dependency_hashes() == frozen, "Dependencies changed; discard this case"
                (args.output/"chains.json").write_text(json.dumps(report,indent=2)+"\n")
                print(str(dtype),nodes,mode,metrics[0]["wall_ms"]["median"],metrics[1]["wall_ms"]["median"],row["paired_wall_speedup"]["median"],flush=True)


if __name__ == "__main__":
    main()
