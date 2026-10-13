"""Validate frozen v5 namespace/model binding without CUDA or native compilation."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from benchmark_models import build_model
from paired_model_common import PARAMETER_COUNTS, sha256
from v5_snapshot import load_v5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide GPUs for this CPU-only check"
    torch.set_num_threads(1)
    adapt, provenance = load_v5(args.root)
    assert not torch.cuda.is_initialized()
    original_cuda = torch.nn.Module.cuda
    torch.nn.Module.cuda = lambda self, device=None: self
    records = {}
    try:
        for name in PARAMETER_COUNTS:
            original = build_model(name, 4, torch.float64, 2)
            frozen = adapt(original)
            convolutions = [module for module in frozen.modules() if (
                type(module).__module__, type(module).__name__
            ) == ("tace.models._e3nn.tece_oam_rra", "Convolution")]
            expected = 2 * len(convolutions)
            assert frozen._fasteq_v5_frozen_method_count == expected
            if name != "tece":
                assert expected == 0
            assert len(dict(frozen.named_parameters())) == PARAMETER_COUNTS[name]
            assert tuple(dict(original.named_parameters())) == tuple(dict(frozen.named_parameters()))
            assert tuple(original.state_dict()) == tuple(frozen.state_dict())
            for key, tensor in frozen.state_dict().items():
                assert torch.equal(tensor, original.state_dict()[key]), key
            plans, postprocesses = [], []
            for path, module in frozen.named_modules():
                plan = getattr(module, "_fasteq_plan", None)
                if plan is not None:
                    qualified = type(plan).__module__ + "." + type(plan).__name__
                    plans.append({"path": path, "type": qualified})
                    assert qualified.startswith("fasteq_v5."), (path, qualified)
                postprocess = getattr(module, "_fasteq_postprocess", None)
                if postprocess is not None:
                    assert module.forward.__func__ is type(module).forward
                    assert type(postprocess).__module__ == "fasteq_v5.adapters.fusion"
                    postprocesses.append(path)
            for conv in convolutions:
                assert conv.forward.__func__.__module__.endswith("._fasteq_v5_tece_oam_rra")
                assert conv.rotary_score.__func__.__module__.endswith("._fasteq_v5_tece_oam_rra")
            if name == "prophet":
                assert frozen.node_energies.__func__ is type(frozen).node_energies
                assert len(postprocesses) == 4
            assert not torch.cuda.is_initialized()
            records[name] = {
                "named_parameters": len(dict(frozen.named_parameters())),
                "so2_convolutions": len(convolutions), "frozen_method_count": expected,
                "plan_types": plans, "postprocess_paths": postprocesses,
                "state_dict_equal": True, "pure_class_descriptors_shared": True,
            }
    finally:
        torch.nn.Module.cuda = original_cuda
    runtime = sys.modules["fasteq_v5.cuda.jit"].runtime
    assert runtime.cache_info().currsize == 0
    result = {
        "status": "CPU-only constructor and adapter validation; no forward/native compile/performance",
        "constructor_cuda_override": "Module.cuda temporarily returns self, restored after construction",
        "cuda_context_initialized": torch.cuda.is_initialized(), "runtime_cache_entries": 0,
        "driver_sha256": sha256(Path(__file__)), "provenance": provenance, "models": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k not in (
        "plan_types", "postprocess_paths"
    )} for key, value in records.items()}, indent=2))


if __name__ == "__main__":
    main()
