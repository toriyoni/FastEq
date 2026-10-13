"""Profile-only evidence for the actual kernels bound to each native Linear plan."""

import hashlib
import re

import torch


def install():
    from fasteq.cuda import jit

    compiler = jit.runtime()
    cls = compiler.IndexedLinearPlan
    kernel_sources, plans, calls, restore = {}, {}, [], []
    state = {"profile_active": False}
    original_kernels, original_constructor = jit.kernels, compiler.IndexedLinearPlan

    def kernels(sources, *args, **kwargs):
        values = original_kernels(sources, *args, **kwargs)
        for source, kernel in values.items():
            role = re.search(r"#define FASTEQ_LINEAR_ROLE\s+(\d+)", source)
            kernel_sources[id(kernel)] = {"cuda_source_role": int(role.group(1)) if role else None,
                                          "cuda_source_sha256": hashlib.sha256(source.encode()).hexdigest()}
        return values

    def constructor(*args, **kwargs):
        plan = original_constructor(*args, **kwargs)
        plans[id(plan)] = {
            "kernel_slots": [kernel_sources.get(id(kernel), {}) for kernel in args[0]],
            "paths": args[1].detach().cpu().tolist(), "din": args[6], "dout": args[7],
            "weight_numel": args[8], "experts": args[9], "ir_mul": args[10],
            "legacy_tile_counts": [tensor.shape[0] for tensor in args[5]],
            "configured_tile_counts": [0, 0, 0], "max_angular_dim": 0,
        }
        return plan

    def configure(name, original):
        def wrapped(plan, *args):
            result = original(plan, *args)
            row = plans[id(plan)]
            if name == "configure_simt_tiles":
                row["configured_tile_counts"] = [tensor.shape[0] for tensor in args[:3]]
                row["max_angular_dim"] = args[3]
            elif name == "configure_tensor_core":
                row["configured_tile_counts"][2] = args[0].shape[0]
                row["max_angular_dim"] = args[1]
            else:
                row["configured_tile_counts"][:2] = [tensor.shape[0] for tensor in args]
            return result
        return wrapped

    original_forward = cls.forward

    def forward(plan, features, weight, node_types):
        description = plans[id(plan)]
        optimized = (weight.shape[0] == 1 and features.shape[0] >= 1024
                     and description["configured_tile_counts"][2] > 0 and len(description["kernel_slots"]) >= 6)
        slot = 5 if optimized else 2
        selected = description["kernel_slots"][slot]
        index = len(calls)
        row = {"id": index, "plan_id": id(plan), "profile_active": state["profile_active"],
               "features_shape": list(features.shape), "weight_shape": list(weight.shape),
               "dtype": str(features.dtype), "slot_selected_by_verified_cpp_dispatch": slot,
               **selected, "optimized_configured": optimized,
               "configured_tile_counts": description["configured_tile_counts"].copy(),
               "max_angular_dim": description["max_angular_dim"]}
        calls.append(row)
        with torch.profiler.record_function(f"linear_dispatch::{index}"):
            return original_forward(plan, features, weight, node_types)

    restore.extend(((jit, "kernels", original_kernels), (compiler, "IndexedLinearPlan", original_constructor),
                    (cls, "forward", original_forward)))
    jit.kernels, compiler.IndexedLinearPlan, cls.forward = kernels, constructor, forward
    for name in ("configure_simt_tiles", "configure_tensor_core", "configure_derivative_tensor_core"):
        original = getattr(cls, name)
        setattr(cls, name, configure(name, original))
        restore.append((cls, name, original))

    def undo():
        for owner, name, original in reversed(restore):
            setattr(owner, name, original)

    return {"scope": "actual source→resident Kernel binding→native plan constructor/tile configuration/Cpp.forward args; selection uses verified C++ dispatch and is checked against trace grid",
            "plans": plans, "forward_calls": calls}, state, undo
