"""Correctness-only call/adjoint observations; no profiler or timed-path hooks.

SIMT selection is proved from successful real calls and the source-verified
native dispatcher predicate, rather than from a flag alone. Native modulation
is identified by its C++ autograd Node, including actual first/second invocations.
No tensors or autograd Nodes are retained in the report.
"""

import hashlib
import os

import torch


class CandidateExecutionObservers:
    def __init__(self):
        self.report = {"scope": "correctness-only observations; no profiler, kernel timings or extra execution",
                       "indexed_linear_calls": [], "modulation_calls": [],
                       "native_modulation_first_adjoint_invocations": 0,
                       "native_modulation_second_adjoint_invocations": 0,
                       "simt_forward_source_proven_calls": 0,
                       "native_modulation_autograd_forward_calls": 0}
        self._originals, self._handles = [], []

    def _node(self, node):
        if node is None:
            return
        name = node.name()
        if name not in ("FastEqModulationBackward", "FastEqModulationDoubleBackward"):
            return
        phase = "first" if name == "FastEqModulationBackward" else "second"

        def prehook(gradients):
            self.report["native_modulation_" + phase + "_adjoint_invocations"] += 1

        def posthook(gradients, outputs):
            nodes = [value.grad_fn for value in gradients if value is not None and value.grad_fn is not None]
            for child in {id(child): child for child in nodes}.values():
                self._node(child)

        self._handles.extend((node.register_prehook(prehook), node.register_hook(posthook)))

    def install(self):
        import fasteq.ops.indexed_linear as linear
        from fasteq.ops import modulation

        original_linear = linear.indexed_linear

        def indexed(metadata, features, weight, node_type):
            result = original_linear(metadata, features, weight, node_type)
            _, _, _, experts, _, paths = linear._decode(metadata)
            selected = (features.dtype == torch.float32 and features.shape[0] >= 1024
                        and weight.shape[0] == 1 and bool(paths)
                        and all(path[3] >= 64 and path[4] >= 64 for path in paths)
                        and os.environ.get("FASTEQ_EXPERIMENTAL_LINEAR_SIMT") == "1")
            self.report["indexed_linear_calls"].append({
                "dtype": str(features.dtype), "features": list(features.shape), "weight": list(weight.shape),
                "experts": experts, "metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest(),
                "simt_forward_source_proven": selected,
                "proof": "successful real indexed_linear call + frozen current Python plan and C++ launch predicate; kernel duration not observed",
            })
            self.report["simt_forward_source_proven_calls"] += int(selected)
            return result

        original_modulation = modulation.ModulatedResidualPlan.__call__

        def modulated(plan, projected, gates, residual):
            result = original_modulation(plan, projected, gates, residual)
            name = result.grad_fn.name() if result.grad_fn is not None else None
            native = name == "FastEqModulationBackward"
            self.report["modulation_calls"].append({
                "dtype": str(projected.dtype), "features": list(projected.shape),
                "output_grad_fn": name, "native_autograd_node_observed": native,
                "native_cached_executables": [value.cached_executables for value in getattr(plan, "_native", {}).values()],
            })
            self.report["native_modulation_autograd_forward_calls"] += int(native)
            self._node(result.grad_fn)
            return result

        self._originals = [(linear, "indexed_linear", original_linear),
                           (modulation.ModulatedResidualPlan, "__call__", original_modulation)]
        linear.indexed_linear = indexed
        modulation.ModulatedResidualPlan.__call__ = modulated

    def close(self):
        for owner, name, original in self._originals:
            setattr(owner, name, original)
        self._originals.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
