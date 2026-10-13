"""CPU-only trace attribution and nonoverlapping CPU interval accounting."""

import argparse
import bisect
import heapq
import json
from collections import defaultdict
from pathlib import Path


class Ranges:
    def __init__(self, events):
        self.threads = defaultdict(list)
        for event in events:
            self.threads[event["tid"]].append(event)
        self.index = {}
        for thread, values in self.threads.items():
            values.sort(key=lambda e: e["ts"])
            ends = []
            maximum = 0
            for event in values:
                maximum = max(maximum, event["ts"] + event["dur"])
                ends.append(maximum)
            self.index[thread] = (values, [e["ts"] for e in values], ends)

    def enclosing(self, event, skip=None):
        if event["tid"] not in self.index:
            return None
        values, starts, ends = self.index[event["tid"]]
        end = event["ts"] + event.get("dur", 0)
        i = bisect.bisect_right(starts, event["ts"]) - 1
        best = None
        while i >= 0 and ends[i] >= end:
            candidate = values[i]
            if (
                candidate is not skip
                and candidate["ts"] + candidate["dur"] >= end
                and (best is None or candidate["dur"] < best["dur"])
            ):
                best = candidate
            i -= 1
        return best


def label(event):
    _, family, path, phase = event["name"].split("::", 3)
    return family, path, phase


def mechanism(name):
    text = name.lower()
    if "uniform1d" in text:
        return "tensor_product_cuda"
    if name == "run":
        return "native_generated_cuda"
    if name in ("forward", "backward", "double_backward"):
        return "fused_gate_norm_cuda"
    if "gate_" in text:
        return "standalone_gate_cuda"
    if name in (
        "_forward_kernel",
        "_backward_kernel",
        "_double_kernel",
        "_tree_reduce_rows",
    ):
        return "norm_cuda"
    if name in ("_forward", "_backward", "_double", "_second_x", "_second_g"):
        return "ffn_cuda"
    if any(
        x in text
        for x in ("indexing", "index_", "radix", "cub::", "scatter", "sort", "nonzero")
    ):
        return "index_scatter_sort"
    if any(x in text for x in ("direct_copy", "copy_kernel")):
        return "copy_kernel"
    if any(
        x in text
        for x in ("gemm", "cutlass", "ampere", "volta", "hgemm", "dgemm", "sgemm")
    ):
        return "dense_gemm"
    if any(x in text for x in ("reduce", "reduction", "layernorm", "layer_norm")):
        return "reduction_norm"
    if any(x in text for x in ("elementwise", "fill", "pointwise")):
        return "elementwise"
    if "spherical_harmonic" in text:
        return "spherical_harmonics"
    return "other_kernel"


def component(event):
    name = event["name"].lower()
    if event.get("cat") in ("cuda_driver", "cuda_runtime"):
        if "launch" in name:
            return "cuda_launch_api"
        if "sync" in name or "event" in name:
            return "cuda_sync_event_api"
        if "memcpy" in name or "memset" in name:
            return "cuda_copy_memset_api"
        if "alloc" in name or "free" in name:
            return "cuda_allocator_api"
        return "cuda_other_api"
    if event["name"].startswith("profile_prophet_v4::"):
        return "python_unscoped_and_wait"
    if event["name"].startswith("family::"):
        return "python_family_wrapper"
    if event["name"].startswith("block::"):
        return "python_block_wrapper"
    if "autograd::engine" in name:
        return "autograd_engine"
    if any(x in name for x in ("aten::empty", "aten::resize", "aten::zeros")):
        return "tensor_allocation_op"
    if "nonzero" in name:
        return "dynamic_nonzero_op"
    if any(
        x in name
        for x in ("aten::copy", "aten::clone", "aten::_to_copy", "aten::contiguous")
    ):
        return "tensor_copy_op"
    return "framework_operator"


def analyze(case, profile):
    events = json.loads(Path(profile["trace"]).read_text())["traceEvents"]
    cpu = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation")
    ]
    runtime = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("cuda_driver", "cuda_runtime")
    ]
    annotations = Ranges([e for e in cpu if e["name"].startswith("family::")])
    backwards = Ranges([e for e in cpu if "Backward" in e["name"]])
    sequence = {}
    event_tags = {}
    for event in cpu:
        annotation = annotations.enclosing(event)
        if annotation:
            event_tags[id(event)] = label(annotation)
    for _ in range(3):
        for event in cpu:
            number = event.get("args", {}).get("Sequence number")
            if (
                number is not None
                and number >= 0
                and not event["name"].startswith("autograd::engine")
                and id(event) in event_tags
                and event.get("args", {}).get("Fwd thread id", 0) == 0
            ):
                sequence[number] = event_tags[id(event)]
        for event in cpu:
            if id(event) in event_tags:
                continue
            number = event.get("args", {}).get("Sequence number")
            if "Backward" in event["name"] and number in sequence:
                family, path, phase = sequence[number]
                event_tags[id(event)] = (
                    family,
                    path,
                    "second" if phase in ("first", "second") else "first",
                )
            else:
                ancestor = backwards.enclosing(event, skip=event)
                if ancestor and id(ancestor) in event_tags:
                    event_tags[id(event)] = event_tags[id(ancestor)]

    def tag(event):
        annotation = annotations.enclosing(event)
        if annotation:
            return (*label(annotation), "annotation")
        if id(event) in event_tags:
            return (*event_tags[id(event)], "forward_sequence")
        ancestor = backwards.enclosing(event)
        if ancestor and id(ancestor) in event_tags:
            return (*event_tags[id(ancestor)], "backward_ancestor")
        return "unscoped_framework", "", "unknown", "unscoped"

    by_external = {
        e["args"]["External id"]: e for e in cpu if "External id" in e.get("args", {})
    }
    by_correlation = {
        e["args"]["correlation"]: e
        for e in runtime
        if "correlation" in e.get("args", {})
    }
    families = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    mechanisms = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    source_owners = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    methods = defaultdict(int)
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        launch = by_correlation.get(event.get("args", {}).get("correlation"))
        owner = by_external.get(event.get("args", {}).get("External id"))
        info = (
            tag(launch)
            if launch
            else ("unscoped_framework", "", "unknown", "missing_launch")
        )
        if info[0] == "unscoped_framework" and owner:
            info = tag(owner)
        family, path, phase, method = info
        if family == "unscoped_framework":
            name = event["name"]
            op = owner["name"] if owner else ""
            if "uniform1d" in name.lower():
                family = "tensor_product"
                method = "known_specialized_kernel"
            elif "IndexedLinear" in op or "indexed_linear" in op:
                family = "eq_linear"
                method = "known_native_operator"
            elif "spherical_harmonic" in name:
                family = "sh_rotation"
                method = "known_specialized_kernel"
        methods[method] += 1
        key = (family, phase)
        families[key]["count"] += 1
        families[key]["duration_us"] += event["dur"]
        mech = mechanism(event["name"])
        mechanisms[mech]["count"] += 1
        mechanisms[mech]["duration_us"] += event["dur"]
        key = (
            family,
            path,
            phase,
            owner["name"] if owner else "",
            str(owner.get("args", {}).get("Input Dims", [])) if owner else "",
            event["name"],
        )
        source_owners[key]["count"] += 1
        source_owners[key]["duration_us"] += event["dur"]
    # Split every CPU thread at all interval boundaries. The innermost active
    # complete event gets the segment exactly once, including driver/runtime.
    threads = defaultdict(list)
    for event in (*cpu, *runtime):
        threads[event["tid"]].append(event)
    cpu_components = defaultdict(float)
    cpu_logical = defaultdict(float)
    cpu_method = defaultdict(float)
    cpu_threads = {}
    for thread, items in threads.items():
        thread_components = defaultdict(float)
        boundaries = []
        for i, event in enumerate(items):
            boundaries.extend(((event["ts"], 1, i), (event["ts"] + event["dur"], 0, i)))
        boundaries.sort()
        active = set()
        heap = []
        previous = boundaries[0][0]
        for timestamp, action, i in boundaries:
            while heap and heap[0][2] not in active:
                heapq.heappop(heap)
            if timestamp > previous and heap:
                owner = items[heap[0][2]]
                duration = timestamp - previous
                family, _, phase, method = tag(owner)
                cpu_components[component(owner)] += duration
                thread_components[component(owner)] += duration
                cpu_logical[(family, phase)] += duration
                cpu_method[method] += duration
            if action:
                active.add(i)
                heapq.heappush(heap, (items[i]["dur"], -items[i]["ts"], i))
            else:
                active.discard(i)
            previous = timestamp
        cpu_threads[str(thread)] = {
            "components_us": dict(thread_components),
            "elapsed_union_us": sum(thread_components.values()),
        }
    return {
        "backend": profile["backend"],
        "dtype": case["dtype"],
        "mode": case["mode"],
        "normal_wall_ms": case["normal_timing"][profile["backend"]]["wall_ms"],
        "exclusive_kernel_families": [
            {"family": k[0], "phase": k[1], **v}
            for k, v in sorted(
                families.items(), key=lambda x: x[1]["duration_us"], reverse=True
            )
        ],
        "exclusive_kernel_mechanisms": [
            {"mechanism": k, **v}
            for k, v in sorted(
                mechanisms.items(), key=lambda x: x[1]["duration_us"], reverse=True
            )
        ],
        "kernel_source_owners": [
            {
                "family": k[0],
                "module_path": k[1],
                "phase": k[2],
                "owner": k[3],
                "input_dims": k[4],
                "kernel": k[5],
                **v,
            }
            for k, v in sorted(
                source_owners.items(), key=lambda x: x[1]["duration_us"], reverse=True
            )
        ],
        "attribution_methods": dict(methods),
        "exclusive_cpu_thread_elapsed_us": dict(cpu_components),
        "exclusive_cpu_threads": cpu_threads,
        "exclusive_cpu_logical_families": [
            {"family": k[0], "phase": k[1], "duration_us": v}
            for k, v in sorted(cpu_logical.items(), key=lambda x: x[1], reverse=True)
        ],
        "cpu_attribution_methods_us": dict(cpu_method),
        "kernel_count": profile["kernel_count"],
        "kernel_sum_us": profile["sum_kernel_duration_us"],
        "timeline": profile["stream_timeline"],
        "cuda_memory": profile["cuda_memory"],
        "transfers": profile["transfers"],
        "notes": [
            "CPU exclusive numbers partition each thread elapsed intervals; threads can overlap, so totals may exceed whole-step wall time.",
            "GPU families and mechanisms are each independent exhaustive disjoint partitions; do not add the two views.",
            "Single-step instrumented gaps/work are diagnostic; speedups use normal paired rounds only.",
            "Unscoped framework activity stays explicit; family inference uses actual annotations, forward sequence or named specialized operator only.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    raw = json.loads((args.directory / "profile.json").read_text())
    rows = []
    for case in raw["results"]:
        for profile in case["profiles"]:
            result = analyze(case, profile)
            assert (
                sum(row["count"] for row in result["exclusive_kernel_families"])
                == profile["kernel_count"]
            )
            assert (
                abs(
                    sum(
                        row["duration_us"]
                        for row in result["exclusive_kernel_families"]
                    )
                    - profile["sum_kernel_duration_us"]
                )
                < 1e-6
            )
            rows.append(result)
            print(
                case["dtype"],
                case["mode"],
                profile["backend"],
                result["attribution_methods"],
                flush=True,
            )
    (args.directory / "analysis.json").write_text(
        json.dumps({"source": "profile.json", "results": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
