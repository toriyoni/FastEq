"""One guarded TECE regression profile; no normal benchmark timing is measured."""

import argparse
import gc
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

for flag in ("FASTEQ_EXPERIMENTAL_LINEAR_SIMT", "FASTEQ_EXPERIMENTAL_NATIVE_MODULATION"):
    os.environ.setdefault(flag, "1")
    assert os.environ[flag] == "1", flag

import profile_tece_v4 as instrumentation
import torch
from benchmark_scaling_case import ExclusiveGuard, atomic_json, verify_manifest
from profile_tece64 import event_rows, shape
from scaling_model_case import build_case
from tece_native_dispatch_observer_20261013 import install as install_dispatch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_metadata():
    """Call metadata only; parent family scopes retain exclusive attribution."""
    import fasteq.ops.segmented_product as segmented
    import fasteq.ops.uniform1d_jit as uniform

    records, restore = [], []

    def wrapper(function, name):
        def call(*args, **kwargs):
            index = len(records)
            row = {"id": index, "kind": name}
            if name == "segmented_execute":
                metadata, role, required, rows, values = args
                dimensions, terms, _ = segmented._parse(metadata)
                row.update(role=role, required=required, rows=rows, dimensions=dimensions,
                           terms=len(terms), modes=sorted({term[0] for term in terms}),
                           metadata_sha256=hashlib.sha256(metadata.encode()).hexdigest(), values=shape(values))
            elif name == "uniform_forward":
                row.update(inputs=shape(args[:3]), u_dim=args[5]["u_dim"],
                           out_seg_num=args[5]["out_seg_num"], paths=args[5]["i_list"].numel())
            else:
                row.update(arguments={key: shape(value) for key, value in kwargs.items()
                                      if key in ("w", "x", "y", "grad_out", "grad_grad_w", "grad_grad_x", "grad_grad_y",
                                                 "grad_w", "mode", "u_dim", "out_seg_num")})
            records.append(row)
            with torch.profiler.record_function(f"kernelmeta::{name}::{index}"):
                result = function(*args, **kwargs)
            row["outputs"] = shape(result)
            return result
        return call

    for module, attribute, name in ((segmented, "_execute", "segmented_execute"),
                                    (uniform, "fast_uniform1d_jit", "uniform_forward"),
                                    (uniform, "_run_bwd", "uniform_first"),
                                    (uniform, "_run_double_bwd", "uniform_double")):
        original = getattr(module, attribute)
        setattr(module, attribute, wrapper(original, name))
        restore.append((module, attribute, original))

    def undo():
        for module, attribute, original in restore:
            setattr(module, attribute, original)
    return records, undo


def classifier(model):
    old = instrumentation.family
    modules = dict(model.named_modules())
    linear_paths = {path for path, module in modules.items()
                    if old(module) == "eq_linear" or type(module).__name__ in ("IndexedLinear", "MoEIndexedLinear")}
    paths = {id(module): path for path, module in modules.items()}

    def family(module):
        path = paths.get(id(module), "")
        if any(path == parent or path.startswith(parent + ".") for parent in linear_paths):
            return "eq_linear"
        plan = getattr(module, "_fasteq_plan", None)
        if type(plan).__name__ == "TensorProductPlan":
            return "tensor_product/segmented" if plan.vector else "tensor_product/uniform1d"
        if (type(module).__module__, type(module).__name__) == ("tace.models._e3nn.tece_oam_rra", "Convolution"):
            return "convolution_unpartitioned"
        return old(module)
    return old, family


def enriched_partition(trace, records, dispatch):
    result = instrumentation.partition(trace)
    events = json.loads(trace.read_text())["traceEvents"]
    launches = {event["args"]["correlation"]: event for event in events
                if event.get("ph") == "X" and "LaunchKernel" in event.get("name", "")
                and "correlation" in event.get("args", {})}
    annotations = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") == "user_annotation" and event.get("name", "").startswith(("kernelmeta::", "linear_dispatch::")):
            annotations[event["tid"]].append(event)
    excluded = [row for row in result["kernel_rows"] if row["owner"].startswith("family::profile_primer::")]
    result["kernel_rows"] = [row for row in result["kernel_rows"] if row not in excluded]
    result["exclusive_gpu_families"] = [row for row in result["exclusive_gpu_families"] if row["family"] != "profile_primer"]
    result["primer_excluded_kernel_count"] = len(excluded)
    groups = defaultdict(lambda: {"count": 0, "gpu_us": 0.0})
    for row in result["kernel_rows"]:
        launch = launches.get(row["args"].get("correlation"))
        matches = [] if launch is None else [item for item in annotations[launch["tid"]]
                  if item["ts"] <= launch["ts"] and item["ts"] + item["dur"] >= launch["ts"] + launch["dur"]]
        meta = min(matches, key=lambda item: item["dur"])["name"] if matches else None
        row["kernel_metadata_scope"] = meta
        if meta is not None and meta.startswith("kernelmeta::"):
            index = int(meta.rsplit("::", 1)[1])
            records[index]["gpu_us"] = records[index].get("gpu_us", 0.0) + row["duration_us"]
            records[index]["kernel_count"] = records[index].get("kernel_count", 0) + 1
        elif meta is not None:
            index = int(meta.rsplit("::", 1)[1])
            call = dispatch["forward_calls"][index]
            row["linear_dispatch_call"] = index
            row["bound_cuda_source_role"] = call["cuda_source_role"]
            call.setdefault("trace_kernels", []).append({"name": row["name"], "gpu_us": row["duration_us"],
                                                        "grid": row["args"].get("grid"), "block": row["args"].get("block")})
        key = row["owner"], row["name"], json.dumps(row["args"].get("grid")), json.dumps(row["args"].get("block"))
        groups[key]["count"] += 1
        groups[key]["gpu_us"] += row["duration_us"]
    result["grouped_kernels"] = sorted([{"owner": key[0], "name": key[1], "grid": json.loads(key[2]),
                                           "block": json.loads(key[3]), **value} for key, value in groups.items()],
                                       key=lambda row: row["gpu_us"], reverse=True)
    result["kernel_sum_us"] = sum(row["duration_us"] for row in result["kernel_rows"])
    result["matched_kernel_count"] = sum(row["owner"] != "unclassified" and not row["owner"].startswith("family::unclassified::") for row in result["kernel_rows"])
    result["unclassified_kernel_count"] = len(result["kernel_rows"]) - result["matched_kernel_count"]
    direct = [row for row in result["kernel_rows"] if row["owner"] != "unclassified" and "::aten_fallback::" not in row["owner"]]
    result["direct_annotation_kernel_count"] = len(direct)
    result["direct_annotation_kernel_duration_us"] = sum(row["duration_us"] for row in direct)
    result["cpu_annotation_calls"] = [row for row in result["cpu_annotation_calls"] if not row["name"].startswith("family::profile_primer::")]
    intervals = sorted((row["ts"], row["ts"] + row["duration_us"]) for row in result["kernel_rows"])
    end, busy = None, 0.0
    for first, last in intervals:
        busy += last - max(first, end or first) if end is None or last > end else 0
        end = max(end or last, last)
    span = max(stop for _, stop in intervals) - intervals[0][0] if intervals else 0
    result.update(kernel_union_busy_us=busy, first_to_last_kernel_timeline_us=span, timeline_gap_us=span - busy)
    result["exclusive_annotated_cpu_intervals_us"].pop("profile_primer", None)
    for row in result["exclusive_gpu_families"]:
        if row["category"] == "kernel":
            row["kernel_sum_fraction"] = row["duration_us"] / result["kernel_sum_us"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--mode", choices=("energy", "forces", "force_loss"), required=True)
    parser.add_argument("--backend", choices=("original", "v6"), required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--simt-off-ablation", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.warmup <= 3
    uuid = "GPU-3a6d868b-775e-3236-988f-2772678876f2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == uuid, "This worker owns GPU2 only"
    os.sched_setaffinity(0, {170})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), "Use a fresh output path"
    manifest = args.root / "performance/fasteq_optimization/v6/scaling_20261013/manifest.json"
    source = verify_manifest(manifest)
    own_dependencies = {str(Path(__file__).resolve()): digest(__file__),
                        str(Path(instrumentation.__file__)): digest(instrumentation.__file__),
                        str(Path(__file__).with_name("tece_native_dispatch_observer_20261013.py")): digest(Path(__file__).with_name("tece_native_dispatch_observer_20261013.py"))}
    guard = ExclusiveGuard(uuid)
    before = guard.check()

    def check():
        guard.check()
        assert verify_manifest(manifest) == source
        assert {path: digest(path) for path in own_dependencies} == own_dependencies

    case = build_case("tece", args.nodes, args.dtype, "cuda", args.mode, variant=args.backend)
    if args.simt_off_ablation:
        assert args.backend == "v6"
        os.environ["FASTEQ_EXPERIMENTAL_LINEAR_SIMT"] = "0"
    dispatch, dispatch_state, undo_dispatch = install_dispatch()
    for _ in range(args.warmup):
        result = case.run(args.backend)
        del result
    case.clear_gradients()
    torch.cuda.synchronize()
    check()
    model = case.models[args.backend]
    old_classifier, selected = classifier(model)
    instrumentation.family = selected
    descriptors, nodes, undo = instrumentation.annotate(model, args.backend, args.mode)
    records, undo_metadata = install_metadata()
    torch.cuda.reset_peak_memory_stats()
    trace = args.output.with_suffix(".trace.json")
    dispatch_state["profile_active"] = True
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=False, with_stack=False) as profiler:
            with torch.profiler.record_function("family::profile_primer::tiny_torch::excluded"):
                primer = torch.ones(1, device="cuda").add_(1)
                torch.cuda.synchronize()
                del primer
            with torch.profiler.record_function("profile_step::" + args.backend):
                result = case.run(args.backend)
            torch.cuda.synchronize()
    finally:
        undo_metadata()
        undo()
        instrumentation.family = old_classifier
        undo_dispatch()
    check()
    profiler.export_chrome_trace(str(trace))
    partition = enriched_partition(trace, records, dispatch)
    math_path = args.output.with_suffix(".math.pt")
    torch.save({"outputs": {key: value.detach().cpu() if value is not None else None for key, value in result.items()},
                "parameter_gradients": {key: parameter.grad.detach().cpu() if parameter.grad is not None else None
                                        for key, parameter in model.named_parameters()}}, math_path)
    del result
    report = {"status": "guarded_profile_complete", "metadata": case.metadata,
              "gpu_before": before, "gpu_after": guard.check(), "source_guard": source,
              "profile_driver_sha256": own_dependencies, "profile_trace_sha256": digest(trace),
              "cpu_affinity": [170], "warmup_excluded": args.warmup,
              "profile_executions": 1, "normal_timing_measured": False,
              "variant": "v6_simt_off_ablation" if args.simt_off_ablation else args.backend,
              "ablation_not_formal_v6": args.simt_off_ablation,
              "effective_flags": {key: os.environ[key] for key in ("FASTEQ_EXPERIMENTAL_LINEAR_SIMT", "FASTEQ_EXPERIMENTAL_NATIVE_MODULATION")},
              "module_descriptors": descriptors, "node_records": nodes, "kernel_metadata_calls": records,
              "native_linear_dispatch": dispatch, "math_snapshot_path": str(math_path), "math_snapshot_sha256": digest(math_path),
              "shape_groups": event_rows(profiler), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "attribution_note": "CUDA correlation→runtime/driver launch→deepest same-thread CPU family annotation; GPU annotations excluded. CPU nested inclusive times are not summed. Separate kernelmeta scopes preserve parent family and attach actual roles/shapes.",
              **partition}
    atomic_json(args.output, report)
    check()
    print(json.dumps({"backend": report["variant"], "nodes": args.nodes, "dtype": args.dtype, "mode": args.mode,
                      "kernel_sum_us": report["kernel_sum_us"], "families": sorted(
                          [row for row in report["exclusive_gpu_families"] if row["category"] == "kernel"],
                          key=lambda row: row["duration_us"], reverse=True)[:8]}, indent=2), flush=True)
    del model, case
    gc.collect()


if __name__ == "__main__":
    main()
