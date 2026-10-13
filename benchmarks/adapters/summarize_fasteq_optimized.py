"""Audit and summarize one complete four-way FastEq model measurement.

Only the explicitly selected models.json supplies timing conclusions. Source
guards and paired-round statistics are audited before any output is written.
"""

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np

MODELS = ("tace", "tece", "prophet")
DTYPES = ("float32", "float64")
NODES = (64, 256)
MODES = ("energy", "forces", "force_loss")
BACKENDS = ("original", "mixed", "v3", "fasteq")
COMPARISONS = ("original_over_fasteq", "mixed_over_fasteq", "v3_over_fasteq", "original_over_mixed")
STATS = ("median", "q25", "q75", "min", "max")


def stats(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values), values
    return {"median": statistics.median(values),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "min": min(values), "max": max(values)}


def assert_statistics(stored, observed, label):
    for name in STATS:
        assert math.isclose(stored[name], observed[name], rel_tol=2e-10, abs_tol=2e-12), (
            label, name, stored[name], observed[name])


def case_key(row):
    return row["model"], row["dtype"], row["nodes"], row["mode"]


def source_audit(data, root, source_root=None):
    source_root = source_root or root
    manifest = data["metadata"]["fasteq_source_sha256"]
    assert manifest, "Missing production source hashes"
    for relative, expected in manifest.items():
        assert len(expected) == 64 and all(c in "0123456789abcdef" for c in expected)
        path = (source_root / relative).resolve()
        assert path.is_relative_to(source_root.resolve()), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    digest_cases = 0
    for row in data["results"]:
        if "fasteq_source_sha256_digest" in row:
            assert row["fasteq_source_sha256_digest"] == digest, case_key(row)
            digest_cases += 1
        if "fasteq_source_sha256" in row:
            assert row["fasteq_source_sha256"] == manifest, case_key(row)
    driver_root = source_root / "benchmark_driver"
    if not driver_root.is_dir():
        driver_root = root / "src/FastEq/benchmarks/adapters"
    driver = driver_root / "benchmark_fasteq_optimized_models.py"
    recorded_drivers = data["metadata"].get("benchmark_driver_sha256", {})
    for name, expected in recorded_drivers.items():
        assert hashlib.sha256((driver_root/name).read_bytes()).hexdigest() == expected, name
    driver_bytes = driver.read_bytes()
    tree = ast.parse(driver_bytes.decode())
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)
              and ast.unparse(node.test) == "source_hashes() == frozen_sources"]
    assert len(guards) >= 2, "Benchmark must check the complete source manifest before/after a case"
    return {"verified_source_files": len(manifest), "manifest_sha256": digest,
        "verified_source_root": str(source_root),
        "source_files_match_recorded_manifest": True,
        "per_case_digest_recorded_and_verified": digest_cases,
        "per_case_digest_not_recorded": len(data["results"])-digest_cases,
        "case_consistency_basis": "Driver asserts the complete source manifest before and after every case; resumed input requires an identical metadata manifest. No missing per-case digest is fabricated.",
        "driver_source_audited_at_summary": str(driver),
        "driver_sha256_at_summary": hashlib.sha256(driver_bytes).hexdigest(),
        "recorded_driver_hashes_verified": recorded_drivers,
        "recorded_cpu_affinity": data["metadata"].get("cpu_affinity"),
        "driver_source_guard_assertions": len(guards)}


def audit_rows(data):
    rows = data["results"]
    required = set(itertools.product(MODELS, DTYPES, NODES, MODES))
    keys = [case_key(row) for row in rows]
    assert len(rows) == len(set(keys)) == len(required) == 36, "Require 36 unique completed model cases"
    assert set(keys) == required, ("Unexpected or missing cases", required-set(keys), set(keys)-required)
    rounds = data["metadata"]["protocol"]["rounds"]
    assert rounds >= 3
    assert data["metadata"]["protocol"]["warmup"] > 0
    assert data["metadata"]["protocol"]["iterations"] > 0
    known_devices = {run["visible_device"] for run in data["metadata"]["runs"]}
    assert known_devices
    for row in rows:
        assert row["visible_device"] in known_devices, case_key(row)
        for backend in BACKENDS:
            samples = row[backend]["samples"]
            assert len(samples) == rounds, (case_key(row), backend, len(samples), rounds)
            for field in ("wall_ms", "cuda_event_ms"):
                assert_statistics(row[backend][field], stats([s[field] for s in samples]),
                                  (case_key(row), backend, field))
        for comparison in COMPARISONS:
            numerator, denominator = comparison.split("_over_")
            ratios = [a["wall_ms"]/b["wall_ms"]
                      for a,b in zip(row[numerator]["samples"], row[denominator]["samples"], strict=True)]
            assert_statistics(row["paired_speedup"][comparison], stats(ratios),
                              (case_key(row), comparison))
    return sorted(rows, key=lambda r: (MODELS.index(r["model"]), DTYPES.index(r["dtype"]),
                                      r["nodes"], MODES.index(r["mode"])))


def flatten(row):
    result = {key: row[key] for key in ("model", "dtype", "nodes", "mode", "visible_device")}
    for backend in BACKENDS:
        for name in STATS:
            suffix = "" if name == "median" else "_" + name
            result[f"{backend}_wall{suffix}_ms"] = row[backend]["wall_ms"][name]
        for name in ("median", "q25", "q75"):
            suffix = "" if name == "median" else "_" + name
            result[f"{backend}_cuda_event{suffix}_ms"] = row[backend]["cuda_event_ms"][name]
    for comparison in COMPARISONS:
        for name in STATS:
            result[f"{comparison}_paired_{name}"] = row["paired_speedup"][comparison][name]
    return result


def ranges(rows):
    result = {}
    for comparison in COMPARISONS:
        values = [row["paired_speedup"][comparison]["median"] for row in rows]
        result[comparison] = {"min": min(values), "max": max(values), "case_count": len(rows),
            "median_below_one_cases": sum(v < 1 for v in values),
            "IQR_contains_one_cases": sum(row["paired_speedup"][comparison]["q25"] <= 1
                <= row["paired_speedup"][comparison]["q75"] for row in rows),
            "min_cases": [list(case_key(row)) for row,v in zip(rows,values,strict=True) if v == min(values)],
            "max_cases": [list(case_key(row)) for row,v in zip(rows,values,strict=True) if v == max(values)]}
    return result


def validation_maxima(rows):
    result = {}
    for dtype in DTYPES:
        result[dtype] = {}
        for backend in ("mixed", "v3", "fasteq"):
            result[dtype][backend] = {}
            for metric in ("energy", "forces", "gradient_relative_l2"):
                selected = [row["validation"][backend][metric] for row in rows
                    if row["dtype"] == dtype and metric in row["validation"][backend]]
                expected = {"energy":18, "forces":12, "gradient_relative_l2":6}[metric]
                assert len(selected) == expected, (dtype, backend, metric, len(selected), expected)
                assert all(math.isfinite(v) and v >= 0 for v in selected)
                result[dtype][backend][metric] = {"max": max(selected), "case_count": len(selected)}
    return result


def summarize(data, root, source, status, source_root=None):
    rows = audit_rows(data)
    audit = source_audit(data, root, source_root)
    flat = [flatten(row) for row in rows]
    return rows, flat, {"status": status, "source": str(source), "case_count": len(rows),
        "metadata": data["metadata"], "source_audit": audit,
        "definition": "Speedups are within-round numerator/denominator wall ratios; report their median and [q25,q75]. For *_over_fasteq, values above one mean optimized FastEq is faster; original_over_mixed compares original with mixed. IQR is sample spread, not a confidence interval. CUDA events are stream timelines including dispatch gaps, not summed kernel work.",
        "paired_statistics_recomputed_and_verified": True,
        "N256": [row for row in flat if row["nodes"] == 256],
        "ranges": ranges(rows),
        "ranges_by_model": {model:ranges([r for r in rows if r["model"] == model]) for model in MODELS},
        "ranges_by_mode": {mode:ranges([r for r in rows if r["mode"] == mode]) for mode in MODES},
        "ranges_by_model_mode": {model:{mode:ranges([r for r in rows if r["model"] == model and r["mode"] == mode])
                                  for mode in MODES} for model in MODELS},
        "ranges_by_dtype": {dtype:ranges([r for r in rows if r["dtype"] == dtype]) for dtype in DTYPES},
        "validation_maxima": validation_maxima(rows),
        "regressions": {comparison:[flatten(r) for r in rows if r["paired_speedup"][comparison]["median"] < 1]
                        for comparison in ("v3_over_fasteq", "mixed_over_fasteq", "original_over_fasteq")},
        "exclusions": "No timing values imported from snapshots, smoke runs, microbenchmarks or profiler traces."}


def plots(rows, metadata, output, status):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = metadata["model_config"]
    hardware = f"{metadata['gpu']} / C{config['channels']} / N256"
    labels = {"original":"Original", "mixed":"Mixed CUDA", "v3":"Frozen FastEq v3", "fasteq":"Optimized FastEq"}
    colors = {"original":"#7b8794", "mixed":"#568ac4", "v3":"#f2a65a", "fasteq":"#278e7b"}
    lookup = {case_key(r):r for r in rows}
    x = np.arange(len(MODES))
    for kind in ("latency", "speedup"):
        fig, axes = plt.subplots(2,3,figsize=(14,7.5),sharey="row")
        for i,dtype in enumerate(DTYPES):
            for j,model in enumerate(MODELS):
                ax=axes[i,j]
                selected=[lookup[model,dtype,256,mode] for mode in MODES]
                if kind == "latency":
                    keys=BACKENDS
                    metrics=[{backend:r[backend]["wall_ms"] for backend in keys} for r in selected]
                    names=labels
                    palette=colors
                    width=.20
                else:
                    keys=("v3_over_fasteq","mixed_over_fasteq")
                    metrics=[r["paired_speedup"] for r in selected]
                    names={"v3_over_fasteq":"v3 / optimized", "mixed_over_fasteq":"mixed / optimized"}
                    palette={"v3_over_fasteq":colors["v3"], "mixed_over_fasteq":colors["mixed"]}
                    width=.32
                for k,key in enumerate(keys):
                    ys=[r[key]["median"] for r in metrics]
                    low=[y-r[key]["q25"] for y,r in zip(ys,metrics,strict=True)]
                    high=[r[key]["q75"]-y for y,r in zip(ys,metrics,strict=True)]
                    ax.bar(x+(k-(len(keys)-1)/2)*width,ys,width*.95,
                           label=names[key],color=palette[key],yerr=[low,high],capsize=2)
                ax.set_xticks(x,("Energy","Forces","Force loss"))
                ax.set_title(f"{model.upper()} / {dtype.upper()}")
                ax.grid(axis="y",alpha=.15)
                ax.set_axisbelow(True)
                if kind == "speedup":
                    ax.axhline(1,color="#444444",linewidth=.8,linestyle="--")
                if j == 0:
                    ax.set_ylabel("Wall latency (ms; lower is faster)" if kind == "latency" else "Paired wall speedup (higher is faster)")
        handles,names=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,names,loc="upper center",ncol=4 if kind=="latency" else 2,frameon=False)
        fig.suptitle(f"{status.upper()}: {hardware}; median and IQR\nPrebuilt neighbors; force loss excludes optimizer update and stress",y=.93)
        fig.tight_layout(rect=(0,0,1,.87))
        for suffix in ("png","pdf"):
            fig.savefig(output/f"model_{kind}.{suffix}",dpi=180)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--source-root",type=Path,
                        help="Verify production hashes against this saved source tree instead of live files")
    parser.add_argument("--status",choices=("candidate","final"),default="candidate")
    args=parser.parse_args()
    target=args.root/"performance/fasteq_optimization"
    source=args.source or target/"models_final/models.json"
    output=args.output or target
    payload=source.read_bytes()
    data=json.loads(payload)
    rows,flat,summary=summarize(data,args.root,source,args.status,args.source_root)
    summary["source_sha256"] = hashlib.sha256(payload).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    for filename,selected in (("models.csv",flat),("N256.csv",summary["N256"])):
        with (output/filename).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(selected)
    summary["artifacts"]={key:str(output/key) for key in (
        "models.csv","N256.csv","summary.json","model_latency.png","model_latency.pdf",
        "model_speedup.png","model_speedup.pdf")}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    plots(rows,data["metadata"],output,args.status)
    print(json.dumps({"status":args.status,"source":str(source),"case_count":len(rows),
        "source_audit":summary["source_audit"],"ranges":summary["ranges"]},indent=2))


if __name__ == "__main__":
    main()
