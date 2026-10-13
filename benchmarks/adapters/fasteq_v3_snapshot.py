"""Verified historical FastEq v3 postprocess for isolated chain comparisons."""

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType


def load_v3_postprocess(root):
    archive = Path(root) / "performance/fasteq_optimization/v3_snapshot"
    manifest = json.loads((archive / "manifest.json").read_text())
    for row in manifest["files"]:
        assert hashlib.sha256((archive / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
    base = archive / "src/FastEq/fasteq"
    for suffix in ("", ".ops", ".triton", ".adapters"):
        name = "fasteq_v3" + suffix
        package = ModuleType(name)
        package.__path__ = [str(base / suffix[1:])] if suffix else [str(base)]
        sys.modules.setdefault(name, package)
    importlib.import_module("fasteq_v3.ops.gate")
    importlib.import_module("fasteq_v3.ops.equivariant_norm")
    source_path = base / "adapters/fusion.py"
    source = source_path.read_text()
    replacements = (
        ("from fasteq.ops.gate import GatePlan", "from fasteq_v3.ops.gate import GatePlan"),
        ("from .programs import NormPlan", "from fasteq_v3.ops.equivariant_norm import NormPlan"),
    )
    for old, new in replacements:
        assert old in source
        source = source.replace(old, new)
    module = ModuleType("fasteq_v3.adapters.fusion")
    module.__file__, module.__package__ = str(source_path), "fasteq_v3.adapters"
    sys.modules[module.__name__] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)  # noqa: S102
    return module.ProphetPostprocess, {"archive": str(archive),
        "verified_files": len(manifest["files"]),
        "source_changes": "Import namespaces only; Gate and Norm math/execution unchanged"}


__all__ = ["load_v3_postprocess"]
