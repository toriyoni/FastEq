"""Load only the frozen FastEq package in a dedicated diagnostic process."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

root = Path(os.environ["TACE_PROPHET_ROOT"])
archive = root / "performance/fasteq_optimization/v3_snapshot"
manifest = json.loads((archive / "manifest.json").read_text())
for row in manifest["files"]:
    assert (
        hashlib.sha256((archive / row["path"]).read_bytes()).hexdigest()
        == row["sha256"]
    ), row["path"]
assert not any(name == "fasteq" or name.startswith("fasteq.") for name in sys.modules)
mirror = root / "tmp/prophet_v3_profile_frozen_package"
if mirror.exists():
    shutil.rmtree(mirror)
shutil.copytree(archive / "src/FastEq", mirror)
# This existing legacy extension is untouched by the native FastEq optimization.
legacy = list((root / "src/FastEq/fasteq/cuda").glob("_cuda*.so"))
assert len(legacy) == 1
(mirror / "fasteq/cuda" / legacy[0].name).symlink_to(legacy[0])
# Keep the archived C++ implementation separate from concurrent live builds.
jit = mirror / "fasteq/cuda/jit.py"
source = jit.read_text()
assert 'name="fasteq_cuda_jit"' in source
jit.write_text(
    source.replace('name="fasteq_cuda_jit"', 'name="fasteq_cuda_jit_v3_profile"')
)
sys.path.insert(0, str(mirror))
import fasteq.adapters
from fasteq.adapters import adapt_model

assert Path(fasteq.adapters.__file__).is_relative_to(mirror)

stub = ModuleType("v3_snapshot")


def load_v3(_root):
    return adapt_model, {
        "archive": str(archive),
        "manifest": manifest,
        "loaded_package": str(mirror),
        "isolation": "fresh process; frozen complete FastEq package; no live FastEq module imports",
        "source_changes": "native extension module name only; legacy unmodified binary symlink",
        "legacy_extension": {
            "path": str(legacy[0]),
            "sha256": hashlib.sha256(legacy[0].read_bytes()).hexdigest(),
        },
    }


stub.load_v3 = load_v3
sys.modules["v3_snapshot"] = stub
from profile_prophet_v3 import main

main()
