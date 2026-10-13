"""Verified mixed FastEq/EQX implementation beside FastEq-owned adapters.

The unchanged EQX implementation is shared only by the historical baseline.
Production FastEq operators and the archived scalar executor have distinct
dispatcher namespaces. Historical TACE methods are bound per instance.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import MethodType, ModuleType


def load_mixed(root):
    root = Path(root)
    archive = root / "performance/fasteq_reuse/mixed_baseline"
    manifest = json.loads((archive / "manifest.json").read_text())
    for row in manifest["files"]:
        data = (archive / row["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
        if (
            row["path"].startswith("src/tace/eqx/")
            or row["path"] == "src/prophet/prophet/model.py"
        ):
            assert (root / row["path"]).read_bytes() == data, row["path"]

    def module(name, relative, replacements=(), package=None):
        path = archive / relative
        value = ModuleType(name)
        value.__file__ = str(path)
        value.__package__ = package or name.rpartition(".")[0]
        source = path.read_text()
        for old, new in replacements:
            assert old in source, (relative, old)
            source = source.replace(old, new)
        sys.modules[name] = value
        exec(compile(source, str(path), "exec"), value.__dict__)  # noqa: S102
        return value

    package = ModuleType("fasteq_mixed")
    package.__path__ = [str(archive / "src/FastEq/fasteq/adapters")]
    sys.modules[package.__name__] = package
    loaded = []
    for name in ("cache", "harmonics", "programs", "tensor_product", "fusion"):
        replacements = (
            [("fasteq::scalar_expression", "fasteq_mixed::scalar_expression")]
            if name == "harmonics"
            else []
        )
        loaded.append(
            module(
                "fasteq_mixed." + name,
                "src/FastEq/fasteq/adapters/" + name + ".py",
                replacements,
            ).__file__
        )
    adapter = module(
        "fasteq_mixed", "src/FastEq/fasteq/adapters/__init__.py", package="fasteq_mixed"
    )
    adapter.__path__ = package.__path__
    linear = module(
        "tace.models._fasteq_mixed_linear", "src/tace/tace/models/linear.py"
    )
    tece = module(
        "tace.models._e3nn._fasteq_mixed_tece",
        "src/tace/tace/models/_e3nn/tece_oam_rra.py",
    )

    def adapt(model):
        result = adapter.adapt_model(model)
        for child in result.modules():
            cls, origin = type(child).__name__, type(child).__module__
            if origin == "tace.models.linear" and cls in (
                "e3nnElementLinear",
                "e3nnMoEElementLinear",
            ):
                child.forward = MethodType(getattr(linear, cls).forward, child)
            if cls == "Convolution" and origin.endswith("tece_oam_rra"):
                # Helpers such as rotary_attention also changed their opt-in
                # branches. Bind every historical method, rather than letting
                # an old forward call a new FastEq helper on this instance.
                for name, method in vars(tece.Convolution).items():
                    if name != "__init__" and inspect.isfunction(method):
                        setattr(child, name, MethodType(method, child))
        return result

    return adapt, {
        "archive": str(archive),
        "manifest": manifest,
        "loaded": loaded + [linear.__file__, tece.__file__],
        "baseline_runtime": "verified unchanged EQX; historical TACE methods bound per instance",
        "source_changes": "scalar dispatcher namespace and import location only",
    }
