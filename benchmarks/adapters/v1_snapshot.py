"""Load the immutable first implementation beside the optimized adapters.

Only dispatcher names and import locations are renamed. CUDA sources,
preparation and backward formulas come from the verified v1 source archive.
The model-independent compiler/launcher ABI is shared with the current runtime.
"""

import hashlib
import json
import sys
from pathlib import Path
from types import MethodType, ModuleType


def load_v1(root):
    archive = Path(root) / "performance/optimization/v1_snapshot"
    manifest = json.loads((archive / "manifest.json").read_text())
    for row in manifest["files"]:
        source = (archive / row["path"]).read_bytes()
        assert hashlib.sha256(source).hexdigest() == row["sha256"], row["path"]

    def module(name, relative, replacements=(), package=None):
        path = archive / relative
        value = ModuleType(name)
        value.__file__ = str(path)
        value.__package__ = name.rpartition(".")[0] if package is None else package
        source = path.read_text()
        for old, new in replacements:
            assert old in source, (relative, old)
            source = source.replace(old, new)
        sys.modules[name] = value
        exec(compile(source, str(path), "exec"), value.__dict__)  # noqa: S102 -- verified local source snapshot
        return value

    codegen = module("eqx.conv._fasteq_v1_codegen", "src/tace/eqx/conv/codegen.py")
    edge = module(
        "eqx.conv._fasteq_v1_edge",
        "src/tace/eqx/conv/edge.py",
        [
            ('"eqx::edge_program"', '"eqx_v1::edge_program"'),
            ("from .codegen import launch", "from ._fasteq_v1_codegen import launch"),
        ],
    )
    cuda = module("eqx.o3._fasteq_v1_cuda", "src/tace/eqx/o3/cuda.py")
    contraction = module(
        "eqx.o3._fasteq_v1_contraction",
        "src/tace/eqx/o3/contraction.py",
        [
            ('"eqx::element_linear"', '"eqx_v1::element_linear"'),
            ("from .cuda import launch", "from ._fasteq_v1_cuda import launch"),
        ],
    )
    linear = module(
        "eqx.o3._fasteq_v1_linear",
        "src/tace/eqx/o3/linear.py",
        [
            (
                "from .contraction import contract",
                "from ._fasteq_v1_contraction import contract",
            ),
        ],
    )
    package = ModuleType("fasteq_v1")
    package.__path__ = [str(archive / "src/FastEq/fasteq/adapters")]
    sys.modules[package.__name__] = package
    module(
        "fasteq_v1.harmonics",
        "src/FastEq/fasteq/adapters/harmonics.py",
        [("fasteq::scalar_expression", "fasteq_v1::scalar_expression")],
    )
    programs = module(
        "fasteq_v1.programs",
        "src/FastEq/fasteq/adapters/programs.py",
        [
            (
                "from eqx.conv.edge import evaluate",
                "from eqx.conv._fasteq_v1_edge import evaluate",
            ),
            (
                "from eqx.o3.linear import ElementLinear",
                "from eqx.o3._fasteq_v1_linear import ElementLinear",
            ),
        ],
    )
    tensor_product = module(
        "fasteq_v1.tensor_product", "src/FastEq/fasteq/adapters/tensor_product.py"
    )
    adapter = module(
        "fasteq_v1", "src/FastEq/fasteq/adapters/__init__.py", package="fasteq_v1"
    )
    adapter.__path__ = package.__path__

    def adapt(model):
        result = adapter.adapt_model(model)
        # TACE constructs its existing indexed maps before adaptation. Bind
        # those instances to the v1 dispatcher as well, without global patches.
        from eqx.o3.linear import ElementLinear

        for child in result.modules():
            if isinstance(child, ElementLinear):
                child.forward = MethodType(linear.ElementLinear.forward, child)
        return result

    return adapt, {
        "archive": str(archive),
        "manifest": manifest,
        "namespace_changes_only": True,
        "shared_runtime": "model-independent NVRTC/compiler/launch ABI",
        "loaded": [
            codegen.__file__,
            edge.__file__,
            cuda.__file__,
            contraction.__file__,
            linear.__file__,
            programs.__file__,
            tensor_product.__file__,
        ],
    }
