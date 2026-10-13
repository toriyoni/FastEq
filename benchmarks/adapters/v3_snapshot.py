"""Load verified FastEq v3 beside optimized operators in one CUDA process.

Only import, dispatcher and native C++ type namespaces change. The legacy
Uniform1D implementation is shared after source verification. Native v3
compiler/Linear plans are built independently to preserve task dispatch.
"""

import hashlib
import inspect
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from types import CellType, FunctionType, MethodType, ModuleType


def _rebase_class_cell(function, owner):
    """Keep archived globals while making zero-argument super() valid."""
    if "__class__" not in function.__code__.co_freevars:
        return function
    closure = tuple(
        CellType(owner) if name == "__class__" else cell
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__, strict=True
        )
    )
    result = FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        closure,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__dict__.update(function.__dict__)
    result.__module__, result.__qualname__ = function.__module__, function.__qualname__
    result.__doc__ = function.__doc__
    return result


def _bind_methods(instance, archived_class):
    """Preserve instance/static/class binding and calls between old methods."""
    owner = type(instance)
    for name, descriptor in vars(archived_class).items():
        if name == "__init__":
            continue
        if isinstance(descriptor, staticmethod):
            method = _rebase_class_cell(descriptor.__func__, owner)
        elif isinstance(descriptor, classmethod):
            method = MethodType(_rebase_class_cell(descriptor.__func__, owner), owner)
        elif inspect.isfunction(descriptor):
            method = MethodType(_rebase_class_cell(descriptor, owner), instance)
        else:
            continue
        setattr(instance, name, method)


@lru_cache(maxsize=1)
def load_v3(root):
    root = Path(root)
    archive = root / "performance/fasteq_optimization/v3_snapshot"
    manifest = json.loads((archive / "manifest.json").read_text())
    shared_dependencies = []
    for row in manifest["files"]:
        assert (
            hashlib.sha256((archive / row["path"]).read_bytes()).hexdigest()
            == row["sha256"]
        ), row["path"]

    def share(relative, purpose):
        original = (archive / relative).read_bytes()
        current = (root / relative).read_bytes()
        assert current == original, relative
        shared_dependencies.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(current).hexdigest(),
                "purpose": purpose,
            }
        )

    def package(name, relative):
        value = ModuleType(name)
        value.__path__ = [str(archive / relative)]
        value.__package__ = name
        sys.modules[name] = value
        parent = sys.modules.get(name.rpartition(".")[0])
        if parent is not None:
            setattr(parent, name.rpartition(".")[2], value)
        return value

    package("fasteq_v3", "src/FastEq/fasteq")
    for name in ("cuda", "triton", "ops", "adapters"):
        package("fasteq_v3." + name, "src/FastEq/fasteq/" + name)

    compatibility = root / "cache/v3_cuda_sources"
    compatibility.mkdir(parents=True, exist_ok=True)
    for name in ("jit_runtime.h", "jit_runtime.cpp", "indexed_linear_plan.cpp"):
        source = (archive / "src/FastEq/fasteq/cuda/src" / name).read_text()
        source = source.replace("fasteq_jit", "fasteq_v3_jit")
        source = re.sub(r"\bLinearPlan\b", "V3LinearPlan", source)
        target = compatibility / name
        if target.exists():
            assert target.read_text() == source, target
        else:
            target.write_text(source)

    loaded = []

    def module(name, relative, replacements=(), package_override=None):
        path = archive / relative
        value = ModuleType(name)
        value.__file__ = str(path)
        value.__package__ = package_override or name.rpartition(".")[0]
        source = path.read_text().replace("from fasteq.", "from fasteq_v3.")
        source = source.replace('"fasteq::', '"fasteq_v3::')
        for old, new in replacements:
            assert old in source, (relative, old)
            source = source.replace(old, new)
        sys.modules[name] = value
        parent = sys.modules.get(name.rpartition(".")[0])
        if parent is not None:
            setattr(parent, name.rpartition(".")[2], value)
        exec(compile(source, str(path), "exec"), value.__dict__)  # noqa: S102
        loaded.append(str(path))
        return value

    module(
        "fasteq_v3.cuda.jit",
        "src/FastEq/fasteq/cuda/jit.py",
        [
            (
                'directory = Path(__file__).parent / "src"',
                "directory = Path(" + repr(str(compatibility)) + ")",
            ),
            ('name="fasteq_cuda_jit"', 'name="fasteq_v3_cuda_jit"'),
        ],
    )
    for name in (
        "fused_equivariant_gate",
        "fused_equivariant_layer_norm",
        "spherical_harmonics",
        "irreps_invariants",
    ):
        module("fasteq_v3.triton." + name, "src/FastEq/fasteq/triton/" + name + ".py")
    for name in (
        "indexed_linear",
        "gate",
        "equivariant_norm",
        "spherical_harmonics",
        "scalar_expression",
        "rotation",
        "local_product",
        "segmented_product",
    ):
        module("fasteq_v3.ops." + name, "src/FastEq/fasteq/ops/" + name + ".py")

    # Uniform1D uses the original extension/template generator rather than the
    # new native runtime. Every snapshotted dependency is unchanged this round.
    for row in manifest["files"]:
        name = row["path"]
        if name.startswith("src/FastEq/fasteq/cuda/") and name.rsplit("/", 1)[
            -1
        ] not in (
            "jit.py",
            "jit_runtime.cpp",
            "jit_runtime.h",
            "indexed_linear_plan.cpp",
            "indexed_linear.cu",
        ):
            share(name, "unchanged Uniform1D CUDA source dependency")
        if name in (
            "src/FastEq/fasteq/ops/uniform1d_jit.py",
            "src/FastEq/fasteq/ops/uniform1d_auto_schedule.py",
        ):
            share(name, "unchanged Uniform1D Python implementation")
        if name.startswith(
            ("src/tace/tace/models/", "src/prophet/prophet/", "src/tace/eqx/")
        ):
            share(
                name,
                "unchanged model construction/interface/reference; equivariant plans are frozen",
            )
    import fasteq.ops.uniform1d_jit as uniform

    sys.modules["fasteq_v3.ops.uniform1d_jit"] = uniform
    for name in ("cache", "harmonics", "programs", "tensor_product", "fusion"):
        module(
            "fasteq_v3.adapters." + name, "src/FastEq/fasteq/adapters/" + name + ".py"
        )
    adapter = module(
        "fasteq_v3.adapters",
        "src/FastEq/fasteq/adapters/__init__.py",
        package_override="fasteq_v3.adapters",
    )
    adapter.__path__ = [str(archive / "src/FastEq/fasteq/adapters")]
    adapter.__package__ = "fasteq_v3.adapters"
    tece = module(
        "tace.models._e3nn._fasteq_v3_tece",
        "src/tace/tace/models/_e3nn/tece_oam_rra.py",
        package_override="tace.models._e3nn",
    )

    def adapt(model, *, inplace=False):
        result = adapter.adapt_model(model, inplace=inplace)
        for child in result.modules():
            if type(child).__name__ == "Convolution" and type(
                child
            ).__module__.endswith("tece_oam_rra"):
                _bind_methods(child, tece.Convolution)
        return result

    return adapt, {
        "archive": str(archive),
        "manifest": manifest,
        "loaded": loaded,
        "namespace_changes_only": True,
        "native_runtime": "independently built v3 compiler and Linear plans",
        "shared_implementation": "source-verified unchanged Uniform1D",
        "compatibility_sources": str(compatibility),
        "shared_dependencies": shared_dependencies,
        "tece_binding": "all snapshot instance/static/class methods; frozen imports; zero-argument super class cell rebased",
    }
