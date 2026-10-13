"""Load frozen FastEq v4 with independent Python, dispatcher and C++ runtime.

The native indexed Linear plan, Norm mask helper, and resident Triton launcher
come from the verified snapshot. Uniform1D alone is shared after every archived
dependency is compared byte for byte. Model construction interfaces must remain
unchanged and share class descriptors. Only the two TECE methods importing
FastEq bind frozen globals, in addition to the frozen adapter methods.
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
    if "__class__" not in function.__code__.co_freevars:
        return function
    closure = tuple(
        CellType(owner) if name == "__class__" else cell
        for name, cell in zip(function.__code__.co_freevars, function.__closure__, strict=True)
    )
    result = FunctionType(function.__code__, function.__globals__, function.__name__,
                          function.__defaults__, closure)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__dict__.update(function.__dict__)
    result.__module__, result.__qualname__, result.__doc__ = (
        function.__module__, function.__qualname__, function.__doc__)
    return result


def _bind_frozen_methods(instance, classes):
    """Rebase super to the declaring live MRO class, keeping v4 adapter methods."""
    count = 0
    for owner in reversed(type(instance).__mro__):
        if (owner.__module__, owner.__name__) != (
            "tace.models._e3nn.tece_oam_rra", "Convolution"
        ):
            continue
        archived = classes.get((owner.__module__, owner.__name__))
        if archived is None:
            continue
        for name, descriptor in vars(archived).items():
            if name not in ("forward", "rotary_score"):
                continue
            existing = instance.__dict__.get(name)
            if isinstance(existing, MethodType) and existing.__func__.__module__.startswith("fasteq_v4."):
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
            count += 1
    return count


@lru_cache(maxsize=1)
def load_v4(root):
    root = Path(root).resolve()
    archive = root / "performance/fasteq_optimization/v4_snapshot"
    manifest = json.loads((archive / "manifest.json").read_text())
    hashes = manifest["files_sha256"]
    for relative, expected in hashes.items():
        assert hashlib.sha256((archive / relative).read_bytes()).hexdigest() == expected, relative
    for name, expected in manifest["benchmark_driver_sha256"].items():
        assert hashlib.sha256((archive / "benchmark_driver" / name).read_bytes()).hexdigest() == expected, name
    shared = []

    def share(relative, purpose):
        original = (archive / relative).read_bytes()
        assert (root / relative).read_bytes() == original, relative
        shared.append({"path": relative, "sha256": hashes[relative], "purpose": purpose})

    def package(name, relative):
        value = ModuleType(name)
        value.__path__, value.__package__ = [str(archive / relative)], name
        sys.modules[name] = value
        parent = sys.modules.get(name.rpartition(".")[0])
        if parent is not None:
            setattr(parent, name.rpartition(".")[2], value)
        return value

    package("fasteq_v4", "src/FastEq/fasteq")
    for name in ("cuda", "triton", "ops", "adapters"):
        package("fasteq_v4." + name, "src/FastEq/fasteq/" + name)
    compatibility = root / "cache/v4_cuda_sources"
    compatibility.mkdir(parents=True, exist_ok=True)
    compatibility_hashes = {}
    for name in ("jit_runtime.h", "jit_runtime.cpp", "indexed_linear_plan.cpp", "norm_plan.cpp"):
        source = (archive / "src/FastEq/fasteq/cuda/src" / name).read_text()
        source = source.replace("fasteq_jit", "fasteq_v4_jit")
        for original, replacement in (("LinearPlan", "V4LinearPlan"),
                                      ("LinearDerivative", "V4LinearDerivative"),
                                      ("bind_indexed_linear", "bind_v4_indexed_linear"),
                                      ("bind_norm", "bind_v4_norm")):
            source = re.sub(r"\b" + original + r"\b", replacement, source)
        target = compatibility / name
        if target.exists():
            assert target.read_text() == source, target
        else:
            target.write_text(source)
        compatibility_hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    loaded = []

    def module(name, relative, replacements=(), package_override=None):
        path = archive / relative
        assert relative in hashes, relative
        value = ModuleType(name)
        value.__file__, value.__package__ = str(path), package_override or name.rpartition(".")[0]
        source = path.read_text().replace("from fasteq.", "from fasteq_v4.")
        source = source.replace('"fasteq::', '"fasteq_v4::')
        for original, replacement in replacements:
            assert original in source, (relative, original)
            source = source.replace(original, replacement)
        sys.modules[name] = value
        parent = sys.modules.get(name.rpartition(".")[0])
        if parent is not None:
            setattr(parent, name.rpartition(".")[2], value)
        exec(compile(source, str(path), "exec"), value.__dict__)  # noqa: S102
        loaded.append({"namespace": name, "path": relative, "sha256": hashes[relative]})
        return value

    module("fasteq_v4.cuda.jit", "src/FastEq/fasteq/cuda/jit.py", [
        ('directory = Path(__file__).parent / "src"', "directory = Path(" + repr(str(compatibility)) + ")"),
        ('name="fasteq_cuda_jit"', 'name="fasteq_v4_cuda_jit"'),
    ])
    module("fasteq_v4.cuda.triton_launch", "src/FastEq/fasteq/cuda/triton_launch.py")
    for name in ("fused_equivariant_gate", "fused_equivariant_layer_norm", "spherical_harmonics",
                 "irreps_invariants", "gate_norm"):
        module("fasteq_v4.triton." + name, "src/FastEq/fasteq/triton/" + name + ".py")
    for name in ("indexed_linear", "gate", "equivariant_norm", "gate_norm", "modulation",
                 "spherical_harmonics", "scalar_expression", "rotation", "local_product", "segmented_product"):
        module("fasteq_v4.ops." + name, "src/FastEq/fasteq/ops/" + name + ".py")
    isolated_cpp = {"jit.py", "jit_runtime.cpp", "jit_runtime.h", "indexed_linear_plan.cpp",
                    "indexed_linear.cu", "norm_plan.cpp", "triton_launch.py"}
    for relative in hashes:
        if relative.startswith("src/FastEq/fasteq/cuda/") and relative.rsplit("/", 1)[-1] not in isolated_cpp:
            share(relative, "unchanged Uniform1D CUDA source dependency")
        if relative in ("src/FastEq/fasteq/ops/uniform1d_jit.py",
                        "src/FastEq/fasteq/ops/uniform1d_auto_schedule.py"):
            share(relative, "unchanged Uniform1D Python implementation")
        if relative.startswith(("src/tace/tace/models/", "src/prophet/prophet/", "src/tace/eqx/")):
            share(relative, "unchanged model/interface class descriptors; only TECE FastEq import methods frozen")
    import fasteq.ops.uniform1d_jit as uniform

    sys.modules["fasteq_v4.ops.uniform1d_jit"] = uniform
    for name in ("cache", "harmonics", "programs", "tensor_product", "fusion"):
        module("fasteq_v4.adapters." + name, "src/FastEq/fasteq/adapters/" + name + ".py")
    adapter = module("fasteq_v4.adapters", "src/FastEq/fasteq/adapters/__init__.py",
                     package_override="fasteq_v4.adapters")
    adapter.__path__, adapter.__package__ = [str(archive / "src/FastEq/fasteq/adapters")], "fasteq_v4.adapters"

    classes = {}
    for original in ("tace.models.linear", "tace.models.adapter", "tace.models._e3nn.base",
                     "tace.models._e3nn.prod", "tace.models._e3nn.inter",
                     "tace.models._e3nn.tece_oam_rra", "tace.models._e3nn.representation",
                     "tace.models._e3nn.tace", "prophet.layer_norm", "prophet.model"):
        relative = ("src/prophet/" if original.startswith("prophet.") else "src/tace/")
        relative += original.replace(".", "/") + ".py"
        package_name, _, short_name = original.rpartition(".")
        alias = package_name + "._fasteq_v4_" + short_name
        frozen = module(alias, relative, package_override=package_name)
        for name, cls in vars(frozen).items():
            if isinstance(cls, type) and cls.__module__ == alias:
                classes[(original, name)] = cls

    def adapt(model, *, inplace=False):
        result = adapter.adapt_model(model, inplace=inplace)
        rebound = sum(_bind_frozen_methods(child, classes) for child in result.modules())
        assert result._fasteq_backend_version == "fasteq-optimized-v4"
        result._fasteq_v4_frozen_method_count = rebound
        return result

    return adapt, {
        "archive": str(archive), "manifest": manifest,
        "manifest_sha256": hashlib.sha256((archive / "manifest.json").read_bytes()).hexdigest(),
        "namespace_changes_only": True, "loaded": loaded,
        "native_runtime": "independent frozen v4 C++ Kernel/LinearPlan/Norm helper and resident launcher",
        "extension_name": "fasteq_v4_cuda_jit",
        "compatibility_sources": str(compatibility), "compatibility_sha256": compatibility_hashes,
        "shared_implementation": "source-verified unchanged Uniform1D only",
        "shared_dependencies": shared,
        "model_binding": "only TECE Convolution.forward/rotary_score bind snapshot FastEq imports; snapshot adapter forwards retained; pure model helpers share source-verified unchanged class descriptors; super class cell rebased to declaring live MRO class",
        "bound_model_classes": ["tace.models._e3nn.tece_oam_rra.Convolution"],
        "bound_model_methods": ["forward", "rotary_score"],
        "loaded_model_classes": sorted(module + "." + name for module, name in classes),
    }
