"""Forbid EQX CUDA execution while retaining its CPU/interface references."""

import sys
from contextlib import contextmanager

import pytest
from torch.utils._python_dispatch import TorchDispatchMode


class _NoEQXOperators(TorchDispatchMode):
    def __init__(self, calls):
        self.calls = calls

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name.split("::", 1)[0].startswith("eqx"):
            self.calls.append(func._schema.name)
            raise AssertionError("EQX CUDA operator invoked: " + func._schema.name)
        return func(*args, **(kwargs or {}))


@contextmanager
def forbid_eqx_cuda():
    """Reject compiler/launcher aliases, resident plans and EQX dispatcher ops.

    Run the reference model before entering this context. CUDA kernels imported
    from FastEq and ordinary ATen CUDA operations remain available throughout.
    """
    from eqx.kernels import cuda

    calls = []
    patch = pytest.MonkeyPatch()
    originals = {
        name: getattr(cuda, name) for name in ("runtime", "kernels", "launch_config")
    }
    # Existing modules may have bound `from .cuda import kernels` aliases.
    for name, module in list(sys.modules.items()):
        if not name.startswith("eqx.") or module is None:
            continue
        for attribute, value in list(vars(module).items()):
            match = next((key for key, fn in originals.items() if value is fn), None)
            if match is not None:

                def forbidden(*args, _name=name + "." + attribute, **kwargs):
                    calls.append(_name)
                    raise AssertionError("EQX compiler/launcher invoked: " + _name)

                patch.setattr(module, attribute, forbidden)
    # Plans already compiled during the reference run may bypass runtime().
    if originals["runtime"].cache_info().currsize:
        module = originals["runtime"]()
        for attribute, value in list(vars(module).items()):
            if attribute == "launch":

                def forbidden_launch(*args, **kwargs):
                    calls.append("eqx.native.launch")
                    raise AssertionError("EQX native launch invoked")

                patch.setattr(module, attribute, forbidden_launch)
            elif isinstance(value, type) and (
                "Plan" in attribute or attribute == "Kernel"
            ):
                for method in ("launch", "execute", "contract", "run"):
                    if hasattr(value, method):

                        def forbidden_plan(
                            *args, _name=attribute + "." + method, **kwargs
                        ):
                            calls.append("eqx.native." + _name)
                            raise AssertionError("EQX resident plan invoked: " + _name)

                        patch.setattr(value, method, forbidden_plan)
    try:
        with _NoEQXOperators(calls):
            yield calls
    finally:
        patch.undo()
