"""Keep real eager paths intact while forbidding external/Triton launchers."""
from contextlib import contextmanager, nullcontext
import sys


@contextmanager
def native_only():
    """The generic EQX spy uses DispatchMode, which selects operator fallbacks.

    Retain its compiler/launcher/resident-plan guards, and replace its mode
    with explicit EQX operator namespace/alias guards. Profile additionally
    rejects EQX CPU dispatch events. C++-initiated dispatcher calls cannot be
    intercepted by these Python aliases and require the independent profile.
    No spies are used in normal timing.
    """
    import pytest
    import torch
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tests/adapters'))
    import eqx_cuda_spy
    from check_native_gate_invariants_v8 import native_only as guarded
    calls=[]
    patch=pytest.MonkeyPatch()
    def forbidden(*_args,**_kwargs):
        calls.append('eqx.dispatcher');raise AssertionError('External EQX operator invoked')
    try:
        old_overload=torch._ops.OpOverload.__call__
        old_packet=torch._ops.OpOverloadPacket.__call__
        def overload_call(operator,*args,**kwargs):
            if operator._schema.name.split('::',1)[0].startswith('eqx'):return forbidden(*args,**kwargs)
            return old_overload(operator,*args,**kwargs)
        def packet_call(operator,*args,**kwargs):
            if operator._qualified_op_name.split('::',1)[0].startswith('eqx'):return forbidden(*args,**kwargs)
            return old_packet(operator,*args,**kwargs)
        patch.setattr(torch._ops.OpOverload,'__call__',overload_call)
        patch.setattr(torch._ops.OpOverloadPacket,'__call__',packet_call)
        patch.setattr(eqx_cuda_spy,'_NoEQXOperators',lambda _calls:nullcontext())
        names=[name for name in torch._C._dispatch_get_all_op_names()if name.split('::',1)[0].startswith('eqx')]
        for namespace,base in sorted({(name.split('::',1)[0],name.split('::',1)[1].split('.',1)[0])for name in names}):
            packet=getattr(getattr(torch.ops,namespace),base)
            aliases=[packet]+[getattr(packet,overload)for overload in packet.overloads()]
            for module_name,module in list(sys.modules.items()):
                if not module_name.startswith('eqx.')or module is None:continue
                for attribute,value in list(vars(module).items()):
                    if any(value is alias for alias in aliases):patch.setattr(module,attribute,forbidden)
            patch.setattr(getattr(torch.ops,namespace),base,forbidden)
        with guarded():yield calls
        assert not calls
    finally:
        patch.undo()
