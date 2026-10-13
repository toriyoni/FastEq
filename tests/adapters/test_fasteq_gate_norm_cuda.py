"""Fused FastEq residual/Gate/RMS, including the indirect force-loss route."""

from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from eqx_cuda_spy import forbid_eqx_cuda
from fasteq.adapters.fusion import ProphetPostprocess
from prophet.layer_norm import RMSLayerNorm
from prophet.model import Gate

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def modules(dtype, channels, affine=True, centering=False, pre=False):
    out = o3.Irreps(f"{channels}x0e+{channels}x1o+{channels}x2e")
    ir = o3.Irreps(f"{3*channels}x0e+{channels}x1o+{channels}x2e")
    gate = Gate(ir).cuda().to(dtype)
    norm = RMSLayerNorm(out, affine=affine, centering=centering).cuda().to(dtype)
    before = RMSLayerNorm(ir, centering=False).cuda().to(dtype) if pre else None
    with torch.no_grad():
        for p in norm.parameters():
            p.uniform_(0.5, 1.5)
    conv = SimpleNamespace(gate=gate, layer_norm=before,
                           linear_2=SimpleNamespace(irreps_out=ir))
    return gate, norm, before, ProphetPostprocess(conv, norm)


def close(left, right, dtype, factor=1):
    atol, rtol = (4e-5, 4e-4) if dtype == torch.float32 else (2e-10, 2e-9)
    torch.testing.assert_close(left, right, atol=atol*factor, rtol=rtol*factor)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [3, 64])
@pytest.mark.parametrize("affine", [False, True])
def test_gate_norm_all_first_and_second(dtype, channels, affine):
    torch.manual_seed(733)
    gate, norm, _, plan = modules(dtype, channels, affine)
    assert plan.fused is not None
    # Sliced leading axes exercise contiguity and view connectivity.
    p = torch.randn(2, 3, 2*plan.dim, device="cuda", dtype=dtype)[..., ::2].requires_grad_()
    s = torch.randn_like(p, requires_grad=True)
    pr, sr = p.detach().clone().requires_grad_(), s.detach().clone().requires_grad_()
    parameters = tuple(norm.parameters())
    reference = norm(gate((pr+sr).reshape(-1, plan.dim))).reshape(2, 3, plan.outdim)
    seedr = torch.randn_like(reference, requires_grad=True)
    firstr = torch.autograd.grad(reference, (pr, sr, *parameters), seedr, create_graph=True)
    directions = tuple(torch.randn_like(g) for g in firstr)
    secondr = torch.autograd.grad(sum((g*d).sum() for g,d in zip(firstr,directions)),
                                 (pr, sr, seedr, *parameters), allow_unused=True)
    seed = seedr.detach().clone().requires_grad_()
    actual = plan(p, s)
    first = torch.autograd.grad(actual, (p, s, *parameters), seed, create_graph=True)
    second = torch.autograd.grad(sum((g*d).sum() for g,d in zip(first,directions)),
                                (p, s, seed, *parameters), allow_unused=True)
    assert plan.fused._launchers
    with forbid_eqx_cuda():
        close(plan(p,s),reference,dtype)
    close(actual, reference, dtype)
    for a,b in zip(first,firstr):
        close(a,b,dtype,5)
    for a,b in zip(second,secondr):
        if b is None:
            assert a is None or not torch.count_nonzero(a)
        else:
            close(a,b,dtype,15)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_force_loss_saved_input_and_upstream_seed(dtype):
    torch.manual_seed(739)
    gate, norm, _, plan = modules(dtype, 4)
    x = torch.randn(5, plan.dim, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn_like(x, requires_grad=True)
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    parameters = tuple(norm.parameters())

    def execute(x, w, fn):
        # Both saved inputs depend on trainable weights and coordinates. The
        # nonlinear energy makes grad_output depend on the custom output too.
        y = fn(x*w, x.sin()*w.square()/9)
        energy = y.sin().sum() + y.square().sum()/7
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = force.square().mean() + energy.square()/100
        gradients = torch.autograd.grad(loss, (x,w,*parameters))
        return y, energy, force, loss, gradients

    reference = execute(xr,wr,lambda p,s:norm(gate(p+s)))
    actual = execute(x,w,plan)
    with forbid_eqx_cuda():
        # The traced/Triton fallback has the same full derivative semantics.
        guarded = execute(x,w,plan)
    for a,b in zip(actual[:4],reference[:4]):
        close(a,b,dtype,5)
    for a,b in zip(actual[4],reference[4]):
        close(a,b,dtype,30)
    for a,b in zip(guarded[4],reference[4]):
        close(a,b,dtype,30)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_inference_first_and_parameter_replacement(dtype):
    gate, norm, _, plan = modules(dtype, 2)
    with torch.inference_mode():
        plan(torch.randn(2,plan.dim,device="cuda",dtype=dtype),
             torch.randn(2,plan.dim,device="cuda",dtype=dtype))
    old = tuple(plan.parameters)
    norm.load_state_dict({k:torch.full_like(v,1.7) for k,v in norm.state_dict().items()},assign=True)
    assert all(a is not b for a,b in zip(old,plan.parameters))
    x = torch.randn(3,plan.dim,device="cuda",dtype=dtype,requires_grad=True)
    close(plan(x,x),norm(gate(x+x)),dtype)
    torch.autograd.grad(plan(x,x).square().sum(),(x,*norm.parameters()))
    x = torch.empty(0,plan.dim,device="cuda",dtype=dtype,requires_grad=True)
    y = plan(x,x)
    assert y.shape == (0,plan.outdim)
    first = torch.autograd.grad(y.sum(),(x,*norm.parameters()),create_graph=True)
    torch.autograd.grad(sum(g.sum() for g in first),(x,*norm.parameters()),allow_unused=True)


@pytest.mark.parametrize("centering,pre", [(True,False),(False,True),(True,True)])
def test_fallback_keeps_existing_composition(centering,pre):
    gate,norm,before,plan=modules(torch.float64,2,centering=centering,pre=pre)
    assert plan.fused is None
    x=torch.randn(3,plan.dim,device="cuda",dtype=torch.float64,requires_grad=True)
    value=x+x
    if before is not None:
        value=before(value)
    reference=norm(gate(value))
    with forbid_eqx_cuda():
        actual=plan(x,x)
    close(actual,reference,torch.float64)


def test_native_launcher_alignment_stream_and_graph():
    gate,norm,_,plan=modules(torch.float64,3)
    storage=torch.randn(1+3*plan.dim,device="cuda",dtype=torch.float64)
    x=storage[1:].view(3,plan.dim).requires_grad_()
    assert x.data_ptr()%16
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual=plan(x,x)
        first=torch.autograd.grad(actual.sin().sum(),(x,*norm.parameters()),create_graph=True)
        second=torch.autograd.grad(sum(g.square().sum() for g in first),(x,*norm.parameters()))
    stream.synchronize()
    xr=x.detach().clone().requires_grad_()
    reference=norm(gate(xr+xr))
    firstr=torch.autograd.grad(reference.sin().sum(),(xr,*norm.parameters()),create_graph=True)
    secondr=torch.autograd.grad(sum(g.square().sum() for g in firstr),(xr,*norm.parameters()))
    close(actual,reference,torch.float64)
    for a,b in zip(second,secondr):
        close(a,b,torch.float64,15)
    static=x.detach().clone()
    for _ in range(3):
        plan(static,static)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay=plan(static,static)
    graph.replay()
    torch.cuda.synchronize()
    close(replay,norm(gate(static+static)),torch.float64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("target", ["input", "one_gamma"])
def test_selective_first_keeps_second_parameter_routes(dtype,target):
    torch.manual_seed(757)
    gate,norm,_,plan=modules(dtype,3)
    x=torch.randn(5,plan.dim,device="cuda",dtype=dtype,requires_grad=True)
    skip=torch.randn_like(x,requires_grad=True)
    xr,sr=x.detach().clone().requires_grad_(),skip.detach().clone().requires_grad_()
    gamma=tuple(norm.parameters())
    launches=[]
    original_launch=plan.fused.launch

    def spy(fn,args,value,**kwargs):
        launches.append((fn.__name__,kwargs.copy()))
        return original_launch(fn,args,value,**kwargs)

    plan.fused.launch=spy
    y=plan(x,skip)
    reference=norm(gate(xr+sr))
    wanted=x if target=="input" else gamma[0]
    wanted_ref=xr if target=="input" else gamma[0]
    first=torch.autograd.grad(y.sin().sum(),wanted,create_graph=True)[0]
    firstr=torch.autograd.grad(reference.sin().sum(),wanted_ref,create_graph=True)[0]
    backward=[row for name,row in launches if name=="backward"][-1]
    if target=="input":
        assert backward["WIDTH"]==0
        assert not any(backward["GGRADFLAGS"])
    else:
        assert backward["WIDTH"]==gamma[0].numel()
        assert backward["GGRADFLAGS"]==(True,False,False)
        assert not backward["DZFLAG"]
    second=torch.autograd.grad(first.square().sum(),(x,skip,*gamma),allow_unused=True)
    secondr=torch.autograd.grad(firstr.square().sum(),(xr,sr,*gamma),allow_unused=True)
    close(first,firstr,dtype,5)
    for a,b in zip(second,secondr):
        if b is None:
            assert a is None or not torch.count_nonzero(a)
        else:
            close(a,b,dtype,30)
