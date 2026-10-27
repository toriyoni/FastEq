"""Extension fixtures: zero-copy metadata, mixed Gate roles and physical leaves."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from e3nn import o3
from prophet.layer_norm import RMSLayerNorm
from fasteq.ops.modulation import ModulatedResidualPlan
from fasteq.ops.gate_norm import GateNormPlan
from run_native_norm_v8 import reference as norm_reference


def compare(actual,expected,dtype):
    for a,b in zip(actual,expected,strict=True):
        if a is None or b is None:
            value=b if a is None else a
            assert value is None or not torch.count_nonzero(value)
        else:torch.testing.assert_close(a,b,atol=8e-5 if dtype==torch.float32 else 2e-9,rtol=2e-4 if dtype==torch.float32 else 2e-8)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('enabled',[False,True])
def test_modulation_ragged_tail(dtype,enabled,monkeypatch):
    monkeypatch.setenv('FASTEQ_EXPERIMENTAL_NATIVE_MODULATION','1' if enabled else '0')
    plan=ModulatedResidualPlan(o3.Irreps('0x0e+2x0e+0x1o+7x6e+2x2e'));index=torch.tensor(plan._index,device='cuda')
    values=[(torch.randn(17,width*2,device='cuda',dtype=dtype)*.2)[:,::2].detach().requires_grad_() for width in (plan.dim,plan.channels,plan.dim)]
    refs=[v.detach().clone().requires_grad_() for v in values];seed=torch.randn_like(values[0]).requires_grad_();seedr=seed.detach().clone().requires_grad_()
    def chain(fn,args,dy):
        out=fn(*args);first=torch.autograd.grad(out,args,dy,create_graph=True)
        second=torch.autograd.grad(sum(g.square().sum() for g in first),(*args,dy),allow_unused=True)
        return(out,*first,*second)
    compare(chain(plan,values,seed),chain(lambda x,g,r:r+x*g.index_select(-1,index),refs,seedr),dtype)


def fused_fixture(dtype):
    # Mathematical Gate interface with all activation opcodes, signed scales,
    # shared gates, self gates and zero-copy Norm parameter slots.
    output=16;dim=10;src=torch.tensor([i%dim for i in range(output)],dtype=torch.int32)
    gates=torch.tensor([-1 if i%4==0 else (i*3)%dim for i in range(output)],dtype=torch.int32)
    act=torch.tensor([i%4 for i in range(output)],dtype=torch.int8)
    scale=torch.tensor([(-1 if i%3==0 else 1)*(.7+.02*i) for i in range(output)],dtype=torch.float64)
    fans=[[] for _ in range(dim)]
    for i,(s,g) in enumerate(zip(src.tolist(),gates.tolist())):
        fans[s].append((i,0 if g<0 else 1))
        if g>=0:fans[g].append((i,2))
    fan=max(map(len,fans));outs=torch.full((dim,fan),-1,dtype=torch.int32);roles=torch.full_like(outs,-1,dtype=torch.int8)
    for i,entries in enumerate(fans):
        for j,(out,role) in enumerate(entries):outs[i,j]=out;roles[i,j]=role
    gate=SimpleNamespace(dim=dim,outdim=output,metadata=(src,gates,act,scale,outs,roles))
    norm=RMSLayerNorm(o3.Irreps('0x0e+2x0e+0x1o+3x1o+1x2e'),centering=False,eps=1e-7).to(device='cuda',dtype=dtype)
    plan=GateNormPlan(gate,norm)
    return norm,plan


def gate_reference(plan,x):
    src,gates,act,scale,*_=plan._metadata;src=src.to(device=x.device,dtype=torch.long);gates=gates.to(device=x.device,dtype=torch.long)
    xs=x.index_select(-1,src);z=torch.where(gates>=0,x.index_select(-1,gates.clamp_min(0)),xs)
    code=act.to(x.device);scale=scale.to(device=x.device,dtype=x.dtype)
    activated=torch.where(code==0,z,torch.where(code==1,torch.nn.functional.silu(z),torch.where(code==2,z.sigmoid(),z.tanh())))*scale
    return torch.where(gates>=0,xs*activated,activated)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('rows',[0,17,257])
def test_gate_norm_ragged_mixed_roles(dtype,rows):
    norm,plan=fused_fixture(dtype)
    p=(torch.randn(rows,plan.dim*2,device='cuda',dtype=dtype)*.2)[:,::2].detach().requires_grad_();s=torch.randn_like(p).mul_(.1).requires_grad_()
    pr,sr=p.detach().clone().requires_grad_(),s.detach().clone().requires_grad_();seed=torch.randn(rows,plan.outdim,device='cuda',dtype=dtype).requires_grad_()
    seedr=seed.detach().clone().requires_grad_();parameters=tuple(norm.parameters())
    callback=globals().get('_CAPTURE_PRESSURE_INPUTS')
    if callback is not None:callback(norm,plan,p,s,seed)
    def chain(fn,p,s,seed):
        y=fn(p,s);first=torch.autograd.grad(y,(p,s,*parameters),seed,create_graph=True)
        second=torch.autograd.grad(sum(g.square().sum() for g in first),(p,s,seed,*parameters),allow_unused=True)
        return(y,*first,*second)
    oracle=lambda p,s:norm_reference(norm,gate_reference(plan,p+s),plan.norm)
    compare(chain(lambda p,s:plan(norm,p,s),p,s,seed),chain(oracle,pr,sr,seedr),dtype)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_gate_norm_physical_all_parameters(dtype):
    norm,plan=fused_fixture(dtype)
    leaves=[(torch.randn(shape,device='cuda',dtype=dtype)*.1).requires_grad_() for shape in ((19,3),(3,plan.dim),(3,plan.dim),(plan.outdim,))]
    ref=[v.detach().clone().requires_grad_() for v in leaves];parameters=tuple(norm.parameters())
    def execute(args,native):
        coords,w,skip,readout=args;p=(coords@w).sin();s=(coords@skip).tanh()
        y=plan(norm,p,s) if native else norm_reference(norm,gate_reference(plan,p+s),plan.norm)
        energy=((y*readout).sin()+.03*y.square()).sum();force=-torch.autograd.grad(energy,coords,create_graph=True)[0]
        loss=(force-.013).square().mean();gradients=torch.autograd.grad(loss,(*args,*parameters),allow_unused=True)
        return(y,energy,force,loss,*gradients)
    compare(execute(leaves,True),execute(ref,False),dtype)


def test_gate_norm_live_full_training_graph():
    dtype=torch.float64;norm,plan=fused_fixture(dtype);reference_norm=deepcopy(norm)
    x=torch.randn(33,plan.dim,device='cuda',dtype=dtype).mul_(.1).requires_grad_();skip=torch.randn_like(x).mul_(.1).requires_grad_()
    seed=torch.randn(33,plan.outdim,device='cuda',dtype=dtype).requires_grad_();parameters=tuple(norm.parameters())
    def chain(fn,x,skip,seed,parameters):
        y=fn(x,skip);first=torch.autograd.grad(y,(x,skip,*parameters),seed,create_graph=True)
        second=torch.autograd.grad(sum(g.square().sum() for g in first),(x,skip,seed,*parameters),allow_unused=True)
        return(y,*first,*second)
    operation=lambda p,s:plan(norm,p,s);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):chain(operation,x,skip,seed,parameters)
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=chain(operation,x,skip,seed,parameters)
    for _ in range(2):
        with torch.no_grad():
            x.copy_(torch.randn_like(x)*.1);skip.copy_(torch.randn_like(skip)*.1);seed.copy_(torch.randn_like(seed))
            for parameter in parameters:parameter.copy_(.7+torch.rand_like(parameter)*.3)
        graph.replay();reference_norm.load_state_dict(norm.state_dict())
        xr,sr,gr=[v.detach().clone().requires_grad_() for v in (x,skip,seed)]
        oracle=lambda p,s:norm_reference(reference_norm,gate_reference(plan,p+s),plan.norm)
        compare(captured,chain(oracle,xr,sr,gr,tuple(reference_norm.parameters())),dtype)
