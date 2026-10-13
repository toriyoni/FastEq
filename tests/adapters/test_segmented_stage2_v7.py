"""All multilinear roles, sparse residuals, and physical force-loss for stage2."""
import os
from pathlib import Path
import sys

import pytest
import torch
import fasteq
from fasteq.ops import segmented_product as implementation

sys.path.insert(0, str(Path(__file__).parents[2] / 'benchmarks/adapters'))
from benchmark_scaling_case import ExclusiveGuard

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def stage2(monkeypatch):
    assert Path(fasteq.__file__).is_relative_to(Path(__file__).parents[2])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-13222c8f-3c05-b93a-e9d6-1192c18a00ce'
    guard = ExclusiveGuard(os.environ['CUDA_VISIBLE_DEVICES'])
    guard.check()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    for suffix in ('TASK_MASK','REDUCE','GEMM','PAIR'):
        monkeypatch.setenv('FASTEQ_EXPERIMENTAL_SEGMENTED_'+suffix,'1')
    torch.manual_seed(14912)
    yield
    torch.cuda.synchronize()
    guard.check()


def description(mixed=False):
    dims = (8*9*11 + 8, 21, 24, 27)
    dense = (3,0,1,2,2,2,3,2,8,9,11,1)
    # Two angular terms share weights and overlap output channels. Coefficients
    # are retained as stored values, including a sign-changing sparse term.
    terms = [dense,(3,0,0,2,1,2,3,2,8,9,11,1)]
    coefficients = [.11329173,-.0813439]
    if mixed:
        terms += [(0,8*9*11,1,2,2,2,3,2,8,8,8,1),
                  (1,0,1,2,0,1,3,2,8,1,8,0)]
        coefficients += [.0391311,.023912]
    return dims,tuple(terms),tuple(coefficients)


def reference(metadata,w,x,y):
    dims,terms,coefficients = implementation._parse(metadata)
    result = x.new_zeros(x.shape[0],dims[3])
    for term,coefficient in zip(terms,coefficients,strict=True):
        mode,wo,xb,xs,yb,ys,zb,zs,c1,c2,co,weighted = term
        a=x[:,xb+torch.arange(c1,device=x.device)*xs]
        b=y[:,yb+torch.arange(c2,device=x.device)*ys]
        width = c1*c2*co if mode==3 else c1*c2 if mode==2 else c1
        weight = w[:,wo:wo+width] if weighted else x.new_ones(1,width)
        if mode==3:
            value=torch.einsum('eu,ev,euvw->ew',a,b,weight.reshape(-1,c1,c2,co))
        elif mode==2:
            value=torch.einsum('eu,ev,euv->eu',a,b,weight.reshape(-1,c1,c2))
        else:
            value=a*b*weight
        result.index_add_(1,zb+torch.arange(co,device=x.device)*zs,value*coefficient)
    return result


def chain(fn,metadata,leaves,subset=False):
    z=fn(metadata,*leaves)
    first=torch.autograd.grad((z.sin()+.1*z.square()).mean(),
                              leaves[1:] if subset else leaves,create_graph=True)
    loss=sum(g.square().mean() for g in first)
    higher=torch.autograd.grad(loss,leaves,allow_unused=True)
    return (z,*first,*[v if v is not None else torch.zeros_like(p)
                       for v,p in zip(higher,leaves,strict=True)])


def close(actual,expected,dtype):
    tolerance=dict(atol=3e-6,rtol=4e-5) if dtype==torch.float32 else dict(atol=2e-11,rtol=2e-10)
    for a,b in zip(actual,expected,strict=True):
        torch.testing.assert_close(a,b,**tolerance)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('mixed,per_row,noncontiguous,rows',
                         [(False,False,False,17),(True,False,True,1025),
                          (True,True,True,17),(False,False,False,0)])
def test_sparse_all_roles_and_mixed_second(dtype,mixed,per_row,noncontiguous,rows):
    metadata=repr(description(mixed))
    dims=description(mixed)[0]
    source=[]
    for n,d in ((rows if per_row else 1,dims[0]),(rows,dims[1]),(rows,dims[2])):
        value=torch.randn(d,n,device='cuda',dtype=dtype).T if noncontiguous else torch.randn(n,d,device='cuda',dtype=dtype)
        source.append(value.requires_grad_())
    if rows:
        close(chain(implementation.segmented_product,metadata,source),
              chain(reference,metadata,source),dtype)
    else:
        z=implementation.segmented_product(metadata,*source)
        assert z.shape==(0,dims[3])
        gradients=torch.autograd.grad(z.sum(),source)
        for gradient in gradients: assert torch.count_nonzero(gradient)==0


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_physical_coordinate_weight_dependence_nondefault_stream(dtype):
    metadata=repr(description(True));dims=description(True)[0]
    coords=torch.randn(1025,3,device='cuda',dtype=dtype,requires_grad=True)
    px=torch.randn(3,dims[1],device='cuda',dtype=dtype,requires_grad=True)*.05
    py=torch.randn(3,dims[2],device='cuda',dtype=dtype,requires_grad=True)*.05
    w=torch.randn(1,dims[0],device='cuda',dtype=dtype,requires_grad=True)*.05
    pw=torch.randn(3,dims[0],device='cuda',dtype=dtype,requires_grad=True)*.001
    leaves=(coords,px,py,w,pw)
    def physical(fn):
        x=(coords@px).sin();y=(coords@py).cos()
        weights=w+(coords.mean(0,keepdim=True)@pw).sin()
        z=fn(metadata,weights,x,y)
        energy=(z.sin()+.1*z.square()).sum()
        force=-torch.autograd.grad(energy,coords,create_graph=True)[0]
        gradients=torch.autograd.grad((force-.13).square().mean(),leaves)
        return z,force,*gradients
    expected=physical(reference)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): actual=physical(implementation.segmented_product)
    torch.cuda.current_stream().wait_stream(stream)
    close(actual,expected,dtype)
    with torch.no_grad(): w.add_(.021);px.mul_(.7)
    close(physical(implementation.segmented_product),physical(reference),dtype)


def test_requested_task_pruning_preserves_later_weight_gradients(monkeypatch):
    metadata=repr(description(True));dims=description(True)[0]
    leaves=[torch.randn(n,d,device='cuda',dtype=torch.float64,requires_grad=True)
            for n,d in ((1,dims[0]),(17,dims[1]),(17,dims[2]))]
    seen=[];execute=implementation._execute
    def record(*args): seen.append((args[1],args[2]));return execute(*args)
    monkeypatch.setattr(implementation,'_execute',record)
    z=implementation.segmented_product(metadata,*leaves)
    first=torch.autograd.grad(z.sin().mean(),leaves[1:],create_graph=True)
    assert [role for role,_ in seen]==[3]
    assert all(value.grad_fn.name()=='_FeaturePairBackward' for value in first)
    torch.autograd.grad(sum(g.square().sum() for g in first),leaves)
    assert any(role==0 for role,_ in seen[1:])
    close(chain(implementation.segmented_product,metadata,leaves,subset=True),
          chain(reference,metadata,leaves,subset=True),torch.float64)


def test_dense_plan_full_precision_with_global_tf32_enabled():
    metadata=repr(description());dims=description()[0]
    leaves=[torch.randn(n,d,device='cuda',requires_grad=True)
            for n,d in ((1,dims[0]),(1025,dims[1]),(1025,dims[2]))]
    expected=chain(implementation.segmented_product,metadata,leaves)
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        actual=chain(implementation.segmented_product,metadata,leaves)
        assert torch.backends.cuda.matmul.allow_tf32
        close(actual,expected,torch.float32)
    finally: torch.backends.cuda.matmul.allow_tf32=False


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_bounded_chunk_tail_dense_all_roles(dtype):
    # E4099 exceeds the 64MiB role0 scratch row limit in both dtypes. Every
    # transpose traverses a partial final tile; nonzero weight storage offset
    # verifies that GEMM views use live Tensor addresses rather than storage base.
    c=64;rows=4099
    metadata=repr(((c**3,c,c,c),((3,0,0,1,0,1,0,1,c,c,c,1),),(1/c,)))
    leaves=[torch.randn(c**3+1,device='cuda',dtype=dtype)[1:].view(1,c**3).requires_grad_(),
            torch.randn(rows,c,device='cuda',dtype=dtype).requires_grad_(),
            torch.randn(rows,c,device='cuda',dtype=dtype).requires_grad_()]
    close(chain(implementation.segmented_product,metadata,leaves),
          chain(reference,metadata,leaves),dtype)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('mixed',[False,True])
def test_real_e3nn_sparse_cg_and_state_load(dtype,mixed):
    from e3nn import o3
    from fasteq.adapters import adapt_model
    from test_equivariant_cuda import check
    if mixed:
        module=o3.TensorProduct('8x0e+9x1o','8x0e+9x1o','8x0e+9x1o',
                                [(0,0,0,'uuu',True),(1,1,0,'uvw',True)]).to(dtype)
    else:
        module=o3.TensorProduct('8x2e','9x2e','11x2e',[(0,0,0,'uvw',True)]).to(dtype)
    inputs=[torch.randn(7,ir.dim,device='cuda',dtype=dtype)
            for ir in (module.irreps_in1,module.irreps_in2)]
    check(module,inputs,atol=3e-5 if dtype==torch.float32 else 2e-9,
          rtol=3e-5 if dtype==torch.float32 else 2e-8)
    native=adapt_model(module)
    native(*inputs) # Construct the cached immutable descriptor plan.
    with torch.no_grad():
        for name,buffer in module.named_buffers():
            if '_w3j_' in name: buffer.mul_(.73)
        module.weight.add_(.011)
    native.load_state_dict(module.state_dict())
    torch.testing.assert_close(native(*inputs),module(*inputs),
                               atol=3e-5 if dtype==torch.float32 else 2e-9,
                               rtol=3e-5 if dtype==torch.float32 else 2e-8)


def test_dense_cuda_execution_never_uses_eqx():
    from eqx_cuda_spy import forbid_eqx_cuda
    metadata=repr(description(True));dims=description(True)[0]
    leaves=[torch.randn(n,d,device='cuda',dtype=torch.float64,requires_grad=True)
            for n,d in ((1,dims[0]),(17,dims[1]),(17,dims[2]))]
    expected=chain(reference,metadata,leaves)
    with forbid_eqx_cuda() as calls:
        actual=chain(implementation.segmented_product,metadata,leaves)
    assert calls==[]
    close(actual,expected,torch.float64)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('mode',[0,1,2,3])
def test_metadata_specialization_and_partial_gradient_roles(dtype,mode):
    c1,c2,co=8,(1 if mode==1 else 9 if mode in (2,3) else 8),(11 if mode==3 else 8)
    width=c1*c2*co if mode==3 else c1*c2 if mode==2 else c1
    metadata=repr(((width,21,24,27),((mode,0,1,2,2,2,3,2,c1,c2,co,1),),(.09131987,)))
    leaves=[torch.randn(n,d,device='cuda',dtype=dtype,requires_grad=True)
            for n,d in ((1,width),(1025,21),(1025,24))]
    for requested in ((0,),(1,),(2,),(0,2)):
        results=[]
        for fn in (reference,implementation.segmented_product):
            z=fn(metadata,*leaves)
            grads=torch.autograd.grad(z.sin().mean(),[leaves[r] for r in requested],create_graph=True)
            higher=torch.autograd.grad(sum(g.square().mean() for g in grads),leaves,allow_unused=True)
            results.append((z,*grads,*[v if v is not None else torch.zeros_like(p)
                                      for v,p in zip(higher,leaves,strict=True)]))
        close(results[1],results[0],dtype)


def test_dense_cuda_graph_replay_observes_live_weights():
    metadata=repr(description(True));dims=description(True)[0]
    leaves=[torch.randn(n,d,device='cuda',dtype=torch.float64,requires_grad=True)
            for n,d in ((1,dims[0]),(17,dims[1]),(17,dims[2]))]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):chain(implementation.segmented_product,metadata,leaves)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        actual=chain(implementation.segmented_product,metadata,leaves)
    with torch.no_grad():leaves[0].add_(.13);leaves[1].mul_(.71)
    graph.replay()
    close(actual,chain(reference,metadata,leaves),torch.float64)


@pytest.mark.parametrize('chosen',[0,1])
def test_paired_feature_vjp_unused_output_seed(chosen):
    metadata=repr(description(True));dims=description(True)[0]
    leaves=[torch.randn(n,d,device='cuda',dtype=torch.float64,requires_grad=True)
            for n,d in ((1,dims[0]),(17,dims[1]),(17,dims[2]))]
    results=[]
    for fn in (reference,implementation.segmented_product):
        z=fn(metadata,*leaves)
        first=torch.autograd.grad(z.sin().mean(),leaves[1:],create_graph=True)
        second=torch.autograd.grad(first[chosen].square().mean(),leaves)
        results.append((z,*first,*second))
    close(results[1],results[0],torch.float64)


def test_mixed_dense_channel_shapes_and_third_derivatives():
    dims=(1944,24,17,20)
    terms=((3,0,0,1,0,1,0,1,8,9,11,1),
           (3,792,8,1,9,1,11,1,16,8,9,1))
    metadata=repr((dims,terms,(.13,-.07)))
    leaves=[torch.randn(n,d,device='cuda',dtype=torch.float64,requires_grad=True)
            for n,d in ((1,dims[0]),(5,dims[1]),(5,dims[2]))]
    results=[]
    for fn in (reference,implementation.segmented_product):
        z=fn(metadata,*leaves)
        first=torch.autograd.grad(z.sin().sum(),leaves,create_graph=True)
        second=torch.autograd.grad(sum(g.square().sum() for g in first),leaves,create_graph=True)
        third=torch.autograd.grad(sum(g.sin().mean() for g in second),leaves)
        results.append((z,*first,*second,*third))
    close(results[1],results[0],torch.float64)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_diagonal_angular_batched_gemm_storage_offsets(dtype):
    dims=(792,26,29,11)
    terms=tuple((3,0,k,3,k,3,0,1,8,9,11,1) for k in range(3))
    metadata=repr((dims,terms,(.093,.093,.093)))
    leaves=[torch.randn(n*d+1,device='cuda',dtype=dtype)[1:].view(n,d).requires_grad_()
            for n,d in ((1,dims[0]),(1025,dims[1]),(1025,dims[2]))]
    close(chain(implementation.segmented_product,metadata,leaves),
          chain(reference,metadata,leaves),dtype)
