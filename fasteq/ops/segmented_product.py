"""Native CUDA segmented tensor products with complete recursive gradients.

The descriptor keeps angular coupling sparse and channel contraction dense.
Unlike the historical one-hot FCTP specialization, both feature operands and
the weights may be arbitrary differentiable floating point tensors. All
transpose roles use this same multilinear CUDA contraction, including second
and higher derivatives needed for force-loss training.
"""

from ast import literal_eval
from functools import lru_cache
import math
import os

import torch


@lru_cache(maxsize=256)
def _parse(metadata):
    return literal_eval(metadata)


@lru_cache(maxsize=256)
def _tables(metadata, dtype, device):
    dimensions, terms, coefficients = _parse(metadata)
    with torch.inference_mode(False), torch.no_grad():
        descriptors = torch.tensor(terms, dtype=torch.int32, device=device).reshape(
            -1, 12
        )
        scales = torch.tensor(coefficients, dtype=dtype, device=device)
    return dimensions, descriptors, scales


@lru_cache(maxsize=256)
def _source(dtype, role, required, chunked=False, compact=False):
    scalar = "double" if dtype == torch.float64 else "float"
    # The four operands are (weight, input1, input2, output/cotangent). An
    # unweighted path has no weight factor. Required bit zero prevents a
    # subsequent transpose from reviving a path removed by a weight adjoint.
    return f"""
    using T={scalar};
    using I=long long;
    __device__ __forceinline__ T term_value(const T* a0,const T* a1,const T* a2,const T* a3,
      I r0,I r1,I r2,I r3,int d0,int d1,int d2,int d3,I e,
      int wi,int xi,int yi,int zi,bool weighted,T coefficient) {{
      T value=coefficient;
      {"" if role == 0 else "if(weighted) value*=a0[(r0==1?0:e)*d0+wi];"}
      {"" if role == 1 else "value*=a1[(r1==1?0:e)*d1+xi];"}
      {"" if role == 2 else "value*=a2[(r2==1?0:e)*d2+yi];"}
      {"" if role == 3 else "value*=a3[(r3==1?0:e)*d3+zi];"}
      return value;
    }}
    extern "C" __global__ void run(const T* a0,const T* a1,const T* a2,const T* a3,
      const int* terms,const T* scales,T* result,I count,I rows,
      I r0,I r1,I r2,I r3,int d0,int d1,int d2,int d3,int nt{',I span' if chunked else ''}{',const int* active,int active_width' if compact else ''}) {{
      const int dims[4]={{d0,d1,d2,d3}};
      I index=(I)blockIdx.x*blockDim.x+threadIdx.x;
      int width={'active_width' if compact else f'dims[{role}]'};
      if(index>=count*width) return;
      I output_row=index/width; int coordinate={'active[index%width]' if compact else 'index%width'};
      T total=0;
      I begin={'(I)blockIdx.y*span' if chunked else 'count==1?0:output_row'}, end={'min(rows,begin+span)' if chunked else 'count==1?rows:output_row+1'};
      for(I e=begin;e<end;++e) {{
        for(int t=0;t<nt;++t) {{
          const int* p=terms+12*t;
          int mode=p[0],wo=p[1],xb=p[2],xs=p[3],yb=p[4],ys=p[5],zb=p[6],zs=p[7];
          int c1=p[8],c2=p[9],co=p[10]; bool weighted=p[11];
          if(({required}&1) && !weighted) continue;
          int u0=0,u1=0,v0=0,v1=0,w0=0,w1=0;
          if({role}==3) {{
            int local=coordinate-zb;
            if(local<0 || local%zs || local/zs>=co) continue;
            w0=local/zs;w1=w0+1;
            if(mode==3) {{u0=0;u1=c1;v0=0;v1=c2;}}
            else {{u0=w0;u1=u0+1;v0=(mode==0?w0:0);v1=(mode==2?c2:v0+1);}}
          }} else if({role}==0) {{
            if(!weighted) continue;
            int local=coordinate-wo;
            int size=mode==3?c1*c2*co:mode==2?c1*c2:c1;
            if(local<0 || local>=size) continue;
            if(mode==3) {{u0=local/(c2*co);v0=(local/co)%c2;w0=local%co;}}
            else if(mode==2) {{u0=local/c2;v0=local%c2;w0=u0;}}
            else {{u0=local;v0=mode==0?u0:0;w0=u0;}}
            u1=u0+1;v1=v0+1;w1=w0+1;
          }} else if({role}==1) {{
            int local=coordinate-xb;
            if(local<0 || local%xs || local/xs>=c1) continue;
            u0=local/xs;u1=u0+1;
            if(mode==3) {{v0=0;v1=c2;w0=0;w1=co;}}
            else {{v0=mode==0?u0:0;v1=mode==2?c2:v0+1;w0=u0;w1=w0+1;}}
          }} else {{
            int local=coordinate-yb;
            if(local<0 || local%ys || local/ys>=c2) continue;
            v0=local/ys;v1=v0+1;
            if(mode==3) {{u0=0;u1=c1;w0=0;w1=co;}}
            else if(mode==0) {{u0=v0;u1=u0+1;w0=v0;w1=w0+1;}}
            else {{u0=0;u1=c1;w0=0;w1=1;}}
          }}
          for(int u=u0;u<u1;++u) for(int v=v0;v<v1;++v) for(int w=w0;w<w1;++w) {{
            int out_channel=(mode==3?w:u);
            int wi=wo+(mode==3?(u*c2+v)*co+out_channel:mode==2?u*c2+v:u);
            total+=term_value(a0,a1,a2,a3,r0,r1,r2,r3,d0,d1,d2,d3,e,
              wi,xb+u*xs,yb+v*ys,zb+out_channel*zs,weighted,scales[t]);
          }}
        }}
      }}
      result[{ '(I)blockIdx.y*width+index' if chunked else f'output_row*dims[{role}]+coordinate' }]=total;
    }}
    """


@lru_cache(maxsize=4)
def _reduction_source(dtype, compact=False):
    scalar = "double" if dtype == torch.float64 else "float"
    return f"""
    using T={scalar}; using I=long long;
    extern "C" __global__ void run(const T* partial,T* result,I width,I chunks{',const int* active' if compact else ''}) {{
      I i=(I)blockIdx.x*blockDim.x+threadIdx.x;
      if(i>=width) return;
      T total=0;
      for(I c=0;c<chunks;++c) total+=partial[c*width+i];
      result[{'active[i]' if compact else 'i'}]=total;
    }}
    """


@lru_cache(maxsize=256)
def _specialized_source(metadata, dtype, role, required, chunked, compact=False):
    """Constant-fold small sparse descriptors without a per-row table scan."""
    code = _source(dtype, role, required, chunked, compact)
    _, terms, coefficients = _parse(metadata)
    scalar_y = terms and all(term[0] in (1, 2) and term[9] == 1 for term in terms)
    limit = 16 if scalar_y else 8
    if not terms or len(terms) > limit or not all(map(math.isfinite, coefficients)):
        return code
    marker = "for(int t=0;t<nt;++t) {"
    start = code.index(marker)
    opening = start + len(marker) - 1
    depth, end = 1, opening + 1
    while depth:
        depth += (code[end] == "{") - (code[end] == "}")
        end += 1
    body = code[opening + 1:end - 1]
    declarations_start = body.index("const int* p=")
    declarations_end = body.index("bool weighted=p[11];") + len("bool weighted=p[11];")
    blocks = []
    for term, coefficient in zip(terms, coefficients, strict=True):
        names = ("mode", "wo", "xb", "xs", "yb", "ys", "zb", "zs", "c1", "c2", "co")
        constants = "constexpr int " + ",".join(
            f"{name}={value}" for name, value in zip(names, term[:11], strict=True)
        ) + f"; constexpr bool weighted={'true' if term[11] else 'false'};"
        constants += f" constexpr T coefficient=(T)({coefficient!r});"
        specialized = body[:declarations_start] + constants + body[declarations_end:]
        specialized = specialized.replace("scales[t]", "coefficient")
        # A one-iteration loop preserves each descriptor's early continue.
        blocks.append("for(int term_once=0;term_once<1;++term_once) {" + specialized + "}")
    return code[:start] + "\n".join(blocks) + code[end:]


@lru_cache(maxsize=256)
def _scalar_y_warp_eligible(metadata):
    _, terms, coefficients = _parse(metadata)
    return (0 < len(terms) <= 16 and all(map(math.isfinite, coefficients))
            and all(term[0] in (1, 2) and term[9] == 1 and term[8] >= 32
                    for term in terms))


@lru_cache(maxsize=256)
def _scalar_y_warp_source(metadata, dtype, required, chunked, compact=False):
    """One warp contracts a scalar-Y adjoint over the channel dimension."""
    code = _specialized_source(metadata, dtype, 2, required, chunked, compact)
    code = code.replace(
        "I index=(I)blockIdx.x*blockDim.x+threadIdx.x;",
        "I index=((I)blockIdx.x*blockDim.x+threadIdx.x)/32; int lane=threadIdx.x&31;",
    )
    code = code.replace("for(int u=u0;u<u1;++u)",
                        "for(int u=u0+lane;u<u1;u+=32)")
    code = code.replace(
        "      result[",
        "      for(int offset=16;offset;offset/=2) total+=__shfl_down_sync(0xffffffff,total,offset);\n"
        "      if(lane==0) result[",
    )
    return code


@lru_cache(maxsize=256)
def _active_coordinates(metadata, role, required, device):
    dimensions, terms, _ = _parse(metadata)
    ranges = []
    estimate = 0
    for term in terms:
        if ((required & 1) or role == 0) and not term[11]:
            continue
        if role == 0:
            size = (term[8] * term[9] * term[10] if term[0] == 3
                    else term[8] * term[9] if term[0] == 2 else term[8])
            base, stride = term[1], 1
        else:
            base, stride, size = term[2 * role], term[2 * role + 1], term[7 + role]
        estimate += size
        if estimate * 2 >= dimensions[role]:
            return None
        ranges.append((base, stride, size))
    coordinates = sorted({base + i * stride for base, stride, size in ranges for i in range(size)})
    with torch.inference_mode(False), torch.no_grad():
        return torch.tensor(coordinates, dtype=torch.int32, device=device)


@lru_cache(maxsize=32)
def _dense_source(dtype, channels=None):
    scalar = "double" if dtype == torch.float64 else "float"
    source = f"""
    using T={scalar}; using I=long long;
    extern "C" __global__ void pack(const T* a,const T* b,T* out,int op,
      I start,I n,I ra,I rb,int da,int db,int ba,int sa,int ca,int bb,int sb,int cb,int co) {{
      I i=(I)blockIdx.x*blockDim.x+threadIdx.x;
      I size=op==2?(I)ca*cb*co:n*ca*(op==1?cb:1);
      if(i>=size) return;
      if(op==2) {{
        int w=i%co,u=(i/co)%ca,v=i/(co*ca);
        out[i]=a[ba+(u*cb+v)*co+w];
      }} else if(op==1) {{
        I e=start+i/((I)ca*cb);int u=(i/cb)%ca,v=i%cb;
        out[i]=a[(ra==1?0:e)*da+ba+u*sa]*b[(rb==1?0:e)*db+bb+v*sb];
      }} else {{
        I e=start+i/ca;int u=i%ca;
        out[i]=a[(ra==1?0:e)*da+ba+u*sa];
      }}
    }}
    extern "C" __global__ void scatter(const T* temp,const T* partner,T* out,int op,
      I start,I n,I rp,int dp,int pb,int ps,int dout,int ob,int os,int channels,int contract,T coefficient) {{
      I i=(I)blockIdx.x*blockDim.x+threadIdx.x;
      if(i>=n*channels) return;
      I row=i/channels,e=start+row;int c=i%channels;
      T value=0;
      if(op==0) {{
        for(int v=0;v<contract;++v)
          value+=temp[(row*contract+v)*channels+c]*partner[(rp==1?0:e)*dp+pb+v*ps];
      }} else if(op==2) {{
        for(int k=0;k<contract;++k)
          value+=temp[(row*channels+c)*contract+k]*partner[(rp==1?0:e)*dp+pb+k*ps];
      }} else if(op==3) {{
        for(int k=0;k<contract;++k)
          value+=temp[(row*contract+k)*channels+c]*partner[(rp==1?0:e)*dp+pb+k*ps];
      }} else value=temp[i];
      out[e*dout+ob+c*os]+=coefficient*value;
    }}
    extern "C" __global__ void angular_outer(const T* x,const T* y,T* out,
      const int* terms,const T* scales,I start,I n,I rx,I ry,int dx,int dy,
      int begin,int end,int c1,int c2) {{
      I i=(I)blockIdx.x*blockDim.x+threadIdx.x;
      if(i>=n*c1*c2) return;
      I e=start+i/((I)c1*c2);int u=(i/c2)%c1,v=i%c2;
      T total=0;
      for(int t=begin;t<end;++t) {{
        const int* p=terms+12*t;
        total+=scales[t]*x[(rx==1?0:e)*dx+p[2]+u*p[3]]*y[(ry==1?0:e)*dy+p[4]+v*p[5]];
      }}
      out[i]=total;
    }}
    """
    if channels is not None:
        c1, c2, co = channels
        source = source.replace(
            "int begin,int end,int c1,int c2) {",
            f"int begin,int end,int runtime_c1,int runtime_c2) {{ constexpr int c1={c1},c2={c2};",
        )
        source = source.replace(
            "int channels,int contract,T coefficient) {",
            "int runtime_channels,int runtime_contract,T coefficient) {"
            + f" int channels=(op==0 || op==1)?{co}:(op==2?{c1}:{c2});"
            + f" int contract=(op==0 || op==2)?{c2}:{c1};",
        )
        source = source.replace(
            "int bb,int sb,int cb,int co) {",
            f"int bb,int sb,int cb,int co) {{ if(op==0) ca={co};",
        )
    return source


@lru_cache(maxsize=256)
def _dense_plan(metadata, dtype, device):
    from fasteq.cuda.jit import kernels, runtime
    dimensions, terms, coefficients = _parse(metadata)
    dense, scales, residual, residual_scales = [], [], [], []
    for term, coefficient in zip(terms, coefficients, strict=True):
        # Tiny channels and unsupported scratch shapes keep the existing
        # FastEq CUDA contraction. Sparse angular coefficients are unchanged.
        fixed = term[8] * term[9] * term[10] * (8 if dtype == torch.float64 else 4)
        if term[0] == 3 and term[11] and min(term[8:11]) >= 8 and fixed < 32 * 1024 * 1024:
            dense.append(term)
            scales.append(coefficient)
        else:
            residual.append(term)
            residual_scales.append(coefficient)
    if not dense:
        return None, None
    # Group only equal shared weight tensors and equal output angular slices.
    # CUDA fuses their sparse angular outer products before one channel GEMM.
    groups = {}
    for term, coefficient in zip(dense, scales, strict=True):
        key = (term[1], term[6], term[7], *term[8:11])
        groups.setdefault(key, []).append((term, coefficient))
    grouped = [entry for group in groups.values() for entry in group]
    dense, scales = zip(*grouped, strict=True)
    channel_shapes = {tuple(term[8:11]) for term in dense}
    channel_shape = next(iter(channel_shapes)) if len(channel_shapes) == 1 else None
    source = _dense_source(dtype, channel_shape)
    pack = kernels([source], device, entry="pack")[source]
    scatter = kernels([source], device, entry="scatter")[source]
    outer = kernels([source], device, entry="angular_outer")[source]
    _, dense_tables, dense_scales = _tables(repr((dimensions, dense, scales)), dtype, device)
    plan = runtime().SegmentedProductPlan(dimensions, dense, scales, pack, scatter,
                                         outer, dense_tables, dense_scales)
    rest = repr((dimensions, tuple(residual), tuple(residual_scales))) if residual else None
    return plan, rest


def _execute(metadata, role, required, rows, values):
    if os.environ.get("FASTEQ_EXPERIMENTAL_SEGMENTED_GEMM") == "1":
        x = values[1]
        if x.is_cuda and x.dtype in (torch.float32, torch.float64):
            # Role outputs and shared seeds must retain exactly the original
            # broadcast domain; uncommon broadcast/per-row weights fall back.
            supported = values[0].shape[0] == 1 and all(
                values[r].shape[0] == rows for r in (1, 2, 3) if r != role
            ) and (role == 0 or values[role].shape[0] in (0, rows))
            if supported and rows:
                if any(v.device != x.device or v.dtype != x.dtype for v in values):
                    raise ValueError("Segmented product operands must share device and dtype")
                plan, residual = _dense_plan(metadata, x.dtype, x.device)
                if plan is not None:
                    values = [v.contiguous() for v in values]
                    seed = (_execute_cuda(residual, role, required, rows, values)
                            if residual else x.new_empty(0))
                    return plan.evaluate(role, required, rows, values, seed)
    return _execute_cuda(metadata, role, required, rows, values)


def _execute_cuda(metadata, role, required, rows, values):
    from fasteq.cuda.jit import kernels, runtime

    x = values[1]
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
        raise TypeError("Segmented products require CUDA FP32/FP64")
    dimensions, descriptors, scales = _tables(metadata, x.dtype, x.device)
    if any(v.device != x.device or v.dtype != x.dtype for v in values):
        raise ValueError("Segmented product operands must share device and dtype")
    values = [v.contiguous() for v in values]
    counts = [v.shape[0] for v in values]
    count = rows if role == 3 else counts[role]
    result = x.new_empty(count, dimensions[role])
    if not rows or not dimensions[role]:
        return result.zero_()
    optimized = os.environ.get("FASTEQ_EXPERIMENTAL_SEGMENTED_REDUCE") == "1"
    active = _active_coordinates(metadata, role, required, x.device) if optimized else None
    width = dimensions[role] if active is None else active.numel()
    if active is not None:
        result.zero_()
        if not width:
            return result
    # Shared adjoints previously scanned every row in a single thread. Split
    # that exact CUDA contraction into independent row tiles; only bounded
    # partial sums are allocated, never a rows-by-weights outer-product tensor.
    scratch_limit = 64 * 1024 * 1024
    chunks = 1
    if (
        optimized
        and count == 1 and rows > 256
    ):
        chunks = min(65535, (rows + 255) // 256,
                     scratch_limit // (width * x.element_size()))
        chunks = max(1, chunks)
    span = (rows + chunks - 1) // chunks
    partial = x.new_empty(chunks, width) if chunks > 1 else result
    warp = (optimized and role == 2
            and os.environ.get("FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y_WARP") == "1"
            and _scalar_y_warp_eligible(metadata))
    if warp:
        code = _scalar_y_warp_source(metadata, x.dtype, required, chunks > 1, active is not None)
    else:
        code = (_specialized_source(metadata, x.dtype, role, required, chunks > 1, active is not None)
                if optimized
                else _source(x.dtype, role, required, chunks > 1))
    compiled = kernels([code], x.device)[code]
    args = [v.data_ptr() for v in values] + [
        descriptors.data_ptr(),
        scales.data_ptr(),
        partial.data_ptr(),
    ]
    args += [count, rows, *counts, *dimensions, descriptors.shape[0]]
    if chunks > 1:
        args += [span]
    if active is not None:
        args += [active.data_ptr(), width]
    launches = [(compiled, args, (count * width * (32 if warp else 1) + 127) // 128, chunks, 128, 0)]
    if chunks > 1:
        reduce_code = _reduction_source(x.dtype, active is not None)
        reduce_kernel = kernels([reduce_code], x.device)[reduce_code]
        launches.append((reduce_kernel,
                         [partial.data_ptr(), result.data_ptr(), width, chunks] +
                         ([active.data_ptr()] if active is not None else []),
                         (width + 127) // 128, 1, 128, 0))
    runtime().launch(
        launches,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    return result


class _FeaturePair(torch.autograd.Function):
    """Share a live DZ@W transpose within one VJP, with recursive adjoints."""
    @staticmethod
    def forward(ctx, metadata, required, rows, *values):
        ctx.kernel_metadata, ctx.required, ctx.rows = metadata, required, rows
        ctx.save_for_backward(*values)
        ctx.set_materialize_grads(False)
        x = values[1]
        plan, residual = _dense_plan(metadata, x.dtype, x.device)
        values = [v.contiguous() for v in values]
        seedx = (_execute_cuda(residual, 1, required, rows, values)
                 if residual else x.new_empty(0))
        seedy = (_execute_cuda(residual, 2, required, rows, values)
                 if residual else x.new_empty(0))
        return tuple(plan.evaluate_features(required, rows, values, seedx, seedy))

    @staticmethod
    def backward(ctx, gradx, grady):
        w, x, y, z = ctx.saved_tensors
        requested = ctx.needs_input_grad[3:]
        if os.environ.get("FASTEQ_EXPERIMENTAL_SEGMENTED_TASK_MASK") == "1":
            from .equivariant_norm import _requested_gradients
            requested = _requested_gradients(requested)

        def product(role, differentiated, a, b):
            return _Product.apply(ctx.kernel_metadata, role, ctx.required | (1 << differentiated),
                                  ctx.rows, w, a, b, z)

        def add(a, b):
            if a is None:
                return b
            return a if b is None else a + b

        dw = None
        if requested[0]:
            dw = add(product(0, 1, gradx, y) if gradx is not None else None,
                     product(0, 2, x, grady) if grady is not None else None)
        dx = product(1, 2, x, grady) if requested[1] and grady is not None else None
        dy = product(2, 1, gradx, y) if requested[2] and gradx is not None else None
        dz = None
        if requested[3]:
            dz = add(product(3, 1, gradx, y) if gradx is not None else None,
                     product(3, 2, x, grady) if grady is not None else None)
        return None, None, None, dw, dx, dy, dz


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, role, required, rows, *values):
        ctx.kernel_metadata, ctx.role, ctx.required, ctx.rows = (
            metadata,
            role,
            required,
            rows,
        )
        ctx.save_for_backward(*values)
        return _execute(metadata, role, required, rows, values)

    @staticmethod
    def backward(ctx, gradient):
        values = list(ctx.saved_tensors)
        values[ctx.role] = gradient
        requested = ctx.needs_input_grad[4:]
        if os.environ.get("FASTEQ_EXPERIMENTAL_SEGMENTED_TASK_MASK") == "1":
            from .equivariant_norm import _requested_gradients
            requested = _requested_gradients(requested)
        feature_pair = None
        if (ctx.role in (0, 3) and requested[1] and requested[2]
                and os.environ.get("FASTEQ_EXPERIMENTAL_SEGMENTED_GEMM") == "1"
                and os.environ.get("FASTEQ_EXPERIMENTAL_SEGMENTED_PAIR") == "1"
                and values[0].shape[0] == 1 and ctx.rows
                and all(values[r].shape[0] == ctx.rows for r in (1, 2, 3))):
            plan, _ = _dense_plan(ctx.kernel_metadata, values[1].dtype, values[1].device)
            if plan is not None:
                feature_pair = _FeaturePair.apply(ctx.kernel_metadata,
                                                 ctx.required | (1 << ctx.role),
                                                 ctx.rows, *values)
        grads = []
        for role, need in enumerate(requested):
            if role == ctx.role or not need:
                grads.append(None)
            elif feature_pair is not None and role in (1, 2):
                grads.append(feature_pair[role - 1])
            else:
                grads.append(_Product.apply(ctx.kernel_metadata, role,
                                            ctx.required | (1 << ctx.role),
                                            ctx.rows, *values))
        return None, None, None, None, *grads


def segmented_product(metadata, weight, input1, input2):
    """Evaluate a sparse angular/dense channel contraction using FastEq CUDA."""
    dimensions = _parse(metadata)[0]
    rows = input1.shape[0]
    placeholder = input1.new_empty(0, dimensions[3])
    return _Product.apply(metadata, 3, 0, rows, weight, input1, input2, placeholder)
