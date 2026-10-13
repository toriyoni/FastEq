"""Native CUDA segmented tensor products with complete recursive gradients.

The descriptor keeps angular coupling sparse and channel contraction dense.
Unlike the historical one-hot FCTP specialization, both feature operands and
the weights may be arbitrary differentiable floating point tensors. All
transpose roles use this same multilinear CUDA contraction, including second
and higher derivatives needed for force-loss training.
"""

from ast import literal_eval
from functools import lru_cache

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


@lru_cache(maxsize=32)
def _source(dtype, role, required):
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
      I r0,I r1,I r2,I r3,int d0,int d1,int d2,int d3,int nt) {{
      const int dims[4]={{d0,d1,d2,d3}};
      I index=(I)blockIdx.x*blockDim.x+threadIdx.x;
      int width=dims[{role}];
      if(index>=count*width) return;
      I output_row=index/width; int coordinate=index%width;
      T total=0;
      I begin=count==1?0:output_row, end=count==1?rows:output_row+1;
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
      result[index]=total;
    }}
    """


def _execute(metadata, role, required, rows, values):
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
    code = _source(x.dtype, role, required)
    compiled = kernels([code], x.device)[code]
    args = [v.data_ptr() for v in values] + [
        descriptors.data_ptr(),
        scales.data_ptr(),
        result.data_ptr(),
    ]
    args += [count, rows, *counts, *dimensions, descriptors.shape[0]]
    runtime().launch(
        [(compiled, args, (result.numel() + 127) // 128, 1, 128, 0)],
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    return result


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
        grads = [
            _Product.apply(
                ctx.kernel_metadata,
                role,
                ctx.required | (1 << ctx.role),
                ctx.rows,
                *values,
            )
            if role != ctx.role and need
            else None
            for role, need in enumerate(ctx.needs_input_grad[4:])
        ]
        return None, None, None, None, *grads


def segmented_product(metadata, weight, input1, input2):
    """Evaluate a sparse angular/dense channel contraction using FastEq CUDA."""
    dimensions = _parse(metadata)[0]
    rows = input1.shape[0]
    placeholder = input1.new_empty(0, dimensions[3])
    return _Product.apply(metadata, 3, 0, rows, weight, input1, input2, placeholder)
