"""Fused FastEq CUDA Gate/Norm with the existing GateNorm autograd nodes."""
import ast
from functools import lru_cache
from pathlib import Path
import torch
from .native_equivariant_norm import _warp_primitive, _parameter_function, _partials_source, _flatten_tensors

_ROLES={'forward':0,'backward':1,'backward_direct':1,'double_backward':2,'double_backward_direct':2}
_ENTRIES=('fasteq_gate_norm_forward','fasteq_gate_norm_backward','fasteq_gate_norm_double')


@lru_cache(maxsize=256)
def _source(metadata,dtype):
    role,items=ast.literal_eval(metadata);kw=dict(items);scalar='double' if dtype==torch.float64 else 'float'
    headers=Path(__file__).parents[1]/'cuda/src';threads=128
    normdim=kw['DOUT'];block=max(threads,1<<(normdim-1).bit_length())
    shared=2*normdim*(8 if dtype==torch.float64 else 4)+4*(threads//32)*(8 if dtype==torch.float64 else 4)<=48*1024
    definitions={'FASTEQ_NORM_TYPE':scalar,'FASTEQ_NORM_FP64':int(dtype==torch.float64),'FASTEQ_NORM_ROLE':role,
        'FASTEQ_NORM_DIM':normdim,'FASTEQ_NORM_BLOCK':block,'FASTEQ_NORM_THREADS':threads,'FASTEQ_NORM_CENTERS':0,
        'FASTEQ_GN_ENTRY':_ENTRIES[role],'FASTEQ_GN_IN':kw['DIN'],'FASTEQ_GN_FAN':kw.get('MAX_FAN',1),
        'FASTEQ_GN_ROLE':role,'FASTEQ_GN_EPS':format(kw['EPS'],'.17g'),'FASTEQ_GN_HAS_U':int(kw.get('HAS_U',False)),
        'FASTEQ_GN_DZFLAG':int(kw.get('DZFLAG',kw.get('DDZFLAG',False))),'FASTEQ_GN_DDYFLAG':int(kw.get('DDYFLAG',False)),
        'FASTEQ_GN_GPARTIAL':int(any(kw.get('GGRADFLAGS',()))),'FASTEQ_GN_CACHE_TRANSPOSE':int(shared)}
    gamma=[f'g{i}' for i in range(len(kw['GFLAGS']))];h=[f'h{i}' for i in range(len(kw.get('HFLAGS',())))]
    metadata_names=['src','gate','act','scale','in_out','in_role','coefficient']
    if role==0:names=['projected','skip',*metadata_names[:4],'coefficient',*gamma,'y'];outputs={'y'}
    elif role==1:names=['projected','skip','dy',*metadata_names,*gamma,'dz','partials'];outputs={'dz','partials'}
    else:names=['projected','skip','dy','u',*h,*metadata_names,*gamma,'dz','ddy','partials'];outputs={'dz','ddy','partials'}
    integer={'src':'int','gate':'int','act':'signed char','in_out':'int','in_role':'signed char'}
    definitions['FASTEQ_GN_SIGNATURE']=','.join(('' if n in outputs else 'const ')+integer.get(n,scalar)+'* '+n for n in names)
    rematerialized=[]
    for start,channels,dimension,slot,base,*_ in kw['PARTS']:
        if not channels or slot<0 or not kw.get('GGRADFLAGS',()) or not kw['GGRADFLAGS'][slot]:continue
        padded=1<<(dimension-1).bit_length();offset=kw['GOFFSETS'][slot]+base
        direction='u' if role==2 else 'nullptr'
        rematerialized += [f'for(int channel=threadIdx.x;channel<{channels};channel+=NormThreads) {{',
            f'NormT component[{padded}];for(int angle=0;angle<{padded};++angle) {{component[angle]=NormT(0);',
            f'if(angle<{dimension}) {{const int i={start}+channel*{dimension}+angle;',
            f'const auto v=gate_norm_value(projected,skip,{direction},src,gate,act,scale,row,i);',
            'component[angle]=gate_norm_adjoint(v,coefficient[i],dy[row*NormD+i],FASTEQ_GN_GAMMA(i),FASTEQ_GN_H(i),r,a,s,c).parameter;}}',
            f'for(int distance={padded//2};distance;distance>>=1)for(int angle=0;angle<distance;++angle)component[angle]=norm_add(component[angle],component[angle+distance]);',
            f'partials[row*{kw.get("WIDTH",0)}+{offset}+channel]=component[0];}}']
    definitions['FASTEQ_GN_REMATERIALIZED_PARTIALS']=''.join(rematerialized)
    parameters=''
    for tag,prefix,flags,default in [('GAMMA','g',kw['GFLAGS'],1),('H','h',kw.get('HFLAGS',()),0)]:
        function,call=_parameter_function('fasteq_gn_'+tag.lower(),scalar,prefix,kw['PARTS'],flags,False,default=default)
        parameters+=function;definitions['FASTEQ_GN_'+tag+'(i)']=call
    # Reuse the verified Norm CTA statistics and local ordered tree verbatim.
    # The extraction excludes Norm's global kernels; no duplicate normalization
    # implementation or Triton backend is compiled for this fused operation.
    core=(headers/'native_equivariant_norm.cuh').read_text().split('extern "C" __global__ void FASTEQ_NORM_ENTRY',1)[0]+'\n#endif\n'
    parameters+='__device__ __constant__ int fasteq_norm_center_start[]={0};\n__device__ __constant__ int fasteq_norm_center_width[]={0};\n'
    gate='#define FASTEQ_GATE_ACTIVATION_ONLY 1\n'+(headers/'native_gate.cuh').read_text()+'\n#undef FASTEQ_GATE_ACTIVATION_ONLY\n'
    prefix=''.join(f'#define {k} {v}\n' for k,v in definitions.items())
    normalized={**kw,'D':normdim,'IR_MUL':False,'BGRADFLAGS':()}
    partials=_partials_source(scalar,role,normalized) if role else ''
    prototype=(f'__device__ __forceinline__ void fasteq_norm_parameter_partials(const {scalar}*,const {scalar}*,{scalar}*,long long);\n' if role else '')
    return prefix+_warp_primitive()+gate+parameters+core+prototype+(headers/'native_gate_norm.cuh').read_text()+partials


def source_for(role,kwargs,dtype):return _source(repr((role,tuple(sorted(kwargs.items())))),dtype)


def launch(plan,fn,arguments,x,kwargs):
    name=getattr(fn,'__name__',None) or getattr(getattr(fn,'fn',None),'__name__',None)
    if name not in _ROLES:raise ValueError('Unsupported fused native GateNorm entry: '+str(name))
    if not x.shape[0]:return
    role=_ROLES[name];source=source_for(role,kwargs,x.dtype);values=_flatten_tensors(arguments)
    key=(fn,x.device,x.dtype,tuple(kwargs.items()),source)
    prepared=plan._launchers.get(key)
    if prepared is None:
        from fasteq.cuda.jit import kernels
        kernel=kernels([source],x.device,entry=_ENTRIES[role])[source]
        prepared=(kernel,128,0);plan._launchers[key]=prepared
    kernel,threads,shared=prepared
    kernel.launch([v.data_ptr() for v in values],x.shape[0],1,threads,torch.cuda.current_stream(x.device).cuda_stream,shared)


__all__=['launch','source_for']
