"""Hand-written FastEq CUDA modulation; preserves the native C++ pointer ABI."""
import ast
from functools import lru_cache
from pathlib import Path

import torch
from .native_equivariant_norm import _partials_source

_ROLES={'_forward':0,'_backward':1,'_double':2}
_ENTRIES=('fasteq_modulation_forward','fasteq_modulation_backward','fasteq_modulation_double')


@lru_cache(maxsize=256)
def _source(metadata,dtype):
    role,fields=ast.literal_eval(metadata);kw=dict(fields);scalar='double' if dtype==torch.float64 else 'float'
    if role==0:names=['x','g','residual','index','output'];outputs={'output'}
    elif role==1:names=['x','g','dy','index','dx','dg'];outputs={'dx','dg'}
    else:names=['x','g','dy','u','v','index','dx','dg','ddy'];outputs={'dx','dg','ddy'}
    signature=','.join(('const int* ' if name=='index' else ('' if name in outputs else 'const ')+scalar+'* ')+name for name in names)
    partials=''
    prototype=''
    if kw.get('NEED_G',False):
        # Reuse NativeNorm's FastEq per-copy angular transpose emitter; the
        # modulation channel offsets become a single packed parameter slot.
        normalized={'D':kw['D'],'WIDTH':kw['C'],'IR_MUL':False,'GGRADFLAGS':(True,),
                    'BGRADFLAGS':(),'GOFFSETS':(0,),
                    'PARTS':tuple((start,channels,dimension,0,gate_base,-1,0)
                                  for start,channels,dimension,gate_base in kw['PARTS'])}
        partials=_partials_source(scalar,1,normalized).replace('fasteq_norm_parameter_partials','fasteq_modulation_parameter_partials').replace('norm_add(', 'mod_add(')
        prototype=f'__device__ __forceinline__ void fasteq_modulation_parameter_partials(const {scalar}*,const {scalar}*,{scalar}*,long long);\n'
    defines={'FASTEQ_MOD_TYPE':scalar,'FASTEQ_MOD_FP64':int(dtype==torch.float64),'FASTEQ_MOD_ROLE':role,
        'FASTEQ_MOD_DIM':kw['D'],'FASTEQ_MOD_CHANNELS':kw['C'],'FASTEQ_MOD_ENTRY':_ENTRIES[role],
        'FASTEQ_MOD_SIGNATURE':signature,'FASTEQ_MOD_NEED_X':int(kw.get('NEED_X',False)),
        'FASTEQ_MOD_NEED_G':int(kw.get('NEED_G',False)),'FASTEQ_MOD_NEED_DY':int(kw.get('NEED_DY',False)),
        'FASTEQ_MOD_HAS_U':int(kw.get('HAS_U',False)),'FASTEQ_MOD_HAS_V':int(kw.get('HAS_V',False))}
    prefix=''.join(f'#define {name} {value}\n' for name,value in defines.items())
    return prefix+prototype+(Path(__file__).parents[1]/'cuda/src/native_modulation.cuh').read_text()+partials


def source_for(role,kwargs,dtype):
    return _source(repr((role,tuple(sorted(kwargs.items())))),dtype)


def prepare(fn,arguments,x,kwargs,cache=None):
    """Return the unchanged (resident Kernel, threads, dynamic shared) tuple."""
    if not x.numel():return None
    name=getattr(fn,'__name__',None) or getattr(getattr(fn,'fn',None),'__name__',None)
    if name not in _ROLES:raise ValueError('Unknown native modulation entry: '+str(name))
    role=_ROLES[name];source=source_for(role,kwargs,x.dtype)
    from fasteq.cuda.jit import kernels
    kernel=kernels([source],x.device,entry=_ENTRIES[role])[source]
    prepared=(kernel,128,0)
    if cache is not None:cache[(role,x.device,x.dtype,source)]=prepared
    return prepared


def launch(fn,arguments,x,kwargs,cache=None):
    prepared=prepare(fn,arguments,x,kwargs,cache)
    if prepared is None:return
    kernel,threads,shared=prepared
    kernel.launch([value.data_ptr() for value in arguments],x.shape[0],1,threads,
                  torch.cuda.current_stream(x.device).cuda_stream,shared)


__all__=['prepare','launch','source_for']
