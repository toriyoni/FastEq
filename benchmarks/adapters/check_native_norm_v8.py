"""CPU preparation and opt-in targeted GPU QA for native FastEq Norm.

Default --prepare-only uses NVRTC without a CUDA context and checks the
existing FastEq analytic transpose against independent CPU Torch autograd.
GPU mode is intentionally explicit; it binds the two integration hooks in a
test context and leaves the production equivariant_norm.py unchanged.
"""
import argparse
import ctypes
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType

sys.dont_write_bytecode = True
import torch

ROOT = Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
CANDIDATE = ROOT / 'src/FastEq_v8'


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def prepare_configs():
    return [
        dict(D=16, PARTS=((0,4,1,0,0,0,0),(4,4,3,1,0,-1,0)),
             CENTERS=((0,4),), IR_MUL=False, GFLAGS=(True,True), BFLAGS=(True,), EPS=1e-5, num_warps=4),
        dict(D=896, PARTS=((0,64,1,0,0,0,0),(64,64,3,0,64,-1,0),(256,64,5,0,128,-1,0),(576,64,5,0,192,-1,0)),
             CENTERS=((0,64),), IR_MUL=True, GFLAGS=(True,), BFLAGS=(True,), EPS=1e-5, num_warps=4),
        dict(D=36, PARTS=((0,0,1,0,0,-1,0),(0,3,1,1,0,0,0),(3,3,3,2,0,-1,0),(12,0,5,3,0,-1,0),(12,8,3,4,0,-1,0)),
             CENTERS=((0,3),), IR_MUL=False, GFLAGS=(True,True,True,True,True), BFLAGS=(True,), EPS=1e-7, num_warps=4),
        dict(D=2240, PARTS=((0,64,1,-1,0,-1,0),(64,64,3,-1,0,-1,0),(256,64,31,-1,0,-1,0)),
             CENTERS=(), IR_MUL=True, GFLAGS=(), BFLAGS=(), EPS=1e-5, num_warps=8),
    ]


def role_kwargs(config, role):
    kw = dict(config)
    if role == 0: return kw
    widths = [0] * len(kw['GFLAGS'])
    beta_widths = [0] * len(kw['BFLAGS'])
    for _, channels, _, gamma_slot, gamma_base, beta_slot, beta_base in kw['PARTS']:
        if gamma_slot >= 0: widths[gamma_slot] = max(widths[gamma_slot], gamma_base + channels)
        if beta_slot >= 0: beta_widths[beta_slot] = max(beta_widths[beta_slot], beta_base + channels)
    gamma_offsets = tuple(sum(widths[:i]) for i in range(len(widths)))
    kw.pop('BFLAGS')
    if role == 1:
        kw.update(GOFFSETS=gamma_offsets, BOFFSETS=tuple(sum(widths) + sum(beta_widths[:i]) for i in range(len(beta_widths))),
                  WIDTH=sum(widths + beta_widths), DXFLAG=True,
                  GGRADFLAGS=(True,)*len(widths), BGRADFLAGS=(True,)*len(beta_widths))
    else:
        kw.update(HFLAGS=tuple(i%2 == 0 for i in range(len(widths))), KFLAGS=(True,)*len(beta_widths),
                  HAS_U=True, GOFFSETS=gamma_offsets, WIDTH=sum(widths), DDXFLAG=True, DDYFLAG=True,
                  GGRADFLAGS=(True,)*len(widths))
    return kw


def nvrtc_compile(source):
    """Compile-only: NVRTC does not create a CUDA device context."""
    library = ctypes.CDLL('/usr/local/cuda-12.8/lib64/libnvrtc.so')
    pointer = ctypes.c_void_p
    library.nvrtcCreateProgram.argtypes = [ctypes.POINTER(pointer), ctypes.c_char_p, ctypes.c_char_p,
                                         ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
    library.nvrtcCompileProgram.argtypes = [pointer, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    library.nvrtcGetProgramLogSize.argtypes = [pointer, ctypes.POINTER(ctypes.c_size_t)]
    library.nvrtcGetProgramLog.argtypes = [pointer, ctypes.c_char_p]
    library.nvrtcGetCUBINSize.argtypes = [pointer, ctypes.POINTER(ctypes.c_size_t)]
    library.nvrtcGetCUBIN.argtypes = [pointer, ctypes.c_void_p]
    library.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(pointer)]
    program = pointer()
    assert library.nvrtcCreateProgram(ctypes.byref(program), source.encode(), b'native_norm.cu', 0, None, None) == 0
    options = (ctypes.c_char_p*3)(b'--std=c++17', b'--gpu-architecture=sm_90', b'--fmad=true')
    try:
        status = library.nvrtcCompileProgram(program, 3, options)
        size = ctypes.c_size_t(); library.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value); library.nvrtcGetProgramLog(program, log)
        if status: raise RuntimeError(log.value.decode())
        library.nvrtcGetCUBINSize(program, ctypes.byref(size))
        binary = ctypes.create_string_buffer(size.value); library.nvrtcGetCUBIN(program, binary)
        return {'cubin_bytes': size.value, 'cubin_sha256': hashlib.sha256(binary.raw).hexdigest(),
                'compiler_log': log.value.decode()}
    finally:
        library.nvrtcDestroyProgram(ctypes.byref(program))


def cpu_formula_check():
    """Check native formulas against an independent nonlinear Torch Hessian VJP."""
    records = []
    torch.manual_seed(8123)
    for centered in (False, True):
        for seeds in ('all', 'u_only', 'parameter_only', 'bias_only'):
            x = (torch.randn(5, 16, dtype=torch.float64)*.2).requires_grad_()
            dy = (torch.randn_like(x)*.1).requires_grad_()
            gamma = (torch.randn(16,dtype=torch.float64)*.1+1).requires_grad_()
            beta = (torch.randn(16,dtype=torch.float64)*.1).requires_grad_()
            coefficient = torch.linspace(.01,.02,16,dtype=torch.float64)
            # Within the centered scalar slice the source coefficient is uniform.
            coefficient[:4] = .015
            def project(value):
                return torch.cat((value[:,:4]-value[:,:4].mean(1,keepdim=True),value[:,4:]),1) if centered else value
            v = project(x)
            r = ((v.square()*coefficient).sum(1,keepdim=True)+1e-5).rsqrt()
            output = v*r*gamma+beta
            first = torch.autograd.grad((output*dy).sum(), (x,gamma,beta), create_graph=True)
            u = torch.randn_like(x)*.2 if seeds in ('all','u_only') else torch.zeros_like(x)
            h = torch.randn_like(gamma)*.2 if seeds in ('all','parameter_only') else torch.zeros_like(gamma)
            k = torch.randn_like(beta)*.2 if seeds in ('all','parameter_only','bias_only') else torch.zeros_like(beta)
            oracle = torch.autograd.grad((first[0]*u).sum()+(first[1]*h).sum()+(first[2]*k).sum(),
                                         (x,dy,gamma), allow_unused=True)
            tangent = project(u); g = dy*gamma
            a = (g*v).sum(1,keepdim=True); s = (tangent*coefficient*v).sum(1,keepdim=True)
            c = (tangent*g+dy*h*v).sum(1,keepdim=True); r3=r*r*r
            ddx = project(r*dy*h-coefficient*v*r3*c+3*coefficient*v*r3*r*r*a*s-r3*(s*g+a*coefficient*tangent))
            ddy = r*(tangent*gamma+h*v)-r3*s*v*gamma+k
            dgamma = (dy*(r*tangent-r3*s*v)).sum(0)
            error=[]
            for actual,expected in zip((ddx,ddy,dgamma),oracle):
                if expected is None: assert not torch.count_nonzero(actual);error.append(0);continue
                torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-11)
                error.append((actual-expected).abs().max().item())
            records.append({'centering':centered,'seeds':seeds,'status':'pass','full_max_abs':error})
    return records


@contextmanager
def native_hooks(plans):
    """Test-only binding; production old source remains untouched until integration."""
    from fasteq.ops import equivariant_norm as old
    from fasteq.ops import native_equivariant_norm as native
    previous = old._parameter_gradients
    state = [(plan, plan.launch, getattr(plan,'backend',None)) for plan in plans]
    old._parameter_gradients = native.parameter_gradients
    for plan in plans:
        plan.launch = MethodType(native.launch,plan)
        plan.backend = 'FastEq native CUDA Norm preparation binding'
    try: yield
    finally:
        old._parameter_gradients = previous
        for plan,launcher,backend in state:
            plan.launch = launcher
            if backend is not None:plan.backend=backend


def prepare(output):
    from fasteq.ops.native_equivariant_norm import source_for
    assert not torch.cuda.is_initialized()
    out = Path(output).parent / 'prepared_sources';out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','scope':'CPU mathematical VJP and NVRTC compile only; not GPU precision/performance',
            'cuda_initialized_before':torch.cuda.is_initialized(),'formulas':cpu_formula_check(),'specializations':[]}
    try:
        for dtype in (torch.float32,torch.float64):
            for case,config in enumerate(prepare_configs()):
                for role in range(3):
                    source=source_for(role,role_kwargs(config,role),dtype)
                    name=f'case{case}_{str(dtype).split(".")[-1]}_role{role}.cu';path=out/name;path.write_text(source)
                    result=nvrtc_compile(source)
                    report['specializations'].append({'case':case,'dtype':str(dtype),'role':role,
                        'source_path':str(path),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**result})
            source=source_for(3,{},dtype);path=out/f'reduce_{str(dtype).split(".")[-1]}.cu';path.write_text(source)
            report['specializations'].append({'case':'reduce','dtype':str(dtype),'role':3,
                'source_path':str(path),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**nvrtc_compile(source)})
        assert not torch.cuda.is_initialized()
        report['cuda_initialized_after']=False;report['status']='pass'
    except BaseException as error:
        report['status']='failed';report['error']=repr(error);raise
    finally:write_json(output,report)
    print(json.dumps({'status':report['status'],'CPU_formula_fixtures':len(report['formulas']),
                      'NVRTC_specializations':len(report['specializations']),'cuda_initialized':False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.prepare_only:raise SystemExit('GPU execution is not implemented/enabled in this CPU preparation stage')
    prepare(args.output)


if __name__=='__main__':main()
