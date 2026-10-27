"""Check the hand-written high-accuracy TF32 extension independently of plans."""
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch

ROOT=Path('/public-data/zhouxibo/zxb/tace_prophet_fasteq')
sys.path.insert(0,str(ROOT/'performance/fasteq_optimization/v7/driver'))
from pair_common import ExclusiveGuard
from fasteq.cuda.jit import kernels


def main():
    guard=ExclusiveGuard('GPU-e7d1b5bc-5939-b204-406e-a57dd38b7341')
    guard.check()
    torch.empty(0,device='cuda')
    torch.backends.cuda.matmul.allow_tf32=False
    header=ROOT/'src/FastEq_v8/fasteq/cuda/src/equi_linear_kernels/matrix_tf32.cuh'
    ptx=(ROOT/'src/FastEq_v8/fasteq/cuda/src/equi_linear_kernels/ptx_inst.cuh').read_text()
    rounding=next(line for line in ptx.splitlines() if line.startswith('#define asm_cvt_tf32_f32'))
    prefix='using int64_t=long long; struct MatrixShape {int64_t m,n,k,ar,ac,ab,br,bc,bb,cr,cc,cb;};\n'+rounding+'\n'
    records=[]
    for transa,transb in ((False,False),(False,True),(True,False),(True,True)):
        source=prefix+header.read_text()+f'''
        extern "C" __global__ void run(const float* a,const float* b,float* c,
          long long m,long long n,long long k,long long ar,long long ac,
          long long br,long long bc,long long cr,long long cc,float alpha,float beta) {{
          MatrixShape s{{m,n,k,ar,ac,0,br,bc,0,cr,cc,0}};
          matrix_tf32_body<64,64,32,{str(not transa).lower()},{str(not transb).lower()}>(
            a,b,c,s,alpha,beta,(m+63)/64,0);
        }}'''
        kernel=kernels([source],'cuda',entry='run')[source]
        for m,n,k in ((7,9,13),(63,65,127),(64,64,64),(33,35,8),(128,64,1024),(17,9,0),(512,64,4096)):
            a=torch.randn((k,m) if transa else (m,k),device='cuda',dtype=torch.float32)*.2
            b=torch.randn((n,k) if transb else (k,n),device='cuda',dtype=torch.float32)*.2
            if transa:a=a.t()
            if transb:b=b.t()
            c=torch.randn(m,n,device='cuda')*.2
            ref=.7*(a@b)-.3*c
            args=[a.data_ptr(),b.data_ptr(),c.data_ptr(),m,n,k,*a.stride(),*b.stride(),*c.stride(),
                  struct.unpack('I',struct.pack('f',.7))[0],struct.unpack('I',struct.pack('f',-.3))[0]]
            kernel.launch(args,(n+63)//64,(m+63)//64,128,torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            delta=(ref-c).double();relative=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.double()).clamp_min(1e-30))
            torch.testing.assert_close(c,ref,atol=3e-5,rtol=5e-5)
            guard.check()
            records.append({'m':m,'n':n,'k':k,'transposed':[transa,transb],
                            'relative_l2':relative,'max_absolute':float(delta.abs().max()),'passed':True})
    out=ROOT/'performance/fasteq_optimization/v8/native_matrix/tf32x3_isolated_qa.json'
    out.write_text(json.dumps({'kind':'Isolated hand CUDA precision check, not normal performance',
                              'header_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),
                              'records':records,'guard':guard.check()},indent=2)+'\n')
    print('TF32x3 isolated 28 transpose/tail/large-K checks passed',flush=True)


if __name__=='__main__':main()
