#pragma once
// FastEq Gate fan-in transpose and Norm CTA statistics are fused in one CTA
// per row. Two shared vectors cache the first/second Norm adjoint for input
// fan-in; large rows rematerialize those values from the same scalar moments.
struct GateNormValue { NormT value, tangent; };
struct GateNormAdjoint { NormT first, second, seed, parameter; };

__device__ __forceinline__ GateNormValue gate_norm_value(
    const NormT* projected,const NormT* skip,const NormT* u,
    const int* src,const int* gate,const signed char* act,const NormT* scale,
    NormIndex row,int output) {
    const int source=src[output],g=gate[output];
    const NormT xs=norm_add(projected[row*FASTEQ_GN_IN+source],skip[row*FASTEQ_GN_IN+source]);
    const NormT z=g<0?xs:norm_add(projected[row*FASTEQ_GN_IN+g],skip[row*FASTEQ_GN_IN+g]);
    const NormT activation=gate_activation(z,act[output],scale[output]);
    GateNormValue result={g<0?activation:norm_mul(xs,activation),NormT(0)};
    if(FASTEQ_GN_ROLE==2 && FASTEQ_GN_HAS_U) {
        const NormT us=u[row*FASTEQ_GN_IN+source];
        const NormT d1=gate_activation_d1(z,act[output],scale[output]);
        result.tangent=g<0?norm_mul(d1,us):norm_add(norm_mul(us,activation),
            norm_mul(norm_mul(xs,d1),u[row*FASTEQ_GN_IN+g]));
    }
    return result;
}

__device__ __forceinline__ GateNormAdjoint gate_norm_adjoint(
    GateNormValue v,NormT weight,NormT seed,NormT gamma,NormT h,
    NormT r,NormT a,NormT s,NormT c) {
    const NormT r3=norm_mul(norm_mul(r,r),r),g=norm_mul(seed,gamma);
    GateNormAdjoint result;
    result.first=norm_add(norm_mul(r,g),-norm_mul(norm_mul(norm_mul(r3,weight),v.value),a));
    result.parameter=norm_mul(norm_mul(seed,v.value),r);
    result.second=result.seed=NormT(0);
    if(FASTEQ_GN_ROLE==2) {
        const NormT t0=norm_mul(norm_mul(r,seed),h);
        const NormT t1=norm_mul(norm_mul(norm_mul(weight,v.value),r3),c);
        const NormT t2=norm_mul(norm_mul(norm_mul(norm_mul(norm_mul(norm_mul(norm_mul(
            NormT(3),weight),v.value),r3),r),r),a),s);
        const NormT t3=norm_mul(r3,norm_add(norm_mul(s,g),norm_mul(norm_mul(a,weight),v.tangent)));
        result.second=norm_add(norm_add(norm_add(t0,-t1),t2),-t3);
        result.seed=norm_add(norm_mul(r,norm_add(norm_mul(v.tangent,gamma),norm_mul(h,v.value))),
            -norm_mul(norm_mul(norm_mul(r3,s),v.value),gamma));
        result.parameter=norm_mul(seed,norm_add(norm_mul(r,v.tangent),-norm_mul(norm_mul(r3,s),v.value)));
    }
    return result;
}

extern "C" __global__ void FASTEQ_GN_ENTRY(FASTEQ_GN_SIGNATURE) {
    const NormIndex row=blockIdx.x;
    __shared__ NormT statistics_workspace[4*NormWarps];
#if FASTEQ_GN_ROLE > 0 && FASTEQ_GN_CACHE_TRANSPOSE
    __shared__ NormT transpose_first[NormD],transpose_second[NormD];
#endif
#if FASTEQ_GN_ROLE > 0 && FASTEQ_GN_GPARTIAL && !FASTEQ_GN_CACHE_TRANSPOSE
    // NativeNorm's one-vector angular transpose fits ordinary/wide model rows.
    // Larger rows use the channel-centric rematerialized path below.
#if FASTEQ_NORM_DIM * (4 + 4 * FASTEQ_NORM_FP64) <= 48000
    __shared__ NormT parameter_values[NormD];
#define FASTEQ_GN_PARAMETER_TILE 1
#else
#define FASTEQ_GN_PARAMETER_TILE 0
#endif
#endif
    GateNormValue values[NormItems];
    NormT variance[NormItems],avalues[NormItems],svalues[NormItems],cvalues[NormItems];
    #pragma unroll
    for(int item=0;item<NormItems;++item) {
        const int i=threadIdx.x+item*NormThreads;
        values[item]={NormT(0),NormT(0)};variance[item]=avalues[item]=svalues[item]=cvalues[item]=NormT(0);
        if(i<NormD) {
#if FASTEQ_GN_ROLE == 2
            values[item]=gate_norm_value(projected,skip,u,src,gate,act,scale,row,i);
#else
            values[item]=gate_norm_value(projected,skip,nullptr,src,gate,act,scale,row,i);
#endif
            variance[item]=norm_mul(norm_mul(values[item].value,values[item].value),coefficient[i]);
#if FASTEQ_GN_ROLE > 0
            const NormT seed=dy[row*NormD+i],g=norm_mul(seed,FASTEQ_GN_GAMMA(i));
            avalues[item]=norm_mul(g,values[item].value);
#if FASTEQ_GN_ROLE == 2
            svalues[item]=norm_mul(norm_mul(values[item].tangent,coefficient[i]),values[item].value);
            cvalues[item]=norm_add(norm_mul(values[item].tangent,g),norm_mul(norm_mul(seed,FASTEQ_GN_H(i)),values[item].value));
#endif
#endif
        }
    }
#if FASTEQ_GN_ROLE == 0
    NormT statistics[1]={norm_local_tree(variance)};
#elif FASTEQ_GN_ROLE == 1
    NormT statistics[2]={norm_local_tree(variance),norm_local_tree(avalues)};
#else
    NormT statistics[4]={norm_local_tree(variance),norm_local_tree(avalues),norm_local_tree(svalues),norm_local_tree(cvalues)};
#endif
    norm_block_statistics(statistics,statistics_workspace);
    const NormT r=rsqrt(norm_add(statistics[0],NormT(FASTEQ_GN_EPS)));
#if FASTEQ_GN_ROLE > 0
    const NormT a=statistics[1];
#if FASTEQ_GN_ROLE == 2
    const NormT s=statistics[2],c=statistics[3];
#else
    const NormT s=0,c=0;
#endif
#endif
    #pragma unroll
    for(int item=0;item<NormItems;++item) {
        const int i=threadIdx.x+item*NormThreads;
        if(i<NormD) {
#if FASTEQ_GN_ROLE == 0
            y[row*NormD+i]=norm_mul(norm_mul(values[item].value,r),FASTEQ_GN_GAMMA(i));
#else
            const GateNormAdjoint adjoint=gate_norm_adjoint(values[item],coefficient[i],dy[row*NormD+i],
                FASTEQ_GN_GAMMA(i),FASTEQ_GN_H(i),r,a,s,c);
#if FASTEQ_GN_CACHE_TRANSPOSE
            transpose_first[i]=FASTEQ_GN_ROLE==2?adjoint.second:adjoint.first;
            transpose_second[i]=adjoint.first;
#elif FASTEQ_GN_GPARTIAL && FASTEQ_GN_PARAMETER_TILE
            parameter_values[i]=adjoint.parameter;
#endif
#if FASTEQ_GN_ROLE == 2
            if(FASTEQ_GN_DDYFLAG)ddy[row*NormD+i]=adjoint.seed;
#endif
#endif
        }
    }
#if FASTEQ_GN_ROLE > 0
    __syncthreads();
    if(FASTEQ_GN_DZFLAG) {
        for(int input=threadIdx.x;input<FASTEQ_GN_IN;input+=NormThreads) {
            NormT total=0;
            #pragma unroll
            for(int fan=0;fan<FASTEQ_GN_FAN;++fan) {
                const int output=in_out[input*FASTEQ_GN_FAN+fan];
                if(output<0)continue;
                const int role=in_role[input*FASTEQ_GN_FAN+fan],source=src[output],g=gate[output];
                const NormT xs=norm_add(projected[row*FASTEQ_GN_IN+source],skip[row*FASTEQ_GN_IN+source]);
                const NormT z=g<0?xs:norm_add(projected[row*FASTEQ_GN_IN+g],skip[row*FASTEQ_GN_IN+g]);
                const NormT d1=gate_activation_d1(z,act[output],scale[output]);
                const NormT jacobian=role==0?d1:role==1?gate_activation(z,act[output],scale[output]):norm_mul(xs,d1);
#if FASTEQ_GN_CACHE_TRANSPOSE
                const NormT direct=transpose_first[output],first=transpose_second[output];
#else
#if FASTEQ_GN_ROLE == 2
                const GateNormValue value=gate_norm_value(projected,skip,u,src,gate,act,scale,row,output);
#else
                const GateNormValue value=gate_norm_value(projected,skip,nullptr,src,gate,act,scale,row,output);
#endif
                const GateNormAdjoint adjoint=gate_norm_adjoint(value,coefficient[output],dy[row*NormD+output],
                    FASTEQ_GN_GAMMA(output),FASTEQ_GN_H(output),r,a,s,c);
                const NormT direct=FASTEQ_GN_ROLE==2?adjoint.second:adjoint.first,first=adjoint.first;
#endif
                NormT contribution=norm_mul(direct,jacobian);
#if FASTEQ_GN_ROLE == 2
                if(FASTEQ_GN_HAS_U) {
                    const NormT us=u[row*FASTEQ_GN_IN+source],ug=g<0?NormT(0):u[row*FASTEQ_GN_IN+g];
                    const NormT d2=gate_activation_d2(z,act[output],scale[output]);
                    const NormT curvature=role==0?norm_mul(d2,us):role==1?norm_mul(d1,ug):
                        norm_add(norm_mul(d1,us),norm_mul(norm_mul(xs,d2),ug));
                    contribution=norm_add(contribution,norm_mul(first,curvature));
                }
#endif
                total=norm_add(total,contribution);
            }
            dz[row*FASTEQ_GN_IN+input]=total;
        }
    }
    __syncthreads();
#if FASTEQ_GN_GPARTIAL
#if FASTEQ_GN_CACHE_TRANSPOSE
    for(int item=0;item<NormItems;++item) {
        const int i=threadIdx.x+item*NormThreads;
        if(i<NormD)transpose_first[i]=gate_norm_adjoint(values[item],coefficient[i],dy[row*NormD+i],
            FASTEQ_GN_GAMMA(i),FASTEQ_GN_H(i),r,a,s,c).parameter;
    }
    __syncthreads();fasteq_norm_parameter_partials(transpose_first,dy,partials,row);
#elif FASTEQ_GN_PARAMETER_TILE
    fasteq_norm_parameter_partials(parameter_values,dy,partials,row);
#else
    FASTEQ_GN_REMATERIALIZED_PARTIALS
#endif
#endif
#endif
}
