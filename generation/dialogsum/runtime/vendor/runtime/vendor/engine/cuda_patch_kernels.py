"""Lazily imported CUDA decoder; validated against CPU references at admission."""
import triton
import triton.language as tl

@triton.jit
def codes_kernel(W,P,N,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<N
    packed=tl.load(P+i//4,mask=valid,other=0).to(tl.int32)
    code=(packed>>((i%4)*2))&3
    raw=tl.load(W+i,mask=valid,other=0).to(tl.int32)&65535
    fixed=raw+tl.where(code==2,1,tl.where(code==1,-1,0))
    tl.store(W+i,fixed.to(tl.int16),mask=valid)

@triton.jit
def scatter_kernel(W,I,V,N,BLOCK:tl.constexpr):
    j=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=j<N
    pos=tl.load(I+j,mask=valid,other=0)
    value=tl.load(V+j,mask=valid,other=0)
    tl.store(W+pos,value,mask=valid)

@triton.jit
def grouped_codes(WP,PP,VP,MODES,NS,JOB_P,JOB_START,BLOCK:tl.constexpr):
    job=tl.program_id(0)
    p=tl.load(JOB_P+job)
    start=tl.load(JOB_START+job)
    i=start+tl.arange(0,BLOCK)
    n=tl.load(NS+p)
    valid=i<n
    w=tl.load(WP+p).to(tl.pointer_type(tl.int16))
    mode=tl.load(MODES+p)
    if mode==2:
        packed_ptr=tl.load(PP+p).to(tl.pointer_type(tl.uint8))
        packed=tl.load(packed_ptr+i//4,mask=valid,other=0).to(tl.int32)
        code=(packed>>((i%4)*2))&3
        raw=tl.load(w+i,mask=valid,other=0).to(tl.int32)&65535
        corrected_bits=raw+tl.where(code==2,1,tl.where(code==1,-1,0))
        tl.store(w+i,corrected_bits.to(tl.int16),mask=valid)
    else:
        v=tl.load(VP+p).to(tl.pointer_type(tl.int16))
        reference_bits=tl.load(v+i,mask=valid,other=0).to(tl.int32)
        tl.store(w+i,reference_bits.to(tl.int16),mask=valid)

@triton.jit
def grouped_scatter(WP,IP,VP,ES,JOB_P,JOB_START,BLOCK:tl.constexpr):
    job=tl.program_id(0)
    p=tl.load(JOB_P+job)
    j=tl.load(JOB_START+job)+tl.arange(0,BLOCK)
    count=tl.load(ES+p)
    valid=j<count
    w=tl.load(WP+p).to(tl.pointer_type(tl.int16))
    ix=tl.load(IP+p).to(tl.pointer_type(tl.int32))
    val=tl.load(VP+p).to(tl.pointer_type(tl.int16))
    pos=tl.load(ix+j,mask=valid,other=0)
    value=tl.load(val+j,mask=valid,other=0)
    tl.store(w+pos,value,mask=valid)
