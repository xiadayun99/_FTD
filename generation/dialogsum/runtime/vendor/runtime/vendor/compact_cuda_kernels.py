"""Same raw-bit decoder, with per-projection cumulative tile offsets."""
import triton
import triton.language as tl


@triton.jit
def _projection(OFFSETS,job,N_PROJ:tl.constexpr,SEARCH_BITS:tl.constexpr):
    p=tl.full((),0,tl.int32)
    for i in tl.static_range(SEARCH_BITS):
        candidate=p+(1<<(SEARCH_BITS-1-i))
        end=tl.load(OFFSETS+candidate,mask=candidate<=N_PROJ,other=2147483647)
        p=tl.where((candidate<=N_PROJ)&(end<=job),candidate,p)
    return p


@triton.jit
def compact_codes(WP,PP,VP,MODES,NS,OFFSETS,N_PROJ:tl.constexpr,SEARCH_BITS:tl.constexpr,BLOCK:tl.constexpr):
    job=tl.program_id(0)
    p=_projection(OFFSETS,job,N_PROJ,SEARCH_BITS)
    start=(job-tl.load(OFFSETS+p))*BLOCK
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
def compact_scatter(WP,IP,VP,ES,OFFSETS,N_PROJ:tl.constexpr,SEARCH_BITS:tl.constexpr,BLOCK:tl.constexpr):
    job=tl.program_id(0)
    p=_projection(OFFSETS,job,N_PROJ,SEARCH_BITS)
    j=(job-tl.load(OFFSETS+p))*BLOCK+tl.arange(0,BLOCK)
    count=tl.load(ES+p)
    valid=j<count
    w=tl.load(WP+p).to(tl.pointer_type(tl.int16))
    ix=tl.load(IP+p).to(tl.pointer_type(tl.int32))
    val=tl.load(VP+p).to(tl.pointer_type(tl.int16))
    pos=tl.load(ix+j,mask=valid,other=0)
    value=tl.load(val+j,mask=valid,other=0)
    tl.store(w+pos,value,mask=valid)
