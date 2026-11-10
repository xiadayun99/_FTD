"""New target-residue decoder; the legacy scatter kernel remains unchanged."""
import triton
import triton.language as tl
from compact_cuda_kernels import _projection


@triton.jit
def target_codes(WP,PP,VP,MODES,NS,OFFSETS,N_PROJ:tl.constexpr,SEARCH_BITS:tl.constexpr,BLOCK:tl.constexpr):
    job=tl.program_id(0)
    p=_projection(OFFSETS,job,N_PROJ,SEARCH_BITS)
    start=(job-tl.load(OFFSETS+p))*BLOCK
    i=start+tl.arange(0,BLOCK);n=tl.load(NS+p);valid=i<n
    w=tl.load(WP+p).to(tl.pointer_type(tl.int16));mode=tl.load(MODES+p)
    if mode==2:
        packed_ptr=tl.load(PP+p).to(tl.pointer_type(tl.uint8))
        packed=tl.load(packed_ptr+i//4,mask=valid,other=0).to(tl.int32)
        residue=(packed>>((i%4)*2))&3
        raw=tl.load(w+i,mask=valid,other=0).to(tl.int32)&65535
        difference=(residue-(raw&3))&3
        step=tl.where(difference==1,1,tl.where(difference==3,-1,0))
        corrected=(raw+step)&65535
        tl.store(w+i,corrected.to(tl.int16),mask=valid)
    else:
        v=tl.load(VP+p).to(tl.pointer_type(tl.int16))
        reference_bits=tl.load(v+i,mask=valid,other=0).to(tl.int32)
        tl.store(w+i,reference_bits.to(tl.int16),mask=valid)
