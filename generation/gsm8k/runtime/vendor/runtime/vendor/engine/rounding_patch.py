"""Exact 16-bit correction after deterministic low-rank arithmetic.

This is a prediction-residual codec, not quantization of model weights.
The decoder is exact for the predictor input recorded at admission.
It is NOT a repair for arbitrary weights or changed GEMM/backend contracts.
"""
from dataclasses import dataclass
import hashlib
import math
import numpy as np
import torch

VERSION="ftd_canonical_patch_v1"

def bits(tensor):
    if tensor.dtype not in (torch.float16,torch.bfloat16):
        raise TypeError("Only plain 16-bit float tensors are supported")
    if not tensor.is_contiguous():
        raise ValueError("Contiguous tensor required")
    return tensor.detach().cpu().view(torch.int16).numpy().reshape(-1).copy().view(np.uint16)

def digest(tensor):
    h=hashlib.sha256()
    h.update(str(tuple(tensor.shape)).encode())
    h.update(str(tensor.dtype).encode())
    h.update(bits(tensor).tobytes())
    return h.hexdigest()

def bit_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(bits(a),bits(b))

@dataclass
class Patch:
    shape: tuple
    dtype: str
    mode: str
    packed: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    predictor_sha256: str
    target_sha256: str
    stats: dict

    @property
    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in [self.packed,self.indices,self.values])

    def to(self,device):
        return Patch(self.shape,self.dtype,self.mode,
                     self.packed.to(device),self.indices.to(device),self.values.to(device),
                     self.predictor_sha256,self.target_sha256,dict(self.stats))

    def metadata(self):
        return dict(shape=list(self.shape),dtype=self.dtype,mode=self.mode,bytes=self.nbytes,
                    predictor_sha256=self.predictor_sha256,target_sha256=self.target_sha256,**self.stats)

    def apply_(self,tensor,verify_input=False):
        if tuple(tensor.shape)!=self.shape or str(tensor.dtype)!=self.dtype or not tensor.is_contiguous():
            raise ValueError("Patch shape/dtype/layout differs from admitted operator")
        if tensor.device!=self.packed.device:
            raise ValueError("Patch and parameter must be on the same device")
        if verify_input and digest(tensor)!=self.predictor_sha256:
            raise ValueError("Predictor input differs; refusing to relabel a dirty state as canonical")
        with torch.no_grad():
            if tensor.device.type=="cuda":
                _apply_cuda(self,tensor)
            elif tensor.device.type=="cpu":
                _apply_cpu(self,tensor)
            else:
                raise NotImplementedError("Only CPU and CUDA decoders are implemented")
        return tensor

def encode(predictor,target):
    """CPU-side admission encoding. Signed-zero and escape bits are preserved."""
    if predictor.shape!=target.shape or predictor.dtype!=target.dtype:
        raise ValueError("Predictor/reference identities differ")
    a=bits(predictor);b=bits(target);n=a.size
    if n>np.iinfo(np.int32).max:
        raise ValueError("Projection exceeds this codec's signed int32 index contract")
    delta=b.astype(np.int32)-a.astype(np.int32)
    changed=np.flatnonzero(delta).astype(np.int32)
    escapes=np.flatnonzero((delta < -1)|(delta > 1)).astype(np.int32)
    packed_cost=(n+3)//4+6*len(escapes)
    sparse_cost=6*len(changed)
    dense_cost=2*n
    mode=min([("zero",0)] if not len(changed) else
             [("sparse",sparse_cost),("two_bit",packed_cost),("dense",dense_cost)],
             key=lambda p:(p[1],p[0]))[0]
    packed=np.empty(0,dtype=np.uint8)
    indices=np.empty(0,dtype=np.int32)
    values=np.empty(0,dtype=np.int16)
    if mode=="sparse":
        indices=changed
        values=b[indices].copy().view(np.int16)
    elif mode=="dense":
        values=b.copy().view(np.int16)
    elif mode=="two_bit":
        # 00: unchanged; 01: raw 16-bit code - 1; 10: code + 1;
        # 11: unchanged here, then exact raw target bits from the escape list.
        code=np.zeros(n,dtype=np.uint8)
        code[delta==-1]=1
        code[delta==1]=2
        code[escapes]=3
        if n%4:
            code=np.pad(code,(0,4-n%4))
        code=code.reshape(-1,4)
        packed=(code[:,0]|(code[:,1]<<2)|(code[:,2]<<4)|(code[:,3]<<6)).astype(np.uint8)
        indices=escapes
        values=b[indices].copy().view(np.int16)
    patch=Patch(tuple(target.shape),str(target.dtype),mode,
                torch.from_numpy(packed.copy()),torch.from_numpy(indices.copy()),
                torch.from_numpy(values.copy()),digest(predictor),digest(target),
                dict(elements=int(n),changed=int(len(changed)),escapes=int(len(escapes)),
                     changed_fraction=float(len(changed)/max(1,n)),dense_reference_bytes=int(2*n)))
    probe=predictor.detach().cpu().clone().contiguous()
    patch.apply_(probe,verify_input=True)
    if not bit_equal(probe,target):
        raise AssertionError("Lossless codec failed its full-tensor admission check")
    return patch

def _apply_cpu(patch,tensor):
    if patch.mode=="zero":
        return
    raw=tensor.view(torch.int16).numpy().reshape(-1)
    if patch.mode=="dense":
        raw[:]=patch.values.numpy()
    elif patch.mode=="sparse":
        raw[patch.indices.numpy()]=patch.values.numpy()
    elif patch.mode=="two_bit":
        n=raw.size
        i=np.arange(n,dtype=np.int64)
        code=(patch.packed.numpy()[i//4]>>((i%4)*2))&3
        u=raw.view(np.uint16).astype(np.int32)
        u += (code==2).astype(np.int32)-(code==1).astype(np.int32)
        raw[:]=(u&65535).astype(np.uint16).view(np.int16)
        raw[patch.indices.numpy()]=patch.values.numpy()
    else:
        raise ValueError("Unknown patch mode")

_kernels=None
def _cuda_kernels():
    global _kernels
    if _kernels is not None:
        return _kernels
    import cuda_patch_kernels as k
    _kernels=(k.triton,k.codes_kernel,k.scatter_kernel)
    return _kernels

def _apply_cuda(patch,tensor):
    if patch.mode=="zero":
        return
    target=tensor.view(torch.int16).view(-1)
    if patch.mode=="dense":
        target.copy_(patch.values)
        return
    triton,codes,scatter=_cuda_kernels()
    if patch.mode=="two_bit":
        n=target.numel()
        codes[(triton.cdiv(n,1024),)](target,patch.packed,n,BLOCK=1024)
    if patch.indices.numel():
        n=patch.indices.numel()
        scatter[(triton.cdiv(n,256),)](target,patch.indices,patch.values,n,BLOCK=256)

class PatchGroup:
    """Two batched decoder launches across all projections of one phase.

    Descriptor tensors are device-resident and included in nbytes. Parameter
    storage addresses must remain fixed for the lifetime of this group.
    """
    def __init__(self,parameters,patches):
        self.parameters=parameters
        self.patches=patches
        names=sorted(patches)
        modes={"zero":0,"sparse":1,"two_bit":2,"dense":3}
        vectors={k:[] for k in ["wp","pp","ip","vp","modes","ns","es"]}
        codes_p=[];codes_start=[];escape_p=[];escape_start=[]
        for i,n in enumerate(names):
            w=parameters[n];p=patches[n]
            if w.device.type!="cuda" or p.packed.device!=w.device:
                raise ValueError("Grouped decoder requires colocated CUDA parameters and patches")
            if tuple(w.shape)!=p.shape or str(w.dtype)!=p.dtype or not w.is_contiguous():
                raise ValueError("Grouped decoder parameter layout changed")
            for key,value in [("wp",w.data_ptr()),("pp",p.packed.data_ptr()),
                              ("ip",p.indices.data_ptr()),("vp",p.values.data_ptr()),
                              ("modes",modes[p.mode]),("ns",w.numel()),("es",p.indices.numel())]:
                vectors[key].append(value)
            if p.mode in ["two_bit","dense"]:
                starts=range(0,w.numel(),1024)
                codes_start.extend(starts);codes_p.extend([i]*len(starts))
            if p.indices.numel():
                starts=range(0,p.indices.numel(),256)
                escape_start.extend(starts);escape_p.extend([i]*len(starts))
        device=parameters[names[0]].device
        self.arrays={k:torch.tensor(v,dtype=torch.int64 if k.endswith("p") else torch.int32,device=device)
                     for k,v in vectors.items()}
        for k,v in [("code_job_p",codes_p),("code_job_start",codes_start),
                    ("escape_job_p",escape_p),("escape_job_start",escape_start)]:
            self.arrays[k]=torch.tensor(v,dtype=torch.int32,device=device)
        self.code_jobs=len(codes_p);self.escape_jobs=len(escape_p)

    @property
    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in self.arrays.values())

    def apply_(self):
        import cuda_patch_kernels as k
        a=self.arrays
        if self.code_jobs:
            k.grouped_codes[(self.code_jobs,)](a["wp"],a["pp"],a["vp"],a["modes"],a["ns"],
                a["code_job_p"],a["code_job_start"],BLOCK=1024)
        if self.escape_jobs:
            k.grouped_scatter[(self.escape_jobs,)](a["wp"],a["ip"],a["vp"],a["es"],
                a["escape_job_p"],a["escape_job_start"],BLOCK=256)
        return int(bool(self.code_jobs))+int(bool(self.escape_jobs))

class CanonicalSwitch:
    """Inductive state machine: canonical source -> exact base -> canonical target.

    Admission must validate each primitive on the actual parameter allocation
    and backend. No per-request full digest is hidden in the fast path.
    """
    def __init__(self,parameters,factors,unmerge,merge,initial_task,operator_contract):
        self.parameters=parameters
        self.factors=factors
        self.unmerge=unmerge
        self.merge=merge
        self.task=initial_task
        self.contract=operator_contract
        self.valid=True
        self.switches=0
        self.decoder_launches=0
        if set(unmerge)!=set(merge) or set(merge)!=set(factors.values):
            raise ValueError("Task coverage differs between factors and correction banks")
        for task,values in factors.values.items():
            if set(values)!=set(unmerge[task]) or set(values)!=set(merge[task]) or not set(values)<=set(parameters):
                raise ValueError("Projection coverage differs between factors and correction banks")
        first=next(iter(parameters.values()))
        self.groups=None
        if first.device.type=="cuda":
            self.groups={kind:{t:PatchGroup(parameters,bank) for t,bank in tasks.items()}
                         for kind,tasks in [("unmerge",unmerge),("merge",merge)]}

    @property
    def patch_bytes(self):
        return self.auxiliary_bytes+sum(p.nbytes for bank in [self.unmerge,self.merge]
                   for task in bank.values() for p in task.values())

    @property
    def auxiliary_bytes(self):
        return 0 if self.groups is None else sum(g.nbytes for tasks in self.groups.values() for g in tasks.values())

    def correct(self,kind,task):
        bank=self.unmerge if kind=="unmerge" else self.merge
        if self.groups is not None:self.decoder_launches+=self.groups[kind][task].apply_()
        else:
            for name in self.factors.values[task]:bank[task][name].apply_(self.parameters[name])

    def apply(self,target,operator_contract=None):
        if not self.valid:
            raise RuntimeError("State invalid after an interrupted transition; restore a canonical target first")
        if operator_contract is not None and operator_contract!=self.contract:
            raise ValueError("Operator/backend contract changed; regenerate and readmit patches")
        if target not in self.merge:
            raise KeyError(target)
        if target==self.task:
            return "resident"
        source=self.task
        self.valid=False
        with torch.no_grad():
            # Match Factors.add ordering exactly; no dense B@A is retained.
            for name,(a,b,scale) in self.factors.values[source].items():
                w=self.parameters[name]
                w.addmm_(b,a,beta=1.0,alpha=-scale)
            self.correct("unmerge",source)
            for name,(a,b,scale) in self.factors.values[target].items():
                w=self.parameters[name]
                w.addmm_(b,a,beta=1.0,alpha=scale)
            self.correct("merge",target)
        self.task=target
        self.switches+=1
        self.valid=True
        return "canonical_patch"
