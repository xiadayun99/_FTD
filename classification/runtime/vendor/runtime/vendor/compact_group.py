"""O(projections) decoder metadata instead of O(1024-element/256-index tiles).

Encoded correction payloads, tile counts and corrected raw bits are unchanged.
The prefix lookup adds work per GPU block; speed must be measured, not assumed.
"""


def prefix_counts(lengths,block):
    if block<=0:raise ValueError("Positive block size required")
    offsets=[0]
    for n in lengths:
        if n<0:raise ValueError("Negative tile length")
        offsets.append(offsets[-1]+(n+block-1)//block)
    if offsets[-1]>=2**31:raise ValueError("Tile count exceeds the int32 decoder contract")
    return offsets


def locate_job(offsets,job):
    """CPU reference for the binary lifting used in the GPU helper."""
    n=len(offsets)-1
    if n<1 or job<0 or job>=offsets[-1]:raise ValueError("Job outside the launched grid")
    p=0
    for i in range(n.bit_length()):
        candidate=p+(1<<(n.bit_length()-1-i))
        if candidate<=n and offsets[candidate]<=job:p=candidate
    return p,job-offsets[p]


def metadata_layout(records):
    codes=[];escapes=[]
    for row in records:
        mode=row["mode"]
        if mode not in ("zero","sparse","two_bit","dense"):raise ValueError("Unknown correction mode")
        codes.append(row["elements"] if mode in ("two_bit","dense") else 0)
        # Sparse stores ALL changed indices, not just the +/-1 exceptions.
        escapes.append(row["changed"] if mode=="sparse" else row["escapes"] if mode=="two_bit" else 0)
    return prefix_counts(codes,1024),prefix_counts(escapes,256)


class CompactPatchGroup:
    def __init__(self,parameters,patches,torch_module=None):
        if torch_module is None:
            import torch as torch_module
        torch=torch_module
        self.parameters=parameters;self.patches=patches
        names=sorted(patches);self.num_projections=len(names)
        if not names:raise ValueError("Empty grouped correction")
        modes={"zero":0,"sparse":1,"two_bit":2,"dense":3}
        vectors={k:[] for k in ("wp","pp","ip","vp","modes","ns","es")}
        codes=[];escapes=[]
        for name in names:
            w=parameters[name];patch=patches[name]
            if w.device.type!="cuda" or patch.packed.device!=w.device:
                raise ValueError("Compact grouped decoder requires colocated CUDA tensors")
            if tuple(w.shape)!=patch.shape or str(w.dtype)!=patch.dtype or not w.is_contiguous():
                raise ValueError("Compact grouped parameter layout changed")
            if patch.mode not in modes:raise ValueError("Unsupported correction mode")
            for key,value in (("wp",w.data_ptr()),("pp",patch.packed.data_ptr()),
                ("ip",patch.indices.data_ptr()),("vp",patch.values.data_ptr()),
                ("modes",modes[patch.mode]),("ns",w.numel()),("es",patch.indices.numel())):
                vectors[key].append(value)
            codes.append(w.numel() if patch.mode in ("two_bit","dense") else 0)
            escapes.append(patch.indices.numel())
        self.code_offsets=prefix_counts(codes,1024);self.escape_offsets=prefix_counts(escapes,256)
        self.code_jobs=self.code_offsets[-1];self.escape_jobs=self.escape_offsets[-1]
        device=parameters[names[0]].device
        self.arrays={k:torch.tensor(v,dtype=torch.int64 if k.endswith("p") else torch.int32,device=device)
            for k,v in vectors.items()}
        self.arrays["code_offsets"]=torch.tensor(self.code_offsets,dtype=torch.int32,device=device)
        self.arrays["escape_offsets"]=torch.tensor(self.escape_offsets,dtype=torch.int32,device=device)

    @property
    def nbytes(self):return sum(t.numel()*t.element_size() for t in self.arrays.values())

    def apply_(self):
        if not self.code_jobs and not self.escape_jobs:return 0
        import compact_cuda_kernels as k
        a=self.arrays;bits=self.num_projections.bit_length()
        if self.code_jobs:
            k.compact_codes[(self.code_jobs,)](a["wp"],a["pp"],a["vp"],a["modes"],a["ns"],
                a["code_offsets"],N_PROJ=self.num_projections,SEARCH_BITS=bits,BLOCK=1024)
        if self.escape_jobs:
            k.compact_scatter[(self.escape_jobs,)](a["wp"],a["ip"],a["vp"],a["es"],
                a["escape_offsets"],N_PROJ=self.num_projections,SEARCH_BITS=bits,BLOCK=256)
        return int(bool(self.code_jobs))+int(bool(self.escape_jobs))
