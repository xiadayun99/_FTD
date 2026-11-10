"""Experimental one-addmm canonical transitions for three fixed LoRA tasks.

Rank concatenation is established adapter algebra, not a novelty claim.
The new, operator-specific correction must be rebuilt and admitted on the GPU.
All-pair storage is quadratic in task count; this probe covers only 3 tasks.
"""
import itertools
import math


def edge_key(source,target):
    if source==target:raise ValueError("Self transitions need no edge correction")
    return source+"__to__"+target


class PairFactors:
    """One unordered pair shares its concatenation across both directions.

For i<j: A=[A_i; A_j], B=[-B_i, B_j]. With a common scale s,
s*BA=D_j-D_i. Reversing the transition changes alpha's sign.
No dense transition matrix or per-switch concatenation is retained.
"""
    def __init__(self,torch,factors):
        self.torch=torch;self.tasks=tuple(sorted(factors.values));self.values={}
        if len(self.tasks)!=3:raise ValueError("This bounded probe requires exactly three admitted tasks")
        coverage=[set(factors.values[t]) for t in self.tasks]
        if not coverage[0] or any(v!=coverage[0] for v in coverage):
            raise ValueError("All tasks must cover the same projections")
        self.bytes=0;self.geometry=[]
        with torch.no_grad():
            for first,second in itertools.combinations(self.tasks,2):
                values={}
                for name in sorted(coverage[0]):
                    a1,b1,s1=factors.values[first][name];a2,b2,s2=factors.values[second][name]
                    if s1!=s2 or not math.isfinite(s1):
                        raise ValueError("This probe requires equal finite scales; no silent rescaling/quantization")
                    tensors=(a1,b1,a2,b2)
                    if any(t.dtype!=torch.float16 or not t.is_contiguous() for t in tensors):
                        raise ValueError("Contiguous FP16 factors required")
                    if a1.shape[1]!=a2.shape[1] or b1.shape[0]!=b2.shape[0]:
                        raise ValueError("Factor projection shapes differ")
                    a=torch.cat((a1,a2),dim=0).contiguous()
                    b=torch.cat((-b1,b2),dim=1).contiguous()
                    values[name]=(a,b,s1)
                    self.bytes+=(a.numel()+b.numel())*a.element_size()
                    self.geometry.append(dict(pair=[first,second],name=name,rank=a.shape[0],
                        a_shape=list(a.shape),b_shape=list(b.shape),scale=s1))
                self.values[first,second]=values

    def projection_values(self,source,target):
        pair=tuple(sorted((source,target)))
        if source==target or pair not in self.values:raise KeyError((source,target))
        return self.values[pair],1. if target==pair[1] else -1.

    def add(self,parameters,source,target):
        values,sign=self.projection_values(source,target)
        with self.torch.no_grad():
            for name,(a,b,scale) in values.items():
                parameters[name].addmm_(b,a,beta=1.,alpha=sign*scale)
        return len(values)


def build_banks(torch,parameters,pairs,targets,encode):
    banks={"edge":{}};records=[]
    with torch.no_grad():
        for source in pairs.tasks:
            for target in pairs.tasks:
                if source==target:continue
                key=edge_key(source,target);bank={}
                values,sign=pairs.projection_values(source,target)
                for name,(a,b,scale) in values.items():
                    w=parameters[name]
                    w.copy_(targets[source][name]);w.addmm_(b,a,beta=1.,alpha=sign*scale)
                    if not torch.isfinite(w).all():raise FloatingPointError("Nonfinite direct predictor")
                    patch=encode(w,targets[target][name]);bank[name]=patch
                    records.append(dict(source=source,target=target,name=name,**patch.metadata()))
                banks["edge"][key]=bank
    return banks,records


def validate_primitives(torch,parameters,pairs,targets,banks,bit_equal,repeats=1):
    count=0
    with torch.no_grad():
        for _ in range(repeats):
            for source in pairs.tasks:
                for target in pairs.tasks:
                    if source==target:continue
                    values,sign=pairs.projection_values(source,target)
                    for name,(a,b,scale) in values.items():
                        w=parameters[name]
                        w.copy_(targets[source][name]);w.addmm_(b,a,beta=1.,alpha=sign*scale)
                        banks["edge"][edge_key(source,target)][name].apply_(w,verify_input=True)
                        if not bit_equal(w,targets[target][name]):
                            raise ValueError("Full-projection direct correction mismatch: "+edge_key(source,target)+":"+name)
                        count+=1
    torch.cuda.synchronize()
    return dict(all_equal=True,repeats=repeats,full_projection_checks=count)


class DirectSwitch:
    """Canonical source -> one low-rank update -> exact canonical target."""
    def __init__(self,torch,parameters,pairs,banks,initial_task,contract,group_class):
        self.torch=torch;self.parameters=parameters;self.pairs=pairs
        self.banks=banks;self.task=initial_task;self.contract=contract
        self.valid=True;self.switches=0;self.decoder_launches=0;self.addmm_calls=0
        expected={edge_key(s,t) for s in pairs.tasks for t in pairs.tasks if s!=t}
        if initial_task not in pairs.tasks or set(banks.get("edge",{}))!=expected:
            raise ValueError("Directed correction coverage differs")
        for source in pairs.tasks:
            for target in pairs.tasks:
                if source==target:continue
                vals,_=pairs.projection_values(source,target)
                if set(vals)!=set(banks["edge"][edge_key(source,target)]) or not set(vals)<=set(parameters):
                    raise ValueError("Direct correction projection coverage differs")
        self.groups={k:group_class(parameters,v) for k,v in banks["edge"].items()}

    @property
    def auxiliary_bytes(self):return sum(g.nbytes for g in self.groups.values())

    def apply(self,target,operator_contract=None):
        if not self.valid:raise RuntimeError("Interrupted direct transition: restore a canonical state first")
        if operator_contract is not None and operator_contract!=self.contract:
            raise ValueError("Direct operator contract changed")
        if target not in self.pairs.tasks:raise KeyError(target)
        if target==self.task:return "resident"
        source=self.task;self.valid=False
        self.addmm_calls+=self.pairs.add(self.parameters,source,target)
        self.decoder_launches+=self.groups[edge_key(source,target)].apply_()
        self.task=target;self.switches+=1;self.valid=True
        return "direct_edge_patch"
