"""Three target groups, with O(projections) descriptors and exact escape writes."""


class SharedPatchGroup:
    def __init__(self,parameters,patches):
        import torch
        from compact_group import prefix_counts
        self.parameters,self.patches=parameters,patches
        names=sorted(patches);self.num_projections=len(names)
        if not names:raise ValueError('Empty target group')
        vectors={key:[] for key in ('wp','pp','ip','vp','modes','ns','es')};codes=[];escapes=[]
        for name in names:
            weight=parameters[name];patch=patches[name]
            if weight.device.type!='cuda' or patch.packed.device!=weight.device:raise ValueError('CUDA colocation required')
            if tuple(weight.shape)!=patch.shape or str(weight.dtype)!=patch.dtype or not weight.is_contiguous():
                raise ValueError('Target group parameter layout differs')
            if patch.mode not in ('target_low2','dense'):raise ValueError('Unsupported target mode')
            mode=2 if patch.mode=='target_low2' else 3
            for key,value in (('wp',weight.data_ptr()),('pp',patch.packed.data_ptr()),('ip',patch.indices.data_ptr()),
                ('vp',patch.values.data_ptr()),('modes',mode),('ns',weight.numel()),('es',patch.indices.numel())):
                vectors[key].append(value)
            codes.append(weight.numel());escapes.append(patch.indices.numel())
        self.code_offsets=prefix_counts(codes,1024);self.escape_offsets=prefix_counts(escapes,256)
        self.code_jobs=self.code_offsets[-1];self.escape_jobs=self.escape_offsets[-1]
        device=parameters[names[0]].device
        self.arrays={key:torch.tensor(value,dtype=torch.int64 if key.endswith('p') else torch.int32,device=device)
                     for key,value in vectors.items()}
        self.arrays['code_offsets']=torch.tensor(self.code_offsets,dtype=torch.int32,device=device)
        self.arrays['escape_offsets']=torch.tensor(self.escape_offsets,dtype=torch.int32,device=device)

    @property
    def nbytes(self):return sum(t.numel()*t.element_size() for t in self.arrays.values())

    def apply_(self):
        import shared_cuda as new
        import compact_cuda_kernels as legacy
        a=self.arrays;bits=self.num_projections.bit_length()
        if self.code_jobs:
            new.target_codes[(self.code_jobs,)](a['wp'],a['pp'],a['vp'],a['modes'],a['ns'],a['code_offsets'],
                N_PROJ=self.num_projections,SEARCH_BITS=bits,BLOCK=1024)
        if self.escape_jobs:
            legacy.compact_scatter[(self.escape_jobs,)](a['wp'],a['ip'],a['vp'],a['es'],a['escape_offsets'],
                N_PROJ=self.num_projections,SEARCH_BITS=bits,BLOCK=256)
        return int(bool(self.code_jobs))+int(bool(self.escape_jobs))


class SharedSwitch:
    def __init__(self,torch,parameters,pairs,banks,initial,contract):
        self.torch,self.parameters,self.pairs,self.banks=torch,parameters,pairs,banks
        if set(banks)!= {'target'} or set(banks['target'])!=set(pairs.tasks):raise ValueError('Target bank coverage differs')
        self.groups={task:SharedPatchGroup(parameters,values) for task,values in banks['target'].items()}
        self.task=initial;self.contract=dict(contract);self.valid=True
        self.switches=0;self.addmm_calls=0;self.decoder_launches=0
        self.auxiliary_bytes=sum(group.nbytes for group in self.groups.values())

    def apply(self,target):
        if not self.valid:raise RuntimeError('An interrupted target-shared switch needs canonical restoration')
        if target==self.task:return 'resident'
        source=self.task;self.valid=False
        self.addmm_calls+=self.pairs.add(self.parameters,source,target)
        self.decoder_launches+=self.groups[target].apply_()
        self.task=target;self.switches+=1;self.valid=True
        return 'direct_shared_target_patch'


def attach_executor(previous):
    class TargetExecutor(previous):
        @property
        def uses_target_shared(self):
            return self.policy=='canonical_patch_cached' and self.spec.get('canonical_transition_operator')=='target_shared'

        def reset(self):
            if not self.uses_target_shared:return super().reset()
            policy=self.policy;self.state=None;self.policy='host_copy_cached'
            try:super().reset()
            finally:self.policy=policy
            self.state=SharedSwitch(self.torch,self.params[0],self.factors,self.banks,'mrpc',self.contract)
            self.fixed_bytes=self.payload_bytes+self.state.auxiliary_bytes
            if self.fixed_bytes>self.spec['cap_bytes']:raise MemoryError('Shared target descriptors exceed logical cap')
            self.torch.cuda.synchronize();self.logical_max=self.fixed_bytes

        def install(self,target):
            if not self.uses_target_shared:return super().install(target)
            if target not in self.banks['target']:raise ValueError('Unknown task')
            if self.state.task!=self.task or not self.state.valid:raise ValueError('Parameter identities disagree')
            if target==self.task:return 'resident'
            action=self.state.apply(target);self.addmm_calls=self.state.addmm_calls
            self.task=target;self.switches+=1;self.version='canonical:'+self.e.LOCAL_TARGET_HASHES[target]
            return action
    return TargetExecutor
