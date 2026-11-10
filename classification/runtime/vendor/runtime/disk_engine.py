"""One comparison model per worker; ordinary disk checkpoints between phases."""
from pathlib import Path
from types import SimpleNamespace
import os
import time

import disk_protocol as sp
import disk_store as ds
from disk_training import load_base,release
from disk_audit import audit_records,audit_admission


def runtime(port,root,initial=False):
    import torch
    port.require_no_observed_foreign(os.getpgrp())
    core,e,c,p,o=port.load_core();contract=port.runtime_contract(torch,e,p,'strict_eager')
    if not initial and contract!=sp.read(root/'runtime_contract.json'):
        raise ValueError('GPU/software arithmetic contract changed')
    if torch.cuda.mem_get_info()[0]<48*sp.GIB:raise MemoryError('Need at least 48 GiB currently free GPU memory')
    return torch,core,e,c,p,o,contract


def locations(root,toy):
    scope=root/'tiny' if toy else root
    return scope,(scope/'base' if toy else sp.MODEL_DIR),scope/'trained'


def tiny_setup(root,port):
    import gc
    import shutil
    from transformers import Qwen3Config,Qwen3ForCausalLM,AutoTokenizer
    from peft import LoraConfig,get_peft_model
    from safetensors.torch import load_file,save_file
    from disk_training import collate,save_fp16_adapter
    torch,core,e,c,p,o,contract=runtime(port,root,initial=True)
    import bounded_io as bio
    if bio.memory_snapshot()['MemAvailable_bytes']<sp.HOST_START:
        raise MemoryError('Need 104 GiB available host memory for the later standard-save validation stages')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 training is unavailable')
    decoder=core.validate_cuda_decoder_vectors(torch)
    folder=root/'tiny';folder.mkdir()
    torch.manual_seed(17)
    base=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        pad_token_id=0,eos_token_id=2)).to(device='cuda:0',dtype=torch.bfloat16)
    base.config._attn_implementation='eager';base.name_or_path=str(folder/'base')
    base.save_pretrained(folder/'base',safe_serialization=True)
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=.05,bias='none',
        task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train()
    trainable=[value for value in model.parameters() if value.requires_grad]
    for value in trainable:value.data=value.data.float()
    before=[value.detach().clone() for value in trainable]
    optimizer=torch.optim.AdamW(trainable,lr=2e-4,weight_decay=0.)
    loss=model(**collate(torch,[([3,4,5,6,2],[-100,-100,5,6,2]),([7,8,9,2],[-100,-100,9,2])],0)).loss
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Tiny training loss is nonfinite')
    loss.backward();optimizer.step()
    if not any(not torch.equal(a,b) for a,b in zip(before,trainable)):raise ValueError('Tiny training changed no adapter')
    model.eval();model.config.use_cache=True;save_fp16_adapter(torch,model,folder/'adapter')
    del optimizer,trainable,before,loss,model,base;gc.collect();release(torch)
    for i,task in enumerate(sp.TASKS):
        directory=folder/'trained/lora'/task;directory.mkdir(parents=True)
        shutil.copy2(folder/'adapter/adapter_config.json',directory/'adapter_config.json')
        tensors=load_file(str(folder/'adapter/adapter_model.safetensors'))
        for name in tensors:
            if '.lora_B.' in name:tensors[name]=(tensors[name]*(i+1)).contiguous()
        save_file(tensors,str(directory/'adapter_model.safetensors'))
    tokenizer=AutoTokenizer.from_pretrained(str(sp.MODEL_DIR),local_files_only=True,trust_remote_code=False)
    work,_=sp.prepare_workload(tokenizer);sp.dump(root/'engine/input/workload.json',work)
    sp.dump(root/'runtime_contract.json',contract)
    report=dict(status='VERIFIED',decoder=decoder,tiny_train_save_reload=True,gradient_checkpointing=True,
        saved_adapter_dtype='float16',gpu_allocation_after_setup_bytes=torch.cuda.memory_allocated(),
        allocation_is_diagnostic_not_a_model_leak_classifier=True,new_8b_training_performed=False,
        runtime_contract=contract)
    sp.dump(folder/'setup_report.json',report)
    return [p for p in folder.rglob('*') if p.is_file()]+[root/'runtime_contract.json',root/'engine/input/workload.json']


def prepare_reference(root,port,task,toy=False):
    from peft import PeftModel
    import bounded_io as bio
    torch,core,e,c,p,o,contract=runtime(port,root)
    scope,model_dir,trained=locations(root,toy);directory=scope/'artifacts/references'
    directory.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    names=e.mutable_names_from_factors(trained/'lora/mrpc')
    if len(names)!=(7 if toy else 252):raise ValueError('Mutable projection count differs')
    required=sp.DISK_FLOOR+(0 if toy else sp.MUTABLE_BYTES*(2 if task=='mrpc' else 1))+16*sp.MIB
    ds.require_space(root,required)
    if not toy and bio.memory_snapshot()['MemAvailable_bytes']<48*sp.GIB:
        raise MemoryError('Need 48 GiB available host memory to construct one disk reference')
    model=load_base(torch,model_dir,'cuda:0',torch.float16)
    if any(not bool(torch.isfinite(v).all()) for v in list(model.parameters())+list(model.buffers())):
        raise FloatingPointError('Nonfinite base parameter/buffer')
    params=dict(model.named_parameters())
    common=e.tensor_digest(torch,[(n,v) for n,v in model.named_parameters() if n not in names]+
        [('buffer:'+n,v) for n,v in model.named_buffers()])
    outputs=[]
    if task=='mrpc':
        values={n:params[n].detach().cpu().clone().contiguous() for n in names}
        saved=ds.save_tensors(values,directory/'base.safetensors')
        sp.dump(directory/'base.json',dict(**saved,tensor_sha256=e.tensor_digest(torch,values.items()),
            shared_state_sha256=common,finite=True,mutable_projections=len(names),toy=toy))
        outputs.extend([directory/'base.safetensors',directory/'base.json']);del values
        bio.trim_heap()
    wrapped=PeftModel.from_pretrained(model,trained/'lora'/task,is_trainable=False,
        autocast_adapter_dtype=False,local_files_only=True)
    merged=wrapped.merge_and_unload(safe_merge=True)
    merged_common=e.tensor_digest(torch,[(n,v) for n,v in merged.named_parameters() if n not in names]+
        [('buffer:'+n,v) for n,v in merged.named_buffers()])
    if merged_common!=common:raise ValueError('Merging changed a non-mutable parameter or buffer')
    params=dict(merged.named_parameters())
    values={n:params[n].detach().cpu().clone().contiguous() for n in names}
    if any(v.dtype!=torch.float16 or not bool(torch.isfinite(v).all()) for v in values.values()):
        raise ValueError('Merged reference must be finite FP16')
    saved=ds.save_tensors(values,directory/(task+'.safetensors'))
    report=dict(**saved,task=task,tensor_sha256=e.tensor_digest(torch,values.items()),shared_state_sha256=common,
        finite=True,mutable_projections=len(names),toy=toy,wall_seconds=time.perf_counter()-began,
        adapter_sha256=sp.sha(trained/'lora'/task/'adapter_model.safetensors'),memory=bio.memory_snapshot(),
        storage='standard safetensors on disk',runtime_contract=contract)
    sp.dump(directory/(task+'.json'),report)
    return outputs+[directory/(task+'.safetensors'),directory/(task+'.json')]


def assemble_references(root,toy=False):
    scope,model_dir,trained=locations(root,toy);directory=scope/'artifacts/references'
    records={task:sp.read(directory/(task+'.json')) for task in sp.TASKS}
    base=sp.read(directory/'base.json');common=base['shared_state_sha256']
    if any(r['shared_state_sha256']!=common or r['mutable_projections']!=base['mutable_projections']
           or r['toy']!=toy or not r['finite'] for r in records.values()):raise ValueError('Reference records disagree')
    metadata=dict(target_slice_sha256={task:r['tensor_sha256'] for task,r in records.items()},
        base_slice_sha256=base['tensor_sha256'],shared_state_sha256=common,mutable_projections=base['mutable_projections'],
        reference_tensor_bytes=sum(r['tensor_bytes'] for r in records.values()),base_tensor_bytes=base['tensor_bytes'],
        reference_build_wall_seconds=sum(r['wall_seconds'] for r in records.values()),toy=toy,
        file_sha256={task:records[task]['file_sha256'] for task in sp.TASKS})
    sp.dump(directory/'metadata.json',metadata)
    return [directory/'metadata.json']


class AdmissionEncoder:
    def __init__(self,original,directory,fixed,port,toy):
        self.original,self.directory,self.fixed,self.port,self.toy=original,directory,fixed,port,toy
        self.payload=0;self.count=0
    def __call__(self,predictor,target):
        import bounded_io as bio
        if not self.toy and bio.memory_snapshot()['MemAvailable_bytes']<12*sp.GIB+64*predictor.numel()+sp.GIB:
            raise MemoryError('Insufficient host headroom for one original CPU encoding operation')
        patch=self.original(predictor,target);self.payload+=patch.nbytes;self.count+=1
        sp.append(self.directory/'encoding_progress.jsonl',dict(projection=self.count,payload_bytes=self.payload,
            fixed_state_bytes=self.fixed,memory=bio.memory_snapshot()))
        if self.fixed+self.payload>sp.CAP:raise MemoryError('Operator state exceeds frozen 32 GiB logical cap')
        if self.count%36==0:
            self.port.require_no_observed_foreign(os.getpgrp());bio.trim_heap()
            print('ENCODE:',self.count,'payload GiB=',round(self.payload/sp.GIB,3),flush=True)
        return patch


class Session:
    """Exactly one model is created; this worker exits after one operation."""
    def __init__(self,root,port,method,toy=False,with_banks=True):
        from peft import PeftModel
        import bounded_io as bio
        began=time.perf_counter()
        self.root,self.port,self.method,self.toy=root,port,method,toy
        self.torch,self.core,self.e,self.c,self.p,self.o,contract=runtime(port,root)
        from direct_transition import PairFactors
        torch,e=self.torch,self.e
        if not toy and bio.memory_snapshot()['MemAvailable_bytes']<sp.HOST_START:
            raise MemoryError('Need 104 GiB available host memory before loading one reference bundle')
        self.scope,self.model_dir,self.trained=locations(root,toy)
        self.targets,self.base,self.metadata=ds.reference_bundle(torch,e,self.scope)
        self.names=sorted(self.base)
        self.contract=dict(contract,canonical_target_hashes=e.LOCAL_TARGET_HASHES,
            shared_state_sha256=self.metadata['shared_state_sha256'],
            scale_protocol_sha256=sp.sha(sp.HERE/'disk_protocol.py'),
            scale_engine_sha256=sp.sha(sp.HERE/'disk_engine.py'))
        model=load_base(torch,self.model_dir,'cuda:0',torch.float16)
        common=e.tensor_digest(torch,[(n,v) for n,v in model.named_parameters() if n not in self.names]+
            [('buffer:'+n,v) for n,v in model.named_buffers()])
        if common!=self.metadata['shared_state_sha256']:raise ValueError('Fresh base common-state hash differs')
        self.factors=None;self.banks=None;self.factor_preparation_seconds=0.
        if method=='native':
            model=PeftModel.from_pretrained(model,self.trained/'lora/mrpc',adapter_name='mrpc',
                is_trainable=False,autocast_adapter_dtype=False,local_files_only=True)
            for task in sp.TASKS[1:]:
                model.load_adapter(self.trained/'lora'/task,adapter_name=task,is_trainable=False,
                    autocast_adapter_dtype=False,local_files_only=True)
        elif method in ('node','direct'):
            began=time.perf_counter()
            runtime_state=SimpleNamespace(torch=torch,tasks=list(sp.TASKS),artifact=self.trained,
                device=torch.device('cuda:0'),dtype=torch.float16,manifest={'mutable_parameters':self.names},
                sync=lambda:torch.cuda.synchronize())
            node=e.Factors(runtime_state,dict(model.named_parameters()))
            self.factors=PairFactors(torch,node) if method=='direct' else node
            del node
            self.factor_preparation_seconds=time.perf_counter()-began
        self.model=model.eval()
        if method in ('node','direct') and with_banks:
            import rounding_patch as rp
            self.banks,self.bank_metadata=ds.load_banks(torch,rp.Patch,self.scope/'artifacts/patches'/method,self.contract)
        bio.trim_heap();self.allocation_after_load_bytes=torch.cuda.memory_allocated()
        self.setup_wall_seconds=time.perf_counter()-began

    def build(self):
        import rounding_patch as rp
        from direct_transition import build_banks
        directory=self.scope/'artifacts/patches'/self.method;directory.mkdir(parents=True,exist_ok=False)
        ds.require_space(self.root,sp.CAP-sp.MODEL_BYTES+sp.DISK_FLOOR if not self.toy else sp.DISK_FLOOR)
        projections=len(self.names);descriptors=6*(4*8*projections+3*4*projections+2*(projections+1)*4)
        fixed=self.e.model_payload_bytes(self.model)+self.factors.bytes+descriptors
        encoder=AdmissionEncoder(rp.encode,directory,fixed,self.port,self.toy)
        params=dict(self.model.named_parameters());began=time.perf_counter()
        if self.method=='node':
            original=rp.encode;rp.encode=encoder
            try:banks,records,_=self.core.build_patches(self.torch,params,self.factors,self.base,self.targets)
            finally:rp.encode=original
        else:banks,records=build_banks(self.torch,params,self.factors,self.targets,encoder)
        encoding_seconds=time.perf_counter()-began;save_began=time.perf_counter()
        metadata=ds.save_banks(directory,banks,self.contract)
        if encoder.count!=6*projections or metadata['tensor_bytes']!=encoder.payload:
            raise ValueError('Correction build accounting differs')
        sp.dump(directory/'encoding_records.json',records)
        sp.dump(directory/'build_report.json',dict(method=self.method,payload_bytes=encoder.payload,
            projection_records=encoder.count,fixed_state_before_corrections_bytes=fixed,
            encoding_wall_seconds=encoding_seconds,serialization_wall_seconds=time.perf_counter()-save_began,
            factor_preparation_seconds=self.factor_preparation_seconds,
            wall_seconds=time.perf_counter()-began,corrections_stored_on_disk=True,toy=self.toy))
        return [p for p in directory.rglob('*') if p.is_file()]

    def validate(self,full_paths):
        import rounding_patch as rp
        from direct_transition import validate_primitives,DirectSwitch
        from compact_group import CompactPatchGroup
        params=dict(self.model.named_parameters());torch=self.torch;e=self.e
        print('VALIDATE:',self.method,'full projections',6*len(self.names),flush=True)
        if self.method=='direct':
            value=validate_primitives(torch,params,self.factors,self.targets,self.banks,rp.bit_equal,repeats=1)
            make=lambda initial:DirectSwitch(torch,params,self.factors,self.banks,initial,self.contract,CompactPatchGroup)
        else:
            value=self.core.validate_primitives(torch,params,self.factors,self.base,self.targets,self.banks,repeats=1)
            make=lambda initial:rp.CanonicalSwitch(params,self.factors,self.banks['unmerge'],self.banks['merge'],initial,self.contract)
        paths=self.port.full_path_checks(torch,e,params,self.targets,make) if full_paths else None
        common=e.tensor_digest(torch,[(n,v) for n,v in self.model.named_parameters() if n not in self.names]+
            [('buffer:'+n,v) for n,v in self.model.named_buffers()])
        if common!=self.metadata['shared_state_sha256']:raise ValueError('Common state changed after validation')
        return dict(primitives=value,grouped_paths=paths,shared_state_sha256=common)

    def tiny_check(self):
        from direct_transition import DirectSwitch
        from compact_group import CompactPatchGroup
        import rounding_patch as rp
        torch,e,p,o=self.torch,self.e,self.p,self.o
        validation=self.validate(True) if self.method in ('node','direct') else None
        checks={};params=dict(self.model.named_parameters())
        if self.method!='native':
            e.install_host_target(torch,params,self.targets['mrpc'])
            if self.method=='direct':switch=DirectSwitch(torch,params,self.factors,self.banks,'mrpc',self.contract,CompactPatchGroup)
            elif self.method=='node':switch=rp.CanonicalSwitch(params,self.factors,self.banks['unmerge'],self.banks['merge'],'mrpc',self.contract)
        for task in sp.TASKS:
            if self.method=='native':self.model.set_adapter(task);self.model.eval()
            elif self.method=='reference':e.install_host_target(torch,params,self.targets[task])
            else:switch.apply(task)
            if self.method!='native':
                digest=e.tensor_digest(torch,[(n,params[n]) for n in self.names])
                if digest!=e.LOCAL_TARGET_HASHES[task]:raise ValueError('Tiny full mutable-weight hash differs')
            full,bundle=o.batch_score(torch,e,p,self.model,[3,4,5,6,7],{'a':[8],'b':[9,10]},3,retain_head=True)
            template=o.extract_batch_head(p,bundle)
            before=e.tensor_digest(torch,[(str(i),t) for i,t in enumerate(e.cache_tensors(template))])
            cached,_=o.batch_score(torch,e,p,self.model,[3,4,5,6,7],{'a':[8],'b':[9,10]},3,template=template)
            after=e.tensor_digest(torch,[(str(i),t) for i,t in enumerate(e.cache_tensors(template))])
            if before!=after:raise ValueError('Tiny immutable prefix changed')
            checks[task]=dict(full=full,cached=cached,immutable=True)
            del template,bundle
        path=self.scope/'checks'/(self.method+'.json')
        sp.dump(path,dict(status='VERIFIED',method=self.method,validation=validation,scoring=checks,
            fresh_process=True,one_comparison_model=True,allocation_after_load_bytes=self.allocation_after_load_bytes,
            test_is_tiny_gpu_api_check_not_8b_performance=True))
        return [path]

    def trace(self,tag,admission=False):
        import numerical_guard as ng
        import contextlib
        folder=self.root/'stages'/tag;folder.mkdir(parents=True,exist_ok=False)
        self.core.ROOT=self.root/'engine';limit=6 if admission else 384
        spec=sp.specification(self.root,self.method)
        with self.port.Telemetry(folder):
            self.port.require_no_observed_foreign(os.getpgrp())
            validation=self.validate(True) if self.method in ('node','direct') else None
            sp.dump(folder/'pre_trace_validation.json',dict(validation=validation,runtime_contract=self.contract,
                one_comparison_model_per_process=True,allocation_after_load_bytes=self.allocation_after_load_bytes,
                setup_wall_seconds=self.setup_wall_seconds))
            context=ng.record_probe_cases(self.torch,self.core,folder) if admission or self.method=='reference' else contextlib.nullcontext()
            with context:
                report=self.core.execute_trace(folder,spec,spec['policy'],self.model,self.factors,self.targets,self.base,
                    self.contract,self.e,self.c,self.p,self.o,banks=self.banks,limit=limit,
                    write_tag='smoke' if admission else 'trace')
            report.update(method=self.method,runtime_contract=self.contract,target_hashes=self.e.LOCAL_TARGET_HASHES,
                validation=validation,mutable_projections=len(self.names),logical_cap_bytes=sp.CAP,retained_kv_cap_bytes=sp.KV_CAP,
                separate_worker_process=True,one_comparison_model_per_process=True,worker_pid=os.getpid())
            sp.dump(folder/'report.json',report)
        records=sp.rows(folder/('smoke_requests.jsonl' if admission else 'trace_requests.jsonl'))
        work=sp.read(self.root/'engine/input/workload.json')['check'][:limit]
        audit_records(records,work,report,self.method)
        audit_admission(report,self.method,admission,len(self.names),limit)
        samples=sp.rows(folder/'telemetry.jsonl')
        if not samples or any(x['observation']['foreign_pids'] for x in samples):
            raise RuntimeError('Missing telemetry or another GPU process was observed during this stage')
        sp.dump(folder/'audit.json',dict(status='AUDITED',requests=len(records),telemetry_samples=len(samples),
            isolation_enumeration_verified=all(x['observation']['process_enumeration_verified'] for x in samples)))
        return [p for p in folder.rglob('*') if p.is_file()]


def worker(root,port,job):
    phase=job['phase'];toy=job.get('toy',False)
    if phase=='tiny_setup':return tiny_setup(root,port)
    if phase=='train':
        from disk_training import train_task
        task=job['task'];train_task(root,task,port)
        return [p for p in (root/'trained/lora'/task).rglob('*') if p.is_file()]+[root/'training_logs'/(task+'.jsonl')]
    if phase=='reference':return prepare_reference(root,port,job['task'],toy)
    if phase=='assemble':return assemble_references(root,toy)
    session=Session(root,port,job['method'],toy,with_banks=phase!='build')
    if phase=='build':return session.build()
    if phase=='tiny_check':return session.tiny_check()
    if phase=='trace':return session.trace(job['tag'],job.get('admission',False))
    raise ValueError('Unknown phase')
