#!/usr/bin/env python3
"""Mechanism admission, exact-weight audit, and matched complete-request pilot.

No training, downloads, dependency changes, or killing other GPU jobs.
The bundled questions are reused development evidence, never a blind test.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import collections
import copy
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
VERSION="ftd_canonical_patch_20261102_v1"
TASKS=("mrpc","qnli","agnews")
POLICIES=("host_copy_cached","device_copy1_cached","device_base_patch_cached","two_slots_copy_full","two_slots_copy_cached",
          "hf_native_cached","increment_exact_kv","increment_old_kv","canonical_patch_cached")
MIB=2**20
HELPERS={
 "frozen_kv_engine.py":"0101cdad4d9b887919e09d2004c6bdf7eb68556039c06dfd98ca32b5c46f18fb",
 "frozen_continuous.py":"e091ef792debb959bd35be8411f07937cbd6f59e4d7dee1f59c985ecb82e2c15",
 "frozen_profile.py":"57f4458daaeb3cdc0e263d0a2f8f1650957f51529303c19a9dd6154a7252ee69",
 "ftd_optimized_continuous_20261102.py":"cd4037686c01c16d162807f6be099f8d96e55a07ef80789d26ee1fe631bb7779"}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def identity(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def append(p,row):
    with Path(p).open("a") as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")

def module(name,filename):
    path=ROOT/"frozen"/filename
    if sha(path)!=HELPERS[filename]:raise ValueError("Frozen helper changed: "+filename)
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj

def helpers():
    e=module("closure_engine","frozen_kv_engine.py")
    c=module("closure_continuous","frozen_continuous.py")
    p=module("closure_profile","frozen_profile.py")
    o=module("closure_optimized","ftd_optimized_continuous_20261102.py")
    return e,c,p,o

def runtime(torch,e,p):
    torch.manual_seed(0)
    e.torch_runtime(torch)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    p.cache_runtime_contract()
    import importlib.metadata as md
    import triton
    return dict(torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,
                transformers=md.version("transformers"),peft=md.version("peft"),
                gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
                driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader","--id=0"],text=True).strip(),
                cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                sdp_flash=torch.backends.cuda.flash_sdp_enabled(),
                sdp_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                sdp_math=torch.backends.cuda.math_sdp_enabled(),
                deterministic=True,tf32=False,
                decoder_sha256=sha(ROOT/"cuda_patch_kernels.py"),
                operator_sha256=sha(ROOT/"rounding_patch.py"))

def load_base(spec,torch,e,targets,native=False,need_factors=True):
    model=e.load_profile_base(spec,torch)
    names=e.mutable_names_from_factors(Path(spec["taskpool_dir"])/"lora"/"mrpc")
    params=dict(model.named_parameters())
    if set(names)!=set(targets["mrpc"]):raise ValueError("Mutable reference/factor union differs")
    metadata=e.read_signed(Path(spec["source_result"])/"target_metadata.json")
    common=e.tensor_digest(torch,[(n,t) for n,t in model.named_parameters() if n not in names]+
        [("buffer:"+n,t) for n,t in model.named_buffers()])
    if common!=metadata["shared_state_sha256"]:raise ValueError("Unchanged model tensors differ from canonical reference")
    base={n:params[n].detach().cpu().clone().contiguous() for n in names}
    if native:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,Path(spec["taskpool_dir"])/"lora"/"mrpc",
            adapter_name="mrpc",is_trainable=False,autocast_adapter_dtype=False,local_files_only=True)
        for t in TASKS[1:]:
            model.load_adapter(Path(spec["taskpool_dir"])/"lora"/t,adapter_name=t,
                               is_trainable=False,autocast_adapter_dtype=False,local_files_only=True)
        model.to(dtype=torch.float16).eval()
        return model,None,base
    factors=None
    if need_factors:
        rt=SimpleNamespace(torch=torch,tasks=list(TASKS),artifact=Path(spec["taskpool_dir"]),
            device=torch.device("cuda:0"),dtype=torch.float16,manifest={"mutable_parameters":names},
            sync=lambda:torch.cuda.synchronize())
        factors=e.Factors(rt,params)
    return model,factors,base

def serialize_patches(root,banks,contract):
    from safetensors.torch import save_file
    tensors={};meta={"contract":contract,"banks":{}}
    for kind,tasks in banks.items():
        meta["banks"][kind]={}
        for task,patches in tasks.items():
            meta["banks"][kind][task]={}
            for i,(name,patch) in enumerate(sorted(patches.items())):
                key=f"{kind}.{task}.{i}"
                for field in ["packed","indices","values"]:tensors[key+"."+field]=getattr(patch,field).cpu().contiguous()
                meta["banks"][kind][task][name]=dict(key=key,metadata=patch.metadata())
    save_file(tensors,str(root/"patches.safetensors"))
    meta["payload_sha256"]=sha(root/"patches.safetensors")
    dump(root/"patch_metadata.json",meta)
    return meta

def deserialize_patches(root,device,contract,kinds=None):
    import torch
    from safetensors.torch import load_file
    from rounding_patch import Patch
    meta=read(root/"patch_metadata.json")
    if contract!=meta["contract"] or sha(root/"patches.safetensors")!=meta["payload_sha256"]:
        raise ValueError("Patch operator/environment/payload changed")
    tensors=load_file(str(root/"patches.safetensors"),device="cpu")
    banks={}
    for kind,tasks in meta["banks"].items():
        if kinds is not None and kind not in kinds:continue
        banks[kind]={}
        for task,patches in tasks.items():
            banks[kind][task]={}
            for name,info in patches.items():
                key=info["key"];m=info["metadata"]
                stats={k:m[k] for k in ["elements","changed","escapes","changed_fraction","dense_reference_bytes"]}
                patch=Patch(tuple(m["shape"]),m["dtype"],m["mode"],
                            *[tensors[key+"."+f] for f in ["packed","indices","values"]],
                            m["predictor_sha256"],m["target_sha256"],stats).to(device)
                banks[kind][task][name]=patch
    return banks,meta

def build_patches(torch,params,factors,base,targets):
    from rounding_patch import encode
    banks={k:{t:{} for t in TASKS} for k in ["unmerge","merge"]}
    start=time.perf_counter();records=[]
    with torch.no_grad():
        for task in TASKS:
            print("Building reference corrections:",task,flush=True)
            for name,(a,b,scale) in factors.values[task].items():
                w=params[name]
                if not w.is_contiguous():raise ValueError("Noncontiguous parameter: "+name)
                if not torch.isfinite(targets[task][name]).all():raise ValueError("Nonfinite target")
                w.copy_(base[name]);w.addmm_(b,a,beta=1.,alpha=scale)
                merge=encode(w,targets[task][name])
                w.copy_(targets[task][name]);w.addmm_(b,a,beta=1.,alpha=-scale)
                unmerge=encode(w,base[name])
                banks["merge"][task][name]=merge
                banks["unmerge"][task][name]=unmerge
                for kind,patch in [("merge",merge),("unmerge",unmerge)]:
                    records.append(dict(kind=kind,task=task,name=name,**patch.metadata()))
    torch.cuda.synchronize()
    return banks,records,time.perf_counter()-start

def validate_primitives(torch,params,factors,base,targets,banks,repeats=2,kinds=("unmerge","merge")):
    """Full predictor digest and full corrected tensor, never sampled weights."""
    from rounding_patch import bit_equal
    counts=collections.Counter()
    with torch.no_grad():
        for repeat in range(repeats):
            for task in TASKS:
                for name,(a,b,scale) in factors.values[task].items():
                    w=params[name]
                    for kind,start,expected,sign in [
                        ("unmerge",targets[task][name],base[name],-1.),
                        ("merge",base[name],targets[task][name],1.)]:
                        if kind not in kinds:continue
                        w.copy_(start);w.addmm_(b,a,beta=1.,alpha=sign*scale)
                        banks[kind][task][name].apply_(w,verify_input=True)
                        if not bit_equal(w,expected):
                            raise ValueError("Full CUDA correction mismatch: "+kind+":"+task+":"+name)
                        counts[kind]+=1
    torch.cuda.synchronize()
    return dict(repeats=repeats,full_projection_checks=dict(counts),all_equal=True)

def audit_paths(torch,e,params,factors,base,targets,banks,contract):
    from rounding_patch import CanonicalSwitch
    logs=[]
    base_hash=e.tensor_digest(torch,base.items())
    # Every directed task pair, followed by repeated three-task cycles.
    for source in TASKS:
        e.install_host_target(torch,params,targets[source])
        state=CanonicalSwitch(params,factors,banks["unmerge"],banks["merge"],source,contract)
        factors.add(params,source,-1.)
        state.correct("unmerge",source)
        returned=e.tensor_digest(torch,[(n,params[n]) for n in base])
        if returned!=base_hash:raise ValueError("Grouped decoder did not restore the exact base")
        factors.add(params,source,1.)
        state.correct("merge",source)
        restored=e.tensor_digest(torch,[(n,params[n]) for n in targets[source]])
        if restored!=e.LOCAL_TARGET_HASHES[source]:raise ValueError("Grouped decoder target admission failed")
        logs.append(dict(source=source,target=source,action="grouped_base_roundtrip",
            mutable_weights_sha256=restored,base_weights_sha256=returned,switches=0,
            grouped_decoder_metadata_bytes=state.auxiliary_bytes))
        path=[t for t in TASKS if t!=source]+[source]+list(TASKS)*3
        for target in path:
            action=state.apply(target,contract)
            actual=e.tensor_digest(torch,[(n,params[n]) for n in targets[target]])
            if actual!=e.LOCAL_TARGET_HASHES[target]:raise ValueError("Full-model canonical closure failed")
            logs.append(dict(source=source,target=target,action=action,
                             mutable_weights_sha256=actual,switches=state.switches,
                             grouped_decoder_metadata_bytes=state.auxiliary_bytes))
    # Continuous operator stress, with zero canonical copies inside the loop.
    e.install_host_target(torch,params,targets["mrpc"])
    state=CanonicalSwitch(params,factors,banks["unmerge"],banks["merge"],"mrpc",contract)
    checkpoints={1,8,32,128,512,1024}
    for step in range(1,1025):
        target=TASKS[step%len(TASKS)]
        state.apply(target,contract)
        if step in checkpoints:
            actual=e.tensor_digest(torch,[(n,params[n]) for n in targets[target]])
            if actual!=e.LOCAL_TARGET_HASHES[target]:raise ValueError("Long continuous exact-weight stress failed")
            logs.append(dict(source="continuous_three_task_stress",target=target,action="stress_checkpoint",
                actual_continuous_switches=state.switches,canonical_restores_inside_stress=0,
                mutable_weights_sha256=actual,switches=state.switches,
                grouped_decoder_metadata_bytes=state.auxiliary_bytes))
    return logs

def validate_cuda_decoder_vectors(torch):
    import numpy as np
    from rounding_patch import encode,bit_equal,PatchGroup
    cases=[]
    raw=np.full(4099,15000,dtype=np.uint16)
    target=raw.astype(np.int32)+np.resize(np.array([-1,0,1]),raw.size)
    target[-1]=40000
    cases.append(("two_bit_tail_escape",torch.from_numpy(raw.view(np.int16).copy()).view(torch.float16),
                  torch.from_numpy(target.astype(np.uint16).view(np.int16).copy()).view(torch.float16)))
    x=torch.ones(4097,dtype=torch.float16);y=x.clone();y[0]=-0.;y[-1]=1e-4
    cases.append(("sparse_signed_zero",x,y))
    x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
    cases.append(("all_raw_patterns",x,torch.roll(x,32767)))
    cases.append(("zero",torch.ones(17,dtype=torch.float16),torch.ones(17,dtype=torch.float16)))
    report=[];group_parameters={};group_patches={}
    for name,a,b in cases:
        p=encode(a,b).to("cuda:0");value=a.to("cuda:0")
        p.apply_(value,verify_input=True);torch.cuda.synchronize()
        if not bit_equal(value,b):raise ValueError("CUDA decoder vector mismatch: "+name)
        report.append(dict(case=name,mode=p.mode,elements=a.numel(),all_bits_equal=True))
        group_parameters[name]=a.to("cuda:0")
        group_patches[name]=p
    grouped=PatchGroup(group_parameters,group_patches)
    launches=grouped.apply_();torch.cuda.synchronize()
    for name,a,b in cases:
        if not bit_equal(group_parameters[name],b):raise ValueError("Grouped CUDA decoder vector mismatch: "+name)
    report.append(dict(case="mixed_group",mode="grouped_all_modes",all_bits_equal=True,
                       descriptor_bytes=grouped.nbytes,issued_decoder_launches=launches))
    return report

def admit(root,spec):
    import torch
    from rounding_patch import CanonicalSwitch
    e,c,p,o=helpers();contract=runtime(torch,e,p)
    decoder_checks=validate_cuda_decoder_vectors(torch)
    targets,_=e.target_cache_load(torch,Path(spec["source_result"]))
    model,factors,base=load_base(spec,torch,e,targets)
    params=dict(model.named_parameters())
    t0=time.perf_counter()
    cpu_banks,records,build_seconds=build_patches(torch,params,factors,base,targets)
    serialize_patches(root,cpu_banks,contract)
    banks={k:{t:{n:v.to("cuda:0") for n,v in pp.items()} for t,pp in tasks.items()} for k,tasks in cpu_banks.items()}
    patch_bytes=sum(r["bytes"] for r in records)
    payload=e.model_payload_bytes(model)+factors.bytes+patch_bytes
    # Compilation and correctness admission are separate from request timing.
    checks=validate_primitives(torch,params,factors,base,targets,banks)
    paths=audit_paths(torch,e,params,factors,base,targets,banks,contract)
    descriptor_bytes=max(r["grouped_decoder_metadata_bytes"] for r in paths)
    payload+=descriptor_bytes
    report=dict(status="ADMITTED_EXACT_CANONICAL_CLOSURE",contract=contract,
        codec_records=records,patch_bytes=patch_bytes,model_bytes=e.model_payload_bytes(model),
        factor_bytes=factors.bytes,grouped_decoder_metadata_bytes=descriptor_bytes,
        mutable_target_bytes=sum(v.numel()*v.element_size() for v in targets["mrpc"].values()),
        retained_tensor_bytes_before_kv=payload,logical_cap_bytes=spec["cap_bytes"],
        state_fits_before_kv=payload<spec["cap_bytes"],
        nonzero_correction_elements=sum(r["changed"] for r in records),decoder_checks=decoder_checks,
        primitive_checks=checks,full_weight_path_checks=paths,
        build_seconds=build_seconds,total_preparation_seconds=time.perf_counter()-t0,
        host_reference_tensor_bytes=sum(v.numel()*v.element_size() for t in targets.values() for v in t.values()),
        host_base_tensor_bytes=sum(v.numel()*v.element_size() for v in base.values()),
        fp32_anchor_retained_bytes_before_kv=e.model_payload_bytes(model)+factors.bytes+
            2*sum(v.numel()*v.element_size() for v in base.values()),
        all_target_device_cache_retained_bytes_before_kv=e.model_payload_bytes(model)+
            sum(v.numel()*v.element_size() for values in targets.values() for v in values.values()),
        patch_scope="deterministic FP16 operators on these verified fixed assets and backend",
        independent_test=False,new_method_performance_established=False)
    dump(root/"admission.json",report)
    if not report["state_fits_before_kv"]:return "NO_GO_STATE_MEMORY_BEFORE_KV"
    if not report["nonzero_correction_elements"]:return "NO_GO_NO_NONTRIVIAL_CORRECTION_ON_THESE_ASSETS"
    # Bounded functional smoke before the larger comparison.
    smoke=execute_trace(root,spec,"canonical_patch_cached",model,factors,targets,base,
                        contract,e,c,p,o,banks=banks,limit=32,write_tag="smoke")
    dump(root/"smoke_report.json",smoke)
    return "ADMITTED_AND_SMOKE_COMPLETED"

class TraceExecutor:
    def __init__(self,torch,e,c,p,o,spec,model,factors,targets,base,policy,contract,banks):
        from rounding_patch import CanonicalSwitch
        self.torch,self.e,self.c,self.p,self.o=torch,e,c,p,o
        self.spec,self.models,self.params=spec,[model],[dict(model.named_parameters())]
        self.factors,self.targets,self.base,self.policy=factors,targets,base,policy
        self.target_bytes={t:sum(v.numel()*v.element_size() for v in values.values()) for t,values in targets.items()}
        self.base_bytes=sum(v.numel()*v.element_size() for v in base.values())
        self.contract,self.banks=contract,banks
        self.task="mrpc";self.version="canonical:"+e.LOCAL_TARGET_HASHES["mrpc"]
        self.ticks=0;self.slot_tasks=["mrpc"];self.slot_ages=[0];self.slot=0
        self.switches=0;self.exact_installs=0;self.device_target=None
        self.device_base=None;self.anchor_groups=None;self.anchor_descriptor_bytes=0
        self.bank=collections.OrderedDict();self.bank_cap=int(spec.get("kv_bank_cap_bytes",192*MIB))
        self.evictions=0;self.logical_max=0
        self.fixed_bytes=e.model_payload_bytes(model)+(0 if factors is None else factors.bytes)
        self.per_token=2*model.config.num_hidden_layers*model.config.num_key_value_heads*getattr(
            model.config,"head_dim",model.config.hidden_size//model.config.num_attention_heads)*2
        if policy=="canonical_patch_cached":
            self.fixed_bytes+=sum(v.nbytes for tasks in banks.values() for values in tasks.values() for v in values.values())
        if policy=="device_base_patch_cached":
            from rounding_patch import PatchGroup
            size=sum(v.numel()*v.element_size() for v in base.values())
            self.fixed_bytes+=size+sum(v.nbytes for values in banks["merge"].values() for v in values.values())
            if self.fixed_bytes>spec["cap_bytes"]:raise MemoryError("GPU base/merge corrections exceed logical cap")
            self.device_base={n:v.to("cuda:0") for n,v in base.items()}
            self.anchor_groups={t:PatchGroup(self.params[0],values) for t,values in banks["merge"].items()}
            self.anchor_descriptor_bytes=sum(g.nbytes for g in self.anchor_groups.values())
            self.fixed_bytes+=self.anchor_descriptor_bytes
        if policy=="device_copy1_cached":
            size=sum(v.numel()*v.element_size() for v in targets["mrpc"].values())
            if self.fixed_bytes+size>spec["cap_bytes"]:raise MemoryError("Retained device target cache exceeds logical cap")
            self.device_target={n:v.to("cuda:0") for n,v in targets["mrpc"].items()}
            self.fixed_bytes+=size
        if policy.startswith("two_slots_copy_"):
            size=e.model_payload_bytes(model)
            if self.fixed_bytes+size>spec["cap_bytes"]:raise MemoryError("Two executable replicas exceed logical cap")
            second,_,extra=load_base(spec,torch,e,targets,need_factors=False)
            del extra
            self.models.append(second);self.params.append(dict(second.named_parameters()))
            self.fixed_bytes+=e.model_payload_bytes(second)
            self.slot_tasks.append(None);self.slot_ages.append(-1)
        if self.fixed_bytes>spec["cap_bytes"]:raise MemoryError("Retained model/factor/patch tensors exceed logical cap")
        self.payload_bytes=self.fixed_bytes
        self.state=None
        self.reset()

    @property
    def model(self):return self.models[self.slot]
    @property
    def bank_bytes(self):
        return self.e.tensor_storage_bytes([t for v in self.bank.values() for t in self.e.cache_tensors(v["cache"])])

    def reset(self):
        from rounding_patch import CanonicalSwitch
        self.bank.clear();self.evictions=0
        self.invalidated_entries=0;self.version_gc_events=0
        self.slot=0;self.task="mrpc";self.ticks=0
        self.slot_tasks=["mrpc"]+[None]*(len(self.models)-1)
        self.slot_ages=[0]+[-1]*(len(self.models)-1)
        self.version=("native:" if self.policy=="hf_native_cached" else "canonical:")+self.e.LOCAL_TARGET_HASHES["mrpc"]
        if self.policy=="hf_native_cached":
            self.models[0].set_adapter("mrpc");self.models[0].eval()
        else:self.e.install_host_target(self.torch,self.params[0],self.targets["mrpc"])
        self.switches=0;self.exact_installs=0
        self.addmm_calls=0;self.anchor_decoder_launches=0
        self.host_copy_bytes=0;self.device_copy_bytes=0
        if self.policy=="canonical_patch_cached":
            self.state=CanonicalSwitch(self.params[0],self.factors,self.banks["unmerge"],
                                       self.banks["merge"],"mrpc",self.contract)
        self.fixed_bytes=self.payload_bytes+(0 if self.state is None else self.state.auxiliary_bytes)
        if self.fixed_bytes>self.spec["cap_bytes"]:raise MemoryError("Grouped decoder descriptors exceed logical state cap")
        self.torch.cuda.synchronize()
        self.logical_max=self.fixed_bytes

    def install(self,target):
        if target not in TASKS:raise ValueError("Unsupported task")
        if self.policy.startswith("two_slots_copy_"):
            self.ticks+=1
            before=self.task
            if target in self.slot_tasks:
                self.slot=self.slot_tasks.index(target);action="slot_hit"
            else:
                self.slot=min(range(2),key=lambda i:self.slot_ages[i])
                self.e.install_host_target(self.torch,self.params[self.slot],self.targets[target])
                self.slot_tasks[self.slot]=target
                self.exact_installs+=1;action="host_copy"
                self.host_copy_bytes+=self.target_bytes[target]
            self.slot_ages[self.slot]=self.ticks
            self.task=target;self.version="canonical:"+self.e.LOCAL_TARGET_HASHES[target]
            self.switches+=int(before!=target)
            return action
        if target==self.task:return "resident"
        source=self.task
        if self.policy=="canonical_patch_cached":
            action=self.state.apply(target)
            self.addmm_calls+=len(self.factors.values[source])+len(self.factors.values[target])
            self.version="canonical:"+self.e.LOCAL_TARGET_HASHES[target]
        elif self.policy=="device_base_patch_cached":
            self.e.install_host_target(self.torch,self.params[0],self.device_base)
            self.factors.add(self.params[0],target,1.)
            self.anchor_decoder_launches+=self.anchor_groups[target].apply_()
            self.addmm_calls+=len(self.factors.values[target])
            self.device_copy_bytes+=self.base_bytes
            action="device_base_merge_correct"
            self.version="canonical:"+self.e.LOCAL_TARGET_HASHES[target]
        elif self.policy=="hf_native_cached":
            self.models[0].set_adapter(target);self.models[0].eval()
            action="native_select";self.version="native:"+self.e.LOCAL_TARGET_HASHES[target]
        elif self.policy.startswith("increment_"):
            self.factors.add(self.params[0],source,-1.)
            self.factors.add(self.params[0],target,1.)
            self.addmm_calls+=len(self.factors.values[source])+len(self.factors.values[target])
            action="increment";self.version="dirty:"+identity([self.version,source,target])
        else:
            cached=target=="mrpc" and self.device_target is not None
            self.e.install_host_target(self.torch,self.params[0],self.device_target if cached else self.targets[target])
            self.exact_installs+=1;action="device_copy" if cached else "host_copy"
            amount=self.target_bytes[target]
            if cached:self.device_copy_bytes+=amount
            else:self.host_copy_bytes+=amount
            self.version="canonical:"+self.e.LOCAL_TARGET_HASHES[target]
        self.task=target;self.switches+=1
        return action

    def collect_dead_versions(self):
        if not self.spec.get("collect_dead_kv_versions",False):return
        if self.policy!="increment_exact_kv":
            raise ValueError("Version GC is admitted only for the conservative increment baseline")
        # Dirty identities hash their entire transition history. After a switch,
        # an earlier dirty identity cannot recur in this fixed serving protocol.
        # Canonical/native identities and approximate same-adapter keys have
        # different reuse semantics and are never collected by this rule.
        dead=[key for key in self.bank if key[0]!=self.version]
        for key in dead:self.bank.pop(key)
        self.invalidated_entries+=len(dead)
        self.version_gc_events+=int(bool(dead))

    def trim(self,allowance):
        limit=max(0,min(self.bank_cap,allowance))
        while self.bank and self.bank_bytes>limit:
            self.bank.popitem(last=False);self.evictions+=1

    def live_check(self,*caches):
        all_tensors=[t for v in self.bank.values() for t in self.e.cache_tensors(v["cache"])]
        for cache in caches:all_tensors.extend(self.e.cache_tensors(cache))
        value=self.fixed_bytes+self.e.tensor_storage_bytes(all_tensors)
        self.logical_max=max(self.logical_max,value)
        if value>self.spec["cap_bytes"]:raise MemoryError("Observed model/factor/patch/KV bytes exceed logical cap")

    def serve(self,case,heads,tokenizer):
        torch=self.torch
        torch.cuda.synchronize();start=time.perf_counter_ns()
        head=heads[case["head_id"]]
        prefix=tokenizer.encode(head["text"]+case["query"],add_special_tokens=False)
        if prefix!=case["prefix"] or prefix[:len(head["tokens"])]!=head["tokens"]:
            raise ValueError("Effective request/prefix identity changed")
        labels=case["label_ids"]
        tokenized=time.perf_counter_ns()
        before=self.version
        actions=[]
        for t in list(case.get("via",[]))+[case["task"]]:
            actions.append(dict(target=t,action=self.install(t)))
        torch.cuda.synchronize();installed=time.perf_counter_ns()
        # No per-request weight scan is hidden in this timed serving path.
        use_cache=self.policy!="two_slots_copy_full"
        maximum=len(prefix)+max(map(len,labels.values()))
        # Bound the actual batched working KV geometry, with the same rule for
        # all methods. Activations/backend workspace are reported separately.
        allowance=self.spec["cap_bytes"]-self.fixed_bytes-len(labels)*maximum*self.per_token
        self.collect_dead_versions()
        self.trim(allowance if use_cache else 0)
        stamp=("same_adapter:"+case["task"]) if self.policy=="increment_old_kv" else self.version
        key=(stamp,case["head_id"])
        item=self.bank.get(key) if use_cache else None
        if item is not None:self.bank.move_to_end(key)
        template=None if item is None else item["cache"]
        hit=template is not None
        retained=use_cache and not hit and len(head["tokens"])*self.per_token<=min(self.bank_cap,max(0,allowance))
        origin=None if item is None else item["produced_by"]
        decided=time.perf_counter_ns()
        result,bundle=self.o.batch_score(torch,self.e,self.p,self.model,prefix,labels,
            len(head["tokens"]),template,retained,"cuda:0",self.live_check)
        torch.cuda.synchronize();scored=time.perf_counter_ns()
        admitted=False;stored=None
        if retained:
            needed=len(head["tokens"])*self.per_token
            self.trim(allowance-needed)
            stored=self.o.extract_batch_head(self.p,bundle)
            self.live_check(bundle["cache"],stored)
            self.bank[key]=dict(cache=stored,produced_by=self.version)
            self.trim(allowance)
            admitted=key in self.bank
        del bundle,stored
        torch.cuda.synchronize();end=time.perf_counter_ns()
        return dict(result,case_id=case["case_id"],request_key=case["request_key"],
            position=case["position"],task=case["task"],gold=case["gold"],head_id=case["head_id"],
            state_before=before,state_after=self.version,parameter_actions=actions,
            total_switches=self.switches,exact_installs=self.exact_installs,slot=self.slot,
            issued_addmm_calls=self.addmm_calls,
            issued_patch_decoder_launches=self.anchor_decoder_launches+(0 if self.state is None else self.state.decoder_launches),
            host_target_copy_bytes=self.host_copy_bytes,device_dense_copy_bytes=self.device_copy_bytes,
            cache_hit=hit,cache_admitted=admitted,cache_origin_version=origin,
            approximate_cache_used=bool(hit and origin!=self.version),
            cache_bytes=self.bank_bytes,cache_evictions=self.evictions,
            cache_invalidated_entries=self.invalidated_entries,
            cache_version_gc_events=self.version_gc_events,cache_bank_entries=len(self.bank),
            score_kind="cached" if hit else ("full_store" if retained else "full"),
            prefix_tokens=len(prefix),fixed_tensor_bytes=self.fixed_bytes,
            logical_live_max_bytes=self.logical_max,
            tokenize_ms=(tokenized-start)/1e6,parameter_ms=(installed-tokenized)/1e6,
            decision_ms=(decided-installed)/1e6,score_ms=(scored-decided)/1e6,
            admission_ms=(end-scored)/1e6,request_ms=(end-start)/1e6,
            started_monotonic_ns=start,finished_monotonic_ns=end)

def execute_trace(root,spec,policy,model,factors,targets,base,contract,e,c,p,o,
                  banks=None,limit=None,write_tag=None):
    import torch
    from transformers import AutoTokenizer
    from rounding_patch import bit_equal
    workload=read(ROOT/"input/workload.json")
    heads=workload["heads"]
    tokenizer=AutoTokenizer.from_pretrained(spec["model_dir"],local_files_only=True,trust_remote_code=False)
    server=TraceExecutor(torch,e,c,p,o,spec,model,factors,targets,base,policy,contract,banks)
    warm_checks=[]
    seen=set()
    for case in workload["warmup"]:
        if case["head_id"] in seen:continue
        seen.add(case["head_id"])
        first=server.serve(case,heads,tokenizer)
        tensors=[(str(i),t) for i,v in enumerate(server.bank.values()) for t in e.cache_tensors(v["cache"])]
        # Give each tensor a distinct name, retaining actual full tensor bytes.
        tensors=[(str(i),t) for i,(_,t) in enumerate(tensors)]
        before=e.tensor_digest(torch,tensors) if tensors else None
        again=server.serve(case,heads,tokenizer)
        after=e.tensor_digest(torch,tensors) if tensors else None
        if before!=after:raise ValueError("Immutable cache template was mutated during warmup")
        warm_checks.append(dict(head_id=case["head_id"],immutable=True,
                                observed_cached=again["cache_hit"],
                                repeated_prediction_equal=first["prediction"]==again["prediction"]))
    server.reset()
    torch.cuda.reset_peak_memory_stats()
    tag=write_tag or "trace"
    output=root/(tag+"_requests.jsonl")
    if output.exists():raise FileExistsError("Refusing to overwrite an earlier trace")
    sequence=workload["check"][:limit] if limit else workload["check"]
    audit=[]
    for case in sequence:
        row=server.serve(case,heads,tokenizer)
        append(output,row)
        if write_tag=="smoke":
            # Smoke timing is diagnostic only; this audit occurs AFTER timing.
            actual=e.tensor_digest(torch,[(n,server.params[server.slot][n]) for n in targets[case["task"]]])
            if actual!=e.LOCAL_TARGET_HASHES[case["task"]]:raise ValueError("Smoke full-weight audit failed")
            audit.append(dict(position=case["position"],mutable_weights_sha256=actual))
    torch.cuda.synchronize()
    final_hash=None
    if policy!="hf_native_cached":
        final_hash=e.tensor_digest(torch,[(n,server.params[server.slot][n]) for n in targets[server.task]])
        if policy in ["canonical_patch_cached","host_copy_cached","device_copy1_cached","device_base_patch_cached","two_slots_copy_full","two_slots_copy_cached"]:
            if final_hash!=e.LOCAL_TARGET_HASHES[server.task]:raise ValueError("Final canonical state hash differs")
    return dict(status="COMPLETED",policy=policy,requests=len(sequence),warmup_checks=warm_checks,
        final_task=server.task,final_mutable_weights_sha256=final_hash,
        fixed_tensor_bytes=server.fixed_bytes,logical_live_max_bytes=server.logical_max,
        allocator_peak_bytes=torch.cuda.max_memory_allocated(),
        reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        resident_snapshot=e.nvml_owned_snapshot(),switches=server.switches,
        exact_target_installs_inside_trace=server.exact_installs,
        smoke_full_weight_audit=audit,
        host_reference_tensor_bytes=sum(v.numel()*v.element_size() for t in targets.values() for v in t.values()),
        host_base_tensor_bytes=sum(v.numel()*v.element_size() for v in base.values()),
        retained_patch_bytes=0 if banks is None else sum(v.nbytes for tasks in banks.values() for values in tasks.values() for v in values.values()),
        grouped_decoder_metadata_bytes=server.anchor_descriptor_bytes+(0 if server.state is None else server.state.auxiliary_bytes),
        strict_physical_memory_guarantee=False,independent_test=False)

def worker(root,job):
    verify_release()
    import torch
    e,c,p,o=helpers()
    spec=read(root/"spec.json")
    group=os.getpgrp()
    e.require_idle(group=group,samples=1)
    stop=threading.Event()
    monitor=threading.Thread(target=e.monitor_stage,args=(stop,root/(job["tag"]+"_telemetry.jsonl"),group),daemon=True)
    monitor.start()
    try:
        preflight_dir=root/"preflight"/job["tag"]
        preflight_dir.mkdir(parents=True,exist_ok=True)
        e.preflight(dict(spec,scope="smoke"),preflight_dir)
        if job["kind"]=="admit":
            decision=admit(root,spec)
            dump(root/"admit_stage.json",dict(status="COMPLETED",decision=decision))
            return 0
        contract=runtime(torch,e,p)
        expected=read(root/"patch_metadata.json")["contract"]
        if contract!=expected:raise ValueError("Runtime differs from correction admission")
        policy=job["policy"]
        targets,_=e.target_cache_load(torch,Path(spec["source_result"]))
        need=policy.startswith("increment_") or policy in ["canonical_patch_cached","device_base_patch_cached"]
        model,factors,base=load_base(spec,torch,e,targets,native=policy=="hf_native_cached",need_factors=need)
        banks=None;validation=None
        if policy=="canonical_patch_cached":
            banks,_=deserialize_patches(root,"cuda:0",contract)
            # Admit on the actual allocation used for this timing process.
            validation=validate_primitives(torch,dict(model.named_parameters()),factors,base,targets,banks,repeats=1)
        elif policy=="device_base_patch_cached":
            banks,_=deserialize_patches(root,"cuda:0",contract,kinds=("merge",))
            validation=validate_primitives(torch,dict(model.named_parameters()),factors,base,targets,banks,repeats=1,kinds=("merge",))
        try:
            report=execute_trace(root,spec,policy,model,factors,targets,base,contract,e,c,p,o,
                                 banks=banks,limit=spec.get("limit"),write_tag=job["tag"])
            report["contract"]=contract;report["repeat"]=job["repeat"]
            report["allocation_primitive_validation"]=validation
        except (MemoryError,torch.OutOfMemoryError) as exc:
            report=dict(status="INFEASIBLE",policy=policy,repeat=job["repeat"],
                        reason=str(exc),logical_cap_bytes=spec["cap_bytes"])
        dump(root/(job["tag"]+"_report.json"),report)
        return 0
    finally:
        stop.set();monitor.join(timeout=3)

def percentile(v,q):
    s=sorted(v);x=(len(s)-1)*q;i=math.floor(x);j=math.ceil(x)
    return s[i]+(s[j]-s[i])*(x-i)

def summarize(root,jobs):
    metrics=[];flags=[];data={}
    for job in jobs:
        if job["kind"]!="requests":continue
        tag=job["tag"];report=read(root/(tag+"_report.json"))
        telemetry=rows(root/(tag+"_telemetry.jsonl"))
        if not telemetry or any(x.get("status")!="ok" or x.get("other_compute_process_count",0) for x in telemetry):
            flags.append(tag+": isolation telemetry incomplete or foreign GPU compute process observed")
        if report["status"]=="INFEASIBLE":
            metrics.append(dict(round=job["repeat"],policy=job["policy"],status="INFEASIBLE",reason=report["reason"]))
            continue
        rr=rows(root/(tag+"_requests.jsonl"))
        spec=read(root/"spec.json")
        expected=read(ROOT/"input/workload.json")["check"]
        if spec.get("limit"):expected=expected[:spec["limit"]]
        if len(rr)!=len(expected):raise ValueError("Incomplete request trace: "+tag)
        task="mrpc";expected_switches=0
        for case in expected:
            for target in list(case.get("via",[]))+[case["task"]]:
                expected_switches+=int(task!=target);task=target
        if report["switches"]!=expected_switches:raise ValueError("Actual parameter/adapter transitions differ from the frozen workload")
        for r,c in zip(rr,expected):
            if any(r[k]!=c[k] for k in ["position","request_key","case_id","task","gold"]):
                raise ValueError("Request identity changed")
            if r["finished_monotonic_ns"]<r["started_monotonic_ns"]:
                raise ValueError("Invalid timing order")
            elapsed=(r["finished_monotonic_ns"]-r["started_monotonic_ns"])/1e6
            if not math.isclose(elapsed,r["request_ms"],abs_tol=1e-8):raise ValueError("Timing boundaries differ")
        if len({r["request_key"] for r in rr})!=len(rr):raise ValueError("Repeated question keys in one trace")
        data[job["repeat"],job["policy"]]=rr
        m=dict(round=job["repeat"],policy=job["policy"],status="COMPLETED",n=len(rr),
               mean_ms=statistics.mean(r["request_ms"] for r in rr),
               p95_ms=percentile([r["request_ms"] for r in rr],.95),
               correct=sum(r["prediction"]==r["gold"] for r in rr),
               cache_hits=sum(r["cache_hit"] for r in rr),
               cache_admissions=sum(r["cache_admitted"] for r in rr),
               approximate_cache_uses=sum(r["approximate_cache_used"] for r in rr),
               fixed_tensor_bytes=report["fixed_tensor_bytes"],
               logical_peak_bytes=report["logical_live_max_bytes"],
               allocator_peak_bytes=report["allocator_peak_bytes"],
               reserved_peak_bytes=report["reserved_peak_bytes"],
               final_hash=report["final_mutable_weights_sha256"])
        metrics.append(m)
    comparisons=[]
    get=lambda i,p:next(m for m in metrics if m["round"]==i and m["policy"]==p)
    for i in range(read(root/"spec.json")["rounds"]):
        if (i,"host_copy_cached") not in data:raise ValueError("Canonical execution reference unavailable")
        reference={r["request_key"]:r for r in data[i,"host_copy_cached"]}
        proposed=get(i,"canonical_patch_cached")
        if proposed["status"]!="COMPLETED":
            flags.append("Proposed mechanism infeasible");continue
        prop=data[i,"canonical_patch_cached"]
        for policy in POLICIES:
            if (i,policy) not in data:continue
            rr=data[i,policy];m=get(i,policy)
            m["canonical_prediction_changes"]=sum(r["prediction"]!=reference[r["request_key"]]["prediction"] for r in rr)
            m["losses_vs_reference"]=sum(reference[r["request_key"]]["prediction"]==r["gold"] and r["prediction"]!=r["gold"] for r in rr)
            m["gains_vs_reference"]=sum(reference[r["request_key"]]["prediction"]!=r["gold"] and r["prediction"]==r["gold"] for r in rr)
            m["max_score_difference"]=max(abs(v-reference[r["request_key"]]["scores"][k]) for r in rr for k,v in r["scores"].items())
            if policy!="canonical_patch_cached":
                comparisons.append(dict(round=i,baseline=policy,
                    mean_reduction_fraction=1-proposed["mean_ms"]/m["mean_ms"],
                    p95_reduction_fraction=1-proposed["p95_ms"]/m["p95_ms"],
                    baseline_correct=m["correct"],proposed_correct=proposed["correct"]))
        if proposed["canonical_prediction_changes"] or proposed["approximate_cache_uses"]:
            flags.append("Proposed path did not preserve measured reference predictions/provenance")
    strong=["host_copy_cached","device_copy1_cached","device_base_patch_cached","two_slots_copy_full","two_slots_copy_cached","hf_native_cached"]
    wins=[r for r in comparisons if r["baseline"] in strong]
    # Engineering screen fixed before device data; not a significance test.
    qualifies=(not flags and len(wins)==len(strong)*read(root/"spec.json")["rounds"] and
               all(r["mean_reduction_fraction"]>=.05 and r["p95_reduction_fraction"]>=-.05 for r in wins))
    if flags:decision="REVIEW_REQUIRED_CORRECTNESS_OR_MEASUREMENT_FLAGS"
    elif qualifies:decision="GO_TO_INDEPENDENT_VALIDATION"
    else:decision="NO_GO_PILOT_PERFORMANCE_OR_FEASIBILITY"
    report=dict(status="COMPLETED_MECHANISM_PILOT",decision=decision,flags=flags,
        metrics=metrics,comparisons=comparisons,strong_controls=strong,
        engineering_screen=">=5% mean reduction and <=5% P95 increase in both rounds versus every listed strong control; not statistical significance",
        unique_development_questions=read(root/"spec.json").get("limit") or 192,
        repeated_questions_not_independent=True,project_wide_blind_test=False,
        theorem_scope="canonical weight reconstruction under admitted deterministic operator, not task accuracy or score equality across execution paths",
        controller_contribution_claim=False,submission_ready=False,
        next_if_go="Freeze this mechanism; confirm on genuinely independent input and Jetson/model-scale conditions before a submission claim")
    dump(root/"summary.json",report)
    return report

def archive_result(root):
    # Keep large, regenerable correction tensors on the server. The returned
    # archive contains their full layout/size/digest metadata and all raw logs.
    omitted=[]
    for p in [root/"patches.safetensors"]:
        if p.exists():omitted.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p),
            reason="Retained on server; reproducible from the pinned model/operator; excluded to keep upload small"))
    dump(root/"artifact_scope.json",dict(omitted_regenerable_payloads=omitted,all_request_rows_preserved=True))
    snapshot=root/"source_snapshot";snapshot.mkdir(exist_ok=True)
    for name in read(ROOT/"RELEASE_MANIFEST.json"):
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    shutil.copy2(ROOT/"RELEASE_MANIFEST.json",snapshot/"RELEASE_MANIFEST.json")
    selected=[p for p in sorted(root.rglob("*")) if p.is_file() and p.name!="manifest.json" and p.name!="patches.safetensors"]
    inv={str(p.relative_to(root)):sha(p) for p in selected}
    dump(root/"manifest.json",inv)
    dest=root.with_suffix(".tar.gz")
    with tarfile.open(dest,"w:gz") as tf:
        for p in selected+[root/"manifest.json"]:tf.add(p,arcname=root.name+"/"+str(p.relative_to(root)),recursive=False)
    return dest

def verify_release():
    inv=read(ROOT/"RELEASE_MANIFEST.json")
    for name,expected in inv.items():
        rel=Path(name)
        if rel.is_absolute() or ".." in rel.parts or not (ROOT/rel).is_file() or sha(ROOT/rel)!=expected:
            raise ValueError("Package file missing or changed: "+name)

def run_stage(root,job,timeout):
    e,_,_,_=helpers()
    e.require_idle(samples=2)
    tag=job["tag"]
    dump(root/"active_stage.json",job)
    log=root/(tag+".log")
    cmd=[sys.executable,str(Path(__file__).resolve()),"_worker","--worker-root",str(root)]
    started=time.perf_counter()
    with log.open("w") as f:
        process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            # Only stop the process group created for this one owned stage.
            e.stop_owned_process(process)
            raise
    if process.returncode:
        tail="\n".join(log.read_text(errors="replace").splitlines()[-14:])
        raise RuntimeError(tag+" failed; see "+str(log)+"\n"+tail)
    job["stage_wall_seconds_including_model_setup"]=time.perf_counter()-started

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["run","admit","_worker","inspect"])
    parser.add_argument("--output-dir",type=Path,default=Path.home()/"ftd_canonical_patch_outputs")
    parser.add_argument("--model-dir",type=Path)
    parser.add_argument("--taskpool-dir",type=Path)
    parser.add_argument("--reference-root",type=Path)
    parser.add_argument("--cap-mib",type=int,default=2560)
    parser.add_argument("--limit",type=int,choices=[32,64,192],default=192)
    parser.add_argument("--stage-timeout",type=int,default=3600)
    parser.add_argument("--worker-root",type=Path)
    args=parser.parse_args(argv)
    if args.mode=="_worker":
        root=args.worker_root
        try:return worker(root,read(root/"active_stage.json"))
        except BaseException:
            (root/"worker_error.txt").write_text(traceback.format_exc())
            traceback.print_exc();return 2
    verify_release()
    if args.mode=="inspect":
        print(json.dumps(dict(version=VERSION,policies=POLICIES,default_cap_mib=2560,
            input_questions=len(read(ROOT/"input/workload.json")["check"]),
            independent_test=False,source_paths=read(ROOT/"input/source_spec.json")),indent=2))
        return 0
    args.output_dir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root=Path(tempfile.mkdtemp(prefix="ftd_canonical_patch_"+stamp+"_",dir=args.output_dir))
    state=dict(version=VERSION,status="RUNNING",mode=args.mode,stages=[])
    rc=0
    try:
        source=read(ROOT/"input/source_spec.json")
        spec=dict(model_dir=str(args.model_dir or source["model_dir"]),
            taskpool_dir=str(args.taskpool_dir or source["taskpool_dir"]),
            source_result=str(args.reference_root or source["source_result"]),
            cap_bytes=args.cap_mib*MIB,limit=args.limit,rounds=2,
            development_only=True,project_wide_blind_test=False,
            input_sha256=sha(ROOT/"input/workload.json"),
            source_release_sha256=sha(ROOT/"RELEASE_MANIFEST.json"),version=VERSION)
        if args.cap_mib<1200:raise ValueError("Logical state cap too small for this model")
        for key in ["model_dir","taskpool_dir","source_result"]:
            if not Path(spec[key]).is_dir():raise FileNotFoundError(key+" missing: "+spec[key]+"; use the corresponding CLI override")
        if os.environ.get("CUDA_VISIBLE_DEVICES","0")!="0":
            raise ValueError("This pinned NVML/engine protocol requires physical GPU 0")
        dump(root/"spec.json",spec)
        dump(root/"status.json",state)
        job=dict(kind="admit",tag="admit")
        print("STAGE: full-weight admission and 32-request audited smoke",flush=True)
        run_stage(root,job,args.stage_timeout)
        state["stages"].append(job)
        decision=read(root/"admit_stage.json")["decision"]
        if decision!="ADMITTED_AND_SMOKE_COMPLETED":
            state.update(status=decision)
        elif args.mode=="admit":
            state.update(status="ADMISSION_AND_SMOKE_COMPLETED",next="Run the full pilot only if admitted memory and operator costs are acceptable")
        else:
            jobs=[]
            for repeat in range(2):
                order=POLICIES if repeat==0 else tuple(reversed(POLICIES))
                for policy in order:
                    jobs.append(dict(kind="requests",tag=f"r{repeat}_{policy}",policy=policy,repeat=repeat))
            for job in jobs:
                print("STAGE:",job["tag"],flush=True)
                run_stage(root,job,args.stage_timeout)
                state["stages"].append(job)
                dump(root/"status.json",state)
            report=summarize(root,jobs)
            state.update(status="COMPLETED_MECHANISM_PILOT",decision=report["decision"])
    except KeyboardInterrupt:
        rc=130;state.update(status="INTERRUPTED")
    except Exception as exc:
        rc=2;state.update(status="FAILED",error_type=type(exc).__name__,error=str(exc))
        (root/"error.txt").write_text(traceback.format_exc())
        print("Stopped:",str(exc),flush=True)
    finally:
        dump(root/"status.json",state)
        result=archive_result(root)
        print("STATUS:",state["status"],flush=True)
        print("SEND_THIS_ARCHIVE:",result,flush=True)
    return rc

if __name__=="__main__":sys.exit(main())
