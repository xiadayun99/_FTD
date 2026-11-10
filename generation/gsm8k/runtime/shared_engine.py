"""Convert admitted edge repairs, then validate and compare one GPU worker at a time."""
from pathlib import Path
import contextlib
import os
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/runtime'))
import disk_protocol as sp
import disk_store as ds
import disk_engine as legacy_engine
import run_8b_disk as legacy_driver
from shared_codec import TargetPatch,encode_target,decode_bits,payload_bytes


def operator_hashes():
    return {name:sp.sha(HERE/name) for name in ('shared_codec.py','shared_cuda.py','shared_group.py','shared_engine.py')}


def extended_contract(base):return dict(base,target_shared_operator_sha256=operator_hashes())


def make_patch(torch,target,encoded,predictor_hashes,target_sha256):
    return TargetPatch(tuple(target.shape),str(target.dtype),encoded['mode'],
        torch.from_numpy(encoded['packed'].copy()),torch.from_numpy(encoded['indices'].copy()),
        torch.from_numpy(encoded['values'].copy().view(np.int16)),dict(predictor_hashes),target_sha256)


def vectors(root,source):
    torch,core,e,c,p,o,contract=legacy_engine.runtime(legacy_driver.port_module(),source)
    import rounding_patch as rp
    from shared_group import SharedPatchGroup
    target=np.arange(65536,dtype=np.uint16)
    a=np.maximum(target.astype(np.int32)-1,0).astype(np.uint16)
    b=np.minimum(target.astype(np.int32)+1,65535).astype(np.uint16)
    fixtures=[('all_low2',target,a,b),
        ('odd_length',target[:4099],a[:4099],b[:4099]),
        ('escapes',target[:4099],np.bitwise_xor(a[:4099],np.where(np.arange(4099)%97==0,32768,0).astype(np.uint16)),b[:4099]),
        ('dense',target[:1027],np.bitwise_xor(target[:1027],32768),target[:1027])]
    parameters={};patches={};expected={};checks=[]
    for name,t,a,b in fixtures:
        groups=[np.flatnonzero(np.abs(t.astype(np.int32)-x.astype(np.int32))>1).astype(np.int32) for x in (a,b)]
        encoded=encode_target(t,groups)
        target_tensor=torch.from_numpy(t.copy().view(np.int16)).view(torch.float16)
        pred_tensors=[torch.from_numpy(x.copy().view(np.int16)).view(torch.float16) for x in (a,b)]
        patch=make_patch(torch,target_tensor,encoded,{str(i):rp.digest(value) for i,value in enumerate(pred_tensors)},rp.digest(target_tensor)).to('cuda:0')
        for i,prediction in enumerate(pred_tensors):
            value=prediction.cuda();patch.apply_(value,str(i),verify_input=True);torch.cuda.synchronize()
            if not rp.bit_equal(value,target_tensor):raise ValueError('Target-residue CUDA vector failed: '+name)
        parameters[name]=pred_tensors[0].cuda();patches[name]=patch;expected[name]=target_tensor
        checks.append(dict(name=name,mode=patch.mode,elements=t.size,bits_equal=True))
    SharedPatchGroup(parameters,patches).apply_();torch.cuda.synchronize()
    if any(not rp.bit_equal(parameters[name],value) for name,value in expected.items()):raise ValueError('Mixed grouped target decoder failed')
    path=root/'gpu_vectors.json';sp.dump(path,dict(status='VERIFIED',checks=checks,mixed_group=True,
        legacy_runtime_contract=contract,operator_sha256=operator_hashes(),eight_b_checked=False))
    return [path]


def convert(root,source):
    """Read legacy two-bit escape lists; no new model or GEMM is needed here."""
    import torch
    from safetensors import safe_open
    import bounded_io as bio
    port=legacy_driver.port_module();core,e,c,p,o=port.load_core()
    import rounding_patch as rp
    old=sp.read(source/'artifacts/patches/direct/patch_metadata.json')
    bankfile=source/'artifacts/patches/direct/patches.safetensors'
    if sp.sha(bankfile)!=old['payload_sha256']:raise ValueError('Legacy correction payload changed')
    directory=root/'artifacts/shared_target';directory.mkdir(parents=True,exist_ok=False)
    banks={'target':{}};records=[];began=time.perf_counter()
    with safe_open(str(bankfile),framework='pt',device='cpu') as legacy:
        for target in sp.TASKS:
            banks['target'][target]={};reference=source/'artifacts/references'/(target+'.safetensors')
            record=sp.read(source/'artifacts/references'/(target+'.json'))
            if sp.sha(reference)!=record['file_sha256']:raise ValueError('Reference file changed')
            incoming={src:old['banks']['edge'][src+'__to__'+target] for src in sp.TASKS if src!=target}
            names=sorted(next(iter(incoming.values())))
            if len(names)!=252 or any(set(values)!=set(names) for values in incoming.values()):raise ValueError('Incoming projection coverage differs')
            with safe_open(str(reference),framework='pt',device='cpu') as refs:
                for name in names:
                    tensor=refs.get_tensor(name)
                    if tensor.dtype!=torch.float16 or not tensor.is_contiguous():raise ValueError('FP16 contiguous target required')
                    raw=tensor.view(torch.int16).numpy().view(np.uint16)
                    digest=rp.digest(tensor);groups=[];hashes={}
                    for src,values in incoming.items():
                        entry=values[name];m=entry['metadata'];key=entry['key']
                        if m['mode']!='two_bit' or m['target_sha256']!=digest or tuple(m['shape'])!=tuple(tensor.shape):
                            raise ValueError('Converter requires the admitted two-bit incoming patch and exact target')
                        indices=legacy.get_tensor(key+'.indices').numpy()
                        values_raw=legacy.get_tensor(key+'.values').numpy().view(np.uint16)
                        if len(indices)!=m['escapes'] or not np.array_equal(raw.reshape(-1)[indices],values_raw):
                            raise ValueError('Legacy escape values differ from canonical target')
                        groups.append(indices.copy());hashes[src]=m['predictor_sha256']
                    encoded=encode_target(raw,groups)
                    patch=make_patch(torch,tensor,encoded,hashes,digest)
                    banks['target'][target][name]=patch
                    if patch.nbytes!=payload_bytes(encoded):raise ValueError('Shared byte count differs')
                    records.append(dict(target=target,name=name,**patch.metadata()))
                    sp.append(directory/'progress.jsonl',dict(projections=len(records),payload_bytes=sum(x['bytes'] for x in records),
                        memory=bio.memory_snapshot()))
                    if len(records)%36==0:
                        bio.trim_heap();print('SHARE:',len(records),'/756',flush=True)
    payload=sum(patch.nbytes for values in banks['target'].values() for patch in values.values())
    bound=sp.read(HERE/'analysis/verified_8b_audit.json')['proposed_target_sharing']['new_correction_upper_bytes']
    if payload>bound:raise ValueError('Shared payload exceeds the recorded all-two-bit bound')
    old_build=sp.read(source/'artifacts/patches/direct/build_report.json')
    if old_build['fixed_state_before_corrections_bytes']+payload>sp.CAP:raise MemoryError('Shared fixed state exceeds 32 GiB')
    encoding_seconds=time.perf_counter()-began
    meta=ds.save_banks(directory,banks,extended_contract(old['contract']))
    sp.dump(directory/'records.json',records)
    sp.dump(directory/'build_report.json',dict(status='CONVERTED_NOT_YET_GPU_ADMITTED',payload_bytes=payload,
        old_payload_bytes=old['tensor_bytes'],encoding_wall_seconds=encoding_seconds,wall_seconds=time.perf_counter()-began,
        mode_counts={mode:sum(x['mode']==mode for x in records) for mode in ('target_low2','dense')},
        projections=len(records),new_gemm_calibration_performed=False,source_predictor_hashes_preserved=True,
        source_payload_sha256=old['payload_sha256'],new_payload_sha256=meta['payload_sha256']))
    return [p for p in directory.rglob('*') if p.is_file()]


def load_shared(torch,root,contract):
    from safetensors.torch import load_file
    directory=root/'artifacts/shared_target';meta=sp.read(directory/'patch_metadata.json');file=directory/'patches.safetensors'
    if meta['contract']!=contract or sp.sha(file)!=meta['payload_sha256']:raise ValueError('Shared payload/contract changed')
    raw=load_file(str(file),device='cpu');banks={'target':{}};seen=set();total=0
    if set(meta['banks'])!={'target'} or set(meta['banks']['target'])!=set(sp.TASKS):raise ValueError('Shared bank coverage differs')
    for task,entries in meta['banks']['target'].items():
        banks['target'][task]={}
        for name,entry in entries.items():
            m=entry['metadata'];key=entry['key'];fields=[]
            if m['codec']!='target_low2_union_v1':raise ValueError('Shared codec version differs')
            for field in ('packed','indices','values'):
                label=key+'.'+field;seen.add(label);fields.append(raw[label])
            patch=TargetPatch(tuple(m['shape']),m['dtype'],m['mode'],*fields,m['predictor_hashes'],m['target_sha256']).to('cuda:0')
            if patch.nbytes!=m['bytes']:raise ValueError('Shared loaded byte count differs')
            banks['target'][task][name]=patch;total+=patch.nbytes
    if seen!=set(raw) or total!=meta['tensor_bytes']:raise ValueError('Shared payload index differs')
    return banks,meta


def validate_shared(session):
    import rounding_patch as rp
    from shared_group import SharedSwitch
    torch,e=session.torch,session.e;params=dict(session.model.named_parameters());count=0
    with torch.no_grad():
        for source in sp.TASKS:
            for target in sp.TASKS:
                if source==target:continue
                values,sign=session.factors.projection_values(source,target)
                for name,(a,b,scale) in values.items():
                    w=params[name];w.copy_(session.targets[source][name]);w.addmm_(b,a,beta=1.,alpha=sign*scale)
                    session.banks['target'][target][name].apply_(w,source,verify_input=True)
                    if not rp.bit_equal(w,session.targets[target][name]):raise ValueError('Shared full projection correction differs')
                    count+=1
                print('SHARED_EDGE_VERIFIED:',source,'->',target,flush=True)
    make=lambda initial:SharedSwitch(torch,params,session.factors,session.banks,initial,session.contract)
    paths=session.port.full_path_checks(torch,e,params,session.targets,make)
    common=e.tensor_digest(torch,[(n,v) for n,v in session.model.named_parameters() if n not in session.names]+
        [('buffer:'+n,v) for n,v in session.model.named_buffers()])
    if count!=1512 or common!=session.metadata['shared_state_sha256']:raise ValueError('Shared allocation admission differs')
    return dict(primitives=dict(all_equal=True,repeats=1,full_projection_checks=count),grouped_paths=paths,
        shared_state_sha256=common,incoming_predictor_hashes_checked=True)


def trace(root,source,method,tag,admission=False):
    import disk_audit as audit
    import numerical_guard as ng
    port=legacy_driver.port_module()
    base_method='native' if method=='native' else 'direct'
    session=legacy_engine.Session(source,port,base_method,with_banks=method!='shared')
    if method=='shared':
        from shared_group import attach_executor
        session.contract=extended_contract(session.contract)
        session.banks,_=load_shared(session.torch,root,session.contract)
        session.core.TraceExecutor=attach_executor(session.core.TraceExecutor)
    folder=root/'stages'/tag;folder.mkdir(parents=True,exist_ok=False)
    session.core.ROOT=root/'engine';limit=6 if admission else 384
    spec=sp.specification(source,base_method)
    if method=='shared':spec['canonical_transition_operator']='target_shared'
    with port.Telemetry(folder):
        port.require_no_observed_foreign(os.getpgrp())
        validation=validate_shared(session) if method=='shared' else session.validate(True) if method=='direct' else None
        sp.dump(folder/'pre_trace_validation.json',dict(validation=validation,source_result=str(source),
            runtime_contract=session.contract,one_comparison_model_per_process=True))
        context=ng.record_probe_cases(session.torch,session.core,folder) if admission else contextlib.nullcontext()
        with context:
            report=session.core.execute_trace(folder,spec,spec['policy'],session.model,session.factors,session.targets,session.base,
                session.contract,session.e,session.c,session.p,session.o,banks=session.banks,limit=limit,
                write_tag='smoke' if admission else 'trace')
        report.update(method=method,runtime_contract=session.contract,target_hashes=session.e.LOCAL_TARGET_HASHES,
            validation=validation,mutable_projections=252,logical_cap_bytes=sp.CAP,retained_kv_cap_bytes=sp.KV_CAP,
            worker_pid=os.getpid(),one_comparison_model_per_process=True,new_operator_sha256=operator_hashes())
        sp.dump(folder/'report.json',report)
    records=sp.rows(folder/('smoke_requests.jsonl' if admission else 'trace_requests.jsonl'))
    cases=sp.read(root/'engine/input/workload.json')['check'][:limit]
    audit_method='direct' if method=='shared' else method
    audit.audit_records(records,cases,report,audit_method)
    audit.audit_admission(report,audit_method,admission,252,limit)
    samples=sp.rows(folder/'telemetry.jsonl')
    if not samples or any(x['observation']['foreign_pids'] for x in samples):raise ValueError('Missing or contaminated telemetry')
    sp.dump(folder/'audit.json',dict(status='AUDITED',requests=len(records),telemetry_samples=len(samples),
        isolation_enumeration_verified=all(x['observation']['process_enumeration_verified'] for x in samples)))
    return [p for p in folder.rglob('*') if p.is_file()]


def worker(root,source,job):
    if job['phase']=='vectors':return vectors(root,source)
    if job['phase']=='convert':return convert(root,source)
    if job['phase']=='trace':return trace(root,source,job['method'],job['tag'],job.get('admission',False))
    raise ValueError('Unknown target-shared phase')
