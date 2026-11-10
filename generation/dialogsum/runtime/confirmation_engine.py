"""Reuse the admitted numerical implementation; only inputs/baselines change."""
import contextlib,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/runtime'))
import disk_protocol as sp
import disk_engine as engine
import run_8b_disk as legacy_driver
import disk_audit as audit

def audit_mode(method):return 'direct' if method=='shared' else 'reference' if method=='host_copy' else method

def trace(root,source,shared,job):
    import shared_engine as se
    import numerical_guard as ng
    method=job['method'];base_method=audit_mode(method)
    session=engine.Session(source,legacy_driver.port_module(),base_method,with_banks=method!='shared')
    if method=='shared':
        from shared_group import attach_executor
        session.contract=se.extended_contract(session.contract)
        session.banks,_=se.load_shared(session.torch,shared,session.contract)
        session.core.TraceExecutor=attach_executor(session.core.TraceExecutor)
    folder=root/'stages'/job['tag'];folder.mkdir(parents=True,exist_ok=False)
    session.core.ROOT=root/'engine';spec=sp.specification(source,base_method)
    if method=='shared':spec['canonical_transition_operator']='target_shared'
    with session.port.Telemetry(folder):
        session.port.require_no_observed_foreign(os.getpgrp())
        validation=se.validate_shared(session) if method=='shared' else session.validate(True) if method in ('direct','node') else None
        sp.dump(folder/'pre_trace_validation.json',dict(validation=validation,runtime_contract=session.contract,
            one_comparison_model_per_process=True,source_result=str(source),shared_result=str(shared)))
        context=ng.record_probe_cases(session.torch,session.core,folder) if job.get('reference') else contextlib.nullcontext()
        with context:
            report=session.core.execute_trace(folder,spec,spec['policy'],session.model,session.factors,session.targets,session.base,
                session.contract,session.e,session.c,session.p,session.o,banks=session.banks,limit=384,write_tag='trace')
        report.update(method=method,runtime_contract=session.contract,target_hashes=session.e.LOCAL_TARGET_HASHES,
            validation=validation,mutable_projections=252,logical_cap_bytes=sp.CAP,retained_kv_cap_bytes=sp.KV_CAP,
            worker_pid=os.getpid(),one_comparison_model_per_process=True,serving_operator_sha256=se.operator_hashes(),
            workload_sha256=sp.sha(root/'engine/input/workload.json'),fresh_inputs_relative_to_recorded_ledger=True,
            new_tasks=0,independent_test=False)
        sp.dump(folder/'report.json',report)
    records=sp.rows(folder/'trace_requests.jsonl');work=sp.read(root/'engine/input/workload.json')
    audit.audit_records(records,work['check'],report,base_method)
    audit.audit_admission(report,base_method,False,252,384)
    samples=sp.rows(folder/'telemetry.jsonl')
    if not samples or any(x['observation']['foreign_pids'] or not x['observation']['process_enumeration_verified'] for x in samples):
        raise ValueError('GPU process observation incomplete or contaminated')
    sp.dump(folder/'audit.json',dict(status='AUDITED',requests=len(records),telemetry_samples=len(samples)))
    return [p for p in folder.rglob('*') if p.is_file()]

def worker(root,source,shared,job):
    if job['phase']=='prepare':
        from confirmation_inputs import prepare
        return prepare(root,source)
    if job['phase']=='trace':return trace(root,source,shared,job)
    raise ValueError('Unknown confirmation phase')
