"""Reconstruct every formal record; keep disagreements and repeated-input limits."""
import math,statistics as st,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor/runtime'))
import disk_protocol as sp
import disk_audit as da
from confirmation_protocol import METHODS
from confirmation_engine import audit_mode

def metrics(rows,report):
    values=sorted(x['request_ms'] for x in rows);at=(len(values)-1)*.95;lo=int(at);hi=math.ceil(at)
    return dict(mean_ms=st.mean(values),p95_ms=values[lo]+(values[hi]-values[lo])*(at-lo),
        parameter_mean_ms=st.mean(x['parameter_ms'] for x in rows),score_mean_ms=st.mean(x['score_ms'] for x in rows),
        correct=sum(x['prediction']==x['gold'] for x in rows),cache_hits=sum(x['cache_hit'] for x in rows),
        fixed_tensor_bytes=report['fixed_tensor_bytes'],logical_live_max_bytes=report['logical_live_max_bytes'],
        allocator_peak_bytes=report['allocator_peak_bytes'],retained_patch_bytes=report['retained_patch_bytes'])

def summarize(root,jobs,phase_done):
    if not all(phase_done(root,job) for job in jobs):raise ValueError('Incomplete confirmation stages')
    work=sp.read(root/'engine/input/workload.json');raw={};stages=[]
    if len({(x['task'],x['source_id']) for x in work['check']})!=384:raise ValueError('Unexpected unique question count')
    for job in [x for x in jobs if x['phase']=='trace']:
        folder=root/'stages'/job['tag'];report=sp.read(folder/'report.json');rows=sp.rows(folder/'trace_requests.jsonl')
        da.audit_records(rows,work['check'],report,audit_mode(job['method']))
        da.audit_admission(report,audit_mode(job['method']),False,252,384)
        if sp.read(folder/'audit.json')['status']!='AUDITED':raise ValueError('Stage not audited')
        raw[job['tag']]=rows
        if 'block' in job:stages.append(dict(job=job,**metrics(rows,report)))
    if len(stages)!=50:raise ValueError('Expected fifty formal stages')
    reference=raw['preflight_reference'];aggregate={}
    cache_fields=('cache_hit','cache_admitted','cache_bytes','cache_evictions','cache_bank_entries')
    for method in METHODS:
        mm=[x for x in stages if x['job']['method']==method];groups=[raw[f'r{i}_{method}'] for i in range(10)]
        aggregate[method]=dict(mean_ms=st.mean(x['mean_ms'] for x in mm),mean_block_p95_ms=st.mean(x['p95_ms'] for x in mm),
            parameter_mean_ms=st.mean(x['parameter_mean_ms'] for x in mm),score_mean_ms=st.mean(x['score_mean_ms'] for x in mm),
            correct_each_block=[x['correct'] for x in mm],cache_hits_each_block=[x['cache_hits'] for x in mm],
            fixed_tensor_bytes=max(x['fixed_tensor_bytes'] for x in mm),retained_patch_bytes=max(x['retained_patch_bytes'] for x in mm),
            allocator_peak_bytes=max(x['allocator_peak_bytes'] for x in mm),
            maximum_score_difference_vs_reference=[max(abs(x['scores'][k]-y['scores'][k]) for x,y in zip(rr,reference) for k in x['scores']) for rr in groups],
            prediction_changes_vs_reference=[sum(x['prediction']!=y['prediction'] for x,y in zip(rr,reference)) for rr in groups],
            cache_behavior_mismatches_vs_reference=[sum(any(x[k]!=y[k] for k in cache_fields) for x,y in zip(rr,reference)) for rr in groups],
            prediction_changes_between_blocks=[sum(x['prediction']!=y['prediction'] for x,y in zip(rr,groups[0])) for rr in groups],
            task_accuracy_first_block={t:dict(correct=sum(x['prediction']==x['gold'] for x in groups[0] if x['task']==t),
                questions=sum(x['task']==t for x in groups[0])) for t in sp.TASKS},
            total_timed_host_target_copy_bytes_each_block=[rr[-1]['host_target_copy_bytes'] for rr in groups])
    result=dict(status='FRESH_CONFIRMATION_COMPLETED',aggregates=aggregate,stages=stages,
        workload_sha256=sp.sha(root/'engine/input/workload.json'),new_unique_questions=384,new_tasks=0,
        formal_requests=19200,formal_blocks=10,selection=sp.read(root/'data/selection_report.json'),
        project_wide_blind_test=False,official_test_benchmark=False,new_independent_test=False,
        canonical_score_equality_observed=all(max(aggregate[m]['maximum_score_difference_vs_reference'])==0 for m in ('shared','direct','node')),
        repeated_blocks_are_new_accuracy_examples=False,previous_timings_pooled=False,
        no_method_or_input_selected_by_quality=True,submission_ready=False)
    sp.dump(root/'summary.json',result);return result
