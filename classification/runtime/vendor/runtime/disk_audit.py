"""CPU-only request and complete-weight admission audits."""
import math
import statistics as st
import disk_protocol as sp


def audit_records(records,cases,report,method):
    if not records or len(records)!=len(cases) or report['requests']!=len(cases) or report['status']!='COMPLETED':
        raise ValueError('Incomplete trace')
    previous=None;task='mrpc';switches=0
    for row,case in zip(records,cases):
        if any(row[k]!=case[k] for k in ('position','request_key','case_id','task','gold')):raise ValueError('Request identity differs')
        elapsed=(row['finished_monotonic_ns']-row['started_monotonic_ns'])/1e6
        if not math.isclose(elapsed,row['request_ms'],abs_tol=1e-8) or previous is not None and row['started_monotonic_ns']<previous:
            raise ValueError('Timing boundaries differ')
        previous=row['finished_monotonic_ns']
        parts=[row[k] for k in ('tokenize_ms','parameter_ms','decision_ms','score_ms','admission_ms')]
        if any(not math.isfinite(v) or v<0 for v in parts) or not math.isclose(sum(parts),row['request_ms'],abs_tol=1e-8):
            raise ValueError('Invalid timing components')
        scores=row['scores']
        if set(scores)!=sp.LABELS[case['task']] or any(not math.isfinite(v) for v in scores.values()):raise ValueError('Invalid scores')
        if row['prediction']!=max(sorted(scores),key=lambda k:scores[k]):raise ValueError('Score/tie rule differs')
        path=list(case.get('via',[]))+[case['task']]
        if [a['target'] for a in row['parameter_actions']]!=path:raise ValueError('Transition path differs')
        for target in path:
            switches+=int(target!=task);task=target
        if row['total_switches']!=switches:raise ValueError('Transition count differs from workload')
        expected_version=('native:' if method=='native' else 'canonical:')+report['target_hashes'][case['task']]
        if row['state_after']!=expected_version:raise ValueError('State identity differs')
        if row['approximate_cache_used'] or row['cache_hit'] and row['cache_origin_version']!=row['state_after']:raise ValueError('Wrong cache producer')
        if row['logical_live_max_bytes']>sp.CAP or report['logical_live_max_bytes']>sp.CAP:raise ValueError('Logical cap exceeded')
        if row['cache_bytes']>sp.KV_CAP:raise ValueError('Retained KV cap exceeded')
        if method in ('node','direct') and (row['host_target_copy_bytes'] or row['device_dense_copy_bytes']):raise ValueError('Dense timed canonical copy')
        if method in ('node','direct'):
            expected=report['mutable_projections']*switches*(1 if method=='direct' else 2)
            if row['issued_addmm_calls']!=expected:raise ValueError('Dynamic projection call count differs')
    if report['switches']!=switches or report['final_task']!=task:raise ValueError('Final transition accounting differs')
    if method!='native' and report['final_mutable_weights_sha256']!=report['target_hashes'][report['final_task']]:raise ValueError('Final canonical hash differs')
    return True


def audit_admission(report,method,admission,projections,limit):
    if not report['warmup_checks'] or any(not x['immutable'] for x in report['warmup_checks']):
        raise ValueError('Warmup immutable-cache check failed')
    if method not in ('node','direct'):return True
    v=report['validation'];primitive=v['primitives']
    count=primitive['full_projection_checks']
    count=sum(count.values()) if isinstance(count,dict) else count
    if not primitive['all_equal'] or count!=6*projections:raise ValueError('Full projection validation missing')
    checks=v['grouped_paths'];tasks=set(sp.TASKS)
    if not isinstance(checks,list) or len(checks)!=24:raise ValueError('Grouped edge validation missing')
    edges={(x.get('source'),x['target']) for x in checks if 'source' in x}
    if len(checks)!=24 or edges!={(a,b) for a in tasks for b in tasks if a!=b}:
        raise ValueError('Grouped edge validation missing')
    if [x['target'] for x in checks if x.get('cycle')]!=['qnli','agnews','mrpc']*6:
        raise ValueError('Continuous grouped path differs')
    if any(x['mutable_weights_sha256']!=report['target_hashes'][x['target']] for x in checks):
        raise ValueError('Grouped path target hash differs')
    if admission:
        smoke=report['smoke_full_weight_audit']
        work=sp.read(sp.HERE/'input/workload_source.json')['check'][:limit]
        if len(smoke)!=limit or any(x['position']!=case['position'] or
            x['mutable_weights_sha256']!=report['target_hashes'][case['task']] for x,case in zip(smoke,work)):
            raise ValueError('Full weight smoke audit missing')
    return True


def summarize(root):
    """Summarize audited stages without selecting a result by quality or speed."""
    from run_8b_disk import jobs,phase_done
    work=sp.read(root/'engine/input/workload.json');metrics=[];raw={};pids=[]
    reference=sp.rows(root/'stages/preflight_reference/trace_requests.jsonl')
    for job in [x for x in jobs() if x.get('phase')=='trace' and 'block' in x]:
        if not phase_done(root,job):raise ValueError('A formal phase is not committed')
        directory=root/'stages'/job['tag'];report=sp.read(directory/'report.json')
        rr=sp.rows(directory/'trace_requests.jsonl');raw[job['tag']]=rr;pids.append(report['worker_pid'])
        audit_records(rr,work['check'],report,job['method'])
        audit_admission(report,job['method'],False,252,384)
        if sp.read(directory/'audit.json')['status']!='AUDITED':raise ValueError('Missing stage audit')
        values=sorted(x['request_ms'] for x in rr);at=(len(values)-1)*.95;lo=int(at);hi=math.ceil(at)
        metrics.append(dict(job=job,mean_ms=st.mean(values),p95_ms=values[lo]+(values[hi]-values[lo])*(at-lo),
            parameter_mean_ms=st.mean(x['parameter_ms'] for x in rr),score_mean_ms=st.mean(x['score_ms'] for x in rr),
            correct=sum(x['prediction']==x['gold'] for x in rr),cache_hits=sum(x['cache_hit'] for x in rr),
            fixed_tensor_bytes=report['fixed_tensor_bytes'],logical_live_max_bytes=report['logical_live_max_bytes'],
            allocator_peak_bytes=report['allocator_peak_bytes'],reserved_peak_bytes=report['reserved_peak_bytes'],
            retained_patch_bytes=report['retained_patch_bytes'],worker_pid=report['worker_pid']))
    if len(metrics)!=18:raise ValueError('Expected eighteen formal stages')
    aggregates={}
    for method in ('direct','node','native'):
        mm=[x for x in metrics if x['job']['method']==method];groups=[raw[f'r{i}_{method}'] for i in range(6)]
        aggregates[method]=dict(mean_ms=st.mean(x['mean_ms'] for x in mm),
            mean_block_p95_ms=st.mean(x['p95_ms'] for x in mm),
            parameter_mean_ms=st.mean(x['parameter_mean_ms'] for x in mm),score_mean_ms=st.mean(x['score_mean_ms'] for x in mm),
            correct_each_block=[x['correct'] for x in mm],cache_hits_each_block=[x['cache_hits'] for x in mm],
            fixed_tensor_bytes=max(x['fixed_tensor_bytes'] for x in mm),
            maximum_logical_live_bytes=max(x['logical_live_max_bytes'] for x in mm),
            maximum_allocator_peak_bytes=max(x['allocator_peak_bytes'] for x in mm),
            prediction_changes_vs_device_reference=[sum(x['prediction']!=y['prediction'] for x,y in zip(group,reference)) for group in groups],
            max_candidate_score_difference_vs_reference=[max(abs(x['scores'][k]-y['scores'][k])
                for x,y in zip(group,reference) for k in x['scores']) for group in groups])
    result=dict(status='EIGHT_B_DISK_COMPARISON_COMPLETED',model='Qwen3-8B',stages=metrics,aggregates=aggregates,
        formal_requests=6912,formal_blocks=6,unique_questions=384,new_questions=0,new_independent_test=False,
        separate_worker_process_per_stage=True,observed_formal_worker_pids=pids,
        not_pooled_with_06b_timings=True,controller_claim=False,submission_ready=False,
        paired_node_minus_direct_ms=[next(x['mean_ms'] for x in metrics if x['job']['block']==i and x['job']['method']=='node')-
            next(x['mean_ms'] for x in metrics if x['job']['block']==i and x['job']['method']=='direct') for i in range(6)],
        preparation=dict(references=sp.read(root/'artifacts/references/metadata.json'),
            corrections={method:sp.read(root/'artifacts/patches'/method/'build_report.json') for method in ('node','direct')},
            one_time_costs_and_disk_loading_excluded_from_serving_latency=True),
        inference_logical_cap_bytes=sp.CAP,retained_kv_cap_bytes=sp.KV_CAP,strict_physical_memory_guarantee=False)
    sp.dump(root/'summary.json',result)
    return result
