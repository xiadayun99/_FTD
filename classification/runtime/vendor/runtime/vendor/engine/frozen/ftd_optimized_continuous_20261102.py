#!/usr/bin/env python3
"""Matched optimized continuous-state controls; no presumed policy benefit."""
import argparse
import collections
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback

VERSION='ftd_optimized_continuous_20261102_v1'
PROFILE_SHA='57f4458daaeb3cdc0e263d0a2f8f1650957f51529303c19a9dd6154a7252ee69'
CONTINUOUS_SHA='e091ef792debb959bd35be8411f07937cbd6f59e4d7dee1f59c985ecb82e2c15'
ENGINE_SHA='0101cdad4d9b887919e09d2004c6bdf7eb68556039c06dfd98ca32b5c46f18fb'
DEFAULT_PROFILE=Path('./assets/legacy/ftd_kv_bottleneck_outputs/ftd_kv_bottleneck_20261102T042538Z_8vflgvzl')
SCRIPT='ftd_optimized_continuous_20261102.py'
POLICIES=('fixed_inc','simple_rule','fixed_exact','hf_native','length_rule','joint')
BASE_CALIBRATION=POLICIES[:4]
TASKS=('mrpc','qnli')
LIVE_CAP=2048*2**20
BANK_CAP=192*2**20


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_rows(path,values):
    with Path(path).open('w') as f:
        for value in values:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def helpers(root):
    path=root/'frozen_profile.py'
    if sha(path)!=PROFILE_SHA:raise ValueError('Frozen optimized scorer identity changed')
    spec=importlib.util.spec_from_file_location('optimized_frozen_profile',path)
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    m,e=p.helpers(root)
    return p,m,e


def batch_score(torch,e,p,model,prefix,labels,head_length,template=None,retain_head=False,device='cuda:0',check=None):
    """Same batched candidate objective for every control, with optional KV output.

    The returned bundle is transient. Extracting an immutable single-row prefix
    is separately timed by the serving loop; no second head prefill is needed.
    """
    names=sorted(labels)
    suffix=prefix[head_length:] if template is not None else prefix
    if not suffix or any(not labels[n] for n in names):raise ValueError('Empty suffix/label')
    if template is not None and template.get_seq_length()!=head_length:raise ValueError('Wrong prefix cache length')
    seqs=[suffix+labels[n] for n in names]
    width=max(map(len,seqs));keep=max(map(len,labels.values()))+1
    tail=e.tail_argument(model)
    if tail is None:raise ValueError('Tail-logit API required')
    with torch.inference_mode():
        pad=model.config.pad_token_id if model.config.pad_token_id is not None else 0
        ids=torch.full((len(names),width),pad,device=device,dtype=torch.long)
        newmask=torch.zeros_like(ids)
        for i,seq in enumerate(seqs):
            ids[i,width-len(seq):]=torch.tensor(seq,device=device)
            newmask[i,width-len(seq):]=1
        past=head_length if template is not None else 0
        mask=torch.cat([torch.ones((len(names),past),device=device,dtype=torch.long),newmask],dim=-1)
        positions=(mask.cumsum(-1)-1).clamp(min=0)[:,past:]
        working=p.fork_cache(template,'shared',len(names)) if template is not None else None
        if check:check(working)
        out=model(input_ids=ids,attention_mask=mask,position_ids=positions,past_key_values=working,
                  use_cache=template is not None or retain_head,**{tail:keep})
        cache=getattr(out,'past_key_values',None)
        if check:check(working,cache)
        scores={}
        for i,name in enumerate(names):
            targets=labels[name];begin=keep-len(targets)-1
            logits=out.logits[i,begin:begin+len(targets)].float()
            token_ids=torch.tensor(targets,device=device)
            scores[name]=float(torch.log_softmax(logits,dim=-1).gather(1,token_ids[:,None]).mean().item())
        bundle=None
        if retain_head and template is None:
            if cache is None:raise ValueError('Full scoring failed to return requested KV')
            row=max(range(len(names)),key=lambda i:len(seqs[i]))
            bundle=dict(cache=cache,row=row,offset=width-len(seqs[row]),head_length=head_length)
    return p.finalize(scores),bundle


def extract_batch_head(p,bundle,row=None,offset=None):
    cache=bundle['cache'];p.validate_dynamic(cache)
    row=bundle['row'] if row is None else row
    offset=bundle['offset'] if offset is None else offset
    length=bundle['head_length']
    if length<=0 or offset<0 or offset+length>cache.get_seq_length():raise ValueError('Invalid batch-prefix slice')
    result=copy.copy(cache);result.layers=[copy.copy(layer) for layer in cache.layers]
    for layer in result.layers:
        if not 0<=row<layer.keys.shape[0]:raise ValueError('Invalid batch row')
        layer.keys=layer.keys[row:row+1,:,offset:offset+length,:].clone().contiguous()
        layer.values=layer.values[row:row+1,:,offset:offset+length,:].clone().contiguous()
    if result.get_seq_length()!=length:raise ValueError('Extracted prefix length differs')
    return result


def choose(policy,state,bank,head,costs=None,threshold=None):
    if policy=='fixed_inc':return 'inc_full',False
    if policy in ('simple_rule','hf_native'):return 'inc_auto',True
    if policy=='fixed_exact':return 'exact_auto',True
    if policy=='length_rule':
        if threshold is None:raise ValueError('Length rule threshold must be frozen')
        return ('inc_auto',True) if len(head['tokens'])>=threshold else ('inc_full',False)
    if policy!='joint':raise ValueError('Unknown policy')
    c=costs[head['task']+':'+head['context']]
    store=c['cached_ms']<c['full_ms'] and c['admission_extra_ms']<2*(c['full_ms']-c['cached_ms'])
    same=state.task==head['task']
    inc=0. if same else costs['increment_ms']
    restore=0. if state.version==state.canonical(head['task']) else costs['restore_ms']
    current_hit=same and bank.get(state.version,head,touch=False) is not None
    canonical_hit=bank.get(state.canonical(head['task']),head,touch=False) is not None
    # Admission is an explicit two-future-hit heuristic; no future trace is read.
    options=[('inc_full',inc+(c['full_store_ms'] if store and not current_hit else c['full_ms']))]
    if current_hit:options.append(('keep_cached',c['cached_ms']))
    if canonical_hit:options.append(('exact_cached',restore+c['cached_ms']))
    options.append(('exact_full',restore+(c['full_store_ms'] if store and not canonical_hit else c['full_ms'])))
    priority={'keep_cached':0,'inc_full':1,'exact_cached':2,'exact_full':3}
    return min(options,key=lambda x:(x[1],priority[x[0]]))[0],store


class Server:
    def __init__(self,p,m,e,torch,model,factors,targets,tokenizer,policy,costs=None,threshold=None,
                 device='cuda:0',live_cap=LIVE_CAP,bank_cap=BANK_CAP):
        self.p,self.m,self.e,self.torch=p,m,e,torch
        self.model,self.tokenizer,self.policy=model,tokenizer,policy
        self.costs,self.threshold,self.device=costs,threshold,device
        self.state=m.WeightState(e,torch,model,factors,targets,policy=='hf_native',device)
        self.bank=m.CacheBank(e,bank_cap)
        self.model_bytes=e.model_payload_bytes(model)+(factors.bytes if factors is not None else 0)
        cfg=model.config;size=next(model.parameters()).element_size()
        self.per_token=2*cfg.num_hidden_layers*cfg.num_key_value_heads*getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)*size
        self.live_cap,self.live_max=live_cap,self.model_bytes

    def check(self,*caches):
        tensors=[]
        for cache in list(self.bank.data.values())+list(caches):tensors.extend(self.e.cache_tensors(cache))
        amount=self.model_bytes+self.e.tensor_storage_bytes(tensors)
        self.live_max=max(self.live_max,amount)
        if amount>self.live_cap:raise ValueError('Observed logical model/factor/KV tensors exceed cap')

    def serve(self,head,query,labels):
        self.state.sync();start=time.perf_counter_ns()
        prefix=self.tokenizer.encode(head['text']+query,add_special_tokens=False)
        if prefix[:len(head['tokens'])]!=head['tokens']:raise ValueError('Tokenized shared prefix changed')
        tokenized=time.perf_counter_ns()
        maximum=len(prefix)+max(map(len,labels.values()))
        allowance=self.live_cap-self.model_bytes-4*maximum*self.per_token-64*2**20
        if allowance<0:raise ValueError('Working-cache budget rejected before allocation')
        self.bank.trim(allowance)
        before=self.state.version
        route,store=choose(self.policy,self.state,self.bank,head,self.costs,self.threshold)
        decided=time.perf_counter_ns()
        action=self.state.apply(head['task'],exact=route.startswith('exact'))
        installed=time.perf_counter_ns()
        template=None if route in ('inc_full','exact_full') else self.bank.get(self.state.version,head)
        if route in ('keep_cached','exact_cached') and template is None:raise ValueError('Selected cache unavailable')
        hit=template is not None
        retain=store and not hit and len(head['tokens'])*self.per_token<=min(self.bank.cap,allowance)
        result,bundle=batch_score(self.torch,self.e,self.p,self.model,prefix,labels,len(head['tokens']),
                                  template,retain,self.device,self.check)
        self.state.sync();scored=time.perf_counter_ns()
        admitted=False;cache=None
        if retain:
            expected=len(head['tokens'])*self.per_token
            self.bank.trim(allowance-expected)
            cache=extract_batch_head(self.p,bundle)
            self.check(bundle['cache'],cache)
            admitted=self.bank.put(self.state.version,head,cache,allowance)
        del bundle,cache
        self.state.sync();end=time.perf_counter_ns()
        return dict(result,route=route,score_kind='cached' if hit else ('full_store' if retain else 'full'),
            parameter_action=action,state_before=before,state_after=self.state.version,
            cache_hit=hit,cache_admitted=admitted,cache_bytes=self.bank.bytes,cache_evictions=self.bank.evictions,
            head_tokens=len(head['tokens']),prefix_tokens=len(prefix),logical_live_max_bytes=self.live_max,
            incremental_switches_since_exact=self.state.since_exact,total_switches=self.state.switches,total_exact_restores=self.state.restores,
            tokenize_ms=(tokenized-start)/1e6,decision_ms=(decided-tokenized)/1e6,
            parameter_ms=(installed-decided)/1e6,score_ms=(scored-installed)/1e6,
            store_ms=(end-scored)/1e6,request_ms=(end-start)/1e6)


def make_workload(m,previous,mode):
    w=copy.deepcopy(m.read_workload(previous));w.pop('signature')
    w['source_evaluation_queries']=len(w['evaluation'])
    if mode=='verify':
        seen=collections.Counter();selected=[]
        for c in w['evaluation']:
            key=c['task'],c['context'],c['phase']
            if seen[key]<2:selected.append(c)
            seen[key]+=1
        w['evaluation']=selected
    for i,c in enumerate(w['evaluation']):c['position']=i
    w['mode']=mode;w['reused_previously_observed_evaluation']=True
    w['unique_evaluation_requests']=len({c['request_key'] for c in w['evaluation']})
    w['selection']='Original frozen pilot trace; verify keeps first two per task/context/phase, without correctness or margin selection'
    w['signature']=identity(w)
    return w


def threshold_candidates(workload):
    lengths=sorted({len(workload['heads'][c['head_id']]['tokens']) for c in workload['calibration']})
    return [0]+[(a+b)//2+1 for a,b in zip(lengths,lengths[1:])]+[lengths[-1]+1]


def warmup_validate(p,m,e,torch,model,targets,workload,native,device='cuda:0'):
    checks=[];seen=set();params=dict(model.named_parameters())
    for c in workload['calibration']:
        key=c['task'],c['context']
        if key in seen:continue
        seen.add(key);h=workload['heads'][c['head_id']]
        if native:model.set_adapter(c['task']);model.eval()
        else:e.install_host_target(torch,params,targets[c['task']])
        full,_=batch_score(torch,e,p,model,c['prefix'],c['label_ids'],len(h['tokens']),device=device)
        retained,bundle=batch_score(torch,e,p,model,c['prefix'],c['label_ids'],len(h['tokens']),retain_head=True,device=device)
        cache=extract_batch_head(p,bundle)
        before=e.tensor_digest(torch,[(str(i),t) for i,t in enumerate(e.cache_tensors(cache))])
        cached,_=batch_score(torch,e,p,model,c['prefix'],c['label_ids'],len(h['tokens']),cache,device=device)
        after=e.tensor_digest(torch,[(str(i),t) for i,t in enumerate(e.cache_tensors(cache))])
        if before!=after:raise ValueError('Shared template mutated during preflight')
        checks.append(dict(task=c['task'],context=c['context'],immutable=True,
            prediction_matches=full['prediction']==retained['prediction']==cached['prediction'],
            max_abs_score_difference=max(abs(result['scores'][k]-full['scores'][k]) for result in (retained,cached) for k in full['scores'])))
    return checks


def worker(root):
    import torch
    from transformers import AutoTokenizer
    p,m,e=helpers(root);p.cache_runtime_contract()
    spec,active=read(root/'spec.json'),read(root/'active_stage.json')
    w=m.read_workload(root);torch.manual_seed(0);e.torch_runtime(torch)
    policy=active.get('policy','fixed_inc')
    model,factors,targets=m.model_for_policy(e,torch,spec,policy)
    tokenizer=AutoTokenizer.from_pretrained(spec['model_dir'],local_files_only=True,trust_remote_code=False)
    checks=warmup_validate(p,m,e,torch,model,targets,w,policy=='hf_native')
    tag=active['tag'];dump(root/(tag+'_cache_validation.json'),checks)
    plan=read(root/'plan.json') if policy in ('joint','length_rule') else None
    costs=plan['costs'] if plan else None;threshold=plan['length_threshold_tokens'] if plan else None
    data=w['calibration' if active['kind'] in ('calibration','sweep') else 'evaluation']
    e.require_idle(os.getpgrp());torch.cuda.reset_peak_memory_stats()
    output=[];runtime=[]
    if active['kind']=='sweep':
        for repeat in (0,1):
            order=threshold_candidates(w)
            if repeat:order.reverse()
            for threshold in order:
                server=Server(p,m,e,torch,model,factors,targets,tokenizer,'length_rule',threshold=threshold)
                for c in data:
                    e.require_idle(os.getpgrp())
                    result=server.serve(w['heads'][c['head_id']],c['query'],c['label_ids'])
                    output.append(dict(position=c['position'],request_key=c['request_key'],case_id=c['case_id'],
                        task=c['task'],context=c['context'],phase=c['phase'],head_id=c['head_id'],gold=c['gold'],
                        policy='length_rule',threshold=threshold,calibration_repeat=repeat,**result))
                runtime.append(dict(threshold=threshold,calibration_repeat=repeat,logical_live_max_bytes=server.live_max))
                write_rows(root/(tag+'_requests.jsonl'),output)
                del server
    elif active['kind']=='reference':
        server=Server(p,m,e,torch,model,factors,targets,tokenizer,'fixed_inc');seen=set()
        for c in data:
            if c['request_key'] in seen:continue
            seen.add(c['request_key']);e.require_idle(os.getpgrp())
            server.state.apply(c['task'],exact=True)
            result,_=batch_score(torch,e,p,model,c['prefix'],c['label_ids'],
                len(w['heads'][c['head_id']]['tokens']),device='cuda:0',check=server.check)
            output.append(dict(request_key=c['request_key'],gold=c['gold'],**result))
        runtime.append(dict(logical_live_max_bytes=server.live_max))
    else:
        server=Server(p,m,e,torch,model,factors,targets,tokenizer,policy,costs,threshold)
        for c in data:
            e.require_idle(os.getpgrp())
            result=server.serve(w['heads'][c['head_id']],c['query'],c['label_ids'])
            output.append(dict(position=c['position'],request_key=c['request_key'],case_id=c['case_id'],
                head_id=c['head_id'],task=c['task'],context=c['context'],phase=c['phase'],gold=c['gold'],policy=policy,**result))
            if len(output)%8==0:
                write_rows(root/(tag+'_requests.jsonl'),output)
                print(tag,'completed requests:',len(output),flush=True)
        runtime.append(dict(logical_live_max_bytes=server.live_max))
    e.require_idle(os.getpgrp());write_rows(root/(tag+'_requests.jsonl'),output)
    dump(root/(tag+'_runtime.json'),dict(e.torch_runtime(torch),requests=len(output),policy=policy,
        logical_live_max_bytes=max(x['logical_live_max_bytes'] for x in runtime),trace_details=runtime,
        allocator_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        nvml=e.nvml_owned_snapshot(),strict_physical_memory_claim=False,per_request_hidden_exact_reset=False))


def fit_costs(root):
    w=read(root/'workload.json');record=[];means={}
    for policy in BASE_CALIBRATION:
        rr=rows(root/('cal_'+policy+'_requests.jsonl'))
        validate_trace(rr,w['calibration'],policy)
        means[policy]=statistics.mean(r['request_ms'] for r in rr)
        if policy!='hf_native':record.extend(rr)
    median=lambda v:statistics.median(v) if v else (_ for _ in ()).throw(ValueError('Missing development path component'))
    costs=dict(increment_ms=median([r['parameter_ms'] for r in record if r['parameter_action']=='increment']),
               restore_ms=median([r['parameter_ms'] for r in record if r['parameter_action']=='exact_restore']))
    inc=rows(root/'cal_fixed_inc_requests.jsonl')
    for task in TASKS:
        for ctx in ('short','long'):
            r=[x for x in record if x['task']==task and x['context']==ctx]
            full=median([x['score_ms'] for x in inc if x['task']==task and x['context']==ctx])
            store=median([x['score_ms']+x['store_ms'] for x in r if x['score_kind']=='full_store' and x['cache_admitted']])
            cached=median([x['score_ms'] for x in r if x['cache_hit']])
            costs[task+':'+ctx]=dict(full_ms=full,full_store_ms=store,cached_ms=cached,
                admission_extra_ms=max(0.,store-full),unclamped_admission_extra_ms=store-full)
    dump(root/'plan_costs.json',dict(version=VERSION,costs=costs,base_calibration_means_ms=means,
        source_split='policy_dev only',future_hit_assumption=2,workload_signature=w['signature']))


def freeze_plan(root):
    plan=read(root/'plan_costs.json');w=read(root/'workload.json')
    rr=rows(root/'cal_length_sweep_requests.jsonl');choices=threshold_candidates(w)
    validate_sweep(rr,w)
    means={}
    for threshold in choices:
        means[threshold]=statistics.mean(r['request_ms'] for r in rr if r['threshold']==threshold)
    threshold=min(choices,key=lambda t:(means[t],t))
    fixed=dict(plan['base_calibration_means_ms'],length_rule=means[threshold])
    plan.update(length_threshold_tokens=threshold,length_threshold_means_ms=means,
        calibrated_fixed=min(fixed,key=lambda k:(fixed[k],k)),calibration_means_ms=fixed,
        threshold_selection='Actual continuous development replays, two orders, empty bank and canonical initial state per threshold trace; no evaluation labels or future requests used')
    dump(root/'plan.json',plan)


def schedule(mode):
    result=[dict(kind='calibration',policy=policy,tag='cal_'+policy) for policy in BASE_CALIBRATION]
    result += [dict(kind='costs',tag='fit_costs'),dict(kind='sweep',tag='cal_length_sweep'),
               dict(kind='freeze',tag='freeze_plan'),dict(kind='reference',tag='reference')]
    for i in range(1 if mode=='verify' else 2):
        for policy in (POLICIES if i==0 else tuple(reversed(POLICIES))):
            result.append(dict(kind='evaluation',policy=policy,round=i,tag='r'+str(i)+'_'+policy))
    return result


def validate_trace(rr,expected,policy):
    if len(rr)!=len(expected):raise ValueError('Incomplete continuous trace: '+policy)
    for r,c in zip(rr,expected):
        if r['policy']!=policy or any(r[k]!=c[k] for k in ('position','request_key','case_id','head_id','task','context','phase','gold')):raise ValueError('Trace identity differs: '+policy)
        if not math.isfinite(r['request_ms']) or r['request_ms']<=0 or r['logical_live_max_bytes']>LIVE_CAP:raise ValueError('Invalid time/budget: '+policy)
    if any(a['state_after']!=b['state_before'] for a,b in zip(rr,rr[1:])):raise ValueError('Unrecorded state reset: '+policy)
    if policy in ('fixed_inc','simple_rule','length_rule') and any(r['parameter_action']=='exact_restore' for r in rr):raise ValueError('Unexpected exact reset: '+policy)


def validate_sweep(rr,w):
    choices=threshold_candidates(w)
    if len(rr)!=len(w['calibration'])*len(choices)*2:raise ValueError('Incomplete threshold sweep')
    for threshold in choices:
        for repeat in (0,1):
            part=[r for r in rr if r['threshold']==threshold and r['calibration_repeat']==repeat]
            validate_trace(part,w['calibration'],'length_rule')


def validate_reference(rr,w):
    expected={c['request_key']:c for c in w['evaluation']}
    if len(rr)!=len(expected) or {r['request_key'] for r in rr}!=set(expected):raise ValueError('Incomplete or duplicated canonical reference')
    for r in rr:
        c=expected[r['request_key']]
        if r['gold']!=c['gold'] or set(r['scores'])!=set(c['label_ids']) or not all(math.isfinite(v) for v in r['scores'].values()):raise ValueError('Canonical reference identity/numerics differ')


def percentile(values,fraction):
    ordered=sorted(values);index=(len(ordered)-1)*fraction
    lo,hi=math.floor(index),math.ceil(index)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)


def summarize(root):
    spec,w,plan=read(root/'spec.json'),read(root/'workload.json'),read(root/'plan.json')
    reference_rows=rows(root/'reference_requests.jsonl');validate_reference(reference_rows,w)
    refs={r['request_key']:r for r in reference_rows}
    metrics=[];traces={};flags=[]
    for stage in schedule(spec['mode']):
        if stage['kind'] in ('costs','freeze'):continue
        tag=stage['tag'];te=rows(root/(tag+'_telemetry.jsonl'))
        if not te or any(r.get('status')!='ok' or r.get('other_compute_process_count',0) for r in te):flags.append(tag+': isolation/telemetry not confirmed')
        checks=read(root/(tag+'_cache_validation.json'))
        if len(checks)!=4 or not all(c['immutable'] and c['prediction_matches'] for c in checks):flags.append(tag+': warmup cache/numeric check requires review')
        if stage['kind']!='evaluation':continue
        rr=rows(root/(tag+'_requests.jsonl'));validate_trace(rr,w['evaluation'],stage['policy'])
        times=[r['request_ms'] for r in rr]
        per={}
        for task in TASKS:
            r=[x for x in rr if x['task']==task]
            per[task]=dict(requests=len(r),correct=sum(x['prediction']==x['gold'] for x in r),
                accuracy=sum(x['prediction']==x['gold'] for x in r)/len(r),
                reference_disagreements=sum(x['prediction']!=refs[x['request_key']]['prediction'] for x in r))
        metric=dict(round=stage['round'],policy=stage['policy'],requests=len(rr),mean_ms=statistics.mean(times),p95_ms=percentile(times,.95),
            accuracy=sum(r['prediction']==r['gold'] for r in rr)/len(rr),per_task=per,
            reference_disagreements=sum(r['prediction']!=refs[r['request_key']]['prediction'] for r in rr),
            max_abs_score_difference=max(abs(v-refs[r['request_key']]['scores'][k]) for r in rr for k,v in r['scores'].items()),
            cache_hits=sum(r['cache_hit'] for r in rr),cache_admissions=sum(r['cache_admitted'] for r in rr),
            parameter_actions=dict(collections.Counter(r['parameter_action'] for r in rr)),score_kinds=dict(collections.Counter(r['score_kind'] for r in rr)),
            total_decision_ms=sum(r['decision_ms'] for r in rr),logical_live_max_bytes=max(r['logical_live_max_bytes'] for r in rr))
        metrics.append(metric);traces[stage['round'],stage['policy']]=rr
    comparisons=[];stability=[]
    for i in range(1 if spec['mode']=='verify' else 2):
        get=lambda policy:next(x for x in metrics if x['round']==i and x['policy']==policy)
        joint=get('joint')
        for policy in POLICIES[:-1]:
            baseline=get(policy)
            action=lambda r:(r['parameter_action'],r['state_after'],r['score_kind'],r['cache_hit'],r['cache_admitted'])
            same=all(action(a)==action(b) for a,b in zip(traces[i,'joint'],traces[i,policy]))
            comparisons.append(dict(round=i,baseline=policy,mean_reduction_fraction=1-joint['mean_ms']/baseline['mean_ms'],
                p95_reduction_fraction=1-joint['p95_ms']/baseline['p95_ms'],executed_actions_identical=same,
                per_task_accuracy_difference={t:joint['per_task'][t]['accuracy']-baseline['per_task'][t]['accuracy'] for t in TASKS}))
    if spec['mode']=='pilot':
        for policy in POLICIES:
            for task in TASKS:
                for context in ('short','long'):
                    means=[statistics.mean(r['request_ms'] for r in traces[i,policy] if r['task']==task and r['context']==context) for i in (0,1)]
                    ratio=max(means)/min(means);stability.append(dict(policy=policy,task=task,context=context,ratio=ratio))
                    if ratio>1.20:flags.append(policy+'/'+task+'/'+context+': round mean ratio >1.20')
    if spec['mode']=='verify':decision='VERIFY_ONLY_NOT_A_POLICY_CONCLUSION'
    elif flags:decision='INCONCLUSIVE_NUMERICS_ISOLATION_OR_TIMING'
    elif any(x['reference_disagreements'] for x in metrics if x['policy']=='joint'):decision='NO_GO_REFERENCE_FIDELITY_REQUIRES_REVIEW'
    elif any(c['executed_actions_identical'] for c in comparisons):decision='NO_GO_IDENTICAL_TO_AN_OPTIMIZED_CONTROL'
    elif all(c['mean_reduction_fraction']>=.05 and c['p95_reduction_fraction']>=-.05 and all(v>=0 for v in c['per_task_accuracy_difference'].values()) for c in comparisons):
        decision='CANDIDATE_ONLY_NEEDS_INDEPENDENT_CONFIRMATION'
    else:decision='NO_GO_NO_STABLE_EXTRA_GAIN_AGAINST_ALL_CONTROLS'
    legacy={r['request_key']:r for r in rows(root/'legacy_serial_reference.jsonl')}
    result=dict(status=decision,metrics=metrics,comparisons=comparisons,flags=flags,round_stability=stability,
        calibrated_fixed=plan['calibrated_fixed'],length_threshold_tokens=plan['length_threshold_tokens'],
        reference_accuracy=sum(r['prediction']==r['gold'] for r in refs.values())/len(refs),
        new_batch_vs_legacy_serial_reference_prediction_changes=sum(r['prediction']!=legacy[k]['prediction'] for k,r in refs.items()),
        scope='Matched optimized label scoring, constructed few-shot session trace, previously observed evaluation pool, continuous state; no queueing, network or multi-user throughput',
        online_chooser_executed=True,all_controls_use_matched_optimized_scoring=True,
        true_labels_evaluated=True,application_quality_noninferiority_established=False,controller_benefit_demonstrated=False,
        novelty_established=False,strict_physical_memory_claim=False,
        screen='Pilot needs >=5% mean reduction against all five optimized controls in both rounds, <=5% p95 regression, no per-task observed accuracy drop, no reference disagreement and distinct operations. Engineering screen, not statistical significance or publication readiness.')
    dump(root/'optimized_continuous_summary.json',result)
    return result


def run_stage(e,root,active,timeout):
    e.require_idle(samples=3);dump(root/'active_stage.json',active)
    tag=active['tag'];env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',FTD_TRANSFER_CHUNK_MB='1')
    print('Starting',tag,'; log:',root/'logs'/(tag+'.log'),flush=True)
    with (root/'logs'/(tag+'.log')).open('w') as log:
        child=subprocess.Popen([sys.executable,str(root/SCRIPT),'--worker','--run-dir',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        event=threading.Event();monitor=threading.Thread(target=e.monitor_stage,args=(event,root/(tag+'_telemetry.jsonl'),child.pid),daemon=True)
        monitor.start();start=time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic()-start>timeout:raise TimeoutError(tag+' exceeded timeout')
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:print(tag,'running; see log',flush=True)
            if child.returncode:
                text=(root/'logs'/(tag+'.log')).read_text(errors='replace');print('\n'.join(text.splitlines()[-14:]),flush=True)
                if 'ProbeBlocked:' in text:raise e.ProbeBlocked(tag+' detected external GPU work; completed stages retained')
                raise RuntimeError(tag+' failed; full log retained')
        finally:event.set();monitor.join(timeout=5);e.stop_owned_process(child)


def environment_fingerprint(p,root):
    import torch
    pf=read(root/'preflight.json')
    gpu=pf['gpu']['gpus'][0]
    return dict(torch=str(torch.__version__),torch_cuda=torch.version.cuda,python=sys.version.split()[0],
        packages={k:v.get('version') for k,v in pf['software'].items()},cache_source=p.CACHE_SOURCE_SHA,
        gpu={k:gpu[k] for k in ('index','name','compute_cap','driver_version','memory.total')})


def copy_resume(previous,root,m,mode):
    inv=m.check_inventory(previous);old=read(previous/'status.json')
    required=(SCRIPT,'spec.json','workload.json','environment.json','status.json')
    if any(name not in inv for name in required):raise ValueError('Resume inventory is missing core provenance')
    if old['status']!='BLOCKED_GPU_ISOLATION':raise ValueError('Only a GPU-isolation pause can resume')
    if sha(previous/SCRIPT)!=sha(Path(__file__)):raise ValueError('Resume requires the unchanged runner')
    done=[x['stage'] for x in old['stages'] if x['status']=='COMPLETED']
    jobs=schedule(mode);alltags=[j['tag'] for j in jobs]
    if len(done)!=len(old['stages']) or done!=alltags[:len(done)]:raise ValueError('Completed stages are not a schedule prefix')
    w=m.read_workload(previous)
    for job in jobs[:len(done)]:
        tag=job['tag']
        if job['kind']=='costs':read(previous/'plan_costs.json');continue
        if job['kind']=='freeze':read(previous/'plan.json');continue
        r=rows(previous/(tag+'_requests.jsonl'))
        if job['kind']=='sweep':validate_sweep(r,w)
        elif job['kind']=='reference':validate_reference(r,w)
        else:validate_trace(r,w['calibration' if job['kind']=='calibration' else 'evaluation'],job['policy'])
        for name in (tag+'_requests.jsonl',tag+'_runtime.json',tag+'_telemetry.jsonl',tag+'_cache_validation.json','logs/'+tag+'.log'):
            if name not in inv:raise ValueError('Completed stage metadata missing: '+tag)
        runtime=read(previous/(tag+'_runtime.json'))
        if runtime['requests']!=len(r) or runtime['logical_live_max_bytes']>LIVE_CAP or ('policy' in job and runtime['policy']!=job['policy']):raise ValueError('Completed stage runtime differs: '+tag)
    pending=alltags[len(done)] if len(done)<len(alltags) else ''
    reused={}
    for name in inv:
        if name in ('status.json','active_stage.json','error.txt'):continue
        interrupted=pending and (name.startswith(pending+'_') or name=='logs/'+pending+'.log')
        history=name.startswith('prior_attempt/') or name in ('preflight.json','environment.json','resume_provenance.json','prior_status.json')
        dest=root/('prior_attempt/history/'+name if history else ('prior_attempt/'+name if interrupted else name))
        dest.parent.mkdir(parents=True,exist_ok=True)
        if name==SCRIPT:continue
        shutil.copyfile(previous/name,dest)
        if not interrupted and not history:reused[name]=sha(dest)
    dump(root/'prior_status.json',old)
    dump(root/'resume_provenance.json',dict(previous_result=str(previous),source_manifest_sha256=sha(previous/'manifest.json'),
        reused_stage_tags=done,reused_file_sha256=reused,note='Copied measurements are unchanged original observations, not new repeats; a process round may span an isolation pause.'))
    return dict(status='RUNNING',stages=[dict(stage=t,status='COMPLETED') for t in done]),read(previous/'environment.json')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('verify','pilot'),nargs='?',default='verify')
    parser.add_argument('--profile-result',type=Path,default=DEFAULT_PROFILE)
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'ftd_optimized_continuous_outputs')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--then-pilot',action='store_true')
    parser.add_argument('--stage-timeout',type=int,default=1800)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:worker(args.run_dir.resolve());return 0
    if args.stage_timeout<30:parser.error('Timeout must be >=30 seconds')
    resume_spec_error=None
    if args.resume_from:
        try:
            saved_mode=read(args.resume_from.expanduser()/'spec.json')['mode']
            if saved_mode not in ('verify','pilot'):raise ValueError('Invalid saved experiment mode')
            args.mode=saved_mode
        except Exception as exc:resume_spec_error=exc
    if args.then_pilot and args.mode!='verify':parser.error('--then-pilot requires verify')
    parent=args.output_dir.expanduser();parent.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=Path(tempfile.mkdtemp(prefix='ftd_optimized_'+args.mode+'_'+stamp+'_',dir=parent)).resolve()
    (root/'logs').mkdir();shutil.copyfile(Path(__file__),root/SCRIPT)
    state=dict(status='RUNNING',stages=[]);p=m=e=None;rc=0;oldenv=None
    try:
        if resume_spec_error is not None:raise resume_spec_error
        previous=(args.resume_from if args.resume_from else args.profile_result).expanduser().resolve()
        source_profile=previous if not args.resume_from else Path(read(previous/'spec.json')['source_profile_result'])
        for source,dest in [('ftd_kv_bottleneck_20261102.py','frozen_profile.py'),('frozen_continuous.py','frozen_continuous.py'),('frozen_kv_engine.py','frozen_kv_engine.py')]:
            shutil.copyfile(source_profile/source,root/dest)
        p,m,e=helpers(root);m.check_inventory(source_profile)
        if read(source_profile/'status.json')['status']!='COMPLETED_KV_BOTTLENECK_PROFILE' or read(source_profile/'kv_bottleneck_summary.json')['status']!='OPTIMIZED_KV_HEADROOM_ONLY':
            raise ValueError('Pass the completed positive KV bottleneck profile directory')
        if args.resume_from:
            state,oldenv=copy_resume(previous,root,m,args.mode)
            spec=read(root/'spec.json')
        else:
            old=read(source_profile/'spec.json');source_cont=Path(old['source_continuous_result'])
            m.check_inventory(source_cont)
            if sha(source_cont/'manifest.json')!=old['source_manifest_sha256']:raise ValueError('Original continuous source changed')
            workload=make_workload(m,source_cont,args.mode);dump(root/'workload.json',workload)
            shutil.copyfile(source_cont/'reference_requests.jsonl',root/'legacy_serial_reference.jsonl')
            spec=dict(version=VERSION,mode=args.mode,scope='smoke',model_dir=old['model_dir'],taskpool_dir=old['taskpool_dir'],
                source_result=old['source_result'],source_profile_result=str(source_profile),source_continuous_result=str(source_cont),
                source_profile_manifest_sha256=sha(source_profile/'manifest.json'),script_sha256=sha(Path(__file__)),
                logical_live_cap_bytes=LIVE_CAP,bank_cap_bytes=BANK_CAP,workload_signature=workload['signature'],
                scoring='Matched parallel mean label-token log likelihood; immutable shared-prefix forks and optional compact extraction',
                reused_observed_evaluation=True)
            dump(root/'spec.json',spec)
        if sha(source_profile/'manifest.json')!=spec['source_profile_manifest_sha256']:raise ValueError('Source profile changed since saved run')
        source=Path(spec['source_result'])
        if sha(source/'ftd_complete_request_20261101.py')!=m.SOURCE_SCRIPT_SHA:raise ValueError('Reference builder changed')
        for name,h in e.SOURCE_HASHES.items():
            if sha(source/name)!=h:raise ValueError('Reference metadata changed: '+name)
        e.preflight(spec,root);g=read(root/'preflight.json')['gpu'].get('gpus',[])
        if len(g)!=1 or g[0]['index']!='0' or os.environ.get('CUDA_VISIBLE_DEVICES','0')!='0':raise ValueError('Single physical GPU 0 required')
        p.cache_runtime_contract();fingerprint=environment_fingerprint(p,root)
        if oldenv is not None and oldenv!=fingerprint:raise ValueError('Environment changed since preserved timings; cannot mix sessions')
        dump(root/'environment.json',fingerprint);e.require_idle(samples=3)
        done={s['stage'] for s in state['stages']}
        for active in schedule(args.mode):
            if active['tag'] in done:print('REUSED_UNCHANGED:',active['tag'],flush=True);continue
            dump(root/'status.json',dict(state,active=active))
            if active['kind']=='costs':fit_costs(root)
            elif active['kind']=='freeze':freeze_plan(root)
            else:run_stage(e,root,active,args.stage_timeout)
            state['stages'].append(dict(stage=active['tag'],status='COMPLETED'))
        report=summarize(root)
        state.update(status='COMPLETED_OPTIMIZED_'+args.mode.upper(),decision=report['status'])
        if args.mode=='verify':state['automatic_pilot_ready']=not report['flags'] and all(x['reference_disagreements']==0 for x in report['metrics'] if x['policy']=='fixed_exact')
    except (Exception,KeyboardInterrupt) as exc:
        if e is not None and isinstance(exc,e.ProbeBlocked):rc=3;state.update(status='BLOCKED_GPU_ISOLATION',reason=str(exc))
        else:rc=2;state.update(status='FAILED',error_type=type(exc).__name__,error=str(exc));(root/'error.txt').write_text(traceback.format_exc())
        print('Stopped:',str(exc),flush=True)
    finally:
        dump(root/'status.json',state)
        if m is None:
            import tarfile
            files=[x for x in root.rglob('*') if x.is_file()];dump(root/'manifest.json',{str(x.relative_to(root)):sha(x) for x in files})
            archive=root.with_name(root.name+'.tar.gz')
            with tarfile.open(archive,'w:gz') as tf:
                for f in files+[root/'manifest.json']:tf.add(f,arcname=root.name+'/'+str(f.relative_to(root)),recursive=False)
        else:archive=m.pack(root)
        print('STATUS:',state['status']);print('SEND_THIS_ARCHIVE:',archive,flush=True)
    if rc==0 and args.then_pilot:
        if state.get('automatic_pilot_ready'):
            print('Verify execution checks passed; starting a fresh two-round pilot.',flush=True)
            return main(['pilot','--profile-result',str(source_profile),'--output-dir',str(parent),'--stage-timeout',str(args.stage_timeout)])
        print('Verify needs review; pilot was not started.',flush=True)
    return rc


if __name__=='__main__':sys.exit(main())
