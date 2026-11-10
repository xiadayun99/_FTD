#!/usr/bin/env python3
"""Matched development-only KV/scoring bottleneck profile; no controller claim."""
import argparse
import collections
import copy
import datetime
import functools
import hashlib
import importlib.metadata
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

VERSION = 'ftd_kv_bottleneck_20261102_v1'
CONTINUOUS_SHA = 'e091ef792debb959bd35be8411f07937cbd6f59e4d7dee1f59c985ecb82e2c15'
ENGINE_SHA = '0101cdad4d9b887919e09d2004c6bdf7eb68556039c06dfd98ca32b5c46f18fb'
CACHE_SOURCE_SHA = 'f4e11caf14c75bdefa5e3003b93ef0e8b97fb6103375705ad944b0b344d7da8a'
DEFAULT_SOURCE = Path('./assets/legacy/ftd_continuous_outputs/ftd_continuous_pilot_20261102T032449Z_zkivgemb')
PATHS = ('serial_full_deep', 'serial_cached_deep', 'serial_full_shared', 'serial_cached_shared',
         'batch_full', 'batch_cached_deep', 'batch_cached_shared', 'serial_cold_shared', 'batch_cold_shared')
FULL_PATHS = ('serial_full_deep', 'serial_full_shared', 'batch_full')
WARM_PATHS = ('serial_cached_shared', 'batch_cached_shared')
PRIMITIVES = ('head_fork_deep','head_fork_shared','label_fork_deep','label_fork_shared',
              'extract_deep_crop','extract_direct','head_prefill')
LIVE_CAP = 2048 * 2**20


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def write_rows(p, values):
    with Path(p).open('w') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def load_module(path, expected, name):
    if sha(path) != expected:
        raise ValueError('Frozen helper identity changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers(root):
    m = load_module(root/'frozen_continuous.py', CONTINUOUS_SHA, 'profile_frozen_continuous')
    e = m.load_engine(root/'frozen_kv_engine.py')
    return m, e


@functools.lru_cache(maxsize=1)
def cache_runtime_contract():
    import transformers.cache_utils as c
    version,actual=importlib.metadata.version('transformers'),sha(Path(c.__file__))
    if version != '4.57.6' or actual != CACHE_SOURCE_SHA:
        raise ValueError('Shared-prefix fork requires the inspected transformers 4.57.6 cache_utils.py; observed version='+version+' source_sha256='+actual+'; do not upgrade to bypass this guard')
    return c.DynamicCache, c.DynamicLayer


def validate_dynamic(cache):
    cache_class, layer_class = cache_runtime_contract()
    if type(cache) is not cache_class or getattr(cache, 'offloading', False) or not cache.layers:
        raise TypeError('Only non-offloaded plain DynamicCache is supported')
    for layer in cache.layers:
        if type(layer) is not layer_class or not layer.is_initialized or layer.keys is None or layer.values is None:
            raise TypeError('Static, sliding, quantized, empty, or custom cache layers are unsupported')


def fork_cache(cache, kind='shared', batch=1):
    """New container/layers; plain DynamicLayer.update replaces tensors via cat.

    Sharing eliminates the eager prefix clone. Appending still allocates and
    copies through torch.cat; this is not a paged or zero-copy attention engine.
    """
    validate_dynamic(cache)
    if kind not in ('deep', 'shared') or batch < 1:
        raise ValueError('Invalid fork request')
    if kind == 'deep':
        result = copy.deepcopy(cache)
    else:
        result = copy.copy(cache)
        result.layers = [copy.copy(layer) for layer in cache.layers]
    for layer in result.layers:
        if layer.keys.shape[0] != 1 or layer.values.shape[0] != 1:
            raise ValueError('Template must have batch size one')
        if batch != 1:
            layer.keys = layer.keys.expand(batch, -1, -1, -1)
            layer.values = layer.values.expand(batch, -1, -1, -1)
    return result


def extract_prefix(cache, length):
    if not 0 < length <= cache.get_seq_length():
        raise ValueError('Invalid compact prefix length')
    value = fork_cache(cache)
    for layer in value.layers:
        layer.keys = layer.keys[..., :length, :].clone().contiguous()
        layer.values = layer.values[..., :length, :].clone().contiguous()
    return value


class Meter:
    """Diagnostic-only scopes; do not add their costs to light request timings."""
    def __init__(self, torch, device):
        self.torch = torch
        self.cuda = str(device).startswith('cuda')
        self.spans = []

    def call(self, label, fn):
        a = self.torch.cuda.Event(enable_timing=True) if self.cuda else None
        b = self.torch.cuda.Event(enable_timing=True) if self.cuda else None
        if a is not None:
            a.record()
        start = time.perf_counter_ns()
        result = fn()
        cpu = (time.perf_counter_ns()-start)/1e6
        if b is not None:
            b.record()
        self.spans.append((label, cpu, a, b))
        return result

    def report(self):
        result = {}
        for label, cpu, a, b in self.spans:
            item = result.setdefault(label, dict(calls=0, host_scope_ms=0., cuda_stream_elapsed_ms=0. if self.cuda else None))
            item['calls'] += 1
            item['host_scope_ms'] += cpu
            if a is not None:
                item['cuda_stream_elapsed_ms'] += a.elapsed_time(b)
        return result


def call(meter, label, fn):
    return fn() if meter is None else meter.call(label, fn)


def finalize(scores):
    if any(not math.isfinite(v) for v in scores.values()):
        raise FloatingPointError('Nonfinite label score')
    order = sorted(scores, key=lambda k: (-scores[k], k))
    return dict(scores=scores, prediction=order[0], margin=scores[order[0]]-scores[order[1]])


def serial_scores(torch, engine, model, prefix, labels, head_length, template=None,
                  fork_kind='deep', device='cuda:0', meter=None, check=None):
    tail = engine.tail_argument(model)
    if tail is None:
        raise ValueError('Tail-logit API required')
    with torch.inference_mode():
        working = call(meter, 'head_fork', lambda:fork_cache(template, fork_kind)) if template is not None else None
        if working is not None and working.get_seq_length() != head_length:
            raise ValueError('Head cache length mismatch')
        values = prefix[head_length:] if working is not None else prefix
        if not values:
            raise ValueError('A nonempty query suffix is required')
        ids = call(meter, 'input_setup', lambda:torch.tensor([values], device=device))
        out = call(meter, 'request_prefill', lambda:model(input_ids=ids, past_key_values=working, use_cache=True, **{tail:1}))
        working, first = out.past_key_values, out.logits[0,-1].float()
        del out
        if check:
            check(working)
        scores = {}
        for label in sorted(labels):
            tokens = labels[label]
            branch = call(meter, 'label_fork', lambda:fork_cache(working, fork_kind)) if len(tokens)>1 else None
            logits = first
            total = torch.zeros((), device=device, dtype=torch.float32)
            for i, token in enumerate(tokens):
                total = call(meter, 'score_reduce', lambda:total+torch.log_softmax(logits, dim=-1)[token])
                if i+1 < len(tokens):
                    ids = call(meter, 'input_setup', lambda:torch.tensor([[token]], device=device))
                    out = call(meter, 'label_forward', lambda:model(input_ids=ids, past_key_values=branch, use_cache=True, **{tail:1}))
                    branch, logits = out.past_key_values, out.logits[0,-1].float()
                    del out
                    if check:
                        check(working, branch)
            scores[label] = call(meter, 'result_transfer', lambda:float((total/len(tokens)).item()))
            del branch
    return finalize(scores)


def batched_scores(torch, engine, model, prefix, labels, head_length, template=None,
                   fork_kind='shared', device='cuda:0', meter=None, check=None):
    names = sorted(labels)
    if template is not None and template.get_seq_length() != head_length:
        raise ValueError('Head cache length mismatch')
    suffix = prefix[head_length:] if template is not None else prefix
    if not suffix:
        raise ValueError('A nonempty query suffix is required')
    seqs = [suffix+labels[name] for name in names]
    width, keep = max(map(len, seqs)), max(map(len, labels.values()))+1
    tail = engine.tail_argument(model)
    if tail is None:
        raise ValueError('Tail-logit API required')
    def inputs():
        pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        ids = torch.full((len(names),width),pad,dtype=torch.long,device=device)
        newmask = torch.zeros_like(ids)
        for i, seq in enumerate(seqs):
            ids[i,width-len(seq):] = torch.tensor(seq,device=device)
            newmask[i,width-len(seq):] = 1
        past = head_length if template is not None else 0
        mask = torch.cat([torch.ones((len(names),past),dtype=torch.long,device=device),newmask],dim=-1)
        positions = (mask.cumsum(-1)-1).clamp(min=0)[:,past:]
        return ids,mask,positions
    with torch.inference_mode():
        ids,mask,positions = call(meter, 'input_setup', inputs)
        working = call(meter, 'head_fork',lambda:fork_cache(template,fork_kind,len(names))) if template is not None else None
        out = call(meter, 'batched_forward',lambda:model(input_ids=ids,attention_mask=mask,
            position_ids=positions,past_key_values=working,use_cache=template is not None,**{tail:keep}))
        if check:
            check(working, getattr(out,'past_key_values',None))
        def reduction():
            scores = {}
            for i,name in enumerate(names):
                targets = labels[name]
                begin = keep-len(targets)-1
                logits = out.logits[i,begin:begin+len(targets)].float()
                token_ids = torch.tensor(targets,device=device)
                value = torch.log_softmax(logits,dim=-1).gather(1,token_ids[:,None]).mean()
                scores[name] = float(value.item())
            return scores
        scores = call(meter,'score_reduce_and_transfer',reduction)
    return finalize(scores)


def execute(torch, e, model, path, case, template, device='cuda:0', meter=None, check=None):
    cached = '_cached_' in path
    local = template if cached else None
    if '_cold_' in path:
        local = call(meter,'cold_head_build',lambda:e.prefill_cache(torch,model,case['head_tokens'],device))
        if check:
            check(local)
    # Include the cold head still held by this invocation as well as its
    # growing request and label caches in logical storage accounting.
    active_check=(lambda *caches:check(local,*caches)) if check is not None else None
    if path.startswith('batch_'):
        return batched_scores(torch,e,model,case['prefix'],case['label_ids'],len(case['head_tokens']),
            local,'deep' if path.endswith('_deep') else 'shared',device,meter,active_check)
    return serial_scores(torch,e,model,case['prefix'],case['label_ids'],len(case['head_tokens']),
        local,'deep' if path.endswith('_deep') else 'shared',device,meter,active_check)


def timed(torch, e, device, fn):
    e.sync(torch,device)
    start = time.perf_counter_ns()
    value = fn()
    e.sync(torch,device)
    return value,(time.perf_counter_ns()-start)/1e6


class Budget:
    def __init__(self,e,model,template,cap=LIVE_CAP):
        self.e,self.base,self.template,self.cap = e,e.model_payload_bytes(model),template,cap
        self.maximum = self.base

    def check(self,*caches):
        tensors = []
        for c in (self.template,)+caches:
            tensors.extend(self.e.cache_tensors(c))
        amount = self.base+self.e.tensor_storage_bytes(tensors)
        self.maximum = max(self.maximum,amount)
        if amount>self.cap:
            raise ValueError('Observed logical live tensors exceed 2 GiB')


def select_cases(m, workload):
    result = []
    for task in ('mrpc','qnli'):
        for context in ('short','long'):
            pool = sorted([c for c in workload['calibration'] if c['task']==task and c['context']==context],key=lambda c:c['request_key'])
            seen = set()
            for c in pool:
                if c['request_key'] in seen:
                    continue
                seen.add(c['request_key'])
                result.append(dict(case_id=c['case_id'],request_key=c['request_key'],task=task,context=context,
                    head_id=c['head_id'],head_tokens=workload['heads'][c['head_id']]['tokens'],
                    prefix=c['prefix'],label_ids=c['label_ids'],split='policy_dev'))
                if len(seen)==2:
                    break
            if len(seen)!=2:
                raise ValueError('Need two distinct development inputs per task/context')
    data = dict(cases=result,selection='First two SHA-ordered distinct development requests per task/context; no evaluation queries or labels used')
    data['signature'] = identity(data)
    return data


def read_cases(root):
    d = read(root/'profile_cases.json')
    signature = d.pop('signature')
    if identity(d)!=signature:
        raise ValueError('Profile input signature changed')
    return d['cases']


def worker(root):
    import torch
    m,e = helpers(root)
    spec,active = read(root/'spec.json'),read(root/'active_stage.json')
    cache_runtime_contract()
    torch.manual_seed(0)
    e.torch_runtime(torch)
    targets,_ = e.target_cache_load(torch,Path(spec['source_result']))
    model,_ = e.merged_model_and_factors(spec,Path(spec['source_result']),torch,False)
    params = dict(model.named_parameters())
    main_rows,calibration,diagnostics,primitives,validity = [],[],[],[],[]
    tag,round_index = active['tag'],active['round']
    cases = read_cases(root)
    if round_index:
        cases = list(reversed(cases))
    for case_index,c in enumerate(cases):
        e.require_idle(os.getpgrp())
        e.install_host_target(torch,params,targets[c['task']])
        template = e.prefill_cache(torch,model,c['head_tokens'],'cuda:0')
        validate_dynamic(template)
        budget = Budget(e,model,template)
        cfg = model.config
        per_token = 2*cfg.num_hidden_layers*cfg.num_key_value_heads*getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)*2
        worst = budget.base+e.cache_bytes(template)+6*(len(c['prefix'])+max(map(len,c['label_ids'].values())))*per_token+64*2**20
        if worst>LIVE_CAP:
            raise ValueError('Conservative working-cache reserve exceeds 2 GiB before scoring')
        before = e.tensor_digest(torch,[(str(i),t) for i,t in enumerate(e.cache_tensors(template))])
        reference = serial_scores(torch,e,model,c['prefix'],c['label_ids'],len(c['head_tokens']),device='cuda:0',check=budget.check)
        # Each path warmed once, then independent calibration calls select the
        # full and warm-cache baselines before timed confirmations.
        for path in PATHS:
            execute(torch,e,model,path,c,template,check=budget.check)
        for rep in range(2):
            order = list(PATHS)
            offset = (case_index+rep+round_index)%len(order)
            order = order[offset:]+order[:offset]
            if round_index:
                order.reverse()
            for path in order:
                e.require_idle(os.getpgrp())
                result,ms = timed(torch,e,'cuda:0',lambda:execute(torch,e,model,path,c,template,check=budget.check))
                calibration.append(dict(round=round_index,request_key=c['request_key'],path=path,repeat=rep,wall_ms=ms))
        e.require_idle(os.getpgrp())
        torch.cuda.reset_peak_memory_stats()
        for rep in range(3):
            order = list(PATHS)
            offset = (case_index+rep+round_index)%len(order)
            order = order[offset:]+order[:offset]
            if round_index:
                order.reverse()
            for path in order:
                result,ms = timed(torch,e,'cuda:0',lambda:execute(torch,e,model,path,c,template,check=budget.check))
                main_rows.append(dict(round=round_index,request_key=c['request_key'],case_id=c['case_id'],task=c['task'],context=c['context'],
                    head_tokens=len(c['head_tokens']),prefix_tokens=len(c['prefix']),path=path,repeat=rep,wall_ms=ms,
                    same_prediction_as_serial_full=result['prediction']==reference['prediction'],
                    max_abs_score_difference_vs_serial_full=max(abs(result['scores'][k]-reference['scores'][k]) for k in result['scores']),
                    logical_live_max_bytes=budget.maximum,**result))
            e.require_idle(os.getpgrp())
        # Separately instrumented passes: added event/host overhead is excluded
        # from the lightweight primary timing rows above.
        for path in PATHS:
            meter = Meter(torch,'cuda:0')
            result,ms = timed(torch,e,'cuda:0',lambda:execute(torch,e,model,path,c,template,meter=meter,check=budget.check))
            diagnostics.append(dict(round=round_index,request_key=c['request_key'],path=path,instrumented_wall_ms=ms,components=meter.report()))
        full_cache = e.prefill_cache(torch,model,c['prefix'],'cuda:0')
        full_before = e.tensor_digest(torch,[(str(i),t) for i,t in enumerate(e.cache_tensors(full_cache))])
        functions = {
            'head_fork_deep':lambda:fork_cache(template,'deep'),
            'head_fork_shared':lambda:fork_cache(template,'shared'),
            'label_fork_deep':lambda:fork_cache(full_cache,'deep'),
            'label_fork_shared':lambda:fork_cache(full_cache,'shared'),
            'extract_deep_crop':lambda:m.clone_prefix(full_cache,len(c['head_tokens'])),
            'extract_direct':lambda:extract_prefix(full_cache,len(c['head_tokens'])),
            'head_prefill':lambda:e.prefill_cache(torch,model,c['head_tokens'],'cuda:0')}
        for name,fn in functions.items():
            for rep in range(3):
                value,ms = timed(torch,e,'cuda:0',fn)
                budget.check(full_cache,value)
                source_cache=full_cache if name.startswith(('label_','extract_')) else template
                retained=e.tensor_storage_bytes(e.cache_tensors(source_cache)+e.cache_tensors(value))-e.cache_bytes(source_cache)
                primitives.append(dict(round=round_index,request_key=c['request_key'],primitive=name,repeat=rep,wall_ms=ms,
                    returned_storage_bytes=e.cache_bytes(value),new_retained_unique_kv_storage_bytes=retained))
                del value
        after = e.tensor_digest(torch,[(str(i),t) for i,t in enumerate(e.cache_tensors(template))])
        full_after = e.tensor_digest(torch,[(str(i),t) for i,t in enumerate(e.cache_tensors(full_cache))])
        if before!=after or full_before!=full_after:
            raise ValueError('An immutable cache template was modified; all headroom claims blocked')
        validity.append(dict(request_key=c['request_key'],template_digest_before=before,template_digest_after=after,
            full_template_digest_before=full_before,full_template_digest_after=full_after,passed=True))
        dump(root/(tag+'_'+str(case_index)+'_memory.json'),dict(logical_live_max_bytes=budget.maximum,
            allocator_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
            nvml=e.nvml_owned_snapshot(),strict_physical_memory_claim=False))
        del full_cache,template,budget
        # Preserve finished case records if a later case is interrupted.
        write_rows(root/(tag+'_requests.jsonl'),main_rows)
        write_rows(root/(tag+'_calibration.jsonl'),calibration)
        write_rows(root/(tag+'_components.jsonl'),diagnostics)
        write_rows(root/(tag+'_primitives.jsonl'),primitives)
        dump(root/(tag+'_validity.json'),validity)
        print(tag,'completed development cases:',case_index+1,'/',len(cases),flush=True)
    write_rows(root/(tag+'_requests.jsonl'),main_rows)
    write_rows(root/(tag+'_calibration.jsonl'),calibration)
    write_rows(root/(tag+'_components.jsonl'),diagnostics)
    write_rows(root/(tag+'_primitives.jsonl'),primitives)
    dump(root/(tag+'_validity.json'),validity)
    dump(root/(tag+'_runtime.json'),dict(e.torch_runtime(torch),cache_source_sha256=CACHE_SOURCE_SHA,model_bytes=e.model_payload_bytes(model),
        scope='Canonical weights per development case; static warm-prefix/scorer microprofile, not continuous control or end-to-end adapter switching'))


def run_stage(m,e,root,active,timeout):
    e.require_idle(samples=3)
    dump(root/'active_stage.json',active)
    tag = active['tag']
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',FTD_TRANSFER_CHUNK_MB='1')
    print('Starting',tag,'; log:',root/'logs'/(tag+'.log'),flush=True)
    with (root/'logs'/(tag+'.log')).open('w') as log:
        child = subprocess.Popen([sys.executable,str(root/'ftd_kv_bottleneck_20261102.py'),'--worker','--run-dir',str(root)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        stop = threading.Event()
        observer = threading.Thread(target=e.monitor_stage,args=(stop,root/(tag+'_telemetry.jsonl'),child.pid),daemon=True)
        observer.start();started=time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic()-started>timeout:
                    raise TimeoutError(tag+' exceeded timeout')
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:print(tag,'running; see log',flush=True)
            if child.returncode:
                logtext=(root/'logs'/(tag+'.log')).read_text(errors='replace')
                print('\n'.join(logtext.splitlines()[-14:]),flush=True)
                if 'ProbeBlocked:' in logtext:
                    raise e.ProbeBlocked(tag+' detected external GPU work')
                raise RuntimeError(tag+' failed; full log retained')
        finally:
            stop.set();observer.join(timeout=5);e.stop_owned_process(child)


def summarize(root):
    cases=read_cases(root)
    timing,primitive,checks,flags,decisions=[],[],[],[],[]
    for i in (0,1):
        tag='r'+str(i)
        rr=rows(root/(tag+'_requests.jsonl'));cal=rows(root/(tag+'_calibration.jsonl'))
        if len(rr)!=len(cases)*len(PATHS)*3 or len(cal)!=len(cases)*len(PATHS)*2:
            raise ValueError('Incomplete profile records')
        telemetry=rows(root/(tag+'_telemetry.jsonl'))
        if not telemetry or any(r.get('status')!='ok' or r.get('other_compute_process_count',0) for r in telemetry):
            flags.append(tag+': isolation/telemetry not confirmed')
        validity=read(root/(tag+'_validity.json'))
        if len(validity)!=len(cases) or not all(v['passed'] for v in validity):
            raise ValueError('Cache immutability validation incomplete')
        checks.extend(validity)
        for c in cases:
            scores={}
            for path in PATHS:
                r=[x for x in rr if x['request_key']==c['request_key'] and x['path']==path]
                cr=[x for x in cal if x['request_key']==c['request_key'] and x['path']==path]
                if sorted(x['repeat'] for x in r)!=[0,1,2] or sorted(x['repeat'] for x in cr)!=[0,1]:
                    raise ValueError('Profile repetition identity changed')
                if any(not math.isfinite(x['wall_ms']) or x['wall_ms']<=0 for x in r+cr) or any(x['logical_live_max_bytes']>LIVE_CAP for x in r):
                    raise ValueError('Invalid profile timings/budget')
                same=all(x['same_prediction_as_serial_full'] for x in r)
                if not same:
                    flags.append(tag+'/'+c['request_key']+'/'+path+': prediction differs from serial full')
                scores[path]=dict(round=i,request_key=c['request_key'],case_id=c['case_id'],task=c['task'],context=c['context'],path=path,
                    median_ms=statistics.median(x['wall_ms'] for x in r),mean_ms=statistics.mean(x['wall_ms'] for x in r),
                    calibration_ms=statistics.mean(x['wall_ms'] for x in cr),same_prediction_as_serial_full=same,
                    max_abs_score_difference=max(x['max_abs_score_difference_vs_serial_full'] for x in r))
                timing.append(scores[path])
            for a,b in [('serial_full_deep','serial_full_shared'),('serial_cached_deep','serial_cached_shared'),('batch_cached_deep','batch_cached_shared')]:
                x=[r for r in rr if r['request_key']==c['request_key'] and r['path']==a]
                y=[r for r in rr if r['request_key']==c['request_key'] and r['path']==b]
                if any(max(abs(u['scores'][k]-v['scores'][k]) for k in u['scores'])>1e-5 for u,v in zip(x,y)):
                    flags.append(tag+'/'+c['request_key']+': shared/deep same-algorithm scores differ >1e-5')
            full=min(FULL_PATHS,key=lambda p:(scores[p]['calibration_ms'],p))
            warm=min(WARM_PATHS,key=lambda p:(scores[p]['calibration_ms'],p))
            cold='batch_cold_shared' if warm.startswith('batch') else 'serial_cold_shared'
            gap=scores[full]['median_ms']-scores[warm]['median_ms']
            cold_cost=scores[cold]['median_ms']
            break_even=(max(1,math.floor((cold_cost-scores[warm]['median_ms'])/gap)+1) if gap>0 else None)
            decisions.append(dict(round=i,request_key=c['request_key'],full=full,warm=warm,cold=cold,
                full_median_ms=scores[full]['median_ms'],warm_median_ms=scores[warm]['median_ms'],cold_median_ms=cold_cost,
                warm_reduction_fraction=1-scores[warm]['median_ms']/scores[full]['median_ms'],
                projected_first_profitable_reuse_window=break_even,
                break_even_scope='Algebraic projection from separate microprofile medians; not a measured online request sequence or guaranteed reuse forecast'))
        pr=rows(root/(tag+'_primitives.jsonl'))
        cp=rows(root/(tag+'_components.jsonl'))
        if len(pr)!=len(cases)*len(PRIMITIVES)*3 or len(cp)!=len(cases)*len(PATHS):
            raise ValueError('Incomplete primitive/component diagnostics')
        for c in cases:
            for name in PRIMITIVES:
                subset=[r for r in pr if r['request_key']==c['request_key'] and r['primitive']==name]
                if sorted(r['repeat'] for r in subset)!=[0,1,2] or any(not math.isfinite(r['wall_ms']) or r['wall_ms']<=0 for r in subset):
                    raise ValueError('Invalid primitive timing repetitions')
            for path in PATHS:
                subset=[r for r in cp if r['request_key']==c['request_key'] and r['path']==path]
                if len(subset)!=1 or not subset[0]['components']:
                    raise ValueError('Invalid component diagnostic identity')
        primitive.extend(dict(r,round=i) for r in pr)
    for c in cases:
        for path in PATHS:
            a=[t['median_ms'] for t in timing if t['request_key']==c['request_key'] and t['path']==path]
            if max(a)/min(a)>1.20:
                flags.append(c['request_key']+'/'+path+': round median ratio >1.20')
    candidate=[c['request_key'] for c in cases if all(d['warm_reduction_fraction']>=.05 for d in decisions if d['request_key']==c['request_key'])]
    status='INCONCLUSIVE_NUMERICS_OR_TIMING' if flags else ('OPTIMIZED_KV_HEADROOM_ONLY' if candidate else 'NO_KV_HEADROOM_IN_MATCHED_WORKLOAD')
    report=dict(status=status,timings=timing,decisions=decisions,primitives=primitive,cache_immutability_checks=checks,flags=flags,candidate_cases=candidate,
        scope='Eight existing development context/query pairs, two canonical-state workers, three timing repetitions, nine scoring paths. No new held-out quality test, online admission, queueing, or parameter-switch latency measurement.',
        main_timed_rows=len(cases)*len(PATHS)*3*2,component_passes=len(cases)*len(PATHS)*2,primitive_rows=len(primitive),
        controller_benefit_demonstrated=False,novelty_established=False,true_label_accuracy_established=False,strict_physical_memory_claim=False,
        caution='Shared-prefix fork removes the eager tensor clone; DynamicLayer still concatenates into new tensors. Warm and cold timings must remain distinct. Instrumented component scopes overlap CPU/device waiting and must not be summed into normal request latency. Compare cache reuse with the best calibration-selected full scorer, including batching; do not use only the legacy serial baseline.')
    dump(root/'kv_bottleneck_summary.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuous-result',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'ftd_kv_bottleneck_outputs')
    parser.add_argument('--stage-timeout',type=int,default=1800)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        worker(args.run_dir.resolve());return 0
    if args.stage_timeout<30:parser.error('Timeout must be >=30 seconds')
    parent=args.output_dir.expanduser();parent.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=Path(tempfile.mkdtemp(prefix='ftd_kv_bottleneck_'+stamp+'_',dir=parent)).resolve()
    (root/'logs').mkdir()
    shutil.copyfile(Path(__file__),root/'ftd_kv_bottleneck_20261102.py')
    state=dict(status='RUNNING',stages=[]);m=e=None;rc=0
    try:
        previous=args.continuous_result.expanduser().resolve()
        shutil.copyfile(previous/'ftd_continuous_state_20261102.py',root/'frozen_continuous.py')
        shutil.copyfile(previous/'frozen_kv_engine.py',root/'frozen_kv_engine.py')
        m,e=helpers(root)
        m.check_inventory(previous)
        if read(previous/'status.json')['status']!='COMPLETED_CONTINUOUS_PILOT':
            raise ValueError('Pass the completed continuous pilot directory, not a tar.gz')
        old=read(previous/'spec.json')
        spec=dict(version=VERSION,model_dir=old['model_dir'],taskpool_dir=old['taskpool_dir'],source_result=old['source_result'],
            scope='smoke',source_continuous_result=str(previous),source_manifest_sha256=sha(previous/'manifest.json'),
            script_sha256=sha(Path(__file__)),cache_source_sha256=CACHE_SOURCE_SHA,logical_live_cap_bytes=LIVE_CAP,
            rounds=2,measured_repeats=3,calibration_repeats=2,profile_only=True)
        dump(root/'spec.json',spec)
        source=Path(spec['source_result'])
        if sha(source/'ftd_complete_request_20261101.py')!=m.SOURCE_SCRIPT_SHA:
            raise ValueError('Reference source builder identity changed')
        for name,h in e.SOURCE_HASHES.items():
            if sha(source/name)!=h:raise ValueError('Reference metadata changed: '+name)
        e.preflight(spec,root)
        g=read(root/'preflight.json')['gpu'].get('gpus',[])
        if len(g)!=1 or g[0]['index']!='0' or os.environ.get('CUDA_VISIBLE_DEVICES','0')!='0':
            raise ValueError('Single physical GPU 0 mapping required')
        cache_runtime_contract()
        e.require_idle(samples=3)
        dump(root/'profile_cases.json',select_cases(m,m.read_workload(previous)))
        for i in (0,1):
            active=dict(tag='r'+str(i),round=i)
            run_stage(m,e,root,active,args.stage_timeout)
            state['stages'].append(dict(stage=active['tag'],status='COMPLETED'))
        report=summarize(root)
        state.update(status='COMPLETED_KV_BOTTLENECK_PROFILE',decision=report['status'])
    except (Exception,KeyboardInterrupt) as exc:
        if e is not None and isinstance(exc,e.ProbeBlocked):
            rc=3;state.update(status='BLOCKED_GPU_ISOLATION',reason=str(exc))
        else:
            rc=2;state.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
            (root/'error.txt').write_text(traceback.format_exc())
        print('Stopped:',str(exc),flush=True)
    finally:
        dump(root/'status.json',state)
        if m is None:
            # Standard-library fallback, including a missing prior-directory failure.
            import tarfile
            files=[p for p in root.rglob('*') if p.is_file()]
            dump(root/'manifest.json',{str(p.relative_to(root)):sha(p) for p in files})
            archive=root.with_name(root.name+'.tar.gz')
            with tarfile.open(archive,'w:gz') as tf:
                for p in files+[root/'manifest.json']:tf.add(p,arcname=root.name+'/'+str(p.relative_to(root)),recursive=False)
        else:
            archive=m.pack(root)
        print('STATUS:',state['status'])
        print('SEND_THIS_ARCHIVE:',archive,flush=True)
    return rc


if __name__=='__main__':sys.exit(main())
