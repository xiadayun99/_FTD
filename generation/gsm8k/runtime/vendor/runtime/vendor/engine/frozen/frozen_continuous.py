#!/usr/bin/env python3
"""Bounded continuous-state experiment; no reset between served requests.

Uses the verified original engine for model/LoRA loading, reference identity,
chunked copies, and process isolation. No package installation or downloads.
"""
import argparse
import collections
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback

VERSION = 'ftd_continuous_state_20261102_v1'
ENGINE_SHA = '0101cdad4d9b887919e09d2004c6bdf7eb68556039c06dfd98ca32b5c46f18fb'
SOURCE_SCRIPT_SHA = '389efa3a60d7b539496b6871b3c9d45ac9e656bd4c0f5a940708838dde5e1ed0'
DEFAULT_KV = Path('./assets/legacy/ftd_kv_feasibility_outputs/ftd_kv_resume_20261102T004549Z_8ntazjvf')
POLICIES = ('fixed_inc', 'simple_rule', 'fixed_exact', 'hf_native', 'joint')
TASKS = ('mrpc', 'qnli')
MIB = 2**20
LIVE_CAP = 2048 * MIB
BANK_CAP = 192 * MIB
RESERVE = 32 * MIB


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_rows(path, values):
    with Path(path).open('w') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def load_engine(path):
    if sha(path) != ENGINE_SHA:
        raise ValueError('Original engine checksum changed')
    spec = importlib.util.spec_from_file_location('ftd_continuous_frozen_engine', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_inventory(root):
    inv = read(root / 'manifest.json')
    for name, expected in inv.items():
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe inventory path')
        f = root / name
        if not f.is_file() or f.is_symlink() or sha(f) != expected:
            raise ValueError('Previous file missing or changed: ' + name)
    return inv


def longest_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def demonstration_head(tokenizer, demos, limit, variant):
    ordered = list(demos if variant == 0 else reversed(demos))
    text = 'Use the examples below to answer the new classification item.\n\n'
    used = []
    for c in ordered:
        block = c['raw_prompt'] + '\nAnswer: ' + c['gold'] + '\n\n'
        if len(tokenizer.encode(text + block + 'New item:\n', add_special_tokens=False)) > limit:
            break
        text += block
        used.append(c['case_id'])
    if not used:
        raise ValueError('No complete development demonstration fits context budget')
    return text + 'New item:\n', used


def make_workload(engine, source, tokenizer, mode):
    original = engine.read_signed(source / 'cases.json')
    result = dict(heads={}, calibration=[], evaluation=[], selection='Existing SHA order; no correctness/margin selection',
                  raw_query_count_per_task=4 if mode == 'verify' else 32,
                  reused_previously_observed_evaluation=True,
                  note='Real classification queries with development-only few-shot contexts; constructed session schedule, not production traffic')
    for task in TASKS:
        fit = original['tasks'][task]['fit']
        if len(fit) < 16:
            raise ValueError('Need the recorded 16 development cases per task')
        demos, dev = fit[:8], fit[8:16]
        ev = original['tasks'][task]['eval'][:result['raw_query_count_per_task']]
        if set(c['case_id'] for c in fit) & set(c['case_id'] for c in ev):
            raise ValueError('Development/evaluation overlap')
        for context, limit in [('short', 384), ('long', 1280)]:
            for variant in (0, 1):
                key = task + ':' + context + ':' + str(variant)
                text, used = demonstration_head(tokenizer, demos, limit, variant)
                common = tokenizer.encode(text, add_special_tokens=False)
                for c in dev + ev:
                    full = tokenizer.encode(text + c['raw_prompt'] + '\nAnswer:', add_special_tokens=False)
                    common = longest_prefix(common, full)
                if len(common) < 16:
                    raise ValueError('Shared demonstration prefix is unexpectedly short')
                result['heads'][key] = dict(text=text, tokens=common, token_hash=identity(common),
                                            task=task, context=context, demo_case_ids=used)
    for split, group in [('calibration', 2), ('evaluation', 2 if mode == 'verify' else 4)]:
        for context_index, context in enumerate(('short', 'long')):
            offsets = dict.fromkeys(TASKS, 0)
            for phase in range(4):
                order = ([task for task in TASKS for _ in range(group)] if phase in (0, 3)
                         else [task for _ in range(group) for task in TASKS])
                for task in order:
                    pool = original['tasks'][task]['fit'][8:16] if split == 'calibration' else original['tasks'][task]['eval'][:result['raw_query_count_per_task']]
                    # Calibration and verify intentionally reuse a small pool;
                    # pilot has one distinct raw query per task/context slot.
                    index = context_index * 4 * group + offsets[task]
                    offsets[task] += 1
                    c = pool[index % len(pool)]
                    head_id = task + ':' + context + ':' + str(int(phase >= 2))
                    h = result['heads'][head_id]
                    query = c['raw_prompt'] + '\nAnswer:'
                    prefix = tokenizer.encode(h['text'] + query, add_special_tokens=False)
                    if prefix[:len(h['tokens'])] != h['tokens'] or len(prefix) + max(map(len, c['label_ids'].values())) > 1792:
                        raise ValueError('Request prefix contract/budget mismatch: ' + c['case_id'])
                    request_key = identity([task, prefix, c['label_ids']])
                    result[split].append(dict(position=len(result[split]), task=task, context=context,
                        phase=('sticky', 'churn', 'new_context', 'sticky_after_change')[phase],
                        head_id=head_id, case_id=c['case_id'], query=query, gold=c['gold'],
                        label_ids=c['label_ids'], prefix=prefix, request_key=request_key))
    result['unique_evaluation_requests'] = len({c['request_key'] for c in result['evaluation']})
    result['signature'] = identity(result)
    return result


def read_workload(root):
    workload = read(root / 'workload.json')
    signature = workload.pop('signature')
    if identity(workload) != signature:
        raise ValueError('Workload signature changed')
    workload['signature'] = signature
    return workload


class CacheBank:
    def __init__(self, engine, cap=BANK_CAP):
        self.engine, self.cap = engine, cap
        self.data = collections.OrderedDict()
        self.evictions = 0

    @property
    def bytes(self):
        return sum(self.engine.cache_bytes(c) for c in self.data.values())

    def key(self, version, head):
        return version, head['token_hash']

    def get(self, version, head, touch=True):
        key = self.key(version, head)
        value = self.data.get(key)
        if value is not None and touch:
            self.data.move_to_end(key)
        return value

    def trim(self, allowance):
        allowance = max(0, min(self.cap, allowance))
        while self.data and self.bytes > allowance:
            self.data.popitem(last=False)
            self.evictions += 1

    def put(self, version, head, cache, allowance):
        amount = self.engine.cache_bytes(cache)
        allowance = max(0, min(self.cap, allowance))
        if amount > allowance:
            return False
        key = self.key(version, head)
        self.data.pop(key, None)
        self.trim(allowance - amount)
        self.data[key] = cache
        return True


def clone_prefix(cache, length):
    value = copy.deepcopy(cache)
    value.crop(length)
    # crop may leave views backed by the whole query storage. Compact them.
    for layer in value.layers:
        for name in ('keys', 'values'):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                setattr(layer, name, tensor.clone().contiguous())
    if value.get_seq_length() != length:
        raise ValueError('Stored prefix length mismatch')
    return value


def label_scores(torch, engine, model, prefix, labels, head_length, template=None,
                 save_prefix=False, device='cuda:0', live_check=None):
    """Average candidate token log likelihood; labels themselves are not gold."""
    tail = engine.tail_argument(model)
    if tail is None:
        raise ValueError('Tail-logit forward is required')
    def check(*caches):
        if live_check is not None:
            live_check(caches)
    with torch.inference_mode():
        working = copy.deepcopy(template) if template is not None else None
        if template is not None and template.get_seq_length() != head_length:
            raise ValueError('Wrong shared prefix length')
        ids = prefix[head_length:] if working is not None else prefix
        if not ids:
            raise ValueError('Empty query suffix')
        output = model(input_ids=torch.tensor([ids], device=device), past_key_values=working,
                       use_cache=True, **{tail: 1})
        working = output.past_key_values
        first = output.logits[0, -1].float()
        del output
        check(working)
        scores = {}
        for label in sorted(labels):
            tokens = labels[label]
            if not tokens:
                raise ValueError('Empty label')
            branch = copy.deepcopy(working) if len(tokens) > 1 else None
            logits = first
            total = torch.zeros((), device=device, dtype=torch.float32)
            for i, token in enumerate(tokens):
                total = total + torch.log_softmax(logits, dim=-1)[token]
                if i + 1 < len(tokens):
                    output = model(input_ids=torch.tensor([[token]], device=device), past_key_values=branch,
                                   use_cache=True, **{tail: 1})
                    branch = output.past_key_values
                    logits = output.logits[0, -1].float()
                    del output
                    check(working, branch)
            scores[label] = float((total / len(tokens)).item())
            del branch
        # Return the request cache only when the caller will extract its head;
        # never perform an extra model prefill merely to populate the bank.
        stored = working if save_prefix else None
        check(working)
    ranked = sorted(scores, key=lambda x: (-scores[x], x))
    if any(not math.isfinite(x) for x in scores.values()):
        raise FloatingPointError('Non-finite classification score')
    return dict(scores=scores, prediction=ranked[0], margin=scores[ranked[0]]-scores[ranked[1]]), stored


class WeightState:
    def __init__(self, engine, torch, model, factors, targets, native=False, device='cuda:0'):
        self.engine, self.torch, self.model = engine, torch, model
        self.factors, self.targets, self.native, self.device = factors, targets, native, device
        self.params = dict(model.named_parameters())
        self.task = TASKS[0]
        self.version = self.canonical(self.task)
        self.switches = self.since_exact = self.restores = 0
        # Exactly one initialization before the trace, never one per request.
        if native:
            model.set_adapter(self.task)
            model.eval()
        else:
            engine.install_host_target(torch, self.params, targets[self.task])
        self.sync()

    def sync(self):
        self.engine.sync(self.torch, self.device)

    def canonical(self, task):
        return ('native:' if self.native else 'merged:') + self.engine.LOCAL_TARGET_HASHES[task]

    def apply(self, target, exact=False):
        if self.native:
            if self.task != target:
                self.model.set_adapter(target)
                self.model.eval()
                self.switches += 1
                action = 'native_select'
            else:
                action = 'none'
            self.task, self.version = target, self.canonical(target)
        elif exact and self.version != self.canonical(target):
            self.engine.install_host_target(self.torch, self.params, self.targets[target])
            self.switches += int(self.task != target)
            self.restores += 1
            self.since_exact = 0
            self.task, self.version = target, self.canonical(target)
            action = 'exact_restore'
        elif self.task != target:
            old_task, old_version = self.task, self.version
            self.factors.add(self.params, self.task, -1)
            self.factors.add(self.params, target, +1)
            self.task = target
            self.version = 'dirty:' + identity([old_version, old_task, target])
            self.switches += 1
            self.since_exact += 1
            action = 'increment'
        else:
            action = 'none'
        self.sync()
        return action


def choose_route(policy, state, bank, head, costs):
    target = head['task']
    if policy == 'fixed_inc':
        return 'inc_full', False
    if policy in ('simple_rule', 'hf_native'):
        return 'inc_auto', True
    if policy == 'fixed_exact':
        return 'exact_auto', True
    c = costs[target + ':' + head['context']]
    inc_cost = 0.0 if state.task == target else costs['increment_ms']
    restore = 0.0 if state.version == state.canonical(target) else costs['restore_ms']
    options = [('inc_full', inc_cost + c['full_ms'])]
    if state.task == target and bank.get(state.version, head, touch=False) is not None:
        options.append(('keep_cached', c['cached_ms']))
    if bank.get(state.canonical(target), head, touch=False) is not None:
        options.append(('exact_cached', restore + c['cached_ms']))
    options.append(('exact_full', restore + c['full_ms']))
    priority = {'keep_cached':0, 'inc_full':1, 'exact_cached':2, 'exact_full':3}
    route = min(options, key=lambda item: (item[1], priority[item[0]]))[0]
    # Fixed two-use amortization, fitted only to development component costs.
    # This is a heuristic, not foreknowledge of a future evaluation request.
    store = c['cached_ms'] + c['store_ms'] / 2 < c['full_ms']
    return route, store


class Server:
    def __init__(self, engine, torch, model, factors, targets, tokenizer, policy,
                 costs=None, device='cuda:0', live_cap=LIVE_CAP, bank_cap=BANK_CAP):
        self.engine, self.torch, self.model = engine, torch, model
        self.tokenizer, self.policy, self.costs = tokenizer, policy, costs
        self.device, self.live_cap = device, live_cap
        self.state = WeightState(engine, torch, model, factors, targets, policy == 'hf_native', device)
        self.bank = CacheBank(engine, bank_cap)
        self.model_bytes = engine.model_payload_bytes(model) + (factors.bytes if factors is not None else 0)
        cfg = model.config
        itemsize = next(model.parameters()).element_size()
        self.bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads) * itemsize
        self.observed_live_max = self.model_bytes

    def live_check(self, caches):
        tensors = []
        for c in list(self.bank.data.values()) + list(caches):
            tensors.extend(self.engine.cache_tensors(c))
        amount = self.model_bytes + self.engine.tensor_storage_bytes(tensors)
        self.observed_live_max = max(self.observed_live_max, amount)
        if amount > self.live_cap:
            raise ValueError('Observed live model/factor/KV tensors exceed logical cap')

    def serve(self, head, query, label_ids):
        self.state.sync()
        started = time.perf_counter_ns()
        prefix = self.tokenizer.encode(head['text'] + query, add_special_tokens=False)
        if prefix[:len(head['tokens'])] != head['tokens']:
            raise ValueError('Request does not share its recorded context prefix')
        after_tokenize = time.perf_counter_ns()
        max_length = len(prefix) + max(map(len, label_ids.values()))
        working_reserve = 2 * max_length * self.bytes_per_token + RESERVE
        allowance = self.live_cap - self.model_bytes - working_reserve
        if allowance < 0:
            raise ValueError('Request working-cache budget rejected before forward')
        self.bank.trim(allowance)
        before_version = self.state.version
        route, save = choose_route(self.policy, self.state, self.bank, head, self.costs)
        after_decision = time.perf_counter_ns()
        parameter_action = self.state.apply(head['task'], exact=route.startswith('exact'))
        after_parameter = time.perf_counter_ns()
        template = None if route in ('inc_full', 'exact_full') else self.bank.get(self.state.version, head)
        if route in ('keep_cached', 'exact_cached') and template is None:
            raise ValueError('Chosen KV state is unavailable')
        hit = template is not None
        result, working = label_scores(self.torch, self.engine, self.model, prefix, label_ids,
            len(head['tokens']), template, save and not hit, self.device, self.live_check)
        self.state.sync()
        after_score = time.perf_counter_ns()
        # Extract from the completed request cache; no extra head prefill.
        admitted = False
        stored = None
        if save and not hit:
            amount = len(head['tokens']) * self.bytes_per_token
            if amount <= min(self.bank.cap, allowance):
                self.bank.trim(allowance - amount)
                stored = clone_prefix(working, len(head['tokens']))
                self.live_check((working, stored))
                admitted = self.bank.put(self.state.version, head, stored, allowance)
        del working, stored
        self.state.sync()
        finished = time.perf_counter_ns()
        return dict(result, route=route, parameter_action=parameter_action,
            state_before=before_version, state_after=self.state.version,
            cache_hit=hit, cache_admitted=admitted, cache_bytes=self.bank.bytes,
            cache_evictions=self.bank.evictions, incremental_switches_since_exact=self.state.since_exact,
            total_switches=self.state.switches, total_exact_restores=self.state.restores,
            prefix_tokens=len(prefix), head_tokens=len(head['tokens']),
            logical_live_max_bytes=self.observed_live_max,
            tokenize_ms=(after_tokenize-started)/1e6, decision_ms=(after_decision-after_tokenize)/1e6,
            parameter_ms=(after_parameter-after_decision)/1e6,
            score_ms=(after_score-after_parameter)/1e6, store_ms=(finished-after_score)/1e6,
            request_ms=(finished-started)/1e6)


def model_for_policy(engine, torch, spec, policy):
    source = Path(spec['source_result'])
    targets, _ = engine.target_cache_load(torch, source)
    if policy == 'hf_native':
        from peft import PeftModel
        model = engine.load_profile_base(spec, torch)
        model = PeftModel.from_pretrained(model, Path(spec['taskpool_dir'])/'lora'/'mrpc',
            adapter_name='mrpc', is_trainable=False, autocast_adapter_dtype=False, local_files_only=True)
        for task in ('qnli', 'agnews'):
            model.load_adapter(Path(spec['taskpool_dir'])/'lora'/task, adapter_name=task,
                               is_trainable=False, autocast_adapter_dtype=False, local_files_only=True)
        model.to(dtype=torch.float16).eval()
        return model, None, targets
    model, factors = engine.merged_model_and_factors(spec, source, torch, True)
    return model, factors, targets


def worker(root):
    import torch
    from transformers import AutoTokenizer
    engine = load_engine(root/'frozen_kv_engine.py')
    spec, active = read(root/'spec.json'), read(root/'active_stage.json')
    workload = read_workload(root)
    torch.manual_seed(0)
    engine.torch_runtime(torch)
    policy = active.get('policy', 'fixed_exact')
    model, factors, targets = model_for_policy(engine, torch, spec, policy)
    tokenizer = AutoTokenizer.from_pretrained(spec['model_dir'], local_files_only=True, trust_remote_code=False)
    costs = read(root/'plan.json')['costs'] if policy == 'joint' else None
    # Development-only shape warmup before the single initial state install.
    # No query or template from this warmup is retained in the serving bank.
    warmed = set()
    for c in workload['calibration']:
        key = c['task'], c['context']
        if key in warmed:
            continue
        warmed.add(key)
        if policy == 'hf_native':
            model.set_adapter(c['task']); model.eval()
        label_scores(torch, engine, model, c['prefix'], c['label_ids'],
                     len(workload['heads'][c['head_id']]['tokens']), device='cuda:0')
    if factors is not None:
        params = dict(model.named_parameters())
        factors.add(params, 'mrpc', -1)
        factors.add(params, 'qnli', +1)
    server = Server(engine, torch, model, factors, targets, tokenizer, policy, costs)
    engine.require_idle(os.getpgrp())
    torch.cuda.reset_peak_memory_stats()
    data = workload['calibration' if active['kind'] == 'calibration' else 'evaluation']
    tag = active['tag']
    output = []
    if active['kind'] == 'reference':
        seen = set()
        for c in data:
            if c['request_key'] in seen:
                continue
            seen.add(c['request_key'])
            # Independent reference worker, never interleaved with the tested state.
            server.state.apply(c['task'], exact=True)
            result, _ = label_scores(torch, engine, model, c['prefix'], c['label_ids'],
                len(workload['heads'][c['head_id']]['tokens']), device='cuda:0', live_check=server.live_check)
            output.append(dict(request_key=c['request_key'], gold=c['gold'], **result))
            engine.require_idle(os.getpgrp())
    else:
        for c in data:
            engine.require_idle(os.getpgrp())
            h = workload['heads'][c['head_id']]
            result = server.serve(h, c['query'], c['label_ids'])
            output.append(dict(position=c['position'], case_id=c['case_id'], request_key=c['request_key'],
                head_id=c['head_id'], task=c['task'], context=c['context'], phase=c['phase'],
                gold=c['gold'], policy=policy, **result))
            if len(output) % 8 == 0:
                print(tag, 'completed requests:', len(output), flush=True)
    engine.require_idle(os.getpgrp())
    write_rows(root/(tag+'_requests.jsonl'), output)
    dump(root/(tag+'_runtime.json'), dict(engine.torch_runtime(torch),
        logical_live_max_bytes=server.observed_live_max, model_and_factor_bytes=server.model_bytes,
        cache_bank_cap_bytes=BANK_CAP, logical_live_cap_bytes=LIVE_CAP,
        allocator_peak_bytes=torch.cuda.max_memory_allocated(), reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        nvml=engine.nvml_owned_snapshot(), requests=len(output), policy=policy,
        strict_physical_memory_claim=False, per_request_exact_source_reset=False))


def fit_plan(root):
    calibrated = {}
    records = []
    for policy in POLICIES[:-1]:
        rr = rows(root/('cal_'+policy+'_requests.jsonl'))
        calibrated[policy] = statistics.mean(r['request_ms'] for r in rr)
        if policy != 'hf_native':
            records.extend(rr)
    def median(values):
        if not values:
            raise ValueError('Calibration lacks a required path component')
        return statistics.median(values)
    costs = dict(increment_ms=median([r['parameter_ms'] for r in records if r['parameter_action']=='increment']),
                 restore_ms=median([r['parameter_ms'] for r in records if r['parameter_action']=='exact_restore']))
    for task in TASKS:
        for context in ('short', 'long'):
            rr = [r for r in records if r['task']==task and r['context']==context]
            costs[task+':'+context] = dict(
                full_ms=median([r['score_ms'] for r in rr if not r['cache_hit']]),
                cached_ms=median([r['score_ms'] for r in rr if r['cache_hit']]),
                store_ms=median([r['store_ms'] for r in rr if r['cache_admitted']]))
    plan = dict(version=VERSION, costs=costs,
        calibrated_fixed=min(calibrated, key=lambda p:(calibrated[p], p)),
        calibrated_means_ms=calibrated, source_split='policy_dev only',
        workload_signature=read_workload(root)['signature'],
        policy_description='Online cost-based state/KV choice; two-use cache admission amortization. No quality certificate, no evaluation labels or future requests supplied to chooser.')
    dump(root/'plan.json', plan)


def schedule(mode):
    result = [dict(kind='calibration', policy=p, tag='cal_'+p) for p in POLICIES[:-1]]
    result += [dict(kind='plan', tag='freeze_plan'), dict(kind='reference', tag='reference')]
    for i in range(1 if mode == 'verify' else 2):
        for policy in (POLICIES if i == 0 else tuple(reversed(POLICIES))):
            result.append(dict(kind='evaluation', round=i, policy=policy, tag='r'+str(i)+'_'+policy))
    return result


def run_stage(engine, root, active, timeout):
    engine.require_idle(samples=3)
    dump(root/'active_stage.json', active)
    tag = active['tag']
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', FTD_TRANSFER_CHUNK_MB='1')
    print('Starting', tag, '; log:', root/'logs'/(tag+'.log'), flush=True)
    with (root/'logs'/(tag+'.log')).open('w') as log:
        proc = subprocess.Popen([sys.executable, str(root/'ftd_continuous_state_20261102.py'),
                                 '--worker', '--run-dir', str(root)], stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        event = threading.Event()
        monitor = threading.Thread(target=engine.monitor_stage,
            args=(event, root/(tag+'_telemetry.jsonl'), proc.pid), daemon=True)
        monitor.start()
        started = time.monotonic()
        try:
            while proc.poll() is None:
                if time.monotonic()-started > timeout:
                    raise TimeoutError(tag+' exceeded timeout')
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print(tag, 'running; see log', flush=True)
            if proc.returncode:
                text = (root/'logs'/(tag+'.log')).read_text(errors='replace')
                print('\n'.join(text.splitlines()[-14:]), flush=True)
                if 'ProbeBlocked:' in text:
                    raise engine.ProbeBlocked(tag+' detected external GPU work; partial stages retained')
                raise RuntimeError(tag+' failed; full log included in result archive')
        finally:
            event.set()
            monitor.join(timeout=5)
            engine.stop_owned_process(proc)


def percentile(values, fraction):
    values = sorted(values)
    at = (len(values)-1)*fraction
    low, high = math.floor(at), math.ceil(at)
    return values[low] + (values[high]-values[low])*(at-low)


def summarize(root):
    spec, workload, plan = read(root/'spec.json'), read_workload(root), read(root/'plan.json')
    refs = {r['request_key']:r for r in rows(root/'reference_requests.jsonl')}
    expected = workload['evaluation']
    metrics, traces, flags = [], {}, []
    for stage in schedule(spec['mode']):
        if stage['kind'] not in ('evaluation', 'calibration', 'reference'):
            continue
        tag = stage['tag']
        samples = rows(root/(tag+'_telemetry.jsonl'))
        if not samples or any(r.get('status')!='ok' or r.get('other_compute_process_count',0) for r in samples):
            flags.append(tag+': isolation/telemetry not confirmed')
        if stage['kind'] != 'evaluation':
            continue
        rr = rows(root/(tag+'_requests.jsonl'))
        if len(rr) != len(expected):
            raise ValueError('Incomplete evaluation trace: '+tag)
        for r, c in zip(rr, expected):
            if any(r[k]!=c[k] for k in ('position','request_key','case_id','gold','task','head_id')):
                raise ValueError('Evaluation trace changed: '+tag)
            if not math.isfinite(r['request_ms']) or r['request_ms']<=0:
                raise ValueError('Invalid request timing')
            if r['logical_live_max_bytes'] > LIVE_CAP:
                raise ValueError('Logical tensor cap exceeded in recorded request')
        if any(b['state_before']!=a['state_after'] for a,b in zip(rr,rr[1:])):
            raise ValueError('Unrecorded parameter-state reset between requests: '+tag)
        if stage['policy'] in ('fixed_inc','simple_rule') and any(r['parameter_action']=='exact_restore' for r in rr):
            raise ValueError('Unexpected exact reset in continuous incremental control: '+tag)
        times = [r['request_ms'] for r in rr]
        unique = {r['request_key']:r for r in rr}
        metric = dict(round=stage['round'], policy=stage['policy'], requests=len(rr),
            unique_context_queries=len(unique), mean_ms=statistics.mean(times), p95_ms=percentile(times,.95),
            accuracy=sum(r['prediction']==r['gold'] for r in rr)/len(rr),
            accuracy_last_observation_per_unique_context_query=sum(r['prediction']==r['gold'] for r in unique.values())/len(unique),
            reference_disagreements=sum(r['prediction']!=refs[r['request_key']]['prediction'] for r in rr),
            max_abs_score_difference_vs_reference=max(abs(v-refs[r['request_key']]['scores'][k]) for r in rr for k,v in r['scores'].items()),
            cache_hits=sum(r['cache_hit'] for r in rr), cache_admissions=sum(r['cache_admitted'] for r in rr),
            parameter_actions=dict(collections.Counter(r['parameter_action'] for r in rr)),
            routes=dict(collections.Counter(r['route'] for r in rr)),
            total_decision_ms=sum(r['decision_ms'] for r in rr),
            total_request_ms=sum(times), logical_live_max_bytes=max(r['logical_live_max_bytes'] for r in rr))
        metric['per_task'] = {}
        for task in TASKS:
            subset = [r for r in rr if r['task']==task]
            metric['per_task'][task] = dict(requests=len(subset),
                correct=sum(r['prediction']==r['gold'] for r in subset),
                accuracy=sum(r['prediction']==r['gold'] for r in subset)/len(subset),
                mean_ms=statistics.mean(r['request_ms'] for r in subset),
                reference_disagreements=sum(r['prediction']!=refs[r['request_key']]['prediction'] for r in subset))
        metrics.append(metric)
        traces[stage['round'], stage['policy']] = rr
    comparisons = []
    for i in range(1 if spec['mode']=='verify' else 2):
        get = lambda policy: next(m for m in metrics if m['round']==i and m['policy']==policy)
        joint = get('joint')
        for baseline in POLICIES[:-1]:
            b = get(baseline)
            # Match meaningful executed operations, not the route name strings.
            action = lambda r:(r['parameter_action'],r['cache_hit'],r['cache_admitted'],r['state_after'])
            equal = all(action(a)==action(b) for a,b in zip(traces[i,'joint'],traces[i,baseline]))
            comparisons.append(dict(round=i, baseline=baseline,
                mean_reduction_fraction=1-joint['mean_ms']/b['mean_ms'],
                p95_reduction_fraction=1-joint['p95_ms']/b['p95_ms'],
                accuracy_difference=joint['accuracy']-b['accuracy'],
                per_task_accuracy_difference={t:joint['per_task'][t]['accuracy']-b['per_task'][t]['accuracy'] for t in TASKS},
                executed_actions_identical=equal))
    stability = []
    if spec['mode'] == 'pilot':
        for policy in POLICIES:
            for task in TASKS:
                for context in ('short','long'):
                    means = [statistics.mean(r['request_ms'] for r in traces[i,policy]
                             if r['task']==task and r['context']==context) for i in (0,1)]
                    ratio = max(means)/min(means)
                    stability.append(dict(policy=policy,task=task,context=context,round_mean_ratio=ratio))
                    if ratio > 1.20:
                        flags.append(policy+'/'+task+'/'+context+': round mean ratio >1.20')
    selected = [c for c in comparisons if c['baseline'] in {plan['calibrated_fixed'],'simple_rule'}]
    if spec['mode']=='verify':
        decision = 'VERIFY_ONLY_NOT_A_METHOD_CONCLUSION'
    elif flags:
        decision = 'INCONCLUSIVE_ISOLATION_OR_TIMING'
    elif any(m['reference_disagreements'] for m in metrics if m['policy']=='joint'):
        decision = 'NO_GO_REFERENCE_FIDELITY_REQUIRES_REVIEW'
    elif any(c['executed_actions_identical'] for c in selected):
        decision = 'NO_GO_NO_DISTINCT_GAIN_AGAINST_FIXED_OR_SIMPLE_RULE'
    elif all(c['mean_reduction_fraction']>=.05 and c['p95_reduction_fraction']>=-.05 and
             all(v>=0 for v in c['per_task_accuracy_difference'].values()) for c in selected):
        decision = 'CANDIDATE_ONLY_NEEDS_LARGER_QUALITY_STUDY'
    else:
        decision = 'NO_GO_NO_STABLE_EXTRA_GAIN_ON_THIS_TRACE'
    result = dict(status=decision, metrics=metrics, comparisons=comparisons,
        calibrated_fixed=plan['calibrated_fixed'], flags=flags, round_stability=stability,
        reference_accuracy=sum(r['prediction']==r['gold'] for r in refs.values())/len(refs),
        scope='Real task labels and development-only few-shot contexts; constructed synchronous trace; no queue, network, or open-ended dialog evaluation',
        continuous_state_measured=True, online_chooser_executed=True,
        true_labels_evaluated=True, application_quality_noninferiority_established=False,
        controller_benefit_demonstrated=False, novelty_established=False, strict_physical_memory_claim=False,
        screen='A pilot candidate needs >=5% mean reduction versus both development-selected fixed and simple rule in both rounds, <=5% p95 regression, no lower observed per-task accuracy, no reference disagreement, and distinct meaningful actions. Engineering screen, not significance or quality certification.',
        note='Same raw cases are served across policies and process rounds; counts are not independent samples. Reference agreement and true-label correctness are separate. No policy observes evaluation gold, reference scores, future trace, or phase annotations.')
    dump(root/'continuous_summary.json', result)
    return result


def pack(root):
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts]
    dump(root/'manifest.json', {str(p.relative_to(root)):sha(p) for p in files})
    archive = root.with_name(root.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for p in files+[root/'manifest.json']:
            tf.add(p, arcname=root.name+'/'+str(p.relative_to(root)), recursive=False)
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify','pilot'), nargs='?', default='verify')
    parser.add_argument('--kv-result', type=Path, default=DEFAULT_KV)
    parser.add_argument('--output-dir', type=Path, default=Path.home()/'ftd_continuous_outputs')
    parser.add_argument('--stage-timeout', type=int, default=1800)
    parser.add_argument('--then-pilot', action='store_true', help='After verify passes execution/isolation and exact-cache reference checks, start a fresh two-round pilot')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.run_dir.resolve())
        return 0
    if args.stage_timeout < 30:
        parser.error('Stage timeout must be at least 30 seconds')
    if args.then_pilot and args.mode != 'verify':
        parser.error('--then-pilot is only valid with verify')
    parent = args.output_dir.expanduser(); parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = Path(tempfile.mkdtemp(prefix='ftd_continuous_'+args.mode+'_'+stamp+'_',dir=parent)).resolve()
    (root/'logs').mkdir()
    shutil.copyfile(Path(__file__), root/'ftd_continuous_state_20261102.py')
    state, engine, rc = dict(status='RUNNING', stages=[]), None, 0
    try:
        previous = args.kv_result.expanduser().resolve()
        check_inventory(previous)
        if read(previous/'status.json')['status'] != 'COMPLETED_FEASIBILITY_SCREEN':
            raise ValueError('Pass the completed KV probe directory, not a tar.gz or blocked result')
        shutil.copyfile(previous/'ftd_kv_feasibility_20261101.py', root/'frozen_kv_engine.py')
        engine = load_engine(root/'frozen_kv_engine.py')
        old = read(previous/'spec.json')
        source = Path(old['source_result'])
        if read(source/'status.json')['status']!='COMPLETED_COMPLETE_REQUEST_DIAGNOSTIC' or sha(source/'ftd_complete_request_20261101.py')!=SOURCE_SCRIPT_SHA:
            raise ValueError('Original reference source identity changed')
        for name,h in engine.SOURCE_HASHES.items():
            if sha(source/name)!=h:
                raise ValueError('Original reference metadata changed: '+name)
        spec = dict(version=VERSION, mode=args.mode, scope='smoke',
                    model_dir=old['model_dir'], taskpool_dir=old['taskpool_dir'], source_result=old['source_result'],
                    rounds=1 if args.mode=='verify' else 2,
                    context_demo_token_limits=[384,1280], max_prefix_plus_label_tokens=1792,
                    reused_previously_observed_evaluation=True,
                    script_sha256=sha(Path(__file__)), original_kv_result=str(previous),
                    original_kv_manifest_sha256=sha(previous/'manifest.json'),
                    logical_live_cap_bytes=LIVE_CAP, bank_cap_bytes=BANK_CAP,
                    decode_contract='Mean teacher-forced candidate-label log likelihood; no fixed 16-token generation',
                    synthetic_probe=False, continuous_state=True)
        dump(root/'spec.json',spec)
        engine.preflight(spec,root)
        gpus = read(root/'preflight.json')['gpu'].get('gpus',[])
        if len(gpus)!=1 or gpus[0]['index']!='0' or os.environ.get('CUDA_VISIBLE_DEVICES','0')!='0':
            raise ValueError('Single physical GPU 0 mapping required')
        engine.require_idle(samples=3)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(spec['model_dir'],local_files_only=True,trust_remote_code=False)
        dump(root/'workload.json',make_workload(engine,source,tokenizer,args.mode))
        for active in schedule(args.mode):
            dump(root/'status.json',dict(state,active=active))
            if active['kind']=='plan':
                fit_plan(root)
            else:
                run_stage(engine,root,active,args.stage_timeout)
            state['stages'].append(dict(stage=active['tag'],status='COMPLETED'))
        report = summarize(root)
        state.update(status='COMPLETED_CONTINUOUS_'+args.mode.upper(),decision=report['status'])
        if args.mode == 'verify':
            state['automatic_pilot_ready'] = (not report['flags'] and all(
                x['reference_disagreements']==0 for x in report['metrics'] if x['policy']=='fixed_exact'))
    except (Exception,KeyboardInterrupt) as e:
        if engine is not None and isinstance(e,engine.ProbeBlocked):
            state.update(status='BLOCKED_GPU_ISOLATION',reason=str(e));rc=3
        else:
            state.update(status='FAILED',error_type=type(e).__name__,error=str(e));rc=2
            (root/'error.txt').write_text(traceback.format_exc())
        print('Stopped:',str(e),flush=True)
    finally:
        dump(root/'status.json',state)
        (root/'READ_ME.txt').write_text('Continuous-state diagnostic. Model/reference weights read in place, never included.\nClassification candidate scoring, not the previous fixed-16-token generation benchmark.\nNo quality or novelty guarantee; preserve failures and send the archive.\n')
        archive = pack(root)
        print('STATUS:',state['status'])
        print('SEND_THIS_ARCHIVE:',archive,flush=True)
    if rc == 0 and args.then_pilot:
        if state.get('automatic_pilot_ready'):
            print('Verify execution checks passed; starting fresh two-round pilot. This is not a positive method conclusion.',flush=True)
            return main(['pilot','--kv-result',str(previous),'--output-dir',str(parent),
                         '--stage-timeout',str(args.stage_timeout)])
        print('Verify requires review; pilot was not started. Send the verify archive.',flush=True)
    return rc


if __name__=='__main__':
    sys.exit(main())
