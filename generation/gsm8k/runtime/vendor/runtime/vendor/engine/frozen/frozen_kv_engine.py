#!/usr/bin/env python3
"""FTD parameter/KV coupling feasibility screen; synthetic inputs, no quality or policy-benefit claim."""

from contextlib import ExitStack, contextmanager

EXPECTED_POOL_IDENTITIES = {'mrpc': {'config_sha256': '733f2555c7df69f97dadde08376144be7fe8b1970e3c436a5756d4010ff4ef03', 'training_contract_sha256': '44ef1372adafebf25fbc904ac2c1a85c28312fbfcbf7c8b7372ae4332bfac779', 'policy_dev_sha256': '37dc82ae610b787f28bbf70741a2a5cad6654f4ea433032c807bac451501d239', 'weight_files': {'adapter_model.safetensors': '7936d7b900f580d3e3b75d6c92c894924ebc64eae25533318d167bf103b48d82'}}, 'qnli': {'config_sha256': 'fe294396373e06a6c33fd7e906c0f40fc73e7d6b6795ae82453b0cad0de64488', 'training_contract_sha256': 'fb9ec909f3ff612dbd4b6d9a161efcadab406583bf24db18ab85bae498701207', 'policy_dev_sha256': '41802c1bc910cd27f131607032aeb1b8e5ea230f1b32efc9eb3a95868575156e', 'weight_files': {'adapter_model.safetensors': 'ea9eb356d9bb13fdfd2e4b149d81c6bb78c8dfb601ba75a0bc5b5ff3a56ffb5b'}}, 'agnews': {'config_sha256': 'aab85f4688e0d218939d8769b0eb62b7a8ef93e16f356f8ba96d6840b4ccdccb', 'training_contract_sha256': '3eb86f1c04afb7a95afe069389da620309c287113cd0f6acfb63e51e164c670a', 'policy_dev_sha256': 'dbd24c45750e22c30749813cd860ff47e73bd7c95767f671596cdd536d31cc91', 'weight_files': {'adapter_model.safetensors': '0c0eacae87cdc8bfa18cebafd7cc018bf781f452fedcd15f6b5eca5c0538ce8e'}}}

import argparse

import collections

import csv

import datetime

import hashlib

import importlib.metadata

import io

import json

import math

import os

from pathlib import Path

import platform

import re

import shutil

import struct

import subprocess

import sys

import tarfile

import tempfile

import time

MIB = 1024 * 1024

TASKS = ('sentiment', 'nli', 'boolq', 'mrpc', 'qnli', 'agnews')

RANKS = dict(sentiment=8, nli=8, boolq=8, mrpc=4, qnli=16, agnews=8)

PACKAGES = ('torch', 'transformers', 'peft', 'vllm', 'sglang', 'triton',
            'flashinfer-python', 'flashinfer-cubin', 'flashinfer-jit-cache',
            'safetensors', 'accelerate', 'tokenizers', 'numpy')

BASE_HASHES = {
    'model.safetensors': 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b',
    'config.json': '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd',
    'generation_config.json': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
    'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
}

DTYPE_BITS = {'BOOL': 8, 'U8': 8, 'I8': 8, 'I16': 16, 'U16': 16,
              'I32': 32, 'U32': 32, 'I64': 64, 'U64': 64,
              'F16': 16, 'BF16': 16, 'F32': 32, 'F64': 64,
              'F8_E4M3': 8, 'F8_E5M2': 8, 'F8_E8M0': 8}

BAD_LORA = ('use_dora', 'fan_in_fan_out', 'lora_bias', 'modules_to_save',
            'rank_pattern', 'alpha_pattern', 'layer_replication',
            'trainable_token_indices', 'target_parameters', 'alora_invocation_tokens',
            'use_qalora', 'arrow_config', 'use_bdlora', 'use_rslora')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(MIB), b''):
            h.update(chunk)
    return h.hexdigest()

def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key: ' + str(key))
        out[key] = value
    return out

def read_json(path, max_bytes=256*MIB):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError('JSON exceeds metadata size limit')
    value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value

def header(path):
    """Validate safetensors structure using its header, without loading tensor data."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('Short safetensors prefix')
        n = struct.unpack('<Q', raw)[0]
        if n < 2 or n > 16*MIB or n > size-8:
            raise ValueError('Invalid safetensors header length')
        obj = json.loads(f.read(n).decode('utf-8'), object_pairs_hook=unique_object)
    if not isinstance(obj, dict):
        raise ValueError('Invalid safetensors JSON')
    entries, intervals = {}, []
    payload_size = size-8-n
    for name, meta in obj.items():
        if name == '__metadata__':
            continue
        if not isinstance(meta, dict):
            raise ValueError('Invalid tensor metadata')
        shape, dtype, offsets = meta.get('shape'), meta.get('dtype'), meta.get('data_offsets')
        if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError('Invalid tensor shape')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError('Invalid tensor offsets')
        start, end = offsets
        if not (0 <= start <= end <= payload_size):
            raise ValueError('Tensor offsets outside file')
        numel = math.prod(shape)
        bits = DTYPE_BITS.get(dtype)
        if bits is None:
            raise ValueError('Unsupported serialized dtype: ' + str(dtype))
        if end-start != (numel*bits+7)//8:
            raise ValueError('Tensor byte size disagrees with dtype/shape')
        entries[name] = dict(shape=shape, dtype=dtype, numel=numel, bytes=end-start)
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError('Overlapping tensor data or unused gap')
        cursor = end
    if cursor != payload_size:
        raise ValueError('Unused safetensors trailing payload')
    return dict(file=path.name, file_bytes=size, header_bytes=n,
                tensor_count=len(entries), tensor_payload_bytes=payload_size,
                serialized_elements=sum(x['numel'] for x in entries.values()),
                dtype_counts=dict(collections.Counter(x['dtype'] for x in entries.values())),
                tensors=entries)

def capture_command(argv, timeout=15):
    if not shutil.which(argv[0]):
        return dict(status='not_found', executable=argv[0])
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                           check=False, errors='replace')
        # Do not copy stderr: sites can include account paths/environment in diagnostics.
        return dict(status='ok' if p.returncode == 0 else 'command_failed',
                    returncode=p.returncode, stdout=p.stdout.strip()[:16000])
    except subprocess.TimeoutExpired:
        return dict(status='timeout', executable=argv[0])
    except OSError as e:
        return dict(status='os_error', error_type=type(e).__name__)

def gpu_metadata():
    fields = ['index', 'name', 'compute_cap', 'driver_version',
              'memory.total', 'memory.used', 'memory.free']
    first = capture_command(['nvidia-smi', '--query-gpu='+','.join(fields),
                             '--format=csv,noheader,nounits'])
    attempts = [first]
    if first['status'] == 'command_failed':
        fields.remove('compute_cap')
        attempts.append(capture_command(['nvidia-smi', '--query-gpu='+','.join(fields),
                                         '--format=csv,noheader,nounits']))
    result = attempts[-1]
    gpus = []
    if result['status'] == 'ok':
        for row in csv.reader(io.StringIO(result.get('stdout', ''))):
            if len(row) == len(fields):
                gpus.append(dict(zip(fields, [x.strip() for x in row])))
    return dict(status=result['status'], gpus=gpus,
                attempts=[{k:v for k,v in x.items() if k != 'stdout'} for x in attempts],
                memory_unit='MiB; instantaneous device totals, not FTD retained-state bytes')

def software_metadata():
    result = {}
    for name in PACKAGES:
        try:
            dist = importlib.metadata.distribution(name)
            deps = []
            for req in dist.requires or []:
                # Collect only version constraints of relevant packages, never private URLs.
                if '@' in req or '://' in req:
                    continue
                if re.match(r'(?i)^(torch|transformers|triton|flashinfer|safetensors|peft)(\b|[-_])', req):
                    deps.append(req)
            result[name] = dict(installed=True, version=dist.version,
                                requires_python=dist.metadata.get('Requires-Python'),
                                selected_dependency_constraints=deps)
        except importlib.metadata.PackageNotFoundError:
            result[name] = dict(installed=False)
        except Exception as e:
            result[name] = dict(status='metadata_error', error_type=type(e).__name__)
    return result

def inspect_base(root, hash_weights=False):
    root = Path(root).expanduser().resolve()
    result = dict(path=str(root), exists=root.is_dir(), errors=[], frozen_identity={})
    tensors = {}
    if not root.is_dir():
        return result, tensors
    try:
        cfg = read_json(root/'config.json')
        keep = ('architectures','model_type','hidden_size','intermediate_size',
                'num_hidden_layers','num_attention_heads','num_key_value_heads',
                'vocab_size','tie_word_embeddings','torch_dtype','dtype',
                'max_position_embeddings','quantization_config')
        result['config'] = {k:cfg[k] for k in keep if k in cfg}
    except Exception as e:
        result['errors'].append('config.json: '+str(e))
    result['weight_files'] = []
    files = sorted(root.glob('*.safetensors'))
    for p in files[:256]:
        try:
            meta = header(p)
            duplicated = set(tensors)&set(meta['tensors'])
            if duplicated:
                raise ValueError('Tensor keys repeated across files')
            tensors.update(meta.pop('tensors'))
            result['weight_files'].append(meta)
        except Exception as e:
            result['errors'].append(p.name+': '+str(e))
    if not files:
        result['errors'].append('No safetensors weights; pickle/.bin is not loaded by this probe')
    if len(files) > 256:
        result['errors'].append('Weight-file limit exceeded; inventory incomplete')
    for name, expected in BASE_HASHES.items():
        path = root/name
        row = dict(expected_sha256=expected, exists=path.is_file())
        if path.is_file():
            row['bytes'] = path.stat().st_size
            if name.endswith('.safetensors') and not hash_weights:
                row['status'] = 'not_hashed_use_hash_base'
            else:
                row['sha256'] = sha256(path)
                row['status'] = 'match' if row['sha256'] == expected else 'MISMATCH'
        else:
            row['status'] = 'missing'
        result['frozen_identity'][name] = row
    result['all_frozen_base_hashes_match'] = all(
        x['status'] == 'match' for x in result['frozen_identity'].values())
    result['serialized_tensor_count'] = len(tensors)
    result['serialized_elements'] = sum(x['numel'] for x in tensors.values())
    result['serialized_payload_bytes'] = sum(x['bytes'] for x in tensors.values())
    result['estimated_fp16_tensor_payload_bytes'] = 2*result['serialized_elements']
    result['payload_note'] = 'File tensors only; tied-weight deduplication and engine padding/buffers are not runtime memory measurements.'
    return result, tensors

def dataset_metadata(path):
    path = Path(path)
    if not path.is_file():
        return dict(exists=False)
    h, rows, labels, invalid = hashlib.sha256(), 0, collections.Counter(), 0
    first_keys = None
    with path.open('rb') as f:
        for raw in f:
            h.update(raw)
            if not raw.strip():
                continue
            rows += 1
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError('not object')
                if first_keys is None:
                    first_keys = sorted(obj.keys())
                label = obj.get('label')
                # Whitelisted task labels only; do not copy prompt contents into reports.
                allowed = {'negative','positive','entailment','not_entailment','no','yes',
                           'not_equivalent','equivalent','world','sports','business','science'}
                labels[label if isinstance(label,str) and label in allowed else '<other_or_missing>'] += 1
            except (ValueError, UnicodeError):
                invalid += 1
    return dict(exists=True, bytes=path.stat().st_size, sha256=h.hexdigest(),
                rows=rows, invalid_rows=invalid, first_row_keys=first_keys,
                label_counts=dict(labels))

def inspect_adapter(directory, task, base_tensors):
    directory = Path(directory)
    out = dict(task=task, path=str(directory), exists=directory.is_dir(), errors=[])
    if not directory.is_dir():
        return out
    try:
        cfg = read_json(directory/'adapter_config.json')
        keep = ('peft_type','task_type','r','lora_alpha','lora_dropout','bias',
                'target_modules','init_lora_weights')+BAD_LORA
        out['config'] = {k:cfg[k] for k in keep if k in cfg}
        out['config_sha256'] = sha256(directory/'adapter_config.json')
        bad = [k for k in BAD_LORA if cfg.get(k)]
        if str(cfg.get('peft_type','')).upper() != 'LORA':
            bad.append('peft_type')
        if cfg.get('bias','none') != 'none':
            bad.append('bias')
        if cfg.get('init_lora_weights', True) not in (True, False, 'gaussian'):
            bad.append('base_transforming_initialization')
        rank = cfg.get('r')
        if type(rank) is not int or rank < 1:
            bad.append('rank')
        alpha = cfg.get('lora_alpha')
        if not isinstance(alpha, (int,float)) or not math.isfinite(alpha):
            bad.append('lora_alpha')
        out['requires_manual_variant_review'] = sorted(set(bad))
        out['rank_matches_recorded_task'] = rank == RANKS.get(task)
        out['scale_alpha_over_rank'] = alpha/rank if not bad and rank else None
    except Exception as e:
        out['errors'].append('adapter_config.json: '+str(e))
        return out
    tensors, groups, weight_files = {}, {}, []
    for p in sorted(directory.glob('*.safetensors'))[:64]:
        try:
            meta = header(p)
            if set(tensors)&set(meta['tensors']):
                raise ValueError('Repeated tensor keys across adapter files')
            tensors.update(meta.pop('tensors'))
            meta['sha256'] = sha256(p)
            weight_files.append(meta)
        except Exception as e:
            out['errors'].append(p.name+': '+str(e))
    if not tensors:
        out['errors'].append('No validated safetensors factors')
    for name, meta in tensors.items():
        match = re.fullmatch(r'base_model\.model\.(.+)\.lora_([AB])\.weight', name)
        if not match:
            out['errors'].append('Nonstandard factor key: '+name)
            continue
        weight, letter = match.group(1)+'.weight', match.group(2)
        groups.setdefault(weight, {})[letter] = meta
    for name, pair in groups.items():
        if set(pair) != {'A','B'}:
            out['errors'].append('Missing A/B pair: '+name)
            continue
        a, b = pair['A']['shape'], pair['B']['shape']
        if len(a) != 2 or len(b) != 2 or a[0] != rank or b[1] != rank:
            out['errors'].append('Rank/dimension mismatch: '+name)
            continue
        if base_tensors and (name not in base_tensors or base_tensors[name]['shape'] != [b[0],a[1]]):
            out['errors'].append('Base shape/key mismatch: '+name)
    out.update(weight_files=weight_files, pair_count=len(groups),
               tensor_count=len(tensors),
               fp16_factor_payload_bytes=2*sum(x['numel'] for x in tensors.values()),
               projection_pair_counts=dict(collections.Counter(n.split('.')[-2] for n in groups)),
               factor_tensors=tensors,
               base_shape_crosscheck_available=bool(base_tensors))
    out['ordinary_linear_metadata_checks_pass'] = bool(tensors) and not out['errors'] and not bad
    out['engine_load_verified'] = False
    training_path = directory/'training_contract.json'
    if training_path.is_file():
        try:
            training = read_json(training_path)
            keys = ('task','policy_dev_used_for_training','evaluation_used_for_training',
                    'smoke_only','training_data_sha256','seed','rank','lora_alpha')
            out['training_contract'] = {k:training[k] for k in keys if k in training}
            out['training_contract_sha256'] = sha256(training_path)
        except Exception as e:
            out['errors'].append('training_contract.json: '+str(e))
    return out

import argparse

import contextlib

import csv

import inspect

import signal

import tarfile

import traceback

RUN_VERSION = 'ftd_vllm_quality_20261101_v1'

VERIFY_TASKS = ('mrpc', 'qnli', 'agnews')

LABELS = {'mrpc':['not_equivalent','equivalent'],
          'qnli':['entailment','not_entailment'],
          'agnews':['world','sports','business','science']}

VERSION = 'answer_prefix_head_tail64_v3_20261026'

SUFFIX = '\nAnswer:'

def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def read_rows(path):
    result = []
    with Path(path).open(encoding='utf-8') as f:
        for line_number, line in enumerate(f,1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row,dict) or not isinstance(row.get('prompt'),str) or not row['prompt'].strip():
                raise ValueError(str(path)+': invalid prompt at line '+str(line_number))
            result.append(row)
    if not result:
        raise ValueError('Empty development data: '+str(path))
    return result

def preflight_v1(spec, root):
    """No ML imports or GPU allocation; fail before inference if assets do not match."""
    errors = []
    software = software_metadata()
    if software.get('vllm',{}).get('version') != '0.17.1':
        errors.append('This release targets installed vLLM 0.17.1; do not auto-upgrade it')
    for name in ('torch','transformers','peft','safetensors'):
        if not software.get(name,{}).get('installed'):
            errors.append('Package metadata missing: '+name)
    pool = Path(spec['taskpool_dir'])
    contract = read_json(pool/'base_contract.json')
    if set(contract.get('tasks',[])) != set(VERIFY_TASKS):
        errors.append('Task catalog differs from the recorded MRPC/QNLI/AGNews pool')
    if contract.get('base_files') != BASE_HASHES:
        errors.append('Task-pool base identity differs from the frozen seven-file record')
    base, tensors = inspect_base(Path(spec['model_dir']), hash_weights=True)
    if not base.get('all_frozen_base_hashes_match') or base.get('errors'):
        errors.append('Base identity/header validation failed; see preflight.json')
    adapters = []
    for task in VERIFY_TASKS:
        meta = inspect_adapter(pool/'lora'/task,task,tensors)
        adapters.append(meta)
        if not meta.get('ordinary_linear_metadata_checks_pass') or meta.get('errors'):
            errors.append(task+': adapter tensor/config validation failed')
        if not meta.get('rank_matches_recorded_task') or meta.get('pair_count') != 196:
            errors.append(task+': expected rank or 196-pair coverage does not match')
        cfg = meta.get('config',{})
        if cfg.get('lora_alpha') != 2*RANKS[task]:
            errors.append(task+': expected alpha=2*rank')
        training = meta.get('training_contract')
        if not training or training.get('task') != task or training.get('smoke_only'):
            errors.append(task+': missing/invalid training provenance or smoke-only adapter')
        elif training.get('policy_dev_used_for_training') is not False or training.get('evaluation_used_for_training') is not False:
            errors.append(task+': development/evaluation split contract failed')
        dev = pool/'data'/(task+'_policy_dev.jsonl')
        if not dev.is_file():
            errors.append('Missing development data: '+str(dev))
        else:
            data = dataset_metadata(dev)
            meta['policy_dev_metadata'] = data
            if data['invalid_rows'] or not data['rows']:
                errors.append(task+': invalid/empty development JSONL')
        train = pool/'data'/(task+'_train.jsonl')
        if train.is_file() and training:
            meta['training_data_sha256_actual'] = sha256(train)
            if meta['training_data_sha256_actual'] != training.get('training_data_sha256'):
                errors.append(task+': training-data SHA differs from training provenance')
    report = dict(software=software,gpu=gpu_metadata(),base=base,
                  base_contract_sha256=sha256(pool/'base_contract.json'),adapters=adapters,errors=errors)
    dump(root/'preflight.json',report)
    if errors:
        raise ValueError('; '.join(errors))

def torch_runtime(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot use CUDA in this environment')
    return dict(torch_version=torch.__version__,torch_cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),
                built_arches=torch.cuda.get_arch_list())

def engine_settings(spec):
    return dict(model=spec['model_dir'],tokenizer=spec['model_dir'],dtype='float16',
        trust_remote_code=False,tensor_parallel_size=1,seed=0,
        enable_lora=True,max_loras=3,max_cpu_loras=3,max_lora_rank=16,lora_dtype='float16',
        max_model_len=512,max_num_seqs=4,max_num_batched_tokens=2048,
        enable_chunked_prefill=True,enable_prefix_caching=False,
        enforce_eager=True,compilation_config=0,
        attention_config={'backend':'TRITON_ATTN'},
        gpu_memory_utilization=0.10,kv_cache_memory_bytes=256*MIB,
        generation_config='vllm',logprobs_mode='raw_logprobs',disable_log_stats=True)

def stop_owned_process(process):
    # Clean up only the new process group created by this runner, including
    # descendants if the immediate worker has already exited after an error.
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()

def preflight(spec, root):
    preflight_v1(spec,root)
    pf = read_json(root/'preflight.json')
    errors = []
    for a in pf['adapters']:
        expected = EXPECTED_POOL_IDENTITIES[a['task']]
        weights = {w['file']:w['sha256'] for w in a['weight_files']}
        for key,actual in [('config_sha256',a['config_sha256']),
                           ('training_contract_sha256',a.get('training_contract_sha256')),
                           ('policy_dev_sha256',a['policy_dev_metadata']['sha256']),
                           ('weight_files',weights)]:
            if actual != expected[key]:
                errors.append(a['task']+': differs from successful short test: '+key)
        if spec['scope']=='full':
            path = Path(spec['taskpool_dir'])/'data'/(a['task']+'_eval.jsonl')
            if not path.is_file():
                errors.append('Missing evaluation file: '+str(path))
            else:
                a['eval_metadata'] = dataset_metadata(path)
                if a['eval_metadata']['invalid_rows'] or not a['eval_metadata']['rows']:
                    errors.append(a['task']+': invalid/empty evaluation data')
    pf['quality_identity_errors'] = errors
    dump(root/'preflight.json',pf)
    if errors:
        raise ValueError('; '.join(errors))

def shutdown_llm(llm, root):
    result=dict(engine_shutdown=False,driver_process_group_destroyed=False,errors=[])
    if llm is not None:
        try:
            core=getattr(getattr(llm,'llm_engine',None),'engine_core',None)
            if core is None or not callable(getattr(core,'shutdown',None)):
                raise RuntimeError('vLLM engine-core shutdown API unavailable')
            core.shutdown()
            result['engine_shutdown']=True
        except Exception as e:
            result['errors'].append(type(e).__name__+': '+str(e))
    try:
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            result['driver_process_group_destroyed']=True
    except Exception as e:
        result['errors'].append(type(e).__name__+': '+str(e))
    dump(root/'vllm_cleanup.json',result)

class TensorFiles:
    def __init__(self, path):
        path = Path(path)
        self.files = sorted(path.glob('*.safetensors')) if path.is_dir() else [path]
        if not self.files:
            raise FileNotFoundError(f'No safetensors files: {path}')
        self.entries = {}
        for file in self.files:
            with file.open('rb') as f:
                first = f.read(8)
                if len(first) != 8:
                    raise ValueError(f'Empty/truncated tensor file: {file}')
                size = struct.unpack('<Q', first)[0]
                if size > 100_000_000 or size > file.stat().st_size - 8:
                    raise ValueError(f'Invalid safetensors header: {file}')
                header = json.loads(f.read(size))
            for name, spec in header.items():
                if name == '__metadata__':
                    continue
                if name in self.entries:
                    raise ValueError(f'Duplicate tensor: {name} in {path}')
                begin, end = spec['data_offsets']
                if not 0 <= begin <= end <= file.stat().st_size - 8 - size:
                    raise ValueError(f'Invalid tensor offsets: {file}/{name}')
                self.entries[name] = dict(file=file, shape=tuple(spec['shape']),
                                          dtype=spec['dtype'], bytes=end-begin)

    def require(self, shapes, dtype=None):
        for name, shape in shapes.items():
            if name not in self.entries:
                raise ValueError(f'Missing tensor {name} in {self.files}')
            spec = self.entries[name]
            if tuple(shape) != spec['shape']:
                raise ValueError(f'Shape mismatch for {name}: {spec["shape"]} != {shape}')
            if dtype is not None and spec['dtype'] != dtype:
                raise ValueError(f'Dtype mismatch for {name}: {spec["dtype"]} != {dtype}')

    @contextmanager
    def open(self, names):
        from safetensors import safe_open
        files = sorted({self.entries[n]['file'] for n in names})
        with ExitStack() as stack:
            handles = {p: stack.enter_context(safe_open(str(p), framework='pt', device='cpu'))
                       for p in files}
            yield lambda name: handles[self.entries[name]['file']].get_tensor(name)

    def tensor_bytes(self, names):
        return sum(self.entries[n]['bytes'] for n in names)

def _ftd_transfer_chunk_mib():
    # FTD_CHUNKED_TRANSFER_20261025
    mib = float(os.environ.get('FTD_TRANSFER_CHUNK_MB', '1'))
    if not 0 < mib <= 64:
        raise ValueError('FTD_TRANSFER_CHUNK_MB must be >0 and <=64 MiB')
    return mib

def _ftd_chunked_update(torch, destination, source, operation):
    # Do not flatten via reshape/contiguous: that could allocate a full copy.
    if destination.shape != source.shape:
        raise ValueError('Shape mismatch in chunked transfer')
    if not destination.is_contiguous() or not source.is_contiguous():
        raise ValueError('Chunked v2 currently requires contiguous state tensors')
    if operation not in ['copy', 'sub', 'add']:
        raise ValueError('Unsupported chunk operation: ' + operation)
    budget = max(1, int(_ftd_transfer_chunk_mib() * 2**20))
    elements = max(1, budget // max(destination.element_size(), source.element_size()))
    dst = destination.view(-1)
    src = source.view(-1)
    with torch.no_grad():
        for start in range(0, destination.numel(), elements):
            end = min(start + elements, destination.numel())
            target = dst[start:end]
            value = src[start:end]
            if operation == 'copy':
                # Tensor.copy_ supports cross-device copy. Avoid a full src.to(cuda).
                target.copy_(value, non_blocking=False)
            else:
                update = value.to(device=target.device, dtype=target.dtype)
                if operation == 'sub':
                    target.sub_(update)
                else:
                    target.add_(update)
                del update

class Factors:
    """Ordinary linear LoRA only. Unsupported variants fail explicitly."""
    def __init__(self, runtime, parameters):
        self.runtime, self.values, self.bytes, self.metadata = runtime, {}, 0, {}
        for task in runtime.tasks:
            directory = runtime.artifact / 'lora' / task
            config = json.loads((directory / 'adapter_config.json').read_text())
            if str(config.get('peft_type', '')).upper() != 'LORA':
                raise NotImplementedError('Ordinary LoRA required: ' + task)
            reject = ['use_dora', 'fan_in_fan_out', 'lora_bias', 'modules_to_save',
                      'rank_pattern', 'alpha_pattern', 'layer_replication',
                      'trainable_token_indices', 'target_parameters', 'alora_invocation_tokens',
                      'use_qalora', 'arrow_config', 'use_bdlora']
            bad = [key for key in reject if config.get(key)]
            if config.get('bias', 'none') != 'none':
                bad.append('bias')
            if config.get('init_lora_weights', True) not in (True, 'gaussian', False):
                bad.append('nonstandard initialization/base transformation')
            if bad:
                raise NotImplementedError(task + ': unsupported configuration: ' + ', '.join(bad))
            rank = int(config['r'])
            if rank < 1:
                raise ValueError('Invalid LoRA rank')
            scale = float(config['lora_alpha']) / (math.sqrt(rank) if config.get('use_rslora') else rank)
            if not math.isfinite(scale):
                raise ValueError('Invalid LoRA scale')
            index = TensorFiles(directory)
            grouped = collections.defaultdict(dict)
            for key in index.entries:
                part = None
                for letter in ('A', 'B'):
                    suffix = '.lora_' + letter + '.weight'
                    if key.endswith(suffix):
                        name = key[:-len(suffix)]
                        if not name.startswith('base_model.model.'):
                            raise NotImplementedError('Unrecognized adapter prefix: ' + key)
                        part = name[len('base_model.model.'):] + '.weight'
                        grouped[part][letter] = key
                        break
                if part is None:
                    raise NotImplementedError('Unsupported non-A/B adapter tensor: ' + key)
            if set(grouped) != set(runtime.manifest['mutable_parameters']):
                raise NotImplementedError('LoRA factor coverage differs from mutable checkpoint weights: ' + task)
            self.values[task] = {}
            with index.open(list(index.entries)) as get:
                for name, keys in grouped.items():
                    if set(keys) != {'A', 'B'} or name not in parameters:
                        raise ValueError('Incomplete LoRA pair: ' + name)
                    a, b = get(keys['A']), get(keys['B'])
                    expected = (b.shape[0], a.shape[1])
                    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank or tuple(parameters[name].shape) != expected:
                        raise ValueError('LoRA/model shape mismatch: ' + name)
                    a = a.to(device=runtime.device, dtype=runtime.dtype)
                    b = b.to(device=runtime.device, dtype=runtime.dtype)
                    if runtime.device.type == 'cpu':
                        a, b = a.clone(), b.clone()
                    if not runtime.torch.isfinite(a).all() or not runtime.torch.isfinite(b).all():
                        raise FloatingPointError('Non-finite adapter factor')
                    self.values[task][name] = (a, b, scale)
                    self.bytes += (a.numel() + b.numel()) * a.element_size()
            self.metadata[task] = dict(rank=rank, scale=scale, matrices=len(grouped),
                                      serialized_bytes=sum(p.stat().st_size for p in index.files))
        runtime.sync()

    def add(self, parameters, task, sign):
        with self.runtime.torch.no_grad():
            for name, (a, b, scale) in self.values[task].items():
                # GEMM accumulates into the actual weight; no dense B@A tensor
                # is intentionally retained. Backend workspace is measured.
                parameters[name].addmm_(b, a, beta=1.0, alpha=sign * scale)

import gc

import random

import statistics

from types import SimpleNamespace

from contextlib import ExitStack,contextmanager

PROFILE_VERSION='ftd_runtime_profile_20261101_v1'

FORWARD_PATHS=('hf_merged_serial','hf_unmerged_serial','hf_merged_batch','hf_unmerged_batch','vllm_prompt')

def load_profile_base(spec,torch,device='cuda:0',dtype=None):
    from transformers import AutoModelForCausalLM
    dtype=torch.float16 if dtype is None else dtype
    return AutoModelForCausalLM.from_pretrained(spec['model_dir'],local_files_only=True,
        trust_remote_code=False,torch_dtype=dtype,device_map={'':device},
        low_cpu_mem_usage=True,attn_implementation='sdpa').to(dtype=dtype).eval()

def mutable_names_from_factors(directory):
    names=set()
    for key in TensorFiles(directory).entries:
        for letter in ('A','B'):
            suffix='.lora_'+letter+'.weight'
            if key.startswith('base_model.model.') and key.endswith(suffix):
                names.add(key[len('base_model.model.'):-len(suffix)]+'.weight')
                break
        else: raise ValueError('Unsupported adapter key: '+key)
    return sorted(names)

def tensor_digest(torch,items):
    h=hashlib.sha256()
    for name,tensor in sorted(items):
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode());h.update(str(tensor.dtype).encode())
        # Each matrix is processed separately; no full-model concatenation.
        cpu=tensor.detach().to('cpu',copy=True).contiguous().view(torch.uint8)
        h.update(memoryview(cpu.numpy()))
    return h.hexdigest()

def install_host_target(torch,parameters,target):
    for name,value in target.items(): _ftd_chunked_update(torch,parameters[name],value,'copy')

def model_payload_bytes(model):
    storages={}
    for tensor in list(model.parameters())+list(model.buffers()):
        if tensor.device.type=='meta': raise ValueError('Meta tensor in active model')
        storage=tensor.untyped_storage()
        storages[(str(tensor.device),storage.data_ptr(),storage.nbytes())]=storage.nbytes()
    return sum(storages.values())

def describe(values):
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                minimum=min(values),maximum=max(values))

NATIVE_VERSION = 'ftd_vllm_labelonly_20261101_v1'

NATIVE_PATHS = ('hf_merged_batch', 'hf_unmerged_batch', 'vllm_prompt', 'vllm_labelonly')

CONTROL_LOGPROB_TOLERANCE = 0.10

FROZEN_SOURCE_FILES = {
    'quality': {
        'cases.json': '5a376010dbcfafe98b0a04aa8c089401133a3aee46169f0be7e8031de9684f3e',
        'hf_scores.jsonl': 'ed77707f3562b77863ddad691ed7f0fa082efd818a3c81c8afbf3fd185421475',
        'vllm_scores.jsonl': '17877b54533f80ab221f9d508f595357fa035f5c561c4aeab3b7f32b4d996d78',
        'spec.json': '62d4dac27de0d2efd3af727ac627efe20cbeda29c4b1e056820f010e9822fb1c',
        'comparison.json': 'd8b1d1fe43cef485b73873c42a27a0a64bd8986f88225a68411260b0550f83e8'},
    'profile': {
        'profile_cases.json': '7320d1969e1d167627ab5a77c5e6781d1fe44c6be0c0b328ea1fefed9e0951dc',
        'profile_summary.json': '6fd3dace22c95fb855870a9913ae52a7bf3202e814d27706186527f6e072be49',
        'spec.json': '86cd6eb0db463edbcf3d7dde660b8d0765008c337cd56e71d758c16bb2a70f0a'}}

LOCAL_TARGET_HASHES = {
    'mrpc': 'fda248ef92334ea9912ee70c767ef8e95b6310fde042da73edd34c9abffa698e',
    'qnli': '96e00644dc2b3a97f512cad5bc7a8ecc8d3a2fef6cc028dd69ed8854b2dca24f',
    'agnews': 'aa7d0a239444e8f519ee203c8e68906f6a7a4aaca7e72a88b44be98e43888b46'}

def jsonl_rows(path):
    return [json.loads(line, object_pairs_hook=unique_object)
            for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def nvml_owned_snapshot():
    # No arbitrary Python RPC, pickle fallback, environment permission changes,
    # or subprocess sampling inside the timed scoring region.
    try:
        import pynvml
    except ImportError:
        return dict(status='unavailable', reason='pynvml is not installed; no package was changed')
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        rows = []
        group = os.getpgrp()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                try:
                    owned = os.getpgid(proc.pid) == group
                except ProcessLookupError:
                    continue
                if owned:
                    value = int(proc.usedGpuMemory)
                    rows.append(dict(gpu_index=index, pid=int(proc.pid),
                        used_gpu_memory_bytes=value if 0 <= value < 2**60 else None))
        valid = [r['used_gpu_memory_bytes'] for r in rows]
        return dict(status='ok' if rows else 'no_owned_process_visible', processes=rows,
            sum_process_reported_bytes=sum(valid) if rows and all(v is not None for v in valid) else None,
            measurement='Point-in-time process-reported resident bytes; not a physical peak; IPC may double count')
    except Exception as e:
        return dict(status='unavailable', error_type=type(e).__name__, reason=str(e))
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

import itertools

import resource

import threading

E2E_VERSION='ftd_complete_request_20261101_v1'

WORKLOADS=('churn','sticky8')

RESET_PERIODS=(8,32,128,512)

METHODS=('exact_copy','hf_unmerged','periodic128','guard8','guard32','guard128','guard512',
         'calibrated_fixed','policy','vllm_prompt')

SOURCE_CASES_SHA='5a376010dbcfafe98b0a04aa8c089401133a3aee46169f0be7e8031de9684f3e'

STATE_BUDGET_BYTES=1280*MIB

HOST_REFERENCE_BUDGET_BYTES=3072*MIB

EXPECTED_MODEL_BYTES=1192099968

EXPECTED_FACTOR_BYTES=35323904

FIT_PER_TASK=16

AUDIT_PER_TASK=16

EVAL_PER_TASK=192

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def save_signed(path,value):
    obj=dict(value)
    obj['signature']=canonical_hash(value)
    dump(path,obj)
    return obj

def read_signed(path):
    obj=read_json(path)
    signature=obj.pop('signature')
    if canonical_hash(obj)!=signature:
        raise ValueError('Frozen experiment metadata changed: '+str(path))
    obj['signature']=signature
    return obj

def tail_argument(model):
    underlying=model.get_base_model() if callable(getattr(model,'get_base_model',None)) else model
    return next((n for n in ('logits_to_keep','num_logits_to_keep')
                 if n in inspect.signature(underlying.forward).parameters),None)

def target_cache_load(torch,root):
    metadata=read_signed(root/'target_metadata.json')
    targets={}
    for task in VERIFY_TASKS:
        path=root/'reference_slices'/(task+'.safetensors')
        reader=TensorFiles(path)
        with reader.open(list(reader.entries)) as get:
            targets[task]={name:get(name).clone().contiguous() for name in reader.entries}
        actual=tensor_digest(torch,targets[task].items())
        if actual!=metadata['target_slice_sha256'][task] or actual!=LOCAL_TARGET_HASHES[task]:
            raise ValueError('Exact local reference tensor identity changed: '+task)
    amount=sum(t.numel()*t.element_size() for values in targets.values() for t in values.values())
    if amount>HOST_REFERENCE_BUDGET_BYTES:
        raise ValueError('CPU reference tensor budget exceeded')
    return targets,metadata

def merged_model_and_factors(spec,root,torch,need_factors):
    model=load_profile_base(spec,torch)
    names=mutable_names_from_factors(Path(spec['taskpool_dir'])/'lora'/'mrpc')
    metadata=read_signed(root/'target_metadata.json')
    shared=tensor_digest(torch,[(n,t) for n,t in model.named_parameters() if n not in names]+
                         [('buffer:'+n,t) for n,t in model.named_buffers()])
    if shared!=metadata['shared_state_sha256']:
        raise ValueError('Common model state differs from exact reference')
    factors=None
    if need_factors:
        runtime=SimpleNamespace(torch=torch,tasks=list(VERIFY_TASKS),artifact=Path(spec['taskpool_dir']),
            device=torch.device('cuda:0'),dtype=torch.float16,manifest={'mutable_parameters':names},
            sync=lambda:torch.cuda.synchronize())
        factors=Factors(runtime,dict(model.named_parameters()))
    dtypes={str(t.dtype) for t in model.parameters() if t.is_floating_point()}
    if dtypes!={'torch.float16'}:
        raise ValueError('Online model is not fully FP16')
    state=model_payload_bytes(model)+(0 if factors is None else factors.bytes)
    if state>STATE_BUDGET_BYTES:
        raise ValueError('Actual GPU retained tensor state exceeds 1280 MiB')
    return model,factors

def monitor_stage(stop,path,group):
    try:
        import pynvml as nv
        nv.nvmlInit()
    except Exception as e:
        dump(path.with_suffix('.unavailable.json'),dict(status='unavailable',error_type=type(e).__name__))
        return
    try:
        handle=nv.nvmlDeviceGetHandleByIndex(0)
        with path.open('w') as f:
            while not stop.is_set():
                row=dict(monotonic_ns=time.perf_counter_ns(),physical_gpu_index=0,
                         host_loadavg=list(os.getloadavg()),logical_cpu_count=os.cpu_count())
                try:
                    processes=nv.nvmlDeviceGetComputeRunningProcesses(handle)
                    owned=[]
                    others=0
                    for process in processes:
                        try:
                            belongs=os.getpgid(process.pid)==group
                        except ProcessLookupError:
                            continue
                        if belongs:
                            owned.append(int(process.usedGpuMemory))
                        else:
                            others+=1
                    row.update(other_compute_process_count=others,
                        owned_process_reported_bytes=sum(owned) if all(0<=v<2**60 for v in owned) else None,
                        status='ok')
                    queries=dict(gpu_util_pct=lambda:nv.nvmlDeviceGetUtilizationRates(handle).gpu,
                        memory_util_pct=lambda:nv.nvmlDeviceGetUtilizationRates(handle).memory,
                        sm_clock_mhz=lambda:nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_SM),
                        memory_clock_mhz=lambda:nv.nvmlDeviceGetClockInfo(handle,nv.NVML_CLOCK_MEM),
                        temperature_c=lambda:nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU),
                        power_w=lambda:nv.nvmlDeviceGetPowerUsage(handle)/1000,
                        device_used_bytes=lambda:nv.nvmlDeviceGetMemoryInfo(handle).used)
                    for name,query in queries.items():
                        try:
                            row[name]=query()
                        except Exception:
                            row[name]=None
                except Exception as e:
                    row.update(status='partial',error_type=type(e).__name__)
                f.write(json.dumps(row,allow_nan=False)+'\n')
                f.flush()
                stop.wait(1.0)
    except Exception as e:
        dump(path.with_suffix('.unavailable.json'),dict(status='unavailable',error_type=type(e).__name__))
    finally:
        try:
            nv.nvmlShutdown()
        except Exception:
            pass

import copy

PROBE_VERSION='ftd_parameter_kv_feasibility_20261101_v1'
PROBE_TASKS=('mrpc','qnli')
CONTEXT_LENGTHS=(512,2048)
DECODE_TOKENS=16
PROBE_METHODS=('inc_full','exact_cached','exact_full','merged_hot_cached','hf_native_cached','vllm_native_cached')
LIVE_TENSOR_CAP=2048*MIB
SOURCE_RUN='ftd_complete_request_20261101T112924Z_6xy3wqgy'
SOURCE_HASHES={'cases.json':'26e82328503725c5d3d9e2d97b56c66f4c5dbbcf85dafef00220fcff817950ca',
               'target_metadata.json':'f7b7aa5ac9a4fb8380cab3c775b755134babee3be8b06d5950c297da65e0292d'}


class ProbeBlocked(RuntimeError):
    pass


class CacheIdentity:
    """Conservative provenance. Version identity is not a float-error certificate."""
    @staticmethod
    def exact(target_digest,backend='hf_merged_fp16_sdpa'):
        return backend+':exact:'+target_digest

    @staticmethod
    def increment(previous,source,target):
        return 'incremental:'+canonical_hash([previous,source,target])

    @staticmethod
    def key(version,prefix,contract):
        return canonical_hash(dict(weight_state_version=version,prefix_ids=prefix,execution_contract=contract))


class PrefixBank:
    """Store immutable prefix templates; every generation forks its working KV."""
    def __init__(self):
        self.entries={}

    def put(self,version,prefix,contract,cache):
        self.entries[CacheIdentity.key(version,prefix,contract)]=cache

    def get(self,version,prefix,contract):
        return self.entries.get(CacheIdentity.key(version,prefix,contract))

    def require(self,version,prefix,contract):
        cache=self.get(version,prefix,contract)
        if cache is None:
            raise ValueError('KV provenance mismatch: recompute prefix or restore its exact executable state')
        return cache


def choose_parameter_context_path(current_version,canonical_target_version,cache_present,
                                  estimated_ms,live_tensor_bytes,budget,allow_approximate=False):
    """Planning primitive only; no claim of an executed online policy benchmark."""
    feasible=[]
    for action in ('inc_full','exact_cached','exact_full','merged_hot_cached'):
        if action not in estimated_ms or action not in live_tensor_bytes:
            continue
        if not math.isfinite(estimated_ms[action]) or estimated_ms[action]<=0:
            raise ValueError('Invalid calibrated path cost')
        if live_tensor_bytes[action]>budget:
            continue
        if action=='inc_full' and not allow_approximate:
            continue
        if action=='exact_cached' and not cache_present:
            continue
        if action=='merged_hot_cached' and (not cache_present or current_version!=canonical_target_version):
            continue
        feasible.append(action)
    if not feasible:
        raise ValueError('No feasible parameter/context restoration route')
    return min(feasible,key=lambda a:(estimated_ms[a],a))


def cache_tensors(cache):
    if cache is None:
        return []
    if hasattr(cache,'layers'):
        result=[]
        for layer in cache.layers:
            for name in ('keys','values'):
                value=getattr(layer,name,None)
                if value is not None:
                    result.append(value)
        return result
    if isinstance(cache,(tuple,list)):
        return [t for pair in cache for t in pair]
    raise TypeError('Unsupported cache type: '+type(cache).__name__)


def tensor_storage_bytes(tensors):
    storages={}
    for tensor in tensors:
        storage=tensor.untyped_storage()
        storages[(str(tensor.device),storage.data_ptr(),storage.nbytes())]=storage.nbytes()
    return sum(storages.values())


def cache_bytes(cache):
    return tensor_storage_bytes(cache_tensors(cache))


def sync(torch,device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def prefill_cache(torch,model,prefix,device):
    tail=tail_argument(model)
    if tail is None:
        raise ValueError('Tail-logit API is required; do not allocate full-prompt vocabulary logits')
    with torch.inference_mode():
        ids=torch.tensor([prefix],device=device,dtype=torch.long)
        out=model(input_ids=ids,use_cache=True,**{tail:1})
    cache=out.past_key_values
    if cache is None or cache.get_seq_length()!=len(prefix):
        raise ValueError('Prefix cache length mismatch')
    return cache


def greedy_tokens(torch,model,prefix,device,decode_tokens=DECODE_TOKENS,template=None,bank_bytes=0,model_bytes=0,cap=None,expected_working_bytes=None):
    """Fixed token count ignores EOS. Synthetic mechanism timing, not task accuracy."""
    if len(prefix)<2 or decode_tokens<1:
        raise ValueError('Invalid synthetic request length')
    tail=tail_argument(model)
    if tail is None:
        raise ValueError('Tail-logit API unavailable')
    if cap is not None and expected_working_bytes is not None and model_bytes+bank_bytes+expected_working_bytes>cap:
        raise ValueError('Working KV would exceed fixed live tensor cap before allocation')
    sync(torch,device)
    start=time.perf_counter_ns()
    with torch.inference_mode():
        working=copy.deepcopy(template) if template is not None else None
        if template is not None and template.get_seq_length()!=len(prefix)-1:
            raise ValueError('Only the first L-1 tokens may be reused')
        ids=torch.tensor([prefix[-1:] if working is not None else prefix],device=device,dtype=torch.long)
        out=model(input_ids=ids,past_key_values=working,use_cache=True,**{tail:1})
        working=out.past_key_values
        sync(torch,device)
        prefill_end=time.perf_counter_ns()
        first_logits=out.logits[0,-1]
        generated=[]
        for index in range(decode_tokens):
            token=int(out.logits[0,-1].argmax().item())
            generated.append(token)
            if index+1<decode_tokens:
                out=model(input_ids=torch.tensor([[token]],device=device,dtype=torch.long),
                    past_key_values=working,use_cache=True,**{tail:1})
                working=out.past_key_values
        sync(torch,device)
        finished=time.perf_counter_ns()
        active_bytes=cache_bytes(working)
        logical_bytes=model_bytes+bank_bytes+active_bytes
        if cap is not None and logical_bytes>cap:
            raise ValueError('Model + factors + prefix templates + working KV exceeds fixed live tensor cap')
        if template is not None and template.get_seq_length()!=len(prefix)-1:
            raise ValueError('Immutable prefix template was mutated')
    result=dict(generated_ids=generated,generated_tokens=len(generated),
        finished_monotonic_ns=finished,
        generation_ms=(finished-start)/1e6,cache_fork_and_prefill_ms=(prefill_end-start)/1e6,
        decode_ms=(finished-prefill_end)/1e6,working_cache_bytes=active_bytes,logical_live_tensor_bytes=logical_bytes)
    # Only a small reference vector is retained outside timed generation.
    return result,first_logits.detach().to('cpu').float()


def make_probe_cases(source):
    data=read_signed(Path(source)/'cases.json')
    unit=data['tasks']['mrpc']['fit'][0]['prefix']
    cases=[]
    for length in CONTEXT_LENGTHS:
        prefix=(unit*((length+len(unit)-1)//len(unit)))[:length]
        for task in PROBE_TASKS:
            cases.append(dict(case_id=task+':synthetic_prefix:'+str(length),task=task,
                context_tokens=length,prefix=prefix,kind='synthetic_mechanism_probe',gold=None))
    return cases


def foreign_gpu_process_count(group=None):
    try:
        import pynvml as nv
    except ImportError as e:
        raise ProbeBlocked('NVML is unavailable; no package was installed') from e
    initialized=False
    try:
        nv.nvmlInit()
        initialized=True
        handle=nv.nvmlDeviceGetHandleByIndex(0)
        foreign=0
        for proc in nv.nvmlDeviceGetComputeRunningProcesses(handle):
            try:
                same=group is not None and os.getpgid(proc.pid)==group
            except ProcessLookupError:
                continue
            if not same:
                foreign+=1
        return foreign
    except ProbeBlocked:
        raise
    except Exception as e:
        raise ProbeBlocked('Cannot verify GPU isolation: '+type(e).__name__) from e
    finally:
        if initialized:
            try:nv.nvmlShutdown()
            except Exception:pass


def require_idle(group=None,samples=1):
    for i in range(samples):
        count=foreign_gpu_process_count(group)
        if count:
            raise ProbeBlocked('Observed '+str(count)+' GPU compute process(es) outside this worker group. Coordinate an idle slot, then rerun. No process was stopped.')
        if i+1<samples:
            time.sleep(1.)


def hf_probe_worker(spec,root):
    import torch
    from peft import PeftModel
    torch_runtime(torch)
    torch.manual_seed(0)
    method=spec['method']
    tag=spec['tag']
    source=Path(spec['source_result'])
    cases=read_signed(root/'probe_cases.json')['cases']
    targets={}
    factors=None
    if method=='hf_native_cached':
        model=load_profile_base(spec,torch)
        model=PeftModel.from_pretrained(model,Path(spec['taskpool_dir'])/'lora'/'mrpc',adapter_name='mrpc',
            is_trainable=False,autocast_adapter_dtype=False,local_files_only=True)
        for task in ('qnli','agnews'):
            model.load_adapter(Path(spec['taskpool_dir'])/'lora'/task,adapter_name=task,
                is_trainable=False,autocast_adapter_dtype=False,local_files_only=True)
        model.to(dtype=torch.float16).eval()
    else:
        targets,_=target_cache_load(torch,source)
        model,factors=merged_model_and_factors(spec,source,torch,True)
    params=dict(model.named_parameters())
    if {str(t.dtype) for t in model.parameters() if t.is_floating_point()}!={'torch.float16'}:
        raise ValueError('Probe model parameters must all be FP16')
    model_bytes=model_payload_bytes(model)+(0 if factors is None else factors.bytes)
    dump(root/(tag+'_runtime.json'),dict(torch_runtime(torch),parameter_dtypes=sorted({str(t.dtype) for t in model.parameters() if t.is_floating_point()}),
        model_and_factor_bytes=model_bytes,logical_tensor_cap_bytes=LIVE_TENSOR_CAP,
        strict_physical_memory_claim=False,optimistic_no_switch_bound=method=='merged_hot_cached'))
    rows=[]
    validity=[]
    for length in CONTEXT_LENGTHS:
        selected=[c for c in cases if c['context_tokens']==length]
        bank=PrefixBank()
        contract=dict(backend='hf_sdpa',dtype='float16',base=BASE_HASHES['model.safetensors'],
            config=BASE_HASHES['config.json'],position_start=0,mode=method=='hf_native_cached')
        versions={}
        for c in selected:
            task=c['task']
            if method=='hf_native_cached':
                model.set_adapter(task)
                model.eval()
                version=CacheIdentity.exact(EXPECTED_POOL_IDENTITIES[task]['weight_files']['adapter_model.safetensors'],backend='hf_unmerged_fp16_sdpa')
            else:
                install_host_target(torch,params,targets[task])
                version=CacheIdentity.exact(LOCAL_TARGET_HASHES[task])
            cache=prefill_cache(torch,model,c['prefix'][:-1],'cuda:0')
            bank.put(version,c['prefix'][:-1],contract,cache)
            versions[task]=version
        bank_size=sum(cache_bytes(v) for v in bank.entries.values())
        if model_bytes+bank_size>LIVE_TENSOR_CAP:
            raise ValueError('Prefix templates plus model/factors exceed 2048 MiB tensor cap')
        for c in selected:
            require_idle(os.getpgrp())
            task=c['task']
            other=next(t for t in PROBE_TASKS if t!=task)
            exact_version=versions[task]
            template=bank.require(exact_version,c['prefix'][:-1],contract)
            dirty=CacheIdentity.increment(versions[other],other,task)
            if method!='hf_native_cached':
                assert bank.get(dirty,c['prefix'][:-1],contract) is None
                assert bank.get(versions[other],c['prefix'][:-1],contract) is not None
            def install_source():
                if method=='hf_native_cached':
                    model.set_adapter(other)
                    model.eval()
                elif method=='merged_hot_cached':
                    install_host_target(torch,params,targets[task])
                else:
                    install_host_target(torch,params,targets[other])
                torch.cuda.synchronize()
            def one():
                started=time.perf_counter_ns()
                if method in ('exact_cached','exact_full'):
                    install_host_target(torch,params,targets[task])
                elif method=='inc_full':
                    factors.add(params,other,-1)
                    factors.add(params,task,+1)
                elif method=='hf_native_cached':
                    # All factors are resident, but actual task selection remains timed.
                    model.set_adapter(task)
                    model.eval()
                torch.cuda.synchronize()
                use_template=method in ('exact_cached','merged_hot_cached','hf_native_cached')
                result,logits=greedy_tokens(torch,model,c['prefix'],'cuda:0',
                    template=template if use_template else None,bank_bytes=bank_size,model_bytes=model_bytes,cap=LIVE_TENSOR_CAP,
                    expected_working_bytes=(cache_bytes(template)*(length+DECODE_TOKENS-1)+length-2)//(length-1))
                # Include state installation; exclude CPU copy of diagnostic logits.
                result['request_ms']=(result['finished_monotonic_ns']-started)/1e6
                return result,logits
            # Both exact forms are numerically checked, independent of the chosen timing path.
            if method!='hf_native_cached':
                install_host_target(torch,params,targets[task])
                expected=(cache_bytes(template)*(length+DECODE_TOKENS-1)+length-2)//(length-1)
                full,full_logits=greedy_tokens(torch,model,c['prefix'],'cuda:0',bank_bytes=bank_size,model_bytes=model_bytes,cap=LIVE_TENSOR_CAP,expected_working_bytes=expected)
                cached,cached_logits=greedy_tokens(torch,model,c['prefix'],'cuda:0',template=template,bank_bytes=bank_size,model_bytes=model_bytes,cap=LIVE_TENSOR_CAP,expected_working_bytes=expected)
                validity.append(dict(case_id=c['case_id'],exact_cache_state_key_accepted=True,
                    dirty_state_key_rejected=True,cached_vs_full_max_abs_first_logit_difference=float((full_logits-cached_logits).abs().max()),
                    cached_vs_full_same_generated_ids=full['generated_ids']==cached['generated_ids']))
            calibration=[]
            for _ in range(2):
                install_source()
                result,_=one()
                calibration.append(result['request_ms'])
            require_idle(os.getpgrp())
            before=nvml_owned_snapshot()
            torch.cuda.reset_peak_memory_stats()
            for rep in range(3):
                install_source()
                result,first=one()
                row=dict(method=method,case_id=c['case_id'],task=task,context_tokens=length,repeat=rep,
                    cache_hit_workload=True,synthetic=True,gold=None,decode_tokens=DECODE_TOKENS,
                    calibrated_mean_ms=statistics.mean(calibration),optimistic_no_switch_bound=method=='merged_hot_cached',
                    same_generated_ids_as_exact_full=(result['generated_ids']==full['generated_ids'] if method!='hf_native_cached' else None),
                    max_abs_first_logit_difference_vs_exact_full=(float((first-full_logits).abs().max()) if method!='hf_native_cached' else None),**result)
                rows.append(row)
            require_idle(os.getpgrp())
            dump(root/(tag+'_'+task+'_'+str(length)+'_memory.json'),dict(before=before,after=nvml_owned_snapshot(),
                allocator_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                model_and_factor_bytes=model_bytes,immutable_template_bytes=bank_size,strict_physical_memory_claim=False))
        del cache,template,bank
        gc.collect()
        torch.cuda.empty_cache()
    with (root/(tag+'_requests.jsonl')).open('w') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    dump(root/(tag+'_cache_validity.json'),dict(checks=validity,
        scope='State-key checks and exact cached/full numerical comparison; no true-label accuracy measurement'))


def native_probe_worker(spec,root):
    import torch
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    llm=None
    tag=spec['tag']
    try:
        settings=engine_settings(spec)
        settings.update(max_model_len=2304,max_num_seqs=1,max_num_batched_tokens=4096,
                        enable_prefix_caching=True,kv_cache_memory_bytes=512*MIB)
        dump(root/(tag+'_engine_settings.json'),settings)
        llm=LLM(**settings)
        config=getattr(llm.llm_engine,'model_config',None)
        if config is None:config=getattr(getattr(llm.llm_engine,'vllm_config',None),'model_config',None)
        if str(getattr(config,'dtype',None))!='torch.float16':
            raise ValueError('Native probe dtype is not confirmed FP16')
        adapters={t:LoRARequest(t,i+1,str(Path(spec['taskpool_dir'])/'lora'/t)) for i,t in enumerate(VERIFY_TASKS)}
        sampling=SamplingParams(temperature=0,max_tokens=DECODE_TOKENS,ignore_eos=True,detokenize=False)
        cases=read_signed(root/'probe_cases.json')['cases']
        rows=[]
        for length in CONTEXT_LENGTHS:
            selected=[c for c in cases if c['context_tokens']==length]
            # Deliberate warm prefix hits; not cold unique-prompt inference.
            for c in selected:
                llm.generate([{'prompt_token_ids':c['prefix']}],sampling_params=sampling,lora_request=adapters[c['task']],use_tqdm=False)
            for c in selected:
                require_idle(os.getpgrp())
                def one():
                    started=time.perf_counter_ns()
                    outputs=llm.generate([{'prompt_token_ids':c['prefix']}],sampling_params=sampling,
                        lora_request=adapters[c['task']],use_tqdm=False)
                    if len(outputs)!=1 or list(outputs[0].prompt_token_ids)!=c['prefix'] or len(outputs[0].outputs)!=1:
                        raise ValueError('Native generation response identity mismatch')
                    tokens=list(outputs[0].outputs[0].token_ids)
                    if len(tokens)!=DECODE_TOKENS:
                        raise ValueError('Native fixed decode count changed')
                    finished=time.perf_counter_ns()
                    return dict(request_ms=(finished-started)/1e6,generated_ids=tokens,generated_tokens=len(tokens),
                        native_cached_tokens=getattr(outputs[0],'num_cached_tokens',None))
                calibration=[one()['request_ms'] for _ in range(2)]
                require_idle(os.getpgrp())
                for rep in range(3):
                    rows.append(dict(method='vllm_native_cached',case_id=c['case_id'],task=c['task'],context_tokens=length,
                        repeat=rep,cache_hit_workload=True,synthetic=True,gold=None,decode_tokens=DECODE_TOKENS,
                        calibrated_mean_ms=statistics.mean(calibration),optimistic_no_switch_bound=False,**one()))
                require_idle(os.getpgrp())
        with (root/(tag+'_requests.jsonl')).open('w') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        dump(root/(tag+'_runtime.json'),dict(torch_runtime(torch),same_logical_tensor_budget_verified=False,
            native_baseline_scope='512 MiB native KV pool and resident LoRA; external native baseline, not same-budget admission proof'))
        dump(root/(tag+'_memory.json'),nvml_owned_snapshot())
    finally:
        shutdown_llm(llm,root)
        path=root/'vllm_cleanup.json'
        if path.is_file():path.replace(root/(tag+'_cleanup.json'))


def run_probe_stage(root,spec,active,timeout):
    require_idle(samples=3)
    dump(root/'active_stage.json',active)
    tag=active['tag']
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',FTD_TRANSFER_CHUNK_MB='1')
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--run-dir',str(root)]
    print('Starting',tag,'; log:',root/'logs'/(tag+'.log'),flush=True)
    started=time.monotonic()
    with (root/'logs'/(tag+'.log')).open('w') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        event=threading.Event()
        observer=threading.Thread(target=monitor_stage,args=(event,root/(tag+'_telemetry.jsonl'),child.pid),daemon=True)
        observer.start()
        try:
            while True:
                left=timeout-(time.monotonic()-started)
                if left<=0:raise TimeoutError(tag+' exceeded timeout')
                try:
                    code=child.wait(timeout=min(15,left))
                    break
                except subprocess.TimeoutExpired:
                    print(tag,'running; see its log',flush=True)
            if code:
                tail=(root/'logs'/(tag+'.log')).read_text(errors='replace').splitlines()[-16:]
                print('\n'.join(tail),flush=True)
                if (root/'logs'/(tag+'.log')).read_text(errors='replace').find('ProbeBlocked:')>=0:
                    raise ProbeBlocked(tag+' observed a GPU isolation problem; partial results retained')
                raise RuntimeError(tag+' failed; full log included')
        finally:
            event.set()
            observer.join(timeout=5)
            stop_owned_process(child)


def summarize_probe(root):
    cases=read_signed(root/'probe_cases.json')['cases']
    grouped={}
    for i in range(2):
        for method in PROBE_METHODS:
            rr=jsonl_rows(root/('r'+str(i)+'_'+method+'_requests.jsonl'))
            if len(rr)!=len(cases)*3:raise ValueError('Incomplete probe timing rows')
            for c in cases:
                values=[r for r in rr if r['case_id']==c['case_id']]
                if sorted(r['repeat'] for r in values)!=[0,1,2]:raise ValueError('Probe repetitions changed')
                if any(r['generated_tokens']!=DECODE_TOKENS or len(r['generated_ids'])!=DECODE_TOKENS or r['gold'] is not None or not math.isfinite(r['request_ms']) or r['request_ms']<=0 or not math.isfinite(r['calibrated_mean_ms']) or r['calibrated_mean_ms']<=0 for r in values):
                    raise ValueError('Invalid synthetic timing or output record')
                grouped[i,method,c['case_id']]=dict(mean_ms=statistics.mean(r['request_ms'] for r in values),
                    median_ms=statistics.median(r['request_ms'] for r in values),
                    calibration_ms=values[0]['calibrated_mean_ms'],
                    logical_live_tensor_bytes=max((r.get('logical_live_tensor_bytes',0) for r in values)),
                    observed_same_ids_as_exact_full=(all(r.get('same_generated_ids_as_exact_full') is True for r in values) if method in PROBE_METHODS[:4] else None),
                    confirmed_native_prefix_hit=(all(isinstance(r.get('native_cached_tokens'),int) and r['native_cached_tokens']>0 for r in values) if method=='vllm_native_cached' else None))
    validity=[]
    for i in range(2):
        for method in PROBE_METHODS[:4]:
            validity+=read_json(root/('r'+str(i)+'_'+method+'_cache_validity.json'))['checks']
    telemetry_flags=[]
    for i in range(2):
        for method in PROBE_METHODS:
            path=root/('r'+str(i)+'_'+method+'_telemetry.jsonl')
            if not path.is_file() or not any(r.get('status')=='ok' for r in jsonl_rows(path)):
                telemetry_flags.append(path.name+': no verified telemetry samples')
    for path in root.glob('r*_telemetry.jsonl'):
        samples=jsonl_rows(path)
        if any(r.get('other_compute_process_count',0)>0 for r in samples):
            telemetry_flags.append(path.name+': foreign compute process observed')
    for path in root.glob('r*_telemetry.unavailable.json'):
        telemetry_flags.append(path.name+': telemetry unavailable')
    if any(not v['confirmed_native_prefix_hit'] for (i,m,c),v in grouped.items() if m=='vllm_native_cached'):
        telemetry_flags.append('Native prefix hits not confirmed by num_cached_tokens for every timed case')
    decisions=[]
    followup=[]
    for c in cases:
        per=[]
        for i in range(2):
            scored=lambda m:grouped[i,m,c['case_id']]
            # Freeze action from warmup calibration; timed repetitions are confirmation only.
            costs={m:scored(m)['calibration_ms'] for m in PROBE_METHODS[:4]}
            payloads={m:scored(m)['logical_live_tensor_bytes'] for m in PROBE_METHODS[:4]}
            chosen=choose_parameter_context_path('different_source_state','canonical_target',True,
                costs,payloads,LIVE_TENSOR_CAP,allow_approximate=True)
            canonical_choice=choose_parameter_context_path('different_source_state','canonical_target',True,
                costs,payloads,LIVE_TENSOR_CAP,allow_approximate=False)
            native=min(('hf_native_cached','vllm_native_cached'),key=lambda m:scored(m)['calibration_ms'])
            gain=1-scored(chosen)['median_ms']/scored(native)['median_ms']
            per.append(dict(round=i,calibration_selected_ftd_path=chosen,calibration_selected_native=native,
                canonical_weight_state_only_choice=canonical_choice,
                selected_path_same_ids_as_exact_full=scored(chosen)['observed_same_ids_as_exact_full'],
                selected_path_median_ms=scored(chosen)['median_ms'],native_median_ms=scored(native)['median_ms'],
                optimistic_zero_switch_ftd_median_ms=scored('merged_hot_cached')['median_ms'],
                observed_reduction_fraction=gain))
        candidate=all(r['observed_reduction_fraction']>=.10 and r['selected_path_same_ids_as_exact_full'] for r in per)
        if candidate:followup.append(c['case_id'])
        decisions.append(dict(case_id=c['case_id'],context_tokens=c['context_tokens'],rounds=per,
            clears_prespecified_10pct_engineering_headroom=candidate))
    unstable=[c['case_id'] for c in cases for method in PROBE_METHODS
        if max(grouped[i,method,c['case_id']]['median_ms'] for i in range(2))/min(grouped[i,method,c['case_id']]['median_ms'] for i in range(2))>1.20]
    if unstable:telemetry_flags.append('Some round median ratios exceed 1.20; no records removed')
    numerical_changes=any(not r['cached_vs_full_same_generated_ids'] for r in validity)
    if numerical_changes:telemetry_flags.append('Exact cached vs full prefill token paths differ on some synthetic inputs; inspect numerical sensitivity before application claims')
    status='CANDIDATE_HEADROOM_ONLY' if followup and not telemetry_flags else 'STOP_OR_INCONCLUSIVE'
    report=dict(status=status,scope='Synthetic warm-prefix feasibility screen; not application accuracy or measured online-policy gain',
        timings=[dict(round=i,method=m,case_id=c,**v) for (i,m,c),v in grouped.items()],
        decisions=decisions,cache_validity=validity,timing_or_numerical_flags=telemetry_flags,
        candidate_cases=followup,controller_benefit_demonstrated=False,novelty_established=False,
        true_label_accuracy_measured=False,strict_physical_memory_claim=False,
        no_go='If no stable switching path beats a strong native cached baseline by 10% in both process rounds, stop expansion of this candidate on the measured configuration. The 10% screen is engineering triage, not a statistical guarantee.',
        limitation='All prefixes are constructed and deliberately repeated. Cache availability, steady shared-context traffic and real task quality remain unvalidated. The hot merged path excludes switching and is only an optimistic bound.')
    dump(root/'kv_feasibility_summary.json',report)
    return report


def pack_probe(root):
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json']
    dump(root/'manifest.json',{str(p.relative_to(root)):sha256(p) for p in files})
    archive=root.with_name(root.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for p in files+[root/'manifest.json']:
            tf.add(p,arcname=root.name+'/'+str(p.relative_to(root)),recursive=False)
    return archive


def main(argv=None):
    parser=argparse.ArgumentParser(description='Small parameter-state/KV feasibility screen. No retraining, no new datasets, no claimed policy gain.')
    parser.add_argument('--model-dir',type=Path,default=Path('./assets/legacy/models/Qwen3-0.6B'))
    parser.add_argument('--taskpool-dir',type=Path,default=Path('./assets/legacy/ftd_taskpool_v5_20261027_194301'))
    parser.add_argument('--source-result',type=Path,default=Path('./assets/legacy/ftd_complete_request_outputs')/SOURCE_RUN)
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'ftd_kv_feasibility_outputs')
    parser.add_argument('--stage-timeout',type=int,default=1800)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        root=args.run_dir.resolve()
        spec=read_json(root/'spec.json')
        spec.update(read_json(root/'active_stage.json'))
        if spec['method']=='vllm_native_cached':native_probe_worker(spec,root)
        else:hf_probe_worker(spec,root)
        return 0
    if args.stage_timeout<30:parser.error('Timeout must be at least 30 seconds')
    args.output_dir.expanduser().mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=Path(tempfile.mkdtemp(prefix='ftd_kv_feasibility_'+stamp+'_',dir=args.output_dir.expanduser())).resolve()
    (root/'logs').mkdir()
    shutil.copyfile(Path(__file__),root/Path(__file__).name)
    spec=dict(version=PROBE_VERSION,model_dir=str(args.model_dir.expanduser().resolve()),
        taskpool_dir=str(args.taskpool_dir.expanduser().resolve()),source_result=str(args.source_result.expanduser().resolve()),
        scope='smoke',script_sha256=sha256(Path(__file__)),rounds=2,context_lengths=list(CONTEXT_LENGTHS),
        decode_tokens=DECODE_TOKENS,measured_repeats=3,calibration_repeats=2,
        logical_live_tensor_cap_bytes=LIVE_TENSOR_CAP,synthetic_probe=True)
    dump(root/'spec.json',spec)
    state=dict(status='RUNNING',stages=[])
    rc=0
    try:
        source=Path(spec['source_result'])
        if not source.is_dir() or read_json(source/'status.json')['status']!='COMPLETED_COMPLETE_REQUEST_DIAGNOSTIC':
            raise ValueError('Pass the completed prior result directory, including reference_slices, not its tar.gz')
        if sha256(source/'ftd_complete_request_20261101.py')!='389efa3a60d7b539496b6871b3c9d45ac9e656bd4c0f5a940708838dde5e1ed0':
            raise ValueError('Prior reference builder identity changed')
        for filename,expected in SOURCE_HASHES.items():
            if sha256(source/filename)!=expected:
                raise ValueError('Prior frozen source changed: '+filename)
        preflight(spec,root)
        gpus=read_json(root/'preflight.json')['gpu'].get('gpus',[])
        if len(gpus)!=1 or gpus[0]['index']!='0' or os.environ.get('CUDA_VISIBLE_DEVICES','0') not in ('0',''):
            raise ValueError('This probe targets the recorded single physical GPU 0')
        require_idle(samples=3)
        save_signed(root/'probe_cases.json',dict(cases=make_probe_cases(source),synthetic=True,
            note='Repeated synthetic prefixes made from one development token sequence; no ground truth or application accuracy'))
        for i in range(2):
            order=PROBE_METHODS if i==0 else tuple(reversed(PROBE_METHODS))
            for method in order:
                active=dict(method=method,round=i,tag='r'+str(i)+'_'+method)
                dump(root/'status.json',dict(state,active=active))
                run_probe_stage(root,spec,active,args.stage_timeout)
                state['stages'].append(dict(stage=active['tag'],status='COMPLETED'))
        summary=summarize_probe(root)
        state.update(status='COMPLETED_FEASIBILITY_SCREEN',screen_decision=summary['status'])
    except ProbeBlocked as e:
        rc=3
        state.update(status='BLOCKED_GPU_ISOLATION',reason=str(e))
        print(str(e),flush=True)
    except (Exception,KeyboardInterrupt) as e:
        rc=2
        state.update(status='FAILED',error_type=type(e).__name__,error=str(e))
        (root/'error.txt').write_text(traceback.format_exc())
        print('Probe stopped:',str(e),flush=True)
    finally:
        dump(root/'status.json',state)
        (root/'READ_ME.txt').write_text('Small synthetic parameter/KV feasibility probe. No application quality, policy-gain, novelty or physical-memory guarantee.\nExisting reference weights are read in place and never copied into this archive.\n')
        archive=pack_probe(root)
        print('STATUS:',state['status'])
        print('SEND_THIS_ARCHIVE:',archive,flush=True)
    return rc


if __name__=='__main__':
    sys.exit(main())
