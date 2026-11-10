"""Frozen inputs and experiment settings for disk-checkpointed Qwen3-8B validation."""
import copy
import hashlib
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
MIB=2**20
GIB=2**30
TASKS=('mrpc','qnli','agnews')
RANKS={'mrpc':4,'qnli':16,'agnews':8}
LABELS={'mrpc':{'equivalent','not_equivalent'},'qnli':{'entailment','not_entailment'},
        'agnews':{'world','sports','business','science'}}
CAP=32*GIB
KV_CAP=512*MIB
START_FREE_DISK=100*GIB
DISK_FLOOR=12*GIB
HOST_START=104*GIB
MUTABLE_BYTES=13891534848
MODEL_BYTES=16381470720
MODEL_DIR=Path('./assets/legacy/models/Qwen3-8B')
POOL_DIR=Path('./assets/legacy/ftd_taskpool_v5_20261027_194301')
SOURCE_WORKLOAD_SHA='80eaff3b6634fee9012c0bd4340e36073d3b320c9820c92e75de45e7b6712f68'
MODEL_CONFIG_SHA='f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30'
MODEL_INDEX_SHA='f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc'
INPUT_INVENTORY_SHA='be612d74d3958b4051093938d037fa47240e7eb107706a620b2f74197a8e074a'


def read(path):return json.loads(Path(path).read_text())


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(MIB),b''):h.update(block)
    return h.hexdigest()


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def source_inventory():
    if sha(HERE/'input/server_inventory.json')!=INPUT_INVENTORY_SHA:raise ValueError('Reviewed inventory changed')
    return read(HERE/'input/server_inventory.json')


def verify_inputs(model_dir=MODEL_DIR,pool=POOL_DIR):
    inv=source_inventory()
    if sha(model_dir/'config.json')!=MODEL_CONFIG_SHA or sha(model_dir/'model.safetensors.index.json')!=MODEL_INDEX_SHA:
        raise ValueError('Use the reviewed unquantized Qwen3-8B base')
    info=next(x for x in inv['models'] if x['path']==str(MODEL_DIR))
    for shard in info['shards']:
        p=model_dir/shard['file']
        if not p.is_file() or p.stat().st_size!=shard['bytes']:raise ValueError('Missing/changed model shard: '+str(p))
    if sha(HERE/'input/workload_source.json')!=SOURCE_WORKLOAD_SHA:raise ValueError('Source questions changed')
    ledger={x['file']:x for x in inv['taskpool']['data_files']}
    source=read(HERE/'input/workload_source.json')
    held_hashes={x['raw_prompt_sha256'] for x in source['check']}
    held_ids={t:{str(x['source_id']) for x in source['check'] if x['task']==t} for t in TASKS}
    data={};stats={}
    for task in TASKS:
        rel='data/'+task+'_train.jsonl';path=pool/rel
        if sha(path)!=ledger[rel]['sha256']:raise ValueError('Training file differs: '+task)
        rr=rows(path)
        if len(rr)!=3000:raise ValueError('Expected 3000 training rows per task')
        labels={};overlap=[]
        for row in rr:
            if row.get('task')!=task or row.get('label') not in LABELS[task] or not isinstance(row.get('prompt'),str):
                raise ValueError('Training task/label/schema differs: '+task)
            digest=hashlib.sha256(row['prompt'].encode()).hexdigest()
            sid=str(row.get('source_id',''))
            if digest in held_hashes or (sid.isdecimal() and sid in held_ids[task]):overlap.append(sid)
            labels[row['label']]=labels.get(row['label'],0)+1
        if overlap:raise ValueError('Training/check source overlap: '+task+':'+str(overlap[:8]))
        data[task]=rr;stats[task]=dict(rows=len(rr),label_counts=labels,sha256=sha(path),overlaps=0)
    return data,stats,info


def asset_files(model_dir,pool):
    index=read(model_dir/'model.safetensors.index.json')
    files={str(model_dir/name):sha(model_dir/name) for name in sorted(set(index['weight_map'].values()))}
    for p in model_dir.iterdir():
        if p.is_file() and p.suffix in ('.json','.txt','.model') and p.stat().st_size<64*MIB:files[str(p)]=sha(p)
    for task in TASKS:
        p=pool/'data'/(task+'_train.jsonl');files[str(p)]=sha(p)
    return files


def specification(root,method):
    return dict(model_dir=str(MODEL_DIR),taskpool_dir=str(root/'trained'),
        source_result=str(root/'artifacts/references'),cap_bytes=CAP,kv_bank_cap_bytes=KV_CAP,
        canonical_transition_operator='direct' if method=='direct' else 'node',
        cache_budget_policy='shared',collect_dead_kv_versions=False,limit=384,
        policy={'direct':'canonical_patch_cached','node':'canonical_patch_cached',
                'native':'hf_native_cached','reference':'host_copy_cached'}[method])


def protocol():
    return dict(version='ftd_8b_disk_20261105_v1',model_dir=str(MODEL_DIR),taskpool_dir=str(POOL_DIR),
        tasks=list(TASKS),ranks=RANKS,lora_scale=2,training_rows_per_task=3000,
        epochs=2,microbatch=4,gradient_accumulation=4,optimizer_updates_per_task=375,
        learning_rate=2e-4,weight_decay=0.,adam_betas=[.9,.999],adam_eps=1e-8,
        max_grad_norm=1.,task_seeds={t:100+i for i,t in enumerate(TASKS)},
        lora_dropout=.05,max_train_tokens=512,
        training_base_dtype='bfloat16',training_adapter_dtype='float32',saved_adapter_dtype='float16',inference_dtype='float16',
        attention='eager',fp16_reduced_precision_reduction=False,
        training_tokenization='separate prompt+newline Answer: tokens and admitted label tokens; EOS supervised',
        optimizer_uses_only_train_file=True,checkpoint_selected_by_quality=False,
        cached_vs_full_prediction_differences='reported; do not choose an adapter or precision based on them',
        logical_live_cap_bytes=CAP,retained_kv_cap_bytes=KV_CAP,
        start_free_disk_bytes=START_FREE_DISK,disk_floor_bytes=DISK_FLOOR,
        host_start_available_bytes=HOST_START,projection_expected=252,smoke_requests_per_operator=6,
        full_grouped_checks_per_operator=24,full_grouped_validation_for_each_canonical_worker=True,
        formal_blocks=6,formal_stages=18,
        formal_requests=6912,unique_questions=384,new_questions=0,
        reused_inspected_inputs=True,new_independent_test=False,
        source_workload_sha256=SOURCE_WORKLOAD_SHA,inventory_sha256=INPUT_INVENTORY_SHA,
        target_reference_storage='Standard safetensors files: base mutable slice plus three new 8B merged reference slices',
        correction_storage='Standard safetensors plus JSON metadata, one admitted operator bank loaded per worker',
        execution='Fresh worker process for each preparation, admission and formal stage; one active comparison model per worker',
        independent_benchmark_worker_per_stage=True,allocator_residual_threshold_used=False,
        loading_and_preparation_outside_timing=True,task_switches_within_stage_inside_timing=True,
        checkpoint_resume='Reuse hash-verified completed phases; retrain an interrupted task from its beginning',
        automatic_budget_precision_or_rank_tuning=False,controller_claim=False,submission_ready=False)


def training_batches(task,count):
    """One deterministic pass per epoch; accumulation continues across epochs."""
    if task not in TASKS or count!=3000:raise ValueError('Training plan requires the frozen task and 3000 rows')
    for epoch in range(2):
        order=list(range(count));random.Random(100+TASKS.index(task)+epoch).shuffle(order)
        for start in range(0,count,4):yield epoch,order[start:start+4]


def prepare_workload(tokenizer):
    work=copy.deepcopy(read(HERE/'input/workload_source.json'))
    labels={}
    for case in work['warmup']+work['check']:
        task=case['task']
        if task not in labels:
            labels[task]={}
            for name,ids in case['label_ids'].items():
                text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                if text.strip()!=name:raise ValueError('8B tokenizer does not decode the admitted label convention')
                labels[task][name]=tokenizer.encode(text,add_special_tokens=False)
        case['label_ids']=labels[task]
        head=work['heads'][case['head_id']]
        head['tokens']=tokenizer.encode(head['text'],add_special_tokens=False)
        prefix=tokenizer.encode(head['text']+case['query'],add_special_tokens=False)
        if prefix[:len(head['tokens'])]!=head['tokens']:raise ValueError('Prefix token boundary changed')
        case['prefix']=prefix;case['source_request_key']=case['request_key']
        case['request_key']=identity({k:case[k] for k in ('task','case_id','head_id','prefix','label_ids','via')})
    work.update(source_workload_sha256=SOURCE_WORKLOAD_SHA,new_questions=0,
                reused_previously_observed_inputs=True,new_independent_test=False,
                scope='8B model-scale check with model-specific effective tokens')
    return work,labels


def encode_training(tokenizer,records,label_ids):
    eos=tokenizer.eos_token_id
    if eos is None:raise ValueError('Tokenizer lacks EOS')
    result=[];truncated=0
    for row in records:
        suffix=label_ids[row['label']]+[eos]
        prefix=tokenizer.encode(row['prompt']+'\nAnswer:',add_special_tokens=False)
        budget=512-len(suffix)
        if budget<=64:raise ValueError('Unexpected label length')
        if len(prefix)>budget:prefix=prefix[:budget-64]+prefix[-64:];truncated+=1
        result.append((prefix+suffix,[-100]*len(prefix)+suffix))
    return result,truncated
