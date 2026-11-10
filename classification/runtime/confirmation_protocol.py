"""Predeclared five-method confirmation on new questions from the known tasks."""
import hashlib
METHODS=('shared','direct','node','native','host_copy')
SEED='FTD_FRESH_CONFIRM_20261106_A'
DATASETS={'mrpc':('nyu-mll/glue','mrpc'),'qnli':('nyu-mll/glue','qnli'),'agnews':('fancyzhx/ag_news','default')}
NAMES={'mrpc':['not_equivalent','equivalent'],'qnli':['entailment','not_entailment'],'agnews':['World','Sports','Business','Sci/Tech']}
GOLD={'mrpc':NAMES['mrpc'],'qnli':NAMES['qnli'],'agnews':['world','sports','business','science']}
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def orders():
    rows=[tuple(METHODS[(x+i)%5] for x in (0,1,4,2,3)) for i in range(5)]
    return rows+[tuple(reversed(row)) for row in rows]
def jobs():
    result=[dict(key='prepare_inputs',phase='prepare'),dict(key='preflight_reference',phase='trace',method='host_copy',tag='preflight_reference',reference=True)]
    result.extend(dict(key=f'r{i}_{m}',phase='trace',method=m,tag=f'r{i}_{m}',block=i) for i,row in enumerate(orders()) for m in row)
    return result
def candidate_indices(task,n):
    if task=='mrpc':return list(range(n))
    blocks=sorted(range((n+99)//100),key=lambda i:digest(f'{SEED}:pool:{task}:{i}'))
    result=[]
    for i in blocks:
        result.extend(range(i*100,min(n,(i+1)*100)))
        if len(result)>=4096:break
    return sorted(result[:4096])
def protocol():
    return dict(version='ftd_fresh_confirmation_20261106_v1',seed=SEED,methods=list(METHODS),orders=[list(x) for x in orders()],
        formal_blocks=10,formal_stages=50,formal_requests=19200,new_questions=384,questions_per_task=128,
        source_split='train',candidate_pool='all MRPC; 4096 deterministic block-spread rows per other task',
        selection='SHA256 order; exclude recorded train/eval/dev, old checks/warmups and prefix examples; labels and scores unused',
        query_cap_tokens=768,query_truncation='head 64 plus marker plus tail',known_contexts_and_schedule_reused=True,
        project_wide_blind_test=False,official_test_benchmark=False,new_independent_test=False,
        fresh_input_confirmation_relative_to_supplied_ledger=True,semantic_near_duplicate_exclusion_guaranteed=False,
        inference_dtype='float16',attention='eager',logical_cap_bytes=32*2**30,kv_cap_bytes=512*2**20,
        fresh_process_per_stage=True,full_canonical_validation_per_worker=True,
        host_copy='standard pageable CPU canonical reference to GPU; no tuned transfer engine',
        old_banks_and_training_reused=True,retraining=False,automatic_tuning=False,controller_claim=False,
        repeated_blocks_are_new_accuracy_examples=False,previous_timings_pooled=False,submission_ready=False)
