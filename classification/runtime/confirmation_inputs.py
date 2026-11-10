"""Download/freeze new inputs before any GPU model is loaded."""
import collections,copy,html,importlib,json,os,sys,time,unicodedata
from pathlib import Path
import urllib.parse,urllib.request
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/runtime'))
import disk_protocol as sp
from confirmation_protocol import DATASETS,NAMES,GOLD,SEED,digest,candidate_indices

def norm(text):return ' '.join(unicodedata.normalize('NFKC',html.unescape(text)).casefold().split())

def format_row(task,row):
    value=row['label']
    if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<len(GOLD[task]):raise ValueError('Source label convention changed')
    if task=='mrpc':
        fragments=[row['sentence1'],row['sentence2']];sid=str(row['idx'])
        query=('Instruction: Determine whether the two sentences have equivalent meanings.\n'
            f'Sentence 1: {fragments[0]}\nSentence 2: {fragments[1]}\n'
            'Answer with one label: equivalent or not_equivalent.\nAnswer:')
    elif task=='qnli':
        fragments=[row['question'],row['sentence']];sid=str(row['idx'])
        query=('Instruction: Determine whether the sentence contains an answer to the question.\n'
            f'Question: {fragments[0]}\nSentence: {fragments[1]}\n'
            'Answer with one label: entailment or not_entailment.\nAnswer:')
    else:
        fragments=[row['text']];sid=digest(row['text'])
        query=('Instruction: Classify the news into world, sports, business, or science. Science includes science and technology.\n'
            f'News: {fragments[0]}\nAnswer with one label: world, sports, business, or science.\nAnswer:')
    if any(not isinstance(x,str) or not norm(x) for x in fragments):raise ValueError('Missing source text')
    return dict(task=task,source_id=sid,source_row_idx=row['_row_idx'],query=query,
        raw_prompt_sha256=digest(query),gold=GOLD[task][value],fragments=[norm(x)[:(64 if task=='agnews' else 160)] for x in fragments])

def parse_page(page,offset,task):
    n=page['num_rows_total']
    if not isinstance(n,int) or n<=offset:raise ValueError('Invalid dataset size')
    feature=next(x for x in page['features'] if x['name']=='label')
    if feature['type']['names']!=NAMES[task]:raise ValueError('Dataset class labels changed')
    if len(page['rows'])!=min(100,n-offset):raise ValueError('Incomplete row page')
    result=[]
    for i,item in enumerate(page['rows']):
        if item['row_idx']!=offset+i or item.get('truncated_cells'):raise ValueError('Truncated or reordered source row')
        result.append(dict(item['row'],_row_idx=offset+i))
    return result

def http_page(root,task,offset):
    repo,config=DATASETS[task];params=dict(dataset=repo,config=config,split='train',offset=offset,length=100)
    path=root/'data/raw/viewer'/task/(str(offset)+'.json')
    if path.exists():
        saved=sp.read(path)
        if saved['request']!=params or sp.identity(saved['response'])!=saved['response_sha256']:raise ValueError('Cached input page changed')
        parse_page(saved['response'],offset,task);return saved['response']
    url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(params)
    error=None
    for attempt in range(4):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'FTD-input-confirmation/1.0'})
            with urllib.request.urlopen(request,timeout=45) as response:
                raw=response.read(32*sp.MIB+1)
                headers={k:response.headers.get(k) for k in ('ETag','X-Revision','Date')}
            if len(raw)>32*sp.MIB:raise ValueError('Oversized input response')
            page=json.loads(raw);parse_page(page,offset,task)
            sp.dump(path,dict(request=params,response=page,response_sha256=sp.identity(page),headers=headers))
            return page
        except (OSError,ValueError,KeyError) as exc:
            error=exc
            if attempt<3:time.sleep(1+2*attempt)
    raise RuntimeError('Dataset download failed; cached pages retained for resume: '+task+': '+str(error))

def snapshot(root,task):
    path=root/'data/raw/snapshots'/(task+'.json')
    if path.exists():
        obj=sp.read(path)
        if obj['rows_sha256']!=sp.identity(obj['rows']):raise ValueError('Candidate snapshot changed')
        return obj
    os.environ['HF_HUB_OFFLINE']='0';os.environ['HF_DATASETS_OFFLINE']='0'
    rows=None;method='official_rows_api';details={}
    try:
        ds=importlib.import_module('datasets');repo,config=DATASETS[task]
        data=ds.load_dataset(repo,config,split='train',token=False)
        if list(data.features['label'].names)!=NAMES[task]:raise ValueError('Dataset class labels changed')
        rows=[dict(data[i],_row_idx=i) for i in candidate_indices(task,len(data))]
        method='datasets_cache_or_hub';details=dict(version=ds.__version__,fingerprint=data._fingerprint,total_rows=len(data))
    except (ImportError,OSError,RuntimeError,ValueError,ConnectionError,TimeoutError) as exc:
        details=dict(datasets_issue=type(exc).__name__+': '+str(exc)[:400])
        print('DATA_SOURCE: using official JSON rows API for',task,flush=True)
    if rows is None:
        first=http_page(root,task,0);n=first['num_rows_total'];indices=set(candidate_indices(task,n));rows=[]
        for number,offset in enumerate(sorted({i//100*100 for i in indices})):
            page=first if offset==0 else http_page(root,task,offset)
            if page['num_rows_total']!=n or page['features']!=first['features']:raise ValueError('Dataset structure changed while downloading')
            rows.extend(row for row in parse_page(page,offset,task) if row['_row_idx'] in indices)
            if number%5==0:print('DATA_PAGE:',task,offset,'/',n,flush=True)
        details.update(total_rows=n,raw_pages_preserved=True,git_revision_pinned=False)
    if len(rows)!=len({x['_row_idx'] for x in rows}):raise ValueError('Duplicate source row positions')
    obj=dict(dataset=list(DATASETS[task]),split='train',method=method,details=details,rows=rows,rows_sha256=sp.identity(rows))
    sp.dump(path,obj);return obj

def exclusions(pool,previous,files):
    ids={t:set() for t in sp.TASKS};texts={t:[] for t in sp.TASKS};hashes={t:set() for t in sp.TASKS}
    for rel,expected in files.items():
        path=pool/rel
        if sp.sha(path)!=expected:raise ValueError('Recorded train/eval/dev file changed: '+str(path))
        for row in sp.rows(path):
            task=row['task'];ids[task].add(str(row['source_id']));texts[task].append(norm(row['prompt']))
    for row in previous['warmup']+previous['check']:
        task=row['task'];ids[task].add(str(row['source_id']));texts[task].append(norm(row['query']))
        # Legacy warmups have query text and IDs, but no raw-prompt digest.
        # Retain both the actual query digest and any separately stored raw
        # digest: stored raw prompts can use an earlier prompt template.
        hashes[task].add(digest(row['query']))
        stored=row.get('raw_prompt_sha256')
        if stored is not None:
            if not isinstance(stored,str) or len(stored)!=64 or any(c not in '0123456789abcdef' for c in stored):
                raise ValueError('Invalid stored raw-prompt digest: '+str(row.get('case_id',task)))
            hashes[task].add(stored)
    for key,head in previous['heads'].items():texts[key.split(':')[0]].append(norm(head['text']))
    return ids,texts,hashes

def truncate(tokenizer,query):
    ids=tokenizer.encode(query,add_special_tokens=False)
    if len(ids)<=768:return query,False
    marker=tokenizer.encode('\n[... middle omitted ...]\n',add_special_tokens=False);tail=768-64-len(marker)
    if tail<64:raise ValueError('Unexpected marker tokenization')
    result=tokenizer.decode(ids[:64]+marker+ids[-tail:],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if len(tokenizer.encode(result,add_special_tokens=False))>768:raise ValueError('Truncation retokenization exceeds cap')
    return result,True

def select(task,rows,known_ids,known_texts,known_hashes,tokenizer,count=128):
    candidates=[format_row(task,row) for row in rows]
    candidates.sort(key=lambda x:digest(SEED+':'+task+':'+x['source_id']+':'+x['raw_prompt_sha256']))
    selected=[];seen_ids=set();seen_inputs=set();reasons=collections.Counter()
    for case in candidates:
        sid=case['source_id'];fragments=case.pop('fragments')
        if sid in known_ids or sid in seen_ids:reasons['known_or_duplicate_id']+=1;continue
        if case['raw_prompt_sha256'] in known_hashes or any(all(fragment in text for fragment in fragments) for text in known_texts):
            reasons['known_content']+=1;continue
        query,cut=truncate(tokenizer,case['query']);effective=norm(query)
        if effective in seen_inputs or effective in known_texts:reasons['duplicate_effective_input']+=1;continue
        case.update(query=query,query_truncated=cut,effective_query_sha256=digest(query))
        selected.append(case);seen_ids.add(sid);seen_inputs.add(effective)
        if len(selected)==count:break
    if len(selected)!=count:raise ValueError(f'Only {len(selected)} eligible {task} inputs; require {count}; GPU inference has not started')
    return selected,dict(candidate_rows=len(candidates),selected=len(selected),excluded_before_selection=dict(reasons),
        label_counts=dict(collections.Counter(x['gold'] for x in selected)),truncated=sum(x['query_truncated'] for x in selected))

def prepare(root,source):
    from transformers import AutoTokenizer
    work=copy.deepcopy(sp.read(HERE/'input/previous_workload.json'));binding=sp.read(HERE/'input/source_binding.json')
    ids,texts,hashes=exclusions(sp.POOL_DIR,work,binding['known_data_files'])
    tokenizer=AutoTokenizer.from_pretrained(str(sp.MODEL_DIR),local_files_only=True,trust_remote_code=False)
    selected={};reports={}
    for task in sp.TASKS:
        print('SELECT_NEW_INPUTS:',task,flush=True);data=snapshot(root,task)
        selected[task],reports[task]=select(task,data['rows'],ids[task],texts[task],hashes[task],tokenizer)
    at={t:0 for t in sp.TASKS}
    for case in work['check']:
        task=case['task'];new=selected[task][at[task]];at[task]+=1
        case['schedule_source_request_key']=case['request_key'];case.pop('source_request_key',None);case.update(new)
        case.update(case_id=task+':fresh_train:'+new['source_id'],phase='frozen_fresh_confirmation')
        head=work['heads'][case['head_id']];head['tokens']=tokenizer.encode(head['text'],add_special_tokens=False)
        case['prefix']=tokenizer.encode(head['text']+case['query'],add_special_tokens=False)
        if case['prefix'][:len(head['tokens'])]!=head['tokens']:raise ValueError('Prefix token boundary changed')
        case['request_key']=sp.identity({k:case[k] for k in ('task','case_id','head_id','prefix','label_ids','via')})
    if at!={t:128 for t in sp.TASKS}:raise ValueError('Unexpected task schedule')
    work.update(new_questions=384,new_independent_test=False,project_wide_blind_test=False,official_test_benchmark=False,
        reused_previously_observed_inputs=False,fresh_input_confirmation_relative_to_recorded_ledger=True,
        selection='SHA256 after source-ID, content-fragment and effective-input exclusions; no model/label selection',
        query_truncation_rule='768 tokens, first 64 plus marker and tail; identical for all methods',
        scope='New questions from official train remainder; old prefixes/schedule; no project-wide blindness claim',
        study_scope='Post-freeze new-input confirmation relative to supplied records; three known tasks',
        fresh_holdout_inputs=True,method_freeze_id='FTD_FRESH_CONFIRM_20261106_A')
    work['schedule_source_workload_sha256']=sp.sha(HERE/'input/previous_workload.json');work.pop('source_workload_sha256',None)
    work['cohort_id']=sp.identity([{k:x[k] for k in ('task','source_id','raw_prompt_sha256','effective_query_sha256','gold')} for x in work['check']])
    report=dict(status='FROZEN_BEFORE_GPU_INFERENCE',tasks=reports,known_data_files=binding['known_data_files'],
        content_exclusion='First 64 normalized AG News characters or 160 characters of each sentence/question field must not coexist in a known prompt/prefix',
        source_id_aliases_not_assumed_compatible_for_AG_News=True,semantic_near_duplicate_exclusion_guaranteed=False,
        new_questions=384,new_tasks=0,cohort_id=work['cohort_id'],official_test_benchmark=False,project_wide_blind_test=False,quality_used_for_selection=False)
    sp.dump(root/'engine/input/workload.json',work);sp.dump(root/'data/selection_report.json',report);sp.dump(root/'data/selected_rows.json',selected)
    return [p for folder in (root/'data',root/'engine/input') for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.partial')]
