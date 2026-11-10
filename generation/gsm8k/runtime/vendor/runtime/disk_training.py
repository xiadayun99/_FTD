"""Fresh 8B LoRA training; disk checkpoints preserve completed tasks."""
import gc
import math
import random
import time
from pathlib import Path
import shutil
import sys

import disk_protocol as sp


def load_base(torch,path,device,dtype):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(path),torch_dtype=dtype,
        device_map={'':str(device)},low_cpu_mem_usage=True,attn_implementation='eager',
        local_files_only=True,trust_remote_code=False).to(dtype=dtype).eval()


def release(torch):
    gc.collect()
    import bounded_io
    bounded_io.trim_heap()
    torch.cuda.empty_cache()


def collate(torch,examples,pad):
    width=max(len(x[0]) for x in examples)
    ids=torch.full((len(examples),width),pad,dtype=torch.long,device='cuda:0')
    mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
    for i,(tokens,targets) in enumerate(examples):
        ids[i,:len(tokens)]=torch.tensor(tokens,device='cuda:0')
        mask[i,:len(tokens)]=1;labels[i,:len(tokens)]=torch.tensor(targets,device='cuda:0')
    return dict(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False)


def save_fp16_adapter(torch,model,directory):
    # Only trainable A/B tensors change dtype. The frozen BF16 base is not saved.
    for name,param in model.named_parameters():
        if param.requires_grad:
            if '.lora_A.' not in name and '.lora_B.' not in name:raise ValueError('Unexpected trainable parameter: '+name)
            param.data=param.data.to(dtype=torch.float16)
    model.save_pretrained(str(directory),safe_serialization=True,selected_adapters=['default'])
    from safetensors import safe_open
    with safe_open(str(directory/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        names=list(f.keys())
        if not names:raise ValueError('Saved adapter is empty')
        for name in names:
            value=f.get_tensor(name)
            if value.dtype!=torch.float16 or not bool(torch.isfinite(value).all()):raise ValueError('Saved adapter dtype/finite check failed')


def train_task(root,task,port):
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import AutoTokenizer
    import bounded_io as bio
    if task not in sp.TASKS:raise ValueError('Unknown task')
    port.require_no_observed_foreign(__import__('os').getpgrp())
    core,e,c,p,o=port.load_core();contract=port.runtime_contract(torch,e,p,'strict_eager')
    if contract!=sp.read(root/'runtime_contract.json'):raise ValueError('Frozen runtime changed before training')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 training is unavailable')
    if torch.cuda.mem_get_info()[0]<48*sp.GIB:raise MemoryError('Training requires at least 48 GiB currently free GPU memory')
    data,stats,info=sp.verify_inputs()
    torch.manual_seed(100+sp.TASKS.index(task));random.seed(100+sp.TASKS.index(task))
    tokenizer=AutoTokenizer.from_pretrained(str(sp.MODEL_DIR),local_files_only=True,trust_remote_code=False)
    work,label_ids=sp.prepare_workload(tokenizer)
    examples,truncated=sp.encode_training(tokenizer,data[task],label_ids[task])
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    folder=root/'trained/lora'/task
    if folder.exists():raise FileExistsError('Refusing to overwrite task output: '+str(folder))
    folder.mkdir(parents=True)
    model=load_base(torch,sp.MODEL_DIR,'cuda:0',torch.bfloat16)
    model.config.use_cache=False
    config=LoraConfig(r=sp.RANKS[task],lora_alpha=2*sp.RANKS[task],lora_dropout=.05,
        bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model=get_peft_model(model,config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train()
    for param in model.parameters():
        if param.requires_grad:param.data=param.data.float()
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)
    begun=time.perf_counter();updates=0;micro=0;pending=[]
    # Accumulation crosses the epoch boundary: 1500 microbatches / 4 = 375 updates.
    for epoch,indices in sp.training_batches(task,len(examples)):
        port.require_no_observed_foreign(__import__('os').getpgrp()) if micro%100==0 else None
        batch=collate(torch,[examples[i] for i in indices],tokenizer.pad_token_id)
        loss=model(**batch).loss
        value=float(loss.detach().float().item())
        if not math.isfinite(value):raise FloatingPointError('Nonfinite training loss')
        (loss/4).backward();pending.append(value);micro+=1
        del loss,batch
        if micro%4==0:
            grad=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            sp.append(root/'training_logs'/(task+'.jsonl'),dict(task=task,epoch=epoch,
                update=updates,microbatches=micro,mean_loss=sum(pending)/len(pending),
                grad_norm=float(grad.item()),elapsed_s=time.perf_counter()-begun,
                memory=bio.memory_snapshot()))
            pending=[]
            if updates%5==0:print('TRAIN:',task,updates,'/375 loss=',round(value,5),flush=True)
    if updates!=375 or micro!=1500 or pending:raise ValueError('Training update accounting differs')
    model.eval();model.config.use_cache=True
    del optimizer,trainable;gc.collect()
    save_fp16_adapter(torch,model,folder)
    files={name:sp.sha(folder/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    log=root/'training_logs'/(task+'.jsonl')
    if len(sp.rows(log))!=375:raise ValueError('Training log must contain exactly 375 updates')
    report=dict(status='TRAINED',task=task,runtime_contract=contract,training_data=stats[task],
        optimizer_updates=updates,microbatches=micro,epochs=2,examples_seen=6000,
        train_sequences_truncated=truncated,saved_adapter_dtype='float16',
        adapter_files_sha256=files,training_log_sha256=sp.sha(log),elapsed_seconds=time.perf_counter()-begun,
        trained_new_8b_adapter=True,used_06b_adapter=False,quality_selected_checkpoint=False,
        training_base_dtype='bfloat16',training_adapter_dtype='float32',
        protocol_sha256=sp.sha(root/'protocol.json'),asset_identity_sha256=sp.sha(root/'asset_identity.json'))
    sp.dump(folder/'training_contract.json',report)
    del model;release(torch)
    return report

