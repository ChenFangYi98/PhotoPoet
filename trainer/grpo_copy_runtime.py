# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Standalone native-checkpoint runtime for train_grpo - 副本.py.

The selected script supplies its original reward and CISPO/GRPO objective.
This module provides deterministic prompts, aligned sampling and full archives.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import datasets  # Windows pyarrow/torch DLL ordering
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataset.lm_dataset import pre_processing_chat
from model.model_mindchen import MindChenForCausalLM
from model.checkpoint_config import load_checkpoint_config
from trainer.grpo_control import StopController, StopRequested, live_state
from trainer.grpo_data_quality import prompt_issue
from trainer.grpo_resources import (ROOT, DEFAULT_SOURCE, DEFAULT_REWARD, atomic_json, atomic_save,
    ensure_reward_model, file_digest, project_path, protect_source, snapshot_source, source_weight,
    write_checkpoint_config)
from trainer.grpo_repetition import repetition_diagnostics
from trainer.trainer_utils import LMForRewardModel

FORMAT = 'mindchen_grpo_copy_cispo_v1'


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Native SFT best -> copy-script CISPO/GRPO')
    p.add_argument('--checkpoint',default=str(DEFAULT_SOURCE))
    p.add_argument('--save_dir',default=str(ROOT/'out/grpo_copy'))
    p.add_argument('--run_name',default='')
    p.add_argument('--resume',default='')
    p.add_argument('--save_weight',default='grpo_copy')
    p.add_argument('--data_path',default=str(ROOT/'dataset/rlaif.jsonl'))
    p.add_argument('--reward_model_path',default=str(DEFAULT_REWARD))
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--reward_device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--ref_device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='bfloat16')
    p.add_argument('--batch_size',type=int,default=1)
    p.add_argument('--num_generations',type=int,default=6)
    p.add_argument('--policy_micro_batch_size',type=int,default=1)
    p.add_argument('--accumulation_steps',type=int,default=4)
    p.add_argument('--learning_rate',type=float,default=3e-7)
    p.add_argument('--weight_decay',type=float,default=.01)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--max_steps',type=int,default=100,help='Optimizer updates; 0 means complete epochs')
    p.add_argument('--stop_after_steps',type=int,default=0)
    p.add_argument('--max_seq_len',type=int,default=768)
    p.add_argument('--max_gen_len',type=int,default=256)
    p.add_argument('--temperature',type=float,default=.8)
    p.add_argument('--thinking_ratio',type=float,default=.9)
    p.add_argument('--beta',type=float,default=.1)
    p.add_argument('--loss_type',choices=['cispo','grpo'],default='cispo')
    p.add_argument('--epsilon',type=float,default=.2)
    p.add_argument('--epsilon_high',type=float,default=5.)
    p.add_argument('--grad_clip',type=float,default=1.)
    p.add_argument('--save_interval',type=int,default=20)
    p.add_argument('--log_interval',type=int,default=1)
    p.add_argument('--eval_interval',type=int,default=20)
    p.add_argument('--validation_samples',type=int,default=16)
    p.add_argument('--validation_fraction',type=float,default=.05)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu_threads',type=int,default=4)
    p.add_argument('--num_workers',type=int,default=0)
    p.add_argument('--hidden_size',type=int,default=None)
    p.add_argument('--num_hidden_layers',type=int,default=None)
    p.add_argument('--use_moe',type=int,choices=[0,1],default=None)
    p.add_argument('--rollout_engine',choices=['torch'],default='torch')
    p.add_argument('--debug_mode',action='store_true')
    p.add_argument('--debug_interval',type=int,default=20)
    p.add_argument('--prepare_only',action='store_true')
    a=p.parse_args(argv)
    positive=['batch_size','num_generations','policy_micro_batch_size','accumulation_steps','epochs',
              'max_seq_len','max_gen_len','save_interval','log_interval','eval_interval',
              'validation_samples','cpu_threads','debug_interval']
    if any(getattr(a,n)<1 for n in positive) or a.num_generations<2:
        p.error('Lengths, batch sizes and intervals must be positive; at least two candidates')
    if a.num_workers!=0:
        p.error('Use num_workers=0 for reproducible Windows resume')
    if min(a.max_steps,a.stop_after_steps)<0 or not 0<a.validation_fraction<.5 or not 0<=a.thinking_ratio<=1:
        p.error('Invalid step count or ratio')
    for name in ('learning_rate','weight_decay','temperature','beta','epsilon','epsilon_high','grad_clip'):
        value=getattr(a,name)
        if not math.isfinite(value) or value<0 or (name not in ('weight_decay','beta') and value==0):
            p.error('Invalid finite parameter: '+name)
    for name in ('run_name','save_weight'):
        value=getattr(a,name)
        if (name=='save_weight' and not value) or (value and (Path(value).name!=value or value in ('.','..') or any(c in value for c in '/\\:'))):
            p.error('Invalid filename: '+name)
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        p.error('This native-checkpoint entry is single-process; do not use torchrun')
    return a


def amp(device,dtype):
    return torch.autocast('cuda',dtype=getattr(torch,dtype)) if str(device).startswith('cuda') and dtype!='float32' else nullcontext()


def code_signature(script):
    names=[Path(script).resolve().relative_to(ROOT).as_posix(),'trainer/grpo_copy_runtime.py',
        'model/model_mindchen.py','model/checkpoint_config.py','model/tokenizer.json',
        'model/tokenizer_config.json','dataset/lm_dataset.py','trainer/trainer_utils.py',
        'trainer/grpo_resources.py','trainer/grpo_io.py','trainer/grpo_control.py',
        'trainer/grpo_data_quality.py','trainer/grpo_repetition.py']
    return {name:file_digest(ROOT/name) for name in names}


class CopyPrompts:
    def __init__(self,path,tokenizer,args):
        self.tokenizer,self.args=tokenizer,args
        self.train,self.valid,self.rejected=[],[],[]
        seen=set();self.duplicates=0;self.removed_tails=0
        for number,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
            if not line.strip():continue
            row=json.loads(line)
            messages=[dict(role=m.get('role'),content=m.get('content')) for m in row.get('conversations',row.get('messages',[]))]
            removed=0
            while messages and messages[-1]['role']=='assistant':messages.pop();removed+=1
            self.removed_tails+=int(removed>1)
            issue=prompt_issue(messages)
            if issue:
                self.rejected.append(dict(line=number,reason=issue));continue
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,open_thinking=False)
            key=hashlib.sha256(rendered.encode()).hexdigest()
            if key in seen:self.duplicates+=1;continue
            seen.add(key)
            item=dict(id=f'rlaif_{number}',messages=messages)
            (self.valid if int(key[:8],16)/2**32<args.validation_fraction else self.train).append(item)
        self.valid.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
        if not self.train or not self.valid:raise ValueError('Training and held-out prompts must both be nonempty')

    def prompt(self,row,epoch=0,training=True):
        seed=int(hashlib.sha256(f'{self.args.seed}:{epoch}:{row["id"]}'.encode()).hexdigest()[:16],16)
        rng=random.Random(seed)
        messages=pre_processing_chat(row['messages'],.2 if training else 0.,rng=rng)
        thinking=training and rng.random()<self.args.thinking_ratio
        def render():
            return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,open_thinking=thinking)
        text=render()
        while len(self.tokenizer(text,add_special_tokens=False).input_ids)>self.args.max_seq_len:
            messages=[dict(m) for m in messages]
            if len(messages)>1:
                messages.pop(0)
                while len(messages)>1 and messages[0]['role']!='user':messages.pop(0)
            else:
                ids=self.tokenizer(messages[0]['content'],add_special_tokens=False).input_ids
                extra=len(self.tokenizer(text,add_special_tokens=False).input_ids)-self.args.max_seq_len
                keep=len(ids)-extra-8
                if keep<1:raise ValueError('Prompt window cannot hold its chat template')
                messages[0]['content']=self.tokenizer.decode(ids[-keep:],skip_special_tokens=True)
            text=render()
        return text


def completion_mask(ids,eos):
    is_eos=ids.eq(eos)
    return (is_eos.cumsum(1)-is_eos.long()).eq(0)


def logps(model,ids,mask,keep,args,with_aux=False):
    with amp(ids.device,args.dtype):
        result=model(ids,attention_mask=mask,logits_to_keep=keep+1,use_cache=False)
    values=F.log_softmax(result.logits[:,:-1].float()/args.temperature,dim=-1).gather(-1,ids[:,-keep:,None]).squeeze(-1)
    return (values,result.aux_loss) if with_aux else values


def rollout(model,tokenizer,prompts,args,greedy=False):
    encoded=tokenizer(prompts,padding=True,return_tensors='pt',add_special_tokens=False,
                      return_token_type_ids=False).to(args.device)
    count=1 if greedy else args.num_generations
    prefix=encoded.input_ids.repeat_interleave(count,0)
    prefix_mask=encoded.attention_mask.repeat_interleave(count,0)
    model.eval()
    with torch.no_grad(),amp(args.device,args.dtype):
        ids=model.generate(input_ids=prefix,attention_mask=prefix_mask,max_new_tokens=args.max_gen_len,
            do_sample=not greedy,temperature=args.temperature if not greedy else 1.,top_p=1.,top_k=0,
            repetition_penalty=1.,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id).clone()
    completions=ids[:,prefix.size(1):]
    valid=completion_mask(completions,tokenizer.eos_token_id)
    full_mask=torch.cat([prefix_mask,valid.long()],1)
    texts=tokenizer.batch_decode(completions,skip_special_tokens=True)
    return ids,full_mask,valid,texts,completions.eq(tokenizer.eos_token_id).any(1).tolist()


@torch.no_grad()
def frozen_logps(model,ids,mask,keep,device,args):
    return torch.cat([logps(model,ids[i:i+args.policy_micro_batch_size].to(device),
        mask[i:i+args.policy_micro_batch_size].to(device),keep,args).to(ids.device)
        for i in range(0,len(ids),args.policy_micro_batch_size)])


def rng_state():
    return dict(python=random.getstate(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def export_name(model,args,best=False):
    return f'{args.save_weight}{"_best" if best else ""}_{model.config.hidden_size}{"_moe" if model.config.use_moe else ""}.pth'


def save(run,model,optimizer,scheduler,scaler,args,progress,source,signature,data_hash,status):
    print(f'[save] full checkpoint at update={progress["update"]}',flush=True)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    weight=run/export_name(model,args)
    atomic_save(weight,state);write_checkpoint_config(weight,model.config)
    atomic_save(run/'resume.pth',dict(format=FORMAT,args=vars(args),progress=progress,source=source,
        signature=signature,data_sha256=data_hash,model=state,optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state()))
    atomic_json(run/'summary.json',dict(status=status,**progress,source=source,
        export=str(weight),best_weight=str(run/export_name(model,args,True)),algorithm=args.loss_type,
        reward='original_copy_rules_plus_clipped_InternLM2',source_directory_written=False))
    live_state(run,status,progress,last_saved_update=progress['update'])
    print(f'[saved] update={progress["update"]} {weight}',flush=True)


def validate(selected,run,model,tokenizer,reward_model,data,args,progress,control):
    saved_rng=rng_state();old_count=args.num_generations;args.num_generations=1
    records=[]
    try:
        for i,row in enumerate(data.valid[:args.validation_samples],1):
            control.check();live_state(run,'validating',progress,sample=i,samples=min(args.validation_samples,len(data.valid)))
            print(f'[validation sample] update={progress["update"]} {i}/{min(args.validation_samples,len(data.valid))} {row["id"]}',flush=True)
            prompt=data.prompt(row,training=False)
            _,_,_,texts,ended=rollout(model,tokenizer,[prompt],args,greedy=True)
            reward=float(selected.calculate_rewards([prompt],texts,reward_model)[0])
            if not math.isfinite(reward):raise FloatingPointError('Nonfinite original validation reward')
            records.append(dict(id=row['id'],prompt=prompt,response=texts[0],reward=reward,
                ended=ended[0],repetition=repetition_diagnostics(texts[0])))
        metrics=dict(update=progress['update'],count=len(records),reward=sum(r['reward'] for r in records)/len(records),
            repetition_rate=sum(r['repetition']['excessive'] for r in records)/len(records),
            truncated_fraction=sum(not r['ended'] for r in records)/len(records),
            empty_fraction=sum(not r['response'].strip() for r in records)/len(records),
            decoding=dict(mode='greedy',max_new_tokens=args.max_gen_len,top_k=0,top_p=1.,repetition_penalty=1.))
        if metrics['reward']>progress.get('best_reward',-float('inf')):
            weight=run/export_name(model,args,True)
            atomic_save(weight,{k:v.detach().cpu() for k,v in model.state_dict().items()})
            write_checkpoint_config(weight,model.config)
            progress.update(best_reward=metrics['reward'],best_update=progress['update'])
        atomic_json(run/f'validation_samples_{progress["update"]:06d}.json',records)
        with (run/'validation_metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(metrics,ensure_ascii=False)+'\n')
        print('[validation] '+json.dumps(metrics,ensure_ascii=False),flush=True)
        return metrics
    finally:
        args.num_generations=old_count;restore_rng(saved_rng)


def train(selected,run,model,tokenizer,reference,reward_model,data,optimizer,scheduler,scaler,
          args,progress,source,signature,data_hash,control):
    started=time.monotonic();first_update=progress['update']
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(progress['epoch'],args.epochs):
        indices=list(range(len(data.train)));random.Random(args.seed+epoch).shuffle(indices)
        batches=[indices[i:i+args.batch_size] for i in range(0,len(indices),args.batch_size)]
        start=progress['next_batch'] if epoch==progress['epoch'] else 0
        for window in range(start,len(batches),args.accumulation_steps):
            control.check();size=min(args.accumulation_steps,len(batches)-window)
            stats=[];samples=[]
            for batch in range(window,window+size):
                rows=[data.train[i] for i in batches[batch]]
                prompts=[data.prompt(r,epoch,True) for r in rows]
                print(f'[rollout] update={progress["update"]+1} batch={batch+1}/{len(batches)}',flush=True)
                ids,mask,valid,texts,ended=rollout(model,tokenizer,prompts,args)
                rewards=selected.calculate_rewards(prompts,texts,reward_model).detach()
                if not torch.isfinite(rewards).all():raise FloatingPointError('Nonfinite original copy reward; no optimizer update')
                grouped=rewards.reshape(-1,args.num_generations)
                advantages=((grouped-grouped.mean(1,keepdim=True))/(grouped.std(1,unbiased=False,keepdim=True)+1e-4)).flatten()
                old=frozen_logps(model,ids,mask,valid.size(1),args.device,args)
                ref=frozen_logps(reference,ids,mask,valid.size(1),args.ref_device,args)
                model.train()  # Loaded config must have dropout=0; retain original MoE auxiliary loss.
                objective=0.;kl_value=0.;aux_value=0.;ratio_error=0.
                for j in range(0,len(ids),args.policy_micro_batch_size):
                    sl=slice(j,j+args.policy_micro_batch_size)
                    current,aux=logps(model,ids[sl],mask[sl],valid.size(1),args,with_aux=True)
                    loss,kl=selected.policy_objective(current,old[sl],ref[sl],advantages[sl],valid[sl],args)
                    if not torch.isfinite(loss+aux):raise FloatingPointError('Nonfinite CISPO/GRPO loss')
                    proportion=len(ids[sl])/len(ids)
                    scaler.scale((loss+aux)*proportion/size).backward()
                    objective+=float(loss.detach())*proportion;aux_value+=float(aux.detach())*proportion
                    kl_value+=float(kl)*proportion
                    ratio_error=max(ratio_error,float((current.detach()-old[sl])[valid[sl]].abs().max()))
                diagnoses=[repetition_diagnostics(text) for text in texts]
                stats.append(dict(loss=objective+aux_value,policy_loss=objective,aux_loss=aux_value,kl=kl_value,
                    reward=float(rewards.mean()),advantage_std=float(advantages.std(unbiased=False)),
                    mean_tokens=float(valid.sum(1).float().mean()),truncated_fraction=sum(not e for e in ended)/len(ended),
                    repetition_rate=sum(d['excessive'] for d in diagnoses)/len(diagnoses),
                    zero_variance_groups=float((grouped.std(1,unbiased=False)<1e-6).float().mean()),
                    old_current_logp_max_error=ratio_error))
                samples.extend(dict(id=rows[i//args.num_generations]['id'],prompt=prompts[i//args.num_generations],
                    response=text,reward=float(rewards[i]),ended=ended[i],repetition=diagnoses[i]) for i,text in enumerate(texts))
                del ids,mask,valid,current,loss,aux,old,ref,rewards,advantages
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True)
            progress.update(update=progress['update']+1,epoch=epoch,next_batch=window+size)
            if progress['next_batch']==len(batches):progress.update(epoch=epoch+1,next_batch=0)
            record=dict(update=progress['update'],epoch=epoch+1,grad_norm=float(norm),
                learning_rate=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-started,
                **{k:sum(s[k] for s in stats)/len(stats) for k in stats[0]})
            with (run/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
            atomic_json(run/'latest_samples.json',samples)
            live_state(run,'training',progress,last_metrics=record)
            if progress['update']%args.log_interval==0:print('[train] '+json.dumps(record),flush=True)
            if args.debug_mode and progress['update']%args.debug_interval==0:print(json.dumps(samples,ensure_ascii=False),flush=True)
            stop=(progress['update']>=progress['total_updates'] or control.requested or
                  (args.stop_after_steps and progress['update']-first_update>=args.stop_after_steps))
            if not control.requested and (stop or progress['update']%args.eval_interval==0):
                validate(selected,run,model,tokenizer,reward_model,data,args,progress,control)
            if stop or progress['update']==1 or progress['update']%args.save_interval==0 or progress['epoch']>epoch:
                status='completed' if progress['update']>=progress['total_updates'] else ('stopped' if stop else 'running')
                save(run,model,optimizer,scheduler,scaler,args,progress,source,signature,data_hash,status)
            if stop:return


def main(selected,argv=None):
    args=parse_args(argv);selected.args=args
    supplied={value.split('=')[0] for value in (sys.argv[1:] if argv is None else argv) if value.startswith('--')}
    torch.set_num_threads(args.cpu_threads)
    signature=code_signature(selected.__file__)
    checkpoint=None
    if args.resume:
        path=project_path(args.resume)
        if path.name!='resume.pth':raise ValueError('Resume requires the full native resume.pth')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint.get('format')!=FORMAT or checkpoint['signature']!=signature:
            raise ValueError('Copy-script resume code/protocol mismatch; original checkpoint was not changed')
        mutable={name:getattr(args,name) for name in ('device','ref_device','reward_device','stop_after_steps','debug_mode')}
        for key,value in checkpoint['args'].items():
            if key not in mutable and key!='resume' and '--'+key in supplied and getattr(args,key)!=value:
                raise ValueError('Resume cannot change '+key)
        args=argparse.Namespace(**checkpoint['args'])
        for name,value in mutable.items():setattr(args,name,value)
        args.resume=str(path);run=path.parent;source=checkpoint['source']
        snapshot=run/'source_snapshot'/Path(source['snapshot']).name
        if file_digest(snapshot)!=source['sha256']:raise ValueError('Reference snapshot changed')
        config,_=load_checkpoint_config(snapshot)
    else:
        source_path=source_weight(args.checkpoint)
        root=project_path(args.save_dir)
        if not root.is_relative_to(ROOT/'out'):raise ValueError('All training outputs must remain under project/out')
        protect_source(root,source_path)
        run=root/(args.run_name or datetime.now().strftime('cispo_%Y%m%d_%H%M%S_%f'))
        run.mkdir(parents=True,exist_ok=False)
        print(f'[prepare] source snapshot -> {run}',flush=True)
        snapshot,config,source=snapshot_source(source_path,run)
        for name,digest in signature.items():
            target=run/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
            if file_digest(target)!=digest:raise ValueError('Code changed during snapshot: '+name)
    if not run.resolve().is_relative_to((ROOT/'out').resolve()):raise ValueError('Output must stay under project/out')
    protect_source(run,source['path']);selected.args=args
    torch.set_num_threads(args.cpu_threads)
    for key in ('hidden_size','num_hidden_layers','use_moe'):
        if getattr(args,key) is not None and getattr(args,key)!=getattr(config,key):
            raise ValueError('CLI architecture conflicts with bound SFT config: '+key)
    if config.dropout!=0:raise ValueError('Aligned original MoE training requires dropout=0')
    if args.max_seq_len+args.max_gen_len>config.max_position_embeddings:raise ValueError('Context exceeds saved model positions')
    print(f'[run] {run}\n[source read only] {source["path"]}\n[snapshot sha256] {source["sha256"]}',flush=True)
    print(f'[algorithm] {args.loss_type}; original copy rewards; FP32 parameters/{args.dtype} compute\n'
          f'[config] hidden={config.hidden_size} layers={config.num_hidden_layers} moe={config.use_moe} norm_topk_prob={config.norm_topk_prob}',flush=True)
    args.data_path=str(project_path(args.data_path));args.reward_model_path=str(ensure_reward_model(args.reward_model_path))
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True);tokenizer.padding_side='left'
    data=CopyPrompts(args.data_path,tokenizer,args)
    if not checkpoint:
        snapshot_dir=run/'data_snapshot';snapshot_dir.mkdir()
        for split,rows in [('train',data.train),('valid',data.valid)]:
            (snapshot_dir/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    data_hash=dict(original=file_digest(args.data_path),**{s:file_digest(run/'data_snapshot'/f'{s}.jsonl') for s in ('train','valid')})
    if checkpoint and checkpoint['data_sha256']!=data_hash:raise ValueError('Original or snapshotted training data changed')
    if checkpoint:
        for split,rows in [('train',data.train),('valid',data.valid)]:
            saved=[json.loads(line) for line in (run/'data_snapshot'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
            if saved!=rows:raise ValueError('Prompt preprocessing changed')
    atomic_json(run/'data_quality.json',dict(train=len(data.train),valid=len(data.valid),duplicates=data.duplicates,
        normalized_assistant_tails=data.removed_tails,rejected=data.rejected,original_files_modified=False))
    if not checkpoint:
        atomic_json(run/'run_config.json',dict(args=vars(args),source=source,model_config=config.to_dict(),
            signature=signature,data_sha256=data_hash,algorithm=args.loss_type,reward='original_copy_rules'))
    print(f'[data] train={len(data.train)} valid={len(data.valid)} quarantined={len(data.rejected)} duplicates={data.duplicates}',flush=True)
    if args.prepare_only:
        atomic_json(run/'summary.json',dict(status='prepared',source=source,optimizer_updates=0));return
    random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    state=torch.load(snapshot,map_location='cpu',weights_only=True)
    model=MindChenForCausalLM(config);model.load_state_dict(state,strict=True);model.to(args.device)
    reference=MindChenForCausalLM(config);reference.load_state_dict(state,strict=True)
    reference.to(args.ref_device).eval().requires_grad_(False);del state
    print(f'[reward] loading local InternLM2 on {args.reward_device}',flush=True)
    reward_model=LMForRewardModel(args.reward_model_path,device=args.reward_device,
        dtype=torch.float16 if args.reward_device.startswith('cuda') else torch.float32)
    reward_model.model.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    total=math.ceil(math.ceil(len(data.train)/args.batch_size)/args.accumulation_steps)*args.epochs
    total=min(total,args.max_steps) if args.max_steps else total
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total,eta_min=args.learning_rate/10)
    scaler=torch.amp.GradScaler('cuda',enabled=args.device.startswith('cuda') and args.dtype=='float16')
    progress=dict(update=0,epoch=0,next_batch=0,total_updates=total)
    if checkpoint:
        model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
        progress=checkpoint['progress'];restore_rng(checkpoint['rng']);del checkpoint
        path=run/'metrics.jsonl'
        if path.exists():
            retained=[line for line in path.read_text(encoding='utf-8').splitlines() if json.loads(line)['update']<=progress['update']]
            path.write_text(''.join(line+'\n' for line in retained),encoding='utf-8')
    if progress['update']>=progress['total_updates']:
        print('[completed] No updates remain',flush=True);return
    with StopController(run) as control:
        try:
            if progress['update']==0:validate(selected,run,model,tokenizer,reward_model,data,args,progress,control)
            print(f'[start] update={progress["update"]}/{total}; algorithm={args.loss_type}',flush=True)
            train(selected,run,model,tokenizer,reference,reward_model,data,optimizer,scheduler,scaler,
                  args,progress,source,signature,data_hash,control)
        except StopRequested:
            save(run,model,optimizer,scheduler,scaler,args,progress,source,signature,data_hash,'stopped')
        except BaseException as exc:
            live_state(run,'failed',progress,error=repr(exc));raise
