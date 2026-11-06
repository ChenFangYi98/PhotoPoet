# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""纯SFT自适应训练：任务配比、学习型prompt分类、有界动态权重、无目标丢失窗口。

默认从指定继续预训练best开始SFT；不自动覆盖旧模型、旧数据或旧断点。
--prepare_only只检查，--smoke_test最多3次更新；正式运行交给用户手动开始。
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
# 直接运行本文件即使用这些路径；随项目目录移动，不依赖当前终端目录。
# manifest 同时引用 clean_v1 原库、作者数据和 v3 新增确定性数据，勿改回原始 JSONL。
DEFAULT_MANIFEST = ROOT/'dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready/manifest.json'
DEFAULT_PRETRAIN = ROOT/'out/continue_pretrain/poetry_gujilab_20261022_190751_427063/pretrain_poetry_gujilab_768_moe_best.pth'
DEFAULT_OUTPUT = ROOT/'out/adaptive_sft'
sys.path.insert(0, str(ROOT))
import datasets
import torch
import torch.nn.functional as F
from model.model_mindchen import MindChenConfig
from model.checkpoint_config import write_checkpoint_config
from trainer.trainer_utils import init_model, setup_seed
from trainer.train_full_sft import amp_context, export_weights, lr_for_update
from trainer.sft_data import SFTCollator
from trainer.continue_pretrain_utils import (atomic_json, atomic_torch_save, file_info,
    sha256_file, make_logger, MetricsWriter, rng_state, restore_rng, rewind_metrics)
from trainer.adaptive_sft_data import (TASKS, IndexedCorpus, lossless_features, prepare_sft_row,
    prompt_batch)
from trainer.adaptive_sft_sampling import (CHECKPOINT_FORMAT, SAMPLING_VERSION,
    draw_examples, sampling_plan, require_checkpoint_format)
from trainer.adaptive_sft_objective import TaskRouter, WeightController, window_loss
from trainer.factual_sft_utils import correct_answer, entity_correct_answer
from trainer.adaptive_sft_safety import (quarantine_reason, training_signature,
    stratified_indices, generation_diagnostics, SAFETY_VERSION)

CHECKPOINT_FORMAT = 'adaptive_sft_probability_v1'


def require_checkpoint_format(checkpoint):
    if checkpoint.get('format_version') != CHECKPOINT_FORMAT:
        raise ValueError('旧运行不能恢复到概率门控版。请新建运行；切换前代码备份位于 '
                         'out/adaptive_sft/probability_gate_20261027/before。'
                         '只复用 LM 权重请显式使用 --init_weight_path，优化器和验证基线重新开始。')

FIELDS = ['timestamp','phase','update','loss','router_loss','router_macro_accuracy','router_mix',
    'fact_nll','open_nll','general_nll','fact_weight','open_weight','general_weight',
    'balanced_nll','fact_em','fact_count','author_em','verified_em','difficulty','sample_counts',
    'source_counts','source_samples','windows','windowed_samples','target_tokens','target_retention',
    'lr','grad_norm','gpu_peak_gib','seconds','draw_digest','accepted_for_best',
    'clarification_nll','clarification_count','quality_rejections','fact_macro_em',
    'entity_em','answer_format_em','open_diagnostics','task_em']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(DEFAULT_MANIFEST),help='已划分训练/验证并包含新增数据的 v3 清单')
    p.add_argument('--init_weight_path',default=str(DEFAULT_PRETRAIN),help='首次 SFT 的继续预训练 best 权重；不是旧 SFT 权重')
    p.add_argument('--output_dir',default=str(DEFAULT_OUTPUT),help='每次新建运行子目录，保留旧权重')
    p.add_argument('--run_name',default='')
    p.add_argument('--resume',default='')
    p.add_argument('--max_updates',type=int,default=0,help='0按epochs×可索引训练条数/每次样本数设置采样预算；正数用于短实验')
    p.add_argument('--epochs',type=int,default=1,help='全库样本数等量采样预算，不保证每条都访问')
    p.add_argument('--stop_after_updates',type=int,default=0,help='0不提前停止；正数在该绝对 update 保存退出，不改变学习率总预算')
    # 64 条原样本合成一次更新；LM 前向每批最多两个窗口，分类前向每批八条。
    p.add_argument('--examples_per_update',type=int,default=64)
    p.add_argument('--window_batch_size',type=int,default=2)
    p.add_argument('--router_batch_size',type=int,default=8,help='prompt分类分批反传，避免有效batch变大导致显存膨胀')
    p.add_argument('--task_ratios',type=float,nargs=3,default=None,metavar=('FACT','OPEN','GENERAL'),help='省略则从全库数量按n^0.35自动平滑配比')
    p.add_argument('--max_seq_len',type=int,default=2048)
    p.add_argument('--context_overlap',type=int,default=512)
    p.add_argument('--add_system_ratio',type=float,default=.2,help='训练普通对话按此概率添加 mindchen 身份 system；验证不添加')
    p.add_argument('--context_policy',choices=('full_context','windows'),default='full_context',
                   help='默认隔离超窗口整条对话并同通道补样；windows保留目标但可能丢失远端依据')
    p.add_argument('--verify_source_sha256',action='store_true',help='启动时完整重算原 JSONL 哈希，首次正式长训建议启用')
    p.add_argument('--norm_topk_prob',type=int,choices=(0,1),default=0,
                   help='默认0：保留选中专家的全局 softmax 概率；1仅供旧归一化对照')
    p.add_argument('--moe_router_gradient',choices=('legacy','straight_through'),default='legacy',
                   help='legacy使用所选前向公式的真实梯度；straight_through仅允许norm_topk_prob=1的实验对照')
    # 从预训练权重开始全参数 SFT；BF16 前向配 FP32 主参数与 AdamW 状态。
    p.add_argument('--learning_rate',type=float,default=1e-5)
    p.add_argument('--router_learning_rate',type=float,default=3e-4,help='新初始化的小分类头使用独立学习率')
    p.add_argument('--warmup_ratio',type=float,default=.03)
    p.add_argument('--min_lr_ratio',type=float,default=.1)
    p.add_argument('--weight_decay',type=float,default=.01)
    p.add_argument('--grad_clip',type=float,default=1.)
    p.add_argument('--weight_mode',choices=('uniform','static','dynamic'),default='dynamic')
    p.add_argument('--fact_boost',type=float,default=.6)
    p.add_argument('--max_sample_weight',type=float,default=1.8)
    p.add_argument('--router_loss_weight',type=float,default=.05)
    p.add_argument('--router_warmup_updates',type=int,default=100)
    p.add_argument('--router_min_accuracy',type=float,default=.75)
    p.add_argument('--validation_examples',type=int,default=128)
    p.add_argument('--clarification_validation_examples',type=int,default=16,
                   help='额外无重复的澄清专项 NLL 探针；不混入 general 均值或 best 排序，0关闭')
    p.add_argument('--generation_per_source',type=int,default=32,help='verified按任务分层，默认覆盖全部已知类型')
    p.add_argument('--generation_tokens',type=int,default=32)
    p.add_argument('--open_generation_examples',type=int,default=4,help='额外固定创作题，仅保存样本与重复诊断，不自动判定文学质量')
    p.add_argument('--open_generation_samples',type=int,default=2)
    p.add_argument('--open_generation_tokens',type=int,default=128)
    p.add_argument('--preservation_tolerance',type=float,default=.15)
    p.add_argument('--eval_interval',type=int,default=1000)
    p.add_argument('--save_interval',type=int,default=1000)
    p.add_argument('--log_interval',type=int,default=50)
    p.add_argument('--early_stopping_patience',type=int,default=0,help='0关闭；预训练初始化的早期能力尚未形成，避免过早停止')
    p.add_argument('--hidden_size',type=int,default=768)
    p.add_argument('--num_hidden_layers',type=int,default=8)
    p.add_argument('--use_moe',type=int,default=1,choices=(0,1))
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype',choices=('bfloat16','float16','float32'),default='bfloat16')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke_test',action='store_true')
    p.add_argument('--prepare_only',action='store_true')
    return p.parse_args(argv)


def materialize(corpus, draws, tokenizer, args, rejection_counts=None, rejection_examples=None):
    """风险样本隔离后同通道确定性补样，不把题干截掉或悄悄改变任务配额。"""
    examples = []
    seed = int(hashlib.sha256(json.dumps(draws).encode()).hexdigest()[:16],16) + args.seed
    rng = random.Random(seed)
    actual_draws, used = [], set()
    for position,(channel,index) in enumerate(draws):
        for attempt in range(1000):
            row, meta = corpus.read(channel,index)
            reason = quarantine_reason(row)
            feature = None
            if not reason:
                if channel.startswith('train_'):
                    # 每次更新及样本位置固定，断点重启时重现同一 system 提示。
                    item_seed=int.from_bytes(hashlib.sha256(f'{seed}:{position}:{channel}:{index}'.encode()).digest()[:8],'big')
                    row=prepare_sft_row(row,item_seed,getattr(args,'add_system_ratio',.2))
                feature = lossless_features(row,tokenizer,args.max_seq_len,args.context_overlap)
                if args.context_policy=='full_context' and feature['full_tokens']>args.max_seq_len:
                    reason = 'context_exceeds_window'
            if not reason and (channel.startswith('train_') or (channel,index) not in used):
                break
            if reason:
                if rejection_counts is not None: rejection_counts[reason]+=1
                if rejection_examples is not None and len(rejection_examples)<50:
                    rejection_examples.append(dict(channel=channel,index=index,reason=reason,
                        prompt=row['conversations'][0].get('content','')[:200]))
            index = rng.randrange(len(corpus.indices[channel]))
        else:
            raise ValueError('同通道无法补齐安全样本：'+channel)
        feature['meta'] = meta
        examples.append(feature)
        actual_draws.append((channel,index)); used.add((channel,index))
    # 下游统计/恢复哈希必须记录实际参与训练的行，而不是被替换的候选。
    draws[:] = actual_draws
    return examples


def router_forward(model, router, examples, tokenizer, args):
    ids, mask = prompt_batch(examples, tokenizer.pad_token_id, args.device)
    with amp_context(args):
        hidden, _, aux = model.model(ids, attention_mask=mask)
    return router(hidden, mask), aux


def answer_is_correct(text,gold,task_meta=None,aliases=None):
    # 新确定性任务含数组、分数、时间和原文；逗号/冒号属于答案，不能剥掉后评分。
    # 按题面“只输出指定格式”严格比较，保留现有旧探针的姓名/数字评分方式。
    if aliases or (task_meta or {}).get('kind')=='literature_author':
        return text.strip() in {a.strip() for a in (aliases or [gold])}
    return text.strip()==gold.strip() if task_meta else correct_answer(text,[gold])


@torch.inference_mode()
def creative_probes(model,tokenizer,args):
    prompts = ['写一首原创七言绝句，描写雨后山村，只输出诗。',
               '写一个有关失而复得的原创微型故事，结尾留有余味。',
               '用三种不同意象表达离别，避免直接写出离别二字。',
               '写一段古风散文，将秋日与新的希望联系起来。']
    records=[]
    devices=list(range(torch.cuda.device_count())) if args.device.startswith('cuda') else []
    # 验证采样不能改变后续训练的随机数轨迹；同一检查点可复现。
    with torch.random.fork_rng(devices=devices):
        for i,prompt in enumerate(prompts[:getattr(args,'open_generation_examples',0)]):
            for sample in range(args.open_generation_samples):
                torch.manual_seed(args.seed+82000+i*101+sample)
                rendered=tokenizer.apply_chat_template([dict(role='user',content=prompt)],
                    tokenize=False,add_generation_prompt=True,open_thinking=False)
                inputs=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).to(args.device)
                budget=min(args.open_generation_tokens,args.max_seq_len-inputs['input_ids'].shape[1])
                if budget<1: raise ValueError('创作探针超出上下文')
                with amp_context(args):
                    ids=model.generate(inputs=inputs['input_ids'],attention_mask=inputs['attention_mask'],
                        max_new_tokens=budget,do_sample=True,temperature=.85,top_p=.95,top_k=0,
                        eos_token_id=tokenizer.eos_token_id)
                text=tokenizer.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                records.append(dict(prompt=prompt,sample=sample,seed=args.seed+82000+i*101+sample,
                    prediction=text,max_tokens_reached=ids.shape[1]-inputs['input_ids'].shape[1]>=budget))
    per_prompt={p:generation_diagnostics([r['prediction'] for r in records if r['prompt']==p])
                for p in dict.fromkeys(r['prompt'] for r in records)}
    return records,per_prompt


def sample_nll(model, examples, tokenizer, args):
    totals = torch.zeros(len(examples),device=args.device)
    counts = torch.tensor([e['target_tokens'] for e in examples],device=args.device)
    flattened = [(owner,window) for owner,e in enumerate(examples) for window in e['windows']]
    flattened.sort(key=lambda item:len(item[1]['input_ids']))
    collator = SFTCollator(tokenizer.pad_token_id)
    for start in range(0,len(flattened),args.window_batch_size):
        current = flattened[start:start+args.window_batch_size]
        owners = torch.tensor([owner for owner,_ in current],device=args.device)
        batch = collator([window for _,window in current])
        labels = batch['labels'].to(args.device)
        with amp_context(args):
            out = model(input_ids=batch['input_ids'].to(args.device),attention_mask=batch['attention_mask'].to(args.device))
        _, nll, _ = window_loss(out.logits,labels,owners,counts,torch.ones(len(examples),device=args.device))
        totals.scatter_add_(0,owners,nll)
    return totals/counts


@torch.inference_mode()
def evaluate(model,router,examples,generation_examples,tokenizer,args,clarification_examples=()):
    old_model,old_router = model.training,router.training
    model.eval();router.eval()
    try:
        nll = sample_nll(model,examples,tokenizer,args)
        tasks = torch.tensor([e['meta']['task'] for e in examples],device=args.device)
        parts=[]
        for start in range(0,len(examples),getattr(args,'router_batch_size',8)):
            part,_=router_forward(model,router,examples[start:start+getattr(args,'router_batch_size',8)],tokenizer,args)
            parts.append(part)
        logits=torch.cat(parts)
        predicted = logits.argmax(-1)
        confidences = torch.tensor([e['meta']['label_confidence'] for e in examples],device=args.device)
        metrics,records,accuracies = {},[],[]
        for task,name in enumerate(TASKS):
            mask = tasks.eq(task)
            if not mask.any(): raise ValueError('固定开发集缺少任务类别')
            metrics[name+'_nll'] = float(nll[mask].mean())
            router_mask = mask & confidences.gt(0)
            accuracies.append(float(predicted[router_mask].eq(task).float().mean()) if router_mask.any() else 0.)
        metrics['balanced_nll'] = sum(metrics[name+'_nll'] for name in TASKS)/3
        metrics['clarification_count'] = len(clarification_examples)
        if clarification_examples:
            metrics['clarification_nll'] = float(sample_nll(model,clarification_examples,tokenizer,args).mean())
        metrics['router_macro_accuracy'] = sum(accuracies)/3
        metrics['router_accuracy_by_task'] = dict(zip(TASKS,accuracies))
        corrects = Counter();numbers = Counter()
        for channel,index,row in generation_examples:
            messages = row['conversations'][:-1]
            prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,open_thinking=False)
            inputs = tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(args.device)
            # 生成探针均为短确定性任务；不截长prompt以假装是原题。
            if inputs['input_ids'].shape[1]+args.generation_tokens > args.max_seq_len:
                raise ValueError('生成探针超出窗口，需调整固定探针')
            with amp_context(args):
                ids = model.generate(inputs=inputs['input_ids'],attention_mask=inputs['attention_mask'],
                    max_new_tokens=args.generation_tokens,do_sample=False,temperature=1.,top_p=1.,top_k=0,
                    repetition_penalty=1.,eos_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            gold = row['conversations'][-1]['content']
            aliases=row.get('_eval_aliases')
            correct = answer_is_correct(answer,gold,row.get('task_meta'),aliases)
            source = 'author' if '_author_' in channel else 'verified'
            corrects[source]+=int(correct); numbers[source]+=1
            kind=row.get('task_meta',{}).get('kind','author' if source=='author' else 'legacy_numeric')
            corrects['task:'+kind]+=int(correct); numbers['task:'+kind]+=1
            is_author=source=='author' or kind=='literature_author'
            entity=entity_correct_answer(answer,aliases or [gold],row.get('task_meta',{}).get('title')) if is_author else correct
            corrects['entity']+=int(entity);numbers['entity']+=1
            quality=row.get('_eval_quality',row.get('task_meta',{}).get('quality',source))
            if source=='author':corrects[quality]+=int(correct);numbers[quality]+=1
            records.append(dict(channel=channel,index=index,prompt=messages[-1]['content'],gold=gold,
                quality=quality,task=kind,prediction=answer,correct=correct,entity_correct=entity,
                max_tokens_reached=ids.shape[1]-inputs['input_ids'].shape[1]>=args.generation_tokens))
        for name in ('author','verified'): metrics[name+'_em'] = corrects[name]/max(1,numbers[name])
        metrics['author_external_em']=corrects['externally_reviewed']/max(1,numbers['externally_reviewed'])
        metrics['author_corpus_em']=corrects['corpus_metadata']/max(1,numbers['corpus_metadata'])
        metrics['generation_counts']=dict(numbers)
        metrics['fact_count'] = numbers['author']+numbers['verified']
        metrics['fact_em'] = (corrects['author']+corrects['verified'])/max(1,metrics['fact_count'])
        metrics['answer_format_em']=metrics['fact_em']
        metrics['entity_em']=corrects['entity']/max(1,numbers['entity'])
        metrics['task_em']={k[5:]:corrects[k]/n for k,n in numbers.items() if k.startswith('task:')}
        metrics['fact_macro_em']=sum(metrics['task_em'].values())/max(1,len(metrics['task_em']))
        open_records,metrics['open_diagnostics']=creative_probes(model,tokenizer,args)
        return metrics,dict(predictions=records,router_predictions=[dict(task=int(t),predicted=int(p),probabilities=prob)
            for t,p,prob in zip(tasks.cpu(),predicted.cpu(),logits.softmax(-1).cpu().tolist())],open_predictions=open_records)
    finally:
        model.train(old_model);router.train(old_router)


def main():
    args = parse_args(); session = datetime.now().strftime('%Y%m%d_%H%M%S_%f'); resume = None
    if args.resume:
        if args.prepare_only: raise ValueError('prepare_only仅用于新运行')
        resume = torch.load(args.resume,map_location='cpu',weights_only=True)
        require_checkpoint_format(resume)
        provided = {v[2:].split('=')[0] for v in sys.argv[1:] if v.startswith('--')}
        for k,v in resume['args'].items():
            if k in ('resume','stop_after_updates','prepare_only'): continue
            if k in provided and k in ('log_interval','eval_interval','save_interval'): continue
            if k in provided and getattr(args,k)!=v: raise ValueError('恢复不能改变 '+k)
            setattr(args,k,v)
        folder = Path(args.resume).resolve().parent
    else:
        if args.smoke_test:
            args.max_updates = min(args.max_updates or 3,3)
            args.examples_per_update = 6; args.validation_examples = 9; args.generation_per_source = 2
            args.eval_interval = 3; args.save_interval = 3; args.log_interval = 1
            args.open_generation_examples=1;args.open_generation_tokens=32
        name = args.run_name or ('smoke_' if args.smoke_test else 'adaptive_')+session
        if Path(name).name!=name or name in ('.','..'): raise ValueError('run_name必须为目录名')
        folder = Path(args.output_dir).resolve()/name; folder.mkdir(parents=True,exist_ok=False)
    if not 32<=args.max_seq_len<=32768 or not 1<=args.context_overlap<args.max_seq_len: raise ValueError('窗口/重叠设置错误')
    if args.examples_per_update<3 or args.validation_examples<3 or min(args.router_batch_size,args.window_batch_size,args.epochs,args.eval_interval,args.save_interval,args.log_interval)<1 or args.max_updates<0: raise ValueError('批量/步数必须正数')
    if (args.task_ratios is not None and any(v<=0 for v in args.task_ratios)) or min(args.learning_rate,args.router_learning_rate,args.grad_clip)<=0: raise ValueError('比例/学习率/梯度裁剪设置错误')
    if min(args.generation_per_source,args.generation_tokens)<1 or min(args.early_stopping_patience,args.router_warmup_updates,args.stop_after_updates,args.clarification_validation_examples)<0: raise ValueError('评测/停止参数错误')
    if not 0<=args.router_min_accuracy<=1 or args.fact_boost<0 or args.max_sample_weight<1 or args.router_loss_weight<0: raise ValueError('权重设置错误')
    if not 0<=args.warmup_ratio<1 or not 0<=args.min_lr_ratio<=1 or args.weight_decay<0: raise ValueError('优化器设置错误')
    if not math.isfinite(args.add_system_ratio) or not 0<=args.add_system_ratio<=1: raise ValueError('add_system_ratio 必须在 [0,1] 内')
    if not 0<=args.open_generation_examples<=4 or min(args.open_generation_samples,args.open_generation_tokens)<1: raise ValueError('创作验证参数错误')
    if not args.norm_topk_prob and args.moe_router_gradient!='legacy':
        raise ValueError('保留原概率已具有真实梯度，无需替代梯度；straight_through要求 --norm_topk_prob 1')
    setup_seed(args.seed)
    logger = make_logger(folder,'adaptive_sft');writer = None; corpus = None; old_signal = None
    started = time.perf_counter(); stop_requested = [False]
    progress = dict(update=0,best_rank=None,best_update=None,initial_open_nll=None,initial_general_nll=None,stale_evals=0)
    try:
        if args.device.startswith('cuda'):
            torch.cuda.set_device(args.device)
            if args.dtype=='bfloat16' and not torch.cuda.is_bf16_supported(): raise ValueError('GPU不支持BF16')
        corpus = IndexedCorpus(args.manifest)
        if args.verify_source_sha256:
            for item in corpus.sources:
                logger.info('核验源数据内容哈希：%s',item['path'])
                if sha256_file(item['path'])!=item['sha256']: raise ValueError('源数据 SHA256 不一致：'+item['path'])
        if args.task_ratios is None:args.task_ratios=corpus.manifest['recommended_task_ratios']
        if not args.max_updates:
            args.max_updates=math.ceil(sum(corpus.manifest['indexed_counts']['train'].values())*args.epochs/args.examples_per_update)
        if not corpus.manifest['complete_scan'] and not args.smoke_test: raise ValueError('正式运行需要完整扫描索引')
        source = file_info(args.init_weight_path,checksum=True)
        config = MindChenConfig(hidden_size=args.hidden_size,num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),norm_topk_prob=bool(args.norm_topk_prob),
            moe_mask_padding=True,moe_router_gradient=args.moe_router_gradient)
        if resume and resume['model_config']!=config.to_dict():
            raise ValueError('完整断点模型配置与恢复参数不一致')
        model,tokenizer = init_model(config,'none',device=args.device)
        model.load_state_dict(resume['model'] if resume else torch.load(args.init_weight_path,map_location='cpu',weights_only=True),strict=True)
        router = TaskRouter(config.hidden_size).to(args.device)
        if resume: router.load_state_dict(resume['router'])
        controller = WeightController(args.fact_boost,args.max_sample_weight,warmup=args.router_warmup_updates,minimum_accuracy=args.router_min_accuracy)
        signature = training_signature(ROOT,tokenizer,config,args.manifest)
        if resume and (signature!=resume['signature'] or source!=resume['source']): raise ValueError('数据/代码/初始化改变，拒绝恢复')
        # 已到绝对停止上限时直接返回，不更新、不覆盖旧 summary 或 metrics。
        if resume and (resume['progress']['update']>=args.max_updates or
                       (args.stop_after_updates and resume['progress']['update']>=args.stop_after_updates)):
            logger.info('断点 update=%d 已达到停止上限，未执行任何更新；原概要保留',resume['progress']['update'])
            return
        rejection_counts=Counter();rejection_examples=[]
        dev_draws = draw_examples(corpus,args.validation_examples,[1,1,1],args.seed+9101,'valid')
        dev = materialize(corpus,dev_draws,tokenizer,args,rejection_counts,rejection_examples)
        physical_rows = [(int(corpus.indices[key][index]['source']),int(corpus.indices[key][index]['offset']))
                         for key,index in dev_draws]
        if len(set(physical_rows))!=len(dev_draws): raise ValueError('开发索引包含重复原行，请重新准备索引')
        # 少量澄清题单独监控，既不放大其训练频率，也不让固定答法拉低通用评估。
        clarification_key = 'valid_author_general'
        clarification_size = len(corpus.indices.get(clarification_key, []))
        clarification_draws = [(clarification_key,i) for i in random.Random(args.seed+9104).sample(
            range(clarification_size),min(clarification_size,args.clarification_validation_examples))]
        clarification_dev = materialize(corpus,clarification_draws,tokenizer,args,rejection_counts,rejection_examples)
        generation = []
        for channel in ('valid_author_fact','valid_verified_fact'):
            size = len(corpus.indices[channel])
            candidates=list(range(size));quality_by_offset={};meta_by_offset={}
            if channel=='valid_author_fact':
                # 生成评测只用原关系留出题；不能把带答案资料的grounded视图当闭卷题。
                original_sources={i for i,s in enumerate(corpus.sources) if Path(s['path']).name=='facts_valid.jsonl'}
                candidates=[i for i in candidates if int(corpus.indices[channel][i]['source']) in original_sources]
                source_path=Path(corpus.sources[next(iter(original_sources))]['path'])
                if sha256_file(source_path.parent/'manifest.json')!=corpus.manifest['factual_manifest_sha256']:
                    raise ValueError('作者数据来源清单改变，需要重建索引')
                factual_manifest=json.loads((source_path.parent/'manifest.json').read_text(encoding='utf-8'))
                if sha256_file(source_path.with_name('facts_valid.meta.jsonl'))!=factual_manifest['files']['facts_valid.meta.jsonl']['sha256']:
                    raise ValueError('作者评测元数据改变')
                offset=0
                with source_path.open('rb') as raw, source_path.with_name('facts_valid.meta.jsonl').open(encoding='utf-8') as meta_stream:
                    for line,meta_line in zip(raw,meta_stream):
                        metadata=json.loads(meta_line)
                        meta_by_offset[offset]=metadata
                        quality_by_offset[offset]=metadata['quality'];offset+=len(line)
                reviewed=[i for i in candidates if quality_by_offset[int(corpus.indices[channel][i]['offset'])]=='externally_reviewed']
                fixed=reviewed[:min(args.generation_per_source,len(reviewed))]
                other=[i for i in candidates if i not in set(fixed)]
                indices=fixed+random.Random(args.seed+9102).sample(other,min(len(other),args.generation_per_source-len(fixed)))
            else:indices=stratified_indices(corpus,channel,candidates,min(size,args.generation_per_source),random.Random(args.seed+9102))
            for i in indices:
                row=corpus.read(channel,i)[0]
                if quality_by_offset:
                    offset=int(corpus.indices[channel][i]['offset'])
                    row['_eval_quality']=quality_by_offset[offset]
                    row['_eval_aliases']=meta_by_offset[offset].get('aliases',[row['conversations'][-1]['content']])
                generation.append((channel,i,row))
        audit_draws = draw_examples(corpus,max(24,args.examples_per_update),args.task_ratios,args.seed+9103)
        audit = materialize(corpus,audit_draws,tokenizer,args,rejection_counts,rejection_examples)
        atomic_json(folder/('run_config.json' if not resume else f'resume_config_{session}.json'),dict(args=vars(args),signature=signature,
            source=source,counts=corpus.manifest['indexed_counts'],dev_draws=dev_draws,
            clarification_draws=clarification_draws,
            safety=dict(version=SAFETY_VERSION,policy=args.context_policy,
                        rejections=dict(rejection_counts),examples=rejection_examples,
                        generation_tasks=dict(Counter(r.get('task_meta',{}).get('kind',
                            'author' if '_author_' in k else 'legacy_numeric') for k,_,r in generation))),
            sampling=dict(version=SAMPLING_VERSION,
                train_plan=sampling_plan(corpus,args.examples_per_update,args.task_ratios),
                validation_plan=sampling_plan(corpus,args.validation_examples,[1,1,1],'valid'),
                validation_without_replacement=True),
            audit=dict(samples=len(audit),windowed=sum(e['windowed'] for e in audit),
                       windows=sum(len(e['windows']) for e in audit),target_tokens=sum(e['target_tokens'] for e in audit),target_retention=1.),
            task_labels='Weak prompt-task labels; not a truth detector. Router is an auxiliary SFT head.',
            identity_system=dict(training_probability=args.add_system_ratio,validation_probability=0,
                scope='Only adds a system prompt; original assistant answers and source JSONL remain unchanged.'),
            window_scope='full_context rejects over-window rows without truncation; windows explicitly permits remote context loss.'))
        atomic_json(folder/'model_config.json',config.to_dict())
        logger.info('初始化=%s | 完整分类索引=%s | 每次更新原样本=%d 窗口=%d 重叠=%d | mode=%s 计划更新=%d',
            args.init_weight_path,corpus.manifest['indexed_counts'],args.examples_per_update,args.max_seq_len,args.context_overlap,args.weight_mode,args.max_updates)
        logger.info('数据=%s | 输出=%s | device=%s dtype=%s | 任务比例=%s',
            args.manifest,folder,args.device,args.dtype,args.task_ratios)
        logger.info('训练普通对话身份 system 概率=%.2f；验证不添加，已有 system/工具对话保留',args.add_system_ratio)
        logger.info('上下文策略=%s 初始化隔离=%s | MoE梯度=%s 有效token负载统计开启',
            args.context_policy,dict(rejection_counts),args.moe_router_gradient)
        logger.info('MoE norm_topk_prob=%s | 选中概率%s保留 | 模型配置随每份权重导出',
            config.norm_topk_prob,'不' if config.norm_topk_prob else '')
        if config.use_moe and not config.norm_topk_prob:
            logger.info('本次改变旧预训练模型的专家输出尺度；初始验证建立新基线，损失不能与旧运行直接比较')
        logger.info('采样=%s | open/general按条数抽样 | 验证原行去重=%d/%d | 澄清专项=%d（单独统计）',
            SAMPLING_VERSION,len(set(physical_rows)),len(dev_draws),len(clarification_draws))
        logger.info('主干峰值lr=%.2g 分类头峰值lr=%.2g warmup=%d updates | 验证/保存间隔=%d/%d',
            args.learning_rate,args.router_learning_rate,int(args.max_updates*args.warmup_ratio),args.eval_interval,args.save_interval)
        if args.prepare_only:
            atomic_json(folder/'summary.json',dict(status='prepared',optimizer_updates=0));return
        parameters = list(model.parameters())+list(router.parameters())
        optimizer = torch.optim.AdamW([dict(params=model.parameters(),lr=args.learning_rate),
            dict(params=router.parameters(),lr=args.router_learning_rate)],weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler('cuda',enabled=args.device.startswith('cuda') and args.dtype=='float16')
        if resume:
            optimizer.load_state_dict(resume['optimizer']); scaler.load_state_dict(resume['scaler'])
            progress = resume['progress']; controller.load_state_dict(resume['controller']); restore_rng(resume['rng'])
            rewind_metrics(folder,progress['update'],session,FIELDS)
            del resume
        writer = MetricsWriter(folder,FIELDS)
        old_signal = signal.signal(signal.SIGINT,lambda *_:stop_requested.__setitem__(0,True))
        if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        prefix = f'adaptive_{args.hidden_size}'+('_moe' if args.use_moe else '')
        def record(phase,**values):
            return dict(timestamp=datetime.now().isoformat(timespec='seconds'),phase=phase,update=progress['update'],
                gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30 if args.device.startswith('cuda') else 0.,seconds=time.perf_counter()-started,**values)
        def export_pair(best=False):
            suffix = '_best' if best else ''
            export_weights(model,folder/(prefix+suffix+'.pth'))
            exported=write_checkpoint_config(folder/(prefix+suffix+'.pth'),config)
            atomic_torch_save(folder/('task_router'+suffix+'.pth'),dict(state_dict=router.state_dict(),
                hidden_size=args.hidden_size,num_hidden_layers=args.num_hidden_layers,use_moe=bool(args.use_moe),
                update=progress['update'],tasks=TASKS,weight_file=prefix+suffix+'.pth',
                weight_sha256=exported['weight_sha256'],signature=signature))
        final_metrics = None
        def validate():
            metrics,predictions = evaluate(model,router,dev,generation,tokenizer,args,clarification_dev)
            controller.router_macro_accuracy = metrics['router_macro_accuracy']
            if progress['initial_open_nll'] is None:
                progress['initial_open_nll']=metrics['open_nll']; progress['initial_general_nll']=metrics['general_nll']
            references=progress.setdefault('preservation_reference',dict(
                open=progress['initial_open_nll'],general=progress['initial_general_nll']))
            accepted_guard = all(metrics[k+'_nll'] <= references[k]+args.preservation_tolerance for k in ('open','general'))
            for k in references: references[k]=min(references[k],metrics[k+'_nll'])
            rank = (metrics['fact_macro_em'],-metrics['balanced_nll'])
            accepted = accepted_guard and (progress['best_rank'] is None or rank>tuple(progress['best_rank']))
            improved = accepted and (progress['best_rank'] is None or rank[0]>progress['best_rank'][0]+1e-8)
            progress['stale_evals'] = 0 if improved else progress['stale_evals']+1
            if accepted:
                progress.update(best_rank=list(rank),best_update=progress['update']); export_pair(True)
            writer.write(record('validation',accepted_for_best=accepted,**metrics))
            atomic_json(folder/f'validation_{progress["update"]:06d}.json',dict(metrics=metrics,**predictions,accepted_for_best=accepted))
            logger.info('验证@%d factEM=%d/%d router宏准确率=%.3f | NLL fact=%.4f open=%.4f general=%.4f best@%s',
                progress['update'],round(metrics['fact_em']*metrics['fact_count']),metrics['fact_count'],metrics['router_macro_accuracy'],
                metrics['fact_nll'],metrics['open_nll'],metrics['general_nll'],progress['best_update'])
            if metrics['clarification_count']:
                logger.info('澄清专项 NLL=%.4f 样本=%d；不计入 general 均值或 best 排序',
                    metrics['clarification_nll'],metrics['clarification_count'])
            return metrics
        saved = -1
        def save():
            nonlocal saved
            if saved==progress['update']: return
            logger.info('开始写入权重与完整断点 update=%d；包含 AdamW 状态，磁盘写入可能较慢',progress['update'])
            export_pair()
            atomic_torch_save(folder/'latest_resume.pth',dict(format_version=CHECKPOINT_FORMAT,args=vars(args),
                model_config=config.to_dict(),
                model={k:v.detach().cpu() for k,v in model.state_dict().items()},router=router.state_dict(),
                optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),controller=controller.state_dict(),
                progress=progress.copy(),signature=signature,source=source))
            saved=progress['update']; logger.info('保存权重、分类头与完整断点 update=%d',saved)
        if progress['update']==0: final_metrics=validate()
        model.train();router.train()
        # Time only updates completed by this process, including validation and saves.
        eta_started = time.perf_counter()
        eta_start_update = progress['update']
        eta_target = min(args.max_updates, args.stop_after_updates) if args.stop_after_updates else args.max_updates
        for update in range(progress['update']+1,args.max_updates+1):
            draws = draw_examples(corpus,args.examples_per_update,args.task_ratios,args.seed+update*1000003)
            update_rejections=Counter();update_rejection_examples=[]
            examples = materialize(corpus,draws,tokenizer,args,update_rejections,update_rejection_examples)
            if update_rejections:
                with (folder/'quarantine.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(update=update,counts=dict(update_rejections),
                        examples=update_rejection_examples),ensure_ascii=False)+'\n')
            tasks = torch.tensor([e['meta']['task'] for e in examples],device=args.device)
            confidences = torch.tensor([e['meta']['label_confidence'] for e in examples],device=args.device)
            reliability = torch.tensor([e['meta']['reliability'] for e in examples],device=args.device)
            target_counts = torch.tensor([e['target_tokens'] for e in examples],device=args.device)
            optimizer.zero_grad(set_to_none=True)
            # 按整个更新的标签质量总量归一，再分批反传；与一次性分类计算等价。
            router_parts=[];router_loss=0.;router_aux_total=0.
            denominator=confidences.sum().clamp_min(1e-8)
            for start in range(0,len(examples),args.router_batch_size):
                end=min(len(examples),start+args.router_batch_size)
                partial,prompt_aux=router_forward(model,router,examples[start:end],tokenizer,args)
                ce=(F.cross_entropy(partial,tasks[start:end],reduction='none')*confidences[start:end]).sum()/denominator
                aux_part=prompt_aux*confidences[start:end].sum()/denominator
                if args.router_loss_weight:scaler.scale(args.router_loss_weight*(ce+aux_part)).backward()
                router_parts.append(partial.detach());router_loss+=float(ce.detach());router_aux_total+=float(aux_part.detach())
            router_logits=torch.cat(router_parts)
            if args.weight_mode=='dynamic': weights,details = controller.weights(router_logits,tasks,reliability,confidences,update)
            elif args.weight_mode=='static':
                weights = (1+args.fact_boost*reliability*confidences*tasks.eq(0)).clamp(1,args.max_sample_weight).detach()
                details=dict(router_mix=0.,difficulty=1.)
            else: weights=torch.ones(len(examples),device=args.device);details=dict(router_mix=0.,difficulty=1.)
            flattened = [(owner,window) for owner,e in enumerate(examples) for window in e['windows']]
            flattened.sort(key=lambda item:len(item[1]['input_ids']))
            collator = SFTCollator(tokenizer.pad_token_id)
            sample_sums = torch.zeros(len(examples),device=args.device)
            lm_total,aux_total = 0.,0.
            for start in range(0,len(flattened),args.window_batch_size):
                current = flattened[start:start+args.window_batch_size]
                owners = torch.tensor([owner for owner,_ in current],device=args.device)
                batch = collator([window for _,window in current])
                labels = batch['labels'].to(args.device)
                with amp_context(args):
                    out = model(input_ids=batch['input_ids'].to(args.device),attention_mask=batch['attention_mask'].to(args.device))
                    lm,nll,window_counts = window_loss(out.logits,labels,owners,target_counts,weights)
                    mass = (weights[owners]*window_counts/target_counts[owners]).sum()/weights.sum()
                    aux = out.aux_loss*mass
                    loss = lm+aux
                if not torch.isfinite(loss): raise FloatingPointError('非有限SFT损失')
                scaler.scale(loss).backward()
                sample_sums.scatter_add_(0,owners,nll)
                lm_total += float(lm.detach()); aux_total += float(aux.detach())
                del out,loss,lm,aux
            scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(parameters,args.grad_clip,error_if_nonfinite=True))
            lr = lr_for_update(update,args.max_updates,args.learning_rate,args.warmup_ratio,args.min_lr_ratio)
            optimizer.param_groups[0]['lr']=lr
            optimizer.param_groups[1]['lr']=lr*args.router_learning_rate/args.learning_rate
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            sample_losses=sample_sums/target_counts; controller.observe(sample_losses,tasks)
            progress['update']=update
            values={name+'_nll':float(sample_losses[tasks.eq(i)].mean()) for i,name in enumerate(TASKS)}
            values.update({name+'_weight':float(weights[tasks.eq(i)].mean()) for i,name in enumerate(TASKS)})
            writer.write(record('train',loss=lm_total+aux_total+args.router_loss_weight*(router_loss+router_aux_total),
                router_loss=router_loss,router_macro_accuracy=controller.router_macro_accuracy,
                **{k:details[k] for k in ('router_mix','difficulty')},**values,
                sample_counts=dict(Counter(TASKS[int(t)] for t in tasks)),source_counts=dict(Counter(k for k,_ in draws)),
                quality_rejections=dict(update_rejections),
                source_samples=len(examples),windows=len(flattened),windowed_samples=sum(e['windowed'] for e in examples),
                target_tokens=int(target_counts.sum()),target_retention=1.,lr=lr,grad_norm=norm,
                draw_digest=hashlib.sha256(json.dumps(draws).encode()).hexdigest()))
            if update==1 or update%args.log_interval==0:
                elapsed = time.perf_counter() - eta_started
                remaining_seconds = elapsed / (update - eta_start_update) * max(0, eta_target - update)
                logger.info('update=%d/%d LM=%.4f | weight fact=%.3f open=%.3f general=%.3f routerMix=%.2f | 原样本=%d 窗口=%d 保留=100%% grad=%.3f lr=%.3g GPU峰值=%.2fGiB | ETA=%.1fh (target update=%d)',
                    update,args.max_updates,lm_total,values['fact_weight'],values['open_weight'],values['general_weight'],
                    details['router_mix'],len(examples),len(flattened),norm,lr,
                    torch.cuda.max_memory_allocated()/2**30 if args.device.startswith('cuda') else 0.,
                    remaining_seconds / 3600, eta_target)
            stop = stop_requested[0] or (args.stop_after_updates and update>=args.stop_after_updates)
            if stop or update==args.max_updates or update%args.eval_interval==0:
                final_metrics=validate()
                if args.early_stopping_patience and progress['stale_evals']>=args.early_stopping_patience: stop=True
            if stop or update==args.max_updates or update%args.save_interval==0: save()
            if stop: break
        save()
        atomic_json(folder/'summary.json',dict(status='completed' if progress['update']>=args.max_updates else 'stopped',
            **progress,final_metrics=final_metrics,source_sha256=source['sha256'],signature=signature,
            note='Development metrics and repetition diagnostics do not establish factual truth or creative quality; independent final testing required.'))
    except BaseException as exc:
        logger.exception('自适应SFT失败；原始数据/权重未覆盖')
        atomic_json(folder/'failure.json',dict(error=str(exc),progress=progress));raise
    finally:
        if old_signal is not None: signal.signal(signal.SIGINT,old_signal)
        if writer: writer.close()
        if corpus: corpus.close()


if __name__=='__main__': main()
