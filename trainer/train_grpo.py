# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""混合 RLAIF 与古诗的 GRPO/CISPO；SFT 来源只读，所有产物写入独立运行。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets  # noqa: F401  Windows pyarrow/torch DLL 初始化顺序
import argparse
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model_mindchen import MindChenForCausalLM
from model.checkpoint_config import load_checkpoint_config, weight_sha256
from trainer.grpo_resources import (DEFAULT_SOURCE, DEFAULT_REWARD, atomic_json, atomic_save,
    ensure_reward_model, file_digest, project_path, protect_source, snapshot_source, source_weight,
    write_checkpoint_config)
from trainer.grpo_poetry_data import DEFAULT_DATA, MixedPromptDataset, collate_prompts, ensure_poetry_data
from trainer.grpo_poetry_reward import PoetryRewardModel, calculate_rewards, rep_penalty
from trainer.grpo_control import StopController, StopRequested, live_state
from trainer.grpo_repetition import (repeated_token_mask, unlikelihood_loss,
    preserves_reference, aggregate_repetition)
from trainer.grpo_replay import SupervisedReplay

FORMAT = 'mindchen_poetry_grpo_v5_replay'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(DEFAULT_SOURCE), help='SFT 目录或原生 .pth，仅复制读取')
    p.add_argument('--save_dir', default=str(ROOT/'out/grpo'), help='独立输出根目录，每次新建运行子目录')
    p.add_argument('--run_name', default='', help='新运行名称，已存在则拒绝覆盖')
    p.add_argument('--resume', default='', help='本脚本生成的 resume.pth，不接受 SFT 完整断点')
    p.add_argument('--save_weight', default='grpo_poetry')
    p.add_argument('--data_path', default=str(ROOT/'dataset/rlaif.jsonl'), help='原通用 RL 数据，只读')
    p.add_argument('--poetry_data_path', default=str(DEFAULT_DATA), help='新增古诗提示数据，不存在时自动生成')
    p.add_argument('--poetry_ratio', type=float, default=0.5, help='古诗采样比例，其余为原 RLAIF')
    p.add_argument('--validation_fraction', type=float, default=0.05, help='原 RLAIF 按实际 prompt 哈希分组留出比例')
    p.add_argument('--eval_interval', type=int, default=20)
    p.add_argument('--validation_per_channel', type=int, default=16)
    p.add_argument('--eval_max_gen_len', type=int, default=256)
    p.add_argument('--general_regression_tolerance', type=float, default=0.15, help='选 best 时普通问答有界评分允许下降量')
    p.add_argument('--quality_regression_tolerance', type=float, default=.1)
    p.add_argument('--poetry_reward_regression_tolerance', type=float, default=.15)
    p.add_argument('--semantic_regression_tolerance', type=float, default=.05)
    p.add_argument('--completion_regression_tolerance', type=float, default=.05)
    p.add_argument('--reward_model_path', default=str(DEFAULT_REWARD))
    p.add_argument('--reward_device', default='cpu', help='默认 CPU，减少与 SFT 争用显存')
    p.add_argument('--ref_device', default='cpu')
    p.add_argument('--reward_max_length', type=int, default=4096)
    p.add_argument('--reward_model_weight', type=float, default=1.)
    p.add_argument('--repetition_weight', type=float, default=3., help='两类回答统一的过度重复惩罚')
    p.add_argument('--unlikelihood_weight', type=float, default=.2, help='降低重复 token 概率，即使组内奖励相同也能学习')
    p.add_argument('--replay_data_path', default='', help='仅训练集的格式/通用监督回放 JSONL')
    p.add_argument('--replay_weight', type=float, default=.1)
    p.add_argument('--repetition_regression_tolerance', type=float, default=.05)
    p.add_argument('--poetry_judge_model', default='qwen3.5:latest', help='本地 Ollama 古诗语义评审，启动时校准并固定模型摘要')
    p.add_argument('--poetry_judge_url', default='http://localhost:11434')
    p.add_argument('--poetry_judge_timeout', type=int, default=120)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--cpu_threads', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_generations', type=int, default=4)
    p.add_argument('--policy_micro_batch_size', type=int, default=1)
    p.add_argument('--accumulation_steps', type=int, default=4)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max_steps', type=int, default=100, help='optimizer 更新上限，0=完整 epochs')
    p.add_argument('--stop_after_steps', type=int, default=0, help='本次额外执行的更新上限，0=不限制')
    p.add_argument('--max_seq_len', type=int, default=768)
    p.add_argument('--max_gen_len', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.8, help='采样与 logprob 使用同一温度，不作 top-k/p 截断')
    p.add_argument('--learning_rate', type=float, default=3e-7)
    p.add_argument('--weight_decay', type=float, default=0.)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--loss_type', choices=['grpo', 'cispo'], default='grpo')
    p.add_argument('--epsilon', type=float, default=0.2)
    p.add_argument('--epsilon_high', type=float, default=5.)
    p.add_argument('--grad_clip', type=float, default=1.)
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--log_interval', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0, help='Windows 默认单进程读取')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--debug_mode', action='store_true')
    p.add_argument('--prepare_only', action='store_true', help='下载评分模型、生成数据、复制快照，不训练')
    p.add_argument('--eval_only', action='store_true', help='只运行固定验证，不作 optimizer 更新')
    p.add_argument('--smoke_test', action='store_true', help='默认 CPU；混合两类提示，完成一次真实更新后停止')
    args = p.parse_args(argv)
    supplied = {s.split('=')[0] for s in (sys.argv[1:] if argv is None else argv) if s.startswith('--')}
    if args.smoke_test and not args.resume:
        for name, value in dict(device='cpu', batch_size=2, num_generations=2, max_gen_len=64,
                                accumulation_steps=1, max_steps=2, stop_after_steps=1, save_interval=1,
                                validation_per_channel=1, eval_max_gen_len=64, eval_interval=1).items():
            if '--'+name not in supplied:
                setattr(args, name, value)
    positive = ['batch_size', 'num_generations', 'policy_micro_batch_size', 'accumulation_steps', 'epochs',
                'max_seq_len', 'max_gen_len', 'save_interval', 'log_interval', 'cpu_threads', 'reward_max_length',
                'eval_interval', 'validation_per_channel', 'eval_max_gen_len', 'poetry_judge_timeout']
    if any(getattr(args, n) < 1 for n in positive) or args.num_generations < 2:
        p.error('批量、长度、间隔必须为正数，num_generations 至少为 2')
    if (min(args.max_steps, args.stop_after_steps, args.num_workers) < 0 or not 0 < args.poetry_ratio < 1
        or not 0 < args.validation_fraction < 0.5 or not 0 <= args.general_regression_tolerance <= 2):
        p.error('步数不能为负数；poetry_ratio 必须在 0 和 1 之间')
    if (not all(math.isfinite(getattr(args, n)) for n in ['temperature', 'learning_rate', 'weight_decay', 'beta',
            'epsilon', 'epsilon_high', 'grad_clip', 'reward_model_weight', 'repetition_weight',
            'unlikelihood_weight', 'repetition_regression_tolerance', 'replay_weight']) or
        min(args.temperature, args.learning_rate, args.grad_clip, args.epsilon, args.epsilon_high) <= 0 or
        min(args.beta, args.weight_decay, args.reward_model_weight, args.repetition_weight,
            args.unlikelihood_weight, args.repetition_regression_tolerance, args.replay_weight) < 0 or
        args.repetition_regression_tolerance > 1):
        p.error('学习率/温度/裁剪需为有限正数，KL/衰减/评分权重需非负')
    for name in ['save_weight', 'run_name']:
        value = getattr(args, name)
        if value and (Path(value).name != value or value in ('.', '..') or any(c in value for c in '/\\:')):
            p.error(name+' 必须是单个文件/目录名前缀')
    if not args.save_weight:
        p.error('save_weight 不能为空')
    for name in ('quality_regression_tolerance', 'poetry_reward_regression_tolerance',
                 'semantic_regression_tolerance', 'completion_regression_tolerance'):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            p.error(name+' 必须是 0..1 的有限值')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('此入口使用单进程训练，请勿以 torchrun 多进程启动')
    return args, supplied


def amp_context(device, dtype):
    if str(device).startswith('cuda') and dtype != 'float32':
        return torch.autocast('cuda', dtype=getattr(torch, dtype))
    return nullcontext()


def completion_mask(completions, eos_id):
    """计到首个 EOS（含）；后续重复 EOS/padding 不参与 loss。"""
    eos = completions.eq(eos_id)
    return (eos.cumsum(-1)-eos.long()).eq(0)


def completion_ended(completions, eos_id):
    """An EOS before padding still means the candidate finished normally."""
    return completions.eq(eos_id).any(dim=1)


def token_logps(model, ids, mask, n_keep, temperature=1., dtype='float32'):
    with amp_context(ids.device, dtype):
        logits = model(ids, attention_mask=mask, logits_to_keep=n_keep+1, use_cache=False).logits[:, :-1]
    return F.log_softmax(logits.float()/temperature, dim=-1).gather(-1, ids[:, -n_keep:].unsqueeze(-1)).squeeze(-1)


def sample_group(model, tokenizer, rows, args):
    prompt = tokenizer([r['prompt'] for r in rows], return_tensors='pt', padding=True,
                       add_special_tokens=False, return_token_type_ids=False).to(args.device)
    prompt_ids = prompt.input_ids.repeat_interleave(args.num_generations, 0)
    prompt_mask = prompt.attention_mask.repeat_interleave(args.num_generations, 0)
    # eval 不禁用梯度；固定 dropout 并停用 MoE 辅助项，让新旧策略前向一致。
    model.eval()
    with torch.no_grad(), amp_context(args.device, args.dtype):
        ids = model.generate(input_ids=prompt_ids, attention_mask=prompt_mask, do_sample=True,
            max_new_tokens=args.max_gen_len, temperature=args.temperature, top_k=0, top_p=1.,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id).clone()
    completions = ids[:, prompt_ids.size(1):]
    cmask = completion_mask(completions, tokenizer.eos_token_id)
    full_mask = torch.cat([prompt_mask, cmask.long()], dim=1)
    return ids, full_mask, cmask, tokenizer.batch_decode(completions, skip_special_tokens=True)


def group_advantages(rewards, num_generations):
    groups = rewards.reshape(-1, num_generations)
    return ((groups-groups.mean(1, keepdim=True))/(groups.std(1, unbiased=False, keepdim=True)+1e-4)).flatten()


def policy_objective(current, old, reference, advantages, mask, args):
    delta = (reference-current).clamp(-20, 20)
    kl = delta.exp()-delta-1
    ratio = (current-old).clamp(-20, 20).exp()
    if args.loss_type == 'cispo':
        pg = -ratio.clamp(max=args.epsilon_high).detach()*advantages[:, None]*current
    else:
        clipped = ratio.clamp(1-args.epsilon, 1+args.epsilon)
        pg = -torch.minimum(ratio*advantages[:, None], clipped*advantages[:, None])
    loss = (((pg+args.beta*kl)*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()
    return loss, ((kl*mask).sum()/mask.sum().clamp_min(1)).detach()


def frozen_logps(model, ids, full_mask, n_keep, device, args):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(ids), args.policy_micro_batch_size):
            sl = slice(start, start+args.policy_micro_batch_size)
            result = token_logps(model, ids[sl].to(device), full_mask[sl].to(device), n_keep, args.temperature, args.dtype)
            chunks.append(result.to(ids.device))
    return torch.cat(chunks)


def code_signature():
    names = ['trainer/train_grpo.py', 'trainer/grpo_resources.py', 'trainer/grpo_poetry_data.py',
             'trainer/grpo_poetry_reward.py', 'model/model_mindchen.py', 'model/checkpoint_config.py',
             'model/tokenizer.json', 'model/tokenizer_config.json', 'trainer/grpo_control.py',
             'trainer/grpo_data_quality.py', 'trainer/grpo_poetry_judge.py', 'trainer/grpo_repetition.py',
             'trainer/grpo_io.py', 'trainer/grpo_replay.py', 'trainer/sft_data.py']
    return {name: file_digest(ROOT/name) for name in names}


def save_training(run_dir, model, optimizer, scheduler, scaler, args, progress, source, signature, tokenizer,
                  status='running'):
    suffix = '_moe' if model.config.use_moe else ''
    weight = run_dir/f'{args.save_weight}_{model.config.hidden_size}{suffix}.pth'
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f'[save] writing export and optimizer checkpoint at update={progress["update"]}', flush=True)
    # 极小 RL 学习率下，提前转 fp16 会抹掉很多已学更新；导出也保留训练参数精度。
    atomic_save(weight, state)
    write_checkpoint_config(weight, model.config)
    rng = dict(torch=torch.get_rng_state(), python=random.getstate(),
               cuda=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else [])
    atomic_save(run_dir/'resume.pth', dict(format=FORMAT, args=vars(args), progress=progress,
        source=source, signature=signature, model=state, optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng))
    atomic_json(run_dir/'summary.json', dict(status=status,
        **progress, source=source, export=str(weight), source_directory_written=False))
    tokenizer.save_pretrained(run_dir/'tokenizer')
    print(f'[save] update={progress["update"]} -> {weight}', flush=True)
    live_state(run_dir, status, progress, last_saved_update=progress['update'])


def epoch_batches(dataset, args, epoch):
    indices = dataset.epoch_indices(args.seed+epoch, args.poetry_ratio)
    if args.smoke_test:
        # 工程验证固定覆盖原通用数据与古诗数据；正式训练完全使用混合随机顺序。
        short = min(range(dataset.general_count), key=lambda i: len(dataset[i]['prompt']))
        indices = [short, dataset.general_count]+[i for i in indices if i not in (short, dataset.general_count)]
    return [indices[i:i+args.batch_size] for i in range(0, len(indices), args.batch_size)]


def selection_decision(result, progress, args):
    if 'baseline_general' not in progress:
        progress['baseline_general'] = result['general']['model_score']
    if 'baseline_repetition' not in progress:
        progress['baseline_repetition'] = {c: result[c]['repetition_rate'] for c in ('general', 'poetry')}
        progress['baseline_poetry_qualified'] = result['poetry']['qualified_pass_rate']
    score = .5*(result['general']['model_score']+result['poetry']['reward'])
    general_guard = result['general']['model_score'] >= progress['baseline_general']-args.general_regression_tolerance
    repetition_guard = all(result[c]['repetition_rate'] <= progress['baseline_repetition'][c]+
                           args.repetition_regression_tolerance for c in ('general', 'poetry'))
    quality_guard = result['poetry']['qualified_pass_rate'] >= progress['baseline_poetry_qualified']
    # Low repetition cannot compensate for a collapse in useful content, even at
    # a zero-qualified baseline. Compare with both the source and accepted best.
    quality = dict(score=score, poetry_reward=result['poetry']['reward'],
                   semantic=result['poetry'].get('semantic_quality', 0.))
    quality.update({f'{c}_{k}': result[c].get(k, 0.) for c in ('general', 'poetry')
                    for k in ('empty_fraction', 'truncated_fraction')})
    baseline = progress.setdefault('baseline_quality', dict(quality))
    anchor = progress.get('best_quality', baseline)
    guards = {}
    for key, option, default in [('score', 'quality_regression_tolerance', .1),
            ('poetry_reward', 'poetry_reward_regression_tolerance', .15),
            ('semantic', 'semantic_regression_tolerance', .05)]:
        guards[key] = quality[key] >= max(baseline[key], anchor[key])-getattr(args, option, default)
    for key in quality.keys()-{'score', 'poetry_reward', 'semantic'}:
        guards[key] = quality[key] <= min(baseline[key], anchor[key])+getattr(args, 'completion_regression_tolerance', .05)
    quality_guard = quality_guard and all(guards.values())
    result.update(selection_score=score, general_guard_passed=general_guard,
                  repetition_guard_passed=repetition_guard, poetry_guard_passed=quality_guard,
                  quality_guards=guards, selection_quality=quality)
    repetition_rate = .5*(result['general']['repetition_rate']+result['poetry']['repetition_rate'])
    return score, (-repetition_rate, result['poetry']['qualified_pass_rate'], score), (general_guard and repetition_guard and quality_guard)


@torch.no_grad()
def validate(args, run_dir, model, tokenizer, reward_model, dataset, progress, control=None):
    rows = dataset.validation_rows(args.validation_per_channel)
    if not any(r['channel'] == 'general' for r in rows) or not any(r['channel'] == 'poetry' for r in rows):
        raise ValueError('验证必须同时包含通用和古诗数据，请补充数据或调整划分比例')
    records = []
    model.eval()
    for index, row in enumerate(rows, 1):
        if control is not None:
            control.check()
        live_state(run_dir, 'validating', progress, sample=index, samples=len(rows))
        print(f'[validation sample] update={progress["update"]} {index}/{len(rows)} '
              f'{row["channel"]} {row["id"]}', flush=True)
        inputs = tokenizer(row['prompt'], return_tensors='pt', add_special_tokens=False,
                           return_token_type_ids=False).to(args.device)
        with amp_context(args.device, args.dtype):
            ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                max_new_tokens=args.eval_max_gen_len, do_sample=False, temperature=1., top_k=0, top_p=1.,
                eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(ids[0, inputs.input_ids.size(1):], skip_special_tokens=True)
        ended = bool(completion_ended(ids[:, inputs.input_ids.size(1):], tokenizer.eos_token_id)[0])
        rewards, details = calculate_rewards([row], [text], reward_model, 1, 'cpu', args.reward_model_weight,
                                             completion_ended=[ended], repetition_weight=args.repetition_weight)
        records.append(dict(id=row['id'], channel=row['channel'], task=(row.get('spec') or {}).get('task', 'template'), prompt=row['messages'], response=text,
                            reward=details[0], truncated=not ended))
        print(f'[validation result] reward={details[0]["total"]:.3f} truncated={not ended} '
              f'format={details[0]["rules"].get("format_exact", "n/a")}', flush=True)
    result = dict(update=progress['update'], repetition_detector='grpo_excessive_v1',
                  decoding=dict(mode='greedy', repetition_penalty=1., top_k=0, top_p=1.,
                                max_new_tokens=args.eval_max_gen_len))
    for channel in ['general', 'poetry']:
        group = [r for r in records if r['channel'] == channel]
        result[channel] = dict(count=len(group), reward=sum(r['reward']['total'] for r in group)/len(group),
            model_score=sum(r['reward']['model_bounded'] for r in group)/len(group),
            empty_fraction=sum(not r['response'].strip() for r in group)/len(group),
            truncated_fraction=sum(r['truncated'] for r in group)/len(group))
        result[channel].update(aggregate_repetition(group))
    poems = [r for r in records if r['channel'] == 'poetry']
    result['poetry']['format_pass_rate'] = sum(r['reward']['rules'].get('format_exact', 0) for r in poems)/len(poems)
    result['poetry']['qualified_pass_rate'] = sum(r['reward'].get('qualified', False) for r in poems)/len(poems)
    result['poetry']['semantic_quality'] = sum(r['reward'].get('semantic', {}).get('quality', 0.) for r in poems)/len(poems)
    result['poetry']['tasks'] = {}
    for task in sorted({r['task'] for r in poems}):
        group = [r for r in poems if r['task'] == task]
        result['poetry']['tasks'][task] = dict(count=len(group),
            qualified_pass_rate=sum(r['reward'].get('qualified', False) for r in group)/len(group),
            format_pass_rate=sum(r['reward']['rules'].get('format_exact', 0) for r in group)/len(group),
            **aggregate_repetition(group))
    score, rank, eligible = selection_decision(result, progress, args)
    if eligible and rank > tuple(progress.get('best_rank', [-float('inf')]*3)):
        progress.update(best_score=score, best_rank=list(rank), best_update=progress['update'])
        progress['best_quality'] = dict(result['selection_quality'])
        suffix = '_moe' if model.config.use_moe else ''
        weight = run_dir/f'{args.save_weight}_best_{model.config.hidden_size}{suffix}.pth'
        atomic_save(weight, {k: v.detach().cpu() for k, v in model.state_dict().items()})
        write_checkpoint_config(weight, model.config)
    with (run_dir/'validation_metrics.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False)+'\n')
    atomic_json(run_dir/f'validation_samples_{progress["update"]:06d}.json', records)
    print('[validation] '+json.dumps(result, ensure_ascii=False), flush=True)
    return result


def train(args, run_dir, model, tokenizer, ref_model, reward_model, dataset, optimizer, scheduler,
          scaler, progress, source, signature, control=None):
    started, first_update = time.monotonic(), progress['update']
    replay = SupervisedReplay(project_path(args.replay_data_path), tokenizer, args.max_seq_len) if args.replay_data_path and args.replay_weight else None
    metrics_path = run_dir/'metrics.jsonl'
    if metrics_path.exists():
        retained = [line for line in metrics_path.read_text(encoding='utf-8').splitlines()
                    if json.loads(line)['update'] <= progress['update']]
        metrics_path.write_text(''.join(line+'\n' for line in retained), encoding='utf-8')
    optimizer.zero_grad(set_to_none=True)
    start_epoch = progress['epoch']
    for epoch in range(start_epoch, args.epochs):
        batches = epoch_batches(dataset, args, epoch)
        first_batch = progress['next_batch'] if epoch == start_epoch else 0
        loader = DataLoader(dataset, batch_sampler=batches[first_batch:], collate_fn=collate_prompts,
                            num_workers=args.num_workers, generator=torch.Generator().manual_seed(args.seed+epoch))
        accumulation, stats, samples = 0, [], []
        for batch_index, rows in enumerate(loader, first_batch):
            if control is not None and control.requested and accumulation == 0:
                raise StopRequested()
            window_size = min(args.accumulation_steps, len(batches)-(batch_index-accumulation))
            print(f'[rollout] update={progress["update"]+1} batch={batch_index+1} '
                  f'channels={[r["channel"] for r in rows]}', flush=True)
            ids, full_mask, cmask, texts = sample_group(model, tokenizer, rows, args)
            ended = completion_ended(ids[:, -cmask.size(1):], tokenizer.eos_token_id).tolist()
            rewards, details = calculate_rewards(rows, texts, reward_model, args.num_generations,
                args.device, args.reward_model_weight, completion_ended=ended,
                repetition_weight=args.repetition_weight)
            exempt = [preserves_reference(rows[i//args.num_generations], text) or
                      not details[i]['repetition']['excessive'] for i, text in enumerate(texts)]
            repeat_mask = repeated_token_mask(ids[:, -cmask.size(1):], cmask, exempt)
            advantages = group_advantages(rewards, args.num_generations)
            old = frozen_logps(model, ids, full_mask, cmask.size(1), args.device, args)
            reference = frozen_logps(ref_model, ids, full_mask, cmask.size(1), args.ref_device, args)
            if not torch.isfinite(old[cmask]).all() or not torch.isfinite(reference[cmask]).all():
                raise FloatingPointError('策略 logprob 非有限；不进行更新')
            loss_value, kl_value, ul_value = 0., 0., 0.
            for start in range(0, len(ids), args.policy_micro_batch_size):
                sl = slice(start, start+args.policy_micro_batch_size)
                current = token_logps(model, ids[sl], full_mask[sl], cmask.size(1), args.temperature, args.dtype)
                loss, kl = policy_objective(current, old[sl], reference[sl], advantages[sl], cmask[sl], args)
                ul = unlikelihood_loss(current, repeat_mask[sl])
                loss = loss + args.unlikelihood_weight*ul
                if not torch.isfinite(loss):
                    raise FloatingPointError('策略 loss 非有限；未保存异常更新')
                weight = len(ids[sl])/len(ids)
                scaler.scale(loss*weight/window_size).backward()
                loss_value += loss.item()*weight
                kl_value += kl.item()*weight
                ul_value += ul.item()*weight
            replay_nll = 0.
            if replay is not None:
                replay_nll = replay.backward(model, scaler, args.device,
                    lambda: amp_context(args.device, args.dtype),
                    args.seed+epoch*100000000+batch_index, args.replay_weight, window_size)
            stats.append(dict(loss=loss_value+args.replay_weight*replay_nll, kl=kl_value, reward=rewards.mean().item(),
                replay_nll=replay_nll,
                unlikelihood_loss=ul_value, repeated_token_fraction=(repeat_mask.sum()/cmask.sum().clamp_min(1)).item(),
                advantage_std=advantages.std(unbiased=False).item(), mean_tokens=cmask.sum(1).float().mean().item(),
                truncated_fraction=sum(not e for e in ended)/len(ended),
                zero_variance_groups=float((rewards.reshape(-1, args.num_generations).std(1, unbiased=False) < 1e-6).sum())))
            samples.extend(dict(id=rows[i//args.num_generations]['id'], channel=rows[i//args.num_generations]['channel'],
                                prompt=rows[i//args.num_generations]['messages'], response=text, reward=detail,
                                truncated=not ended[i])
                           for i, (text, detail) in enumerate(zip(texts, details)))
            del ids, full_mask, cmask, old, reference, rewards, advantages, current, loss, ul, repeat_mask
            accumulation += 1
            if accumulation < window_size:
                continue
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress.update(update=progress['update']+1, epoch=epoch, next_batch=batch_index+1)
            if progress['next_batch'] == len(batches):
                progress.update(epoch=epoch+1, next_batch=0)
            record = dict(update=progress['update'], epoch=epoch+1, grad_norm=float(norm),
                learning_rate=optimizer.param_groups[0]['lr'], seconds=time.monotonic()-started,
                **{key: sum(s[key] for s in stats)/len(stats) for key in stats[0]})
            record['channels'] = {}
            for channel in ('general', 'poetry'):
                group = [s for s in samples if s['channel'] == channel]
                if group:
                    record['channels'][channel] = dict(candidates=len(group),
                        reward=sum(s['reward']['total'] for s in group)/len(group),
                        truncated_fraction=sum(s['truncated'] for s in group)/len(group))
                    record['channels'][channel].update(aggregate_repetition(group))
                    if channel == 'poetry':
                        record['channels'][channel].update(
                            format_pass_rate=sum(s['reward']['rules'].get('format_exact', 0) for s in group)/len(group),
                            qualified_pass_rate=sum(s['reward'].get('qualified', False) for s in group)/len(group),
                            semantic_quality=sum(s['reward'].get('semantic', {}).get('quality', 0.) for s in group)/len(group))
            with metrics_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            if progress['update'] % args.log_interval == 0:
                print('[train] '+json.dumps(record, ensure_ascii=False), flush=True)
            atomic_json(run_dir/'latest_samples.json', samples)
            live_state(run_dir, 'training', progress, last_metrics=record)
            if args.debug_mode:
                print(json.dumps(samples, ensure_ascii=False, indent=2), flush=True)
            stop = (progress['update'] >= progress['total_updates'] or
                    (args.stop_after_steps and progress['update']-first_update >= args.stop_after_steps)
                    or (control is not None and control.requested))
            if not (control is not None and control.requested) and (stop or progress['update'] % args.eval_interval == 0):
                validate(args, run_dir, model, tokenizer, reward_model, dataset, progress, control)
            if stop or progress['update'] == 1 or progress['update'] % args.save_interval == 0 or progress['epoch'] > epoch:
                status = 'completed' if progress['update'] >= progress['total_updates'] else ('stopped' if stop else 'running')
                save_training(run_dir, model, optimizer, scheduler, scaler, args, progress, source, signature, tokenizer, status)
            accumulation, stats, samples = 0, [], []
            if stop:
                return


def main(argv=None):
    args, supplied = parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    resume = None
    if args.resume:
        resume_path = project_path(args.resume)
        if resume_path.name != 'resume.pth' or not (resume_path.parent/'source.json').is_file():
            raise ValueError('--resume 只接受本 GRPO 独立运行的 resume.pth，不能恢复正在训练的 SFT 断点')
        resume = torch.load(resume_path, map_location='cpu', weights_only=True)
        if resume.get('format') != FORMAT:
            raise ValueError('GRPO 断点版本不匹配')
        mutable = {'resume', 'device', 'ref_device', 'reward_device', 'cpu_threads', 'num_workers',
                   'stop_after_steps', 'debug_mode', 'log_interval', 'prepare_only', 'eval_only',
                   'poetry_judge_url', 'poetry_judge_timeout'}
        for key, value in resume['args'].items():
            if key in mutable:
                continue
            if '--'+key in supplied and getattr(args, key) != value:
                raise ValueError('恢复训练不能改变 '+key)
            setattr(args, key, value)
        run_dir = resume_path.parent
        source = resume['source']
        protect_source(run_dir, source['path'])
        snapshot = run_dir/'source_snapshot'/Path(source['snapshot']).name
        config, _ = load_checkpoint_config(snapshot)
        signature = code_signature()
        if signature != resume['signature']:
            raise ValueError('代码/模板与断点不一致，请新建 GRPO 运行')
        if weight_sha256(snapshot) != source['sha256']:
            raise ValueError('运行内参考快照被更改')
    else:
        source_path = source_weight(args.checkpoint)
        output_root = project_path(args.save_dir)
        protect_source(output_root, source_path)
        run_dir = output_root/(args.run_name or datetime.now().strftime('poetry_%Y%m%d_%H%M%S_%f'))
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot, config, source = snapshot_source(source_path, run_dir)
        signature = code_signature()
        for name, checksum in signature.items():
            saved_code = run_dir/'code_snapshot'/name
            saved_code.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, saved_code)
            if file_digest(saved_code) != checksum:
                raise RuntimeError('Code changed during startup; no training performed: '+name)
    print(f'[run] {run_dir}\n[source read only] {source["path"]}\n[snapshot sha256] {source["sha256"]}', flush=True)
    args.data_path = str(project_path(args.data_path))
    args.poetry_data_path = str(ensure_poetry_data(project_path(args.poetry_data_path)))
    args.reward_model_path = str(ensure_reward_model(args.reward_model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'model'), local_files_only=True)
    tokenizer.padding_side = 'left'
    dataset = MixedPromptDataset(args.data_path, args.poetry_data_path, tokenizer, args.max_seq_len, args.validation_fraction)
    atomic_json(run_dir/'data_quality.json', dict(quarantined=len(dataset.quarantine),
        rejected=dataset.quarantine, deduplicated=dataset.duplicates, trimmed=dataset.trimmed,
        general_train=dataset.general_count, general_valid=len(dataset.general_valid),
        poetry_train=dataset.poetry_count, poetry_valid=len(dataset.poetry_valid),
        original_files_modified=False))
    data_hash = dict(general=file_digest(args.data_path), poetry=file_digest(args.poetry_data_path),
                    poetry_valid=file_digest(Path(args.poetry_data_path).with_name('valid.jsonl')))
    if args.replay_data_path:
        args.replay_data_path = str(project_path(args.replay_data_path))
        data_hash['replay_train'] = file_digest(args.replay_data_path)
    if resume and data_hash != resume['progress']['data_sha256']:
        raise ValueError('恢复时训练数据已改变')
    if args.max_seq_len+max(args.max_gen_len, args.eval_max_gen_len) > config.max_position_embeddings:
        raise ValueError('提示与生成长度之和超出模型位置编码上限')
    print(f'[config] hidden={config.hidden_size}, layers={config.num_hidden_layers}, moe={config.use_moe}, '
          f'norm_topk_prob={config.norm_topk_prob}\n[data] general={dataset.general_count}, poetry={dataset.poetry_count}, '
          f'poetry_ratio={args.poetry_ratio}, trimmed_general={dataset.trimmed}, '
          f'general_valid={len(dataset.general_valid)}, poetry_valid={len(dataset.poetry_valid)}, '
          f'normalized_tails={dataset.normalized_tails}, deduplicated={dataset.duplicates}', flush=True)
    print(f'[data quarantine] excluded={len(dataset.quarantine)}; originals unchanged', flush=True)
    if not resume:
        atomic_json(run_dir/'run_config.json', dict(args=vars(args), model_config=config.to_dict(),
            source=source, signature=signature, data_sha256=data_hash))
    if args.prepare_only:
        if not resume:
            atomic_json(run_dir/'summary.json', dict(status='prepared', optimizer_updates=0, source=source,
                                                     source_directory_written=False))
        print('[prepared] 资源、数据和快照已完成；未执行训练', flush=True)
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
        free, _ = torch.cuda.mem_get_info(args.device)
        print(f'[device] {args.device}, free={free/2**30:.1f} GiB; reward={args.reward_device}, ref={args.ref_device}', flush=True)
    state = torch.load(snapshot, map_location='cpu', weights_only=True)
    model = MindChenForCausalLM(config)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    ref_model = MindChenForCausalLM(config)
    ref_model.load_state_dict(state, strict=True)
    ref_model.to(args.ref_device).eval().requires_grad_(False)
    del state
    reward_model = PoetryRewardModel(args.reward_model_path, args.reward_device, args.reward_max_length,
        judge_model=args.poetry_judge_model, judge_url=args.poetry_judge_url, judge_timeout=args.poetry_judge_timeout)
    judge_provenance = None
    if getattr(reward_model, 'judge', None) is not None:
        reward_model.judge.audit_path = run_dir/'semantic_judgments.jsonl'
        calibration = reward_model.judge.calibrate()
        atomic_json(run_dir/'judge_calibration.json', calibration)
        judge_provenance = calibration['provenance']
        if resume and resume['progress'].get('judge_provenance') != judge_provenance:
            raise ValueError('Local semantic judge revision changed; start a new run')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total = math.ceil(len(epoch_batches(dataset, args, 0))/args.accumulation_steps)*args.epochs
    total = min(total, args.max_steps) if args.max_steps else total
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total, 1), eta_min=args.learning_rate/10)
    scaler = torch.amp.GradScaler('cuda', enabled=str(args.device).startswith('cuda') and args.dtype == 'float16')
    progress = dict(update=0, epoch=0, next_batch=0, total_updates=total, data_sha256=data_hash,
                    judge_provenance=judge_provenance)
    if resume:
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler'])
        progress = resume['progress']
        random.setstate(resume['rng']['python'])
        torch.set_rng_state(resume['rng']['torch'])
        if str(args.device).startswith('cuda') and resume['rng']['cuda']:
            torch.cuda.set_rng_state_all(resume['rng']['cuda'])
        del resume
    if args.eval_only:
        validate(args, run_dir, model, tokenizer, reward_model, dataset, progress)
        if not args.resume:
            atomic_json(run_dir/'summary.json', dict(status='evaluated', optimizer_updates=0, source=source,
                                                     source_directory_written=False))
        return
    if progress['update'] >= progress['total_updates']:
        print('[completed] 此运行已完成，无须再次更新', flush=True)
        return
    with StopController(run_dir) as control:
        try:
            if progress['update'] == 0:
                validate(args, run_dir, model, tokenizer, reward_model, dataset, progress, control)
            print(f'[start] update={progress["update"]}/{total}; policy={args.device}; reward={args.reward_device}; ref={args.ref_device}', flush=True)
            train(args, run_dir, model, tokenizer, ref_model, reward_model, dataset, optimizer, scheduler,
                  scaler, progress, source, signature, control)
        except StopRequested:
            print('[stop] safe optimizer boundary reached; saving full checkpoint', flush=True)
            save_training(run_dir, model, optimizer, scheduler, scaler, args, progress, source, signature,
                          tokenizer, status='stopped')
        except BaseException as exc:
            live_state(run_dir, 'failed', progress, error=repr(exc))
            raise


if __name__ == '__main__':
    main()
