# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""在指定768 MoE预训练权重上做全量SFT（单GPU，默认参数可直接运行）。

正式训练：python trainer/train_full_sft.py
小规模验证：python trainer/train_full_sft.py --smoke_test
完整数据准备：python trainer/train_full_sft.py --prepare_only
恢复：python trainer/train_full_sft.py --resume <运行目录>/latest_resume.pth

说明：所有模型参数参与更新；用户/system/工具返回和padding不计loss，
只监督assistant区间。验证独立读取valid文件，绝不从训练文件随机分出。
"""
import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import datasets  # Windows下保持pyarrow/datasets先于torch加载
import torch
import transformers
from torch.utils.data import DataLoader

from model.model_mindchen import MindChenConfig
from trainer.trainer_utils import init_model, setup_seed
from trainer.continue_pretrain_utils import (
    EpochBatchSampler, METRIC_FIELDS, MetricsWriter, atomic_json, atomic_torch_save,
    file_info, make_logger, restore_rng, rewind_metrics, rng_state, sha256_file,
)
from trainer.sft_data import AssistantDataset, SFTCollator, audit_dataset, load_split
from trainer.project_paths import relocate_project_path, relocate_sft_checkpoint

SFT_FIELDS = METRIC_FIELDS + ['source_samples', 'skipped_samples', 'truncated_samples',
    'total_source_samples', 'total_skipped_samples', 'total_truncated_samples',
    'val_samples', 'val_source_samples', 'val_skipped_samples', 'val_truncated_samples', 'weights_update']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='MindChen全量SFT：指定预训练权重、独立train/valid、完整日志和断点')
    # from_weight在这里是完整文件路径的兼容别名，不再是pretrain这种文件名前缀。
    p.add_argument('--init_weight_path', '--from_weight', dest='init_weight_path',
        default=str(ROOT / 'out/continue_pretrain/poetry_gujilab_20261022_190751_427063/pretrain_poetry_gujilab_768_moe_best.pth'))
    p.add_argument('--train_data_path', '--data_path', dest='train_data_path',
        default=str(ROOT / 'dataset/sft_t2t_with_poetry/sft_t2t_with_poetry_train.jsonl'))
    p.add_argument('--valid_data_path', default=str(ROOT / 'dataset/sft_t2t_with_poetry/sft_t2t_with_poetry_valid.jsonl'))
    p.add_argument('--save_dir', default=str(ROOT / 'out/full_sft'))
    p.add_argument('--run_name', default='')
    p.add_argument('--save_weight', default='full_sft_poetry')
    p.add_argument('--resume', default='', help='本脚本的完整latest_resume.pth，自动恢复原参数')
    # 14GB混合数据先训练1轮；后续依据固定验证集和独立任务评测决定是否加轮数。
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--accumulation_steps', type=int, default=8, help='8×8=64条有效batch；最后不足窗口按实际监督token归一')
    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--min_lr_ratio', type=float, default=0.1)
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--num_hidden_layers', type=int, default=8)
    p.add_argument('--use_moe', type=int, choices=[0, 1], default=1)
    p.add_argument('--max_seq_len', type=int, default=1024)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--export_dtype', choices=['float32', 'float16'], default='float32',
        help='默认FP32保留SFT小幅更新；选择FP16可减小纯权重文件，但会量化舍入')
    p.add_argument('--num_workers', type=int, default=0, help='Windows下默认0，避免进程启动和tokenizer状态问题')
    p.add_argument('--log_interval', type=int, default=50, help='控制台间隔；CSV/JSONL每次更新均记录')
    p.add_argument('--save_interval', type=int, default=1000)
    p.add_argument('--eval_interval', type=int, default=1000)
    p.add_argument('--validation_samples', type=int, default=1024, help='每次周期验证的固定子集；0表示每次跑全部valid')
    p.add_argument('--validation_batch_size', type=int, default=8)
    p.add_argument('--skip_final_full_validation', action='store_true', help='不对导出的best跑最终完整验证；默认会跑')
    p.add_argument('--train_samples', type=int, default=0, help='0读全部训练数据；正数只读文件前N条，供调试')
    p.add_argument('--valid_samples', type=int, default=0, help='0读全部valid；正数只读前N条，供调试')
    p.add_argument('--audit_samples', type=int, default=256, help='均匀抽样检查截断、监督token和结束标记')
    p.add_argument('--max_updates', type=int, default=0, help='缩短整个LR计划；0为完整计划')
    p.add_argument('--stop_after_updates', type=int, default=0, help='到累计第N次更新安全暂停，保持原LR计划')
    p.add_argument('--smoke_test', action='store_true', help='192条train+16条valid、3次更新，独立目录')
    p.add_argument('--prepare_only', action='store_true', help='严格加载权重、读取完整数据并审计，不训练')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args(argv)


def amp_context(args):
    if args.device.startswith('cuda') and args.dtype != 'float32':
        return torch.autocast('cuda', dtype=getattr(torch, args.dtype))
    return nullcontext()


def lr_for_update(index, total, peak, warmup_ratio, min_ratio):
    warmup = int(total * warmup_ratio)
    if warmup and index <= warmup:
        return peak * index / warmup
    progress = max(0.0, min(1.0, (index-warmup) / max(1, total-warmup)))
    return peak * (min_ratio + (1-min_ratio) * 0.5 * (1+math.cos(math.pi*progress)))


def export_weights(model, path, dtype='float32'):
    # 默认纯FP32 state_dict，保留低学习率下的小幅更新；不等于关闭BF16混合精度。
    # 完整断点另外保存FP32主参数、AdamW、RNG和数据游标。
    atomic_torch_save(path, {k: v.detach().to(getattr(torch, dtype)).cpu() for k, v in model.state_dict().items()})


def evaluate(model, loader, args):
    """仅对assistant监督token计算加权NLL/PPL，不把不同长度batch简单平均。"""
    was_training = model.training
    totals = dict(val_tokens=0, val_samples=0, val_source_samples=0, val_skipped_samples=0, val_truncated_samples=0)
    nll_sum = 0.0
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                totals['val_source_samples'] += batch['source_samples']
                totals['val_skipped_samples'] += batch['skipped_samples']
                totals['val_truncated_samples'] += batch['truncated_samples']
                if batch['input_ids'] is None:
                    continue
                n = int((batch['labels'][:, 1:] != -100).sum())
                with amp_context(args):
                    loss = model(batch['input_ids'].to(args.device), attention_mask=batch['attention_mask'].to(args.device),
                                 labels=batch['labels'].to(args.device)).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('验证loss为非有限值')
                nll_sum += float(loss) * n
                totals['val_tokens'] += n
                totals['val_samples'] += batch['input_ids'].shape[0]
    finally:
        model.train(was_training)
    if not totals['val_tokens']:
        raise ValueError('验证集没有assistant监督token；请检查模板和max_seq_len')
    nll = nll_sum / totals['val_tokens']
    return dict(val_nll=nll, val_ppl=math.exp(nll) if nll < 700 else None, **totals)


def main():
    args = parse_args()
    session = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    resume = None
    relocation = {}
    if args.resume:
        checkpoint_path = Path(relocate_project_path(args.resume, ROOT)).resolve()
        resume = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if resume.get('format_version') != 'full_sft_v1':
            raise ValueError('resume必须是本SFT脚本的完整断点，不能使用预训练断点或纯权重')
        # 旧断点可能保存C盘绝对路径。按原项目内相对位置迁到当前ROOT，
        # 保留权重哈希、文件大小和时间戳校验，避免误加载其他数据。
        relocation = relocate_sft_checkpoint(resume, ROOT)
        provided = {v[2:].split('=')[0] for v in sys.argv[1:] if v.startswith('--')}
        if 'from_weight' in provided: provided.add('init_weight_path')
        if 'data_path' in provided: provided.add('train_data_path')
        for key, value in resume['args'].items():
            if key in ('resume', 'stop_after_updates', 'prepare_only', 'skip_final_full_validation'):
                continue
            if key in provided and key in ('log_interval', 'eval_interval', 'save_interval'):
                continue
            if key in provided and getattr(args, key) != value:
                raise ValueError(f'续训不能改变{key}：原值{value}，新值{getattr(args,key)}')
            setattr(args, key, value)
        folder = checkpoint_path.parent
    else:
        if args.smoke_test:
            args.train_samples = args.train_samples or 192
            args.valid_samples = args.valid_samples or 16
            args.validation_samples = min(args.validation_samples or 16, 16)
            args.max_updates = args.max_updates or 3
            args.log_interval, args.eval_interval, args.save_interval = 1, 1, 2
        name = args.run_name or ('smoke_' if args.smoke_test else 'poetry_sft_') + session
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('run_name只能是目录名')
        folder = Path(args.save_dir).resolve() / name
        folder.mkdir(parents=True, exist_ok=False)
    logger = make_logger(folder, 'full_sft')
    if relocation:
        logger.info('项目路径重定位：%s', json.dumps(relocation, ensure_ascii=False))
    writer, old_signal = None, None
    started = time.perf_counter()
    progress = dict(update=0, next_epoch=0, next_batch=0, total_samples=0, total_valid_tokens=0,
        total_source_samples=0, total_skipped_samples=0, total_truncated_samples=0,
        elapsed_seconds=0.0, best_val_nll=None, best_update=None)
    try:
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('本入口针对当前单GPU环境；不要用torchrun多卡启动')
        for key in ('epochs', 'batch_size', 'accumulation_steps', 'validation_batch_size', 'log_interval', 'eval_interval', 'audit_samples'):
            if getattr(args, key) < 1: raise ValueError(f'{key}必须>=1')
        for key in ('num_workers', 'train_samples', 'valid_samples', 'validation_samples', 'max_updates', 'stop_after_updates', 'save_interval'):
            if getattr(args, key) < 0: raise ValueError(f'{key}不能为负')
        if args.max_seq_len < 32 or args.learning_rate <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
            raise ValueError('检查序列长度、学习率、梯度裁剪和weight_decay')
        if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1:
            raise ValueError('warmup_ratio需在[0,1)，min_lr_ratio需在[0,1]')
        cuda = args.device.startswith('cuda')
        if cuda:
            if not torch.cuda.is_available(): raise RuntimeError('CUDA不可用')
            torch.cuda.set_device(torch.device(args.device))
            if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
                raise RuntimeError('GPU不支持BF16，请改float16或float32')
        for key in ('init_weight_path', 'train_data_path', 'valid_data_path'):
            setattr(args, key, str(Path(getattr(args, key)).resolve()))
        if args.train_data_path == args.valid_data_path:
            raise ValueError('训练与验证路径不能相同')
        source = file_info(args.init_weight_path, checksum=True)
        data_files = dict(train=file_info(args.train_data_path), valid=file_info(args.valid_data_path))
        if resume and (source != resume['source_weight'] or data_files != resume['data_files']):
            raise ValueError('基础权重、训练集或验证集发生变化，拒绝按旧游标恢复')
        if Path(args.save_weight).name != args.save_weight:
            raise ValueError('save_weight只能是文件名前缀')
        setup_seed(args.seed)
        config = MindChenConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
        model, tokenizer = init_model(config, 'none', device=args.device)
        weights = resume['model'] if resume else torch.load(args.init_weight_path, map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        del weights
        suffix = '_moe' if config.use_moe else ''
        latest_path = folder / f'{args.save_weight}_{config.hidden_size}{suffix}.pth'
        best_path = latest_path.with_name(latest_path.stem + '_best.pth')
        if latest_path.resolve() == Path(args.init_weight_path) or best_path.resolve() == Path(args.init_weight_path):
            raise ValueError('输出路径会覆盖基础权重')
        logger.info('严格加载预训练权重成功 | %s | 输出=%s', args.init_weight_path, folder)
        logger.info('读取train/valid并构建或复用Arrow缓存；正式训练与验证分开加载。')
        train_rows = load_split(args.train_data_path, args.train_samples)
        valid_rows = load_split(args.valid_data_path, args.valid_samples)
        train_ds = AssistantDataset(train_rows, tokenizer, args.max_seq_len)
        valid_ds = AssistantDataset(valid_rows, tokenizer, args.max_seq_len)
        collator = SFTCollator(tokenizer.pad_token_id)
        subset_n = min(args.validation_samples or len(valid_ds), len(valid_ds))
        # 使用独立Generator，不消耗训练RNG；恢复时选择同一批验证样本。
        val_indices = sorted(torch.randperm(len(valid_ds), generator=torch.Generator().manual_seed(args.seed+991))[:subset_n].tolist())
        periodic_ds = torch.utils.data.Subset(valid_ds, val_indices)
        make_val_loader = lambda ds: DataLoader(ds, batch_size=args.validation_batch_size, collate_fn=collator,
            num_workers=0, pin_memory=cuda, generator=torch.Generator().manual_seed(args.seed+992))
        val_loader = make_val_loader(periodic_ds)
        batches_per_epoch = math.ceil(len(train_ds) / args.batch_size)
        full_updates = math.ceil(batches_per_epoch / args.accumulation_steps) * args.epochs
        total_updates = min(full_updates, args.max_updates) if args.max_updates else full_updates
        if resume and total_updates != resume['total_updates']:
            raise ValueError('训练样本数或计划更新数与断点不一致')
        audit = dict(train=audit_dataset(train_ds, args.audit_samples), valid=audit_dataset(valid_ds, args.audit_samples))
        if not audit['train']['target_tokens'] or not audit['valid']['target_tokens']:
            raise ValueError('抽样未发现可用assistant监督token，请检查模板/长度/数据')
        atomic_json(folder / f'data_audit_{session}.json', audit)
        # 保存输入来源、选择索引和少量原文/标签预览；不重复复制14GB原文件。
        if not resume:
            atomic_json(folder / 'validation_indices.json', val_indices)
            with (folder / 'supervision_preview.jsonl').open('w', encoding='utf-8') as stream:
                for split, ds in [('train', train_ds), ('valid', valid_ds)]:
                    for index in range(min(4, len(ds))):
                        item = ds.encode(ds.samples[index], index)
                        record = dict(split=split, index=index, conversations=ds.samples[index]['conversations'],
                            full_tokens=item['full_tokens'], truncated=item['truncated'], target_tokens=item['target_tokens'],
                            supervised_text=tokenizer.decode([v for v in item['labels'] if v != -100], skip_special_tokens=False))
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        metadata = dict(args=vars(args), project_root=str(ROOT), relocation=relocation, source_weight=source, data_files=data_files,
            train_samples=len(train_ds), valid_samples=len(valid_ds), periodic_validation_samples=subset_n,
            effective_batch_size=args.batch_size*args.accumulation_steps, batches_per_epoch=batches_per_epoch,
            total_updates=total_updates, model_config=config.to_dict(), python=sys.version, executable=sys.executable,
            torch=torch.__version__, transformers=transformers.__version__, datasets=datasets.__version__,
            gpu=torch.cuda.get_device_name() if cuda else None,
            tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            script_sha256=sha256_file(__file__), data_code_sha256=sha256_file(ROOT/'trainer/sft_data.py'),
            objective='assistant区间（含think/工具调用/EOS）token加权NLL+MoE aux；其余角色和padding标签为-100',
            validation_scope='本valid文件按数据报告全部是诗词古文，不能代表通用能力；属于SFT验证，原预训练可能见过作品', audit=audit)
        atomic_json(folder / ('run_config.json' if not resume else f'resume_config_{session}.json'), metadata)
        logger.info('数据 train=%d valid=%d 周期验证=%d | 有效batch=%d | 计划更新=%d',
            len(train_ds), len(valid_ds), subset_n, metadata['effective_batch_size'], total_updates)
        logger.info('抽样检查：%s', json.dumps(audit, ensure_ascii=False))
        if args.prepare_only:
            atomic_json(folder/'summary.json', dict(status='prepared', optimizer_updates=0, train_samples=len(train_ds), valid_samples=len(valid_ds), total_updates=total_updates))
            logger.info('数据与参数准备完成，没有执行训练。')
            return
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=cuda and args.dtype == 'float16')
        if resume:
            optimizer.load_state_dict(resume['optimizer'])
            scaler.load_state_dict(resume['scaler'])
            progress.update(resume['progress'])
            restore_rng(resume['rng'])
            rewind_metrics(folder, progress['update'], session, fieldnames=SFT_FIELDS)
            logger.info('恢复optimizer/scaler/RNG及数据游标：update=%d，下一epoch=%d batch=%d', progress['update'], progress['next_epoch']+1, progress['next_batch']+1)
            del resume
            gc.collect()
        writer = MetricsWriter(folder, fieldnames=SFT_FIELDS)
        stop_requested = [False]
        def request_stop(signum, frame):
            stop_requested[0] = True
            logger.info('收到Ctrl+C，将在完成当前更新后保存并退出。')
        old_signal = signal.signal(signal.SIGINT, request_stop)
        prior_elapsed = progress['elapsed_seconds']
        tick_start = time.perf_counter()
        first_update, current_epoch = progress['update'], progress['next_epoch']
        if cuda: torch.cuda.reset_peak_memory_stats()

        def record(phase, **values):
            return dict(timestamp=datetime.now().isoformat(timespec='seconds'), session=session, phase=phase,
                epoch=current_epoch+1, batch=progress['next_batch'], update=progress['update'], total_updates=total_updates,
                total_samples=progress['total_samples'], total_valid_tokens=progress['total_valid_tokens'],
                total_source_samples=progress['total_source_samples'], total_skipped_samples=progress['total_skipped_samples'],
                total_truncated_samples=progress['total_truncated_samples'],
                elapsed_seconds=prior_elapsed+time.perf_counter()-tick_start, best_val_nll=progress['best_val_nll'],
                gpu_allocated_gb=round(torch.cuda.memory_allocated()/2**30,3) if cuda else 0,
                gpu_reserved_gb=round(torch.cuda.memory_reserved()/2**30,3) if cuda else 0,
                gpu_peak_gb=round(torch.cuda.max_memory_allocated()/2**30,3) if cuda else 0, **values)

        def validate_subset():
            metrics = evaluate(model, val_loader, args)
            if progress['best_val_nll'] is None or metrics['val_nll'] < progress['best_val_nll']:
                progress.update(best_val_nll=metrics['val_nll'], best_update=progress['update'])
                export_weights(model, best_path, args.export_dtype)
            writer.write(record('validation', **metrics))
            logger.info('固定子集验证 update=%d | NLL=%.5f PPL=%.3f | 样本=%d/%d 排除=%d | best=%.5f@%d',
                progress['update'], metrics['val_nll'], metrics['val_ppl'] or float('inf'), metrics['val_samples'],
                metrics['val_source_samples'], metrics['val_skipped_samples'], progress['best_val_nll'], progress['best_update'])

        last_saved = -1
        def save():
            nonlocal last_saved
            if last_saved == progress['update']: return
            progress['elapsed_seconds'] = prior_elapsed + time.perf_counter()-tick_start
            export_weights(model, latest_path, args.export_dtype)
            atomic_torch_save(folder/'latest_resume.pth', dict(format_version='full_sft_v1', project_root=str(ROOT), args=vars(args),
                model={k:v.detach().cpu() for k,v in model.state_dict().items()}, optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(), progress=progress.copy(), rng=rng_state(), source_weight=source,
                data_files=data_files, total_updates=total_updates))
            last_saved = progress['update']
            logger.info('保存update=%d | 最新=%s | 最佳=%s | 完整断点=latest_resume.pth', progress['update'], latest_path.name, best_path.name)

        # 训练前先测一次基线；如果SFT后验证没有改善，best可能仍对应update=0。
        if progress['update'] == 0: validate_subset()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        stop = progress['update'] >= total_updates
        for epoch in range(progress['next_epoch'], args.epochs):
            if stop: break
            current_epoch = epoch
            skip = progress['next_batch'] if epoch == progress['next_epoch'] else 0
            sampler = EpochBatchSampler(len(train_ds), args.batch_size, args.seed+epoch, skip)
            loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collator, num_workers=args.num_workers,
                pin_memory=cuda, generator=torch.Generator().manual_seed(args.seed+epoch))
            iterator, cursor = iter(loader), skip
            while True:
                # 先收集一个累积窗口，才能按这个窗口真实的assistant token总数归一。
                # 这也正确处理最后不足8个microbatch和被排除的超长prompt样本。
                batches = list(itertools.islice(iterator, args.accumulation_steps))
                if not batches: break
                step_started = time.perf_counter()
                # 模型内部logits向前错一位：第0个输入预测第1个标签，故排除labels首位置。
                counts = [int((b['labels'][:,1:] != -100).sum()) if b['labels'] is not None else 0 for b in batches]
                tokens = sum(counts)
                if not tokens:
                    raise ValueError('整个梯度累积窗口均无assistant监督token；请增加max_seq_len或清理长prompt')
                lm_sum, aux_sum, used = 0.0, 0.0, 0
                for batch, n in zip(batches, counts):
                    if not n: continue
                    with amp_context(args):
                        output = model(batch['input_ids'].to(args.device, non_blocking=cuda),
                            attention_mask=batch['attention_mask'].to(args.device, non_blocking=cuda),
                            labels=batch['labels'].to(args.device, non_blocking=cuda))
                        aux = output.aux_loss if output.aux_loss is not None else output.loss.new_zeros(())
                        # 不除以固定accumulation_steps：不同回答长度的batch应按监督token加权。
                        loss = (output.loss+aux) * (n/tokens)
                    if not torch.isfinite(loss): raise FloatingPointError('训练loss非有限值')
                    scaler.scale(loss).backward()
                    lm_sum += float(output.loss.detach())*n
                    aux_sum += float(aux.detach())*n
                    used += batch['input_ids'].shape[0]
                    del output, loss
                update = progress['update']+1
                lr = lr_for_update(update,total_updates,args.learning_rate,args.warmup_ratio,args.min_lr_ratio)
                for group in optimizer.param_groups: group['lr'] = lr
                # 先还原梯度尺度再裁剪；BF16默认不启用GradScaler，FP16才启用。
                scaler.unscale_(optimizer)
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if cuda: torch.cuda.synchronize()
                cursor += len(batches)
                source_n = sum(b['source_samples'] for b in batches)
                skipped_n = sum(b['skipped_samples'] for b in batches)
                truncated_n = sum(b['truncated_samples'] for b in batches)
                # 游标只在完整optimizer更新后提交；恢复时直接跳过这些原始batch。
                progress.update(update=update,next_epoch=epoch+1 if cursor==batches_per_epoch else epoch,
                    next_batch=0 if cursor==batches_per_epoch else cursor,total_samples=progress['total_samples']+used,
                    total_valid_tokens=progress['total_valid_tokens']+tokens,total_source_samples=progress['total_source_samples']+source_n,
                    total_skipped_samples=progress['total_skipped_samples']+skipped_n,
                    total_truncated_samples=progress['total_truncated_samples']+truncated_n)
                elapsed = time.perf_counter()-tick_start
                row = record('train',lm_loss=lm_sum/tokens,aux_loss=aux_sum/tokens,total_loss=(lm_sum+aux_sum)/tokens,
                    learning_rate=lr,grad_norm=norm,samples=used,valid_tokens=tokens,source_samples=source_n,
                    skipped_samples=skipped_n,truncated_samples=truncated_n,
                    tokens_per_second=tokens/max(time.perf_counter()-step_started,1e-9),
                    eta_seconds=elapsed/max(1,update-first_update)*(total_updates-update))
                row.update(epoch=epoch+1,batch=cursor)
                writer.write(row)
                if update==1 or update%args.log_interval==0 or update==total_updates:
                    logger.info('epoch=%d/%d batch=%d/%d update=%d/%d | LM=%.5f aux=%.6f lr=%.8f grad=%.3f | 排除=%d 截断=%d GPU=%.2fGB ETA=%.1fh',
                        epoch+1,args.epochs,cursor,batches_per_epoch,update,total_updates,row['lm_loss'],row['aux_loss'],lr,norm,
                        progress['total_skipped_samples'],progress['total_truncated_samples'],row['gpu_peak_gb'],row['eta_seconds']/3600)
                stop = stop_requested[0] or update>=total_updates or (args.stop_after_updates and update>=args.stop_after_updates)
                if stop or update%args.eval_interval==0 or cursor==batches_per_epoch: validate_subset()
                if stop or cursor==batches_per_epoch or (args.save_interval and update%args.save_interval==0): save()
                if stop: break
        save()
        completed = progress['update'] >= total_updates
        full_result = None
        if completed and not args.skip_final_full_validation and not stop_requested[0]:
            logger.info('对导出的best运行完整valid验证（%d条）；不与周期子集混合选择best。',len(valid_ds))
            # 此时最新训练断点已保存。临时加载best做部署权重验证，最后恢复最新FP32参数，
            # 绝不把best权重配上latest优化器另存成错误恢复断点。
            latest_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            try:
                model.load_state_dict(torch.load(best_path,map_location='cpu',weights_only=True),strict=True)
                full_result = evaluate(model,make_val_loader(valid_ds),args)
                writer.write(record('validation_full_best',weights_update=progress['best_update'],**full_result))
                atomic_json(folder/'full_validation_best.json',dict(weights=str(best_path),weights_sha256=sha256_file(best_path),
                    weights_update=progress['best_update'],validation_file=data_files['valid'],**full_result))
                logger.info('best完整验证 NLL=%.5f PPL=%.3f | 样本=%d/%d 排除=%d',full_result['val_nll'],
                    full_result['val_ppl'] or float('inf'),full_result['val_samples'],full_result['val_source_samples'],full_result['val_skipped_samples'])
            finally:
                model.load_state_dict(latest_state,strict=True)
                del latest_state
        atomic_json(folder/'summary.json',dict(status='completed' if completed else 'stopped',**progress,
            total_updates=total_updates,latest_weight=str(latest_path),best_weight=str(best_path),
            resume_checkpoint=str(folder/'latest_resume.pth'),full_validation_best=full_result,
            source_weight_sha256=source['sha256'],wall_seconds=time.perf_counter()-started))
        logger.info('%s | update=%d/%d | 输出=%s','completed' if completed else 'stopped',progress['update'],total_updates,folder)
    except BaseException as exc:
        logger.exception('SFT未完成；原始权重未覆盖，可从最近的SFT断点恢复。')
        atomic_json(folder/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',progress=progress,
            timestamp=datetime.now().isoformat(timespec='seconds')))
        raise
    finally:
        if old_signal is not None: signal.signal(signal.SIGINT,old_signal)
        if writer: writer.close()


if __name__ == '__main__':
    main()
