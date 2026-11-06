# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""从指定基础权重继续预训练；默认参数已填写，完整训练由用户启动。

python trainer/train_continue_pretrain.py --smoke_test
python trainer/train_continue_pretrain.py
python trainer/train_continue_pretrain.py --resume <运行目录>/latest_resume.pth
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
# 保留Windows下datasets/pyarrow先于torch导入的顺序。
import datasets
import torch
import transformers
from torch.utils.data import DataLoader

from dataset.lm_dataset import PretrainDataset
from model.model_mindchen import MindChenConfig
from trainer.trainer_utils import init_model, setup_seed
from trainer.continue_pretrain_utils import (
    EpochBatchSampler, MetricsWriter, atomic_json, atomic_torch_save, file_info,
    make_logger, restore_rng, rewind_metrics, rng_state, sha256_file,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='MindChen 基础权重 + 诗词古籍混合数据继续预训练')
    p.add_argument('--init_weight_path', default=str(ROOT / 'out/pretrain_768_moe.pth'))
    p.add_argument('--data_path', default=str(ROOT / 'dataset/pretrain_t2t_with_poetry_gujilab.jsonl'))
    p.add_argument('--save_dir', default=str(ROOT / 'out/continue_pretrain'))
    p.add_argument('--run_name', default='')
    p.add_argument('--save_weight', default='pretrain_poetry_gujilab')
    p.add_argument('--resume', default='', help='本脚本保存的latest_resume.pth；自动恢复原训练参数')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--accumulation_steps', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--min_lr_ratio', type=float, default=0.1)
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--num_hidden_layers', type=int, default=8)
    p.add_argument('--use_moe', type=int, choices=[0, 1], default=1)
    p.add_argument('--max_seq_len', type=int, default=340)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--log_interval', type=int, default=50, help='控制台打印间隔；CSV/JSONL记录每次更新')
    p.add_argument('--save_interval', type=int, default=1000, help='每N次更新保存完整断点；0仅结束/中止保存')
    p.add_argument('--eval_interval', type=int, default=1000)
    p.add_argument('--validation_samples', type=int, default=256)
    p.add_argument('--validation_batch_size', type=int, default=8)
    p.add_argument('--train_samples', type=int, default=0, help='0使用全部；大于0只读取文件前N条训练样本及验证样本')
    p.add_argument('--max_updates', type=int, default=0, help='0使用完整计划；正数缩短计划并相应调整学习率调度')
    p.add_argument('--stop_after_updates', type=int, default=0, help='本次运行达到该累计更新数即安全退出，保持原调度用于续训')
    p.add_argument('--smoke_test', action='store_true', help='仅192条训练+最多8条验证、3次更新，单独输出目录')
    p.add_argument('--prepare_only', action='store_true', help='只校验权重并读取/缓存完整数据、生成配置，不执行训练')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args(argv)


def lr_for_update(update_index, total_updates, peak_lr, warmup_ratio, min_lr_ratio):
    warmup = int(total_updates * warmup_ratio)
    if warmup and update_index <= warmup:
        return peak_lr * update_index / warmup
    progress = max(0.0, min(1.0, (update_index - warmup) / max(1, total_updates - warmup)))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


class TextDataset(PretrainDataset):
    def __init__(self, samples, tokenizer, max_length):
        # 复用原项目完全相同的BOS/EOS、截断、padding和labels处理。
        self.samples, self.tokenizer, self.max_length = samples, tokenizer, max_length


def read_samples(args):
    if args.train_samples:
        rows = []
        with open(args.data_path, encoding='utf-8-sig') as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get('text'), str) or not row['text'].strip():
                    raise ValueError(f'数据第{line_no}行缺少非空text字符串')
                rows.append({'text': row['text']})
                if len(rows) >= args.train_samples + args.validation_samples:
                    break
        samples = datasets.Dataset.from_list(rows)
    else:
        samples = datasets.load_dataset('json', data_files=args.data_path, split='train')
    if len(samples) <= args.validation_samples or 'text' not in samples.column_names:
        raise ValueError('数据不足或缺少text列；必须至少留有一条训练样本')
    for row in samples.select(range(min(16, len(samples)))):
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError('数据text必须为非空字符串')
    val = samples.select(range(args.validation_samples))
    train = samples.select(range(args.validation_samples, len(samples)))
    return train, val


def amp_context(args):
    if args.device.startswith('cuda') and args.dtype != 'float32':
        return torch.autocast('cuda', dtype=getattr(torch, args.dtype))
    return nullcontext()


def evaluate(model, loader, args):
    was_training = model.training
    total_nll, tokens = 0.0, 0
    model.eval()
    try:
        with torch.inference_mode():
            for inputs, labels in loader:
                n = int((labels[:, 1:] != -100).sum())
                with amp_context(args):
                    loss = model(inputs.to(args.device), labels=labels.to(args.device)).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('验证loss为非有限值')
                total_nll += float(loss) * n
                tokens += n
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError('验证集没有可评估token')
    nll = total_nll / tokens
    return dict(val_nll=nll, val_ppl=math.exp(nll) if nll < 700 else None, val_tokens=tokens)


def export_weights(model, path):
    atomic_torch_save(path, {k: v.detach().half().cpu() for k, v in model.state_dict().items()})


def main():
    args = parse_args()
    session = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    resume = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
        resume = torch.load(resume_path, map_location='cpu', weights_only=True)
        if resume.get('format_version') != 1:
            raise ValueError('resume必须为本脚本保存的完整训练断点，不能使用纯权重文件')
        provided = {v[2:].split('=')[0] for v in sys.argv[1:] if v.startswith('--')}
        runtime_keys = {'resume', 'stop_after_updates', 'prepare_only'}
        for key, value in resume['args'].items():
            if key in runtime_keys:
                continue
            if key in provided and key in ('log_interval', 'save_interval', 'eval_interval'):
                continue
            if key in provided and getattr(args, key) != value:
                raise ValueError(f'恢复时不能改变训练参数 {key}：原值{value}，新值{getattr(args, key)}')
            setattr(args, key, value)
        folder = resume_path.parent
    else:
        if args.smoke_test:
            args.train_samples = args.train_samples or 192
            args.validation_samples = min(args.validation_samples, 8)
            args.max_updates = args.max_updates or 3
            args.log_interval, args.eval_interval, args.save_interval = 1, 1, 2
        name = args.run_name or ('smoke_' if args.smoke_test else 'poetry_gujilab_') + session
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('run_name只能是目录名')
        folder = Path(args.save_dir).resolve() / name
        folder.mkdir(parents=True, exist_ok=False)
    logger = make_logger(folder)
    writer = None
    started = time.perf_counter()
    state = dict(update=0, next_epoch=0, next_batch=0, total_samples=0, total_valid_tokens=0,
                 elapsed_seconds=0.0, best_val_nll=None)
    try:
        for key in ('epochs', 'batch_size', 'accumulation_steps', 'log_interval', 'eval_interval', 'validation_batch_size'):
            if getattr(args, key) < 1:
                raise ValueError(f'{key}必须>=1')
        if min(args.validation_samples, args.num_workers, args.train_samples, args.max_updates, args.stop_after_updates, args.save_interval) < 0:
            raise ValueError('样本数、worker、步数、保存间隔不能为负')
        if args.max_seq_len < 4 or args.learning_rate <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
            raise ValueError('检查序列长度、学习率、梯度裁剪和weight_decay')
        if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1:
            raise ValueError('warmup_ratio必须在[0,1)，min_lr_ratio在[0,1]')
        cuda = args.device.startswith('cuda')
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA不可用')
            torch.cuda.set_device(torch.device(args.device))
            if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
                raise RuntimeError('GPU不支持bfloat16，请改float16或float32')
        args.init_weight_path = str(Path(args.init_weight_path).resolve())
        args.data_path = str(Path(args.data_path).resolve())
        source = file_info(args.init_weight_path, checksum=True)
        data_info = file_info(args.data_path)
        suffix = '_moe' if args.use_moe else ''
        weight_path = folder / f'{args.save_weight}_{args.hidden_size}{suffix}.pth'
        if Path(args.save_weight).name != args.save_weight or weight_path.resolve() == Path(args.init_weight_path):
            raise ValueError('保存名称非法或会覆盖基础权重')
        if resume and (source != resume['source_weight'] or data_info != resume['data_file']):
            raise ValueError('原权重或数据文件已变化，拒绝按原游标续训')
        setup_seed(args.seed)
        config = MindChenConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
        model, tokenizer = init_model(config, 'none', device=args.device)
        weights = resume['model'] if resume else torch.load(args.init_weight_path, map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        del weights
        logger.info('严格加载权重成功 | %s | 输出: %s', args.init_weight_path if not resume else args.resume, folder)
        logger.info('读取数据；首次完整运行可能需要构建HuggingFace Arrow缓存。冒烟测试只读取前N条。')
        train_rows, val_rows = read_samples(args)
        train_ds = TextDataset(train_rows, tokenizer, args.max_seq_len)
        val_ds = TextDataset(val_rows, tokenizer, args.max_seq_len)
        batches_per_epoch = math.ceil(len(train_ds) / args.batch_size)
        full_updates = math.ceil(batches_per_epoch / args.accumulation_steps) * args.epochs
        total_updates = min(full_updates, args.max_updates) if args.max_updates else full_updates
        if resume and total_updates != resume['total_updates']:
            raise ValueError('数据长度或计划更新数与断点不符')
        metadata = dict(args=vars(args), source_weight=source, data_file=data_info,
                        train_samples=len(train_ds), validation_samples=len(val_ds),
                        validation_note='新阶段固定留出；可能与基础模型预训练数据重合，不是独立泛化测试',
                        effective_batch_size=args.batch_size * args.accumulation_steps,
                        batches_per_epoch=batches_per_epoch, total_updates=total_updates,
                        model_config=config.to_dict(), python=sys.version, executable=sys.executable,
                        torch=torch.__version__, transformers=transformers.__version__, datasets=datasets.__version__,
                        gpu=torch.cuda.get_device_name() if cuda else None,
                        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                        script_sha256=sha256_file(__file__),
                        truncation_note='继承原预训练处理；max_seq_len包含BOS/EOS，超长样本仅取前max_seq_len-2 token')
        atomic_json(folder / ('run_config.json' if not resume else f'resume_config_{session}.json'), metadata)
        validation_file = folder / 'validation_samples.jsonl'
        if not resume:
            with validation_file.open('w', encoding='utf-8') as f:
                for row in val_rows:
                    f.write(json.dumps({'text': row['text']}, ensure_ascii=False) + '\n')
        if args.prepare_only:
            atomic_json(folder / 'summary.json', dict(status='prepared', train_samples=len(train_ds),
                        validation_samples=len(val_ds), total_updates=total_updates,
                        source_weight_sha256=source['sha256'], optimizer_updates=0))
            logger.info('准备完成，未执行训练 | 训练=%d 验证=%d 计划更新=%d | 配置=%s',
                        len(train_ds), len(val_ds), total_updates, folder / 'run_config.json')
            return
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=cuda and args.dtype == 'float16')
        if resume:
            optimizer.load_state_dict(resume['optimizer'])
            scaler.load_state_dict(resume['scaler'])
            state.update(resume['progress'])
            restore_rng(resume['rng'])
            rewind_metrics(folder, state['update'], session)
            logger.info('恢复optimizer/scaler/RNG | update=%d | 下一epoch=%d batch=%d', state['update'], state['next_epoch'] + 1, state['next_batch'] + 1)
            del resume
            gc.collect()
        writer = MetricsWriter(folder)
        val_loader = DataLoader(val_ds, batch_size=args.validation_batch_size, num_workers=0,
                                generator=torch.Generator().manual_seed(args.seed), pin_memory=cuda)
        logger.info('训练=%d 验证=%d | batch=%d 累积=%d 有效batch=%d | 计划更新=%d | lr=%g dtype=%s',
                    len(train_ds), len(val_ds), args.batch_size, args.accumulation_steps,
                    metadata['effective_batch_size'], total_updates, args.learning_rate, args.dtype)
        requested_stop = [False]
        def stop_handler(signum, frame):
            requested_stop[0] = True
            logger.info('收到停止请求：完成当前更新后保存断点并退出。')
        previous_signal = signal.signal(signal.SIGINT, stop_handler)
        prior_elapsed = state['elapsed_seconds']
        train_started = time.perf_counter()
        first_update = state['update']
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        current_epoch = state['next_epoch']

        def record(phase, **values):
            elapsed = prior_elapsed + time.perf_counter() - train_started
            return dict(timestamp=datetime.now().isoformat(timespec='seconds'), session=session, phase=phase,
                        epoch=current_epoch + 1, batch=state['next_batch'], update=state['update'],
                        total_updates=total_updates, total_samples=state['total_samples'],
                        total_valid_tokens=state['total_valid_tokens'], elapsed_seconds=round(elapsed, 3),
                        gpu_allocated_gb=round(torch.cuda.memory_allocated() / 2**30, 3) if cuda else 0,
                        gpu_reserved_gb=round(torch.cuda.memory_reserved() / 2**30, 3) if cuda else 0,
                        gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else 0,
                        best_val_nll=state['best_val_nll'], **values)

        def validate():
            metrics = evaluate(model, val_loader, args)
            improved = state['best_val_nll'] is None or metrics['val_nll'] < state['best_val_nll']
            if improved:
                state['best_val_nll'] = metrics['val_nll']
                export_weights(model, folder / f'{args.save_weight}_{args.hidden_size}{suffix}_best.pth')
            writer.write(record('validation', **metrics))
            logger.info('验证 update=%d | NLL=%.5f PPL=%.3f token=%d | best=%.5f',
                        state['update'], metrics['val_nll'], metrics['val_ppl'] or float('inf'), metrics['val_tokens'], state['best_val_nll'])

        last_saved = -1
        def save():
            nonlocal last_saved
            if last_saved == state['update']:
                return
            state['elapsed_seconds'] = prior_elapsed + time.perf_counter() - train_started
            export_weights(model, weight_path)
            atomic_torch_save(folder / 'latest_resume.pth', dict(format_version=1, args=vars(args),
                model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), progress=state.copy(),
                rng=rng_state(), source_weight=source, data_file=data_info, total_updates=total_updates))
            last_saved = state['update']
            logger.info('保存 update=%d | 推理权重=%s | 完整断点=latest_resume.pth', state['update'], weight_path.name)

        if not state['update'] and len(val_ds):
            validate()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        stop = state['update'] >= total_updates
        for epoch in range(state['next_epoch'], args.epochs):
            if stop:
                break
            current_epoch = epoch
            skip = state['next_batch'] if epoch == state['next_epoch'] else 0
            sampler = EpochBatchSampler(len(train_ds), args.batch_size, args.seed + epoch, skip)
            loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=cuda,
                                generator=torch.Generator().manual_seed(args.seed + epoch))
            iterator, batch_cursor = iter(loader), skip
            while True:
                microbatches = list(itertools.islice(iterator, args.accumulation_steps))
                if not microbatches:
                    break
                tick = time.perf_counter()
                counts = [int((labels[:, 1:] != -100).sum()) for _, labels in microbatches]
                tokens = sum(counts)
                if not tokens:
                    raise ValueError('训练窗口没有有效token')
                lm_sum, aux_sum, samples = 0.0, 0.0, 0
                for (inputs, labels), n in zip(microbatches, counts):
                    with amp_context(args):
                        result = model(inputs.to(args.device, non_blocking=cuda), labels=labels.to(args.device, non_blocking=cuda))
                        aux = result.aux_loss if result.aux_loss is not None else result.loss.new_zeros(())
                        loss = (result.loss + aux) * (n / tokens)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'非有限loss: epoch={epoch+1}, batch={batch_cursor+1}')
                    scaler.scale(loss).backward()
                    lm_sum += float(result.loss.detach()) * n
                    aux_sum += float(aux.detach()) * n
                    samples += inputs.shape[0]
                    del result, loss
                update = state['update'] + 1
                lr = lr_for_update(update, total_updates, args.learning_rate, args.warmup_ratio, args.min_lr_ratio)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if cuda:
                    torch.cuda.synchronize()
                step_seconds = time.perf_counter() - tick
                batch_cursor += len(microbatches)
                state.update(update=update, next_epoch=epoch + 1 if batch_cursor == batches_per_epoch else epoch,
                             next_batch=0 if batch_cursor == batches_per_epoch else batch_cursor,
                             total_samples=state['total_samples'] + samples,
                             total_valid_tokens=state['total_valid_tokens'] + tokens)
                elapsed_session = time.perf_counter() - train_started
                eta = elapsed_session / max(1, update - first_update) * (total_updates - update)
                row = record('train', lm_loss=lm_sum/tokens, aux_loss=aux_sum/tokens,
                             total_loss=(lm_sum+aux_sum)/tokens, learning_rate=lr, grad_norm=grad_norm,
                             samples=samples, valid_tokens=tokens, tokens_per_second=tokens/max(step_seconds, 1e-9), eta_seconds=eta)
                row.update(epoch=epoch+1, batch=batch_cursor)
                writer.write(row)
                if update == 1 or update % args.log_interval == 0 or update == total_updates:
                    logger.info('epoch=%d/%d batch=%d/%d update=%d/%d | LM=%.5f aux=%.6f loss=%.5f | lr=%.8f grad=%.3f | %.0f tok/s GPU=%.2fGB ETA=%.1fh',
                        epoch+1, args.epochs, batch_cursor, batches_per_epoch, update, total_updates,
                        row['lm_loss'], row['aux_loss'], row['total_loss'], lr, grad_norm,
                        row['tokens_per_second'], row['gpu_peak_gb'], eta/3600)
                stop = requested_stop[0] or update >= total_updates or (args.stop_after_updates > 0 and update >= args.stop_after_updates)
                if len(val_ds) and (update % args.eval_interval == 0 or stop or batch_cursor == batches_per_epoch):
                    validate()
                if stop or batch_cursor == batches_per_epoch or (args.save_interval and update % args.save_interval == 0):
                    save()
                if stop:
                    break
        save()
        signal.signal(signal.SIGINT, previous_signal)
        status = 'completed' if state['update'] >= total_updates else 'stopped'
        atomic_json(folder / 'summary.json', dict(status=status, **state, total_updates=total_updates,
                    source_weight_sha256=source['sha256'], output_weight=str(weight_path),
                    resume_checkpoint=str(folder / 'latest_resume.pth'), wall_seconds=time.perf_counter()-started))
        logger.info('%s | update=%d/%d | 输出目录=%s', status, state['update'], total_updates, folder)
    except BaseException as exc:
        logger.exception('训练未完成；可从最近的latest_resume.pth恢复。')
        atomic_json(folder / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}', progress=state,
                    timestamp=datetime.now().isoformat(timespec='seconds')))
        raise
    finally:
        if writer:
            writer.close()


if __name__ == '__main__':
    main()
