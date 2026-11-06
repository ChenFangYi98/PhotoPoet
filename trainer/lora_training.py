# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Bounded native-checkpoint LoRA SFT shared by the standalone LLM/VLM projects.

Training never imports the old random identity/system augmentation. Resume files
bind adapters, optimizer, scaler, RNG and dataset position to an immutable base.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import datasets  # Load Arrow before torch on Windows.
import torch
from PIL import Image
from transformers import AutoTokenizer

from model.checkpoint_compat import SUPPORTED_CONFIG_SCHEMAS
from model.lora_adapter import attach_lora, adapter_state, load_adapter, atomic_save, save_merged

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def project_path(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def load_native(checkpoint, kind, device='cpu', vision_model=None):
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Use a standalone native model .pth, not a training resume/adapter file')
    embed = state['model.embed_tokens.weight']
    layers = sorted({int(k.split('.')[2]) for k in state if k.startswith('model.layers.')})
    inferred = dict(hidden_size=embed.shape[1], vocab_size=embed.shape[0], num_hidden_layers=len(layers),
                    use_moe=any('.experts.' in k for k in state))
    if layers != list(range(len(layers))):
        raise ValueError('Checkpoint has non-contiguous language layers')
    config_values = dict(inferred)
    sidecar = Path(str(checkpoint) + '.config.json')
    if sidecar.exists():
        saved = json.loads(sidecar.read_text(encoding='utf-8-sig'))
        if saved.get('schema') not in SUPPORTED_CONFIG_SCHEMAS or saved.get('weight_sha256') != sha256(checkpoint):
            raise ValueError('Checkpoint/config checksum mismatch')
        config_values.update(saved['model_config'])
        for key, value in inferred.items():
            if config_values.get(key) != value:
                raise ValueError(f'Checkpoint/config architecture mismatch: {key}')
    # These routing settings cannot be inferred from bare tensor dimensions.
    config_values.setdefault('norm_topk_prob', True)
    if inferred['use_moe']:
        experts = {int(k.split('.experts.')[1].split('.')[0]) for k in state if '.experts.' in k}
        config_values.setdefault('num_experts', len(experts))
        config_values.setdefault('moe_intermediate_size', state['model.layers.0.mlp.experts.0.gate_proj.weight'].shape[0])
    else:
        config_values.setdefault('intermediate_size', state['model.layers.0.mlp.gate_proj.weight'].shape[0])
    config_values.pop('model_type', None)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    processor = None
    if kind == 'vlm':
        from model.model_vlm import MindChenVLM, VLMConfig
        model = MindChenVLM(VLMConfig(**config_values), vision_model_path=str(vision_model or ROOT / 'model/siglip2-base-p32-256-ve'))
        if model.vision_encoder is None or model.processor is None:
            raise FileNotFoundError('Local vision encoder/processor could not be loaded')
        processor = model.processor
        incompatible = model.load_state_dict(state, strict=False)
        missing = [k for k in incompatible.missing_keys if not k.startswith('vision_encoder.')]
        if missing or incompatible.unexpected_keys:
            raise ValueError(f'VLM state mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}')
    else:
        from model.model_mindchen import MindChenForCausalLM, MindChenConfig
        model = MindChenForCausalLM(MindChenConfig(**config_values))
        model.load_state_dict(state, strict=True)
    del state
    return model.to(device), tokenizer, processor


class ChatData:
    def __init__(self, path, tokenizer, config, max_length=256, kind='llm', processor=None, limit=2000):
        self.path, self.tokenizer, self.config = Path(path), tokenizer, config
        self.max_length, self.kind, self.processor = max_length, kind, processor
        self.rows = []
        if self.path.suffix.lower() == '.parquet':
            import pyarrow.parquet as pq
            source = pq.ParquetFile(self.path)
            columns = [n for n in ('conversations', 'image_bytes', 'image', 'images') if n in source.schema_arrow.names]
            for batch in source.iter_batches(batch_size=64, columns=columns):
                self.rows.extend(batch.to_pylist()[:max(0, limit - len(self.rows))])
                if len(self.rows) >= limit: break
        else:
            with self.path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if line.strip(): self.rows.append(json.loads(line))
                    if len(self.rows) >= limit: break
        if not self.rows: raise ValueError(f'Empty dataset: {self.path}')
        self.begin = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
        self.end = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self): return len(self.rows)

    def fingerprint(self):
        digest = hashlib.sha256()
        for row in self.rows:
            record = dict(row)
            if record.get('image_bytes'):
                record['image_bytes'] = hashlib.sha256(record['image_bytes']).hexdigest()
            paths = record.get('images') or ([record['image']] if record.get('image') else [])
            if paths:
                record['resolved_image_hashes'] = []
                for name in paths:
                    p = Path(name); p = p if p.is_absolute() else self.path.parent/p
                    record['resolved_image_hashes'].append(sha256(p))
            digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode('utf8'))
        return digest.hexdigest()

    def __getitem__(self, index):
        row = self.rows[index]
        messages = row['conversations']
        if isinstance(messages, str): messages = json.loads(messages)
        messages = [dict(m) for m in messages]
        image = None
        paths = row.get('images') or ([row['image']] if row.get('image') else [])
        has_image = bool(row.get('image_bytes')) or bool(paths)
        if has_image and self.kind != 'vlm': raise ValueError('Image rows require the VLM entry point')
        if has_image:
            if len(paths) > 1: raise ValueError('This LoRA collator supports one image per sample')
            if row.get('image_bytes'):
                from io import BytesIO
                image = Image.open(BytesIO(row['image_bytes'])).convert('RGB')
            else:
                p = Path(paths[0]); p = p if p.is_absolute() else self.path.parent / p
                with Image.open(p) as opened: image = opened.convert('RGB')
            count = sum(m['content'].count('<image>') for m in messages)
            if count == 0:
                user = next(m for m in messages if m['role'] == 'user')
                user['content'] = '<image>\n' + user['content']
            elif count != 1: raise ValueError('Expected one <image> marker for a single image')
            marker = self.config.image_special_token * self.config.image_token_len
            for message in messages: message['content'] = message['content'].replace('<image>', marker)
        elif any('<image>' in m['content'] for m in messages):
            raise ValueError('Image marker without an image')
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, open_thinking=False)
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[:self.max_length]
        if has_image and ids.count(self.config.image_ids[0]) != self.config.image_token_len:
            raise ValueError(f'Image markers were truncated at row {index}; increase max_seq_len')
        labels = [-100] * len(ids)
        cursor = 0
        while cursor < len(ids):
            if ids[cursor:cursor + len(self.begin)] != self.begin:
                cursor += 1; continue
            start = cursor + len(self.begin); end = start
            while end < len(ids) and ids[end:end + len(self.end)] != self.end: end += 1
            stop = min(len(ids), end + len(self.end))
            labels[start:stop] = ids[start:stop]; cursor = stop
        if not any(v != -100 for v in labels[1:]):
            raise ValueError(f'No supervised assistant tokens at row {index}; check data/max_seq_len')
        pixels = self.processor(images=image, return_tensors='pt') if image is not None else None
        return dict(input_ids=ids, labels=labels, pixel_values=pixels)


def collate(items, tokenizer, device):
    width = max(len(x['input_ids']) for x in items)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch = dict(input_ids=torch.tensor([x['input_ids'] + [pad] * (width - len(x['input_ids'])) for x in items], device=device),
                 labels=torch.tensor([x['labels'] + [-100] * (width - len(x['labels'])) for x in items], device=device),
                 attention_mask=torch.tensor([[1] * len(x['input_ids']) + [0] * (width - len(x['input_ids'])) for x in items], device=device))
    exemplar = next((x['pixel_values'] for x in items if x['pixel_values'] is not None), None)
    if exemplar is not None:
        batch['pixel_values'] = {k: torch.cat([x['pixel_values'][k] if x['pixel_values'] is not None else torch.zeros_like(v)
                                             for x in items], dim=0).to(device) for k, v in exemplar.items()}
    return batch


def evaluate_loss(model, data, tokenizer, device, batch_size):
    model.eval(); numerator = denominator = 0
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            batch = collate([data[i] for i in range(start, min(start + batch_size, len(data)))], tokenizer, device)
            result = model(**batch)
            count = int((batch['labels'][:, 1:] != -100).sum())
            numerator += float(result.loss) * count; denominator += count
    return numerator / max(1, denominator)


def parse_args(kind):
    parser = argparse.ArgumentParser(description=f'Bounded {kind.upper()} LoRA SFT; outputs stay in project out/')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', required=True, help='JSONL conversations (+ image for VLM) or Parquet')
    parser.add_argument('--eval_data_path')
    parser.add_argument('--run_dir', help='New directory under project out/; auto-generated by default')
    parser.add_argument('--resume', help='resume.pth in an existing LoRA run')
    parser.add_argument('--max_steps', type=int, default=100, help='Total optimizer updates, including resumed updates')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--max_train_samples', type=int, default=2000)
    parser.add_argument('--max_eval_samples', type=int, default=128)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--target_modules', default='q_proj,v_proj')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument('--eval_interval', type=int, default=25)
    parser.add_argument('--save_interval', type=int, default=25)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vision_model', default=str(ROOT / 'model/siglip2-base-p32-256-ve'))
    parser.add_argument('--export_merged', action='store_true')
    args = parser.parse_args()
    for key in ('max_steps', 'batch_size', 'accumulation_steps', 'max_seq_len', 'max_train_samples', 'max_eval_samples', 'eval_interval', 'save_interval'):
        if getattr(args, key) < 1: parser.error(f'{key} must be positive')
    if args.learning_rate <= 0: parser.error('learning_rate must be positive')
    return args


def main(kind):
    args = parse_args(kind)
    if int(__import__('os').environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Short LoRA entry supports one process; launch with python, not torchrun')
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    checkpoint, data_path = project_path(args.checkpoint), project_path(args.data_path)
    evaluation = project_path(args.eval_data_path) if args.eval_data_path else None
    run = project_path(args.run_dir) if args.run_dir else ROOT / 'out/lora_runs' / (kind + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    resume_path = project_path(args.resume) if args.resume else None
    if resume_path and not args.run_dir: run = resume_path.parent
    if not run.resolve().is_relative_to((ROOT / 'out').resolve()):
        raise ValueError('Training outputs must stay under this project out/')
    if run.exists() and not resume_path: raise FileExistsError('Choose a new run_dir, or explicitly resume')
    if resume_path and run.resolve() != resume_path.parent: raise ValueError('Resume must use its original run directory')
    run.mkdir(parents=True, exist_ok=True)
    targets = tuple(s.strip() for s in args.target_modules.split(',') if s.strip())
    tracked = [Path(__file__), ROOT/'model/lora_adapter.py', ROOT/'model/model_mindchen.py',
               ROOT/'model/tokenizer.json', ROOT/'model/tokenizer_config.json', Path(sys.argv[0]).resolve()]
    if kind == 'vlm': tracked.append(ROOT/'model/model_vlm.py')
    if Path(str(checkpoint) + '.config.json').exists(): tracked.append(Path(str(checkpoint) + '.config.json'))
    def file_identity(p):
        if p is None: return None
        stat=p.stat()
        return dict(path=str(p),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    signature = dict(kind=kind, base_sha256=sha256(checkpoint), data_source=file_identity(data_path),
                     eval_source=file_identity(evaluation),
                     sources={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p) for p in tracked},
                     training={k: getattr(args, k) for k in ('batch_size','accumulation_steps','max_seq_len','max_train_samples','max_eval_samples','rank','alpha','dropout','target_modules','learning_rate','seed','dtype')})
    if kind == 'vlm':
        vision = project_path(args.vision_model)
        signature['vision'] = {p.name: sha256(p) for p in sorted(vision.iterdir()) if p.is_file()}
    model, tokenizer, processor = load_native(checkpoint, kind, args.device, project_path(args.vision_model))
    names = attach_lora(model, args.rank, args.alpha, args.dropout, targets)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda') and args.dtype == 'float16')
    train = ChatData(data_path, tokenizer, model.config, args.max_seq_len, kind, processor, args.max_train_samples)
    valid = ChatData(evaluation, tokenizer, model.config, args.max_seq_len, kind, processor, args.max_eval_samples) if evaluation else None
    signature['selected_train_sha256'] = train.fingerprint()
    signature['selected_eval_sha256'] = valid.fingerprint() if valid else None
    info = dict(schema='native_lora_v1', kind=kind, base_checkpoint=str(checkpoint), signature=signature,
                lora=dict(rank=args.rank, alpha=args.alpha, dropout=args.dropout, targets=list(targets), modules=names),
                model_config=model.config.to_dict(), trainable_parameters=sum(p.numel() for p in params),
                train_rows=len(train), eval_rows=len(valid) if valid else 0)
    epoch = cursor = step = 0; best = float('inf')
    if resume_path:
        restored = torch.load(resume_path, map_location='cpu', weights_only=True)
        if restored['signature'] != signature: raise ValueError('Resume base/data/source/training signature mismatch')
        load_adapter(model, restored); optimizer.load_state_dict(restored['optimizer']); scaler.load_state_dict(restored['scaler'])
        epoch, cursor, step, best = (restored[k] for k in ('epoch','cursor','step','best_eval_loss'))
        random.setstate(restored['random_state']); torch.set_rng_state(restored['torch_rng'])
        if args.device.startswith('cuda'): torch.cuda.set_rng_state_all(restored['cuda_rng'])
    (run/'run.json').write_text(json.dumps(dict(**info, args=vars(args)), ensure_ascii=False, indent=2), encoding='utf8')
    if not resume_path:
        import shutil
        snapshot=run/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for p in tracked:
            target=snapshot/(p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('base_config')/p.name)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    print(f'LoRA {kind}: {len(names)} modules, {info["trainable_parameters"]:,} trainable parameters; {len(train)} rows; run={run}', flush=True)
    def payload(): return dict(**info, adapter=adapter_state(model))
    def save():
        atomic_save(payload(), run/'adapter.pth')
        atomic_save(dict(**payload(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         epoch=epoch, cursor=cursor, step=step, best_eval_loss=best,
                         random_state=random.getstate(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else []), run/'resume.pth')
    started = time.monotonic()
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
    while step < args.max_steps:
        model.train()
        if kind == 'vlm': model.vision_encoder.eval()
        optimizer.zero_grad(set_to_none=True); total = 0
        for _ in range(args.accumulation_steps):
            if cursor >= len(order):
                epoch += 1; cursor = 0
                order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
            indices = order[cursor:cursor + args.batch_size]; cursor += len(indices)
            batch = collate([train[i] for i in indices], tokenizer, args.device)
            dtype = getattr(torch, args.dtype)
            context = torch.autocast('cuda', dtype=dtype) if args.device.startswith('cuda') and dtype != torch.float32 else nullcontext()
            with context:
                result = model(**batch); loss = result.loss + result.aux_loss
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss; reduce learning rate or inspect data')
            total += float(loss.detach()); scaler.scale(loss / args.accumulation_steps).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        previous_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < previous_scale:
            raise FloatingPointError('FP16 overflow: optimizer step was skipped; retry with bfloat16/float32')
        step += 1
        metric = dict(step=step, loss=total / args.accumulation_steps, seconds=time.monotonic() - started)
        if valid and (step % args.eval_interval == 0 or step == args.max_steps):
            metric['eval_loss'] = evaluate_loss(model, valid, tokenizer, args.device, args.batch_size)
            if metric['eval_loss'] < best:
                best = metric['eval_loss']; atomic_save(payload(), run/'best_loss_adapter.pth')
        with (run/'metrics.jsonl').open('a', encoding='utf8') as stream: stream.write(json.dumps(metric) + '\n')
        print(json.dumps(metric), flush=True)
        if step % args.save_interval == 0 or step == args.max_steps: save()
    if args.export_merged: save_merged(model, run/f'merged_step_{step:06}.pth')
    print(f'Complete: {step} optimizer steps; adapter and resume saved in {run}', flush=True)
