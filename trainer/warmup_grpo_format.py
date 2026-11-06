# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Bounded format/EOS warmup on a separate SFT-best copy, with general retention checks."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import datasets  # Windows DLL ordering
import torch
from transformers import AutoTokenizer
from model.model_mindchen import MindChenForCausalLM
from model.checkpoint_config import load_checkpoint_config
from trainer.grpo_resources import (atomic_json, atomic_save, file_digest, protect_source, snapshot_source,
                                    project_path, write_checkpoint_config)
from trainer.grpo_control import StopController, StopRequested, live_state
from trainer.grpo_data_quality import prompt_issue
from trainer.grpo_poetry_reward import poetry_rule_score
from trainer.sft_data import AssistantDataset, SFTCollator
from trainer.adaptive_sft_data import IndexedCorpus
from trainer.continue_pretrain_utils import rng_state, restore_rng
from trainer.train_grpo import amp_context

FORMAT = 'poetry_format_warmup_v2_balanced'


def balanced_probes(rows, count):
    if count < 4 or count % 4:
        raise ValueError('Probe count must be a positive multiple of four')
    result = []
    for shape in ((4, 5), (4, 7), (8, 5), (8, 7)):
        group = [r for r in rows if (r['spec']['lines'], r['spec']['width']) == shape]
        group.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        if len(group) < count//4:
            raise ValueError(f'Insufficient independent probes for {shape}')
        result.extend(group[:count//4])
    return result


def quality_gate(metrics, baseline, min_format=.5, min_restore=.25, general_tolerance=.1):
    return (metrics['format_pass_rate'] >= min_format and metrics['restore_exact_rate'] >= min_restore
            and metrics['general_nll'] <= baseline['general_nll']+general_tolerance)


def repair_pair(row):
    spec = row['poetry_spec']
    body = spec['reference_body']
    lines = [body[i:i+spec['width']] for i in range(0, len(body), spec['width'])]
    answer = '\n'.join(v+('，' if i % 2 == 0 else '。') for i, v in enumerate(lines))
    if len(lines) != spec['lines']:
        raise ValueError('Invalid reference body')
    return dict(conversations=row['messages']+[dict(role='assistant', content=answer)],
                id=row['id'], spec=spec, reference_body=body)


def build_data(tokenizer, archive_workspace, max_length=768, retention_cache=None):
    poetry = {}
    for split in ('train', 'valid'):
        path = ROOT/f'dataset/grpo_poetry_v2/{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        poetry[split] = [repair_pair(row) for row in rows if row['poetry_spec'].get('task') == 'format']
    bodies = [{r['reference_body'] for r in poetry[s]} for s in ('train', 'valid')]
    if bodies[0] & bodies[1]:
        raise ValueError('Warmup poem train/validation overlap')
    encoder = AssistantDataset([], tokenizer, max_length)
    def encode(row):
        item = encoder.encode(row)
        if item['truncated'] or not item['target_tokens'] or not item['eos_supervised']:
            return None
        item.pop('rendered', None)
        return item
    features = {s: [encode(r) for r in poetry[s]] for s in ('train', 'valid')}
    if any(x is None for group in features.values() for x in group):
        raise ValueError('Poetry warmup would lose target tokens or EOS')
    if retention_cache:
        cache=project_path(retention_cache)
        manifest=json.loads((cache/'run_config.json').read_text(encoding='utf-8'))
        general,general_rows={},{}
        for split,needed in [('train',768),('valid',128)]:
            path=cache/f'{split}.jsonl'
            if file_digest(path)!=manifest['data_sha256'][split]:
                raise ValueError('Retention cache hash mismatch')
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            rows=[row for row in rows if not row.get('spec')]
            if len(rows)<needed:raise ValueError('Insufficient verified retention examples')
            general_rows[split]=rows[:needed]
            general[split]=[encode(row) for row in rows[:needed]]
            if any(item is None for item in general[split]):raise ValueError('Retention cache would lose target/EOS')
        return poetry,features,general,general_rows
    corpus = IndexedCorpus(ROOT/'dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready/manifest.json')
    # Read the already verified SSD archive, without changing the original manifest.
    corpus.sources = [dict(s, path=str(archive_workspace/Path(s['path']).relative_to(ROOT))) for s in corpus.sources]
    general, general_rows = {}, {}
    rng = random.Random(712)
    try:
        for split, needed in [('train', 768), ('valid', 128)]:
            key = split+'_base_general'; selected, rows = [], []
            for index in rng.sample(range(len(corpus.indices[key])), min(len(corpus.indices[key]), needed*20)):
                row, _ = corpus.read(key, index)
                messages = row.get('conversations', [])
                users = [m for m in messages if m.get('role') == 'user']
                if not users or any('\ufffd' in m.get('content', '') for m in messages):
                    continue
                if prompt_issue([users[-1]]):
                    continue
                item = encode(row)
                if item is not None:
                    selected.append(item); rows.append(row)
                if len(selected) == needed:
                    break
            if len(selected) != needed:
                raise ValueError('Insufficient short, complete general retention examples')
            general[split], general_rows[split] = selected, rows
    finally:
        corpus.close()
    return poetry, features, general, general_rows


@torch.no_grad()
def measure(model, tokenizer, features, general, probes, device, control=None):
    model.eval(); collate = SFTCollator(tokenizer.pad_token_id)
    result = {}
    for name, examples in [('poetry', features[:128]), ('general', general[:128])]:
        total, count = 0., 0
        for start in range(0, len(examples), 4):
            if control is not None: control.check()
            batch = collate(examples[start:start+4]); labels = batch['labels'].to(device)
            with amp_context(device, 'bfloat16'):
                output = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=labels)
            n = int(labels[:, 1:].ne(-100).sum()); total += float(output.loss)*n; count += n
        result[name+'_nll'] = total/count
    predictions = []
    for index, row in enumerate(probes, 1):
        if control is not None: control.check()
        print(f'[warmup probe] {index}/{len(probes)} {row["id"]}', flush=True)
        prompt = tokenizer.apply_chat_template(row['conversations'][:-1], tokenize=False,
                                              add_generation_prompt=True, open_thinking=False)
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
        with amp_context(device, 'bfloat16'):
            ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                max_new_tokens=256, do_sample=False, temperature=1., top_k=0, top_p=1.,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(ids[0, inputs.input_ids.size(1):], skip_special_tokens=True).strip()
        _, parts = poetry_rule_score(row['conversations'][:-1], text, row['spec'])
        ended = bool(ids[0, inputs.input_ids.size(1):].eq(tokenizer.eos_token_id).any())
        predictions.append(dict(id=row['id'], response=text, ended=ended,
            format_exact=parts.get('format_exact', 0), exact=text == row['conversations'][-1]['content']))
    result.update(format_pass_rate=sum(p['format_exact'] for p in predictions)/len(predictions),
                  restore_exact_rate=sum(p['exact'] for p in predictions)/len(predictions), predictions=predictions)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=''); p.add_argument('--run_dir', default='')
    p.add_argument('--archive_workspace', default=''); p.add_argument('--updates', type=int, default=400)
    p.add_argument('--resume', default='')
    p.add_argument('--retention_cache',default='',help='哈希绑定的旧预热数据，仅复用通用训练/验证样本')
    p.add_argument('--eval_interval', type=int, default=50)
    p.add_argument('--save_interval', type=int, default=100)
    p.add_argument('--validation_probes', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=5e-6)
    p.add_argument('--min_format_rate', type=float, default=.5)
    p.add_argument('--min_restore_rate', type=float, default=.25)
    p.add_argument('--general_regression_tolerance', type=float, default=.1)
    p.add_argument('--stop_after_steps', type=int, default=0)
    p.add_argument('--device', default='cuda:0'); args = p.parse_args(argv)
    if (min(args.updates,args.eval_interval,args.save_interval) < 1 or args.validation_probes<4
        or args.validation_probes%4 or not 0 < args.learning_rate < 1
        or not 0 <= args.min_format_rate <= 1 or not 0 <= args.min_restore_rate <= 1
        or not 0 <= args.general_regression_tolerance <= 1 or args.stop_after_steps<0):
        p.error('Invalid update count, probes, learning rate or quality thresholds')
    signature = {name: file_digest(ROOT/name) for name in ['trainer/warmup_grpo_format.py',
        'trainer/sft_data.py', 'model/model_mindchen.py', 'model/checkpoint_config.py',
        'trainer/grpo_resources.py','trainer/grpo_io.py','trainer/grpo_control.py',
        'trainer/grpo_poetry_reward.py','trainer/grpo_repetition.py','trainer/train_grpo.py',
        'model/tokenizer.json','model/tokenizer_config.json','trainer/adaptive_sft_data.py',
        'trainer/grpo_data_quality.py','trainer/continue_pretrain_utils.py']}
    restored = None
    if args.resume:
        resume_path=project_path(args.resume)
        restored=torch.load(resume_path,map_location='cpu',weights_only=True)
        if restored.get('format') != FORMAT or restored['signature'] != signature:
            raise ValueError('Warmup resume code/protocol mismatch')
        device,stop_after=args.device,args.stop_after_steps
        args=argparse.Namespace(**restored['args']);args.device=device;args.stop_after_steps=stop_after
        folder=resume_path.parent;source=restored['source']
        snapshot=folder/'source_snapshot'/Path(source['snapshot']).name
        if file_digest(snapshot)!=source['sha256']:raise ValueError('Source snapshot changed')
        config,_=load_checkpoint_config(snapshot)
    else:
        if not all((args.checkpoint,args.run_dir,args.archive_workspace)):
            p.error('New run requires checkpoint, run_dir and archive_workspace')
        folder = project_path(args.run_dir)
        protect_source(folder,project_path(args.checkpoint))
        folder.mkdir(parents=True, exist_ok=False)
        snapshot, config, source = snapshot_source(project_path(args.checkpoint), folder)
    protect_source(folder, source['path'])
    print(f'[warmup source] {source["path"]}\n[snapshot sha256] {source["sha256"]}',flush=True)
    torch.set_num_threads(4); torch.manual_seed(42); random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'model', local_files_only=True)
    if restored:
        encoder=AssistantDataset([],tokenizer,768)
        poems,features,general={},{},{}
        for split in ('train','valid'):
            path=folder/f'{split}.jsonl'
            if file_digest(path)!=restored['data_sha256'][split]:raise ValueError('Warmup data changed')
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            poems[split]=[r for r in rows if r.get('spec')]
            features[split]=[encoder.encode(r) for r in poems[split]]
            general[split]=[encoder.encode(r) for r in rows if not r.get('spec')]
    else:
        print('[warmup data] encoding real restoration targets and verified general retention examples',flush=True)
        poems, features, general, general_rows = build_data(tokenizer, project_path(args.archive_workspace),
                                                          retention_cache=args.retention_cache or None)
        for split in ('train', 'valid'):
            with (folder/f'{split}.jsonl').open('w', encoding='utf-8') as stream:
                for row in poems[split]+general_rows[split]:
                    stream.write(json.dumps(dict(row,split=split), ensure_ascii=False)+'\n')
    probes=balanced_probes(poems['valid'],args.validation_probes)
    data_sha256={s:file_digest(folder/f'{s}.jsonl') for s in ('train', 'valid')}
    if not restored:
        for name in signature:
            target=folder/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        atomic_json(folder/'run_config.json', dict(args=vars(args), source=source, model_config=config.to_dict(),
            signature=signature, train_poems=len(poems['train']), valid_poems=len(poems['valid']),
            train_general=len(general['train']), valid_general=len(general['valid']), data_sha256=data_sha256,
            validation_probe_ids=[r['id'] for r in probes],
            note='Real restoration targets, shape-balanced updates, independent probes and general retention. No fake original poems.'))
    model = MindChenForCausalLM(config)
    model.load_state_dict(torch.load(snapshot, map_location='cpu', weights_only=True), strict=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    collate = SFTCollator(tokenizer.pad_token_id)
    progress = dict(update=0, total_updates=args.updates, best_rank=[-1., -1., -float('inf')], best_update=0)
    latest, best = folder/'format_warmup.pth', folder/'format_warmup_best.pth'
    baseline = None; best_metrics = None
    if restored:
        model.load_state_dict(restored['model'],strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        progress=restored['progress'];baseline=restored['baseline'];best_metrics=restored['best_metrics']
        restore_rng(restored['rng']);del restored
    first_update=progress['update']
    shape_groups={shape:[feature for row,feature in zip(poems['train'],features['train'])
                        if (row['spec']['lines'],row['spec']['width'])==shape]
                  for shape in ((4,5),(4,7),(8,5),(8,7))}
    if any(not group for group in shape_groups.values()):raise ValueError('All four forms need training targets')
    def save(status):
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        atomic_save(latest, state); write_checkpoint_config(latest, config)
        atomic_save(folder/'resume.pth', dict(format=FORMAT, model=state,
            optimizer=optimizer.state_dict(), progress=progress, args=vars(args), rng=rng_state(),
            source=source, signature=signature, baseline=baseline, best_metrics=best_metrics, data_sha256=data_sha256))
        atomic_json(folder/'summary.json', dict(status=status, **progress, baseline=baseline,
            best_metrics=best_metrics, best_weight=str(best), source=source,
            quality_gate_passed=bool(best_metrics and quality_gate(best_metrics,baseline,
                args.min_format_rate,args.min_restore_rate,args.general_regression_tolerance))))
        live_state(folder, status, progress)
    def evaluate():
        nonlocal baseline, best_metrics
        live_state(folder,'validating',progress)
        metrics = measure(model, tokenizer, features['valid'], general['valid'], probes, args.device,control)
        if baseline is None: baseline = metrics
        rank = [metrics['restore_exact_rate'], metrics['format_pass_rate'], -metrics['poetry_nll']]
        accepted = metrics['general_nll'] <= baseline['general_nll']+args.general_regression_tolerance and rank > progress['best_rank']
        if accepted:
            progress.update(best_rank=rank, best_update=progress['update']); best_metrics=metrics
            atomic_save(best, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            write_checkpoint_config(best, config)
        atomic_json(folder/f'validation_{progress["update"]:06d}.json', dict(**metrics, accepted=accepted))
        print('[warmup validation] '+json.dumps(dict(update=progress['update'], accepted=accepted,
            **{k:v for k,v in metrics.items() if k!='predictions'})), flush=True)
    started=time.monotonic()
    with StopController(folder) as control:
        try:
            if progress['update']==0:evaluate()
            print(f'[warmup start] {progress["update"]}/{args.updates}; probes={len(probes)}',flush=True)
            for update in range(progress['update']+1, args.updates+1):
                control.check(); model.train(); optimizer.zero_grad(set_to_none=True)
                rng=random.Random(42+update)
                examples=[rng.choice(group) for group in shape_groups.values()]+rng.sample(general['train'],4)
                # Equal task mass, rather than letting longer general answers dominate.
                denominators=[sum(e['target_tokens'] for e in examples[:4]),
                              sum(e['target_tokens'] for e in examples[4:])]
                loss_value=0.
                for start in range(0,8,2):
                    batch=collate(examples[start:start+2]); labels=batch['labels'].to(args.device)
                    n=int(labels[:,1:].ne(-100).sum())
                    with amp_context(args.device,'bfloat16'):
                        output=model(input_ids=batch['input_ids'].to(args.device),attention_mask=batch['attention_mask'].to(args.device),labels=labels)
                        loss=(output.loss+output.aux_loss)*(.5*n/denominators[start//4])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite format warmup loss')
                    loss.backward();loss_value+=float(loss.detach())
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                optimizer.step(); progress['update']=update
                row=dict(update=update,loss=loss_value,grad_norm=norm,seconds=time.monotonic()-started)
                with (folder/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                live_state(folder,'training',progress,last_metrics=row)
                if update==1 or update%10==0:print('[warmup train] '+json.dumps(row),flush=True)
                stop=args.stop_after_steps and update-first_update>=args.stop_after_steps
                if update%args.eval_interval==0 or update==args.updates:evaluate()
                if update==1 or update%args.save_interval==0 or update==args.updates or stop:
                    final_status=('completed' if quality_gate(best_metrics,baseline,args.min_format_rate,
                        args.min_restore_rate,args.general_regression_tolerance) else 'quality_gate_failed') if update==args.updates else 'running'
                    save(final_status if update==args.updates else ('stopped' if stop else 'running'))
                if stop:return
            if best_metrics is None or not quality_gate(best_metrics,baseline,args.min_format_rate,
                    args.min_restore_rate,args.general_regression_tolerance):
                raise RuntimeError('Expanded held-out format/general quality gate failed; GRPO must not start')
        except StopRequested:
            save('stopped');return
        except BaseException as exc:
            live_state(folder,'failed',progress,error=repr(exc));raise
    print('[warmup ready] '+str(best),flush=True)


if __name__=='__main__':main()
