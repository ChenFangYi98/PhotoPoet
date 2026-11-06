# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""独立重读清洗产物：哈希、去重、分割、旧身份残留、原文与工具保真，并抽样检查真实监督标签。"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.clean_sft_dataset import (
    answer_keys, conversation_key, contains_old_name, dump, identity_rows, is_poetry,
    RECITE, ZH_IDENTITY, EN_IDENTITY,
)


def length_audit(samples, max_length):
    # 复用实际训练的模板与角色范围，禁止用字符数冒充token统计。
    from trainer.sft_data import AssistantDataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'model'), local_files_only=True)
    report = {}
    for name, pairs in samples.items():
        results = {}
        for variant in ('before', 'after'):
            rows = [pair[variant] for pair in pairs]
            ds = AssistantDataset(rows, tokenizer, 10**9)
            counts = Counter()
            lengths = []
            for i in range(len(ds)):
                feature = ds[i]
                labels = feature['labels']
                retained = sum(v != -100 for v in labels[1:max_length])
                full_targets = sum(v != -100 for v in labels[1:])
                lengths.append(feature['full_tokens'])
                counts['samples'] += 1
                counts['truncated'] += feature['full_tokens'] > max_length
                counts['no_target_at_cap'] += retained == 0
                counts['assistant_tokens_total'] += full_targets
                counts['assistant_tokens_at_cap'] += retained
            lengths.sort()
            results[variant] = dict(counts, truncation_fraction=counts['truncated']/max(1,len(rows)),
                assistant_token_retention=counts['assistant_tokens_at_cap']/max(1,counts['assistant_tokens_total']),
                p50_tokens=lengths[len(lengths)//2] if lengths else 0,
                p95_tokens=lengths[min(len(lengths)-1,int(len(lengths)*.95))] if lengths else 0,
                max_tokens=max(lengths, default=0))
        report[name] = results
    return dict(max_seq_len=max_length, sampling='Deterministic reservoir over each retained output split, seed=42; not a full-corpus token scan.', splits=report)


def validate(folder, audit_samples=1024, max_length=1024, dedup_backend='memory'):
    started = time.perf_counter()
    report = json.loads((folder/'clean_report.json').read_text(encoding='utf-8'))
    if report['status'] not in ('cleaned_pending_independent_validation', 'validated'):
        raise ValueError('Cleaning is not complete')
    # 当前机器内存充足，SHA256集合约占1GB，可避免数百万次随机磁盘写。
    # 内存有限的机器可显式用sqlite；两个后端检查相同的完整SHA256，不使用概率去重。
    db, seen = None, set()
    if dedup_backend == 'sqlite':
        db_path = folder/'validation_index.sqlite'
        if db_path.exists():
            raise FileExistsError(f'Validation index already exists: {db_path}; preserve completed results and validate a new output directory')
        db = sqlite3.connect(db_path)
        db.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-131072; CREATE TABLE seen(key BLOB PRIMARY KEY) WITHOUT ROWID;')
    protected, valid_prompts = set(), set()
    results, samples = {}, {}
    identity_path = report.get('identity_source', str(ROOT/'dataset/lora_identity.jsonl'))
    identities = list(identity_rows(Path(identity_path))) if identity_path else []
    for split in ('valid', 'train'):
        path = folder/f'sft_t2t_with_poetry_{split}.jsonl'
        source_path = Path(report['sources'][split]['path'])
        assert source_path.stat().st_size == report['sources'][split]['bytes']
        assert source_path.stat().st_mtime_ns == report['sources'][split]['mtime_ns']
        counts, sha, reservoir, rng = Counter(), hashlib.sha256(), [], random.Random(42)
        source_line = 0
        with path.open('rb') as stream, (folder/f'provenance_{split}.jsonl').open(encoding='utf-8') as provenance, source_path.open('rb') as original:
            for index, pair in enumerate(itertools.zip_longest(stream, provenance), 1):
                line, metadata = pair
                assert line is not None and metadata is not None, 'Output/provenance length mismatch'
                sha.update(line)
                row, meta = json.loads(line), json.loads(metadata)
                assert meta['output_line'] == index-1
                assert meta['source'] in (split, 'identity')
                if meta['source'] == 'identity':
                    assert split == 'train'
                    before = identities[meta['input_line']-1]
                    counts['identity_source_rows'] += 1
                else:
                    assert meta['input_line'] > source_line, 'Source order changed or row repeated'
                    while source_line < meta['input_line']:
                        source_raw = next(original)
                        source_line += 1
                    before = json.loads(source_raw)
                assert not contains_old_name(line.decode('utf-8')), f'Legacy identity remains at {split}:{index}'
                messages = row['conversations']
                assert all(m['role'] in ('system','user','assistant','tool') and isinstance(m['content'],str) for m in messages)
                if is_poetry(before):
                    assert messages[0]['role'] == 'user' and len(messages) == 2
                    assert messages[1]['content'] == before['conversations'][2]['content'], 'Poetry body changed'
                    counts['exact_poetry_answers_checked'] += 1
                    if messages[0]['content'].startswith(RECITE):
                        assert '首句提示：' not in messages[0]['content']
                        counts['unhinted_recitations'] += 1
                else:
                    assert [m['role'] for m in messages] == [m['role'] for m in before['conversations']]
                    for a,b in zip(before['conversations'], messages):
                        for field in ('tools','tool_calls'):
                            assert a.get(field) == b.get(field), 'Tool structure changed'
                        if a['role'] == 'tool':
                            assert a == b, 'Tool return changed'
                            counts['tool_messages_checked'] += 1
                key = conversation_key(row)
                if db is not None:
                    assert db.execute('INSERT OR IGNORE INTO seen VALUES(?)',(key,)).rowcount == 1, 'Duplicate conversation'
                else:
                    assert key not in seen, 'Duplicate conversation'
                    seen.add(key)
                keys = answer_keys(row)
                prompt_key = hashlib.sha256(dump({'conversations': [m for m in messages if m['role'] != 'system'][:-1]}).encode('utf-8')).digest()
                if split == 'valid':
                    protected.update(keys)
                    valid_prompts.add(prompt_key)
                else:
                    assert not (keys & protected), 'Train/valid exact normalized assistant answer overlap'
                    assert prompt_key not in valid_prompts, 'Train/valid exact prompt overlap'
                counts['rows'] += 1
                sample = dict(before=before, after=row)
                if len(reservoir) < audit_samples:
                    reservoir.append(sample)
                else:
                    slot = rng.randrange(index)
                    if slot < audit_samples:
                        reservoir[slot] = sample
                if index % 250000 == 0:
                    if db is not None: db.commit()
                    print(f'validate {split}: {index:,} rows; elapsed={time.perf_counter()-started:.1f}s',flush=True)
        assert sha.hexdigest() == report['outputs'][split]['sha256'], 'Output SHA256 mismatch'
        assert counts['rows'] == report['outputs'][split]['counts']['output_rows']
        if db is not None: db.commit()
        results[split] = dict(counts, sha256=sha.hexdigest())
        samples[split] = reservoir
        print(f'validate {split}: {counts["rows"]:,} passed',flush=True)
    if db is not None:
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        db.close()
    seen.clear()
    # 先写全库结构检查结果；token审计若失败不会误报整体通过。
    validation = dict(status='structure_passed_token_audit_pending', splits=results,
        duplicate_conversations=0, train_valid_exact_answer_overlap=0, train_valid_exact_prompt_overlap=0, legacy_identity_hits=0,
        original_membership='Each retained row traces to same original split; no work group resplitting.',
        source_unchanged=True, dedup_backend=dedup_backend, seconds=time.perf_counter()-started)
    target = folder/'validation_report.json'
    target.write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    validation['token_audit'] = length_audit(samples,max_length)
    validation.update(status='passed',seconds=time.perf_counter()-started)
    target.write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    report['status'] = 'validated'
    report['validation_report'] = str(target.resolve())
    (folder/'clean_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir',type=Path,default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1')
    p.add_argument('--audit_samples',type=int,default=1024)
    p.add_argument('--max_seq_len',type=int,default=1024)
    p.add_argument('--dedup_backend',choices=('memory','sqlite'),default='memory')
    args=p.parse_args()
    validate(args.input_dir.resolve(),args.audit_samples,args.max_seq_len,args.dedup_backend)
