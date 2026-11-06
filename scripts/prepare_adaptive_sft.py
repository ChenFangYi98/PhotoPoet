# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""全库流式建立 SFT 分类偏移索引，并新增可由程序核验的确定性问答。

不复制14GB基础库。按作品/提示哈希划出内部开发集，并保护原valid作品组。
任务分类是可审计弱标注，不伪称自动事实核验。默认完整扫描所有输入行。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trainer.adaptive_sft_data import INDEX_DTYPE, TASKS, classify_prompt, prompt_of, work_group, digest
from trainer.continue_pretrain_utils import atomic_json, file_info, sha256_file
import numpy as np


def numeric_records(train_groups=2048, valid_groups=256):
    rng = random.Random(20261026)
    seen, counts = set(), Counter()
    while counts['train'] < train_groups or counts['valid'] < valid_groups:
        mode = rng.randrange(4)
        if mode < 3:
            a, b = rng.randrange(1, 100), rng.randrange(1, 100)
            op = ('+', '-', '×')[mode]
            if mode == 1 and a < b: a, b = b, a
            if mode == 2: a, b = a % 13, b % 13
            if mode != 1 and a > b: a, b = b, a
            expression = f'{a}{op}{b}'
            answer = str(a+b if mode==0 else a-b if mode==1 else a*b)
            prompts = [f'计算 {expression}，只输出整数结果。', f'{expression} 等于多少？只回答数字。']
            group = digest('arithmetic\0'+expression)
        else:
            value = rng.randrange(1, 500)
            source, target, factor = rng.choice([('千米','米',1000),('米','厘米',100),('千克','克',1000),('小时','分钟',60),('分钟','秒',60)])
            answer = str(value*factor)
            prompts = [f'{value}{source}等于多少{target}？只输出整数数值。',
                       f'将{value}{source}换算为{target}，只回答数值，不带单位。']
            group = digest(f'unit\0{value}\0{source}\0{target}')
        if group in seen: continue
        seen.add(group)
        split = 'valid' if int.from_bytes(group[:2], 'little') % 10 == 0 else 'train'
        if counts[split] >= (valid_groups if split=='valid' else train_groups): continue
        counts[split] += 1
        for prompt in prompts:
            yield split, dict(conversations=[dict(role='user',content=prompt),dict(role='assistant',content=answer)]), group


def prepare(base, out, max_rows=0):
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    original = json.loads((base/'clean_report.json').read_text(encoding='utf-8'))
    fact_path = base/'factual_v1/manifest.json'
    fact_manifest = json.loads(fact_path.read_text(encoding='utf-8'))
    if fact_manifest['status'] != 'validated': raise ValueError('先验证 factual_v1')
    sources, writers, counts, rejected = [], {}, Counter(), Counter()

    def add_source(path, expected=None):
        item = file_info(path)
        if expected: item['expected_sha256'] = expected
        sources.append(item)
        return len(sources)-1

    def write_index(split, channel, task, confidence, reliability, group, source, offset, size):
        key = f'{split}_{channel}_{TASKS[task]}'
        if key not in writers: writers[key] = (out/(key+'.bin')).open('wb')
        record = np.array([(offset,size,source,task,confidence,reliability,group)],dtype=INDEX_DTYPE)
        writers[key].write(record.tobytes())
        counts[key] += 1

    # 先生成确定性增补。全部答案可由四则运算/固定单位规则独立核验。
    numeric_files = {s:(out/f'verified_{s}.jsonl').open('wb') for s in ('train','valid')}
    for split,row,group in numeric_records():
        numeric_files[split].write((json.dumps(dict(**row, fact_group=group.hex()),ensure_ascii=False)+'\n').encode())
    for stream in numeric_files.values(): stream.close()

    protected_groups, protected_prompts = set(), set()
    # 原验证文件以及新增作者验证集中的作品都禁止进入训练。
    base_valid = base/'sft_t2t_with_poetry_valid.jsonl'
    for path, meta_path in [(base_valid,None),(base/'factual_v1/facts_valid.jsonl',base/'factual_v1/facts_valid.meta.jsonl'),(out/'verified_valid.jsonl',None)]:
        meta_stream = meta_path.open(encoding='utf-8') if meta_path else None
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                meta = json.loads(next(meta_stream)) if meta_stream else None
                group = bytes.fromhex(row['fact_group']) if 'fact_group' in row else work_group(row,meta)
                protected_groups.add(group); protected_prompts.add(digest(prompt_of(row)))
        if meta_stream: meta_stream.close()

    jobs = [(base/f'sft_t2t_with_poetry_{s}.jsonl',s,'base',None,
             original['outputs'][s]['sha256']) for s in ('valid','train')]
    jobs += [(base/f'factual_v1/facts_{s}.jsonl',s,'author',base/f'factual_v1/facts_{s}.meta.jsonl',
              fact_manifest['files'][f'facts_{s}.jsonl']['sha256']) for s in ('valid','train')]
    jobs += [(out/f'verified_{s}.jsonl',s,'verified',None,None) for s in ('valid','train')]
    source_counts = {}
    try:
        for path,original_split,channel,meta_path,expected in jobs:
            if meta_path and sha256_file(meta_path)!=fact_manifest['files'][meta_path.name]['sha256']:
                raise ValueError('作者元数据校验失败：'+str(meta_path))
            source_id = add_source(path, expected)
            sha, offset, raw_count = hashlib.sha256(), 0, 0
            metadata = meta_path.open(encoding='utf-8') if meta_path else None
            with path.open('rb') as stream:
                for line in stream:
                    sha.update(line); raw_count += 1
                    row = json.loads(line)
                    meta = json.loads(next(metadata)) if metadata else None
                    prompt = prompt_of(row)
                    if not any(m.get('role')=='assistant' and (m.get('content','').strip() or m.get('tool_calls') or m.get('reasoning_content')) for m in row.get('conversations',[])):
                        rejected['empty_prompt_or_answer'] += 1; offset += len(line); continue
                    group = bytes.fromhex(row['fact_group']) if 'fact_group' in row else work_group(row,meta)
                    if original_split=='train' and (group in protected_groups or digest(prompt) in protected_prompts):
                        rejected['protected_validation_group_or_prompt'] += 1; offset += len(line); continue
                    task,confidence,reliability = classify_prompt(prompt)
                    # 多轮任务可能变化：只看首段prompt无法判断未来轮，保守不训练分类头。
                    # LM仍保留所有assistant回答、reasoning和tool call监督。
                    if sum(m.get('role')=='user' for m in row['conversations'])>1:
                        task,confidence,reliability=2,0.,0.
                    if channel=='author':
                        task = 2 if meta['task']=='clarify_ambiguity' else 0
                        confidence,reliability = 1., (meta['confidence'] if task==0 else 0.)
                    elif channel=='verified': task,confidence,reliability = 0,1.,1.
                    # 基础库按作品/提示组留出3%作内部开发；原valid不回流。
                    split = 'valid' if original_split=='valid' or (channel in ('base','author') and int.from_bytes(group[:2],'little')%100<3) else 'train'
                    write_index(split,channel,task,confidence,reliability,group,source_id,offset,len(line))
                    offset += len(line)
                    if raw_count % 500000 == 0:
                        print(f'{channel} {original_split}: {raw_count:,} rows',flush=True)
                    if max_rows and channel=='base' and raw_count>=max_rows: break
            if metadata: metadata.close()
            if not max_rows and expected and sha.hexdigest()!=expected: raise ValueError('源文件SHA256不符：'+str(path))
            sources[source_id]['sha256'] = sha.hexdigest()
            source_counts[str(path)] = raw_count
            print(f'finished {channel} {original_split}: {raw_count:,}',flush=True)
    finally:
        for stream in writers.values(): stream.close()
    indices = {key:dict(file=key+'.bin',count=counts[key],sha256=sha256_file(out/(key+'.bin'))) for key in writers}
    by_task = {split:{task:sum(n for key,n in counts.items() if key.startswith(split+'_') and key.endswith('_'+task)) for task in TASKS} for split in ('train','valid')}
    masses=np.array([by_task['train'][task] for task in TASKS],dtype=np.float64)**.35
    task_ratios=(masses/masses.sum()).tolist()
    manifest = dict(version='adaptive_sft_v2',status='ready',sources=sources,indices=indices,
        indexed_counts=by_task,source_rows=source_counts,rejected=dict(rejected),complete_scan=not bool(max_rows),
        factual_manifest_sha256=sha256_file(fact_path),index_dtype=INDEX_DTYPE.descr,
        recommended_task_ratios=task_ratios,sampling_temperature=.35,fact_source_ratios=dict(author=.45,verified=.20,base=.35),
        label_scope='Prompt-only weak task labels, not truth verification; corpus reliability is a heuristic, not a probability.',
        split_scope='Original valid works/prompts protected; base train internal dev grouped by work or exact prompt. Not proof of globally unseen pretraining facts.',
        seconds=time.perf_counter()-started)
    atomic_json(out/'manifest.json',manifest)
    print(json.dumps(dict(status='ready',counts=by_task,rejected=dict(rejected),seconds=manifest['seconds']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_dir',type=Path,default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1')
    parser.add_argument('--output_dir',type=Path,default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1/adaptive_v2_ready')
    parser.add_argument('--max_rows',type=int,default=0,help='仅供调试；0扫描全部输入')
    args=parser.parse_args();prepare(args.base_dir.resolve(),args.output_dir.resolve(),args.max_rows)
