# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""为确定性增补提取更多文学关系，保持现有训练/开发作品归属。"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.build_factual_sft import collect
from trainer.adaptive_sft_data import IndexedCorpus,work_group
from trainer.continue_pretrain_utils import atomic_json


def collect_more(base,out):
    out.mkdir(parents=True,exist_ok=False)
    corpus=IndexedCorpus(base/'adaptive_v2_ready/manifest.json')
    train=set(bytes(g) for k,r in corpus.indices.items() if k.startswith('train_') for g in np.unique(r['group']))
    valid=set(bytes(g) for k,r in corpus.indices.items() if k.startswith('valid_') for g in np.unique(r['group']))
    old=set()
    for split in ('train','valid'):
        with (base/f'factual_v1/facts_{split}.meta.jsonl').open(encoding='utf-8') as stream:
            old.update(json.loads(line)['id'] for line in stream)
    records,_,clues,sources=collect(base)
    expected=json.loads((base/'clean_report.json').read_text(encoding='utf-8'))
    for split,source in sources.items():
        if source['sha256']!=expected['outputs'][split]['sha256']:raise ValueError('基础库改变')
    counts={'train':0,'valid':0};rejected={}
    with (out/'literature_candidates.jsonl').open('w',encoding='utf-8') as target:
        for ident,record in sorted(records.items()):
            if ident in old or not record['body'] or len(record['splits'])!=1:continue
            clue=record.get('clue','')
            if not 10<=len(clue)<=120 or len(clues[clue])!=1:continue
            lines=record['body'].splitlines()
            if len(lines)<2 or not lines[1].strip() or len(lines[1])>40 or len(record['author'])>12:continue
            group=work_group({},record).rstrip(b'\0')
            split='train' if group in train else 'valid' if group in valid else None
            if split is None or counts[split]>=(24000 if split=='train' else 3000):continue
            value={k:v for k,v in record.items() if k not in ('splits',)}
            value.update(split=split,first_line=lines[0],next_line=lines[1],fact_group=work_group({},record).hex())
            target.write(json.dumps(value,ensure_ascii=False)+'\n');counts[split]+=1
    corpus.close()
    atomic_json(out/'literature_catalog_report.json',dict(counts=counts,sources=sources,
        scope='Local metadata and exact source text; not independently verified historical attribution. Existing factual_v1 work ids excluded.'))
    print(counts,flush=True)


if __name__=='__main__':
    collect_more(ROOT/'dataset/sft_t2t_with_poetry_clean_v1',ROOT/'dataset/sft_t2t_with_poetry_clean_v1/deterministic_v3')
