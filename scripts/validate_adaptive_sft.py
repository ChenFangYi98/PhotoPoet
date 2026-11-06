# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""完整索引分割校验、增补算术独立核验和分层窗口保留率审计。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
from transformers import AutoTokenizer
from trainer.adaptive_sft_data import IndexedCorpus,lossless_features
from trainer.continue_pretrain_utils import atomic_json


def numeric_gold(prompt):
    match=re.search(r'(\d+)\s*([+\-×])\s*(\d+)',prompt)
    if match:
        a,b=int(match[1]),int(match[3])
        return str({'+':lambda:a+b,'-':lambda:a-b,'×':lambda:a*b}[match[2]]())
    match=re.search(r'(\d+)(千米|千克|小时|分钟|米)',prompt)
    if not match:raise ValueError('无法独立验证数值题：'+prompt)
    factors={'千米':1000,'千克':1000,'小时':60,'分钟':60,'米':100}
    return str(int(match[1])*factors[match[2]])


def validate(manifest_path,samples_per_channel=24):
    corpus=IndexedCorpus(manifest_path)
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    rng=random.Random(42)
    train=np.unique(np.concatenate([rows['group'] for key,rows in corpus.indices.items() if key.startswith('train_') and len(rows)]))
    valid=np.unique(np.concatenate([rows['group'] for key,rows in corpus.indices.items() if key.startswith('valid_') and len(rows)]))
    overlap=np.intersect1d(train,valid,assume_unique=True)
    if len(overlap):raise ValueError(f'训练/验证组重叠 {len(overlap)}')
    report=dict(status='passed',train_valid_group_overlap=0,channels={},numeric_verified=0,
        scope='Full index hashes/group split check; stratified token audit only, not full-corpus tokenization or historical fact verification.')
    previews=[]
    try:
        for key,rows in corpus.indices.items():
            if not len(rows):continue
            stats=Counter()
            # 验证全部新增数值题，其余通道固定随机抽样窗口。
            indices=list(range(len(rows))) if '_verified_' in key else rng.sample(range(len(rows)),min(samples_per_channel,len(rows)))
            for index in indices:
                row,meta=corpus.read(key,index)
                if '_verified_' in key:
                    assert numeric_gold(row['conversations'][0]['content'])==row['conversations'][-1]['content']
                    report['numeric_verified']+=1
                feature=lossless_features(row,tokenizer)
                stats['samples']+=1;stats['windowed']+=feature['windowed']
                stats['targets']+=feature['target_tokens'];stats['retained_targets']+=sum(w['target_tokens'] for w in feature['windows'])
                stats['windows']+=len(feature['windows'])
                if len(previews)<12 and feature['windowed']:
                    previews.append(dict(channel=key,index=index,full_tokens=feature['full_tokens'],targets=feature['target_tokens'],
                        windows=[dict(start=w['start'],end=w['end'],targets=w['target_tokens'],eos=w['eos_supervised']) for w in feature['windows']]))
            assert stats['targets']==stats['retained_targets']
            report['channels'][key]=dict(stats)
            print(key,dict(stats),flush=True)
        report['audit_target_retention']=1.
        atomic_json(Path(manifest_path).parent/'validation_report.json',report)
        atomic_json(Path(manifest_path).parent/'window_previews.json',previews)
    finally:corpus.close()
    print(json.dumps({k:v for k,v in report.items() if k!='channels'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1/adaptive_v2_ready/manifest.json')
    p.add_argument('--samples_per_channel',type=int,default=24)
    args=p.parse_args();validate(args.manifest,args.samples_per_channel)
