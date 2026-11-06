# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""新增6万条确定性训练数据，合并为新版本索引，保留v2及原始基础库。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
from transformers import AutoTokenizer
from trainer.adaptive_sft_data import INDEX_DTYPE,IndexedCorpus,digest,prompt_of
from trainer.continue_pretrain_utils import atomic_json,file_info,sha256_file
from scripts.deterministic_tasks import TARGETS,problem,solve


def build(base,out):
    corpus=IndexedCorpus(base/'adaptive_v2_ready/manifest.json')
    folder=base/'deterministic_v3'
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    train_groups=set(bytes(g) for k,r in corpus.indices.items() if k.startswith('train_') for g in np.unique(r['group']))
    valid_groups=set(bytes(g) for k,r in corpus.indices.items() if k.startswith('valid_') for g in np.unique(r['group']))
    seed=json.loads((ROOT/'dataset/factual_seed_sources.json').read_text(encoding='utf-8'))
    books={'train':[],'valid':[]}
    for source in seed['sources']:
        for title,_ in source['works']:
            group=digest(title+'\0'+source['author']).rstrip(b'\0')
            split='train' if group in train_groups else 'valid' if group in valid_groups else None
            if split:books[split].append(dict(title=title,author=source['author'],url=source['url']))
    counts=Counter();seen_prompts=set();seen_groups=set();streams={s:(folder/f'deterministic_{s}.jsonl').open('x',encoding='utf-8') for s in ('train','valid')}
    def emit(split,kind,prompt,answer,group,meta,reliability=1.):
        ph=digest(prompt);g=group.rstrip(b'\0')
        forbidden=valid_groups if split=='train' else train_groups
        if g in forbidden or ph in forbidden or ph in seen_prompts:return False
        if len(tokenizer.encode(answer,add_special_tokens=False))>24:return False
        row=dict(conversations=[dict(role='user',content=prompt),dict(role='assistant',content=answer)],
            fact_group=group.hex(),task_meta=dict(kind=kind,task=0,label_confidence=1.,reliability=reliability,**meta))
        streams[split].write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        seen_prompts.add(ph);counts[split+'_'+kind]+=1
        return True
    try:
        selected=Counter()
        with (folder/'literature_candidates.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                rec=json.loads(line);split=rec['split']
                if selected[split]>=(12000 if split=='train' else 1200):continue
                author,next_line=rec['author'],rec['next_line']
                if max(len(tokenizer.encode(a,add_special_tokens=False)) for a in (author,next_line))>24:continue
                title,clue=rec['title'],rec['first_line'];group=bytes.fromhex(rec['fact_group'])
                p1=f'作品《{title}》开篇为“{clue}”。请指出作者，仅输出姓名。'
                p2=f'请按已有作品原文回答：{author}《{title}》的第一行为“{clue}”。紧接的下一行是什么？只输出下一行原文。'
                if any(digest(p) in seen_prompts for p in (p1,p2)):continue
                meta=dict(title=title,author=author,first_line=clue,next_line=next_line,source=rec['source'],quality='corpus_metadata')
                a=emit(split,'literature_author',p1,author,group,meta,.6)
                b=emit(split,'literature_next_line',p2,next_line,group,meta,.6)
                if not a or not b:raise ValueError('文学样本配对失败')
                selected[split]+=1
        if selected!={'train':12000,'valid':1200}:raise ValueError('文学候选不足：'+str(selected))
        print('literature works',dict(selected),flush=True)
        rng=random.Random(20261026)
        for kind,limit in TARGETS.items():
            attempts=0
            while counts['train_'+kind]<limit or counts['valid_'+kind]<limit//10:
                attempts+=1
                if attempts>limit*1000:raise ValueError('生成空间不足：'+kind)
                requested='valid' if rng.randrange(10)==0 else 'train'
                prompt,answer,key,meta=problem(kind,rng,books[requested])
                group=digest(key);g=group.rstrip(b'\0')
                split='train' if g in train_groups else 'valid' if g in valid_groups else ('valid' if int.from_bytes(group[:2],'little')%10==0 else 'train')
                # 书目训练/验证使用互不相交的作品池，允许同一资料组合的不同排列。
                if kind=='book_evidence':split=requested
                if counts[split+'_'+kind]>=(limit if split=='train' else limit//10) or (group in seen_groups and kind!='book_evidence'):continue
                if solve(kind,prompt)!=answer:raise ValueError('独立求解不一致：'+prompt)
                if emit(split,kind,prompt,answer,group,dict(quality='seed_grounded' if kind=='book_evidence' else 'program_verified',**meta)):
                    seen_groups.add(group)
            print(kind,counts['train_'+kind],counts['valid_'+kind],flush=True)
    finally:
        for stream in streams.values():stream.close()
    report=dict(counts=dict(counts),train_rows=sum(v for k,v in counts.items() if k.startswith('train_')),
        valid_rows=sum(v for k,v in counts.items() if k.startswith('valid_')),seed_sha256=sha256_file(ROOT/'dataset/factual_seed_sources.json'),
        scope='Literature local metadata/exact text; books source-grounded, not new independently verified facts; remaining tasks program verifiable.')
    atomic_json(folder/'build_report.json',report)
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(corpus.path.read_text(encoding='utf-8'))
    # 复用已校验完整基础索引，只重建verified通道；保留旧数据与旧清单。
    for key,info in manifest['indices'].items():
        if '_verified_' not in key:shutil.copyfile(corpus.path.parent/info['file'],out/info['file'])
    for split in ('train','valid'):
        combined=out/f'verified_{split}.jsonl';idx=out/f'{split}_verified_fact.bin'
        source_id=len(manifest['sources']);offset=0;count=0
        with combined.open('wb') as target,idx.open('wb') as index:
            for path,legacy in [(corpus.path.parent/f'verified_{split}.jsonl',True),(folder/f'deterministic_{split}.jsonl',False)]:
                with path.open('rb') as stream:
                    for line in stream:
                        row=json.loads(line);meta=row.get('task_meta',{});reliability=1. if legacy else meta['reliability']
                        group=bytes.fromhex(row['fact_group'])
                        target.write(line)
                        index.write(np.array([(offset,len(line),source_id,0,1.,reliability,group)],dtype=INDEX_DTYPE).tobytes())
                        offset+=len(line);count+=1
        manifest['sources'].append(dict(**file_info(combined),sha256=sha256_file(combined)))
        key=f'{split}_verified_fact'
        manifest['indices'][key]=dict(file=idx.name,count=count,sha256=sha256_file(idx))
    manifest['indices']={k:dict(info,sha256=sha256_file(out/info['file'])) for k,info in manifest['indices'].items()}
    manifest['indexed_counts']={s:{t:sum(i['count'] for k,i in manifest['indices'].items() if k.startswith(s+'_') and k.endswith('_'+t)) for t in ('fact','open','general')} for s in ('train','valid')}
    masses=[manifest['indexed_counts']['train'][t]**.35 for t in ('fact','open','general')]
    manifest['recommended_task_ratios']=[x/sum(masses) for x in masses]
    manifest.update(version='adaptive_sft_v3',parent_manifest_sha256=sha256_file(corpus.path),
        deterministic_additions=report,addition_manifest=str((folder/'build_report.json').resolve()))
    atomic_json(out/'manifest.json',manifest)
    corpus.close()
    print(json.dumps(dict(train=report['train_rows'],valid=report['valid_rows'],sampling=manifest['recommended_task_ratios']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base_dir',type=Path,default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1')
    p.add_argument('--output_dir',type=Path,default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready')
    args=p.parse_args();build(args.base_dir.resolve(),args.output_dir.resolve())
