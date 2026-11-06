# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""完整核验新增确定性数据、文学来源、监督长度和新索引的训练/开发分割。"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
from transformers import AutoTokenizer
from trainer.adaptive_sft_data import IndexedCorpus
from trainer.sft_data import AssistantDataset
from trainer.continue_pretrain_utils import atomic_json,sha256_file
from scripts.clean_sft_dataset import contains_old_name
from scripts.deterministic_tasks import solve
from scripts.validate_adaptive_sft import numeric_gold


def validate(base):
    folder=base/'deterministic_v3';index_folder=base/'adaptive_v3_ready'
    corpus=IndexedCorpus(index_folder/'manifest.json')
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    encoder=AssistantDataset([],tokenizer,2048)
    train=np.unique(np.concatenate([r['group'] for k,r in corpus.indices.items() if k.startswith('train_') and len(r)]))
    valid=np.unique(np.concatenate([r['group'] for k,r in corpus.indices.items() if k.startswith('valid_') and len(r)]))
    if np.intersect1d(train,valid,assume_unique=True).size:raise ValueError('全索引训练/验证组相交')
    seed=json.loads((ROOT/'dataset/factual_seed_sources.json').read_text(encoding='utf-8'))
    allowed={(title,s['author'],s['url']) for s in seed['sources'] for title,_ in s['works']}
    seen=set();refs=defaultdict(dict);counts={};books={s:set() for s in ('train','valid')};new_groups={};max_answer_tokens=0
    previews=[]
    try:
        for split in ('train','valid'):
            stats=Counter();offset=0
            rows=corpus.indices[f'{split}_verified_fact']
            with (index_folder/f'verified_{split}.jsonl').open('rb') as stream:
                for index,line in enumerate(stream):
                    item=rows[index];row=json.loads(line)
                    assert int(item['offset'])==offset and int(item['size'])==len(line)
                    assert bytes(item['group'])==bytes.fromhex(row['fact_group']).rstrip(b'\0')
                    offset+=len(line)
                    prompt=row['conversations'][0]['content'];answer=row['conversations'][-1]['content']
                    if 'task_meta' not in row:
                        assert numeric_gold(prompt)==answer
                        stats['legacy_numeric']+=1;continue
                    meta=row['task_meta'];kind=meta['kind']
                    assert meta['task']==0 and meta['label_confidence']==1.
                    assert abs(float(item['reliability'])-meta['reliability'])<1e-6
                    # 来源路径中的 mindchen-master 是项目名，不是模型自我身份。
                    assert not contains_old_name(json.dumps(row['conversations'],ensure_ascii=False))
                    ph=hashlib.sha256(prompt.encode()).digest()
                    assert ph not in seen,'增补提示重复'
                    seen.add(ph)
                    g=row['fact_group']
                    assert g not in new_groups or new_groups[g]==split,'增补分组跨split'
                    new_groups[g]=split
                    if kind.startswith('literature_'):
                        source=meta['source'];refs[source['file']][source['line']]=meta
                        if kind=='literature_author':
                            assert prompt==f'作品《{meta["title"]}》开篇为“{meta["first_line"]}”。请指出作者，仅输出姓名。'
                            assert answer==meta['author']
                        else:
                            assert prompt==f'请按已有作品原文回答：{meta["author"]}《{meta["title"]}》的第一行为“{meta["first_line"]}”。紧接的下一行是什么？只输出下一行原文。'
                            assert answer==meta['next_line']
                        assert meta['quality']=='corpus_metadata' and meta['reliability']==.6
                    else:
                        assert solve(kind,prompt)==answer,'独立求解不一致'
                        assert meta['reliability']==1.
                        if kind=='book_evidence':
                            for book in meta['book_sources']:
                                assert (book['title'],book['author'],book['url']) in allowed
                                assert f'《{book["title"]}》作者：{book["author"]}' in prompt
                                books[split].add(book['title'])
                    answer_tokens=len(tokenizer.encode(answer,add_special_tokens=False))
                    assert answer_tokens<=24
                    max_answer_tokens=max(max_answer_tokens,answer_tokens)
                    feature=encoder.encode(row)
                    assert not feature['truncated'] and feature['target_tokens']>0 and feature['eos_supervised']==1
                    stats[kind]+=1
                    if kind not in {p['kind'] for p in previews}:previews.append(dict(kind=kind,split=split,prompt=prompt,answer=answer))
            assert index+1==len(rows)
            counts[split]=dict(stats)
            print(split,dict(stats),flush=True)
        assert not books['train']&books['valid'],'书目证据的作品池跨分割'
        checked=0
        parent=json.loads((base/'adaptive_v2_ready/manifest.json').read_text(encoding='utf-8'))
        allowed_files={s['path']:s['sha256'] for s in parent['sources'] if Path(s['path']).name in ('sft_t2t_with_poetry_train.jsonl','sft_t2t_with_poetry_valid.jsonl')}
        for path,needed in refs.items():
            assert path in allowed_files,'来源不在基础库内'
            sha=hashlib.sha256();found=0
            with open(path,'rb') as stream:
                for number,line in enumerate(stream,1):
                    sha.update(line)
                    if number not in needed:continue
                    meta=needed[number];row=json.loads(line);body=row['conversations'][1]['content']
                    assert hashlib.sha256(body.encode()).hexdigest()==meta['source']['body_sha256']
                    assert '\n作者：'+meta['author'] in row['conversations'][0]['content']
                    assert row['conversations'][0]['content'].splitlines()[1][1:-1].split(' / ',1)[-1].strip()==meta['title']
                    lines=body.splitlines()
                    assert lines[0]==meta['first_line'] and lines[1]==meta['next_line']
                    found+=1
            assert found==len(needed) and sha.hexdigest()==allowed_files[path]
            checked+=found
            print('source references checked',path,found,flush=True)
        report=dict(status='passed',counts=counts,new_train_rows=sum(v for k,v in counts['train'].items() if k!='legacy_numeric'),
            new_valid_rows=sum(v for k,v in counts['valid'].items() if k!='legacy_numeric'),source_relations_checked=checked,
            full_index_group_overlap=0,new_prompt_duplicates=0,book_evidence_work_overlap=0,max_answer_tokens=max_answer_tokens,
            complete_assistant_supervision=True,base_hashes_unchanged=True,
            manifest_sha256=sha256_file(index_folder/'manifest.json'),
            scope='Program/text/source consistency verification; local corpus attribution is not independently proven historical truth; no claim of unseen pretraining knowledge.')
        atomic_json(folder/'validation_report.json',report);atomic_json(folder/'preview.json',previews)
        atomic_json(index_folder/'validation_report.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:corpus.close()


if __name__=='__main__':validate(ROOT/'dataset/sft_t2t_with_poetry_clean_v1')
