# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""独立核验新增事实通道：作品/提示分割、负例、来源行、全部实体标签与基础库哈希。"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.clean_sft_dataset import contains_old_name


def validate(folder):
    from trainer.factual_sft_utils import read_aligned,entity_feature
    from transformers import AutoTokenizer
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    seed_bytes=Path(manifest['seed_source']).read_bytes()
    assert hashlib.sha256(seed_bytes).hexdigest()==manifest['seed_sha256'],'种子来源清单发生改变'
    seed=json.loads(seed_bytes)
    reviewed={(title,source['author'],source['url']):source['aliases']
              for source in seed['sources'] for title,_ in source['works']}
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    worksets,questions,counts={},set(),{}
    refs=defaultdict(dict)
    for filename,info in manifest['files'].items():
        if hashlib.sha256((folder/filename).read_bytes()).hexdigest()!=info['sha256']:raise ValueError('新增文件哈希改变: '+filename)
    for split in ('valid','train'):
        rows,metas=read_aligned(folder/f'facts_{split}.jsonl',folder/f'facts_{split}.meta.jsonl')
        worksets[split]=set();stats=Counter()
        for row,meta in zip(rows,metas):
            prompt=row['conversations'][0]['content'];answer=row['conversations'][-1]['content']
            assert prompt not in questions,'新增train/valid提示重叠'
            questions.add(prompt)
            assert not contains_old_name(json.dumps(row,ensure_ascii=False))
            assert answer==meta['answer'] and answer in meta['aliases']
            negative=meta.get('negative_answer')
            assert negative not in meta['aliases']
            feature=entity_feature(row,answer,tokenizer,2048)
            entity=tokenizer.decode([v for v,flag in zip(feature['input_ids'],feature['entity_mask']) if flag],skip_special_tokens=False)
            assert entity==answer, f'实体字符/token边界不一致: {entity!r} != {answer!r}'
            assert not feature['truncated']
            worksets[split].add(meta['id'])
            stats['rows']+=1;stats[meta['quality']]+=1
            if meta['quality']=='corpus_metadata':
                source=meta['source']; refs[source['file']][source['line']]=meta
            elif meta['quality']=='externally_reviewed':
                assert meta['source']['url'].startswith(('https://www.dpm.org.cn/','https://www.luxunmuseum.cn/','https://www.chinawriter.com.cn/','https://wyb.chinawriter.com.cn/','https://www.chnmuseum.cn/','https://www.wxg.org.cn/'))
                relation=(meta['title'],meta['author'],meta['source']['url'])
                assert relation in reviewed and meta['aliases']==reviewed[relation],'外部关系与种子清单不符'
            else:assert len(meta['source']['authors'])>1
        counts[split]=dict(stats)
        print(f'validate factual {split}: {len(rows):,} entity targets passed',flush=True)
    assert not worksets['train']&worksets['valid'],'事实作品组跨split'
    source_checked=0
    for info in manifest['base'].values():
        path=Path(info['path']); expected=refs.get(str(path),{});sha=hashlib.sha256();found=0
        with path.open('rb') as stream:
            for line_no,line in enumerate(stream,1):
                sha.update(line)
                if line_no not in expected:continue
                meta=expected[line_no];row=json.loads(line)
                user,assistant=row['conversations']
                assert '\n作者：'+meta['author'] in user['content']
                assert hashlib.sha256(assistant['content'].encode()).hexdigest()==meta['source']['body_sha256']
                found+=1
        assert sha.hexdigest()==info['sha256'],'基础数据发生改变'
        assert found==len(expected),'来源行丢失'
        source_checked+=found
    probes,probe_meta=read_aligned(folder/'probe_seen.jsonl',folder/'probe_seen.meta.jsonl')
    for row,meta in zip(probes,probe_meta):
        assert meta['id'] in worksets['train'] and meta['id'] not in worksets['valid']
        assert row['conversations'][0]['content'] not in questions
    report=dict(status='passed',counts=counts,source_relations_checked=source_checked,
        new_train_valid_work_overlap=0,new_train_valid_prompt_overlap=0,
        complete_entity_targets=True,negative_gold_alias_collisions=0,base_hashes_unchanged=True,
        seen_fact_probe_rows=len(probes),scope='来源一致性与结构检查；语料元数据不等于权威史料核验；基础库/预训练的事实曝光尚未去除。')
    (folder/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest['status']='validated'
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir',type=Path,default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1/factual_v1')
    validate(p.parse_args().input_dir.resolve())
