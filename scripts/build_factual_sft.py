# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""向clean_v1添加有来源的事实通道与混合清单；不重写14GB基础库，不修改lm_dataset.py。

外部核对的现代作品关系与语料元数据关系分层标记。原文片段唯一性、同名作品、
原train/valid归属共同限制生成。验证集是SFT关系/问法留出，不声称预训练未见。
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.clean_sft_dataset import RECITE, contains_old_name


def key(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def norm(value):
    return unicodedata.normalize('NFC',value).strip()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def collect(base):
    records, titles, clues = {}, defaultdict(set), defaultdict(set)
    source_info={}
    prefixes=('请按原文',RECITE)
    for split in ('valid','train'):
        path=base/f'sft_t2t_with_poetry_{split}.jsonl'
        sha=hashlib.sha256(); count=0
        marker='作者：'.encode('utf-8')
        with path.open('rb') as stream:
            for line_no,line in enumerate(stream,1):
                sha.update(line)
                if marker not in line: continue
                row=json.loads(line); ms=row.get('conversations',[])
                if len(ms)!=2 or ms[0]['role']!='user' or ms[1]['role']!='assistant':continue
                user=ms[0]['content']
                if not user.startswith(prefixes):continue
                lines=user.splitlines()
                if len(lines)<3 or not lines[1].startswith('《') or not lines[1].endswith('》') or not lines[2].startswith('作者：'):continue
                title=norm(lines[1][1:-1].split(' / ',1)[-1]); author=norm(lines[2][3:])
                if not author or author in ('佚名','无名氏','不详','未知') or len(author)>20:continue
                ident=key(title+'\0'+author)
                titles[title].add(author)
                rec=records.setdefault(ident,dict(id=ident,title=title,author=author,aliases=[author],
                    descriptor='古典诗词作品',confidence=.6,quality='corpus_metadata',splits=set(),body=None,
                    source=dict(file=str(path.resolve()),line=line_no)))
                rec['splits'].add(split)
                if user.startswith(RECITE):
                    body=ms[1]['content']; clue=norm(body.split('\n')[0])
                    clues[clue].add(author)
                    if rec['body'] is None:
                        rec.update(body=body,clue=clue,source=dict(file=str(path.resolve()),line=line_no,body_sha256=key(body)))
                count+=1
                if line_no%1000000==0:print(f'catalog {split}: input={line_no:,}',flush=True)
        source_info[split]=dict(path=str(path.resolve()),sha256=sha.hexdigest(),bytes=path.stat().st_size,rows=line_no)
        print(f'catalog {split}: metadata={count:,}',flush=True)
    return records,titles,clues,source_info


def seed_facts(seed_path):
    seed=json.loads(seed_path.read_text(encoding='utf-8'))
    for source in seed['sources']:
        for title,descriptor in source['works']:
            yield dict(id=key('book\0'+title),title=title,descriptor=descriptor,author=source['author'],
                aliases=source['aliases'],confidence=1.,quality='externally_reviewed',
                source=dict(url=source['url'],review_date=seed['review_date']),clue='',body=None)


def dialogue(question,answer):
    return dict(conversations=[dict(role='user',content=question),dict(role='assistant',content=answer)])


def views(fact,negative,validation=False,probe=False):
    """负例只用于问题纠错或旁路排序，不作为错误的assistant答案写入正训练JSONL。"""
    label=fact['descriptor']+'《'+fact['title']+'》'
    clue=fact.get('clue','')
    clue_hint=f'开篇是“{clue}”。' if clue else ''
    if probe:
        return [('seen_fact_unseen_wording',f'{label}，{clue_hint}请直接写出创作者的姓名，不附加说明。')]
    if validation:
        return [('heldout_author',f'请指出{label}的作者。{clue_hint}只输出姓名。')]
    evidence=f'书目资料：作品={fact["title"]}；作者={fact["author"]}。'
    result=[('author_recall',f'{label}的作者是谁？{clue_hint}只输出作者姓名。'),
        ('author_paraphrase',f'谁创作了{label}？{clue_hint}只回答姓名。'),
        ('author_grounded',evidence+'\n根据以上资料回答：这部作品的作者是谁？只输出姓名。'),
        ('author_correction',f'有人把{label}的作者说成{negative}。{clue_hint}请给出正确作者，只输出姓名。')]
    return result


def build(base,out,seed_path,train_limit=8000,valid_limit=1000):
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    records,titles,clues,sources=collect(base)
    baseline=json.loads((base/'clean_report.json').read_text(encoding='utf-8'))
    for split,info in sources.items():
        if info['sha256']!=baseline['outputs'][split]['sha256']:raise ValueError('基础库SHA256发生变化')
    candidates={'train':[],'valid':[]}; rejected=Counter()
    for fact in records.values():
        if len(fact['splits'])!=1:rejected['cross_split_work']+=1;continue
        if not fact['body']:continue
        clue=fact['clue']
        if len(clue)<10 or len(clue)>120 or len(clues[clue])!=1:
            rejected['ambiguous_or_short_clue']+=1;continue
        # 同名作品即使作者不同也不输出裸题名问答：所有语料关系带开篇线索。
        split=next(iter(fact.pop('splits')))
        fact['title_author_count']=len(titles[fact['title']])
        fact.pop('body')
        candidates[split].append(fact)
    seeds=list(seed_facts(seed_path))
    # 每位作者至少保留一个外部核对作品作关系留出，避免只有一种作者/领域的验证。
    grouped=defaultdict(list)
    for fact in seeds:grouped[fact['author']].append(fact)
    for facts in grouped.values():
        ordered=sorted(facts,key=lambda f:key('seed42'+f['id']))
        for i,fact in enumerate(ordered):
            candidates['valid' if i==0 else 'train'].append(fact)
    selected={}
    for split,limit in [('train',train_limit),('valid',valid_limit)]:
        ordered=sorted(candidates[split],key=lambda f:key('42:'+f['id']))
        reviewed=[f for f in ordered if f['quality']=='externally_reviewed']
        corpus=[f for f in ordered if f['quality']=='corpus_metadata']
        selected[split]=reviewed+corpus[:max(0,limit-len(reviewed))]
    train_ids={f['id'] for f in selected['train']}; valid_ids={f['id'] for f in selected['valid']}
    if train_ids&valid_ids:raise ValueError('新增事实的作品分割重叠')
    # 作者候选池不取未知/别名；优先从同来源质量级别选择，减少毫无关系的简单负例。
    pools={quality:sorted({f['author'] for split in selected.values() for f in split if f['quality']==quality})
        for quality in ('externally_reviewed','corpus_metadata')}
    counters={}; seen=set(); previews=[]
    for split in ('valid','train'):
        counts=Counter()
        with (out/f'facts_{split}.jsonl').open('w',encoding='utf-8') as data, \
             (out/f'facts_{split}.meta.jsonl').open('w',encoding='utf-8') as meta:
            for fact in selected[split]:
                available=[a for a in pools[fact['quality']] if a not in fact['aliases']]
                if not available:continue
                negative=available[int(key(fact['id'])[:8],16)%len(available)]
                tasks=views(fact,negative,validation=split=='valid')
                for task,question in tasks:
                    row=dialogue(question,fact['author']); signature=key(question)
                    if signature in seen:raise ValueError('生成提示重复')
                    seen.add(signature)
                    metadata={k:v for k,v in fact.items() if k not in ('clue',)}
                    metadata.update(task=task,negative_answer=negative,answer=fact['author'])
                    data.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                    meta.write(json.dumps(metadata,ensure_ascii=False,separators=(',',':'))+'\n')
                    counts[task]+=1;counts['rows']+=1;counts[fact['quality']]+=1
                    if len(previews)<4 or fact['quality']=='externally_reviewed' and len(previews)<12:
                        previews.append(dict(split=split,row=row,metadata=metadata))
        counters[split]=dict(counts)
    # 歧义题只教澄清信息，不伪造“不存在的书”；训练与验证采用不同题名。
    ambiguous=sorted((t for t,a in titles.items() if len(a)>1 and len(t)<=8),key=key)[:64]
    clarification='信息不足，无法唯一确定作者。'
    for split,chosen in [('train',ambiguous[:48]),('valid',ambiguous[48:])]:
        with (out/f'facts_{split}.jsonl').open('a',encoding='utf-8') as data, \
             (out/f'facts_{split}.meta.jsonl').open('a',encoding='utf-8') as meta:
            for title in chosen:
                question=f'只给出题名《{title}》，没有年代、原文或版本信息，能唯一确定作者吗？请简短回答。'
                row=dialogue(question,clarification)
                data.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                meta.write(json.dumps(dict(id=key('ambiguous:'+title),title=title,author=None,answer=clarification,
                    aliases=[clarification],task='clarify_ambiguity',quality='corpus_ambiguity',confidence=1.,negative_answer=None,
                    source=dict(authors=sorted(titles[title]))),ensure_ascii=False,separators=(',',':'))+'\n')
                counters[split]['rows']+=1;counters[split]['clarify_ambiguity']=counters[split].get('clarify_ambiguity',0)+1
    # 训练事实上的未见问法探针与关系留出分开，不能混称未知事实泛化。
    with (out/'probe_seen.jsonl').open('w',encoding='utf-8') as data, (out/'probe_seen.meta.jsonl').open('w',encoding='utf-8') as meta:
        for fact in selected['train'][:256]:
            _,question=views(fact,None,probe=True)[0]
            data.write(json.dumps(dialogue(question,fact['author']),ensure_ascii=False)+'\n')
            meta.write(json.dumps(dict(id=fact['id'],answer=fact['author'],aliases=fact['aliases'],quality=fact['quality'],task='seen_fact_unseen_wording'),ensure_ascii=False)+'\n')
    save(out/'preview.json',previews)
    files={}
    for path in out.glob('*.jsonl'):
        files[path.name]=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    # 不把负例写成常规正答案；新增JSONL与原库通过此清单成为可直接使用的多通道数据集。
    manifest=dict(version='factual_sft_v1',status='built_pending_validation',base=sources,files=files,
        factual_work_counts={s:len(fs) for s,fs in selected.items()},counts=counters,rejected_catalog=dict(rejected),
        evidence_tiers='externally_reviewed uses reviewed institutional pages; corpus_metadata only asserts agreement with local metadata.',
        split_scope='New facts: original base split or seed work holdout; author overlap allowed. Base/pretraining may already contain these facts; no claim of globally unseen knowledge.',
        seed_source=str(seed_path.resolve()),seed_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        base_modified=False,seconds=time.perf_counter()-started)
    save(out/'manifest.json',manifest)
    print(json.dumps(dict(output=str(out),counts=counters,rejected=dict(rejected)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base_dir',type=Path,default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1')
    p.add_argument('--output_dir',type=Path,default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1/factual_v1')
    p.add_argument('--seed_path',type=Path,default=ROOT/'dataset/factual_seed_sources.json')
    p.add_argument('--train_facts',type=int,default=8000)
    p.add_argument('--valid_facts',type=int,default=1000)
    args=p.parse_args()
    build(args.base_dir,args.output_dir,args.seed_path,args.train_facts,args.valid_facts)
