# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""事实SFT的样本/实体归一化、排序与回放KL；不修改原Dataset或模型类。"""
import copy
import json
import random
import re
import unicodedata

import datasets  # Windows DLL加载顺序
import torch
import torch.nn.functional as F
from trainer.sft_data import AssistantDataset, SFTCollator


def read_aligned(data_path,meta_path):
    with open(data_path,encoding='utf-8') as data,open(meta_path,encoding='utf-8') as metadata:
        rows,metas=[json.loads(s) for s in data],[json.loads(s) for s in metadata]
    if len(rows)!=len(metas):raise ValueError('事实数据与元数据行数不一致')
    return rows,metas


def entity_feature(row,answer,tokenizer,max_length):
    """用实际模板字符offset定位答案实体；问题/空think/EOS不能进入实体分数。"""
    feature=AssistantDataset([],tokenizer,max_length).encode(row)
    rendered=feature.pop('rendered')
    start=rendered.rfind(answer)
    if start<0:raise ValueError('模板中未找到实体答案')
    end=start+len(answer)
    offsets=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'][:max_length]
    mask=[label!=-100 and b>a and a>=start and b<=end for label,(a,b) in zip(feature['labels'],offsets)]
    if feature['truncated'] or not any(mask[1:]):
        raise ValueError('事实答案被截断或无完整实体监督，拒绝静默训练残缺标签')
    feature['entity_mask']=mask
    return feature


def encode_facts(rows,metas,tokenizer,max_length):
    examples=[]
    for row,meta in zip(rows,metas):
        answer=row['conversations'][-1]['content']
        if answer!=meta['answer']:raise ValueError('金标准正文与元数据不一致')
        pos=entity_feature(row,answer,tokenizer,max_length)
        negative=meta.get('negative_answer')
        if negative:
            if negative in meta['aliases']:raise ValueError('负例实际上是金标准别名')
            neg_row=copy.deepcopy(row);neg_row['conversations'][-1]['content']=negative
            neg=entity_feature(neg_row,negative,tokenizer,max_length)
        else:neg=pos
        examples.append(dict(row=row,meta=meta,pos=pos,neg=neg))
    return examples


def collate_entities(features,pad_id):
    batch=SFTCollator(pad_id)(features)
    if batch['skipped_samples']:raise ValueError('事实通道不允许无监督样本')
    mask=torch.zeros_like(batch['labels'],dtype=torch.bool)
    for i,feature in enumerate(features):
        mask[i,:len(feature['entity_mask'])]=torch.tensor(feature['entity_mask'])
    batch['entity_mask']=mask
    return batch


def token_nll(logits,labels):
    shifted=labels[:,1:]
    losses=F.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),shifted.reshape(-1),
        ignore_index=-100,reduction='none').view_as(shifted)
    return losses,shifted.ne(-100)


def factual_losses(logits,labels,entity_mask,confidence,entity_fraction=.7):
    """按样本平均而不是按回答长度分配权重；实体项只计算真实作者/作品答案。"""
    nll,valid=token_nll(logits,labels)
    entity=entity_mask[:,1:] & valid
    if torch.any(valid.sum(1)==0) or torch.any(entity.sum(1)==0):raise ValueError('空监督/实体')
    sequence=nll.sum(1)/valid.sum(1)
    entity_nll=(nll*entity).sum(1)/entity.sum(1)
    confidence=confidence.to(nll)
    per_sample=(1-entity_fraction)*sequence+entity_fraction*entity_nll
    balanced=(per_sample*confidence).sum()/confidence.sum()
    token_mean=nll.sum()/valid.sum()
    return dict(balanced=balanced,token_mean=token_mean,entity_nll=entity_nll,
        sequence_nll=sequence,token_sum=nll.sum(),tokens=valid.sum())


def ranking_loss(positive_nll,negative_nll,weights,margin=.5):
    # s=-NLL，softplus(m-(s_pos-s_neg))=softplus(m+NLL_pos-NLL_neg)。
    # 最小化会提高正确作者似然、降低易混淆作者似然；负答案不进入正SFT损失。
    if not torch.any(weights>0):return positive_nll.sum()*0
    losses=F.softplus(margin+positive_nll-negative_nll)
    return (losses*weights).sum()/weights.sum()


def replay_kl(student_logits,reference_logits,labels,temperature=2.):
    mask=labels[:,1:].ne(-100)
    if not mask.any():raise ValueError('KL无有效回放位置')
    # KL(p_ref || p_student)，只约束回放assistant位置；不把参考模型当新事实标签。
    student=F.log_softmax(student_logits[:,:-1,:].float()/temperature,dim=-1)
    teacher=F.softmax(reference_logits[:,:-1,:].detach().float()/temperature,dim=-1)
    divergence=F.kl_div(student,teacher,reduction='none').sum(-1)
    return (divergence[mask].mean()*temperature**2).clamp_min(0)


def sample_indices(metas,count,seed,reviewed_fraction=.25,ambiguity_fraction=.05):
    """每个update独立可复现，恢复无需重放旧batch；普通SFT对照使用同一索引流。"""
    rng=random.Random(seed)
    pools=dict(reviewed=[],corpus=[],ambiguity=[])
    for i,meta in enumerate(metas):
        category='ambiguity' if meta['task']=='clarify_ambiguity' else ('reviewed' if meta['quality']=='externally_reviewed' else 'corpus')
        pools[category].append(i)
    fallback=list(range(len(metas)))
    result=[]
    for _ in range(count):
        if rng.random()<ambiguity_fraction:pool=pools['ambiguity']
        else:pool=pools['reviewed' if rng.random()<reviewed_fraction else 'corpus']
        result.append(rng.choice(pool or fallback))
    return result


def choose_validation(metas,count,seed=42):
    rng=random.Random(seed)
    reviewed=[i for i,m in enumerate(metas) if m['quality']=='externally_reviewed']
    ambiguity=[i for i,m in enumerate(metas) if m['task']=='clarify_ambiguity']
    corpus=[i for i,m in enumerate(metas) if i not in set(reviewed+ambiguity)]
    rng.shuffle(corpus)
    return reviewed+corpus[:max(0,count-len(reviewed))]+ambiguity[:min(8,max(1,count//8))]


def normalize_answer(text):
    text=re.sub(r'<think>[\s\S]*?</think>','',text)
    return re.sub(r'[\s，。！？、,.!?：:;；《》“”"\x27]','',unicodedata.normalize('NFKC',text))


def correct_answer(text,aliases):
    # 禁用子串匹配，不能把“鲁迅或者老舍”算成答对鲁迅。
    return normalize_answer(text) in {normalize_answer(a) for a in aliases}


def entity_correct_answer(text,aliases,title=None):
    """事实实体与只输出姓名的格式合规分开；有限语法提取，禁用任意子串命中。"""
    if correct_answer(text,aliases):return True
    text=re.sub(r'<think>[\s\S]*?</think>','',text).strip()
    match=re.fullmatch(r'(?:《(?P<title>[^》]+)》的?)?(?:作者|创作者)(?:是|为|[:：])\s*(?P<name>[^，。；;!?！？]+)[。.!！]?',text)
    if not match:return False
    if match['title'] and title and normalize_answer(match['title'])!=normalize_answer(title):return False
    return correct_answer(match['name'],aliases)


def checkpoint_rank(metrics,initial_replay_nll,tolerance=.15):
    if tolerance>=0 and metrics['replay_nll']>initial_replay_nll+tolerance:return None
    return (metrics['known_macro_em'],-metrics['fact_entity_nll'],metrics['ambiguity_em'],-metrics['replay_nll'])
