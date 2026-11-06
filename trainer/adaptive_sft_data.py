# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""自适应 SFT 的索引、任务弱标注与无目标丢失窗口；不修改 lm_dataset.py。

任务标签描述回答形式，不保证答案真实。未识别任务归 general，避免把开放解释
误当作唯一答案。索引只保存原文件位置；原始大文件不复制、不改写。
"""
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
import datasets  # Windows DLL 加载顺序
import torch
from dataset.lm_dataset import pre_processing_chat
from trainer.sft_data import AssistantDataset, SFTCollator

TASKS = ('fact', 'open', 'general')
INDEX_DTYPE = np.dtype([('offset', '<u8'), ('size', '<u4'), ('source', 'u1'),
    ('task', 'u1'), ('label_confidence', '<f4'), ('reliability', '<f4'), ('group', 'S16')])
EXACT = ('请背诵以下作品的原文', '请按原文填入', '请按原文补全', '请为下面的古文恢复标点')
OPEN = re.compile(r'原创|创作|写(?:一|首|篇|个|段|诗|词|故事)|作诗|写诗|作词|赏析|鉴赏|改写|润色|续写|模仿|设计|头脑风暴|brainstorm|creative|write (?:a|an)|compose', re.I)
FACT = re.compile(r'(?:作者|朝代|出生年份|逝世年份)(?:是(?:谁|哪)|为谁)|谁(?:创作|写)了|只(?:输出|回答)(?:作者姓名|姓名|整数|数字)|计算\s*\d', re.I)


def prepare_sft_row(row, seed, add_system_ratio=.2):
    """复用普通 SFT 身份 system，使用独立随机数，不改变原始数据或全局 RNG。

    不改写 assistant 答案；已有 system、工具对话由原预处理保护。
    调用方只对训练样本启用，验证使用固定原始题面。
    """
    conversations = row.get('conversations', [])
    # 老数据中部分自我介绍仍写“由 chen 开发”。避免给它叠加相互矛盾的新身份提示；
    # 这只是避免额外冲突，原答案仍需独立清洗，不能宣称已修正身份数据。
    if any(m.get('role') == 'assistant' and re.search(
            r'(?<![a-zA-Z0-9_])chen(?![a-zA-Z0-9_])', m.get('content', ''), re.I)
            and re.search(r'mindchen|(?:我是|I am|I\x27m).{0,40}(?:助手|assistant|模型)',
                          m.get('content', ''), re.I) for m in conversations):
        return row
    processed = pre_processing_chat(conversations, add_system_ratio, rng=random.Random(seed))
    return dict(row, conversations=processed) if processed is not conversations else row


def prompt_of(row):
    """只使用最后一次 user 及其之前的上下文；分类绝不查看最终答案。"""
    messages = row.get('conversations', [])
    users = [m.get('content', '') for m in messages if m.get('role') == 'user']
    return users[-1] if users else ''


def classify_prompt(prompt):
    if prompt.startswith(EXACT):
        return 0, .98, .6
    # 混合任务与分析解释保守归 general，不宣称逐句识别真伪。
    is_open = bool(OPEN.search(prompt))
    is_fact = bool(FACT.search(prompt))
    if is_open and is_fact:
        return 2, .35, 0.
    if is_open:
        return 1, .85, 0.
    if is_fact and len(prompt) < 700:
        return 0, .75, .3
    return 2, .35, 0.


def digest(text):
    return hashlib.sha256(unicodedata.normalize('NFC', text).encode('utf-8')).digest()[:16]


def work_group(row, meta=None):
    if meta and meta.get('title') and meta.get('author'):
        return digest(meta['title'] + '\0' + meta['author'])
    prompt = prompt_of(row)
    title = re.search(r'《([^》]+)》\s*\n作者：([^\n]+)', prompt)
    if title:
        return digest(title[1].split(' / ', 1)[-1].strip() + '\0' + title[2].strip())
    return digest(prompt)


def lossless_features(row, tokenizer, length=2048, overlap=512):
    """完整编码，再切重叠窗口。监督 token 恰好出现一次，EOS 只来自原文。

    保留的是所有 assistant 目标；超过窗口的远端上下文不会全部保留。
    重叠段只作为上下文，不重复计 loss。长 prompt 的纯上下文窗口不进训练。
    """
    if not 1 <= overlap < length or length < 32:
        raise ValueError('要求 length>=32 且 1<=overlap<length')
    full = AssistantDataset([], tokenizer, 2**31 - 1).encode(row)
    ids, labels = full['input_ids'], full['labels']
    target_count = sum(x != -100 for x in labels[1:])
    if not target_count:
        raise ValueError('原样本没有可学习的 assistant token')
    first = next(i for i, v in enumerate(labels) if v != -100)
    prompt_ids = ids[max(0, first - 512):first]
    if not prompt_ids:
        raise ValueError('assistant 之前缺少 prompt 上下文')
    windows, start, covered_until = [], 0, 0
    while start < len(ids):
        end = min(start + length, len(ids))
        part_labels = [v if i >= covered_until else -100
                       for i, v in enumerate(labels[start:end], start)]
        part_labels[0] = -100  # 窗口首 token 没有前一位置，不能参与自回归预测。
        n = sum(v != -100 for v in part_labels[1:])
        if n:
            windows.append(dict(input_ids=ids[start:end], labels=part_labels,
                target_tokens=n, truncated=False, full_tokens=end-start,
                eos_supervised=sum(t == tokenizer.eos_token_id and y != -100
                                   for t, y in zip(ids[start:end], part_labels)), start=start, end=end))
        covered_until = end
        if end == len(ids):
            break
        start = end - overlap
    retained = sum(w['target_tokens'] for w in windows)
    if retained != target_count:
        raise AssertionError(f'目标丢失或重复：{retained} != {target_count}')
    return dict(windows=windows, prompt_ids=prompt_ids, target_tokens=target_count,
        full_tokens=len(ids), eos_supervised=full['eos_supervised'], target_retention=1.,
        windowed=len(ids)>length, row=row)


class IndexedCorpus:
    def __init__(self, manifest_path, check=True):
        from trainer.continue_pretrain_utils import sha256_file, file_info
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
        if self.manifest['status'] != 'ready':
            raise ValueError('数据索引尚未准备完成')
        self.sources = self.manifest['sources']
        for item in self.sources:
            actual = file_info(item['path'])
            if actual['bytes'] != item['bytes'] or actual['mtime_ns'] != item['mtime_ns']:
                raise ValueError('原数据改变，请重新准备索引：' + item['path'])
        self.indices = {}
        for key, item in self.manifest['indices'].items():
            path = self.path.parent / item['file']
            if check and sha256_file(path) != item['sha256']:
                raise ValueError('索引校验失败：' + key)
            self.indices[key] = np.memmap(path, mode='r', dtype=INDEX_DTYPE) if item['count'] else []
        self.streams = {}

    def read(self, key, index):
        item = self.indices[key][index]
        source = int(item['source'])
        if source not in self.streams:
            self.streams[source] = open(self.sources[source]['path'], 'rb')
        stream = self.streams[source]
        stream.seek(int(item['offset']))
        row = json.loads(stream.read(int(item['size'])))
        return row, dict(task=int(item['task']), label_confidence=float(item['label_confidence']),
            reliability=float(item['reliability']), source=source, index=index, channel=key)

    def close(self):
        for stream in self.streams.values():
            stream.close()


def quota_counts(total, ratios):
    if total < 3 or len(ratios) != 3 or any(r <= 0 for r in ratios):
        raise ValueError('每次更新至少3条，三类比例必须为正')
    values = np.asarray(ratios) / sum(ratios) * total
    counts = np.floor(values).astype(int)
    for i in np.argsort(-(values - np.floor(values)))[:total-int(counts.sum())]:
        counts[i] += 1
    for i in range(3):
        if counts[i]==0:
            donor=int(np.argmax(counts)); counts[donor]-=1; counts[i]+=1
    return counts.tolist()


def draw_examples(corpus, count, ratios, seed, split='train'):
    rng = random.Random(seed)
    result = []
    for task, n in enumerate(quota_counts(count, ratios)):
        candidates = [k for k, rows in corpus.indices.items()
                      if k.startswith(split+'_') and k.endswith('_'+TASKS[task]) and len(rows)]
        if not candidates:
            raise ValueError(f'{split} 缺少 {TASKS[task]} 数据')
        # 在事实通道内防止百万原文样本淹没新补充作者/可验证事实。
        weights = [(.45 if '_author_' in k else .20 if '_verified_' in k else .35)
                   if task == 0 else 1. for k in candidates]
        for _ in range(n):
            key = rng.choices(candidates, weights=weights, k=1)[0]
            result.append((key, rng.randrange(len(corpus.indices[key]))))
    rng.shuffle(result)
    return result


def prompt_batch(examples, pad_id, device):
    size = max(len(e['prompt_ids']) for e in examples)
    ids = torch.full((len(examples), size), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, example in enumerate(examples):
        n = len(example['prompt_ids'])
        ids[i, :n] = torch.tensor(example['prompt_ids'], device=device)
        mask[i, :n] = 1
    return ids, mask
