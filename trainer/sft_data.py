# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""本次全量SFT的数据处理：确定性模板、按真实消息范围屏蔽标签、动态padding。

不使用原SFTDataset的随机system/空think处理，以便验证和断点恢复可复现。
不靠搜索整段token里的assistant标记判断角色，避免把用户引用的标记误作答案。
"""
import json
import math
from pathlib import Path

import datasets
import torch
from torch.utils.data import Dataset


def load_split(path, limit=0):
    """小测试只读前N行；正式运行使用Arrow内存映射，不把14GB文本装入Python列表。"""
    if limit:
        rows = []
        with open(path, encoding='utf-8-sig') as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
                    if len(rows) >= limit:
                        break
        result = datasets.Dataset.from_list(rows)
    else:
        # 显式声明可选字段，兼容原项目数据格式，避免后续分块出现字段推断变化。
        fields = datasets.Features({'conversations': [{
            'role': datasets.Value('string'), 'content': datasets.Value('string'),
            'reasoning_content': datasets.Value('string'), 'tools': datasets.Value('string'),
            'tool_calls': datasets.Value('string')}]})
        result = datasets.load_dataset('json', data_files=str(path), split='train', features=fields)
    if not len(result) or 'conversations' not in result.column_names:
        raise ValueError(f'数据为空或缺少conversations: {path}')
    return result


class AssistantDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        if not tokenizer.is_fast:
            raise ValueError('按字符范围屏蔽标签需要Fast tokenizer')
        self.samples, self.tokenizer, self.max_length = samples, tokenizer, max_length

    def __len__(self):
        return len(self.samples)

    def encode(self, row, index=0):
        messages, tools = [], None
        for original in row.get('conversations', []):
            if original.get('role') not in ('system', 'user', 'assistant', 'tool') or not isinstance(original.get('content'), str):
                raise ValueError(f'样本{index}的角色或content格式不受支持')
            message = {k: v for k, v in original.items() if v is not None and k in ('role', 'content', 'reasoning_content', 'tool_calls')}
            if original['role'] == 'system' and original.get('tools'):
                tools = json.loads(original['tools']) if isinstance(original['tools'], str) else original['tools']
            if message.get('tool_calls') and isinstance(message['tool_calls'], str):
                message['tool_calls'] = json.loads(message['tool_calls'])
            messages.append(message)
        if not messages or not any(m['role'] == 'assistant' and (m['content'].strip() or m.get('tool_calls') or m.get('reasoning_content')) for m in messages):
            raise ValueError(f'样本{index}没有非空assistant回答')
        render = lambda ms: self.tokenizer.apply_chat_template(ms, tokenize=False, add_generation_prompt=False, open_thinking=False, tools=tools)
        rendered = render(messages)
        spans, cursor = [], 0
        header = (self.tokenizer.bos_token or '') + 'assistant\n'
        if not tools and not any(m['role'] == 'tool' for m in messages):
            parts = [render([m]) for m in messages]
        else:
            parts = None
        if parts is not None and ''.join(parts) == rendered:
            # 普通对话快路径：逐消息计算一次，避免长多轮对话反复渲染前缀。
            for message, part in zip(messages, parts):
                if message['role'] == 'assistant':
                    if not part.startswith(header):
                        raise ValueError('assistant模板头与tokenizer不匹配')
                    spans.append((cursor + len(header), cursor + len(part)))
                cursor += len(part)
        else:
            # 工具对话包含分组tool_response及工具说明，使用真实消息前缀定位。
            for i, message in enumerate(messages):
                if message['role'] != 'assistant':
                    continue
                before = render(messages[:i]) if i else ''
                through = render(messages[:i+1])
                if not rendered.startswith(through) or not through.startswith(before + header):
                    raise ValueError(f'样本{index}的上下文模板不满足前缀一致性，拒绝错误标注')
                spans.append((len(before) + len(header), len(through)))
        # 回答区间包含think、正文、工具调用、EOS和换行；角色头及工具返回不算loss。
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = encoded['input_ids']
        ids = full_ids[:self.max_length]
        labels = [-100] * len(ids)
        span_index = 0
        for i, (start, end) in enumerate(encoded['offset_mapping'][:self.max_length]):
            while span_index < len(spans) and start >= spans[span_index][1]:
                span_index += 1
            if span_index < len(spans) and end > start and start >= spans[span_index][0] and end <= spans[span_index][1]:
                labels[i] = ids[i]
        supervised = sum(v != -100 for v in labels[1:])
        return dict(input_ids=ids, labels=labels, index=index, full_tokens=len(full_ids),
                    truncated=len(full_ids) > self.max_length, target_tokens=supervised,
                    eos_supervised=sum(token == self.tokenizer.eos_token_id and label != -100 for token, label in zip(ids, labels)),
                    rendered=rendered)

    def __getitem__(self, index):
        result = self.encode(self.samples[int(index)], int(index))
        result.pop('rendered')  # 不把完整字符串传入每个训练batch，减少内存占用。
        return result


class SFTCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        valid = [f for f in features if f['target_tokens'] > 0]
        stats = dict(source_samples=len(features), skipped_samples=len(features)-len(valid),
                     truncated_samples=sum(f['truncated'] for f in features))
        if not valid:
            return dict(input_ids=None, labels=None, attention_mask=None, **stats)
        size = math.ceil(max(len(f['input_ids']) for f in valid) / 8) * 8
        ids = torch.full((len(valid), size), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for i, f in enumerate(valid):
            n = len(f['input_ids'])
            ids[i, :n] = torch.tensor(f['input_ids'])
            labels[i, :n] = torch.tensor(f['labels'])
            mask[i, :n] = 1
        # padding永远是-100；截断时没有EOS就不伪造EOS，避免教模型过早结束。
        return dict(input_ids=ids, labels=labels, attention_mask=mask, **stats)


def audit_dataset(dataset, limit=256):
    n = min(len(dataset), limit)
    indices = [i * len(dataset) // n for i in range(n)]
    stats = dict(audited_samples=n, truncated_samples=0, no_target_samples=0, samples_with_supervised_eos=0,
                 target_tokens=0, max_full_tokens=0)
    for index in indices:
        feature = dataset[index]
        stats['truncated_samples'] += feature['truncated']
        stats['no_target_samples'] += feature['target_tokens'] == 0
        stats['samples_with_supervised_eos'] += feature['eos_supervised'] > 0
        stats['target_tokens'] += feature['target_tokens']
        stats['max_full_tokens'] = max(stats['max_full_tokens'], feature['full_tokens'])
    stats['note'] = '均匀取原文件位置的诊断样本，不是全库长度统计；截断后无监督token的样本在collator中排除并计数'
    return stats
