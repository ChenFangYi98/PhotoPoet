# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""可复现的诗词 GRPO 提示数据；没有伪造的标准诗作或评分标签。"""
import json
import os
import math
import random
import hashlib
import re
from pathlib import Path

from torch.utils.data import Dataset
from trainer.grpo_resources import ROOT, atomic_json, file_digest
from trainer.grpo_poetry_reward import resolve_poetry_spec
from trainer.grpo_data_quality import prompt_issue

DEFAULT_DATA = ROOT / 'dataset/grpo_poetry_v2/train.jsonl'
CORPUS = ROOT / 'dataset/sft_t2t_with_poetry_clean_v1/deterministic_v3/literature_candidates.jsonl'
THEMES = [
    ('春雨', ['春', '雨']), ('秋月', ['秋', '月']), ('冬雪', ['雪', '冬']), ('夏荷', ['荷', '夏']),
    ('山居', ['山', '居']), ('江夜', ['江', '夜']), ('乡愁', ['乡', '归']), ('送别', ['别', '送']),
    ('梅花', ['梅', '寒']), ('竹林', ['竹', '林']), ('松风', ['松', '风']), ('菊香', ['菊', '香']),
    ('孤舟', ['舟', '水']), ('渔火', ['渔', '灯']), ('边塞', ['塞', '关']), ('长城', ['城', '关']),
    ('登高', ['高', '望']), ('田园', ['田', '园']), ('早春', ['春', '芽']), ('暮秋', ['秋', '叶']),
    ('雨后', ['雨', '晴']), ('夜读', ['夜', '书']), ('故友', ['友', '故']), ('远行', ['行', '路']),
    ('思亲', ['亲', '家']), ('归雁', ['雁', '归']), ('落花', ['花', '落']), ('听泉', ['泉', '声']),
    ('古寺', ['寺', '钟']), ('石桥', ['桥', '水']), ('湖光', ['湖', '光']), ('海潮', ['海', '潮']),
    ('清风', ['风', '清']), ('晨露', ['晨', '露']), ('萤火', ['萤', '夜']), ('柳岸', ['柳', '岸']),
    ('采茶', ['茶', '采']), ('丰收', ['禾', '秋']), ('小院', ['院', '庭']), ('灯下', ['灯', '夜']),
    # 验证主题与训练主题分开，不靠随机抽取同一模板的近重复项充当验证。
    ('残雪', ['雪', '残']), ('晓雾', ['雾', '晓']), ('溪云', ['溪', '云']), ('晚钟', ['钟', '晚']),
    ('春燕', ['燕', '春']), ('沙洲', ['沙', '洲']), ('芦花', ['芦', '花']), ('霜晨', ['霜', '晨']),
]
FORMS = [('五言绝句', 4, 5), ('七言绝句', 4, 7), ('五言律诗', 8, 5), ('七言律诗', 8, 7)]
MOODS = ['清新自然', '含蓄悠远', '明朗温暖', '宁静淡雅']


def make_rows(themes, split):
    rows = []
    for topic, keywords in themes:
        for form, lines, width in FORMS:
            for mood in MOODS:
                prompt = (f'请以“{topic}”为题材，原创一首{form}，风格{mood}。'
                          f'共{lines}句，每句{width}个汉字，一句一行；奇数句以逗号收尾，偶数句以句号收尾。'
                          '只输出诗歌正文，不要标题、序号、注释或解释。偶数句尽量押韵，避免重复诗句。')
                spec = dict(form=form, lines=lines, width=width, topic=topic, keywords=keywords,
                            acrostic='', rhyme=True, punctuation=True, linebreaks=True)
                rows.append(dict(id=f'{split}_{len(rows):04d}', messages=[dict(role='user', content=prompt)],
                                 poetry_spec=spec, provenance='deterministic_prompt_template_v1'))
        for width, phrase in [(5, '山河无恙'), (7, '岁月静好')]:
            prompt = (f'围绕“{topic}”写一首{width}言四句藏头诗，各句首字依次为“{phrase}”。'
                      f'每句恰好{width}个汉字，一句一行；奇数句以逗号收尾，偶数句以句号收尾。'
                      '只输出正文，不要标题和解释，避免重复诗句。')
            spec = dict(form='藏头诗', lines=4, width=width, topic=topic, keywords=keywords,
                        acrostic=phrase, rhyme=False, punctuation=True, linebreaks=True)
            rows.append(dict(id=f'{split}_{len(rows):04d}', messages=[dict(role='user', content=prompt)],
                             poetry_spec=spec, provenance='deterministic_prompt_template_v1'))
    return rows


def ensure_poetry_data(path=DEFAULT_DATA, rebuild=False):
    path = Path(path)
    if path.is_file() and not rebuild:
        return path
    if path.resolve() != DEFAULT_DATA.resolve():
        raise FileNotFoundError(f'自定义数据不存在：{path}；自动生成仅用于 {DEFAULT_DATA}')
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    corpus_rows = corpus_prompts() if CORPUS.is_file() else {'train': [], 'valid': []}
    for split, themes in [('train', THEMES[:40]), ('valid', THEMES[40:])]:
        target = path.parent / f'{split}.jsonl'
        rows = make_rows(themes, split) + corpus_rows[split]
        if not target.exists() or rebuild:
            temporary = target.with_suffix('.jsonl.tmp')
            temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
            os.replace(temporary, target)
        counts[split] = dict(rows=len(rows), sha256=file_digest(target), topics=[t[0] for t in themes])
    atomic_json(path.parent / 'manifest.json', dict(version='grpo_poetry_prompts_v2', splits=counts,
        corpus=dict(path=str(CORPUS), sha256=file_digest(CORPUS)) if CORPUS.is_file() else None,
        description='模板提示加本地诗作衍生的原创、改写、格式修复任务；题名分组切分，不包含伪造偏好分数。',
        limitation='语料可能已被 SFT 使用，此验证集仅相对于本次 RL 训练隔离，不代表基座从未见过。'))
    return path


def corpus_prompts(train_seeds=2000, valid_seeds=200):
    output = {'train': [], 'valid': []}
    used, title_counts = set(), {}
    goals = {'train': train_seeds, 'valid': valid_seeds}
    counts = {'train': 0, 'valid': 0}
    for line in CORPUS.open(encoding='utf-8-sig'):
        row = json.loads(line)
        title = re.sub(r'\s+', '', row.get('title', ''))
        if not 2 <= len(title) <= 16 or title.startswith('词牌'):
            continue
        body = row.get('body', '').replace('\\n', '\n')
        verses = [v.strip() for v in re.split(r'[，。！？；,.!?;\n]+', body) if v.strip()]
        if len(verses) not in (4, 8) or len(verses[0]) not in (5, 7):
            continue
        width = len(verses[0])
        if not all(len(v) == width and re.fullmatch(r'[\u3400-\u4dbf\u4e00-\u9fff]+', v) for v in verses):
            continue
        normalized = ''.join(verses)
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in used or title_counts.get(title, 0) >= 2:
            continue
        split = 'valid' if int(hashlib.sha256(title.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'
        # 避免题名与模板中人为保留的另一组题材交叉。
        other_topics = {t[0] for t in (THEMES[:40] if split == 'valid' else THEMES[40:])}
        if title in other_topics or counts[split] >= goals[split]:
            continue
        used.add(digest)
        title_counts[title] = title_counts.get(title, 0)+1
        counts[split] += 1
        form = ('五言' if width == 5 else '七言') + ('绝句' if len(verses) == 4 else '律诗')
        formatted = '\n'.join(v+('，' if i % 2 == 0 else '。') for i, v in enumerate(verses))
        suffix = (f'共{len(verses)}句，每句{width}个汉字，一句一行；奇数句以逗号收尾，偶数句以句号收尾。'
                  '只输出正文，不要标题、序号和解释。')
        prompts = [
            ('original', f'以“{title}”为题意，创作一首{form}，风格{MOODS[title_counts[title]-1]}。要求意象连贯、情感自然，不要背诵已有诗作。'+suffix),
            ('rewrite', f'请将下面的诗作改写为新的{form}，保留大致意境，换用自己的表达，不要原文照抄。'+suffix+'\n参考诗作：\n'+formatted),
            ('format', '请为下面这段诗文恢复分句、换行和中文标点，保持原有汉字及其顺序，一个字也不要增删。'+suffix+'\n待整理诗文：\n'+normalized),
        ]
        for task, prompt in prompts:
            spec = dict(form=form, lines=len(verses), width=width, topic=title, keywords=[], acrostic='',
                rhyme=False, punctuation=True, linebreaks=True, reference_body=normalized,
                preserve_chars=task == 'format', discourage_copy=task != 'format', task=task)
            output[split].append(dict(id=f'{split}_corpus_{digest[:16]}_{task}',
                messages=[dict(role='user', content=prompt)], poetry_spec=spec,
                provenance=dict(kind='local_poem_derived_prompt', source_id=row.get('id'),
                                body_sha256=digest, corpus=str(CORPUS), source=row.get('source'))))
        if all(counts[k] >= goals[k] for k in goals):
            break
    return output


class PoetryPromptDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.rows = []
        with Path(path).open(encoding='utf-8-sig') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get('messages', row.get('conversations', []))
                if messages and messages[-1]['role'] == 'assistant':
                    messages = messages[:-1]
                if not messages or messages[-1]['role'] != 'user':
                    raise ValueError(f'{path}:{line_number} 缺少末尾 user 提问')
                issue = prompt_issue(messages)
                if issue:
                    raise ValueError(f'{path}:{line_number}: {issue}')
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                        open_thinking=False)
                if len(tokenizer(prompt, add_special_tokens=False).input_ids) > max_length:
                    raise ValueError(f'{path}:{line_number} 提示超出 --max_seq_len={max_length}，拒绝截断诗歌要求')
                try:
                    spec = resolve_poetry_spec(messages, row.get('poetry_spec'))
                    if not spec.get('lines') or not spec.get('width'):
                        raise ValueError('必须指定诗歌句数和每句字数，不能默认全部用五言或七言')
                except ValueError as error:
                    raise ValueError(f'{path}:{line_number} {error}') from error
                self.rows.append(dict(prompt=prompt, messages=messages, spec=spec,
                                      id=row.get('id', str(line_number))))
        if not self.rows:
            raise ValueError('GRPO 数据为空')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_prompts(rows):
    return rows


class MixedPromptDataset(Dataset):
    """原 RLAIF 只读载入，与新增诗词分开存储；长普通对话保留最近轮次。"""
    def __init__(self, general_path, poetry_path, tokenizer, max_length=768, validation_fraction=0.05):
        self.poetry = PoetryPromptDataset(poetry_path, tokenizer, max_length)
        self.poetry_valid = PoetryPromptDataset(Path(poetry_path).with_name('valid.jsonl'), tokenizer, max_length)
        self.general = []
        self.general_valid = []
        self.trimmed = 0
        self.normalized_tails = 0
        self.quarantine = []
        seen = set()
        self.duplicates = 0
        with Path(general_path).open(encoding='utf-8-sig') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = [dict(role=m['role'], content=m['content'] or '')
                            for m in row.get('messages', row.get('conversations', []))]
                removed = 0
                while messages and messages[-1]['role'] == 'assistant':
                    messages.pop()
                    removed += 1
                self.normalized_tails += int(removed > 1)
                issue = prompt_issue(messages)
                if issue:
                    self.quarantine.append(dict(id=f'rlaif_{line_number}', line=line_number, reason=issue,
                        source=str(general_path), messages=messages))
                    continue
                def render():
                    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                         open_thinking=False)
                prompt = render()
                trimmed = False
                while len(tokenizer(prompt, add_special_tokens=False).input_ids) > max_length:
                    trimmed = True
                    if len(messages) > 1:
                        # 删除最旧的完整轮次，不把 assistant 历史残片当 user 指令。
                        messages.pop(0)
                        while len(messages) > 1 and messages[0]['role'] != 'user':
                            messages.pop(0)
                    else:
                        ids = tokenizer(messages[0]['content'], add_special_tokens=False).input_ids
                        excess = len(tokenizer(prompt, add_special_tokens=False).input_ids)-max_length
                        keep = len(ids)-excess-8
                        if keep < 1:
                            raise ValueError('max_seq_len 太小，无法保留基本对话模板')
                        messages[0]['content'] = tokenizer.decode(ids[-keep:], skip_special_tokens=True)
                    prompt = render()
                self.trimmed += int(trimmed)
                key = hashlib.sha256(prompt.encode()).hexdigest()
                if key in seen:
                    self.duplicates += 1
                    continue
                seen.add(key)
                # 按最终实际输入分组，重复/被裁为相同上下文的样本不能跨训练和验证。
                bucket = self.general_valid if int(key[:12], 16)/16**12 < validation_fraction else self.general
                bucket.append(dict(prompt=prompt, messages=messages, spec={},
                    id=f'rlaif_{line_number}', channel='general', context_trimmed=trimmed))
        if not self.general:
            raise ValueError('原 RLAIF 数据为空')
        self.general_count = len(self.general)
        self.poetry_count = len(self.poetry)
        if {r['prompt'] for r in self.poetry.rows} & {r['prompt'] for r in self.poetry_valid.rows}:
            raise ValueError('Poetry train/validation prompts overlap')

    def validation_rows(self, count_per_channel=8):
        # 固定选择，不消耗策略采样随机数；题型轮转，避免验证全是同一种格式。
        general = sorted(self.general_valid, key=lambda r: hashlib.sha256(r['prompt'].encode()).hexdigest())[:count_per_channel]
        pool = [dict(r, channel='poetry', context_trimmed=False) for r in self.poetry_valid.rows]
        by_form = {}
        for row in pool:
            form, task = row['spec'].get('form'), row['spec'].get('task', 'template')
            by_form.setdefault(form, {}).setdefault(task, []).append(row)
        for tasks in by_form.values():
            for group in tasks.values():
                group.sort(key=lambda r: hashlib.sha256(r['prompt'].encode()).hexdigest())
        poetry = []
        task_offset = 0
        while len(poetry) < count_per_channel:
            added = False
            for form in sorted(by_form):
                tasks = by_form[form]
                keys = sorted(k for k in tasks if tasks[k])
                if keys and len(poetry) < count_per_channel:
                    poetry.append(tasks[keys[task_offset % len(keys)]].pop(0))
                    added = True
            task_offset += 1
            if not added:
                break
        return general+poetry

    def __len__(self):
        return self.general_count+self.poetry_count

    def __getitem__(self, index):
        if index < self.general_count:
            return self.general[index]
        return dict(self.poetry[index-self.general_count], channel='poetry', context_trimmed=False)

    def epoch_indices(self, seed, poetry_ratio=0.5):
        if not 0 < poetry_ratio < 1:
            raise ValueError('poetry_ratio 必须在 0 和 1 之间，保留两类数据')
        rng = random.Random(seed)
        total = max(self.general_count/(1-poetry_ratio), self.poetry_count/poetry_ratio)
        def expand(start, count, needed):
            result = []
            while len(result) < needed:
                cycle = list(range(start, start+count))
                rng.shuffle(cycle)
                result.extend(cycle[:needed-len(result)])
            return result
        indices = expand(0, self.general_count, math.ceil(total*(1-poetry_ratio)))
        indices += expand(self.general_count, self.poetry_count, math.ceil(total*poetry_ratio))
        rng.shuffle(indices)
        return indices
