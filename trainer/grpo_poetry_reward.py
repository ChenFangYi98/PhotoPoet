# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""可解释诗歌格式奖励 + 有界内容评分；普通回答不套用格律约束。"""
import math
import re
from collections import Counter

import torch
from trainer.grpo_repetition import response_repetition

HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
SPLIT = re.compile(r'[，。！？；,.!?;\n]+')
FORMS = {'五言绝句': (4, 5), '五绝': (4, 5), '七言绝句': (4, 7), '七绝': (4, 7),
         '五言律诗': (8, 5), '五律': (8, 5), '七言律诗': (8, 7), '七律': (8, 7)}
NUMERAL = r'[0-9一二两三四五六七八九十]+'


def rep_penalty(text, n=3, cap=0.8):
    # 中文必须按字统计，不能把一整句中文当成一个 \w+ token。
    tokens = re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]+', text.lower())
    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return min(cap, cap * 2 * (len(grams)-len(set(grams))) / max(len(grams), 1))


def _number(text):
    if text.isdecimal():
        return int(text)
    digits = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    if '十' in text:
        left, right = text.split('十', 1)
        return digits.get(left, 1)*10+digits.get(right, 0)
    return digits[text]


def infer_spec(messages):
    """只从用户要求识别，绝不按模型回答的实际长度反推目标字数。"""
    text = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')
    text = re.split(r'参考诗作[：:]|待整理诗文[：:]', text, maxsplit=1)[0]
    # 题名可能含“五七言”等字样，题材名称不能覆盖真正的创作要求。
    text = re.sub(r'[“《「\"][^”》」\"\n]+[”》」\"]\s*(?:为题意|为题材|为题目|为题)', ' ', text)
    if not re.search(r'写(?:一|个|首|[五七57])|写.{0,6}诗|创作|改写|续写|作一首|生成|整理|恢复', text):
        return {}
    text = text.replace('5言', '五言').replace('7言', '七言')
    requirements = {}
    def require(key, value):
        if key in requirements and requirements[key] != value:
            raise ValueError(f'诗歌要求冲突：{key} 同时指定了 {requirements[key]} 和 {value}')
        requirements[key] = value
    for form, (lines, width) in FORMS.items():
        if form in text:
            require('lines', lines)
            require('width', width)
    for value in re.findall(r'([五七])\s*言', text):
        require('width', _number(value))
    for value in re.findall(r'每句\s*(?:恰好|严格|必须|有|为|是|需为|应为|限制为|限定为)?\s*('+NUMERAL+r')\s*(?:个)?(?:汉字|字)', text):
        require('width', _number(value))
    for value in re.findall(r'(?:总共|一共|全诗共?|共)\s*('+NUMERAL+r')\s*句', text):
        require('lines', _number(value))
    for value in re.findall(r'[五七]言\s*('+NUMERAL+r')\s*句', text):
        require('lines', _number(value))
    if not requirements:
        return {}
    return dict(rhyme='押韵' in text, punctuation=False, linebreaks='一句一行' in text,
                acrostic='', keywords=[], **requirements)


def resolve_poetry_spec(messages, spec=None):
    inferred = infer_spec(messages)
    resolved = dict(inferred)
    if spec:
        for key in ('lines', 'width'):
            if key in inferred and key in spec and inferred[key] != spec[key]:
                raise ValueError(f'poetry_spec.{key}={spec[key]} 与用户要求 {inferred[key]} 不一致')
        resolved.update(spec)
        form = spec.get('form')
        if form in FORMS:
            lines, width = FORMS[form]
            for key, value in [('lines', lines), ('width', width)]:
                if key in resolved and resolved[key] != value:
                    raise ValueError(f'poetry_spec.{key} 与体裁 {form} 不一致')
                resolved[key] = value
    for key in ('lines', 'width'):
        if key in resolved and (type(resolved[key]) is not int or not 1 <= resolved[key] <= 64):
            raise ValueError(f'poetry_spec.{key} 必须是 1..64 的整数')
    return resolved


def legacy_general_rule_score(response):
    """逐项保持原 train_grpo.py 的长度、think、n-gram 评分规则。"""
    pieces = dict(length=0.5 if 20 <= len(response.strip()) <= 800 else -0.5)
    answer = response
    if '</think>' in response:
        thinking, answer = response.split('</think>', 1)
        pieces['thinking_length'] = 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
        pieces['thinking_format'] = 0.25 if response.count('</think>') == 1 else -0.25
        answer = answer.strip()
    tokens = re.findall(r'\w+|[^\w\s]', answer.lower())
    grams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    pieces['repetition'] = -min(0.5, (len(grams)-len(set(grams)))/len(grams)) if grams else 0.
    return sum(pieces.values()), pieces, answer


def legacy_score_messages(messages):
    # 原 LMForRewardModel 会把历史折叠为一次 user 提问，保留相同的评分输入。
    messages = [dict(role=m['role'], content=m['content'].strip()) for m in messages]
    history = '\n'.join(f"{m['role']}: {m['content']}" for m in messages[:-1])
    query = messages[-1]['content'] if messages else ''
    content = f'{history}\n以上是对话历史。我的新问题是：\n{query}' if history else query
    return [dict(role='user', content=content)]


def general_rule_score(response, repetition):
    """No bonus for thinking markup; score the entire generated response."""
    _, pieces, _ = legacy_general_rule_score(response)
    pieces.pop('thinking_length', None)
    pieces.pop('thinking_format', None)
    if '<think>' in response or '</think>' in response:
        valid = re.fullmatch(r'\s*<think>(.*?)</think>(.*)', response, re.S)
        if valid and response.count('<think>') == response.count('</think>') == 1:
            pieces['thinking'] = -1.5 if valid[1].strip() else -.25
            pieces['length'] = .5 if 20 <= len(valid[2].strip()) <= 800 else -.5
        else:
            pieces['thinking'] = -2.
    if repetition.get('allowed_requested_repetition'):
        pieces['repetition'] = 0.
    return sum(pieces.values()), pieces, response


def rhyme_fraction(verses):
    """普通话韵母近似，只作为弱奖励；不冒充平水韵/中古音格律判断。"""
    from pypinyin import pinyin, Style
    endings = [v[-1] for v in verses[1::2] if v]
    if len(endings) < 2 or len(set(endings)) < 2:
        return 0.0
    families = [('a', 'ia', 'ua'), ('o', 'e', 'uo'), ('ie', 've', 'ue'), ('ai', 'uai'),
                ('ei', 'ui', 'uei'), ('ao', 'iao'), ('ou', 'iu', 'iou'), ('an', 'ian', 'uan', 'van'),
                ('en', 'in', 'un', 'uen', 'vn'), ('ang', 'iang', 'uang'), ('eng', 'ing', 'ong', 'iong')]
    mapping = {v: str(i) for i, group in enumerate(families) for v in group}
    options = []
    for char in endings:
        readings = pinyin(char, style=Style.FINALS, heteronym=True, strict=False)[0]
        options.append({mapping.get(r, r) for r in readings if r})
    candidates = set().union(*options)
    best = max((sum(c in opts for opts in options) for c in candidates), default=0)
    return max(0.0, (best - 1) / (len(endings) - 1))


def poetry_rule_score(messages, response, spec=None):
    spec = resolve_poetry_spec(messages, spec)
    answer = response.strip()
    thinking_penalty = 0.0
    if '<think>' in answer or '</think>' in answer:
        thinking_penalty = 0.5
        answer = answer.split('</think>', 1)[-1].strip() if '</think>' in answer else ''
    if not answer:
        return -3.0, dict(empty=-3.0)
    repetition = rep_penalty(answer)
    if not spec or not spec.get('lines') or not spec.get('width'):
        score = -repetition - thinking_penalty
        return score, dict(repetition=-repetition, thinking=-thinking_penalty)
    expected, width = int(spec['lines']), int(spec['width'])
    segments = [s.strip() for s in SPLIT.split(answer) if s.strip()]
    verses = [''.join(HAN.findall(s)) for s in segments]
    count_score = max(0.0, 1 - abs(len(verses)-expected) / expected)
    correct_width = sum(len(v) == width for v in verses) / max(expected, len(verses), 1)
    # 每一句都只包含汉字，额外说明、序号、Markdown 不作为诗歌内容忽略。
    clean_fraction = sum(s == v for s, v in zip(segments, verses)) / max(len(segments), 1)
    exact = len(verses) == expected and all(len(v) == width for v in verses) and clean_fraction == 1
    physical = [s.strip() for s in answer.splitlines() if s.strip()]
    breaks = float(len(physical) == expected and all(len([s for s in SPLIT.split(v) if s.strip()]) == 1 for v in physical))
    punctuation = sum(bool(re.fullmatch(r'[\u3400-\u4dbf\u4e00-\u9fff]+' + ('，' if i % 2 == 0 else '。'), line))
                      for i, line in enumerate(physical)) / max(expected, len(physical), 1)
    duplicate = (len(verses)-len(set(verses))) / max(len(verses), 1)
    chars = ''.join(verses)
    faithful_reference = bool(spec.get('preserve_chars') and chars == spec.get('reference_body'))
    dominance = max(Counter(chars).values(), default=0) / max(len(chars), 1)
    # 防止“月月月月月”或复读同一句拿满格式分。
    degenerate = min(1.0, duplicate + max(0.0, dominance - 0.3) * 2)
    if faithful_reference:
        repetition, degenerate = 0., 0.
    quality_gate = 1.0 - degenerate
    pieces = dict(
        line_count=0.8 * count_score,
        line_width=1.5 * correct_width,
        clean_text=0.2 * clean_fraction,
        linebreaks=0.3 * breaks if spec.get('linebreaks') else 0.,
        punctuation=0.4 * punctuation if spec.get('punctuation') else 0.,
    )
    pieces = {key: value * quality_gate for key, value in pieces.items()}
    pieces['repetition'] = -repetition - 1.5 * degenerate
    pieces['thinking'] = -thinking_penalty
    pieces['extra_text'] = -0.8 * (1 - clean_fraction)
    keywords = spec.get('keywords', [])
    # 关键词仅作弱的覆盖信号，语义/意境交给内容评分；按次数堆词不加分。
    pieces['topic'] = 0.35 * float(any(k in chars for k in keywords)) * quality_gate if keywords else 0.
    acrostic = spec.get('acrostic', '')
    pieces['acrostic'] = 0.75 * sum(i < len(verses) and verses[i].startswith(c) for i, c in enumerate(acrostic)) / len(acrostic) * quality_gate if acrostic else 0.
    if acrostic:
        pieces['acrostic_mismatch'] = -1.5 * (1.0 - pieces['acrostic']/.75)
    pieces['rhyme'] = 0.25 * rhyme_fraction(verses) * quality_gate if spec.get('rhyme') and exact else 0.
    # Dense partial credit remains, but missing lines/width/breaks must have a cost.
    pieces['constraint_violation'] = -2.0 * (1.0 - min(count_score, correct_width))
    if spec.get('linebreaks'):
        pieces['missing_linebreaks'] = -0.75 * (1.0 - breaks)
    if spec.get('punctuation'):
        pieces['wrong_punctuation'] = -0.5 * (1.0 - punctuation)
    reference = spec.get('reference_body', '')
    if reference and spec.get('preserve_chars'):
        # 格式修复任务必须保持原文字序；生成另一首合格诗不算完成任务。
        restored = exact and (breaks or not spec.get('linebreaks')) and (punctuation == 1 or not spec.get('punctuation'))
        pieces['preserve_text'] = 1.0 if chars == reference and restored else (0.0 if chars == reference else -1.0)
    elif reference and spec.get('discourage_copy'):
        grams = {chars[i:i+4] for i in range(max(0, len(chars)-3))}
        source_grams = {reference[i:i+4] for i in range(max(0, len(reference)-3))}
        overlap = len(grams & source_grams)/max(len(grams), 1)
        pieces['copying'] = -2.0 * max(0., (overlap-0.4)/0.6)
    pieces['format_exact'] = float(exact and (breaks or not spec.get('linebreaks')) and
                                   (punctuation == 1 or not spec.get('punctuation')))
    # format_exact 是诊断指标，不重复加进总奖励。
    return sum(v for k, v in pieces.items() if k != 'format_exact'), pieces


class PoetryRewardModel:
    """固定本地 InternLM2 Reward；CPU 默认 float32，独立于训练 GPU。"""
    def __init__(self, path, device='cpu', max_length=1024, judge_model=None,
                 judge_url='http://localhost:11434', judge_timeout=120):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        dtype = torch.float32 if str(device).startswith('cpu') else torch.bfloat16
        self.model = AutoModel.from_pretrained(str(path), trust_remote_code=True, local_files_only=True,
                                               torch_dtype=dtype, attn_implementation='eager')
        self.model.to(device).eval().requires_grad_(False)
        self.max_length = max_length
        self.judge = None
        if judge_model:
            from trainer.grpo_poetry_judge import LocalPoetryJudge
            self.judge = LocalPoetryJudge(judge_model, judge_url, judge_timeout)

    def poetry_rating(self, messages, response):
        if self.judge is None:
            raise RuntimeError('Poetry training requires a calibrated local semantic judge')
        if not response.strip():
            return dict(coherence=0, quality=0., reason='empty response', judge=self.judge.provenance)
        return self.judge.rate(messages, response)

    @torch.no_grad()
    def get_score(self, messages, response):
        # 用官方模板生成 reward_token，采用普通 forward 以显式限制评分上下文，禁用 KV cache。
        chat = [dict(role=m['role'], content=m['content']) for m in messages]
        chat.append(dict(role='assistant', content=response))
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        encoded = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        if encoded.input_ids[0, -1].item() != self.model.reward_token_id:
            encoded['input_ids'] = torch.cat([encoded.input_ids, torch.tensor([[self.model.reward_token_id]])], dim=1)
            encoded['attention_mask'] = torch.ones_like(encoded.input_ids)
        if encoded.input_ids.shape[1] > self.max_length:
            raise ValueError('评分输入过长；请增大 --reward_max_length，拒绝截断诗歌要求或正文')
        encoded = {k: v.to(self.model.device) for k, v in encoded.items() if k in ('input_ids', 'attention_mask')}
        output = self.model(**encoded, use_cache=False)
        # 官方 forward 已提取最后的 reward token，logits 形状为 [batch, 1]。
        score = float(output.logits[0, 0].float().cpu())
        if not math.isfinite(score):
            raise FloatingPointError('评分模型返回非有限值')
        return score


def calculate_rewards(rows, responses, reward_model, num_generations, device='cpu', model_weight=1.0,
                      completion_ended=None, repetition_weight=3.0):
    if len(responses) != len(rows) * num_generations:
        raise ValueError('prompt 与候选回答数量不匹配')
    if completion_ended is not None and len(completion_ended) != len(responses):
        raise ValueError('EOS flags and responses must align')
    totals, details = [], []
    for i, response in enumerate(responses):
        row = rows[i // num_generations]
        repetition = response_repetition(row, response)
        # 通用通道不套用诗歌格律；思考标签没有额外正奖励。
        is_poetry = row.get('channel') == 'poetry' or (
            row.get('channel') != 'general' and bool(row.get('spec')))
        if is_poetry:
            spec = resolve_poetry_spec(row['messages'], row.get('spec'))
            if not spec.get('lines') or not spec.get('width'):
                raise ValueError('古诗数据必须明确句数和每句字数，请填写 poetry_spec.lines/width')
            rule, pieces = poetry_rule_score(row['messages'], response, spec)
            rating = None
            if hasattr(reward_model, 'poetry_rating'):
                rating = reward_model.poetry_rating(row['messages'], response)
                quality = rating['quality']
                positive = sum(max(0., v) for k, v in pieces.items() if k != 'format_exact')
                task_violated = ((spec.get('preserve_chars') and pieces.get('preserve_text') != 1.)
                    or (spec.get('discourage_copy') and pieces.get('copying', 0.) < 0.)
                    or (spec.get('acrostic') and pieces.get('acrostic', 0.) < .75-1e-8))
                positive_gate = quality if quality >= 0.5 and not task_violated else 0.
                pieces['semantic_gate'] = -(1.0 - positive_gate) * positive
                raw = float(rating['coherence'])
                bounded = (2.0 * quality - 1.0) * model_weight
                if task_violated:
                    bounded = min(0., bounded)
            else:
                # Unit-test fake scorers retain the small injectable scoring interface.
                raw = reward_model.get_score(row['messages'], response) if reward_model is not None else 0.
                bounded = math.tanh(raw / 3.) * model_weight
            if not pieces.get('format_exact'):
                bounded = min(0., bounded)
            if completion_ended is not None and not completion_ended[i]:
                pieces['truncated'] = -1.5
            rule = sum(v for k, v in pieces.items() if k != 'format_exact')
        else:
            rule, pieces, answer = general_rule_score(response, repetition)
            score_messages = legacy_score_messages(row['messages'])
            raw = reward_model.get_score(score_messages, answer) if reward_model is not None else 0.
            bounded = max(-3., min(3., raw))
            if completion_ended is not None and not completion_ended[i]:
                pieces['truncated'] = -1.
                rule = sum(pieces.values())
        pieces['repetition_penalty'] = -repetition_weight * repetition['severity']
        rule += pieces['repetition_penalty']
        total = rule + bounded
        if repetition_weight and repetition['excessive']:
            # A semantic scorer must not turn a degenerate loop into a positive target.
            ceiling = -repetition_weight * repetition['severity']
            correction = min(0., ceiling-total)
            pieces['repetition_positive_gate'] = correction
            total += correction
        if not math.isfinite(total):
            raise FloatingPointError('奖励出现非有限值')
        qualified = False
        if is_poetry and rating is not None:
            qualified = (pieces.get('format_exact') == 1 and rating['quality'] >= .75
                and (completion_ended is None or completion_ended[i])
                and (not spec.get('preserve_chars') or pieces.get('preserve_text') == 1.)
                and (not spec.get('discourage_copy') or pieces.get('copying', 0.) >= 0.)
                and (not spec.get('acrostic') or pieces.get('acrostic', 0.) >= .75-1e-8)
                and not repetition['excessive'])
        totals.append(total)
        details.append(dict(total=total, rules=pieces, model_raw=raw, model_bounded=bounded,
                            repetition=repetition,
                            reward_mode=('poetry_reviewed_v4' if is_poetry else 'general_reviewed_v4'),
                            **(dict(semantic=rating, qualified=bool(qualified)) if is_poetry and rating is not None else {})))
    return torch.tensor(totals, device=device, dtype=torch.float32), details
