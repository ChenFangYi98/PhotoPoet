# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""可复核的能力规则和诊断；导入时不依赖 torch，不执行模型生成的代码。"""
import json
import math
import re
from collections import Counter


def strict_json(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('重复 JSON 键')
            obj[key] = value
        return obj

    def invalid(value):
        raise ValueError(f'非标准 JSON 常量: {value}')
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def same_json(a, b):
    # Python 的 True == 1；结构化抽取必须区分这两者。
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return a == b


def grade_capability(case, response):
    r = case['rule']
    typ = r['type']
    text = response.strip()
    details = {}
    if typ == 'numeric':
        candidate = re.sub(r'^(?:答案|答)\s*[：:]\s*', '', text).rstrip('。')
        valid = re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', candidate)
        value = float(candidate) if valid else None
        passed = value is not None and math.isfinite(value) and math.isclose(
            value, float(r['answer']), rel_tol=0, abs_tol=r.get('atol', 1e-6))
        details.update(数值解析成功=bool(valid), 准确匹配=100.0 * passed)
    elif typ == 'choice':
        passed = text.rstrip('。.') == r['answer']
        # 内容指标容忍解释，但多个互相矛盾的选项不会被判正确。
        labels = set(re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', text))
        extracted = next(iter(labels)) if len(labels) == 1 else None
        details.update(准确匹配=100.0 * passed, 选项内容正确=100.0 * (extracted == r['answer']),
                       提取选项=extracted)
    elif typ == 'json':
        try:
            obj = strict_json(text)
            valid = True
            passed = same_json(obj, r['expected'])
        except (ValueError, TypeError, RecursionError):
            valid, passed = False, False
        details.update(JSON格式有效=valid, 准确匹配=100.0 * passed)
    elif typ == 'constraints':
        checks = {}
        for key, expected in r['checks'].items():
            if key == 'contains':
                checks[key] = all(s in text for s in expected)
            elif key == 'not_contains':
                checks[key] = all(s not in text for s in expected)
            elif key == 'line_count':
                checks[key] = len(text.splitlines()) == expected
            elif key == 'max_chars':
                checks[key] = 0 < len(text) <= expected
            elif key == 'min_chars':
                checks[key] = len(text) >= expected
            elif key == 'starts_with':
                checks[key] = text.startswith(expected)
            elif key == 'ends_with':
                checks[key] = text.endswith(expected)
            elif key == 'fullmatch':
                checks[key] = re.fullmatch(expected, text) is not None
            else:
                raise ValueError(f'未知约束: {key}')
        if not checks:
            raise ValueError('constraints 不能为空')
        passed = bool(text) and all(checks.values())
        details.update(约束检查=checks, 全约束通过=passed)
        return dict(auto_score=round(100 * sum(checks.values()) / len(checks), 2) if text else 0.0,
                    measured_coverage_pct=100, details=details, notes=['仅检查明确列出的形式约束，语义质量需另评'])
    else:
        raise ValueError(f'未知评分类型: {typ}')
    details['全约束通过'] = bool(passed)
    return dict(auto_score=100.0 * passed, measured_coverage_pct=100, details=details, notes=[])


def response_diagnostics(text):
    chars = list(re.sub(r'\s+', '', text))
    grams = [tuple(chars[i:i + 4]) for i in range(max(0, len(chars) - 3))]
    repeated = 1 - len(set(grams)) / len(grams) if grams else 0.0
    return dict(empty_response=not text.strip(), response_chars=len(text),
                repeated_char_4gram_ratio=round(repeated, 4),
                unclosed_thinking='<think>' in text and '</think>' not in text)


def infer_architecture(state):
    embedding = state['model.embed_tokens.weight']
    layers = {int(m.group(1)) for key in state if (m := re.match(r'model.layers\.(\d+)\.', key))}
    if not layers or layers != set(range(max(layers) + 1)):
        raise ValueError('checkpoint 层编号不连续')
    prefix = 'model.layers.0.'
    head_dim = state[prefix + 'self_attn.q_norm.weight'].shape[0]
    moe = prefix + 'mlp.gate.weight' in state
    result = dict(vocab_size=embedding.shape[0], hidden_size=embedding.shape[1],
                  num_hidden_layers=len(layers), use_moe=moe, head_dim=head_dim,
                  num_attention_heads=state[prefix + 'self_attn.q_proj.weight'].shape[0] // head_dim,
                  num_key_value_heads=state[prefix + 'self_attn.k_proj.weight'].shape[0] // head_dim)
    if moe:
        result.update(num_experts=state[prefix + 'mlp.gate.weight'].shape[0],
                      moe_intermediate_size=state[prefix + 'mlp.experts.0.gate_proj.weight'].shape[0])
    else:
        result['intermediate_size'] = state[prefix + 'mlp.gate_proj.weight'].shape[0]
    # top-k、RoPE 等训练超参数不能仅靠张量形状识别；沿用本项目配置。
    return result


def select_cases(cases, categories='', limit=0, stratified=False):
    if categories:
        selected = set(categories.split(','))
        unknown = selected - {c['category'] for c in cases}
        if unknown:
            raise ValueError(f'不存在的分类: {sorted(unknown)}')
        cases = [c for c in cases if c['category'] in selected]
    if limit and stratified:
        groups = {}
        for c in cases:
            groups.setdefault(c['category'], []).append(c)
        cases = [group[i] for i in range(max(map(len, groups.values())))
                 for group in groups.values() if i < len(group)]
    return cases[:limit] if limit else cases


def choice_likelihood(model, tokenizer, dev, prompt, case, max_input_tokens):
    """固定 prompt token 边界，比较候选字母的总条件 log P；不评分题干。"""
    import torch
    import torch.nn.functional as F
    prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
    if not prefix or len(prefix) > max_input_tokens:
        raise ValueError('选择题输入为空或超出 max_input_tokens')
    scores, token_counts = {}, {}
    for label in case['rule']['options']:
        suffix = tokenizer(label, add_special_tokens=False)['input_ids']
        if not suffix:
            raise ValueError(f'候选字母编码为空: {label}')
        x = torch.tensor([prefix + suffix], device=dev)
        with torch.inference_mode():
            logits = model(input_ids=x).logits[0, len(prefix) - 1:-1].float()
            value = -F.cross_entropy(logits, torch.tensor(suffix, device=dev), reduction='sum').item()
        if not math.isfinite(value):
            raise RuntimeError('选择题产生非有限 log P')
        scores[label], token_counts[label] = round(value, 6), len(suffix)
    predicted = max(scores, key=scores.get)
    return dict(predicted=predicted, answer=case['rule']['answer'], log_probs=scores,
                candidate_tokens=token_counts, correct=predicted == case['rule']['answer'])
