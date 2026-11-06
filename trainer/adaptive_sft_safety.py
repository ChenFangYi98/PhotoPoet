# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""SFT 审计修正：保守隔离、完整上下文策略和完整恢复指纹。

规则只能隔离明确的风险模式，不能证明事实正确。原始 JSONL 不改写。
"""
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

from trainer.continue_pretrain_utils import sha256_file

SAFETY_VERSION = 'sft_safety_v1'
_DEICTIC = re.compile(r'(这句|这首|以下|下面|上述).*(诗|词).*(作者|谁写)|(?:诗|词).*作者.*(这句|这首|以下|下面)')
_CLARIFY = re.compile(r'请.*提供|提供.*(?:诗|内容)|无法|不能确定|未提供|缺少|更多.*内容|没有.*(?:诗|内容)|需要.*(?:诗|内容)')
_EMPTY_OPTION = re.compile(r'(?:\\n|\n)\s*[ABCD][.．、]\s*[$（(]?\s*$')


def quarantine_reason(row):
    """仅检查无上文的单轮候选，保留要求补充资料的正确回答。"""
    messages = [m for m in row.get('conversations', []) if m.get('role') != 'system']
    if len(messages) != 2 or [m.get('role') for m in messages] != ['user', 'assistant']:
        return None
    prompt, answer = (m.get('content', '') for m in messages)
    if (_DEICTIC.search(prompt) and len(prompt) < 50
            and not re.search(r'《|“|「|\n|\\n', prompt) and not _CLARIFY.search(answer)):
        return 'missing_poem_clue'
    if _EMPTY_OPTION.search(prompt) and not _CLARIFY.search(answer):
        return 'empty_last_option'
    return None


def training_signature(root, tokenizer, config, manifest):
    """模板、模型前向、AMP/LR/保存、依赖版本都属于恢复协议。"""
    files = ['trainer/train_adaptive_sft.py', 'trainer/adaptive_sft_data.py',
             'trainer/adaptive_sft_objective.py', 'trainer/adaptive_sft_sampling.py',
             'trainer/adaptive_sft_safety.py', 'trainer/sft_data.py',
             'trainer/factual_sft_utils.py', 'trainer/trainer_utils.py',
             'trainer/train_full_sft.py', 'trainer/continue_pretrain_utils.py',
             'model/model_mindchen.py', 'model/checkpoint_config.py', 'dataset/lm_dataset.py']
    def digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        default=str).encode('utf-8')).hexdigest()
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'transformers', 'tokenizers', 'datasets', 'numpy')}
    return dict(protocol=SAFETY_VERSION, manifest=sha256_file(manifest),
                code={name: sha256_file(Path(root)/name) for name in files}, versions=versions,
                model_config=digest(config.to_dict()),
                tokenizer=digest(dict(backend=tokenizer.backend_tokenizer.to_str(),
                    chat_template=tokenizer.chat_template, special_tokens=tokenizer.special_tokens_map,
                    padding_side=tokenizer.padding_side, truncation_side=tokenizer.truncation_side)))


def stratified_indices(corpus, channel, candidates, count, rng):
    """按真实任务类型轮转取题，预算足够时覆盖每一类；保持不重复。"""
    buckets = {}
    for index in candidates:
        row, _ = corpus.read(channel, index)
        task = row.get('task_meta', {}).get('kind', 'legacy_numeric')
        buckets.setdefault(task, []).append(index)
    for values in buckets.values():
        rng.shuffle(values)
    names = sorted(buckets)
    result = []
    while len(result) < count:
        advanced = False
        for name in names:
            if buckets[name] and len(result) < count:
                result.append(buckets[name].pop())
                advanced = True
        if not advanced:
            break
    return result


def generation_diagnostics(texts):
    """只记录机械重复/差异，不把字符多样性冒充创造力或文学质量。"""
    grams = Counter(g for text in texts for g in zip(text, text[1:]))
    total = sum(grams.values())
    return dict(distinct_2=len(grams)/max(1, total),
                unique_response_fraction=len(set(texts))/max(1, len(texts)),
                empty_responses=sum(not text.strip() for text in texts))
