# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""修正后的自适应 SFT 抽样；旧抽样实现留在 adaptive_sft_data.py 供旧断点复现。

训练：保留任务配额和事实来源配比，open/general 内按实际条数抽取。
开发：任务内对全部条目均匀、不放回抽取，不按小通道等概率放大。
"""
from bisect import bisect_right
from collections import Counter
import math
import random

from trainer.adaptive_sft_data import TASKS, quota_counts

SAMPLING_VERSION = 'task_stratified_size_weighted_v2'
CHECKPOINT_FORMAT = 'adaptive_sft_sampling_v2'


def _channels(corpus, task, split):
    if split not in ('train', 'valid'):
        raise ValueError('split 必须为 train 或 valid')
    channels = [key for key, index in corpus.indices.items()
                if key.startswith(split+'_') and key.endswith('_'+TASKS[task]) and len(index)
                and not (split == 'valid' and key == 'valid_author_general')]
    if not channels:
        raise ValueError(f'{split} 缺少 {TASKS[task]} 数据')
    return channels


def _family(channel):
    return 'author' if '_author_' in channel else 'verified' if '_verified_' in channel else 'base'


def _source_weights(corpus, channels, task, split):
    # 按通道条数选择，再在通道中均匀选行，等价于在该任务的全部行中均匀抽取。
    if split == 'valid' or task != 0:
        return [len(corpus.indices[key]) for key in channels]
    ratios = getattr(corpus, 'manifest', {}).get('fact_source_ratios',
                                               {'author': .45, 'verified': .20, 'base': .35})
    if any(not math.isfinite(float(ratios.get(_family(key), 0)))
           or float(ratios.get(_family(key), 0)) <= 0 for key in channels):
        raise ValueError('事实来源比例必须为有限正数')
    totals = Counter()
    for key in channels:
        totals[_family(key)] += len(corpus.indices[key])
    return [ratios[_family(key)]*len(corpus.indices[key])/totals[_family(key)] for key in channels]


def sampling_plan(corpus, count, ratios, split='train'):
    """可审计的每通道期望抽样占比；不是 token 占比或已抽样次数。"""
    plan = {}
    for task, quota in enumerate(quota_counts(count, ratios)):
        channels = _channels(corpus, task, split)
        weights = _source_weights(corpus, channels, task, split)
        denominator = sum(weights)
        for key, weight in zip(channels, weights):
            expected = quota*weight/denominator
            plan[key] = dict(rows=len(corpus.indices[key]), expected_samples_per_update=expected,
                             expected_sample_fraction=expected/count)
    return plan


def draw_examples(corpus, count, ratios, seed, split='train'):
    rng = random.Random(seed)
    result = []
    for task, quota in enumerate(quota_counts(count, ratios)):
        channels = _channels(corpus, task, split)
        if split == 'valid':
            # 对虚拟拼接数组采样；只构造少量随机下标，不展开几百万个候选。
            ends, size = [], 0
            for key in channels:
                size += len(corpus.indices[key])
                ends.append(size)
            if quota > size:
                raise ValueError(f'开发集 {TASKS[task]} 只有 {size} 条，无法无重复抽取 {quota} 条')
            for position in rng.sample(range(size), quota):
                owner = bisect_right(ends, position)
                result.append((channels[owner], position-(ends[owner-1] if owner else 0)))
        else:
            weights = _source_weights(corpus, channels, task, split)
            for _ in range(quota):
                key = rng.choices(channels, weights=weights, k=1)[0]
                result.append((key, rng.randrange(len(corpus.indices[key]))))
    rng.shuffle(result)
    return result


def require_checkpoint_format(checkpoint):
    if checkpoint.get('format_version') != CHECKPOINT_FORMAT:
        raise ValueError('旧采样断点不能直接恢复到修正版。请新建运行；若仅复现旧 v3 运行，使用 '
                         'trainer/train_adaptive_sft_v3_legacy.py --resume <旧断点>。'
                         '复用旧纯 LM 权重可用 --init_weight_path <adaptive_768_moe_best.pth>，'
                         '但会新建优化器/验证基线，不会消除旧训练已产生的偏差。')
