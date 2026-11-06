# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""原生 .pth 的配置伴随文件：张量形状无法识别 top-k 概率归一化方式。"""
import hashlib
import json
import os
from pathlib import Path

from model.model_mindchen import MindChenConfig

from model.checkpoint_compat import CONFIG_SCHEMA, SUPPORTED_CONFIG_SCHEMAS, normalize_model_config

SCHEMA = CONFIG_SCHEMA


def weight_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(weight):
    return Path(str(weight)+'.config.json')


def write_checkpoint_config(weight, config, checksum=None):
    """每个 latest/best 单独绑定实际权重哈希；复制权重时须同时复制此文件。"""
    payload = dict(schema=SCHEMA, weight_file=Path(weight).name,
                   weight_sha256=checksum or weight_sha256(weight), model_config=config.to_dict())
    target = sidecar_path(weight)
    temporary = target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,target)
    return payload


def load_checkpoint_config(weight, fallback=None, config_path='', norm_topk_prob=None,
                           expected_architecture=None, inference_rope_scaling=None):
    """优先绑定配置；无配置的历史权重按旧归一化行为回退，绝不套用新 SFT 默认值。

    显式 --norm_topk_prob 仅用于无配置历史权重；有配置时禁止冲突覆盖。
    文件夹级 model_config.json 需要显式指定，避免给复制进来的其他权重误用。
    """
    candidate = Path(config_path) if config_path else sidecar_path(weight)
    saved = None
    if config_path or candidate.exists():
        payload = json.loads(candidate.read_text(encoding='utf-8'))
        if 'schema' in payload:
            if payload['schema'] not in SUPPORTED_CONFIG_SCHEMAS:
                raise ValueError('未知权重配置协议：'+str(candidate))
            if payload['weight_sha256'] != weight_sha256(weight):
                raise ValueError('权重与配置 SHA256 不匹配：'+str(candidate))
            saved = normalize_model_config(payload['model_config'])
        else:
            if not config_path:
                raise ValueError('自动配置必须包含权重哈希：'+str(candidate))
            saved = normalize_model_config(payload)
        if saved.get('model_type','mindchen') != 'mindchen':
            raise ValueError('配置不是 MindChen 模型')
    values = dict(fallback or {})
    # 旧的裸 .pth 无法推断该字段，保持历史默认值；新训练总是写出明确的 False。
    values['norm_topk_prob'] = True
    if saved is not None:
        if 'norm_topk_prob' not in saved:
            raise ValueError('配置缺少 norm_topk_prob，不能确定路由语义')
        values.update(saved)
    if values['norm_topk_prob'] not in (True,False,0,1):
        raise ValueError('norm_topk_prob 必须是布尔值')
    if norm_topk_prob is not None:
        if saved is not None and bool(norm_topk_prob) != bool(values['norm_topk_prob']):
            raise ValueError('--norm_topk_prob 与保存配置冲突，拒绝改变已训练模型的路由计算')
        values['norm_topk_prob'] = bool(norm_topk_prob)
    if inference_rope_scaling is not None:
        values['inference_rope_scaling'] = bool(inference_rope_scaling)
    config = MindChenConfig(**values)
    for key,value in (expected_architecture or {}).items():
        if getattr(config,key) != value:
            raise ValueError(f'保存配置 {key}={getattr(config,key)} 与权重推断值 {value} 不符')
    return config, dict(source=str(candidate) if saved is not None else 'legacy_fallback',
        norm_topk_prob=config.norm_topk_prob,
        weight_bound=saved is not None and payload.get('schema') in SUPPORTED_CONFIG_SCHEMAS)
