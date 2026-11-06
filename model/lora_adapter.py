# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Scoped LoRA for native MindChen/MindChen-V checkpoints; legacy API is untouched."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    def __init__(self, linear, rank=8, alpha=16.0, dropout=0.0):
        nn.Module.__init__(self)
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('rank >= 1, alpha > 0 and 0 <= dropout < 1 are required')
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight, self.bias = linear.weight, linear.bias
        self.rank, self.alpha, self.scale = rank, float(alpha), float(alpha) / rank
        self.lora_dropout = nn.Dropout(dropout)
        options = dict(device=linear.weight.device, dtype=linear.weight.dtype)
        self.lora_A = nn.Linear(self.in_features, rank, bias=False, **options)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False, **options)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale


def attach_lora(model, rank=8, alpha=16.0, dropout=0.0, targets=('q_proj', 'v_proj')):
    """Only model.layers.* is eligible; vision modules and lm_head are excluded."""
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('LoRA is already attached')
    targets = tuple(targets)
    selected = [(n, m) for n, m in model.named_modules()
                if n.startswith('model.layers.') and n.rsplit('.', 1)[-1] in targets
                and isinstance(m, nn.Linear)]
    if not selected:
        raise ValueError(f'No language backbone Linear matched targets={targets}')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = []
    for name, linear in selected:
        parent, leaf = name.rsplit('.', 1)
        model.get_submodule(parent).__setattr__(leaf, LoRALinear(linear, rank, alpha, dropout))
        names.append(name)
    return names


def adapter_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if '.lora_A.' in k or '.lora_B.' in k}


def load_adapter(model, payload):
    expected = adapter_state(model)
    state = payload['adapter']
    if set(expected) != set(state):
        raise ValueError('Adapter module names do not match the attached LoRA configuration')
    if any(expected[k].shape != state[k].shape for k in expected):
        raise ValueError('Adapter tensor shapes do not match')
    model.load_state_dict(state, strict=False)


def load_bound_adapter(model, path, checkpoint):
    from trainer.lora_training import sha256
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'native_lora_v1':
        raise ValueError('Not a native_lora_v1 adapter')
    if payload['signature']['base_sha256'] != sha256(checkpoint):
        raise ValueError('This adapter was trained against a different base checkpoint')
    if payload['model_config'] != model.config.to_dict():
        # Transformers may add bookkeeping keys during loading; compare the
        # architecture and routing fields that affect the forward computation.
        for key in ('hidden_size','num_hidden_layers','use_moe','num_experts','num_experts_per_tok',
                    'norm_topk_prob','num_attention_heads','num_key_value_heads','image_token_len'):
            if key in payload['model_config'] and payload['model_config'][key] != getattr(model.config, key, None):
                raise ValueError(f'Adapter/base runtime config differs: {key}')
    spec = payload['lora']
    attach_lora(model, spec['rank'], spec['alpha'], spec['dropout'], spec['targets'])
    load_adapter(model, payload)
    return payload


def merged_state(model):
    """Return a standalone state dict without modifying the live model or its base."""
    result = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
              if '.lora_A.' not in k and '.lora_B.' not in k and not k.startswith('vision_encoder.')}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            delta = module.lora_B.weight.detach().float().cpu() @ module.lora_A.weight.detach().float().cpu()
            key = name + '.weight'
            result[key] = (result[key].float() + delta * module.scale).to(result[key].dtype)
    return result


def atomic_save(payload, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(target)


def save_merged(model, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite merged output: {target}')
    atomic_save(merged_state(model), target)
    from trainer.lora_training import sha256
    sidecar = dict(schema='mindchen_native_config_v1', weight_file=target.name,
                   weight_sha256=sha256(target), model_config=model.config.to_dict())
    Path(str(target) + '.config.json').write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding='utf8')
