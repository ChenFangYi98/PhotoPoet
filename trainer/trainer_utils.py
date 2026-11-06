from model.checkpoint_compat import SUPPORTED_CONFIG_SCHEMAS
# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""
训练工具函数集合。

训练脚本里很多重复逻辑都放在这里：
- 日志只在主进程打印
- 初始化分布式训练
- 固定随机种子
- 加载 tokenizer 和模型权重
- 保存/恢复 checkpoint
- 断点续训时跳过已经训练过的 batch
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_mindchen import MindChenForCausalLM

def get_model_params(model, config, ignore_patterns=()):
    """打印模型参数量。

    MoE 模型有“总参数量”和“单 token 实际激活参数量”的区别：
    总参数量包含所有专家；激活参数量只包含当前 token 会用到的专家。
    """

    def should_count(name): return not any(pattern in name for pattern in ignore_patterns)
    total = sum(p.numel() for name, p in model.named_parameters() if should_count(name)) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """判断当前进程是否是主进程。

    单卡训练时没有初始化 dist，直接视为主进程。
    多卡 DDP 时只有 rank=0 负责打印日志和保存模型，避免多进程重复写文件。
    """

    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """只在主进程打印日志。"""

    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """余弦退火学习率。

    训练初期学习率接近 lr，后期逐渐降到 lr 的 0.1 倍附近。
    这样通常比固定学习率更稳定。
    """

    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """初始化 DDP 分布式训练。

    torchrun 会自动注入 RANK/LOCAL_RANK 等环境变量。
    如果没有这些变量，说明是普通单进程训练，直接返回 0。
    """

    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """固定随机种子，尽量让实验可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', _exclude_prefixes=(), **kwargs):
    """保存或加载训练断点。

    model 不为 None：保存模式。
    model 为 None：加载模式，只检查是否存在 resume checkpoint。

    保存两份文件：
    - out/xxx.pth：常规模型权重，方便推理或下阶段训练。
    - checkpoints/xxx_resume.pth：续训状态，包含 optimizer、epoch、step、scaler 等。
    """

    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # DDP 会把真实模型包在 .module 里；torch.compile 又可能包在 _orig_mod 里。
        # 保存前先取回原始模型，避免 state_dict key 多一层包装。
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        # 保存 half 权重能显著减小文件体积。
        inference_state = {k: v.half().cpu() for k, v in state_dict.items() if not k.startswith(_exclude_prefixes)}
        # 先写 .tmp 再 os.replace，降低保存中断导致文件损坏的概率。
        if not _exclude_prefixes:
            state_dict = inference_state
        ckp_tmp = ckp_path + '.tmp'
        torch.save(inference_state, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                # 额外对象例如 scaler、scheduler、critic_model 也可以一起保存。
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                # 如果续训时 GPU 数量变化，一个 epoch 内 step 数也会变化，这里按 world_size 粗略换算。
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """初始化 tokenizer 和 MindChen 模型，并按需加载已有权重。"""

    # tokenizer_path 默认指向项目自带 tokenizer。
    import os

    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    tokenizer_path = os.path.join(PROJECT_ROOT, "model")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True
    )




    
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    weight_path = None
    if from_weight != 'none':
        from pathlib import Path
        moe_suffix = '_moe' if lm_config.use_moe else ''
        candidate = Path(from_weight).expanduser()
        if candidate.is_absolute(): weight_path = candidate
        elif candidate.suffix == '.pth': weight_path = Path(PROJECT_ROOT)/candidate
        else: weight_path = Path(save_dir)/f'{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        if Path(str(weight_path)+'.config.json').exists():
            from model.checkpoint_config import load_checkpoint_config
            lm_config, _ = load_checkpoint_config(weight_path, fallback=lm_config.to_dict(), expected_architecture={
                key:getattr(lm_config,key) for key in ('hidden_size','num_hidden_layers','use_moe')})
    model = MindChenForCausalLM(lm_config)

    if weight_path is not None:
        # strict=False 允许某些阶段加载不完全匹配的权重，例如 LoRA/蒸馏/结构实验。
        weights = torch.load(weight_path, map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """断点续训用的 batch sampler。

    假设上次训练在第 step 个 batch 保存，这次恢复时就跳过前 step 个 batch，
    从下一个 batch 继续训练，避免重复跑已经训练过的数据。
    """

    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """Reward Model 包装器。

    PPO/GRPO/RLAIF 阶段会用它给模型生成的回答打分。
    注意：这里假设外部 reward model 实现了 get_score(tokenizer, messages) 方法。
    """

    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        # 把历史对话整理成 reward model 更容易理解的评分输入。
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        # 分数裁剪到 [-3, 3]，避免极端值把强化学习训练带偏。
        return max(min(score, 3.0), -3.0)


def init_vlm_model(vlm_config, from_weight='pretrain_vlm', tokenizer_path=PROJECT_ROOT / 'model', vision_model_path=PROJECT_ROOT / 'model/siglip2-base-p32-256-ve', save_dir=PROJECT_ROOT / 'out', device='cuda', freeze_llm=0):
    """加载分词器、视觉编码器和权重，并设置冻结策略。"""
    from model.model_vlm import MindChenVLM
    weight_path = None
    if from_weight != 'none':
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        candidate = Path(from_weight).expanduser()
        weight_path = candidate if candidate.is_absolute() else (PROJECT_ROOT/candidate if candidate.suffix == '.pth' else Path(save_dir)/f'{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth')
        sidecar = Path(str(weight_path)+'.config.json')
        if sidecar.exists():
            import json
            from trainer.lora_training import sha256
            from model.model_vlm import VLMConfig
            saved = json.loads(sidecar.read_text(encoding='utf-8-sig'))
            if saved.get('schema') not in SUPPORTED_CONFIG_SCHEMAS or saved.get('weight_sha256') != sha256(weight_path):
                raise ValueError('权重与配置 SHA256 不匹配')
            values = vlm_config.to_dict(); values.update(saved['model_config']); values.pop('model_type',None)
            for key in ('hidden_size','num_hidden_layers','use_moe'):
                if values[key] != getattr(vlm_config,key):raise ValueError(f'训练参数与保存配置冲突: {key}')
            vlm_config = VLMConfig(**values)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MindChenVLM(vlm_config, vision_model_path=str(vision_model_path))
    if model.vision_encoder is None or model.processor is None:
        raise FileNotFoundError(f'无法加载视觉编码器或预处理器：{vision_model_path}，请检查文件是否完整。')
    
    if weight_path is not None:
        weights = torch.load(weight_path, map_location='cpu', weights_only=True)
        if isinstance(weights, dict) and isinstance(weights.get('model'), dict):
            weights = weights['model']
        incompatible = model.load_state_dict(weights, strict=False)
        missing_llm = [key for key in incompatible.missing_keys if not key.startswith(('vision_encoder.', 'vision_proj.'))]
        if missing_llm or incompatible.unexpected_keys:
            raise ValueError(f'权重与模型不匹配：missing={missing_llm}, unexpected={incompatible.unexpected_keys}')
        Logger(f'Loaded initial weights: {weight_path}')
        if any(key.startswith('vision_proj.') for key in incompatible.missing_keys):
            Logger('Language-only checkpoint: vision_proj is newly initialized.')
    
    # 1、全部冻结，只打开vision_proj梯度
    for name, param in model.named_parameters():
        if 'vision_proj' not in name:
            param.requires_grad = False

    # 2、判断策略
    if freeze_llm == 0:
        for name, param in model.named_parameters():
            if 'vision_encoder' not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        last_idx = vlm_config.num_hidden_layers - 1
        for name, param in model.model.named_parameters():
            if 'layers.0.' in name or f'layers.{last_idx}.' in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        pass

    get_model_params(model, vlm_config, ignore_patterns=('vision_encoder',))
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vlm_collate_fn(batch):
    """将多个样本堆叠成批次；批内图片数量必须相同。"""
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values


def vlm_checkpoint(vlm_config, *args, **kwargs):
    """Use the shared saver, omitting the frozen encoder from inference exports."""
    return lm_checkpoint(vlm_config, *args, _exclude_prefixes=('vision_encoder.',), **kwargs)
