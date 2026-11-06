# Modified by chenfangyi for the unified MindChen release (2026-11-06).
import torch
from torch import optim, nn


# LoRA（Low-Rank Adaptation）是一种参数高效微调方法。
# 它的核心想法是：不直接改原始大矩阵 W，而是在旁边加一个很小的低秩增量 ΔW。
# 原始线性层输出：y = W x
# 加 LoRA 后输出：y = W x + B(Ax)
# 其中 A 把维度降到 rank，B 再升回原维度；rank 越小，可训练参数越少。


class LoRA(nn.Module):
    """一个最小实现的 LoRA 分支。

    in_features/out_features 和被挂载的 nn.Linear 一致，rank 控制低秩瓶颈大小。
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        # A: [in_features -> rank]，负责把输入压缩到低维空间。
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        # B: [rank -> out_features]，负责把低维增量还原到原输出维度。
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化：刚挂上 LoRA 时，B(Ax)=0，不会立刻改变原模型输出。
        self.B.weight.data.zero_()

    def forward(self, x):
        # 只计算低秩增量，不包含原始线性层输出。
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """给模型中符合条件的线性层挂上 LoRA 分支。

    当前实现只给方阵 Linear 挂 LoRA，即 in_features == out_features。
    在 MindChen 中这会覆盖一部分注意力/FFN 投影层，训练时只更新这些 LoRA 参数。
    """

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # model.device 来自 transformers 的 PreTrainedModel，确保 LoRA 分支和原模型在同一设备上。
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            # 把 lora 注册为当前 Linear 的子模块，之后 state_dict 和 parameters 都能找到它。
            setattr(module, "lora", lora)
            # 保存原始 forward，后面要在它的输出上加 LoRA 增量。
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 原始线性层输出 + LoRA 低秩增量。
                return layer1(x) + layer2(x)

            # monkey-patch 当前 Linear 的 forward。注意：这也是 train_lora.py 里关闭 torch.compile 的原因。
            module.forward = forward_with_lora


def load_lora(model, path):
    """从磁盘加载 LoRA 权重到已经 apply_lora 的模型中。"""

    state_dict = torch.load(path, map_location=model.device)
    # 兼容 DDP 保存时可能带上的 module. 前缀。
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 只挑出属于当前模块的 LoRA 权重，例如 model.layers.0...lora.A.weight。
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """只保存 LoRA 参数，不保存基础模型参数。

    这样微调产物会很小，部署时可以选择“基础模型 + LoRA”加载，也可以后续 merge 成完整权重。
    """

    # torch.compile 之后模型可能包在 _orig_mod 里，这里取回原始模型。
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            # half() 减小文件体积；推理/合并时再按需要加载。
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """把 LoRA 增量合并进基础模型权重，并保存为完整模型权重。

    对线性层来说，LoRA 分支 B(Ax) 等价于 (B @ A) x，所以可以把 B@A 直接加到原始 weight 上。
    合并后推理不再需要 LoRA 分支，速度和普通模型一致。
    """

    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    # 先保存非 LoRA 参数。
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # nn.Linear 的权重形状是 [out_features, in_features]。
                # B.weight @ A.weight 形状也是 [out_features, in_features]，可直接加到原 weight。
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
