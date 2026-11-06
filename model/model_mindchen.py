# Modified by chenfangyi for the unified MindChen release (2026-11-06).
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 这个文件是 MindChen 的核心模型实现。
# 从新手视角看，一条文本进入模型后的主流程是：
# input_ids -> 词向量 Embedding -> 多层 MindChenBlock -> RMSNorm -> lm_head -> 每个位置的下一个 token 概率。
# MindChenBlock 内部又是典型 Decoder-only Transformer 结构：
# RMSNorm -> 自注意力 Attention -> 残差连接 -> RMSNorm -> FFN/MoE -> 残差连接。

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MindChen Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MindChenConfig(PretrainedConfig):
    """模型配置类。

    它只保存“模型应该长什么样”的超参数，不保存具体权重。
    继承 PretrainedConfig 后，模型就能比较自然地接入 HuggingFace Transformers 生态。
    """

    model_type = "mindchen"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # hidden_size 是每个 token 在模型内部的向量维度。
        # num_hidden_layers 是 Transformer Block 堆叠层数。
        # use_moe=False 时使用普通 FFN；True 时使用 MoE FFN。
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        # dropout 在训练时随机丢弃部分神经元输出，用于缓解过拟合；默认 0.0 表示不使用。
        self.dropout = kwargs.get("dropout", 0.0)
        # tokenizer 词表大小。模型最后会输出 vocab_size 维 logits，用来预测下一个 token。
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        # flash_attn=True 时，优先使用 PyTorch 的高效 attention 内核。
        self.flash_attn = kwargs.get("flash_attn", True)
        # num_attention_heads 是 Query 的头数。
        # num_key_value_heads 是 Key/Value 的头数；小于 Query 头数时就是 GQA，可节省 KV cache。
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        # 每个注意力头的维度，通常 hidden_size = num_attention_heads * head_dim。
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        # FFN 激活函数，默认 SiLU，也是很多现代 LLM 常用选择。
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        # FFN 中间层维度。这里用 hidden_size*pi 后向上取整到 64 的倍数，便于矩阵计算。
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        # RoPE 位置编码最多预计算到多少位置。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        # RoPE 的 base，值越大通常越偏向长上下文。
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        # tie_word_embeddings=True 表示输入词表 Embedding 和输出 lm_head 共享同一份权重。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        # 推理时可启用 YaRN 风格 RoPE scaling，用于把训练时较短上下文外推到更长上下文。
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        # MoE 专属配置：use_moe=False 时这些参数不会真正参与主干计算。
        # num_experts 表示每层有多少个专家 FFN。
        # num_experts_per_tok 表示每个 token 选择几个专家参与计算，常见是 top-1 或 top-2。
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        # 是否把 top-k 专家概率重新归一化。保留 True 兼容无配置的历史权重；
        # 新自适应 SFT 显式设 False：保留选中概率，避免 top-1 的 p/p 抵消梯度。
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        # 路由辅助损失系数：鼓励 token 不要全部挤到少数专家上。
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        # 老训练保持旧协议；新 SFT 显式打开有效 token 负载统计。
        self.moe_mask_padding = kwargs.get("moe_mask_padding", False)
        # 实验性替代梯度：前向不变，但反向不再是原归一化函数的真实导数。
        self.moe_router_gradient = kwargs.get("moe_router_gradient", "legacy")
        if self.moe_router_gradient not in ('legacy', 'straight_through'):
            raise ValueError('未知 moe_router_gradient')

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MindChen Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """RMSNorm 归一化层。

    LayerNorm 会减均值再除标准差；RMSNorm 只按均方根缩放，不减均值。
    现代 LLM 中 RMSNorm 很常见，计算更简单，也足够稳定。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        #dim维度
        # 每个 hidden 维度都有一个可学习缩放参数。
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        # rsqrt 表示 1 / sqrt(...)；keepdim=True 保持最后一维，方便广播。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转 float32 提高数值稳定性，再转回输入 dtype，兼容 fp16/bf16 混合精度训练。
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算 RoPE 需要的 cos/sin 表。

    RoPE 的作用是把位置信息“旋转”进 Q/K 向量里。
    训练/推理时每一层 attention 都会复用这张表，所以提前算好可以省时间。
    """

    # freqs 是不同维度上的旋转频率；低维变化慢，高维变化快。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN 对不同频段做平滑缩放，尽量让模型在更长上下文上也能保持可用。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    # t 是位置编号：0, 1, 2, ...
    t = torch.arange(end, device=freqs.device)
    # outer 后形状为 [end, dim/2]，表示每个位置、每个频率上的旋转角度。
    freqs = torch.outer(t, freqs).float()
    # cat 两次是为了和 rotate_half 的半维旋转写法匹配，最终形状 [end, dim]。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """把 RoPE 位置编码应用到 Query 和 Key。

    只旋转 Q/K，不旋转 V，因为 attention 分数来自 Q 和 K 的点积。
    """

    # rotate_half 等价于把向量前后两半组成二维平面并旋转 90 度。
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # q*cos + rotate(q)*sin 是旋转位置编码的标准形式。
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 Key/Value 头复制到和 Query 头数一致。

    GQA 中 Q 头数通常多于 K/V 头数。计算 attention 前需要把 K/V 按组复用。
    例如 Q=8 头、KV=4 头，则每个 KV 头复制 2 份。
    """

    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    """带 RoPE 和 KV Cache 的因果自注意力。

    因果自注意力的核心限制：当前位置只能看见自己和之前的 token，不能偷看未来 token。
    """

    def __init__(self, config: MindChenConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # n_rep 表示每个 KV 头要被多少个 Q 头复用。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        # Q/K/V/O 四个线性层是 attention 的主要可学习参数。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # QK norm 可以提升训练稳定性，尤其对小模型或较长上下文更友好。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # PyTorch 2.x 提供 scaled_dot_product_attention；可用时走更快路径。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        # 1. 从 hidden_states 投影得到 Q/K/V。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 2. 拆成多头形状：
        # Q: [batch, seq, q_heads, head_dim]
        # K/V: [batch, seq, kv_heads, head_dim]
        xq = xq.view (bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # 3. 对 Q/K 做 RMSNorm，再加入 RoPE 位置信息。
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # 4. 推理生成时，过去 token 的 K/V 已经算过，直接拼上即可，避免重复计算。
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # 5. 把张量变成 attention 常用形状 [batch, heads, seq, head_dim]。
        #    K/V 若是 GQA，会先 repeat 到和 Q 头数相同。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # 6. 优先走 Flash Attention；有 padding mask、单 token cache 等情况走手写分支更清晰。
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # attention 分数 = QK^T / sqrt(head_dim)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果 mask：把未来位置置为 -inf，softmax 后概率近似 0。
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # attention_mask 通常用于屏蔽 padding token。
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # 7. 多头结果拼回 hidden_size，再经过输出投影。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    """普通 FFN 前馈网络，也叫 MLP。

    这里使用 SwiGLU 风格结构：down_proj(silu(gate_proj(x)) * up_proj(x))。
    它比简单的 Linear -> 激活 -> Linear 表达能力更强，是现代 LLM 常见写法。
    """

    def __init__(self, config: MindChenConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # gate_proj 经过激活后像“门”一样控制 up_proj 的信息流，再投影回 hidden_size。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    """MoE 版 FFN：每个 token 只路由到少量专家。

    直觉：普通 FFN 是“全班同一个老师教”；MoE 是“根据题目类型找对应专家”。
    总参数量可以很大，但每个 token 激活的专家很少，所以单次计算量可控。
    """

    def __init__(self, config: MindChenConfig):
        super().__init__()
        self.config = config
        # gate 是路由器：输入 token hidden state，输出这个 token 应该分给每个专家的概率。
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 每个专家本质上都是一个独立 FeedForward。
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_dim = x.shape
        # 把 [batch, seq, hidden] 压成 [batch*seq, hidden]，逐 token 路由更方便。
        x_flat = x.view(-1, hidden_dim)
        # scores 表示每个 token 分配给各个专家的概率。
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 只取概率最高的 top-k 个专家，k = num_experts_per_tok。
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            if self.training and self.config.moe_router_gradient == 'straight_through':
                # 同一权重下前向与旧模型相同。只适合独立对照实验，不能宣称已提升能力。
                denominator = denominator.detach()
            topk_weight = topk_weight / denominator
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # mask 标出哪些 token 选择了第 i 个专家。
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                # index_add_ 把该专家输出累加回对应 token 位置。
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 如果某专家本 batch 没被选中，构造一个 0 梯度连接，避免分布式训练中参数未参与图的问题。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # 负载均衡辅助损失：鼓励不同专家都被使用，避免路由器只偏爱一两个专家。
            valid = (attention_mask[:, -seq_len:].reshape(-1).bool()
                     if self.config.moe_mask_padding and attention_mask is not None
                     else torch.ones(len(scores), dtype=torch.bool, device=scores.device))
            if valid.any():
                load = F.one_hot(topk_idx[valid], self.config.num_experts).float().mean(0)
                self.aux_loss = (load * scores[valid].mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
            else:
                self.aux_loss = scores.sum() * 0.
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MindChenBlock(nn.Module):
    """一个 Decoder-only Transformer Block。

    结构顺序：
    1. 输入 RMSNorm
    2. 因果自注意力
    3. 残差连接
    4. 后置 RMSNorm
    5. FFN 或 MoE-FFN
    6. 残差连接
    """

    def __init__(self, layer_id: int, config: MindChenConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Dense 模型用普通 FFN；MoE 模型把 FFN 换成专家混合。
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 第一条残差：attention 只负责增量修正，原 hidden_states 直接加回来。
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        # 第二条残差：FFN/MoE 负责逐 token 的非线性变换。
        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + (self.mlp(normed, attention_mask=attention_mask)
                                        if isinstance(self.mlp, MOEFeedForward) else self.mlp(normed))
        return hidden_states, present_key_value

class MindChenModel(nn.Module):
    """不带 lm_head 的主干模型。

    它输出每个 token 的最终 hidden state，但还没有转换成词表概率。
    """

    def __init__(self, config: MindChenConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # token id -> hidden_size 向量，这是文本进入神经网络的第一步。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 堆叠多个 Transformer Block。
        self.layers = nn.ModuleList([MindChenBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # RoPE cos/sin 注册为 buffer：跟随模型搬到 GPU，但不作为可训练参数保存梯度。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        # 兼容 transformers 新版本里某些 cache 类型；这里统一回到本项目使用的 list[tuple(K,V)]。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 生成时如果已经有 KV cache，新 token 的 RoPE 位置要从 past 长度开始。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # [batch, seq] token id 变成 [batch, seq, hidden] 向量。
        hidden_states = kwargs.pop('inputs_embeds', None)
        if hidden_states is None:
            hidden_states = self.dropout(self.embed_tokens(input_ids))
        # transformers>=5.x 的 meta-device 初始化有时会让 buffer 丢失，这里做一次兜底重建。
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # 当前这段序列要用到的位置编码切片。
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        # 逐层通过 Transformer Block。
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        # 最后一层输出再做一次归一化。
        hidden_states = self.norm(hidden_states)
        # MoE 模型把每层路由辅助损失加起来；Dense 模型这里就是 0。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MindChenForCausalLM(PreTrainedModel, GenerationMixin):
    """带语言模型头的完整 Causal LM。

    Causal LM 的训练目标：给定前面的 token，预测下一个 token。
    例如输入“我 喜欢”，模型在最后位置应该给“你/吃/学习/...”等 token 打分。
    """

    config_class = MindChenConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MindChenConfig = None):
        self.config = config or MindChenConfig()
        super().__init__(self.config)
        self.model = MindChenModel(self.config)
        # lm_head 把 hidden_size 向量映射到 vocab_size，得到每个词的 logits。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享：输入词向量和输出分类头共用一张表，省参数，也常能提升小模型表现。
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # 主干模型先输出 hidden_states，再由 lm_head 变成词表 logits。
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # 推理时有时只需要最后几个位置的 logits，减少显存和计算。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 自回归训练要错位预测：
            # 第 0 个位置的 logits 预测第 1 个 token，第 1 个位置预测第 2 个 token。
            # labels 中的 -100 会被 ignore_index 忽略，常用于 padding 或不希望训练的 prompt 部分。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """简化版自回归生成函数。

        每轮把当前序列喂给模型，取最后一个位置的 logits，采样/贪心选出下一个 token，
        再把新 token 拼回 input_ids，如此循环直到生成 eos 或达到 max_new_tokens。
        """

        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            # 有 KV cache 时，只把新增 token 喂给模型；没有 cache 时喂完整序列。
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # temperature 越高越随机，越低越保守。
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                # 对已经出现过的 token 做惩罚，减少机械重复。
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                # top-k：只保留概率最高的 k 个候选 token。
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                # top-p：按概率从高到低累计，只保留累计概率不超过 p 的候选集合。
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # do_sample=True 走随机采样；False 走 argmax 贪心解码。
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            # 已结束的样本继续补 eos，保证 batch 内长度对齐。
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
