# MindChen 模型架构图 + 代码逐句学习文档

本文专门讲 `model/model_mindchen.py`。目标是让刚入门的人能做到三件事：

1. 看懂 `images/architecture-language.svg` 里的 Dense 模型图。
2. 看懂 `images/architecture-moe.svg` 里的 MoE 模型图。
3. 能把图中的每个方块，对应到代码中的类、函数和关键语句。

建议阅读顺序：

```text
先看图 -> 再看本文 -> 最后打开 model/model_mindchen.py 对照代码
```

Dense 架构图：

![MindChen Dense](../images/architecture-language.svg)

MoE 架构图：

![MindChen MoE](../images/architecture-moe.svg)

---

## 0. 先用最白话的一句话理解模型

MindChen 是一个“根据前文猜下一个字/词”的模型。

假设输入是：

```text
hello
```

模型会输出一堆分数，表示下一个 token 更可能是什么：

```text
world: 最高
there: 次高
...
```

如果模型选中了 `world`，最后就得到：

```text
hello world
```

这正是架构图左侧的流程：

```text
hello
  -> Tokenizer
  -> Input Embedding
  -> Transformer Layer x K
  -> RMSNorm
  -> Linear
  -> Softmax
  -> Tokenizer Decoder / DeTokenizer
  -> world
```

---

## 1. 图和代码的总对应关系

| 图里的模块 | 代码位置 | 小白解释 |
|---|---|---|
| Tokenizer Encoder | `AutoTokenizer.from_pretrained("model")`，训练脚本里调用 | 把文字变成数字 token id。模型只能处理数字，不能直接处理字符串。 |
| Input Embedding | `MindChenModel.embed_tokens` | 把 token id 查表变成向量。 |
| Transformer Layer x K | `MindChenModel.layers` | 堆叠很多层 `MindChenBlock`，默认 8 层。 |
| Layer k | `MindChenBlock` | 一层 Transformer，包括 GQA + FFN/MoE-FFN。 |
| GQA | `Attention` | 让当前 token 去看前面的 token，提取上下文信息。 |
| FFN | `FeedForward` | 对每个 token 自己的向量做加工。 |
| MoE-FFN | `MOEFeedForward` | 多个专家 FFN，每个 token 只选少数专家。 |
| RMSNorm | `RMSNorm` | 让向量数值更稳定。 |
| RoPE | `precompute_freqs_cis` + `apply_rotary_pos_emb` | 给 Q/K 加入位置信息。 |
| mask | `Attention.forward` 里的 causal mask 和 attention_mask | 防止模型偷看未来 token，也屏蔽 padding。 |
| Linear | `lm_head` | 把 hidden 向量变成词表大小的分数。 |
| Softmax | 训练时在 `cross_entropy` 内部做；生成时 `torch.softmax(logits)` | 把分数变成概率。 |
| Tokenizer Decoder / DeTokenizer | `tokenizer.decode` / `batch_decode` | 把 token id 变回文字。 |

---

## 2. 先认识几个最重要的张量形状

看模型代码时，最容易迷路的是张量形状。先记住下面几个符号。

| 符号 | 含义 | 代码里常见名字 |
|---|---|---|
| `B` | batch size，一次喂几条文本 | `batch_size`, `bsz` |
| `S` | sequence length，每条文本多少 token | `seq_len`, `seq_length` |
| `H` | hidden size，每个 token 的向量维度 | `hidden_size` |
| `V` | vocab size，词表大小 | `vocab_size` |
| `QH` | Query 头数 | `num_attention_heads` |
| `KVH` | Key/Value 头数 | `num_key_value_heads` |
| `D` | 每个 attention head 的维度 | `head_dim` |

MindChen 默认配置大致是：

```text
H = 768
QH = 8
KVH = 4
D = H / QH = 96
V = 6400
层数 K = 8
```

一条数据在模型里流动时，形状大概这样变化：

```text
input_ids
[B, S]

Embedding 后
[B, S, H]

经过每层 Transformer
[B, S, H]

lm_head 后
[B, S, V]
```

---

## 3. 从文件开头开始逐句读代码

代码文件：`model/model_mindchen.py`

### 3.1 import 部分

```python
import math, torch, torch.nn.functional as F
```

解释：

- `math`：用来算平方根、对数、向上取整等普通数学函数。
- `torch`：PyTorch 主库。
- `torch.nn.functional as F`：PyTorch 里不带参数的函数式 API，比如 `softmax`、`cross_entropy`、`scaled_dot_product_attention`。

```python
from torch import nn
```

解释：

- `nn` 是 PyTorch 神经网络模块。
- `nn.Linear`、`nn.Embedding`、`nn.Dropout` 都从这里来。

```python
from transformers.activations import ACT2FN
```

解释：

- `ACT2FN` 是 HuggingFace 提供的激活函数字典。
- 例如 `ACT2FN["silu"]` 就能拿到 SiLU 激活函数。

```python
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
```

解释：

- `PretrainedConfig`：让你的配置类能接入 Transformers。
- `PreTrainedModel`：让你的模型能像 HuggingFace 模型一样保存、加载。
- `GenerationMixin`：提供生成相关接口兼容性。

```python
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
```

解释：

- 这是一个输出数据结构。
- 模型 forward 最后会把 `loss`、`logits`、`past_key_values`、`aux_loss` 打包到里面返回。

---

## 4. `MindChenConfig`：模型图纸

对应图里所有“模型有多大、有几层、有几个头”的配置。

```python
class MindChenConfig(PretrainedConfig):
```

解释：

- 定义一个配置类。
- 它继承 `PretrainedConfig`，所以能被 Transformers 生态识别。

```python
model_type = "mindchen"
```

解释：

- 给模型类型起名。
- 保存 config 时会记录它是 `mindchen`。

```python
def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
```

解释：

- `hidden_size=768`：每个 token 用 768 维向量表示。
- `num_hidden_layers=8`：图里的 `Transformer Layer k` 堆叠 8 层。
- `use_moe=False`：默认 Dense 模型；如果设为 True，就走 MoE 图。
- `**kwargs`：允许从外面传更多配置。

```python
super().__init__(**kwargs)
```

解释：

- 调用父类 `PretrainedConfig` 的初始化逻辑。
- 这是接入 HuggingFace 的常规写法。

```python
self.hidden_size = hidden_size
self.num_hidden_layers = num_hidden_layers
self.use_moe = use_moe
```

解释：

- 把传进来的模型大小参数保存到配置对象里。
- 后面创建模型层时都会读这些值。

```python
self.dropout = kwargs.get("dropout", 0.0)
```

解释：

- `dropout` 是训练时随机丢掉一部分信息，减少过拟合。
- 默认是 `0.0`，也就是不丢。

```python
self.vocab_size = kwargs.get("vocab_size", 6400)
```

解释：

- 词表大小是 6400。
- 最后的 `lm_head` 会输出 6400 个分数，每个分数对应一个 token。

```python
self.bos_token_id = kwargs.get("bos_token_id", 1)
self.eos_token_id = kwargs.get("eos_token_id", 2)
```

解释：

- `bos` 是 begin of sequence，表示一句话开始。
- `eos` 是 end of sequence，表示一句话结束。

```python
self.flash_attn = kwargs.get("flash_attn", True)
```

解释：

- 是否优先用 PyTorch 的高效 attention 实现。
- 能用时更快。

```python
self.num_attention_heads = kwargs.get("num_attention_heads", 8)
self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
```

解释：

- Query 有 8 个头。
- Key/Value 有 4 个头。
- 这就是图里的 `GQA`，Grouped Query Attention。

最白话理解：

```text
Q 头多，负责多角度提问；
K/V 头少，被多个 Q 头共享，节省显存和计算。
```

```python
self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
```

解释：

- 每个头的维度。
- 默认 `768 // 8 = 96`。

```python
self.hidden_act = kwargs.get("hidden_act", 'silu')
```

解释：

- FFN 使用 SiLU 激活函数。
- 对应图中 FFN 里的黄色 `SiLU`。

```python
self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
```

解释：

- FFN 中间层维度。
- FFN 会先把向量从 `hidden_size` 放大到 `intermediate_size`，再压回 `hidden_size`。
- 这里用 `hidden_size * π`，再向上凑到 64 的倍数，方便 GPU 矩阵计算。

```python
self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
```

解释：

- RoPE 位置编码最多预先算到 32768 个位置。
- 也就是支持较长上下文。

```python
self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
```

解释：

- RMSNorm 里防止除以 0 的小数。

```python
self.rope_theta = kwargs.get("rope_theta", 1e6)
```

解释：

- RoPE 的频率基数。
- 越大通常越偏向长上下文能力。

```python
self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
```

解释：

- 输入 Embedding 和输出 `lm_head` 共用一份权重。
- 好处是省参数，小模型常用。

```python
self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
```

解释：

- 推理时是否启用 RoPE scaling。
- 用于长文本外推。

```python
self.rope_scaling = {...} if self.inference_rope_scaling else None
```

解释：

- 如果启用，就用 YaRN 风格参数调整 RoPE。
- 如果不启用，就是 `None`，正常 RoPE。

MoE 专属配置：

```python
self.num_experts = kwargs.get("num_experts", 4)
self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
```

解释：

- `num_experts=4`：一层 MoE 里有 4 个专家。
- `num_experts_per_tok=1`：每个 token 只选 1 个专家。
- `moe_intermediate_size`：每个专家 FFN 的中间层大小。
- `norm_topk_prob=True`：选中专家的权重重新归一化。
- `router_aux_loss_coef`：路由均衡损失系数，避免所有 token 都挤到一个专家。

对应 MoE 图：

```text
Router -> Top M -> Routed Experts -> 加权求和
```

---

## 5. `RMSNorm`：图里的灰色 RMSNorm

代码：

```python
class RMSNorm(torch.nn.Module):
```

解释：

- 定义 RMSNorm 层。
- 图里多处灰色 `RMSNorm` 都对应这个类。

```python
def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
```

逐句解释：

- `dim`：最后一维大小，比如 `hidden_size=768` 或 `head_dim=96`。
- `eps`：防止除 0。
- `self.weight`：可学习缩放参数，初始全是 1。

```python
def norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

解释：

1. `x.pow(2)`：每个数平方。
2. `.mean(-1, keepdim=True)`：在最后一维求平均。
3. `+ self.eps`：防止除 0。
4. `torch.rsqrt(...)`：计算 `1 / sqrt(...)`。
5. `x * ...`：把原向量缩放到更稳定的范围。

最白话：

```text
如果向量数值太大，就压小；
如果数值太小，就适当放大；
让后面的计算稳定。
```

```python
def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
```

解释：

- 先把 `x` 转成 float32 做归一化，更稳定。
- 乘上可学习的 `weight`。
- 最后转回输入原来的类型，比如 fp16/bf16。

---

## 6. RoPE：图里 Q/K 旁边的 RoPE

RoPE 的作用：让模型知道 token 的位置。

如果没有位置信息，模型只知道一堆 token，不知道哪个在前哪个在后。

### 6.1 `precompute_freqs_cis`

```python
def precompute_freqs_cis(dim, end, rope_base, rope_scaling):
```

解释：

- `dim`：每个 attention head 的维度。
- `end`：最多预计算多少个位置。
- `rope_base`：RoPE 频率基数。
- `rope_scaling`：长上下文外推配置。

```python
freqs = 1.0 / (rope_base ** (...))
```

解释：

- 生成每个维度对应的旋转频率。
- 不同维度旋转速度不同。

```python
if rope_scaling is not None:
    ...
```

解释：

- 如果启用 YaRN，就调整频率，让模型更适合长上下文。

```python
t = torch.arange(end, device=freqs.device)
freqs = torch.outer(t, freqs).float()
```

解释：

- `t` 是位置编号：0、1、2、3……
- `outer` 得到“每个位置 × 每个频率”的旋转角度表。

```python
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
```

解释：

- 算出每个位置需要的 `cos` 和 `sin`。
- 后面每一层 Attention 都会用这两张表。

### 6.2 `apply_rotary_pos_emb`

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
```

解释：

- 输入 Q、K、cos、sin。
- 输出带位置信息的新 Q、K。
- 注意：RoPE 只加在 Q/K 上，不加在 V 上。

```python
def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
```

解释：

- 把向量后一半取负放前面，前一半放后面。
- 这是 RoPE 旋转公式的一部分。

```python
q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
```

解释：

- 把位置旋转注入 Q/K。
- 这样 Attention 计算 `Q @ K` 时，就能感知位置。

---

## 7. `repeat_kv`：GQA 为什么能省显存

代码：

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
```

解释：

- GQA 中，Q 头数多，K/V 头数少。
- 真正算 attention 时，需要让 K/V 的头数和 Q 对齐。
- `repeat_kv` 就是把 K/V 复制给多个 Q 头共享。

```python
bs, slen, num_key_value_heads, head_dim = x.shape
```

解释：

- `x` 的形状是 `[B, S, KVH, D]`。

```python
if n_rep == 1: return x
```

解释：

- 如果 Q 头数和 KV 头数一样，就不用复制。

```python
return (
    x[:, :, :, None, :]
    .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
    .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
)
```

解释：

- 在 KV 头后面插入一个复制维度。
- 把每个 KV 头复制 `n_rep` 次。
- 最后 reshape 成 `[B, S, QH, D]`。

举例：

```text
QH = 8
KVH = 4
n_rep = 8 / 4 = 2

每个 KV 头复制 2 份：
4 个 KV 头 -> 8 个头
```

---

## 8. `Attention`：图里的 GQA

对应 Dense 图中右侧绿色 `(a) GQA`。

### 8.1 初始化函数

```python
class Attention(nn.Module):
```

解释：

- 定义注意力模块。
- 图里 `GQA` 方块对应这里。

```python
self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
```

解释：

- 如果没有单独指定 KV 头数，就让 KV 头数等于 Q 头数。
- 当前默认指定了 4。

```python
self.n_local_heads = config.num_attention_heads
self.n_local_kv_heads = self.num_key_value_heads
self.n_rep = self.n_local_heads // self.n_local_kv_heads
self.head_dim = config.head_dim
```

解释：

- `n_local_heads` 是 Q 头数。
- `n_local_kv_heads` 是 K/V 头数。
- `n_rep` 是每个 KV 头要复制几份。
- `head_dim` 是每个头的维度。

```python
self.is_causal = True
```

解释：

- 表示因果注意力。
- 当前 token 只能看自己和之前的 token，不能看未来 token。

### 8.2 Q/K/V/O 四个 Linear

```python
self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
```

解释：

- `q_proj`：把输入变成 Query。
- `k_proj`：把输入变成 Key。
- `v_proj`：把输入变成 Value。
- `o_proj`：把多头 attention 输出再合回 `hidden_size`。

对应图：

```text
RMSNorm -> Linear -> Q / K' / V'
...
attention 输出 -> Linear
```

为什么 Q/K/V 都来自同一个输入？

```text
因为这是 self-attention。
当前句子自己产生 Q、K、V，然后自己和自己做匹配。
```

### 8.3 QK Norm

```python
self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

解释：

- 对每个 Q head 和 K head 做 RMSNorm。
- 有助于训练稳定。
- 图里 Q/K 上面的 `N` 就对应这个归一化。

### 8.4 Dropout 和 Flash Attention

```python
self.attn_dropout = nn.Dropout(config.dropout)
self.resid_dropout = nn.Dropout(config.dropout)
self.dropout = config.dropout
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
```

解释：

- `attn_dropout`：attention 权重上的 dropout。
- `resid_dropout`：输出投影后的 dropout。
- `flash`：如果 PyTorch 支持，就使用高效 attention。

---

## 9. `Attention.forward`：GQA 前向传播逐句看

输入：

```python
def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
```

解释：

- `x`：输入 hidden states，形状 `[B, S, H]`。
- `position_embeddings`：RoPE 的 cos/sin。
- `past_key_value`：推理生成时缓存的历史 K/V。
- `use_cache`：是否返回新的 K/V cache。
- `attention_mask`：padding mask。

### 9.1 取形状

```python
bsz, seq_len, _ = x.shape
```

解释：

- `bsz` 是 batch size。
- `seq_len` 是当前输入长度。

### 9.2 生成 Q/K/V

```python
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
```

解释：

- 输入 `x` 经过三个线性层，分别得到 Q、K、V。

形状：

```text
x:  [B, S, H]
xq: [B, S, QH * D]
xk: [B, S, KVH * D]
xv: [B, S, KVH * D]
```

### 9.3 拆成多头

```python
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
```

解释：

- 把最后一维拆成“头数 × 每头维度”。

形状：

```text
xq: [B, S, QH, D]
xk: [B, S, KVH, D]
xv: [B, S, KVH, D]
```

### 9.4 Q/K 归一化 + RoPE

```python
xq, xk = self.q_norm(xq), self.k_norm(xk)
```

解释：

- 对 Q/K 做 RMSNorm。

```python
cos, sin = position_embeddings
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

解释：

- 取出当前位置的 cos/sin。
- 给 Q/K 加 RoPE 位置信息。

### 9.5 KV Cache

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
```

解释：

- 如果是生成阶段，以前 token 的 K/V 已经算过。
- 直接把旧 K/V 和新 K/V 拼起来。

图里没有画 KV cache，但它是推理加速关键。

```python
past_kv = (xk, xv) if use_cache else None
```

解释：

- 如果启用 cache，就返回当前完整 K/V，供下一步生成使用。

### 9.6 转成 attention 需要的形状

```python
xq, xk, xv = (
    xq.transpose(1, 2),
    repeat_kv(xk, self.n_rep).transpose(1, 2),
    repeat_kv(xv, self.n_rep).transpose(1, 2)
)
```

解释：

- `transpose(1, 2)` 把 `[B, S, heads, D]` 变成 `[B, heads, S, D]`。
- K/V 先 `repeat_kv`，让头数从 `KVH` 变成 `QH`。

最终形状：

```text
xq: [B, QH, S, D]
xk: [B, QH, S, D]
xv: [B, QH, S, D]
```

### 9.7 Flash Attention 分支

```python
if self.flash and ...:
    output = F.scaled_dot_product_attention(...)
```

解释：

- 如果条件允许，就用 PyTorch 内置高效 attention。
- 它内部完成 QK 打分、mask、softmax、乘 V。

### 9.8 手写 Attention 分支

```python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

解释：

- `xq @ xk.transpose(-2, -1)`：Q 和 K 做点积。
- 点积越大，说明当前位置越应该关注那个 token。
- 除以 `sqrt(head_dim)` 是缩放，防止分数过大。

形状：

```text
scores: [B, QH, S, S]
```

```python
if self.is_causal:
    scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
```

解释：

- 构造上三角 mask。
- 未来位置加 `-inf`。
- softmax 后未来位置概率变成 0。

这对应图中 GQA 里的 `mask`。

```python
if attention_mask is not None:
    scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
```

解释：

- 如果某些 token 是 padding，就给它们加一个极大负数。
- softmax 后 padding 也不会被关注。

```python
output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
```

解释：

1. `softmax(scores)`：把注意力分数变成概率。
2. `attn_dropout(...)`：训练时可随机丢一点注意力。
3. `@ xv`：用注意力概率对 V 做加权求和。

最白话：

```text
模型先判断“我该看前面哪些词”，再把那些词的信息加权拿过来。
```

### 9.9 拼回 hidden_size

```python
output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
```

解释：

- 从 `[B, QH, S, D]` 变回 `[B, S, QH * D]`。
- 因为 `QH * D = hidden_size`，所以就是 `[B, S, H]`。

```python
output = self.resid_dropout(self.o_proj(output))
```

解释：

- 经过输出线性层 `o_proj`。
- 再做 dropout。

```python
return output, past_kv
```

解释：

- 返回 attention 输出。
- 如果启用 cache，也返回 K/V cache。

---

## 10. `FeedForward`：图里的 FFN

对应 Dense 图右侧灰色 `(b) FFN`。

### 10.1 初始化

```python
class FeedForward(nn.Module):
```

解释：

- 定义普通 Dense FFN。
- Dense 模型每层都用它。

```python
intermediate_size = intermediate_size or config.intermediate_size
```

解释：

- 如果外面传了中间层大小，就用外面的。
- 否则用配置里的默认值。

```python
self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
```

解释：

- `up_proj`：把维度从 `H` 放大到 `I`。
- `gate_proj`：也从 `H` 放大到 `I`，但它会经过 SiLU，当作门。
- `down_proj`：把维度从 `I` 压回 `H`。

图中对应：

```text
RMSNorm
  -> Linear --------------.
  -> Linear -> SiLU -----> 乘法
  -> Linear
  -> Dropout
  -> 残差相加
```

代码中没有单独的 FFN dropout，图是结构示意；当前实现主要在 Attention 输出处使用 dropout。

```python
self.act_fn = ACT2FN[config.hidden_act]
```

解释：

- 根据配置取激活函数。
- 默认取 `silu`。

### 10.2 前向传播

```python
return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

逐步拆开就是：

```python
gate = self.gate_proj(x)
gate = self.act_fn(gate)
up = self.up_proj(x)
middle = gate * up
out = self.down_proj(middle)
```

解释：

- `gate_proj(x)`：生成一扇门。
- `SiLU(gate)`：让门有非线性判断能力。
- `up_proj(x)`：生成要通过的信息。
- `gate * up`：门控制信息通过多少。
- `down_proj(...)`：压回原维度。

为什么这样设计？

```text
普通 FFN 是“直接加工信息”；
SwiGLU 风格 FFN 是“先决定哪些信息重要，再加工”。
```

---

## 11. `MOEFeedForward`：MoE 图里的 Router + Experts

对应 MoE 图右侧大框。

MoE 和 Dense 的区别只在 FFN 部分：

```text
Dense: 每个 token 都经过同一个 FFN
MoE:   每个 token 先经过 Router，选择少数专家 FFN
```

### 11.1 初始化

```python
class MOEFeedForward(nn.Module):
```

解释：

- 定义 MoE 版 FFN。

```python
self.config = config
```

解释：

- 保存配置，后面要用 `num_experts`、`num_experts_per_tok`。

```python
self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
```

解释：

- 这是图里的 `Router`。
- 输入一个 token 的 hidden 向量。
- 输出这个 token 分给每个专家的分数。

形状：

```text
输入: [B*S, H]
输出: [B*S, num_experts]
```

```python
self.experts = nn.ModuleList([
    FeedForward(config, intermediate_size=config.moe_intermediate_size)
    for _ in range(config.num_experts)
])
```

解释：

- 创建 `num_experts` 个专家。
- 每个专家都是一个独立的 `FeedForward`。

如果 `num_experts=4`，就像图中：

```text
E-1, E-2, E-3, E-4
```

### 11.2 前向传播

```python
batch_size, seq_len, hidden_dim = x.shape
```

解释：

- 输入 `x` 是 `[B, S, H]`。

```python
x_flat = x.view(-1, hidden_dim)
```

解释：

- 把 batch 和 seq 合并。
- 从 `[B, S, H]` 变成 `[B*S, H]`。
- 这样每个 token 都能独立路由。

```python
scores = F.softmax(self.gate(x_flat), dim=-1)
```

解释：

- Router 给每个 token 对每个专家打分。
- `softmax` 把分数变成概率。

形状：

```text
scores: [B*S, num_experts]
```

```python
topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
```

解释：

- 选出概率最高的 top-k 个专家。
- `topk_idx` 是专家编号。
- `topk_weight` 是专家权重。

对应 MoE 图：

```text
Router -> Top M -> E-2 / E-N 等专家被激活
```

```python
if self.config.norm_topk_prob:
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
```

解释：

- 把选中专家的权重重新归一化。
- 让它们加起来等于 1。

```python
y = torch.zeros_like(x_flat)
```

解释：

- 创建输出容器。
- 形状和 `x_flat` 一样：`[B*S, H]`。

### 11.3 逐个专家处理 token

```python
for i, expert in enumerate(self.experts):
```

解释：

- 遍历每个专家。
- `i` 是专家编号。
- `expert` 是对应的 FFN。

```python
mask = (topk_idx == i)
```

解释：

- 找出哪些 token 选择了第 `i` 个专家。

```python
if mask.any():
```

解释：

- 如果真的有 token 选中了这个专家，就进入处理。

```python
token_idx = mask.any(dim=-1).nonzero().flatten()
```

解释：

- 找到选中该专家的 token 下标。

```python
weight = topk_weight[mask].view(-1, 1)
```

解释：

- 取出这些 token 对当前专家的权重。

```python
y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
```

解释：

1. `x_flat[token_idx]`：取出应该给这个专家处理的 token。
2. `expert(...)`：专家 FFN 处理这些 token。
3. `* weight`：乘上路由权重。
4. `index_add_`：把专家输出加回对应 token 的位置。

如果一个 token 选多个专家，多个专家结果会加在一起。

```python
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

解释：

- 如果某个专家这次没被任何 token 选中，正常来说它不参与计算图。
- 分布式训练时可能因此出问题。
- 这里加一个“值为 0 的连接”，不改变输出，但让参数挂到计算图里。

### 11.4 MoE 辅助损失

```python
if self.training and self.config.router_aux_loss_coef > 0:
```

解释：

- 只在训练时计算辅助损失。

```python
load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
```

解释：

- 统计每个专家被选中的比例。

```python
self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
```

解释：

- 计算负载均衡损失。
- 目的：让 token 不要全部挤到一个专家。

```python
return y.view(batch_size, seq_len, hidden_dim)
```

解释：

- 把 `[B*S, H]` 还原成 `[B, S, H]`。

---

## 12. `MindChenBlock`：图中被放大的 Layer k

对应图里的紫色 `Layer k`。

Dense 图：

```text
输入
 -> GQA
 -> 残差相加
 -> FFN
 -> 残差相加
 -> 输出
```

MoE 图：

```text
输入
 -> GQA
 -> 残差相加
 -> MoE-FFN
 -> 残差相加
 -> 输出
```

### 12.1 初始化

```python
class MindChenBlock(nn.Module):
```

解释：

- 定义一层 Transformer。

```python
self.self_attn = Attention(config)
```

解释：

- 创建 GQA 注意力层。

```python
self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

解释：

- attention 前一个 RMSNorm。
- FFN/MoE 前一个 RMSNorm。
- 这是 Pre-Norm Transformer 写法。

```python
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

解释：

- 如果 `use_moe=False`，用普通 FFN。
- 如果 `use_moe=True`，用 MoE-FFN。

这就是 Dense 图和 MoE 图的核心差异。

### 12.2 前向传播

```python
residual = hidden_states
```

解释：

- 保存原始输入，后面做残差连接。

```python
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),
    position_embeddings,
    past_key_value,
    use_cache,
    attention_mask
)
```

解释：

1. 先做 `input_layernorm`。
2. 再进入 GQA attention。
3. 输出新的 hidden states 和 KV cache。

对应图：

```text
RMSNorm -> GQA
```

```python
hidden_states += residual
```

解释：

- 残差连接。
- 把 attention 的输出和原输入加起来。

最白话：

```text
不要让 attention 完全重写原信息；
只让它在原信息基础上补充上下文。
```

```python
hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
```

解释：

1. 对当前 hidden states 做 RMSNorm。
2. 送入 FFN 或 MoE-FFN。
3. 再和原 hidden states 相加。

对应图：

```text
RMSNorm -> FFN/MoE-FFN -> 残差相加
```

```python
return hidden_states, present_key_value
```

解释：

- 返回这一层处理后的 hidden states。
- 返回这一层的 KV cache。

---

## 13. `MindChenModel`：图左侧 Transformer Layer x K

`MindChenModel` 是不带最终 `lm_head` 的主干。

### 13.1 初始化

```python
class MindChenModel(nn.Module):
```

解释：

- 定义模型主干。

```python
self.config = config
self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
```

解释：

- 保存配置。
- 保存词表大小和层数。

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
```

解释：

- 图里的 `Input Embedding`。
- 把 token id 变成向量。

形状变化：

```text
input_ids: [B, S]
embedding: [B, S, H]
```

```python
self.dropout = nn.Dropout(config.dropout)
```

解释：

- embedding 后可做 dropout。

```python
self.layers = nn.ModuleList([MindChenBlock(l, config) for l in range(self.num_hidden_layers)])
```

解释：

- 创建很多层 `MindChenBlock`。
- 默认 8 层。
- 对应图里的 `Transformer Layer k × K`。

```python
self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

解释：

- 所有 Transformer 层之后，再做一次最终 RMSNorm。
- 对应图左侧顶部的 `RMSNorm`。

```python
freqs_cos, freqs_sin = precompute_freqs_cis(...)
```

解释：

- 预计算 RoPE 需要的 cos/sin。

```python
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

解释：

- 把 cos/sin 放进模型对象。
- 它们会跟着模型移动到 GPU。
- 但不是可训练参数。

### 13.2 前向传播

```python
batch_size, seq_length = input_ids.shape
```

解释：

- 输入 token id 的形状是 `[B, S]`。

```python
if hasattr(past_key_values, 'layers'): past_key_values = None
```

解释：

- 兼容 Transformers 新版本的 cache 类型。
- 如果不是本项目预期的 cache 格式，就先置空。

```python
past_key_values = past_key_values or [None] * len(self.layers)
```

解释：

- 如果没有传 cache，就给每层都准备一个 `None`。

```python
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
```

解释：

- 生成时，如果前面已经有 cache，新 token 的位置不是 0，而是从历史长度开始。

```python
hidden_states = self.dropout(self.embed_tokens(input_ids))
```

解释：

1. `embed_tokens(input_ids)`：token id 变向量。
2. `dropout(...)`：训练时可随机丢弃一部分。

对应图：

```text
Tokenizer -> Input Embedding
```

```python
position_embeddings = (
    self.freqs_cos[start_pos:start_pos + seq_length],
    self.freqs_sin[start_pos:start_pos + seq_length]
)
```

解释：

- 取出当前这段 token 对应位置的 RoPE cos/sin。

```python
presents = []
```

解释：

- 用来收集每层返回的 KV cache。

```python
for layer, past_key_value in zip(self.layers, past_key_values):
```

解释：

- 逐层通过 Transformer Block。
- 每层配一个自己的 cache。

```python
hidden_states, present = layer(...)
```

解释：

- 当前 hidden states 进入一层 `MindChenBlock`。
- 得到更新后的 hidden states。

```python
presents.append(present)
```

解释：

- 保存这一层的 KV cache。

```python
hidden_states = self.norm(hidden_states)
```

解释：

- 所有层结束后做最终 RMSNorm。

```python
aux_loss = sum([...], hidden_states.new_zeros(1).squeeze())
```

解释：

- 如果是 MoE 模型，把每层 MoE 的路由辅助损失加起来。
- 如果是 Dense 模型，结果就是 0。

```python
return hidden_states, presents, aux_loss
```

解释：

- 返回最终 hidden states。
- 返回每层 KV cache。
- 返回 MoE 辅助损失。

---

## 14. `MindChenForCausalLM`：完整语言模型

图左侧顶部：

```text
RMSNorm -> Linear -> Softmax -> Decoder
```

其中 `Linear` 就是这里的 `lm_head`。

### 14.1 初始化

```python
class MindChenForCausalLM(PreTrainedModel, GenerationMixin):
```

解释：

- 定义完整的因果语言模型。
- 它包含主干 `MindChenModel` 和输出头 `lm_head`。

```python
config_class = MindChenConfig
```

解释：

- 告诉 HuggingFace：这个模型对应的配置类是 `MindChenConfig`。

```python
_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

解释：

- 声明 `lm_head` 和 embedding 可以共享权重。

```python
self.config = config or MindChenConfig()
super().__init__(self.config)
```

解释：

- 如果没有传配置，就用默认配置。
- 调用父类初始化。

```python
self.model = MindChenModel(self.config)
```

解释：

- 创建模型主干。

```python
self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
```

解释：

- 输出层。
- 把 `[B, S, H]` 变成 `[B, S, V]`。
- 每个位置输出词表中每个 token 的分数。

```python
if self.config.tie_word_embeddings:
    self.model.embed_tokens.weight = self.lm_head.weight
```

解释：

- 如果启用权重共享，就让输入 embedding 和输出 lm_head 使用同一张权重表。

```python
self.post_init()
```

解释：

- HuggingFace 模型的初始化收尾逻辑。

### 14.2 forward：训练时怎么计算 loss

```python
def forward(..., labels=None, ...):
```

解释：

- `labels` 不为空时表示训练，需要算 loss。
- `labels` 为空时通常是推理，只要 logits。

```python
hidden_states, past_key_values, aux_loss = self.model(...)
```

解释：

- 先调用主干模型。
- 得到每个 token 的最终 hidden 向量。

```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
```

解释：

- 推理时可以只保留最后几个位置的 logits。
- 省显存。

```python
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

解释：

- 把 hidden states 转成词表分数。

形状：

```text
hidden_states: [B, S, H]
logits:        [B, S, V]
```

```python
loss = None
```

解释：

- 默认不计算 loss。

```python
if labels is not None:
```

解释：

- 如果传了 labels，就进入训练损失计算。

```python
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
```

解释：

- 自回归语言模型要“错位预测”。
- 第 0 个位置预测第 1 个 token。
- 第 1 个位置预测第 2 个 token。

举例：

```text
input:  我  爱  你

位置 0 的输出 -> 预测 “爱”
位置 1 的输出 -> 预测 “你”
```

```python
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

解释：

- `cross_entropy` 内部会做 softmax。
- `ignore_index=-100` 表示 labels 里等于 -100 的位置不算 loss。
- SFT 训练里，用户 prompt 部分通常就是 -100。

```python
return MoeCausalLMOutputWithPast(...)
```

解释：

- 把结果打包返回。
- 包含 `loss`、`aux_loss`、`logits`、`past_key_values`、`hidden_states`。

---

## 15. `generate`：图里从 hello 生成 world 的过程

生成不是一次性生成完，而是循环生成。

### 15.1 函数入口

```python
@torch.inference_mode()
def generate(...):
```

解释：

- `inference_mode` 表示推理模式，不计算梯度。
- 省显存、速度更快。

```python
input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
```

解释：

- 取出输入 token id。
- 如果要生成多个候选，就复制多份。

```python
attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
```

解释：

- attention mask 也同步复制。

```python
past_key_values = kwargs.pop("past_key_values", None)
```

解释：

- 取已有 KV cache。
- 第一步通常没有 cache。

```python
finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
```

解释：

- 记录 batch 里哪些样本已经生成结束。

### 15.2 循环生成

```python
for _ in range(max_new_tokens):
```

解释：

- 最多生成 `max_new_tokens` 个新 token。

```python
past_len = past_key_values[0][0].shape[1] if past_key_values else 0
```

解释：

- 如果有 cache，说明前面 token 已经算过。
- 只需要喂新 token。

```python
outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
```

解释：

- 调用模型 forward。
- 得到最新 logits 和新的 KV cache。

```python
logits = outputs.logits[:, -1, :] / temperature
```

解释：

- 只取最后一个位置的 logits。
- 因为生成下一个 token 只需要最后位置的预测。
- 除以 `temperature` 控制随机性。

```python
if repetition_penalty != 1.0:
    ...
```

解释：

- 惩罚已经出现过的 token。
- 减少重复输出。

```python
if top_k > 0:
    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
```

解释：

- 只保留分数最高的 top-k 个 token。
- 其他 token 设成 `-inf`，不会被采样。

```python
if top_p < 1.0:
    ...
```

解释：

- nucleus sampling。
- 从高概率 token 开始累加，只保留累计概率不超过 `top_p` 的候选。

```python
next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
```

解释：

- 如果 `do_sample=True`，按概率随机采样。
- 如果 `do_sample=False`，直接选分数最高的 token。

```python
input_ids = torch.cat([input_ids, next_token], dim=-1)
```

解释：

- 把新 token 拼到序列末尾。

```python
past_key_values = outputs.past_key_values if use_cache else None
```

解释：

- 保存 KV cache。
- 下一轮生成可以复用。

```python
if finished.all(): break
```

解释：

- 如果 batch 里所有样本都生成了 EOS，就提前结束。

```python
return input_ids
```

解释：

- 返回完整 token 序列。
- 之后 tokenizer 会把它 decode 成文字。

---

## 16. Dense 模型完整路线：从图到代码串起来

Dense 图的路线：

```text
文本 hello
  -> Tokenizer
  -> input_ids
  -> Embedding
  -> Transformer Layer 1
       -> RMSNorm
       -> GQA
       -> 残差
       -> RMSNorm
       -> FFN
       -> 残差
  -> Transformer Layer 2
  -> ...
  -> Transformer Layer K
  -> RMSNorm
  -> Linear lm_head
  -> logits
  -> Softmax
  -> next token
  -> Decoder
  -> world
```

代码对应：

```text
MindChenForCausalLM.forward
  -> MindChenModel.forward
       -> embed_tokens
       -> for layer in self.layers
            -> MindChenBlock.forward
                 -> input_layernorm
                 -> Attention.forward
                 -> residual add
                 -> post_attention_layernorm
                 -> FeedForward.forward
                 -> residual add
       -> final norm
  -> lm_head
  -> cross_entropy 或 generate sampling
```

---

## 17. MoE 模型完整路线：和 Dense 只差 FFN

MoE 图的主线和 Dense 一样，区别在每个 `MindChenBlock` 的 MLP 部分。

代码关键句：

```python
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

如果：

```python
use_moe=False
```

则：

```text
GQA -> FFN
```

如果：

```python
use_moe=True
```

则：

```text
GQA -> MoE-FFN
```

MoE-FFN 内部：

```text
Input Feature
  -> Router
  -> scores
  -> Top M Experts
  -> E-2 / E-N 等专家处理
  -> 乘各自 Weight
  -> 加权求和
  -> Output Feature
```

代码对应：

```text
x_flat = x.view(-1, hidden_dim)
scores = softmax(gate(x_flat))
topk_weight, topk_idx = topk(scores)
for each expert:
    找到分配给这个专家的 token
    expert(token)
    乘权重
    加回输出 y
return y.view(batch_size, seq_len, hidden_dim)
```

---

## 18. 训练时 loss 到底在哪里算

图里画了 `Softmax`，但代码里训练时没有显式写：

```python
softmax(...)
```

原因是：

```python
F.cross_entropy(...)
```

内部已经包含：

```text
log_softmax + negative log likelihood
```

也就是：

```text
logits -> softmax 概率 -> 看正确 token 的概率 -> 概率越低 loss 越大
```

所以训练时：

```python
loss = F.cross_entropy(...)
```

推理生成时才显式写：

```python
torch.softmax(logits, dim=-1)
```

---

## 19. 一次 forward 的最小例子

假设：

```python
input_ids.shape = [2, 10]
```

含义：

- batch 里有 2 条文本。
- 每条文本 10 个 token。

经过 embedding：

```python
hidden_states.shape = [2, 10, 768]
```

进入 GQA：

```python
xq.shape = [2, 10, 8, 96]
xk.shape = [2, 10, 4, 96]
xv.shape = [2, 10, 4, 96]
```

K/V 复制：

```python
xk.shape = [2, 10, 8, 96]
xv.shape = [2, 10, 8, 96]
```

attention 输出：

```python
output.shape = [2, 10, 768]
```

经过所有层和 lm_head：

```python
logits.shape = [2, 10, 6400]
```

这表示：

```text
2 条文本
每条 10 个位置
每个位置都给 6400 个 token 打分
```

---

## 20. 小白常见问题

### 20.1 为什么模型输入是数字？

因为神经网络只能做矩阵计算。文字要先经过 tokenizer 变成 token id。

### 20.2 Embedding 是什么？

Embedding 是一张表。token id 是下标，查表得到一个向量。

### 20.3 Attention 是什么？

Attention 是“当前 token 应该看前面哪些 token”。

### 20.4 为什么不能看未来？

因为训练目标是预测下一个 token。如果允许看未来，模型就作弊了。

### 20.5 FFN 是什么？

Attention 负责混合上下文信息，FFN 负责对每个 token 的向量做进一步加工。

### 20.6 MoE 为什么参数多但计算不一定多？

因为虽然有很多专家，但每个 token 只选少数专家计算。

### 20.7 `aux_loss` 是什么？

MoE 路由辅助损失，防止所有 token 都只找同一个专家。

### 20.8 `past_key_values` 是什么？

生成时缓存过去 token 的 K/V，避免每生成一个 token 都重新算完整历史。

### 20.9 `labels=-100` 是什么？

训练时忽略这些位置，不计算 loss。SFT 中常用它忽略用户输入部分。

### 20.10 `lm_head` 是什么？

最后一层线性层，把 hidden 向量变成词表分数。

---

## 21. 读代码时的推荐路线

不要从第一行一路硬读。建议按执行顺序读：

```text
1. MindChenForCausalLM.forward
2. MindChenModel.forward
3. MindChenBlock.forward
4. Attention.forward
5. FeedForward.forward
6. MOEFeedForward.forward
7. generate
8. MindChenConfig
9. RMSNorm / RoPE 辅助函数
```

原因：

```text
forward 是模型真正跑起来的路线；
先看 forward，知道数据怎么流；
再回头看 __init__，知道模块怎么建。
```

---

## 22. 一张最终总图：Dense 和 MoE 的区别

```text
Dense MindChen

input_ids
  -> Embedding
  -> Block x K
       -> RMSNorm
       -> GQA
       -> Add
       -> RMSNorm
       -> FeedForward
       -> Add
  -> RMSNorm
  -> lm_head
  -> logits
```

```text
MoE MindChen

input_ids
  -> Embedding
  -> Block x K
       -> RMSNorm
       -> GQA
       -> Add
       -> RMSNorm
       -> MOEFeedForward
            -> Router
            -> Top-k Experts
            -> Weighted Sum
       -> Add
  -> RMSNorm
  -> lm_head
  -> logits
  -> aux_loss
```

一句话总结：

```text
Dense 和 MoE 的主干几乎一样；
区别是 Dense 用一个 FFN，MoE 用多个专家 FFN + Router。
```

---

## 23. 你可以自己动手打印一次形状

可以新建一个很小的测试脚本，帮助理解形状：

```python
import torch
from model.model_mindchen import MindChenConfig, MindChenForCausalLM

config = MindChenConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=128,
    max_position_embeddings=64,
)

model = MindChenForCausalLM(config)
input_ids = torch.randint(0, 128, (2, 10))
out = model(input_ids, labels=input_ids)

print(out.loss)
print(out.logits.shape)
```

你会看到：

```text
logits.shape = [2, 10, 128]
```

这说明：

```text
2 条样本
每条 10 个 token
每个位置输出 128 个词表分数
```

---

## 24. 最后记住 5 句话

1. MindChen 是 Decoder-only Causal LM，核心任务是预测下一个 token。
2. 图里的 `Transformer Layer x K` 对应代码里的 `MindChenBlock` 堆叠。
3. 图里的 `GQA` 对应 `Attention`，用 Q/K/V、RoPE、mask、softmax、加权 V 实现。
4. Dense 图里的 `FFN` 对应 `FeedForward`，MoE 图里的 `MoE-FFN` 对应 `MOEFeedForward`。
5. 最后的 `Linear + Softmax` 对应 `lm_head + softmax/cross_entropy`，用于得到下一个 token 的概率或训练 loss。
