# MindChen 新手学习文档

本文按“先模型架构，后训练代码”的顺序讲解当前代码。建议一边读本文，一边打开这些文件对照：

- `model/model_mindchen.py`：模型主体，包含配置、注意力、FFN、MoE、生成函数。
- `model/model_lora.py`：LoRA 参数高效微调实现。
- `dataset/lm_dataset.py`：不同训练阶段的数据如何变成 `input_ids`、`labels`、`mask`。
- `trainer/train_pretrain.py`：预训练。
- `trainer/train_full_sft.py`：全量监督微调。
- `trainer/train_lora.py`：LoRA 微调。
- `trainer/train_dpo.py`：偏好优化。
- `trainer/train_distillation.py`：知识蒸馏。
- `trainer/train_grpo.py`、`trainer/train_ppo.py`、`trainer/train_agent.py`：强化学习和工具调用进阶训练。

## 1. 先用一句话理解 MindChen

MindChen 是一个 Decoder-only Transformer 语言模型。它的任务是：给定前面的 token，预测下一个 token。

例如训练文本是：

```text
我 喜欢 学习
```

模型会学习：

```text
看到 “我”       -> 预测 “喜欢”
看到 “我 喜欢”  -> 预测 “学习”
```

聊天、续写、推理、工具调用，本质上都建立在这个“预测下一个 token”的能力上。

## 2. 文本进入模型后的完整路径

整体流程如下：

```text
原始文本
  -> tokenizer 分词
  -> input_ids: [batch, seq_len]
  -> Embedding: [batch, seq_len, hidden_size]
  -> 多层 MindChenBlock
       -> RMSNorm
       -> Causal Self-Attention
       -> 残差连接
       -> RMSNorm
       -> FeedForward 或 MoEFeedForward
       -> 残差连接
  -> 最终 RMSNorm
  -> lm_head
  -> logits: [batch, seq_len, vocab_size]
  -> 训练时算 loss，推理时采样下一个 token
```

三个最重要的维度：

- `batch`：一次喂给模型多少条样本。
- `seq_len`：每条样本有多少个 token。
- `hidden_size`：每个 token 在模型内部用多少维向量表示，默认是 `768`。

## 3. `MindChenConfig`：模型尺寸说明书

`MindChenConfig` 不保存权重，只保存模型结构参数。你可以把它理解成“图纸”。

常见参数：

- `hidden_size`：模型内部向量维度，越大模型容量越强，显存占用也越大。
- `num_hidden_layers`：Transformer Block 层数，默认 8 层。
- `vocab_size`：词表大小，当前默认 6400。
- `num_attention_heads`：Query 注意力头数。
- `num_key_value_heads`：Key/Value 头数。小于 Query 头数时就是 GQA，可以省 KV cache。
- `intermediate_size`：FFN 中间层维度。
- `max_position_embeddings`：RoPE 位置编码最多预计算多长。
- `use_moe`：是否使用 MoE 专家混合结构。
- `num_experts`：MoE 中每层有多少个专家。
- `num_experts_per_tok`：每个 token 选择几个专家。

## 4. Embedding：把 token id 变成向量

模型不能直接理解中文、英文字符串。Tokenizer 会先把文本变成数字 token id。

例如：

```text
"你好" -> [123, 456]
```

`nn.Embedding(vocab_size, hidden_size)` 会把每个 token id 查表变成一个 `hidden_size` 维向量：

```text
[batch, seq_len] -> [batch, seq_len, hidden_size]
```

这一步在 `MindChenModel.embed_tokens` 中完成。

## 5. RMSNorm：让数值更稳定

`RMSNorm` 是归一化层。它的作用是让向量数值不要忽大忽小，训练更稳定。

和 LayerNorm 相比，RMSNorm 不减均值，只按均方根缩放，计算更简单。很多现代 LLM 都用 RMSNorm。

代码位置：`RMSNorm`。

## 6. RoPE：让模型知道 token 的位置

Transformer 本身看一组 token 时，不天然知道谁在前谁在后。RoPE 的作用就是把“位置”编码进 Query 和 Key。

MindChen 中：

- `precompute_freqs_cis` 提前算好每个位置需要的 `cos/sin`。
- `apply_rotary_pos_emb` 把 `cos/sin` 应用到 Q/K。

为什么只作用在 Q/K 上？

因为注意力分数来自 `Q @ K^T`。只要 Q/K 带有位置信息，模型就能在计算注意力时感知相对位置。

## 7. Attention：模型如何看上下文

`Attention` 是语言模型的核心。它让当前位置的 token 能从前面的 token 中取信息。

MindChen 的 attention 做了这些事：

1. 输入 hidden states。
2. 通过 `q_proj`、`k_proj`、`v_proj` 得到 Q/K/V。
3. 拆成多个 head。
4. 对 Q/K 做 RMSNorm。
5. 给 Q/K 加 RoPE。
6. 如果是生成阶段，把过去的 K/V cache 拼上。
7. 做因果 self-attention。
8. 多头结果拼回去，再过 `o_proj`。

### 7.1 因果 mask

语言模型预测下一个 token 时，不能偷看未来 token。

所以 attention 中会把未来位置的分数设成 `-inf`，softmax 后概率接近 0。

### 7.2 GQA

默认配置中：

```text
num_attention_heads = 8
num_key_value_heads = 4
```

这表示 Query 有 8 个头，但 Key/Value 只有 4 个头。计算时通过 `repeat_kv` 把 K/V 复用到 8 个头。

这样能减少推理时 KV cache 的显存占用。

### 7.3 KV Cache

生成文本时，模型一次生成一个 token。

如果每次都重新计算历史 token 的 K/V，会很浪费。KV cache 会把过去 token 的 K/V 存起来，下一步只计算新 token，再和旧 cache 拼接。

代码中：

- 输入：`past_key_value`
- 输出：`past_kv`
- 控制开关：`use_cache`

## 8. FFN：每个 token 自己做非线性变换

Attention 负责“看上下文”，FFN 负责“加工当前 token 的信息”。

MindChen 的普通 FFN 是 SwiGLU 风格：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

直觉上：

- `up_proj` 提供候选信息。
- `gate_proj + silu` 像一个门，决定哪些信息通过。
- `down_proj` 把中间维度压回 `hidden_size`。

代码位置：`FeedForward`。

## 9. MoE：多个专家，只激活少数专家

当 `use_moe=True` 时，普通 FFN 会换成 `MOEFeedForward`。

MoE 的直觉：

```text
普通 FFN：每个 token 都走同一个 FFN。
MoE：每个 token 先经过 gate，选择最合适的几个专家 FFN。
```

代码流程：

1. `gate(x)` 给每个 token 计算专家概率。
2. `torch.topk` 选出 top-k 专家。
3. 只让选中的专家处理这个 token。
4. 按专家权重加权合并输出。
5. 训练时额外计算 `aux_loss`，鼓励专家负载均衡。

MoE 的好处是总参数量可以更大，但每个 token 实际只激活少量专家，所以计算量可控。

## 10. MindChenBlock：一个完整 Transformer 层

一个 `MindChenBlock` 的结构是：

```text
x
 -> RMSNorm
 -> Attention
 -> 加回原 x，形成残差
 -> RMSNorm
 -> FFN 或 MoE
 -> 再加残差
```

残差连接很重要。它让模型不必每一层都重新学习全部信息，而是在原信息上做增量修改，训练更稳定。

## 11. MindChenModel：堆叠多层 Block

`MindChenModel` 包含：

- `embed_tokens`
- `layers`
- `norm`
- `freqs_cos/freqs_sin`

它输出的是 hidden states，还不是词表概率。

如果配置是默认 8 层，那么输入会依次经过 8 个 `MindChenBlock`。

## 12. MindChenForCausalLM：加上输出头

`MindChenForCausalLM` 在 `MindChenModel` 外面加了一个 `lm_head`：

```text
[batch, seq_len, hidden_size]
  -> lm_head
[batch, seq_len, vocab_size]
```

`vocab_size` 维输出叫 logits，表示每个位置对词表中每个 token 的打分。

### 12.1 训练 loss 的错位

语言模型训练不是让第 i 个位置预测自己，而是预测下一个 token。

代码中：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
```

含义：

```text
第 0 个位置的 logits -> 预测第 1 个 token
第 1 个位置的 logits -> 预测第 2 个 token
```

`labels` 中的 `-100` 会被忽略。SFT 中 user/system prompt 部分就是靠 `-100` 避免计算 loss。

### 12.2 权重共享

当 `tie_word_embeddings=True` 时：

```python
self.model.embed_tokens.weight = self.lm_head.weight
```

输入词向量和输出分类头共用一份权重，可以减少参数量，也常见于小模型。

## 13. generate：模型如何生成回答

`generate` 是自回归循环：

1. 输入 prompt。
2. 前向传播得到最后一个位置的 logits。
3. 根据 logits 选出下一个 token。
4. 拼到输入后面。
5. 重复直到生成 EOS 或达到最大长度。

采样参数：

- `temperature`：越高越随机，越低越保守。
- `top_k`：只保留概率最高的 k 个候选。
- `top_p`：只保留累计概率达到 p 的候选集合。
- `repetition_penalty`：惩罚已经出现过的 token，减少重复。
- `do_sample=False`：使用贪心解码，每次选概率最高的 token。

## 14. LoRA：只训练很少的参数

`model/model_lora.py` 实现了 LoRA。

普通线性层：

```text
y = W x
```

加 LoRA 后：

```text
y = W x + B(Ax)
```

其中：

- `W` 是原模型参数，冻结不训练。
- `A` 是降维矩阵。
- `B` 是升维矩阵。
- `rank` 控制 LoRA 分支有多大。

训练时只更新 `A/B`，所以参数量很小。合并时可以把 `B @ A` 加回原始 `W`。

## 15. 训练代码总览

推荐学习顺序：

```text
train_tokenizer.py     了解文本如何变 token
train_pretrain.py      从零学习语言续写
train_full_sft.py      学会聊天和指令格式
train_lora.py          小成本领域微调
train_dpo.py           用偏好数据优化回答质量
train_distillation.py  用老师模型教学生模型
train_grpo.py / train_ppo.py / train_agent.py  强化学习进阶
```

## 16. Dataset：训练数据如何变成模型输入

### 16.1 PretrainDataset

输入数据通常是：

```json
{"text": "一段普通文本"}
```

输出：

```text
input_ids = [BOS] + tokens + [EOS] + padding
labels = input_ids 的拷贝，但 padding 位置为 -100
```

训练目标：整段文本都参与 next-token prediction。

### 16.2 SFTDataset

输入数据通常是：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮你？"}
  ]
}
```

处理流程：

1. `apply_chat_template` 把多轮消息拼成文本。
2. 分词得到 `input_ids`。
3. `generate_labels` 只把 assistant 回复区间填入 labels。
4. user/system/padding 位置都是 `-100`。

这样模型只会被训练去生成 assistant 的回答，不会被训练去复读用户问题。

### 16.3 DPODataset

DPO 数据有两份回答：

- `chosen`：好回答。
- `rejected`：差回答。

Dataset 会输出：

- `x_chosen`
- `y_chosen`
- `mask_chosen`
- `x_rejected`
- `y_rejected`
- `mask_rejected`

mask 只在 assistant 回复区间为 1。

### 16.4 RLAIFDataset

强化学习阶段只返回 prompt，让当前模型自己生成回答。

生成后的回答再由 reward model 和规则打分。

## 17. 所有训练脚本的共同结构

大多数训练脚本都有类似结构：

```text
1. 解析命令行参数
2. 初始化 DDP 和随机种子
3. 创建 MindChenConfig
4. 设置混合精度 autocast
5. 初始化模型、tokenizer、dataset、optimizer
6. 如果 from_resume=1，恢复 checkpoint
7. 可选 torch.compile 和 DDP 包装
8. for epoch: 构建 DataLoader 并训练
9. 结束时清理分布式进程
```

训练一个 batch 的共同结构：

```text
input_ids/labels 放到 GPU
更新学习率
前向传播
计算 loss
loss / accumulation_steps
反向传播
满足梯度累积步数后：
  unscale
  梯度裁剪
  optimizer.step
  scaler.update
  optimizer.zero_grad
定期打印日志
定期保存权重和续训 checkpoint
```

## 18. 预训练：`train_pretrain.py`

预训练通常从随机初始化开始：

```bash
python trainer/train_pretrain.py --from_weight none
```

它使用 `PretrainDataset`，让模型学习普通文本的 next-token prediction。

核心 loss：

```text
loss = cross_entropy(logits shifted, labels shifted) + aux_loss
```

如果是 Dense 模型，`aux_loss=0`。如果是 MoE 模型，`aux_loss` 是专家负载均衡损失。

预训练后得到：

```text
../out/pretrain_768.pth
```

## 19. 全量 SFT：`train_full_sft.py`

SFT 默认从预训练权重开始：

```bash
python trainer/train_full_sft.py --from_weight pretrain
```

它使用 `SFTDataset`，训练模型按聊天模板回答。

关键点：

- user/system 只是上下文。
- assistant 回复才计算 loss。
- 所有模型参数都会更新。

SFT 后得到：

```text
../out/full_sft_768.pth
```

这个权重通常是后续 LoRA、DPO、PPO、GRPO 的起点。

## 20. LoRA 微调：`train_lora.py`

LoRA 默认从 `full_sft` 权重开始：

```bash
python trainer/train_lora.py --from_weight full_sft
```

核心流程：

1. 加载基础模型。
2. `apply_lora(model)` 给部分线性层挂 LoRA 分支。
3. 冻结所有非 LoRA 参数。
4. optimizer 只接收 LoRA 参数。
5. 保存时只保存 LoRA 权重。

适合：

- 小数据集领域微调。
- 医疗、角色、自我认知等轻量适配。
- 想节省显存和训练成本。

## 21. DPO：`train_dpo.py`

DPO 使用偏好数据：

```text
同一个问题：
  chosen   = 更好的回答
  rejected = 更差的回答
```

训练中有两个模型：

- `model`：策略模型，会训练。
- `ref_model`：参考模型，冻结。

DPO 的直觉：

```text
如果策略模型比参考模型更偏向 chosen，就奖励；
如果策略模型更偏向 rejected，就惩罚。
```

`beta` 控制偏好优化强度。太大可能更新过猛，太小可能效果不明显。

## 22. 知识蒸馏：`train_distillation.py`

蒸馏中有：

- 学生模型：最终要训练和保存。
- 教师模型：冻结，只提供软标签。

总 loss：

```text
loss = alpha * CE + (1 - alpha) * KL
```

含义：

- `CE`：学生学习数据集标准答案。
- `KL`：学生学习老师模型的概率分布。
- `alpha` 越大越信数据，越小越信老师。
- `temperature` 越大，老师分布越软，包含更多暗知识。

## 23. GRPO / PPO：强化学习进阶

强化学习阶段不直接给标准答案，而是让模型生成，再打分。

### 23.1 GRPO

`train_grpo.py` 中，每个 prompt 会生成多个回答。

同组回答比较 reward：

```text
advantage = (reward - 同组平均 reward) / 同组标准差
```

好于平均的回答提高概率，差于平均的回答降低概率。

同时用 `ref_model` 计算 KL 惩罚，防止模型偏离 SFT 模型太远。

### 23.2 PPO

`train_ppo.py` 多了一个 Critic。

- Actor：生成回答。
- Critic：估计 value。
- Reward：来自 reward model 和规则。
- Advantage：由 reward 和 value 通过 GAE 算出。

PPO 使用 clip 限制新旧策略变化，避免训练不稳定。

## 24. Agent RL：`train_agent.py`

Agent RL 是工具调用版本的强化学习。

模型可能输出：

```text
<tool_call>
{"name": "calculate_math", "arguments": {"expression": "1+1"}}
</tool_call>
```

脚本会：

1. 解析 tool call。
2. 执行本地模拟工具。
3. 把 tool response 加回上下文。
4. 继续让模型生成。
5. 根据工具是否正确、最终答案是否命中 gt 来打 reward。

注意：工具返回的 token 是环境观察，不是模型生成的，所以训练时 mask 为 0，不参与 policy loss。

## 25. Checkpoint 和续训

`trainer/trainer_utils.py` 中的 `lm_checkpoint` 负责保存两类东西：

```text
../out/*.pth
  普通模型权重，常用于推理或下一阶段训练

../checkpoints/*_resume.pth
  续训状态，包含 model、optimizer、scaler、epoch、step、scheduler 等
```

如果训练中断，可以加：

```bash
--from_resume 1
```

脚本会尝试从 resume checkpoint 恢复。

## 26. 新手最该先看懂的 6 个问题

1. `input_ids` 是什么？
   是 tokenizer 把文本转成的 token id。

2. `labels=-100` 是什么？
   表示这个位置不参与 loss。SFT 中 user/system 部分通常是 -100。

3. 为什么 logits 和 labels 要错位？
   因为语言模型是用当前位置预测下一个 token。

4. Attention 在做什么？
   让每个 token 从前面的 token 中取信息。

5. FFN 在做什么？
   对每个 token 的向量做非线性加工。

6. LoRA 为什么省参数？
   它冻结原始大矩阵，只训练旁边很小的低秩增量矩阵。

## 27. 推荐阅读顺序

如果你是第一次读这个项目，建议按这个顺序：

```text
1. docs/mindchen_beginner_guide.md
2. model/model_mindchen.py
3. dataset/lm_dataset.py
4. trainer/train_pretrain.py
5. trainer/train_full_sft.py
6. model/model_lora.py + trainer/train_lora.py
7. trainer/train_dpo.py
8. trainer/train_distillation.py
9. trainer/train_grpo.py / trainer/train_ppo.py
10. trainer/train_agent.py
```

先不要一开始就深挖 PPO/Agent RL。先把“模型前向传播”和“SFT 的 labels mask”吃透，后面的训练代码会容易很多。
