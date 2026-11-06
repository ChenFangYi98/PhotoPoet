# 保留选中专家概率的 SFT

当前入口仍为 `trainer/train_adaptive_sft.py`。数据和初始化路径已填写，无需手动更改：

- 数据：`dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready/manifest.json`，继续使用 clean_v1 原库及已有增补数据。
- 初始化：`out/continue_pretrain/poetry_gujilab_20261022_190751_427063/pretrain_poetry_gujilab_768_moe_best.pth`。
- 模型：768 维、8 层、4 个专家、每 token 选择 1 个专家；参数结构没有增加或删减。
- 新默认：`norm_topk_prob=False`，`moe_router_gradient=legacy`。这里 legacy 指直接使用当前前向公式的真实梯度，不是恢复 p/p。
- 主干学习率 `1e-5`、分类头 `3e-4`、预热占总预算 3%、梯度裁剪 1；其余数据配比和安全策略保留。这是可运行起点，未证明最优。

## 现在的计算

```text
p = softmax(gate(hidden))
expert = argmax(p)
输出 = p[expert] × 专家输出
```

不再对 top-1 选中概率重新归一化，所以不会产生 p/p=1 的梯度抵消。专家选择的离散索引本身仍不可微，门控通过连续概率权重获得主任务梯度。此模式不用替代梯度。

该修改保留参数数量与形状，能够 strict=True 加载原预训练权重，但会改变专家输出的尺度。训练器先在修改后的计算方式下做初始验证，再开始更新；原始预训练文件不覆盖。新旧运行的损失不能直接横向比较，最终准确率、创作能力仍需统一评测。

`MindChenConfig` 类的通用默认值仍为 True，用于兼容无配置的历史模型和其他训练入口。当前 SFT 入口和启动脚本显式设置 False，不会依赖那个旧默认值。

## 直接运行

```powershell
cd ./

# 无更新检查，含所有源文件 SHA256 复核
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -PrepareOnly -VerifySourceSha256

# 独立目录中最多 3 次小批量更新
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -SmokeTest

# 用户手动开始新的正式训练：默认就是保留原概率
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -VerifySourceSha256
```

直接 `python trainer/train_adaptive_sft.py` 也使用相同的新默认。每次更新仍为 64 条原样本，任务配比、动态权重和完整上下文筛选不变。日志应出现 `norm_topk_prob=False | 选中概率保留`。

旧完整断点不能直接接入新流程；新协议为 `adaptive_sft_probability_v1`。新完整断点包含模型配置，恢复时不允许改变归一化设置，并核对全部训练依赖。切换前代码备份位于 `out/training_archives/legacy_organize_20261027_122054/adaptive_sft/probability_gate_20261027/before/`。

## 保存与推理必须一致

每个 latest/best 权重都会导出自己的伴随文件，例如：

```text
adaptive_768_moe_best.pth
adaptive_768_moe_best.pth.config.json
```

伴随文件包含完整配置和实际权重 SHA256。`eval_llm.py`、`eval/eval_mindchen_quality.py` 均自动读取它，恢复 False，并拒绝不匹配的权重哈希或冲突的 `--norm_topk_prob`。分类头不是推理必需文件。

复制或移动新权重时，须同时复制对应 `.pth.config.json`。旧裸权重无配置时默认按旧归一化行为加载，并打印配置提示。仅有新裸权重、遗失配置时，已知架构可显式 `--norm_topk_prob 0`；完整的非默认结构信息仍应通过保存配置恢复，不应猜测。

```powershell
python eval_llm.py --checkpoint out/adaptive_sft/你的运行目录/adaptive_768_moe_best.pth --prompt_style chat

python eval/eval_mindchen_quality.py --checkpoint out/adaptive_sft/你的运行目录/adaptive_768_moe_best.pth --prompt_style chat
```

对照旧归一化机制可新建运行并显式 `--norm_topk_prob 1`，PowerShell 对应 `-NormTopkProb 1`。旧 straight-through 实验须再加 `--moe_router_gradient straight_through`；不能与新概率模式混用。

## 验证范围

真实指定 768-MoE 权重，在不包含辅助损失、不执行优化器更新的检查中，8 层门控的 LM 梯度从约 1e-8 恢复到约 0.42～1.98，BF16 也有正常非零梯度。此结果只证明梯度路径恢复，不证明模型知识准确率或创作质量提高。工程测试、真实短训练和保存加载的证据见 `out/adaptive_sft/probability_gate_20261027/`。
