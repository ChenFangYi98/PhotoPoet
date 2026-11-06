# 诗词古文全量 SFT 使用说明

新的优化训练入口见 [README_自适应SFT.md](README_自适应SFT.md)：从指定继续预训练权重初始化，使用 `sft_t2t_with_poetry_clean_v1` 基础库及增补通道，支持任务配比、动态监督权重与完整目标窗口。本文以下说明保留原 `train_full_sft.py` 的行为。

项目当前位置：`./`。训练入口根据自身文件位置自动定位项目，默认权重、数据和输出目录随项目一起迁移，无需填写盘符。

## 直接开始训练

在 PowerShell 执行：

```powershell
cd ./
python trainer/train_full_sft.py
```

也可以使用已填写所有主要参数的启动脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-FullSFT.ps1
```

Python 环境仍在 C 盘，与项目搬盘无关。如果以后重装环境，修改启动脚本中的 `$pythonExe`。

## 默认输入与参数

以下路径均相对于项目根目录：

| 项目 | 默认配置 |
| --- | --- |
| 初始权重 | `out/continue_pretrain/poetry_gujilab_20261022_190751_427063/pretrain_poetry_gujilab_768_moe_best.pth` |
| 训练数据 | `dataset/sft_t2t_with_poetry/sft_t2t_with_poetry_train.jsonl` |
| 验证数据 | `dataset/sft_t2t_with_poetry/sft_t2t_with_poetry_valid.jsonl` |
| 输出目录 | `out/full_sft/poetry_sft_<时间戳>/` |
| 模型结构 | hidden_size=768，8 层，MoE；全部参数参与训练 |
| 训练轮数 | 1 |
| batch 与累积 | batch_size=8，accumulation_steps=8，名义每次更新 64 条 |
| 序列长度 | 最长 1024 token，batch 内动态补齐 |
| 优化器 | AdamW，学习率 1e-5，weight_decay=0.01，grad_clip=1.0 |
| 学习率计划 | 3% warmup，余弦衰减至峰值的 10% |
| 训练设备与精度 | 默认可用 CUDA；BF16 混合精度，FP32 主参数和导出权重 |
| 记录与保存 | 每次更新记录指标；控制台每 50 次；每 1000 次验证并保存 |

完整训练数据为 6,074,666 条，验证数据为 65,385 条，默认计划约 94,917 次参数更新。首次完整加载可能需要生成 Arrow 缓存。原始预训练权重不会被覆盖。

只监督 assistant 的回答、推理内容、工具调用和结束标记；system、user、工具返回及 padding 不计入损失。回答被截断时不人为补结束标记。截断后没有监督 token 的样本会排除并计数。

每次周期验证使用固定的 1024 条验证子集，按 assistant token 数加权计算 NLL/PPL，并按同一子集的 NLL 选择 best。训练前也验证一次，因此如果训练后没有改善，best 可以对应第 0 次更新。训练结束后加载导出的 best，在完整验证集上再验证一次，结果单独保存。

验证文件按现有数据报告主要用于诗词古文，不能据此判断通用能力是否保持；预训练可能已经见过作品。模型能力对比仍应结合 `eval/eval_mindchen_quality.py` 的任务评测和人工评分。

## 训练前检查与小规模试跑

只加载权重、准备完整数据、保存监督预览和审计，不更新模型：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-FullSFT.ps1 -PrepareOnly
```

实际执行小规模训练、验证、权重保存，使用独立输出目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-FullSFT.ps1 -SmokeTest
```

试跑使用前 192 条训练数据、前 16 条验证数据，执行 3 次参数更新，其分数不能代表正式训练效果。此前已完成该 GPU 试跑和断点恢复检查。搬至 F 盘后，已重新完成 21 项单元测试、严格权重加载和小样本数据准备，并核对旧 SFT 断点重定位后的权重哈希及数据文件元信息一致；未启动正式训练。

## 查看训练过程与恢复

每个运行目录保存：

- `train.log`：训练、验证、保存及异常信息。
- `metrics.csv`、`metrics.jsonl`：每次更新的 loss、MoE 辅助损失、学习率、梯度范数、速度、GPU 显存、样本和监督 token 计数、截断/排除计数，以及验证指标。
- `run_config.json`：参数、依赖版本、模型结构、输入文件信息、源权重 SHA256、代码与 tokenizer 指纹。
- `data_audit_*.json`、`supervision_preview.jsonl`、`validation_indices.json`：抽样数据审计、原文/监督区域预览、固定验证索引；不会重复复制整个数据集。
- `full_sft_poetry_768_moe.pth`：最新纯权重；`full_sft_poetry_768_moe_best.pth`：固定验证子集选出的最佳纯权重。默认 FP32，保留低学习率的小幅更新。
- `latest_resume.pth`：FP32 主参数、AdamW、scaler、随机状态、数据游标和进度，供继续训练使用。
- `summary.json`：当前完成状态；`full_validation_best.json`：结束时 best 的完整验证结果；异常时另存 `failure.json`。

按一次 Ctrl+C 后等待当前更新完成，程序会验证、保存并退出。意外终止可以恢复到最近保存的断点。用实际运行目录替换下面的 `<运行目录>`：

```powershell
python trainer/train_full_sft.py --resume "./out\full_sft\<运行目录>\latest_resume.pth"
```

恢复时自动使用断点参数，不用重新填写训练配置。纯权重不能替代完整恢复断点。原 C 盘项目内的路径会在内存中映射到当前项目，仍校验源权重 SHA256、文件大小、修改时间和数据元信息；历史日志不改写。源文件发生变化时会拒绝使用旧游标恢复。
