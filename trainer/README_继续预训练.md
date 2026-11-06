# 在已有768 MoE模型上继续预训练

入口：`trainer/train_continue_pretrain.py`。默认参数已填写，可直接运行。原脚本备份为 `train_continue_pretrain.py.bak`。

基础权重：`./out\pretrain_768_moe.pth`。

新数据：`./dataset\pretrain_t2t_with_poetry_gujilab.jsonl`。

## 现在由你启动正式训练

在项目根目录的PowerShell执行：

```powershell
cd ./
powershell -NoProfile -ExecutionPolicy Bypass -File .\trainer\Start-ContinuePretrain.ps1
```

启动脚本固定使用已经验证的 `python`，不依赖IDE当前选中的解释器。所有训练参数已在脚本中明确填写。本机PowerShell默认禁止直接运行ps1，上述命令仅对当前启动进程指定执行策略，不修改系统设置。

也可使用conda环境直接启动Python入口，两种方式默认配置相同：

```powershell
conda activate mindchen
python trainer/train_continue_pretrain.py
```

每次新训练自动创建 `out/continue_pretrain/poetry_gujilab_<时间>/`，不会覆盖基础权重或旧运行。先前的冒烟训练权重留在独立目录，正式入口仍从你指定的基础权重开始。

## 已验证的参数

| 参数 | 值 |
|---|---|
| 模型 | hidden_size=768，8层，MoE开启；其余采用本项目MindChenConfig |
| 设备/精度 | CUDA:0，bfloat16；模型主参数/AdamW状态为FP32 |
| epochs | 1 |
| batch_size / accumulation_steps | 16 / 4，有效batch=64条 |
| 学习率 | 5e-5，前3%更新线性warmup，后续余弦降到5e-6 |
| weight_decay / grad_clip | 0.01 / 1.0 |
| max_seq_len | 340，保持原预训练长度，含BOS/EOS |
| 数据加载 | num_workers=0，Windows下已验证 |
| 控制台日志 | 每50次优化器更新打印；CSV/JSONL每次更新记录 |
| 验证 / 断点 | 每1000次更新；每轮结束、计划结束或安全停止也执行 |
| 验证样本 / 验证batch | 256 / 8 |
| seed | 42 |

完整数据已实际读取并构建Arrow缓存，共9,011,340条。默认前256条固定作新阶段验证，剩余9,011,084条训练；每轮563,193个microbatch，共140,799次优化器更新。有效batch在最后不足一个窗口时按实际有效token归一化，不丢弃尾部样本。warmup为4,223次更新。

这份混合数据本身已打乱，验证前缀仅用于监控新阶段训练。它可能已出现在基础模型的旧预训练数据中，不能解释成独立泛化测试。验证原文会保存下来，便于复核。

原有PretrainDataset处理被完整复用：只保留每条文本前 `max_seq_len-2` 个token，添加BOS/EOS并屏蔽padding标签。长篇古文并不会被自动切成多段，340之后的内容未被本次样本利用；这是本次保持原训练设置的明确限制。

训练主损失按实际预测token加权；MoE辅助损失单列并加入总损失。日志中的token/s统计非padding的预测token，包含该更新的计算，不包含分词加载、验证与保存；ETA按本次整体进度估计，刚启动时会波动。

## 训练记录和模型文件

| 文件 | 内容 |
|---|---|
| `run_config.json` | 全部参数、模型配置、Python/torch等版本、GPU、基础权重SHA256、数据大小/修改时间、tokenizer与脚本指纹、实际样本数/更新数 |
| `train.log` | 启动、进度、验证、保存、恢复和异常堆栈 |
| `metrics.csv` / `metrics.jsonl` | 每次更新的LM loss、MoE aux loss、总loss、LR、裁剪前梯度范数、样本/token数、速度、显存、ETA；验证NLL/PPL和best NLL |
| `validation_samples.jsonl` | 固定验证原文，未加入本轮训练 |
| `pretrain_poetry_gujilab_768_moe.pth` | 最新纯state_dict推理权重，FP16，可用于评测或下一阶段 |
| `pretrain_poetry_gujilab_768_moe_best.pth` | 当前最低验证NLL对应的纯权重；包括训练前验证，若后续未改善可能仍是基础模型 |
| `latest_resume.pth` | FP32模型、AdamW状态、scaler、随机状态、已提交的数据游标、学习率计划；约2.4 GB |
| `summary.json` | 完成/安全停止状态、累计更新与样本/token数、best NLL、模型路径 |
| `failure.json` | 异常信息；异常时保留最近成功保存的断点 |
| `resume_config_<时间>.json` | 本次恢复参数和环境，保留原run_config |

验证PPL=`exp(token加权NLL)`，统计当前分词器、截断方式下的预测误差，不等同于文学能力分。不要追求一个跨数据集通用的“合格PPL阈值”。每个新运行通常需要约3.2 GB模型/断点空间，保存时还需要临时文件空间；Arrow缓存另占磁盘。

## 停止与恢复

训练时按一次 **Ctrl+C**，脚本会完成当前优化器更新，验证并保存后退出。等待出现 `stopped` 和保存路径，再关闭窗口。强制终止进程只能恢复到最近已保存的断点。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\trainer\Start-ContinuePretrain.ps1 -Resume "./out\continue_pretrain\你的运行目录\latest_resume.pth"
```

或：

```powershell
python trainer/train_continue_pretrain.py --resume "你的运行目录/latest_resume.pth"
```

恢复会自动还原原训练参数，不需要重新填写。恢复原AdamW/scaler/RNG和同一epoch剩余排列，跳过已提交batch，不从头跑数据。基础权重或数据文件变化时拒绝续训。学习率、batch、数据划分等影响训练的参数不能在恢复时偷偷改变；日志/验证/保存间隔可显式覆盖。

若异常前已有超出最近断点的指标行，恢复时会备份原metrics文件，再回退到已保存更新，避免把未提交进度和重跑进度混在一起。

`--max_updates N`会改变整个训练计划及LR调度，用于短实验。`--stop_after_updates N`只在累计第N次更新后停下，保持原计划，适合暂停后继续；二者不能混为一谈。

## 本次验证结果

- `out/continue_pretrain/prepared_gujilab_20261022/`：完整读取9,011,340条数据、严格加载基础权重、生成完整计划；没有执行优化器更新。
- `out/continue_pretrain/verified_smoke_20261022/`：192条训练+8条验证，batch16×累积4，计划3次更新。先运行到update2保存，再从完整断点恢复到update3；恢复从第9个microbatch继续，总计192条训练样本。
- GPU冒烟测试峰值显存约6.2 GB，loss和梯度均为有限值。三次短更新仅验证代码，不能证明继续预训练有效。
- `trainer/test_continue_pretrain.py`检查LR调度、恢复排列、RNG权重安全加载、验证token加权、数据留出与日志回退。

重新做小规模测试或只缓存数据：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\trainer\Start-ContinuePretrain.ps1 -SmokeTest
powershell -NoProfile -ExecutionPolicy Bypass -File .\trainer\Start-ContinuePretrain.ps1 -PrepareOnly
python -m unittest trainer.test_continue_pretrain -v
```

## 训练后评测

继续预训练产物仍使用 `pretrain` 提示模式，先和原始基线比较，再进入SFT：

```powershell
python eval/eval_mindchen_quality.py --checkpoint "你的运行目录/pretrain_poetry_gujilab_768_moe.pth" --domain poetry --prompt_style pretrain --choice_likelihood --no_sample_poetry --run_name continued_pretrain_literature
python eval/compare_runs.py eval/evaluation_results/pretrain_768_moe_literature_baseline_scored eval/evaluation_results/continued_pretrain_literature --output eval/evaluation_results/继续预训练对比.csv
```

看固定验证NLL/PPL、重复情况和诗词古文分类指标是否共同改善。继续预训练并不会自动获得稳定的指令遵循能力，后续仍需对应的监督微调与独立评测。
