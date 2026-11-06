# 确定性与开放任务兼顾的纯 SFT

2026-10-27：当前训练已复用 `dataset/lm_dataset.py` 的 `pre_processing_chat`，默认对没有 system、没有工具调用的普通训练对话以 20% 概率加入 mindchen／chenfangyi 身份提示，验证题面不添加。可通过 `--add_system_ratio 0` 关闭；随机选择由更新抽样确定，恢复时可复现。现有自适应采样、动态权重、助手标签及完整上下文筛选保留。旧自我介绍中明确写旧训练者 chen 的样本不会附加相冲突的新提示，但其答案并未自动清洗；仅添加 system 不能保证模型记住身份或修正旧答案。修改后训练依赖指纹改变，旧完整断点不可直接恢复，新运行使用原预训练初始化。

入口为 `trainer/train_adaptive_sft.py`，启动脚本为 `trainer/Start-AdaptiveSFT.ps1`。默认从用户指定的继续预训练权重开始新的 SFT，数据以 `sft_t2t_with_poetry_clean_v1` 为基础。使用正确 assistant 目标的交叉熵，以及辅助任务分类交叉熵；不使用强化学习、DPO、负答案排序或教师 KL。原 `train_full_sft.py`、`lm_dataset.py`、基础 JSONL 和原权重均保持不变。

2026-10-27 当前版本已加入审计修正，并按用户选择默认保留选中专家的原始概率（`--norm_topk_prob 0`），使 LM 梯度能到达 MoE 门控。当前完整断点协议为 `adaptive_sft_probability_v1`；下文采样版本只描述采样算法。见 [概率门控运行说明](README_概率门控SFT.md) 和 [审计修正说明](README_SFT审计修正.md)。必须新建运行，不能把旧断点接入新代码。

## 数据与占比

逐文件用途和旧运行实测占比见 [当前训练数据集报告](../dataset/sft_t2t_with_poetry_clean_v1/当前训练数据集报告.md)。2026-10-27 已修正发现的采样问题：47 条作者澄清不再占约 25% 的训练抽样。当前 `train_adaptive_sft.py` 使用 `adaptive_sft_sampling.py` 的修正版；旧运行不会因磁盘代码修改而自动切换。

当前可用清单位于 `dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready/manifest.json`。以已完整扫描的基础训练 5,940,346 行、原验证 58,559 行为基础，并纳入 `factual_v1` 作者问答。v3在原有4,096/512条计算题上新增60,000条训练、6,000条验证，覆盖11类任务。明细见 `dataset/sft_t2t_with_poetry_clean_v1/deterministic_v3/扩充说明.md`。计算类独立程序核验，文学关系回查原文和元数据，不等同于逐条史料考证。

索引保存原文件字节位置，训练随机读取，不复制或就地追加 14 GB 基础库。原验证作品/提示受保护；基础训练按作品或相同提示组再留出约 3% 作内部开发，作者问法也遵循同一作品组分割。训练与开发索引的作品/提示组交叉为 0。为避免泄漏而隔离的 12,990 条不进入训练；它们不是因长度而被排除。

当前可索引训练池：

| 任务弱标签 | 条数 | 原池比例 |
| --- | ---: | ---: |
| 确定性 fact | 1,048,282 | 约 17.9% |
| 开放 open | 381,578 | 约 6.5% |
| 一般/未明确识别 general | 4,414,586 | 约 75.5% |

模板、明确创作指令等形成可审计弱标签；混合任务保守归 general。多轮对话中任务可能变化，只用首段问题无法判断未来任务，因此这类样本保留完整 LM 监督，但不用于辅助分类头监督。一般类别不代表“完全没有事实”，任务标签也不保证答案真实。

默认比例根据训练池数量自动计算 `q_task = n_task^0.35 / sum(n^0.35)`，得到约 **29.80% / 20.92% / 49.28%**。温度平滑减少一般类的数量优势，同时保留其最大份额。每次更新默认抽 64 条原样本，整数配额为 **19 / 13 / 32**。这是按原样本计数，不是目标 token 比例；每次更新都记录实际抽样。

事实类内部按作者 45%、新增确定性通道 20%、基础任务 35% 抽样，避免新增小通道被百万原文样本淹没。v3新增通道同时包含程序核验和文学来源样本，逐条来源系数仍区分。比例是当前实验设定，非已证明最优；可用 `--task_ratios FACT OPEN GENERAL` 显式覆盖任务配比。

open/general 类内改为按通道条数选择，再在通道内均匀抽行，等价于在相应任务的全部样本中均匀抽取。47 条澄清保留，理论上约占全部训练抽样的 **0.000532%**，而不是旧版的约 25%。真实索引模拟 10,000 次更新（640,000 次抽样）中出现 4 次。`run_config.json` 的 `sampling.train_plan` 记录各通道期望份额，`metrics.jsonl` 的 `source_counts` 记录实际抽样次数；期望份额不是 token 或损失权重比例。

通用回答替代重复的短澄清模板后，每次更新的实际监督 token 和耗时可能增加；不能把旧版速度直接用于估算新训练时长，也不要把换采样后的损失变化当作故障或能力结论。

## 动态监督权重

新增一个很小的 `TaskRouter`，输入为主干模型在第一个 assistant 回答之前的 prompt 隐状态，最多保留最近 512 个 prompt token。分类头看不到目标回答；训练时分类交叉熵也更新共享主干。

它输出 fact/open/general 三类概率，**不是事实真伪判断或模型知识置信度**。当前按任务/原样本识别，并未实现任意回答内逐句事实分类。

默认权重：

```text
d = clip(sqrt(事实样本平均NLL的EMA / 初始事实NLL), 0.75, 1.25)
p_raw = 任务弱标签与分类头P(fact)的渐进混合
p = max(p_raw, 0.5 × fact弱标签)
w = clip(1 + 0.6 × 来源可靠性系数 × 标签可信系数 × p × d, 1, 1.8)

LM损失 = sum_i w_i × [第i条原样本全部assistant token的平均CE] / sum_i w_i
总损失 = LM损失 + 0.05 × prompt任务分类CE + MoE辅助损失
```

来源可靠性系数是启发式权重，不是概率。未经识别或来源不明的任务不会仅凭模型自信获得大权重。开放/一般样本保留基准权重 1。

分类头需要先学习：默认至少按 100 次更新的进度渐入，且开发集三类弱标签宏一致率达到 0.75 才启用预测参与；单条预测最大概率不足 0.6 时仍回退。这个一致率只衡量与弱标签的一致程度，不等于独立人工评估准确率。

对已标为 fact 且来源可靠的样本，混合概率至少保留 0.5，避免分类头自信错分时完全失去增强。难度仍是事实批次 EMA，不是逐条真伪检测；这项修改需要通过 uniform/static/dynamic 同预算对照检验效果。

权重计算停止梯度，避免模型通过把任务改判为开放类来减轻自身损失。EMA、分类头和优化器状态一同进入完整断点。`uniform / static / dynamic` 三种模式供消融；uniform 仍使用相同任务配额、样本归一与辅助分类，表示“无事实增权”的对照，不等于旧入口的全局 token 平均。

## 长样本与目标保留

默认 `--context_policy full_context`：完整编码后，整条对话超过 2048 token 则隔离，在同一通道确定性补样。参与更新的每条样本保留完整上下文与全部 assistant 目标；`quarantine.jsonl` 记录原因和原索引，源数据不删除。**这会减少可参与训练的数据，不是把模型的上下文能力扩展了。** 对被隔离的长文，后续需要重建独立可解的监督单元或单独做长上下文训练。

显式使用 `--context_policy windows` 才恢复旧分窗策略。窗口 2048 token、重叠 512；先完整编码并精确屏蔽非 assistant 标签，再分窗口：

- 已在前窗口监督的重叠 token 只作上下文，不重复计 loss。
- 同一原样本的全部窗口合计，恰好覆盖原 assistant 目标一次；窗口更多不增加这条样本的总权重。
- 长问题后面的回答仍得到监督；纯上下文窗口不做无效 LM 更新。
- 原对话中的真实 EOS、reasoning 和工具调用保留；不在窗口边界人为添加 EOS。工具返回、user/system 和 padding 不计 loss。
- 更新内将窗口按长度排序，动态补齐，每次最多前向两个窗口；累计所有窗口后才提交一次参数更新。

**保留全部目标 token，不等于保留完整远端上下文。** 超过窗口的很早材料可能不可见，依赖整篇长文的任务仍需扩大窗口或重新设计任务；不能把窗口法宣称为恢复了长上下文理解能力。

## 直接运行

所有参数已填写，项目路径跟随脚本所在目录：

| 配置 | 默认值 |
| --- | --- |
| 数据清单 | `dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready/manifest.json` |
| 初始化权重 | `out/continue_pretrain/poetry_gujilab_20261022_190751_427063/pretrain_poetry_gujilab_768_moe_best.pth` |
| 模型结构 | hidden 768，8 层，MoE |
| 学习率 / 预热 | 主干 `1e-5`，分类头 `3e-4`，总步数前 3% 预热 |
| 更新 / 显存批次 | 每次 64 条原样本；LM 每批 2 个窗口；分类每批 8 条 |
| 长样本 | 默认 full_context，超 2048 token 同通道补样；windows 为显式可选策略 |
| 权重 | dynamic，增强系数 0.6，上限 1.8，预测不可靠时回退标签 |
| 验证 / 保存 | 每 1000 次更新；128 条主 NLL、16 条澄清 NLL、64 道事实生成题、4 题各 2 次创作采样 |
| 总预算 | 1 轮等量采样，91,320 次更新；新建时间戳目录 |

直接运行 `python trainer/train_adaptive_sft.py` 使用相同默认配置。建议使用下面的 PowerShell 启动脚本，它固定到本机已验证的 mindchen Python 环境。参数是可运行的起点，最终能力提升仍需完整训练后的独立评估。

```powershell
cd ./

# 加载默认模型、完整索引并抽样审计，不更新参数
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -PrepareOnly

# 最多3次更新，独立目录，实际验证并保存
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -SmokeTest

# 由你手动启动新的正式SFT
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1

# 同一数据流上的无增权对照
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -WeightMode uniform

# 恢复本入口的完整断点
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -Resume "./out\adaptive_sft\实际目录\latest_resume.pth"
```

这里的 `-Resume` 仅用于修正版产生的断点。默认仍从指定继续预训练 best 权重开始新 SFT，不自动选择旧 SFT 模型。如明确希望把旧 SFT 的知识作为起点，可显式使用纯 LM 权重，新建优化器、分类头和验证基线：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -InitWeightPath "./out\adaptive_sft\旧运行目录\adaptive_768_moe_best.pth"
```

这不是续接旧断点，不能消除此前过采样已经产生的影响。判断修正本身的效果时，应从同一指定预训练权重另开对照；已有训练成果及旧断点仍保留。

默认初始化为 `out/continue_pretrain/poetry_gujilab_20261022_190751_427063/pretrain_poetry_gujilab_768_moe_best.pth`，主干学习率 1e-5，新分类头学习率 3e-4；BF16、3% warmup、梯度裁剪 1。全部 LM 参数参与更新。每次更新 64 条原样本，LM 每批最多 2 个窗口、prompt 分类每批最多 8 条，分别分批反传，累计后才更新参数。

默认 `--epochs 1 --max_updates 0`：以 5,844,446 条可索引训练数据设置等量采样预算，计划 **91,320 次更新**。这是按任务配比有放回采样，不保证逐条遍历全库；某些小通道会重复访问。`-MaxUpdates 2000` 可明确改为短实验，`-Epochs 2` 增加采样预算。更换初始化权重用新目录和 `--init_weight_path`，不能沿用旧优化器游标。

预热共 2739 次更新，因此第 1 次主干实际学习率约 `3.65e-9`，逐步升至 `1e-5`，之后余弦下降至 `1e-6`。日志中的 `lr` 是当次实际值；刚开始很小属于预期。首次训练先做初始验证并保存 best，完整断点还包含 AdamW 状态，写盘时会有等待。

若想按正式学习率计划先跑少量更新，用 `-RunName 自定目录名 -StopAfterUpdates 3`，只限制本次运行终点，不改 91,320 次的总预算。恢复时 `-StopAfterUpdates` 也指绝对更新编号。`-MaxUpdates 3` 则会改变整个学习率计划，仅适合短程实验。无参数启动始终新建正式训练目录；测试输出在独立目录中。

输出目录为 `out/adaptive_sft/<时间戳>/`。纯 LM 权重保持原模型格式，可供现有推理入口加载；配套辅助分类头单独存为 `task_router*.pth`，有 LM 文件哈希校验。任务识别可用：

```powershell
python trainer/inspect_sft_task.py --router "out/adaptive_sft/实际目录/task_router_best.pth" --prompt "请原创一首描写秋天的诗。"
```

## 验证、日志与边界

旧采样版本于 2026-10-26 完成的短程工程验证见 [历史 v3 训练验证报告](../out/training_archives/legacy_organize_20261027_122054/adaptive_sft/v3_训练验证报告.md)，不代表采样策略合理。修正版另增 11 项回归，共 84 项通过，覆盖极小来源通道、无重复验证、事实配比、专项 NLL 隔离与旧协议拒绝。采样统计证据见 `out/adaptive_sft/sampling_fix_distribution_20261027.json`。

2026-10-27 修正版已实际训练 3 次更新，并在第 1 次保存后恢复到第 3 次；累计 192 条原样本、205 个窗口、96,221 个监督目标 token，保留率 100%，本次显存峰值约 7.54 GiB。导出权重、分类头和 AdamW 恢复状态独立核对通过，另测试了旧断点拒绝与旧纯权重加载。完整证据见 [采样修正与验证报告](../out/training_archives/legacy_organize_20261027_122054/adaptive_sft/采样修正与验证报告.md)。测试已经停止，完整新训练仍由用户启动；这不构成能力提升结论。

每次更新记录三类样本数、来源数、平均权重、未加权 NLL、分类损失、预测参与程度、原样本/窗口数、目标保留率、梯度、学习率和显存。`metrics.csv/jsonl`、`train.log`、固定抽样和审计均保留。

新入口控制台的 LM 是按原样本及动态权重归一的损失，不能直接与旧入口的全局 token 平均 LM 数值比较；应在相同配置下比较分层未加权 NLL 和实际生成结果。

默认每 1,000 次更新验证固定 128 条分层开发样本、16 条澄清专项样本及 32 道生成题，每 1,000 次更新及结束时保存。主 NLL 在每个任务的可用原行中均匀、不放回抽取，且检查原行不重复；排除 `valid_author_general`，该通道另抽最多 16 条不重复样本，仅记录 `clarification_nll/count`，不混入 general NLL、分类门槛或 best 排序。澄清专项可用 `--clarification_validation_examples 0` 关闭。

新版固定 NLL 探针与旧版不同，不能直接比较两版 NLL 数值；应在同一套探针上重新评估需要比较的权重。作者生成探针选择保持原逻辑，只取原始关系留出问法，排除提供作者资料的 grounded 视图；先纳入 7 道外部来源作者题，再补 9 道诗词题。另有 16 道确定性通道题。所有生成题贪心解码，保存逐题输出、来源层和分母。

v3的后16道为新增确定性通道探针，可能包含计算、资料提取、作者或原文任务，逐题记录真实来源层。新增题面要求指定格式，因此严格保留数组逗号、时间冒号、分数斜线和原文标点评分，防止把不同答案因去标点而错误判成相同。

开放和一般任务的未加权开发 NLL 若超过初始值 0.15，候选不成为 best；通过该保护后优先按事实生成准确率、再按三类平均 NLL 选择。默认不提前停止，避免从预训练初始化时尚未形成回答能力就停止；需要时显式设置 `--early_stopping_patience`。NLL 保护不能保证创作质量保持，最终仍须人工/独立评测赏析与原创作品。

完整断点保存 LM、分类头、AdamW、scaler、随机状态、动态控制器和已提交 update，恢复核对模型实现、AMP/LR/保存工具、完整聊天模板、特殊 token、模型配置和依赖版本。当前协议为 `adaptive_sft_probability_v1`，独立于数据清单的 v3 版本号；采样算法仍为 `task_stratified_size_weighted_v2`。每份 LM 权重另附 `.pth.config.json`，推理自动读取并核对权重 SHA256。Ctrl+C 在当前更新完成后验证、保存。旧格式断点不会静默续接新策略。

旧 `train_adaptive_sft_v2.py`、`train_adaptive_sft_v3_legacy.py` 与对应旧运行已统一归档到 `_archive/organize_20261027_122054/files/`。如需历史复现，应先按归档清单恢复相应完整代码和运行环境，不要直接执行归档目录中的入口，因为其相对项目路径已改变。历史共享模块 `adaptive_sft_data.py` 仍由当前流程使用，保留原位。**新的训练请始终运行 `train_adaptive_sft.py` 或正式启动器**。归档与恢复方法见项目根目录 `PROJECT_GUIDE.md`。

全索引哈希和训练/验证分组已检查，新增66,000条对话的完整监督与题目/来源一致性已核验，回查13,200条文学来源关系，基础库哈希不变。窗口完整性没有提前 tokenize 全部 600 万行；每个实际训练样本都会检查目标数量严格相等。已有预训练/SFT 可能见过知识，不能据此声称全球知识未曝光测试。

v3独立审计：`scripts/validate_more_deterministic_sft.py`；测试：`python -m unittest trainer.test_adaptive_sft_sampling scripts.test_deterministic_tasks trainer.test_adaptive_sft -q`。新增数据已生成，生成脚本拒绝覆盖已有结果。旧数据、旧断点及对应快照均保留。当前默认训练使用 v3 数据与修正版抽样；更换策略必须新建运行，不能把旧优化器游标冒充新策略的进度。`adaptive_v2/` 是更早准备阶段的 superseded 索引。
