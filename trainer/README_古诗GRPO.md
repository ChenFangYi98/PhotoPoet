# 通用问答 + 古诗 GRPO

## 格式课程与监督回放（v5）

上一轮从 SFT best 直接 GRPO，100 步固定验证的古诗格式通过率仍为 0。
新流程仍从暂停后的同一 SFT best 复制，先进行 400 次格式/EOS 训练，再按门槛进入 GRPO。
四种字数/句数组合在每次预热更新中均衡采样，与通用问答各占一半任务权重。
格式目标只来自真实原文的分句和标点恢复，不把旧诗标为原创。

- 留出格式探针从 4 条增至 32 条，四种组合各 8 条；另检查最多 128 条通用回答 NLL。
- 自动进入 GRPO 要求留出格式通过率至少 50%、原文完整恢复率至少 25%、通用 NLL 相对起点增加不超过 0.1。未通过会保存完整状态并停止，不能视为训练达标。
- 预热支持 `--resume`，保留参数、AdamW、随机状态、更新位置、代码/数据签名和只读来源快照。
- GRPO 使用预热 best，学习率降至 3e-7，保持复读奖励惩罚和 unlikelihood，并加入权重 0.1 的监督回放，帮助保留格式和通用回答能力。
- 回放仅读取预热 `train.jsonl`，逐次均衡抽取一条格式和一条通用样本；拒绝 `split=valid`、截断和缺失 EOS。数据哈希进入 GRPO 完整断点，抽样可精确恢复。
- GRPO 固定验证增至每通道 32 条；100 次更新，每 10 次验证并保存。格式恢复改善不等于原创诗歌质量改善，最终仍以独立 GRPO 验证为准。
- 新协议为 `mindchen_poetry_grpo_v5_replay`；历史运行保持原目录，不混用旧优化器。当前启动配置位于 `out/session_control/fixed_pipeline_config.json`。

```powershell
out/session_control/START_GRPO_FIXED.cmd
# 预热或 GRPO 暂停后，按当前阶段恢复
out/session_control/RESUME_GRPO.cmd
python -X utf8 -m unittest trainer.test_grpo_curriculum trainer.test_grpo_poetry -q
```

## 评分器复读导致的 HTTP 500（当前修复）

已从第 60 步原样复现第 61 步的四个候选回答：第二个候选会让 Qwen 评分器生成评语时循环，
Ollama 的错误正文是 `prediction aborted, token repeat limit reached`。
独立端口并未消除这个确定性错误，之前将其归因于服务重载的判断不完整。

评分请求现使用 JSON Schema，限定 coherence/relevance 为整数 0..4，reason 为有界字符串；
评分提示、模型 digest、温度和上下文保持不变。错误正文会写入日志，临时网络故障在请求内有界重试；
明确的 token-repeat 错误不会再触发整轮训练反复重启，也不会被替换成虚构分数。
执行协议提升为 `poetry_coherence_relevance_v2_structured`。

第 60 步模型、AdamW、调度器、随机状态和批次位置通过一次显式迁移保留到
`out/grpo/judge_recovery/` 的独立运行，原运行和原代码快照完整保留。
迁移只允许评分器这一个代码文件发生已确认的变更，并写 `migration.json` 记录原签名、
新签名、旧断点哈希与评分器协议变化；普通续训的签名检查仍保持严格。

```powershell
python -X utf8 -m unittest trainer.test_grpo_judge_transport trainer.test_grpo_recovery -q
out/session_control/RESUME_GRPO.cmd
```

## 评分服务临时故障后的恢复

2026-10-31，第 60 步保存完成后，本地 Ollama 在准备第 61 步时返回 HTTP 500。
已校验第 60 步完整断点；训练代码、数据、奖励规则和代码签名保持一致，直接续训。
`out/session_control/run_fixed_pipeline.py` 对评分服务的 HTTP 429/500/502/503/504、连接中断和超时，
最多从最近完整断点自动重启三次，等待 5、15、45 秒。未保存的更新会重放，日志追加保留。
没有完整断点、存在 STOP 文件、评分内容无效或代码签名不匹配时不会按此机制重试。
本机评分服务已隔离到 `http://127.0.0.1:11435`：原共享服务会因不同上下文请求反复重载模型，
恢复时复现了 HTTP 500。专用端口仍读取同一个 Qwen 模型，模型 digest 和全部评分参数保持一致。
启动器在专用服务未运行时自动启动隐藏服务；服务日志保存在 `out/session_control/dedicated_judge/`。

```powershell
out/session_control/START_GRPO_FIXED.cmd --resume
python -X utf8 -m unittest trainer.test_grpo_recovery -q
```

## 2026-10-31 代码审查修复（v4，当前版本）

- Windows 共享占用导致的原子替换失败会有界重试，临时文件使用独立名字；实时状态文件持续不可写时告警并继续训练，模型和完整断点无法保存时仍明确报错。启动器、权重绑定配置使用同一套保存逻辑。
- 通用评分取消 think 长度/标签正奖励，评分模型读取完整输出，不能把垃圾前缀藏在 `</think>` 前。非空思考块扣 1.5 分，空块扣 0.25 分，不匹配标签扣 2 分；长度和原有 n-gram 规则、通用评分裁剪 [-3, 3] 保留。
- best 必须同时通过通用评分、复读率、古诗合格率保护。额外比较初始模型及已接受 best 的综合分、古诗奖励、语义质量、空回答率和截断率，防止靠输出变空或内容变差降低复读率。合格率不再允许默认下降一个样本。
- 重复 token 的 unlikelihood 使用数值稳定的平滑公式，概率接近 1 时仍有有限非零梯度。明确要求某个短语重复指定次数的任务，只有输出的内容、次数和指定逐行格式全部符合时才豁免；错误次数、附加内容和泛泛提到“重复”不能豁免。
- `eval_llm.py` 默认读取当前 GRPO 会话的 best，尚未产生 best 时提示缺失；显式 `--checkpoint` 优先，显式 `--save_dir`/`--weight` 可正常选择其它模型。默认关闭 thinking、生成上限 256、温度 0.8、top-p=1；权重保留 FP32，CUDA 支持时使用 BF16 autocast，与训练验证保持一致。
- 使用新协议 `mindchen_poetry_grpo_v4_reviewed`，旧版本产物与修改前代码备份保留；新运行仍从暂停后确定的 SFT best 复制，不混用旧奖励下的优化器。以下 v3/v2 段落是历史记录，与本节冲突时以 v4 为准。

```powershell
python -X utf8 -m unittest trainer.test_grpo_review_fixes trainer.test_grpo_repetition trainer.test_grpo_poetry trainer.test_adaptive_sft_safety trainer.test_probability_gate -q
python -X utf8 eval_llm.py
```

## 2026-10-31 复读修正（v3）

- SFT 已按用户要求正常暂停于第 36,014 步，`latest_resume.pth` 保留模型、router、AdamW、随机状态和控制器，SFT 的 13 个依赖代码签名保持不变。
- 通用通道保留旧奖励作为基础，另外统一检测中文按字、英文按词的循环重复、反复堆叠短语和重复长句。严重复读额外扣分，不能由语义评分补成正奖励；未生成 EOS 的通用回答另扣 1 分。
- 对确认复读候选的完成部分增加 `-log(1-p(token))` 训练项，默认权重 0.2。只惩罚第三次及后续循环或反复出现的 n-gram；提示词、EOS 后 padding、正常两次措辞和正确保留参考原文的格式任务不在此训练项中。这样即使一组候选全都复读、GRPO advantage 为零，仍有纠正梯度。方法背景：[Unlikelihood Training](https://arxiv.org/abs/1908.04319)。
- 训练和验证记录 `repetition_rate`、`repeated_fraction`、distinct-1/2/3；训练另记 `unlikelihood_loss` 和 `repeated_token_fraction`。验证按创作、改写、模板、格式恢复分别记录结果，避免把格式修复改善当作整体创作提升。
- best 在通用评分、两通道复读率和古诗合格率保护通过后，优先选择复读更少的模型，再比较合格率和奖励。贪心验证不使用 repetition penalty 或 n-gram 禁止规则，便于观察模型本身的行为。
- 新协议为 `mindchen_poetry_grpo_v3_repetition`。旧 v2 的模型、完整断点和代码快照仍保留。按用户最新要求，当前入口直接从 `out/adaptive_sft/adaptive_20261027_131212_562620/adaptive_768_moe_best.pth` 复制独立快照开始，重建优化器，不加载旧 GRPO 的 optimizer，也不先经过格式预热。SFT 暂停于 36,014 步，其验证选出的 best 仍为 32,000 步。启动器检查 SFT 已完成保存、来源确为此 best，训练器检查绑定配置和快照 SHA-256。修改前代码另存 `out/grpo/code_before_repetition_20261031/`。
- 本轮入口为 `out/session_control/START_GRPO_FIXED.cmd`；先验证新基线，再训练 100 次更新。第 1 次更新即保存完整断点，之后每 20 次保存；每 10 次验证，日志逐次写入运行目录和控制台日志。

```powershell
python -X utf8 -m unittest trainer.test_grpo_repetition trainer.test_grpo_poetry -q
```

## 2026-10-30 修复版

- 通用 RLAIF 保持原奖励公式；空末尾 user、Unicode 替换乱码及控制字符提示在加载时隔离，原 JSONL 不覆盖。每次运行写 `data_quality.json`。
- 古诗内容改由本机 Ollama `qwen3.5:latest` 评审连贯性和题意相关性。启动必须通过正常诗作/已发现坏样例校准，记录模型 digest；服务异常或校准不通过会停止，绝不静默退回旧内容评分。
- 古诗的正向格式分按内容质量门控；不合格式不能领取正向内容分。缺句、错字数、换行及标点违规额外扣分；纯复制不分句不能领取格式修复加分。保留分项奖励，不将所有不合格回答粗暴归零。
- 默认生成上限从160改为256，采样温度0.8，采样和所有 logprob 使用相同温度。未生成EOS的古诗另扣1.5分。EOS后出现padding仍算正常结束。
- best 优先比较“格式正确、语义相关且连贯、正常结束”的验证通过率，同时保留通用问答退化保护。评分提高不等同于文学能力已经提高。
- `runtime.json` 实时记录PID、训练/验证阶段与更新步数；周期保存的 `summary.json` 为 `running`，真正结束才写 `stopped` 或 `completed`。逐条验证、逐批采样和语义评分均输出进度。
- 在运行目录创建名为 `STOP` 的空文件可正常停止：完成正在累积的优化器更新后保存完整断点；验证中则在当前样例结束后保存。Ctrl+C也支持相同流程，停止文件处理后会改名留档。
- 这是新断点版本，旧版 GRPO 运行保留原目录和代码备份，不混用旧奖励下的优化器。SFT封存与续训文件不修改。
- 本机模型、完整训练断点和日志统一保存在 `./out/`：GRPO 使用 `out/grpo/`，格式预热使用 `out/format_warmup/`，SFT封存使用 `out/training_archives/`。`--poetry_judge_url` 默认 `http://localhost:11434`，只允许本机地址。
- 本次重启先用 `warmup_grpo_format.py` 在独立副本上执行200次格式/EOS预热：监督目标来自真实诗作的分句恢复，不把旧诗标成原创；与原SFT通用样本各占一半任务权重。按独立通用NLL退化上限和固定留出格式探针选择best，未形成任何格式成功时禁止自动进入GRPO。4个预热探针只用于启动检查，不能代表整体诗词能力；正式GRPO仍用自己的独立验证集。

```powershell
# 正常停止修复版运行；把路径替换为实际运行目录
New-Item -ItemType File -Path './out/grpo/<运行名>/STOP'
```

本入口从 `out/adaptive_sft/adaptive_20261027_131212_562620/adaptive_768_moe_best.pth` 启动。
目录参数固定选择这个 best 文件，不自动寻找或切换到其它权重。

## 保留 SFT 的可续训状态

- SFT 是否运行按用户最新指令执行；此次已正常暂停。GRPO 不修改其代码、优化器断点、模型文件或配置。
- 新运行保存到 `out/grpo/<运行名>/`。输出路径与 SFT 来源、`model/`、`dataset/` 相交时拒绝启动。
- Windows 先在 GRPO 目录创建临时只读硬链接别名，再将内容复制成独立快照。这样读取期间原训练仍可原子替换源文件；临时别名用完删除，实际快照不是硬链接。
- 同时校验源文件绑定配置和快照 SHA256。复制遇到 SFT 正在更新时重试；不猜测 MoE 路由。
- 策略和冻结参考模型从同一快照严格加载，保留 `norm_topk_prob=False`。
- 后续 SFT 的新更新不会自动改变这个 GRPO 运行的参考模型。要使用更新后的 SFT，另起 GRPO 运行。
- 推理导出和完整断点都保留训练参数精度，避免小学习率更新被 fp16 提前舍入抹掉。

## 数据与验证

原 `dataset/rlaif.jsonl` 保持不变，共 19,502 条。
加载时处理 4 条末尾连续 assistant 的记录；移除末尾答案，避免把目标答案泄漏给策略。
按实际 prompt 去除 98 条重复后，18,446 条用于训练，958 条按固定哈希分组留作验证。
16 条长普通对话在内存中保留最近轮次；若单条问题仍过长则保留后部，并记录裁剪标志。
古诗提示不允许截断格式约束，超长直接报错。

新增文件位于 `dataset/grpo_poetry_v2/`：

| 数据 | 条数 | 内容 |
| --- | ---: | --- |
| train.jsonl | 6,720 | 720 条题材/风格/藏头提示 + 2,000 首本地诗作各衍生 3 种任务 |
| valid.jsonl | 744 | 144 条保留题材提示 + 200 首不同诗作衍生任务 |
| manifest.json | — | 来源、切分、文件哈希和局限 |

衍生任务为题意原创、保留意境的改写、保持原字序的分行标点修复。
格式修复有可验证的原文字序；原创/改写没有伪造的“标准答案”或人工评分。
按题名分组划分语料，并检查训练/验证正文不交叉。
这些来源可能已用于 SFT，因此这里的验证隔离仅针对本次 RL，不能宣称是基座从未见过的评测。

默认 `--poetry_ratio 0.5`：一半采样来自原通用 RLAIF，一半来自新增古诗。
比例可调且必须介于 0 和 1。完整 epoch 覆盖两类全部训练提示，样本少的一类循环采样。
默认先试跑 100 次 optimizer 更新；`--max_steps 0` 按 `--epochs` 完整训练。

RL 算法能没有验证集地更新，但正式训练建议保留验证集，识别奖励投机、过拟合和通用能力下降。
脚本在更新前、每 20 次更新、停止时验证：普通问答内容评分、古诗格式合格率、内容评分和生成截断率分开报告。
默认每类固定抽取 16 个验证提示，古诗轮转覆盖不同体裁和任务。需要更可靠的判断时增加 `--validation_per_channel`。
验证不执行反向传播，也不用于计算训练 advantage。

`best` 按两类验证综合分选择，并要求通用有界评分相对基线下降不超过 0.15。
如果短期没有提升，best 可能仍是初始 SFT 权重，`summary.json` 中的 `best_update` 会说明。
同一个内容评分模型也用于验证，所以分数不能替代人工审阅诗意、连贯性和事实正确性。

## 奖励

- 句数、每句汉字数、一句一行、中文逗号/句号、正文整洁程度。
- 藏头字、弱题材覆盖、弱押韵奖励。
- 中文按字统计 n-gram 重复，重复整句和单字刷屏扣分。
- 原创/改写大量复制参考诗作扣分；格式修复任务则检查原文字序是否保留。
- 普通回答不强制套用古诗格式。对“解释五绝是什么”这类问题也不强求输出诗歌。
- 通用内容评分使用本地 [InternLM2-1.8B-Reward](https://huggingface.co/internlm/internlm2-1_8b-reward)；古诗语义评分使用上述经校准的本地 Qwen 评审。
- 押韵仅为普通话韵母近似，不是平水韵、平仄或对仗校验。代码不声称自动分数等于文学质量。

2026-10-30 调整：原 RLAIF 通道恢复原脚本的奖励规则，不再应用上一版新的 think 惩罚和 tanh 内容分数。
普通回答保留原长度奖励（20..800 字符为 +0.5，否则 -0.5）、原 think 长度及标签奖励、原 n-gram 惩罚，
以及裁剪到 [-3, 3] 的评分模型分数；评分时按原方式提取 think 后的回答并折叠对话历史。
即使原 RLAIF 某条偶然包含写诗问题，也继续使用原评分，新增古诗通道才使用专门规则。

## 怎样确定每首诗限制几个字

每条新增古诗有独立的 `poetry_spec`，不是整个数据集统一限制成 5 字或 7 字：

| 体裁 | lines：句数 | width：每句汉字数 | 正文汉字总数 |
| --- | ---: | ---: | ---: |
| 五言绝句 / 五绝 | 4 | 5 | 20 |
| 七言绝句 / 七绝 | 4 | 7 | 28 |
| 五言律诗 / 五律 | 8 | 5 | 40 |
| 七言律诗 / 七律 | 8 | 7 | 56 |

例如一条七律数据：

```json
{
  "messages": [{"role": "user", "content": "请写一首七言律诗，每句7个汉字，共8句，一句一行。"}],
  "poetry_spec": {"form": "七言律诗", "lines": 8, "width": 7, "linebreaks": true}
}
```

模型输入来自 `messages`；`poetry_spec` 供评分器使用，不额外塞进模型输入。
生成模板时同时创建用户要求和这份标注；本地诗作衍生任务还检查原作句数、每句汉字数再创建对应要求。
这种结构识别不等于已经证明原作满足律诗的平仄、对仗等完整格律。

如果自定义古诗数据没有标注，加载器识别“我要写一个七言律诗”“帮我写首五律”，
以及“共四句，每句七个汉字”等明确要求。缺少句数/字数且无法从体裁确定时直接报错，要求补齐标注。
要求七律但标注 width=5，或同时写“七言律诗，每句5字”，也会报冲突，避免按错误标准训练。

评分先按标点和换行分句，再统计每句汉字数，标点和换行不计入字数。
字数项 `line_width = 1.5 × 字数合格句数 / max(要求句数, 实际句数)`：
七律 8 句全为 7 字得到 1.5；其中一句为 6 字或 8 字时，该项变为 `1.5 × 7/8 = 1.3125`。
句数、换行、标点等另行评分。字数不合格也可能有其它奖励，并不是直接丢弃候选。

这是通过奖励学习遵守要求，推理时仍可能出现格式错误。
`max_gen_len` / `max_new_tokens` 是 token 上限，不能把它设成 5 或 7 来代替每句汉字数限制。

奖励不直接作为可微 loss。相同 prompt 生成多个候选，先把奖励组内标准化为 advantage，
再按 GRPO 的概率比裁剪目标或 CISPO 目标更新策略，并用参考模型 KL 项约束偏移。
字数检查本身不反向传播；它通过候选之间的奖励差异影响策略梯度。

本次奖励语义已改变，需要新建 GRPO 运行；旧版运行的代码签名不匹配时会拒绝续训。
SFT 的权重和断点不受此次 GRPO 奖励代码调整影响。

评分模型固定来源版本 `25f3593492ab4625ce00fce8c5e67802d6e702ca`，下载至 `model/internlm2-1_8b-reward/`。
缺失资源自动下载、网络中断重试、校验 safetensors 文件完整性；不静默改成规则单独评分。
评分模型和参考模型默认在 CPU 上，策略默认使用 CUDA（可显式 `--device cpu`）。
目前 SFT 正在占用 GPU，工程短测默认全部使用 CPU；正式并行训练仍会争用计算资源。

## 启动

推荐 PowerShell 启动器，优先选择已有 `anaconda3/envs/mindchen/python.exe`：

```powershell
# 仅下载/校验资源、生成数据、复制快照；不作训练更新
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-GRPO.ps1 -PrepareOnly

# 真正完成一次生成、评分、反向、验证和保存；默认 CPU
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-GRPO.ps1 -SmokeTest

# 正式运行：默认 100 次更新，通用/古诗各一半；此命令会启动新训练
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-GRPO.ps1 -MaxSteps 100 -PoetryRatio 0.5

# 恢复自己的 GRPO 运行（使用原固定快照，不重新读取变化中的 SFT）
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-GRPO.ps1 -Resume out/grpo/<运行名>/resume.pth
```

已激活 mindchen 环境时也可直接：

```powershell
python -X utf8 trainer/train_grpo.py --prepare_only
python -X utf8 trainer/train_grpo.py --smoke_test
python -X utf8 trainer/train_grpo.py --max_steps 100 --poetry_ratio 0.5
python -X utf8 trainer/train_grpo.py --eval_only --device cpu --validation_per_channel 32
python -X utf8 trainer/train_grpo.py --resume out/grpo/<运行名>/resume.pth --device cpu
```

旧版本的共享 `../checkpoints`、`--from_weight` 和自动猜测断点已替换为显式 `--checkpoint` / `--resume`。
本入口为单进程 PyTorch，不连接未经核实权重的 SGLang 服务，不支持 torchrun 多进程。

新环境只补充 `python -m pip install -r trainer/requirements_grpo.txt`；不需要升级正在跑 SFT 的 torch/transformers。

## 输出与测试

每个运行包括 `source_snapshot/`、`source.json`、`run_config.json`、
`grpo_poetry_768_moe.pth` 及绑定配置、`grpo_poetry_best_768_moe.pth` 及绑定配置、
`resume.pth`、`metrics.jsonl`、`validation_metrics.jsonl`、验证样例和最新训练候选。
`resume.pth` 保存 FP32 参数、AdamW、学习率调度、随机状态和批次位置；只能在相同代码/数据下恢复。
所有模型输出使用临时文件 + 原子替换；不写原 SFT 的任何检查点。

```powershell
python -X utf8 -m unittest trainer.test_grpo_poetry -v
python -X utf8 eval_llm.py --checkpoint out/grpo/<运行名>/grpo_poetry_768_moe.pth --max_new_tokens 256
```

工程短测仅证明流程可运行，少数更新和少量验证样例不能证明诗词能力已有提升。
