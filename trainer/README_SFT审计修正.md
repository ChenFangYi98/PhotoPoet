# SFT 审计修正与运行说明

入口仍是 `trainer/train_adaptive_sft.py`。数据清单仍引用 `dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready/manifest.json`，初始化仍是指定的 `pretrain_poetry_gujilab_768_moe_best.pth`。本次不改 `lm_dataset.py`、原始 JSONL 和已有模型权重。旧代码备份位于 `out/training_archives/legacy_organize_20261027_122054/adaptive_sft/pipeline_fixes_20261027/before/`。

## 已修正的行为

1. `eval_llm.py --checkpoint` 显式指定文件，实际路径和 SHA256 在加载时输出。省略时按 save_dir/weight 构造文件名，不再静默选择旧 SFT。`--prompt_style chat/raw` 与模型类型对应；默认事实测试不采样，创作需 `--do_sample 1`，随机种子固定。
2. 作者评测读取现有 `aliases`；“钱锺书／钱钟书”均可正确计分。作者实体正确与姓名格式遵循分别记录；数组、时间、分数和原文继续保持严格评分，禁止通过删除标点掩盖格式错误。
3. 无诗句、无作品依据却直接断言作者，以及末选项为空却直接作答的保守风险模式被隔离。要求补充材料的澄清回答保留。运行时同通道补样并记录索引，原数据不删除。规则不能替代完整语义审核。
4. 默认 `full_context`：超过 2048 token 的整条对话不进入该次更新，同通道补样。目标保留和证据完整同时满足；代价是长样本覆盖下降。要利用长文，应重建“完整指令＋必要材料＋完整答案”的独立单元，或进行单独的长上下文实验。不要随意切断材料或答案。
5. MoE 负载均衡只统计 attention_mask 内的有效 token。模型层和参数形状不变，可严格加载原权重；此设置改变训练辅助损失，不改变语言模型的有效 token 前向逻辑。
6. 可信 fact 即使被分类头自信错分，也保留至少一半的事实概率增权依据；open/general 权重仍为 1。权重继续停止梯度。它依旧不是事实真伪检测器，也没有证明优于静态权重。
7. 默认每来源 32 道事实生成题，共 64 题。verified 按 `task_meta.kind` 分层取题，预算足够时覆盖全部类型；best 优先比较任务宏正确率，降低单一作者类别的题量优势。开放/通用 NLL 保护参考改为历次最好 NLL，容忍值仍为 0.15。
8. 固定 4 道创作题，每题温度 0.85 生成 2 次，保存输出、种子及重复诊断。采样恢复随机状态，不干扰训练轨迹。字符多样性不等于创作质量，因此不直接参与 best 排名；仍需人工或独立评审。
9. 验证分类按 router_batch_size 分批。完整恢复指纹覆盖模型、训练工具、配置、聊天模板及依赖版本；原数据可用 `--verify_source_sha256` 完整核验。已达停止上限或完成预算时恢复直接退出，不多训练一步、不覆盖原 summary。

`metrics.jsonl` 新增 `quality_rejections`、`task_em`、`fact_macro_em`、`entity_em`、`answer_format_em`、`open_diagnostics`；`validation_*.json` 保存事实回答和创作回答。`run_config.json` 保存筛选策略、初始化筛选统计和实际生成任务组成。`model_config.json` 保存完整配置。

## MoE 门控梯度：当前默认保留原始概率

原 top-1 归一化将选中专家权重变成 1，LM 对门控的梯度近乎为零。按用户后续选择，当前默认 `--norm_topk_prob 0`，直接使用选中专家的全局 softmax 概率乘专家输出，使用该公式的真实梯度。`moe_router_gradient=legacy` 仅表示不使用替代梯度，不再表示开启概率归一化。参数结构不变，但前向尺度改变，新训练在切换后的模型上重新建立验证基线。详见 [概率门控说明](README_概率门控SFT.md)。

保留的旧实验对照需要同时设置 `--norm_topk_prob 1 --moe_router_gradient straight_through`，在训练时使用：

```text
selected_weight = selected_probability / stop_gradient(sum(selected_probability))
```

同一权重下，数值前向仍保持原归一化结果，反向可以得到非零 LM 门控梯度。这是人为指定的替代梯度，不是原函数的真实导数，也不是已验证有效的新训练方法。门控选择仍然离散，可能出现专家切换及训练不稳定；只建议在独立运行中与 legacy 作短程和完整能力对照，不默认开启。推理不需要新增分类头或该训练梯度开关。

## 运行

使用本机已验证 Python 环境的 PowerShell 脚本：

```powershell
cd ./

# 无更新检查；完整核验 8 个源 JSONL 的 SHA256
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -PrepareOnly -VerifySourceSha256

# 新目录中最多 3 次更新，不会启动完整长训
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -SmokeTest

# 用户手动开始新的正式训练
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -VerifySourceSha256

# 可选：独立的实验性门控梯度对照，先只做短测试
powershell -NoProfile -ExecutionPolicy Bypass -File trainer/Start-AdaptiveSFT.ps1 -SmokeTest -NormTopkProb 1 -MoeRouterGradient straight_through
```

新协议拒绝旧完整断点。新运行默认使用继续预训练权重；若显式用旧 SFT 的纯 LM `.pth` 初始化，只是新优化器的热启动，不能消除旧数据产生的偏差。旧环境复现需要对应的全部旧代码，不能只换回一个入口文件。

事实推理示例（将路径替换成实际新运行的 best 文件）：

```powershell
python eval_llm.py --checkpoint out/adaptive_sft/你的新运行目录/adaptive_768_moe_best.pth --prompt_style chat --do_sample 0
```

创作使用同一 checkpoint，加 `--do_sample 1 --seed 42 --temperature 0.85`。对不同模型做能力对比时固定题目、提示格式和生成预算；事实题和多次创作采样分别报告。

## 还需要实验或后续数据工作的部分

- 用同一初始化、seed、配比、预算比较 uniform/static/dynamic。不要用短训练损失证明最终事实能力提升；现在的动态难度仍是批次 EMA，短答案损失仍含格式 token。
- 开放能力不能靠 NLL 或重复率担保，需要独立创作题与人工/可靠评审，并保留不参与反复选模的最终测试集。
- 有放回采样的一轮预算仍不等于原库完整覆盖。后续应比较按来源循环遍历与当前抽样策略，并审查作者多问法重复，不直接通过无限增加 epochs 解决。
- 保守坏题规则只覆盖已发现模式，其他事实错误仍需来源审核；文学元数据一致不等于史料正确。
- 当前 full_context 会隔离长文，不能宣称已解决长上下文能力。数据重构需要按任务保留足够材料，不能自动摘要后默认答案仍可推得。
