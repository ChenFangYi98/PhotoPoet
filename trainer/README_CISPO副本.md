# SFT Best → 副本 CISPO

当前入口为 `trainer/train_cispo.py`，执行设施位于 `trainer/grpo_copy_runtime.py`。
副本原来的 `rep_penalty`、`calculate_rewards` 函数完整保留，默认目标仍为 CISPO，
也可用 `--loss_type grpo`。MoE 辅助损失保留。修改前完整脚本在
`out/grpo_copy/code_before_start_20261102_101328/`。

本轮从 `out/adaptive_sft/adaptive_20261027_131212_562620/adaptive_768_moe_best.pth`
复制独立快照，严格按 `.pth.config.json` 加载 768 维、8 层 MoE 和 `norm_topk_prob=False`。
不使用该 SFT 的 optimizer；SFT 的完整续训断点及代码保持不变。

- 仅使用原通用 `dataset/rlaif.jsonl` 提示，不接入古诗课程、监督回放或 Ollama。
- 奖励沿用副本：长度奖励、think 长度/标签奖励、原 n-gram 惩罚，以及裁剪到 [-3,3] 的本地 InternLM2 评分。
  这些是原公式的行为，不能把奖励增加直接解释为实际回答质量或诗歌能力提高。
- 对末尾连续 assistant、重复提示和乱码作只读预处理，移除参考答案尾部，生成独立训练/验证提示快照和隔离清单；原 JSONL 不覆盖。
- 采样温度为 0.8，top-k=0、top-p=1；新策略、旧策略、参考策略 logprob 使用一致温度。
- 首轮 100 次 optimizer 更新，batch_size=1、每题 6 个候选、梯度累计 4 次，生成上限 256 token。
- 每次更新记录奖励、KL、梯度、长度、复读率、截断率、旧/新 logprob 误差和采样回答。
  每 20 步使用同一组 16 条留出提示进行贪心验证；按平均验证奖励保存 best。
- 第 1 步、每 20 步及结束时保存 FP32 导出、绑定配置和完整 `resume.pth`。
  完整断点包含 FP32 模型、AdamW、scheduler、scaler、随机状态、更新/批次位置、来源及代码/数据签名。

```powershell
# 打开黑窗，新运行仅执行一次
out/session_control/START_GRPO_COPY.cmd
# 从当前运行的完整断点继续
out/session_control/RESUME_GRPO_COPY.cmd
# 精确的原生入口（新运行自动建独立目录）
python -X utf8 "trainer/train_cispo.py" --checkpoint out/adaptive_sft/adaptive_20261027_131212_562620 --loss_type cispo
# 验证算法梯度、采样一致性、数据隔离和精确续训
python -X utf8 -m unittest trainer.test_grpo_copy -q
```

当前输出与参数见 `out/session_control/copy_pipeline_config.json`，状态见 `copy_session.json`。
`grpo_session.json` 也指向当前副本运行。你的 `eval_llm.py` 当前默认 checkpoint 固定为旧模型，
保留该设置不修改；评估本轮模型请显式传入 `--checkpoint out/grpo_copy/<运行名>/grpo_copy_best_768_moe.pth`。
在运行目录创建 `STOP` 空文件可在安全更新边界保存并停止；恢复时使用上述 resume 入口。
本地可检查的 Torch 单进程实现拒绝不匹配的模型配置、无效参数和不一致的断点签名。
