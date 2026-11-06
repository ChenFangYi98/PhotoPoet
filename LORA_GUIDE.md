# MindChen LoRA

语言和视觉共用 `trainer/lora_training.py` 与 `model/lora_adapter.py`。

```powershell
python trainer/train_lora_sft.py --checkpoint out/llm_base.pth --data_path dataset/train.jsonl --max_steps 100 --export_merged
python trainer/train_lora_vlm.py --checkpoint out/vlm_base.pth --data_path dataset/vlm_train.parquet --max_steps 100 --export_merged
```

两个入口共用训练参数，但分别选择语言和图文模式。视觉模式需要本地视觉编码器。每次训练创建独立的 `out/lora_runs/` 子目录；完整断点绑定基础权重、数据和代码签名。迁移了源码的历史完整断点应使用原工程恢复，合并权重和适配器可通过新工程加载后开始新训练。

身份署名为：我是 MindChen，由本项目的开发者chenfangyi训练和维护。模型自然输出需单独评估；演示界面的身份选项不能代替权重评测。
