# MindChen

MindChen 由 chenfangyi 训练和维护，提供语言模型以及在该语言模型上增加视觉编码器、投影层的视觉语言模型。两个能力共用同一份语言主干、分词器、LoRA 和训练工具。

```text
model/model_mindchen.py       共享语言主干
model/model_vlm.py            SigLIP 编码器、投影层，继承语言模型
model/lora_adapter.py         共享 LoRA
trainer/                     语言、图文训练入口及公共实现
dataset/                     文本和图文读取、PhotoPoet 制作工具
scripts/                     推理工具、可视化和数据处理
eval_llm.py / eval_vlm.py     两类评测入口
```

## 安装

建议使用 Python 3.11，先按设备安装 PyTorch，再安装核心依赖：

```powershell
python -m pip install -r requirements-core.txt
```

其他训练、数据制作和网页功能按 `requirements.txt` 安装扩展依赖。权重和训练数据分别位于发行目录的 `02_weights/`、`03_data/`，不放入源码压缩包。

## 加载和训练

将选定的基础权重及同名 `.pth.config.json` 复制到 `out/`。视觉编码器放到 `model/siglip2-base-p32-256-ve/`，包括配置、预处理配置及编码器权重。

```powershell
python eval_llm.py --checkpoint out/llm_base.pth
python eval_vlm.py --checkpoint out/vlm_base.pth --image_dir path/to/test_images
python scripts/web_demo_vlm.py --checkpoint out/vlm_base.pth
python trainer/train_lora_sft.py --checkpoint out/llm_base.pth --data_path dataset/train.jsonl --max_steps 100 --export_merged
python trainer/train_lora_vlm.py --checkpoint out/vlm_base.pth --data_path dataset/vlm_train.parquet --max_steps 100 --export_merged
python trainer/train_sft_vlm.py --from_weight out/vlm_base.pth --data_path dataset/sft_i2t.parquet --save_weight next_sft --use_moe 1
```

语言数据使用独立数据包中的 `llm/local-training-review/`，图文数据使用 `vlm/local-training-review/`。按原层级复制至同一个 `dataset/`。PhotoPoet 分片保持分开，不要与已混合的 SFT 数据再次全部拼接。迁移自适应 SFT 清单使用：

```powershell
python scripts/relocate_data_manifest.py --manifest dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready/manifest.json --data-root dataset --output out/data_manifest.local.json --verify_sha256
```

训练时将生成的本机清单传给 `--manifest`。完整内容核验会读取所有源文件。

`--mindchen_identity` 在手动对话时为直接身份问题使用项目署名；评测模型自然生成时省略该选项。身份 LoRA 仍是实验候选，存在普通任务退步，详见权重目录。

原生历史权重和适配器继续按文件哈希核验；旧完整训练断点如受原代码签名约束，应在原工程恢复。模型计算层和参数键不因本次命名整理而改变。新手说明见 `docs/`、`新手技术文档.md`，LoRA 用法见 `LORA_GUIDE.md`。

代码参考项目及许可见 [第三方说明](THIRD_PARTY_NOTICES.md)。
