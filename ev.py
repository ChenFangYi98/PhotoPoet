# Modified by chenfangyi for the unified MindChen release (2026-11-06).

import os
import queue
import threading
import warnings
from pathlib import Path

import torch
import gradio as gr

from PIL import Image, ImageOps
from transformers import TextIteratorStreamer

from model.model_vlm import MindChenVLM

warnings.filterwarnings("ignore")

# ============================================================
# 1. 路径和模型配置
# ============================================================

ROOT = Path(__file__).resolve().parent

# 原始 MindChen-V 完整模型
BASE_CHECKPOINT = (
    ROOT / "out" / "sft_vlm_768.pth"
)

# LoRA 训练产物
ARTIFACT_DIR = (
    ROOT / "out" /
    "lora_identity_runs" /
    "20261106_115950_413544_chenfangyi" /
    "artifacts"
)

# LoRA Adapter
ADAPTER_CHECKPOINT = (
    ARTIFACT_DIR / "best_loss_adapter.pth"
)

# 如果有已经合并的权重，可以填写具体路径
# 没有则自动搜索 merged_step_*.pth
MERGED_CHECKPOINT = None

# SigLIP2
VISION_MODEL = (
    ROOT / "model" /
    "siglip2-base-p32-256-ve"
)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# 全局模型
MODEL = None
TOKENIZER = None
PREPROCESSOR = None


# ============================================================
# 2. 加载模型
# ============================================================

def load_model():

    global MODEL, TOKENIZER, PREPROCESSOR

    from trainer.lora_training import load_native

    print("\n" + "=" * 60)
    print("MindChen-V Studio")
    print("正在初始化模型...")
    print("=" * 60)

    if not VISION_MODEL.is_dir():
        raise FileNotFoundError(
            f"找不到 SigLIP2 模型：{VISION_MODEL}"
        )

    model = None
    tokenizer = None
    merged_candidates = []

    # --------------------------------------------------------
    # 查找合并权重
    # --------------------------------------------------------

    if MERGED_CHECKPOINT is not None:

        merged_path = Path(MERGED_CHECKPOINT)

        if not merged_path.is_absolute():
            merged_path = ROOT / merged_path

        if merged_path.is_file():
            merged_candidates.append(merged_path)

    if ARTIFACT_DIR.is_dir():

        candidates = sorted(
            ARTIFACT_DIR.glob("merged_step_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for path in candidates:

            if path not in merged_candidates:
                merged_candidates.append(path)

    # --------------------------------------------------------
    # 优先加载完整合并模型
    # --------------------------------------------------------

    for merged_path in merged_candidates:

        print(f"\n尝试加载合并模型：{merged_path}")

        try:

            model, tokenizer, _ = load_native(
                merged_path,
                "vlm",
                "cpu",
                str(VISION_MODEL)
            )

            print("合并模型加载成功！")
            break

        except Exception as e:

            print(f"合并模型加载失败：{e}")
            print("尝试基础模型 + LoRA 加载方式")

            model = None
            tokenizer = None

    # --------------------------------------------------------
    # 如果没有合并模型，加载基础模型 + LoRA
    # --------------------------------------------------------

    if model is None:

        if not BASE_CHECKPOINT.is_file():

            raise FileNotFoundError(
                "\n找不到基础模型！\n"
                f"当前配置：{BASE_CHECKPOINT}\n\n"
                "请将 BASE_CHECKPOINT 修改为训练 LoRA 时"
                "使用的完整 MindChen-V 模型路径。"
            )

        print("\n加载基础 MindChen-V 模型：")
        print(BASE_CHECKPOINT)

        model, tokenizer, _ = load_native(
            BASE_CHECKPOINT,
            "vlm",
            "cpu",
            str(VISION_MODEL)
        )

        # 加载 LoRA
        if ADAPTER_CHECKPOINT.is_file():

            print("\n正在加载 LoRA Adapter：")
            print(ADAPTER_CHECKPOINT)

            from model.lora_adapter import load_bound_adapter

            load_bound_adapter(
                model,
                ADAPTER_CHECKPOINT,
                BASE_CHECKPOINT
            )

            print("LoRA 加载成功！")

        else:

            raise FileNotFoundError(
                "\n找不到 LoRA Adapter：\n"
                f"{ADAPTER_CHECKPOINT}\n"
                "请检查训练产物目录。"
            )

    # --------------------------------------------------------
    # 模型初始化
    # --------------------------------------------------------

    model.eval()

    if DEVICE == "cuda":
        model = model.half()

    model = model.to(DEVICE)

    MODEL = model
    TOKENIZER = tokenizer
    PREPROCESSOR = model.processor

    print("\n" + "=" * 60)
    print("模型加载完成！")
    print(f"运行设备：{DEVICE}")
    print(f"视觉模型：{VISION_MODEL}")
    print("=" * 60)

    return model, tokenizer, PREPROCESSOR


# ============================================================
# 3. 图片预处理
# ============================================================

def process_image(image):

    if image is None:
        raise ValueError("请先上传一张图片")

    image = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    pixel_values = MindChenVLM.image2tensor(
        image,
        PREPROCESSOR
    )

    pixel_values = {
        key: value.to(DEVICE)
        for key, value in pixel_values.items()
    }

    return pixel_values


# ============================================================
# 4. 生成模型输入
# ============================================================

def build_inputs(messages):

    inputs_text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False
    )

    inputs = TOKENIZER(
        inputs_text,
        return_tensors="pt",
        truncation=True
    ).to(DEVICE)

    return inputs


# ============================================================
# 5. 多轮图片对话
# ============================================================

def chat(
    user_text,
    image,
    display_history,
    model_history,
    max_new_tokens,
    temperature,
    top_p
):

    global MODEL, TOKENIZER

    display_history = list(display_history or [])
    model_history = list(model_history or [])

    user_text = (user_text or "").strip()

    if not user_text:

        yield (
            display_history,
            model_history,
            "",
            "请输入问题。"
        )
        return

    if image is None:

        yield (
            display_history,
            model_history,
            "",
            "请先拖入一张图片。"
        )
        return

    # --------------------------------------------------------
    # 图片处理
    # --------------------------------------------------------

    try:

        pixel_values = process_image(image)

    except Exception as e:

        yield (
            display_history,
            model_history,
            "",
            f"图片处理失败：{e}"
        )
        return

    # --------------------------------------------------------
    # 构造用户消息
    # --------------------------------------------------------

    if len(model_history) == 0:

        # 第一轮对话，将图片 Token 放入消息
        image_token = (
            MODEL.config.image_special_token *
            MODEL.config.image_token_len
        )

        content = image_token + "\n" + user_text

    else:

        # 后续对话保留第一轮图片 Token
        content = user_text

    user_message = {
        "role": "user",
        "content": content
    }

    current_model_history = (
        model_history + [user_message]
    )

    current_display = (
        display_history +
        [{"role": "user", "content": user_text}]
    )

    # 显示用户问题
    yield (
        current_display,
        model_history,
        "",
        "正在分析图片并生成回答..."
    )

    # --------------------------------------------------------
    # 构造完整对话
    # --------------------------------------------------------

    try:

        inputs = build_inputs(
            current_model_history
        )

    except Exception as e:

        yield (
            current_display + [
                {
                    "role": "assistant",
                    "content": f"输入构造失败：{e}"
                }
            ],
            model_history,
            "",
            "输入构造失败"
        )
        return

    # --------------------------------------------------------
    # 流式生成
    # --------------------------------------------------------

    streamer = TextIteratorStreamer(
        TOKENIZER,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=300
    )

    errors = queue.Queue()

    generation_kwargs = {

        "inputs": inputs["input_ids"],

        "attention_mask": inputs["attention_mask"],

        "max_new_tokens": int(max_new_tokens),

        "do_sample": float(temperature) > 0,

        "streamer": streamer,

        "pad_token_id": (
            TOKENIZER.pad_token_id
            if TOKENIZER.pad_token_id is not None
            else TOKENIZER.eos_token_id
        ),

        "eos_token_id": TOKENIZER.eos_token_id,

        "pixel_values": pixel_values
    }

    if float(temperature) > 0:

        generation_kwargs["temperature"] = float(
            temperature
        )

        generation_kwargs["top_p"] = float(top_p)

    # --------------------------------------------------------
    # 在线程中运行 generate
    # --------------------------------------------------------

    def generate_worker():

        try:

            with torch.inference_mode():

                MODEL.generate(
                    **generation_kwargs
                )

        except Exception as e:

            errors.put(e)

            try:
                streamer.end()
            except Exception:
                pass

    worker = threading.Thread(
        target=generate_worker,
        daemon=True
    )

    worker.start()

    answer = ""

    # --------------------------------------------------------
    # 实时接收 Token
    # --------------------------------------------------------

    try:

        for new_text in streamer:

            answer += new_text

            temporary_history = (
                current_display +
                [
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            )

            yield (
                temporary_history,
                model_history,
                "",
                "模型正在生成..."
            )

        worker.join()

        if not errors.empty():

            raise errors.get()

        # ----------------------------------------------------
        # 保存完整对话
        # ----------------------------------------------------

        if not answer.strip():
            answer = "模型没有生成有效文本。"

        new_model_history = (
            current_model_history +
            [
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        )

        final_history = (
            current_display +
            [
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        )

        yield (
            final_history,
            new_model_history,
            "",
            "回答完成"
        )

    except Exception as e:

        if torch.cuda.is_available():

            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()

        error_text = (
            f"推理失败：{type(e).__name__}: {e}"
        )

        yield (
            current_display +
            [
                {
                    "role": "assistant",
                    "content": error_text
                }
            ],
            model_history,
            "",
            "推理异常"
        )


# ============================================================
# 6. 清空对话
# ============================================================

def reset_chat():

    return (
        [],
        [],
        "对话已重置，请重新提问。"
    )


def clear_all():

    return (
        None,
        [],
        [],
        "已清空图片和对话。"
    )


# ============================================================
# 7. 页面样式
# ============================================================

CSS = """
.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    margin-bottom: 4px;
}

#main-subtitle {
    text-align: center;
    color: #8995A8;
    margin-bottom: 20px;
}

#image-panel {
    border-radius: 14px;
}

#chat-panel {
    border-radius: 14px;
}

#send-button {
    min-width: 100px;
}

footer {
    display: none !important;
}
"""


# ============================================================
# 8. Gradio 页面
# ============================================================

with gr.Blocks(
    title="MindChen-V Studio",
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate"
    ),
    css=CSS
) as demo:

    # 保存模型对话历史
    model_state = gr.State([])

    gr.Markdown(
        """
        # 🧠 MindChen-V Studio
        """,
        elem_id="main-title"
    )

    gr.Markdown(
        "本地多模态视觉对话平台 · 拖入图片，开始与你训练的模型对话",
        elem_id="main-subtitle"
    )

    with gr.Row(equal_height=False):

        # ====================================================
        # 左侧：图片区域
        # ====================================================

        with gr.Column(
            scale=4,
            min_width=300
        ):

            with gr.Group(elem_id="image-panel"):

                gr.Markdown("### 🖼️ 图片输入")

                image_input = gr.Image(
                    type="pil",
                    label="拖拽图片到这里，或点击上传",
                    sources=["upload"],
                    height=390
                )

                gr.Markdown(
                    "支持 JPG / JPEG / PNG / WEBP / BMP"
                )

                clear_button = gr.Button(
                    "🗑️ 清空图片与对话",
                    variant="secondary"
                )

            # =================================================
            # 参数设置
            # =================================================

            with gr.Accordion(
                "⚙️ 模型生成参数",
                open=False
            ):

                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="最大生成 Token"
                )

                temperature = gr.Slider(
                    minimum=0,
                    maximum=1.5,
                    value=0.7,
                    step=0.05,
                    label="Temperature"
                )

                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1,
                    value=0.85,
                    step=0.05,
                    label="Top-P"
                )

            with gr.Accordion(
                "📌 使用说明",
                open=False
            ):

                gr.Markdown(
                    """
                    1. 上传一张图片。
                    2. 在右侧输入问题。
                    3. 等待模型流式输出。
                    4. 可以连续追问同一张图片。
                    5. 更换图片后，对话会自动重置。

                    图片和推理均在本地运行。
                    """
                )

        # ====================================================
        # 右侧：对话区域
        # ====================================================

        with gr.Column(
            scale=7,
            min_width=400
        ):

            gr.Markdown("### 💬 图片智能对话")

            chatbot = gr.Chatbot(
                type="messages",
                height=520,
                show_label=False,
                placeholder=(
                    "上传图片后，在下方输入问题。\n\n"
                    "例如：请描述这张图片中的内容。"
                ),
                layout="bubble"
            )

            with gr.Row():

                msg = gr.Textbox(
                    placeholder=(
                        "输入问题，例如：这张图片里有什么？"
                    ),
                    show_label=False,
                    lines=2,
                    scale=8,
                    container=True
                )

                send_button = gr.Button(
                    "发送 ➤",
                    variant="primary",
                    elem_id="send-button",
                    scale=1
                )

            status = gr.Markdown(
                "等待上传图片..."
            )

    # ========================================================
    # 事件绑定
    # ========================================================

    send_button.click(
        fn=chat,
        inputs=[
            msg,
            image_input,
            chatbot,
            model_state,
            max_tokens,
            temperature,
            top_p
        ],
        outputs=[
            chatbot,
            model_state,
            msg,
            status
        ]
    )

    msg.submit(
        fn=chat,
        inputs=[
            msg,
            image_input,
            chatbot,
            model_state,
            max_tokens,
            temperature,
            top_p
        ],
        outputs=[
            chatbot,
            model_state,
            msg,
            status
        ]
    )

    # 上传新图片时，重置旧对话
    image_input.change(
        fn=reset_chat,
        inputs=[],
        outputs=[
            chatbot,
            model_state,
            status
        ]
    )

    # 清空图片和聊天记录
    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            image_input,
            chatbot,
            model_state,
            status
        ]
    )


# ============================================================
# 9. 启动
# ============================================================

if __name__ == "__main__":

    load_model()

    print("\n正在启动 MindChen-V Studio...")

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False
    )
