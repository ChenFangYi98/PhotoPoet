# Modified by chenfangyi for the unified MindChen release (2026-11-06).
import time
import argparse
import os
import warnings
import torch
import random
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_vlm import MindChenVLM, VLMConfig
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent

def init_model(args):
    """加载评估所需的模型、权重、分词器和预处理器。"""
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        from trainer.lora_training import load_native
        moe_suffix = '_moe' if args.use_moe else ''
        # 根据参数选择权重文件；预训练和微调权重使用不同的前缀。
        ckp = Path(args.checkpoint or os.path.join(args.save_dir, f'{args.weight}_{args.hidden_size}{moe_suffix}.pth'))
        ckp = ckp if ckp.is_absolute() else ROOT/ckp
        model, tokenizer, _ = load_native(ckp, 'vlm', 'cpu', args.vision_model)
        if args.lora_path:
            from model.lora_adapter import load_bound_adapter
            adapter = Path(args.lora_path); adapter = adapter if adapter.is_absolute() else ROOT/adapter
            load_bound_adapter(model, adapter, ckp)
    else:
        if args.lora_path: raise ValueError('lora_path requires a native --checkpoint')
        tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = MindChenVLM.get_vision_model(args.vision_model)
    get_model_params(model, model.config)
    model = model.eval()
    if "cuda" in args.device: model = model.half()
    return model.to(args.device), tokenizer, model.processor


def main():
    """解析评估参数，逐张图片生成文字描述。"""
    parser = argparse.ArgumentParser(description="MindChen-V Chat")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀（pretrain_vlm, sft_vlm）")
    parser.add_argument('--checkpoint', default='', help='Explicit native VLM .pth, including merged LoRA weights')
    parser.add_argument('--lora_path', default=None, help='Bound adapter.pth from train_lora_vlm.py')
    parser.add_argument('--vision_model', default=str(ROOT/'model/siglip2-base-p32-256-ve'))
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--image_dir', default='./dataset/eval_images/', type=str, help="测试图像目录")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    args = parser.parse_args()
    
    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 自动测试image_dir中的所有图像
    prompt = "<image>\n请描述这张图中的主要物体和场景。"
    # prompt = "<image>\nPlease illustrate the image through your words."
    for image_file in sorted(os.listdir(args.image_dir)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            setup_seed(random.randint(1, 31415926))
            image_path = os.path.join(args.image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in MindChenVLM.image2tensor(image, preprocess).items()}
            
            messages = [{"role": "user", "content": prompt.replace('<image>', model.config.image_special_token * model.config.image_token_len)}]
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
            inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)
            
            print(f'[图像]: {image_file}')
            print(f"提示: {repr(prompt)}")
            print('模型: ', end='')
            st = time.time()
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, pixel_values=pixel_values
            )
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
