# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Merge a bound LoRA adapter, or record finite identity/ability probes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer.lora_training import load_native, project_path
from model.lora_adapter import load_bound_adapter, save_merged
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['merge', 'probe'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--kind', choices=['llm', 'vlm'], default='vlm' if (ROOT/'model/model_vlm.py').exists() else 'llm')
    parser.add_argument('--output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--vision_model', default=str(ROOT/'model/siglip2-base-p32-256-ve'))
    parser.add_argument('--questions', help='JSON list of question strings')
    parser.add_argument('--system_prompt', default='', help='Default empty: evaluate without an identity hint')
    parser.add_argument('--image', help='Optional image for VLM probes')
    parser.add_argument('--max_new_tokens', type=int, default=96)
    args = parser.parse_args()
    if args.operation == 'merge' and not args.adapter: parser.error('merge requires --adapter')
    extension = '.pth' if args.operation == 'merge' else '.json'
    output = project_path(args.output) if args.output else ROOT/'out/lora_exports'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')/(args.operation + extension)
    if not output.is_relative_to((ROOT/'out').resolve()): parser.error('Outputs must stay under this project out/')
    if output.exists(): parser.error('Output already exists; choose a new file')
    checkpoint = project_path(args.checkpoint)
    model, tokenizer, processor = load_native(checkpoint, args.kind, args.device, project_path(args.vision_model))
    if args.adapter: load_bound_adapter(model, project_path(args.adapter), checkpoint)
    if args.operation == 'merge':
        save_merged(model, output); print(output); return
    questions = json.loads(project_path(args.questions).read_text(encoding='utf-8-sig')) if args.questions else [
        '你是谁？','你叫什么名字？','你的开发者是谁？','Who are you?',
        'What is your name?','Are you ChatGPT?','你是通义千问吗？',
        '计算 12 加 7。','请用一句话解释什么是光合作用。']
    pixels = None
    if args.image:
        if args.kind != 'vlm': parser.error('--image requires a VLM')
        from PIL import Image
        with Image.open(project_path(args.image)) as image:
            pixels = {k: v.to(args.device) for k,v in processor(images=image.convert('RGB'), return_tensors='pt').items()}
    model.eval(); answers = []
    torch.manual_seed(42)
    with torch.inference_mode():
        for question in questions:
            text = question
            if pixels is not None: text = model.config.image_special_token * model.config.image_token_len + '\n' + text
            messages = ([dict(role='system',content=args.system_prompt)] if args.system_prompt else []) + [dict(role='user',content=text)]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
            inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
            kwargs = dict(pixel_values=pixels) if pixels is not None else {}
            result = model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                    max_new_tokens=args.max_new_tokens, do_sample=False, temperature=1.0,
                                    top_p=1.0, top_k=0, pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id, **kwargs)
            answer = tokenizer.decode(result[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            record = dict(question=question, answer=answer); answers.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(checkpoint=str(checkpoint), adapter=args.adapter, system_prompt=args.system_prompt,
                                     image=args.image, answers=answers, limitation='Finite probes; human review is required.'),
                                 ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__': main()
