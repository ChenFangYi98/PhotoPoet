# Modified by chenfangyi for the unified MindChen release (2026-11-06).
import time
import argparse
import warnings
import json
from contextlib import nullcontext
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_mindchen import MindChenConfig, MindChenForCausalLM
from model.checkpoint_config import load_checkpoint_config
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent


def checkpoint_path(args):
    """显式路径优先；目录/前缀模式按参数解析，禁止偷偷选择旧运行。"""
    explicit = getattr(args, 'checkpoint', '')
    if explicit:
        candidate = Path(explicit)
    elif getattr(args, 'save_dir', None) is not None or getattr(args, 'weight', None) is not None:
        candidate = Path(args.save_dir or 'out') / (
            f'{args.weight or "full_sft"}_{args.hidden_size}'+('_moe' if args.use_moe else '')+'.pth')
    else:
        session_path = ROOT/'out/session_control/grpo_session.json'
        if not session_path.is_file():
            raise FileNotFoundError('没有当前 GRPO 会话，请使用 --checkpoint 指定权重')
        session = json.loads(session_path.read_text(encoding='utf-8-sig'))
        folder = Path(session['final_run_dir'])
        config_path = folder/'run_config.json'
        if not config_path.is_file():
            raise FileNotFoundError(f'当前训练尚未准备完毕：{folder}；请等待或显式指定 --checkpoint')
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        saved_args = config['args']
        # Architecture is bound to the source snapshot, never guessed from CLI defaults.
        source = config['source']
        model_config = config.get('model_config')
        if model_config is None:
            from model.checkpoint_config import sidecar_path
            model_config = json.loads(sidecar_path(source['snapshot']).read_text(encoding='utf-8'))['model_config']
        suffix = '_moe' if model_config['use_moe'] else ''
        candidate = folder/f'{saved_args["save_weight"]}_best_{model_config["hidden_size"]}{suffix}.pth'
    return (candidate if candidate.is_absolute() else ROOT/candidate).resolve()


def resolved_prompt_style(args):
    style = getattr(args, 'prompt_style', 'auto')
    if style != 'auto':
        return style
    # 显式权重使用真实文件名，不能再被无关的 --weight 改变模板。
    name = checkpoint_path(args).name if args.load_from == 'model' else args.load_from
    return 'raw' if name.startswith('pretrain') else 'chat'

def init_model(args):
    native = args.load_from == 'model'
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'model') if native else args.load_from)
    if native:
        ckp = checkpoint_path(args)
        if not ckp.is_file():
            raise FileNotFoundError(f'权重不存在：{ckp}；请用 --checkpoint 指定实际 .pth 文件')
        from trainer.continue_pretrain_utils import sha256_file
        print(f'[checkpoint] {ckp}\n[sha256] {sha256_file(ckp)}')
        config,config_info=load_checkpoint_config(ckp,
            fallback=dict(hidden_size=args.hidden_size,num_hidden_layers=args.num_hidden_layers,
                          use_moe=bool(args.use_moe)),
            config_path=getattr(args,'model_config',''),norm_topk_prob=getattr(args,'norm_topk_prob',None),
            inference_rope_scaling=True if args.inference_rope_scaling else None)
        print(f'[model_config] {config_info}')
        if config_info['source']=='legacy_fallback':
            print('[配置提示] 裸权重无伴随配置；旧模型默认归一化。新 SFT 请同时复制 .pth.config.json，或显式 --norm_topk_prob 0。')
        model = MindChenForCausalLM(config)
        model.load_state_dict(torch.load(ckp, map_location='cpu', weights_only=True), strict=True)
        if getattr(args, 'lora_path', None):
            if args.lora_weight != 'None': raise ValueError('Choose lora_path or legacy lora_weight, not both')
            from model.lora_adapter import load_bound_adapter
            adapter = Path(args.lora_path); adapter = adapter if adapter.is_absolute() else ROOT/adapter
            load_bound_adapter(model, adapter, ckp)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, str(ROOT/(args.save_dir or 'out')/f'{args.lora_weight}_{model.config.hidden_size}.pth'))
    else:
        if getattr(args, 'lora_path', None): raise ValueError('lora_path requires a native --checkpoint')
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    # Match GRPO: keep small optimizer updates in FP32 parameters, autocast compute.
    return model.to(device=args.device, dtype=torch.float32).eval(), tokenizer


def inference_context(args):
    dtype = getattr(args, 'dtype', 'auto')
    if str(args.device).startswith('cuda'):
        if dtype == 'auto':
            dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float32'
        if dtype != 'float32':
            return torch.autocast('cuda', dtype=getattr(torch, dtype))
    return nullcontext()

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MindChen模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default=None, type=str, help="模型权重目录；与 weight 均未指定时使用当前 GRPO best")
    parser.add_argument('--checkpoint', default='', help='显式指定原生 .pth 权重，优先于 save_dir/weight')
    parser.add_argument('--model_config',default='',help='默认读取权重旁的 .pth.config.json；可显式指定配置文件')
    parser.add_argument('--norm_topk_prob',type=int,choices=(0,1),default=None,
                        help='无配置旧权重的显式路由设置；默认兼容旧行为，有配置时禁止冲突覆盖')
    parser.add_argument('--prompt_style', choices=('auto','raw','chat'), default='auto', help='SFT 使用 chat；预训练续写使用 raw')
    parser.add_argument('--seed', type=int, default=42, help='固定随机种子，便于复现')
    parser.add_argument('--do_sample', type=int, choices=(0,1), default=0, help='默认贪心事实测试；创作设为1')
    parser.add_argument('--weight', default=None, type=str, help="权重名称前缀；显式目录模式默认 full_sft")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--lora_path', default=None, help='New bound adapter.pth from train_lora_sft.py')
    parser.add_argument('--mindchen_identity', action='store_true', help='直接身份问题使用 MindChen 的项目署名')
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=1, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=256, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.8, type=float, help="采样温度")
    parser.add_argument('--top_p', default=1., type=float, help="nucleus采样阈值")
    parser.add_argument('--open_thinking', default=0, type=int, choices=(0,1), help="是否开启自适应思考")
    parser.add_argument('--dtype', choices=('auto','float32','float16','bfloat16'), default='auto', help='计算精度；参数保留 FP32')
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args(argv)
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.max_new_tokens < 1:
        parser.error('要求 temperature>0 且 0<top_p<=1')
    return args


def main():
    args = parse_args()
    setup_seed(args.seed)
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if args.mindchen_identity and input_mode == 1 and resolved_prompt_style(args) == 'chat':
            from model.project_identity import project_identity_reply
            identity = project_identity_reply(prompt)
            if identity is not None:
                print(f'🧠: {identity}\n')
                conversation.append({'role': 'assistant', 'content': identity})
                continue
        if resolved_prompt_style(args) == 'raw':
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        with torch.inference_mode(), inference_context(args):
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=bool(args.do_sample), streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, top_k=0, repetition_penalty=1
            )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
