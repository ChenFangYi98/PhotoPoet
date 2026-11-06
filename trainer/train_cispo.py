# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""副本的原始 RLAIF 奖励和 CISPO/GRPO 目标；从原生 SFT Best 独立启动。

执行与归档逻辑在 grpo_copy_runtime.py。原副本已完整备份到 out/grpo_copy/code_before_start_*。
不使用古诗课程、监督回放或 Ollama；默认算法仍为本副本的 CISPO。
"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import datasets  # Windows DLL ordering
import re
import torch
from trainer.grpo_copy_runtime import main

def rep_penalty(text, n=3, cap=0.5):
    """重复惩罚：回答重复 n-gram 越多，扣分越多。"""

    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    """给模型生成的回答打分。

    分数来自两部分：
    - 规则奖励/惩罚：长度、think 标签格式、重复度
    - reward_model 分数：外部奖励模型对回答质量的判断
    """

    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[response_idx] -= rep_penalty(answer)

                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards

def policy_objective(current, old, reference, advantages, completion_mask, args):
    """保留本副本的 CISPO/GRPO 目标；MoE aux_loss 在运行时单独加回。

新、旧、参考 logprob 使用与采样一致的温度，采样不作 top-k/p 截断。
delta/ratio 的有限范围保护只在极端数值时生效。
"""
    delta=(reference-current).clamp(-20,20)
    per_token_kl=torch.exp(delta)-delta-1
    ratio=torch.exp((current-old).clamp(-20,20))
    if args.loss_type == 'cispo':
        clamped_ratio=torch.clamp(ratio,max=args.epsilon_high).detach()
        per_token_loss=-(clamped_ratio*advantages.unsqueeze(1)*current-args.beta*per_token_kl)
    else:
        clipped_ratio=torch.clamp(ratio,1-args.epsilon,1+args.epsilon)
        per_token_loss=-(torch.min(ratio*advantages.unsqueeze(1),
            clipped_ratio*advantages.unsqueeze(1))-args.beta*per_token_kl)
    loss=((per_token_loss*completion_mask).sum(1)/completion_mask.sum(1).clamp(min=1)).mean()
    kl=(per_token_kl*completion_mask).sum()/completion_mask.sum().clamp(min=1)
    return loss,kl.detach()


if __name__ == '__main__':
    main(sys.modules[__name__])
