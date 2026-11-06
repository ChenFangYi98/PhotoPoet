# Modified by chenfangyi for the unified MindChen release (2026-11-06).
from torch.utils.data import Dataset
import torch
import json
import os
import random
import math
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 这个文件把 json/jsonl 数据转换成 PyTorch Dataset。
# 训练脚本中的 DataLoader 会不断调用 __getitem__，拿到 input_ids、labels 或偏好训练需要的字段。
# 对新手来说，最重要的是理解：
# 1. input_ids 是模型输入的 token id。
# 2. labels 是希望模型预测的答案。
# 3. labels 中的 -100 表示“这个位置不算 loss”，PyTorch cross_entropy 会忽略它。


def pre_processing_chat(conversations, add_system_ratio=0.2, rng=None):
    """对对话数据做轻量预处理。

    普通对话按概率补充身份、中文表达或古典文学风格提示；已有 system 不覆盖。
    工具定义、工具调用和工具返回均原样保留。rng 可传独立 random.Random，避免污染全局随机状态。
    自适应 SFT 训练时通过独立随机数复用此函数；验证集不随机添加提示。
    """

    if not math.isfinite(add_system_ratio) or not 0<=add_system_ratio<=1:
        raise ValueError('add_system_ratio 必须在 [0,1] 内')
    if not conversations:return conversations
    if add_system_ratio==0:return conversations
    if any(conv.get('tools') or conv.get('tool_calls') or conv.get('role')=='tool' for conv in conversations):return conversations
    rng=random if rng is None else rng
    # 16 条等概率选择：2 条介绍身份，其余引导风格。提示不能代替高质量回答数据。
    # 不强制每条回复自我介绍或写成古文；具体任务及输出格式优先。
    SYSTEM_PROMPTS = (
        '你是mindchen，由chenfangyi训练，专注于古诗词与古文的语言模型。请准确、清楚地回答问题，不确定时明确说明。',
        '你是mindchen，由chenfangyi训练，专注古诗词与古文。依照用户要求回答，表达自然，引用作品时尊重原文。',
        '请用温和自然的语气交流，留意用户的心情和需求，先理解问题，再给出有帮助的回答。',
        '请以真诚、耐心的语气回答；遇到伤感或困惑的话题，可以先回应对方的感受，再适度给出建议。',
        '请让语言清澈而有温度，依据情境选择简洁或细致的表达，不必刻意使用华丽辞藻。',
        '请用亲切平实的中文交流，必要时举具体例子，让解释容易理解。',
        '回答古诗词问题时，留意意象、情境与情感，结合给出的文本解释，避免编造作者或出处。',
        '赏析古文时，先理解字句和上下文，再说明其含义和表达效果；存在不同解释时可说明。',
        '翻译古文时兼顾文意和语气，遵守用户的输出要求，难以确定之处不要随意补造。',
        '原创写作时可以运用自然的意象、节奏和情感，依据题目自由发挥，并清楚区分原创与引用。',
        '引用、背诵或续写已有作品时尊重原文；记不清原句或出处时，坦诚说明，不把猜测当作事实。',
        '讨论诗词时可以从画面、声音、时令或情绪切入，让解释生动，同时以文本为依据。',
        '请根据用户语境调整语气：知识问题清楚准确，创作问题富有想象，日常交流自然亲切。',
        '先满足用户明确提出的格式和篇幅，再选择合适的表达方式；只要求正文时只输出正文。',
        'Be thoughtful and warm. Follow the requested format, distinguish quotation from original writing, and acknowledge uncertainty.',
        'Help with classical Chinese poetry and prose using clear explanations, sensitivity to imagery, and care with attribution.',
    )
    # 概率性添加system
    if not any(conv.get('role')=='system' for conv in conversations):
        if rng.random() < add_system_ratio:
            return [{'role': 'system', 'content': rng.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """处理 chat_template 生成的 prompt 文本。

    模板可能会生成空的 <think></think>，这里大概率移除，保留一小部分让模型也见过这种格式。
    """

    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    """预训练数据集。

    数据格式通常是每行一个 {"text": "..."}。
    预训练阶段目标很直接：给一段普通文本，让模型学习“下一个 token 是什么”。
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # datasets.load_dataset 会按 json/jsonl 读取文件，并返回可索引的数据集对象。
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 预训练文本不需要 chat_template，只做普通分词。
        # max_length - 2 是为了给 BOS/EOS 预留位置。
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # BOS 表示序列开始，EOS 表示序列结束。
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 不足 max_length 的部分补 pad，保证同一个 batch 内长度一致。
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 预训练中 labels 基本等于 input_ids，模型内部会做错位预测。
        labels = input_ids.clone()
        # pad 位置不应该参与 loss。
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    """监督微调 SFT 数据集。

    数据格式通常是 conversations: [{role: user/system/assistant, content: ...}, ...]。
    SFT 和预训练的关键区别：
    - prompt/user/system 部分只作为上下文输入，不计算 loss。
    - assistant 回复部分才是模型要学习输出的内容。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 显式声明字段，兼容工具调用、思考内容等可能为空的列。
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 用 tokenizer 模板中的 assistant 开头/结尾标记来定位“需要算 loss 的回答区间”。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """把结构化 conversations 转成模型实际看到的一整段文本。"""

        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # 如果 system 消息中带工具定义，则传给 chat_template，让模板生成工具说明。
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # tool_calls 可能以字符串形式存储，训练前转回 list/dict。
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """生成 SFT labels。

        初始全是 -100，表示所有位置都不算 loss。
        只有找到 assistant 起始标记后，才把 assistant 回复对应 token 填回 labels。
        """

        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # 先做对话级预处理，再由 tokenizer 的 chat_template 统一拼接格式。
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        # 截断到最大长度，再 pad 到固定长度。
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """DPO 偏好数据集。

    每条样本包含 chosen/rejected 两段回答：
    - chosen 是更好的回答
    - rejected 是较差的回答
    DPO 训练会让策略模型相对参考模型更偏向 chosen。
    """

    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 同 SFT：只在 assistant 回复区间上计算偏好损失。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        # chosen/rejected 都先转成完整对话文本，再分词到固定长度。
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # DPO 脚本手动做错位：
        # x 是输入 token，y 是下一个 token，mask 表示哪些 y 位置参与 DPO。
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """生成 DPO 使用的 loss mask，assistant 回复区间为 1，其它位置为 0。"""

        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """PPO/GRPO 等强化学习阶段的数据集。

    这里只返回 prompt，不直接给标准答案。
    后续训练脚本会让当前模型生成回答，再由 reward_model 和规则打分。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # RL 阶段只把最后一轮 assistant 答案之前的对话作为 prompt，让模型自己生成答案。
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    """Agentic RL 数据集。

    这类数据包含工具定义和可验证的 gt，用来训练模型在多轮工具调用后给出最终答案。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8，Windows 上处理 jsonl 时更稳。
        with open(jsonl_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        # 取出 messages 和工具定义；最后一条通常是目标答案/标注，不作为 prompt 输入。
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass
