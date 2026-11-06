# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""视觉语言模型：把图片特征放入语言模型的 token 序列，再预测回答。"""
import os
import torch
import warnings
from .model_mindchen import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(MindChenConfig):
    """在语言模型配置上补充图片占位符、视觉特征维度和图片 token 数。"""
    model_type = "mindchen-v"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        super().__init__(**kwargs)

class MMVisionProjector(nn.Module):
    """将视觉编码器的每个特征向量映射到语言模型的隐藏维度。

    输入 [批次, 图片块数, 视觉维度]，输出 [批次, 图片块数, 语言维度]。
    两个 token 数参数用于接口兼容；当前 MLP 不改变图片块数量。
    """
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)

# 继承自语言模型，复用词嵌入、Transformer 层和文本生成方法。
class MindChenVLM(MindChenForCausalLM):
    """SigLIP 视觉编码器 + 可训练投影层 + MindChen 自回归语言模型。"""
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):
        self.config = config or VLMConfig()
        super().__init__(self.config)
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)

    @staticmethod
    def get_vision_model(model_path: str):
        """从本地加载预训练视觉编码器与预处理器，冻结视觉编码器参数。"""
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        try:
            model = SiglipVisionModel.from_pretrained(model_path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        """把 PIL 图片转换为视觉编码器可接收的像素张量字典。"""
        # SigLIP expects three channels, including for grayscale/palette inputs.
        if image.mode != 'RGB': image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        """不计算视觉编码器梯度，返回每个图像块的隐藏表示。"""
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        """按顺序用图片特征替换连续的 image_pad 词嵌入。

        h 是 [批次, 序列长度, 隐藏维度]；每组连续占位符对应一张图片。
        图像 token 数和占位符数应一致，否则序列与图片无法正确对齐。
        """
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 先把文字 token ID 转为向量；随后在图片占位符的位置插入视觉向量。
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 自回归生成时只在第一次处理图片，后续 token 复用 KV 缓存。
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                sample_val = next(iter(pixel_values.values()))
                if sample_val.ndim == 5:
                    bs, num = sample_val.shape[:2]
                    vision_tensors = self.vision_proj(MindChenVLM.get_image_embeddings({k: v.flatten(0, 1) for k, v in pixel_values.items()}, self.vision_encoder)).view(bs, num, self.config.image_token_len, -1)
                else:
                    vision_tensors = self.vision_proj(MindChenVLM.get_image_embeddings(pixel_values, self.vision_encoder))
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape
                vision_tensors = torch.stack([self.vision_proj(MindChenVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)) for i in range(num)], dim=1)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])

        # Vision only supplies embeddings; backbone, KV cache, logits and loss are shared.
        output = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
            use_cache=use_cache, logits_to_keep=logits_to_keep, labels=labels,
            inputs_embeds=hidden_states, **args)
        output.aux_loss = output.aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0
        return output

    def generate(self, *args, num_return_sequences=1, **kwargs):
        """需要多条候选回复时，为每条回复复制对应的图片输入。"""
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
