# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Supervised replay of train-only format targets and general answers during RL."""
import json
import random
from pathlib import Path

import torch
from trainer.sft_data import AssistantDataset, SFTCollator


class SupervisedReplay:
    def __init__(self, path, tokenizer, max_length):
        self.path = Path(path)
        encoder = AssistantDataset([], tokenizer, max_length)
        self.groups = {'poetry': [], 'general': []}
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split', 'train') != 'train':
                raise ValueError('Replay accepts training examples only')
            feature = encoder.encode(row)
            if feature['truncated'] or not feature['eos_supervised'] or not feature['target_tokens']:
                raise ValueError('Replay must preserve all target tokens and EOS')
            feature.pop('rendered', None)
            self.groups['poetry' if row.get('spec') else 'general'].append(feature)
        if any(not group for group in self.groups.values()):
            raise ValueError('Replay needs both poetry and general training examples')
        self.collate = SFTCollator(tokenizer.pad_token_id)

    def examples(self, seed):
        rng = random.Random(seed)
        return [rng.choice(self.groups[channel]) for channel in ('poetry', 'general')]

    def backward(self, model, scaler, device, context, seed, weight, denominator=1):
        """Equal channel mass; each example is forwarded separately to bound VRAM."""
        values = []
        for example in self.examples(seed):
            batch = self.collate([example])
            with context():
                output = model(input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
                loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite supervised replay loss')
            scaler.scale(loss*(weight/(2*denominator))).backward()
            values.append(float(loss.detach()))
        return sum(values)/len(values)
