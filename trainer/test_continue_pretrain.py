# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""继续预训练的调度、恢复游标、随机状态与验证损失检查。"""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import datasets  # 与训练入口一致，先于torch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from trainer.continue_pretrain_utils import (
    EpochBatchSampler, MetricsWriter, atomic_torch_save, restore_rng, rewind_metrics, rng_state,
)
from trainer.train_continue_pretrain import evaluate, lr_for_update, parse_args, read_samples


class ContinuePretrainTests(unittest.TestCase):
    def test_scheduler_warmup_and_final_lr(self):
        rates = [lr_for_update(i, 100, 5e-5, 0.1, 0.1) for i in range(1, 101)]
        self.assertAlmostEqual(rates[0], 5e-6)
        self.assertAlmostEqual(rates[9], 5e-5)
        self.assertAlmostEqual(rates[-1], 5e-6)
        self.assertTrue(all(a >= b for a, b in zip(rates[9:], rates[10:])))

    def test_resumed_sampler_keeps_exact_remaining_order(self):
        full = list(EpochBatchSampler(35, 8, seed=42))
        remaining = list(EpochBatchSampler(35, 8, seed=42, skip_batches=2))
        self.assertEqual(full[2:], remaining)
        self.assertEqual(sorted(i for batch in full for i in batch), list(range(35)))
        self.assertNotEqual(full, list(EpochBatchSampler(35, 8, seed=43)))

    def test_rng_checkpoint_weights_only_roundtrip(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        saved = rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rng.pth'
            atomic_torch_save(path, saved)
            loaded = torch.load(path, weights_only=True)
            restore_rng(loaded)
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_validation_is_token_weighted_and_restores_train_mode(self):
        class Model(torch.nn.Module):
            def forward(self, inputs, labels):
                return SimpleNamespace(loss=torch.tensor(2.0 if inputs[0, 0] == 1 else 4.0))
        inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
        labels = torch.tensor([[1, 2, -100, -100], [3, 4, 5, 6]])
        model = Model().train()
        result = evaluate(model, DataLoader(TensorDataset(inputs, labels), batch_size=1), SimpleNamespace(device='cpu', dtype='float32'))
        self.assertEqual(result['val_tokens'], 4)
        self.assertAlmostEqual(result['val_nll'], 3.5)
        self.assertAlmostEqual(result['val_ppl'], math.exp(3.5))
        self.assertTrue(model.training)

    def test_smoke_reads_only_prefix_and_excludes_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.jsonl'
            path.write_text(''.join(json.dumps({'text': str(i)}) + '\n' for i in range(7)) + 'BAD_JSON\n', encoding='utf-8')
            train, val = read_samples(SimpleNamespace(data_path=str(path), train_samples=5, validation_samples=2))
            self.assertEqual(list(val['text']), ['0', '1'])
            self.assertEqual(list(train['text']), ['2', '3', '4', '5', '6'])

    def test_rewind_preserves_committed_metrics_and_backs_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            writer = MetricsWriter(folder)
            for i in range(4):
                writer.write({'update': i, 'phase': 'train'})
            writer.close()
            rewind_metrics(folder, 2, 'session')
            rows = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['update'] for r in rows], [0, 1, 2])
            self.assertTrue((folder / 'metrics.jsonl.before_resume_session').exists())

    def test_defaults_target_requested_weight_and_data(self):
        args = parse_args([])
        self.assertEqual(Path(args.init_weight_path).name, 'pretrain_768_moe.pth')
        self.assertEqual(Path(args.data_path).name, 'pretrain_t2t_with_poetry_gujilab.jsonl')
        self.assertEqual(args.batch_size * args.accumulation_steps, 64)
        self.assertTrue(args.use_moe)


if __name__ == '__main__':
    unittest.main()
