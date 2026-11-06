# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""回归检查：小通道不放大、验证不重复、事实配比保留、旧断点不静默迁移。"""
from collections import Counter
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from trainer.adaptive_sft_sampling import (CHECKPOINT_FORMAT, draw_examples,
    require_checkpoint_format, sampling_plan)


def fake_corpus():
    return SimpleNamespace(indices={
        'train_base_fact': range(953158), 'train_author_fact': range(31028),
        'train_verified_fact': range(64096), 'train_base_open': range(381578),
        'train_base_general': range(4414539), 'train_author_general': range(47),
        'valid_base_fact': range(60), 'valid_author_fact': range(20),
        'valid_verified_fact': range(20), 'valid_base_open': range(100),
        'valid_base_general': range(100), 'valid_author_general': range(17)},
        manifest={'fact_source_ratios': {'base': .35, 'author': .45, 'verified': .20}})


class SamplingTests(unittest.TestCase):
    def test_47_rows_receive_their_size_proportional_share(self):
        plan = sampling_plan(fake_corpus(), 64, [.2979637,.2091945,.4928418])
        expected = .5*47/(4414539+47)
        self.assertAlmostEqual(plan['train_author_general']['expected_sample_fraction'], expected)
        self.assertLess(expected, .00001)
        self.assertAlmostEqual(sum(p['expected_sample_fraction'] for p in plan.values()), 1.)

    def test_actual_draws_do_not_split_general_channels_equally(self):
        corpus = fake_corpus()
        corpus.indices['train_base_general'] = range(1000)
        corpus.indices['train_author_general'] = range(1)
        counts = Counter(key for seed in range(1000)
                         for key,_ in draw_examples(corpus,64,[.3,.2,.5],seed))
        self.assertEqual(counts['train_base_general']+counts['train_author_general'], 32000)
        self.assertLess(counts['train_author_general'], 100)
        self.assertGreater(counts['train_author_general'], 0)

    def test_factual_source_ratios_are_preserved(self):
        plan = sampling_plan(fake_corpus(), 64, [.2979637,.2091945,.4928418])
        for family, ratio in [('base',.35),('author',.45),('verified',.20)]:
            self.assertAlmostEqual(plan[f'train_{family}_fact']['expected_sample_fraction'], 19/64*ratio)

    def test_multiple_sources_in_same_family_share_family_budget(self):
        corpus = fake_corpus()
        corpus.indices['train_extra_author_fact'] = range(31028)
        plan = sampling_plan(corpus,64,[.3,.2,.5])
        left = plan['train_author_fact']['expected_samples_per_update']
        right = plan['train_extra_author_fact']['expected_samples_per_update']
        self.assertAlmostEqual(left,right)
        self.assertAlmostEqual((left+right)/19,.45)

    def test_validation_unique_and_clarification_separate(self):
        draws = draw_examples(fake_corpus(),128,[1,1,1],9143,'valid')
        self.assertEqual(len(draws),128)
        self.assertEqual(len(set(draws)),128)
        self.assertNotIn('valid_author_general',{key for key,_ in draws})
        self.assertEqual(Counter(key.rsplit('_',1)[-1] for key,_ in draws),
                         {'fact':43,'open':43,'general':42})

    def test_validation_uniform_over_rows_not_source_channels(self):
        corpus = fake_corpus()
        plan = sampling_plan(corpus,30,[1,1,1],'valid')
        self.assertAlmostEqual(plan['valid_base_fact']['expected_samples_per_update'],6.)
        self.assertAlmostEqual(plan['valid_author_fact']['expected_samples_per_update'],2.)
        self.assertAlmostEqual(plan['valid_verified_fact']['expected_samples_per_update'],2.)

    def test_validation_rejects_impossible_nonduplicate_budget(self):
        corpus = fake_corpus()
        corpus.indices['valid_base_open'] = range(1)
        with self.assertRaisesRegex(ValueError,'无法无重复'):
            draw_examples(corpus,9,[1,1,1],42,'valid')

    def test_deterministic_draws_preserve_global_rng(self):
        state = random.getstate()
        one = draw_examples(fake_corpus(),64,[.3,.2,.5],42)
        self.assertEqual(state,random.getstate())
        self.assertEqual(one,draw_examples(fake_corpus(),64,[.3,.2,.5],42))
        self.assertNotEqual(one,draw_examples(fake_corpus(),64,[.3,.2,.5],43))

    def test_invalid_source_weights_rejected(self):
        corpus = fake_corpus()
        corpus.manifest['fact_source_ratios']['author'] = float('nan')
        with self.assertRaises(ValueError):draw_examples(corpus,64,[.3,.2,.5],42)

    def test_old_checkpoint_has_actionable_rejection(self):
        with self.assertRaisesRegex(ValueError,'train_adaptive_sft_v3_legacy.py'):
            require_checkpoint_format({'format_version':'adaptive_sft_v2'})
        require_checkpoint_format({'format_version':CHECKPOINT_FORMAT})

    def test_clarification_does_not_lower_general_nll(self):
        from trainer.train_adaptive_sft import evaluate
        model = torch.nn.Linear(1,1)
        router = torch.nn.Linear(1,1)
        examples = [{'meta':{'task':i,'label_confidence':1.}} for i in range(3)]
        with patch('trainer.train_adaptive_sft.sample_nll',side_effect=[
                torch.tensor([1.,2.,3.]),torch.tensor([100.,100.])]), \
             patch('trainer.train_adaptive_sft.router_forward',return_value=(torch.eye(3),0.)):
            metrics,_ = evaluate(model,router,examples,[],None,SimpleNamespace(device='cpu'),[{},{}])
        self.assertEqual(metrics['general_nll'],3.)
        self.assertEqual(metrics['balanced_nll'],2.)
        self.assertEqual(metrics['clarification_nll'],100.)
        self.assertEqual(metrics['clarification_count'],2)
        self.assertTrue(model.training and router.training)


if __name__ == '__main__': unittest.main()
