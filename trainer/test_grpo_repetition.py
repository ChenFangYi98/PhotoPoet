# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Regression tests for repetition rewards, masks and corrective gradients."""
import unittest

import datasets  # Windows DLL ordering
import torch

from trainer.grpo_repetition import (repetition_diagnostics, response_repetition,
    repeated_token_mask, unlikelihood_loss)
from trainer.grpo_poetry_reward import calculate_rewards
from trainer.train_grpo import group_advantages, policy_objective, selection_decision
from types import SimpleNamespace


class AlwaysPositive:
    def get_score(self, messages, answer):
        return 100.


class RepetitionTests(unittest.TestCase):
    def test_chinese_english_sentence_and_unbroken_loops(self):
        examples = ['如何'*80, '岁月静好，\n'*30, '月'*40,
                    'this is a repeated sentence without any useful information. '*8,
                    'abc'*30, '这是一个完整而且包含较多内容的中文句子。'*3]
        for text in examples:
            with self.subTest(text=text[:50]):
                self.assertTrue(repetition_diagnostics(text)['excessive'])

    def test_benign_terms_and_poetic_parallelism_are_not_loops(self):
        for text in ['春风入小院，\n细雨润新芽。\n燕语惊幽梦，\n晨光照落花。',
                     '山重水复疑无路，柳暗花明又一村。',
                     '机器学习包括监督学习和无监督学习，两种学习方式各有用途。',
                     'The model learns from examples. A good model generalizes to new examples.', '哈哈']:
            with self.subTest(text=text):
                self.assertFalse(repetition_diagnostics(text)['excessive'])

    def test_reference_refrain_is_exempt_when_preserved_exactly(self):
        response = '山水有相逢，\n山水有相逢。\n山水有相逢，\n山水有相逢。'
        row = dict(channel='poetry', spec=dict(preserve_chars=True, reference_body='山水有相逢'*4))
        diagnosis = response_repetition(row, response)
        self.assertTrue(diagnosis['allowed_reference'])
        self.assertFalse(diagnosis['excessive'])
        self.assertTrue(response_repetition(row, response+'山水有相逢。')['excessive'])
        appended = response_repetition(row, response+'abc'*50)
        self.assertFalse(appended['allowed_reference'])
        self.assertTrue(appended['excessive'])

    def test_positive_scorer_cannot_reward_chinese_loops(self):
        row = dict(channel='general', messages=[dict(role='user', content='请简要回答问题。')])
        bad = '如何'*80
        good = '解决问题前先明确目标，再核对条件，最后给出具体答案。'
        rewards, details = calculate_rewards([row], [bad, good], AlwaysPositive(), 2,
                                             completion_ended=[False, True])
        self.assertLess(float(rewards[0]), 0.)
        self.assertGreater(float(rewards[1]), float(rewards[0]))
        self.assertTrue(details[0]['repetition']['excessive'])
        self.assertEqual(details[0]['rules']['truncated'], -1.)

    def test_only_completion_repeats_before_eos_are_masked(self):
        ids = torch.tensor([[5,6,5,6,5,6,5,6,2,2,2], [7,8,9,10,11,12,13,14,2,2,2]])
        valid = torch.tensor([[1]*9+[0]*2]*2, dtype=torch.bool)
        mask = repeated_token_mask(ids, valid)
        self.assertFalse(mask[0,:4].any())
        self.assertTrue(mask[0,4:8].all())
        self.assertFalse(mask[:,8:].any())
        self.assertFalse(mask[1].any())
        self.assertFalse(repeated_token_mask(ids, valid, [True, True]).any())

    def test_unlikelihood_learns_when_group_rewards_are_identical(self):
        torch.set_num_threads(2)
        logits = torch.nn.Parameter(torch.tensor([[[2., 0., -1.]]]*2))
        optimizer = torch.optim.SGD([logits], lr=.2)
        old = logits.log_softmax(-1)[...,0].detach()
        advantages = group_advantages(torch.tensor([-5., -5.]), 2)
        self.assertTrue(advantages.eq(0).all())
        options = SimpleNamespace(loss_type='grpo', beta=.1, epsilon=.2, epsilon_high=5.)
        initial = logits.softmax(-1)[...,0].detach().clone()
        current = logits.log_softmax(-1)[...,0]
        pg, _ = policy_objective(current, old, old, advantages, torch.ones_like(current), options)
        loss = pg + .2*unlikelihood_loss(current, torch.ones_like(current, dtype=torch.bool))
        loss.backward()
        optimizer.step()
        self.assertTrue((logits.softmax(-1)[...,0] < initial).all())

    def test_repeated_long_sentence_twice_has_corrective_tokens(self):
        ids = torch.tensor([list(range(10, 26))*2+[2]])
        valid = torch.ones_like(ids, dtype=torch.bool)
        mask = repeated_token_mask(ids, valid)
        self.assertFalse(mask[0,:16].any())
        self.assertTrue(mask[0,16:32].all())
        self.assertFalse(mask[0,32])

    def test_no_repeat_no_auxiliary_gradient_and_finite_near_one(self):
        logps = torch.tensor([[-.5,-1.]], requires_grad=True)
        loss = unlikelihood_loss(logps, torch.zeros_like(logps, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue(logps.grad.eq(0).all())
        self.assertTrue(torch.isfinite(unlikelihood_loss(torch.tensor([[0.]]), torch.tensor([[True]]))))

    def test_best_selection_rejects_higher_reward_with_more_repetition(self):
        options = SimpleNamespace(general_regression_tolerance=.15, repetition_regression_tolerance=.05)
        progress = {}
        baseline = dict(general=dict(model_score=1., repetition_rate=.1),
                        poetry=dict(reward=2., repetition_rate=.1, qualified_pass_rate=.5))
        _, initial_rank, accepted = selection_decision(baseline, progress, options)
        self.assertTrue(accepted)
        worse = dict(general=dict(model_score=2., repetition_rate=.4),
                     poetry=dict(reward=5., repetition_rate=.1, qualified_pass_rate=.6))
        self.assertFalse(selection_decision(worse, progress, options)[2])
        better = dict(general=dict(model_score=1., repetition_rate=0.),
                      poetry=dict(reward=2., repetition_rate=.1, qualified_pass_rate=.5))
        _, rank, accepted = selection_decision(better, progress, options)
        self.assertTrue(accepted)
        self.assertGreater(rank, initial_rank)
        better['general']['model_score'] = -.5
        self.assertFalse(selection_decision(better, progress, options)[2])


if __name__ == '__main__':
    unittest.main()
