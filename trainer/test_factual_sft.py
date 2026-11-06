# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""验证训练原理与保护边界：长度权重、实体位置、排序梯度、KL、确定性和严格评分。"""
import unittest
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from trainer.factual_sft_utils import (factual_losses,ranking_loss,replay_kl,entity_feature,
    collate_entities,sample_indices,correct_answer,entity_correct_answer,checkpoint_rank)

ROOT=Path(__file__).resolve().parent.parent


class FactualSFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)

    def test_sample_normalization_removes_length_advantage(self):
        logits=torch.tensor([[[0.,2.],[0.,0.],[0.,0.],[0.,0.]],
                             [[2.,0.],[2.,0.],[2.,0.],[0.,0.]]],requires_grad=True)
        labels=torch.tensor([[-100,0,-100,-100],[-100,0,0,0]])
        mask=labels.ne(-100)
        result=factual_losses(logits,labels,mask,torch.ones(2),entity_fraction=0.)
        long_easy=F.cross_entropy(torch.tensor([[2.,0.]]),torch.tensor([0]))
        short_hard=F.cross_entropy(torch.tensor([[0.,2.]]),torch.tensor([0]))
        self.assertAlmostEqual(float(result['balanced']),float((short_hard+long_easy)/2),places=6)
        self.assertAlmostEqual(float(result['token_mean']),float((short_hard+3*long_easy)/4),places=6)
        self.assertGreater(float(result['balanced']),float(result['token_mean']))

    def test_entity_focus_ignores_template_and_padding(self):
        tokenizer=self.tokenizer
        row=dict(conversations=[dict(role='user',content='资料写着作者是鲁迅。谁是作者？'),dict(role='assistant',content='鲁迅')])
        feature=entity_feature(row,'鲁迅',tokenizer,256)
        text=tokenizer.decode([v for v,m in zip(feature['input_ids'],feature['entity_mask']) if m])
        self.assertEqual(text,'鲁迅')
        self.assertFalse(any(m and label==-100 for m,label in zip(feature['entity_mask'],feature['labels'])))
        short=entity_feature(dict(conversations=[dict(role='user',content='作者？'),dict(role='assistant',content='老舍')]),'老舍',tokenizer,256)
        batch=collate_entities([feature,short],tokenizer.pad_token_id)
        self.assertFalse(batch['entity_mask'][batch['attention_mask']==0].any())

    def test_no_partial_fact_target(self):
        row=dict(conversations=[dict(role='user',content='很长的问题'*100),dict(role='assistant',content='鲁迅')])
        with self.assertRaises(ValueError):entity_feature(row,'鲁迅',self.tokenizer,32)

    def test_ranking_gradient_raises_gold_and_lowers_wrong_likelihood(self):
        positive=torch.tensor([2.,1.],requires_grad=True)
        negative=torch.tensor([1.,2.],requires_grad=True)
        loss=ranking_loss(positive,negative,torch.tensor([1.,0.]),.5)
        loss.backward()
        self.assertGreater(float(positive.grad[0]),0.)
        self.assertLess(float(negative.grad[0]),0.)
        self.assertEqual(float(positive.grad[1]),0.)
        self.assertEqual(float(negative.grad[1]),0.)

    def test_all_unknown_batch_has_finite_zero_ranking(self):
        p=torch.tensor([2.,3.],requires_grad=True)
        loss=ranking_loss(p,p,torch.zeros(2))
        self.assertEqual(float(loss),0.)
        loss.backward()
        self.assertTrue(torch.equal(p.grad,torch.zeros(2)))

    def test_kl_reference_detached_and_only_supervised_positions(self):
        student=torch.tensor([[[2.,0.],[0.,2.],[1.,1.]]],requires_grad=True)
        teacher=torch.zeros_like(student,requires_grad=True)
        labels=torch.tensor([[-100,0,-100]])
        loss=replay_kl(student,teacher,labels)
        self.assertGreater(float(loss),0.)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad[0,0].abs().sum()),0.)
        self.assertEqual(float(student.grad[0,1:].abs().sum()),0.)
        self.assertAlmostEqual(float(replay_kl(teacher.detach(),teacher.detach(),labels)),0.,places=6)

    def test_confidence_weights_apply_per_sample(self):
        torch.manual_seed(2)
        logits=torch.randn(2,3,4)
        labels=torch.tensor([[-100,1,2],[-100,2,3]])
        result=factual_losses(logits,labels,labels.ne(-100),torch.tensor([1.,.25]),entity_fraction=1.)
        expected=(result['entity_nll'][0]+.25*result['entity_nll'][1])/1.25
        self.assertAlmostEqual(float(result['balanced']),float(expected),places=6)

    def test_draws_resume_without_replaying_or_using_global_rng(self):
        metas=[dict(task='author',quality='externally_reviewed'),dict(task='author',quality='corpus_metadata'),dict(task='clarify_ambiguity',quality='corpus_ambiguity')]
        direct=sample_indices(metas,20,42+1000003*10)
        for step in range(1,10):sample_indices(metas,20,42+1000003*step)
        resumed=sample_indices(metas,20,42+1000003*10)
        self.assertEqual(direct,resumed)

    def test_exact_scoring_does_not_reward_multiple_guesses(self):
        self.assertTrue(correct_answer('钱钟书。',['钱锺书','钱钟书']))
        self.assertFalse(correct_answer('鲁迅或老舍',['鲁迅']))
        self.assertFalse(correct_answer('不知道',['鲁迅']))

    def test_fact_scoring_separates_format_and_checks_work_title(self):
        text='《呐喊》的作者是鲁迅。'
        self.assertFalse(correct_answer(text,['鲁迅']))
        self.assertTrue(entity_correct_answer(text,['鲁迅'],'呐喊'))
        self.assertFalse(entity_correct_answer(text,['鲁迅'],'围城'))
        self.assertFalse(entity_correct_answer('作者是鲁迅或老舍。',['鲁迅'],'呐喊'))

    def test_guard_rejects_forgetting_even_if_accuracy_rises(self):
        metrics=dict(replay_nll=2.,known_macro_em=.9,fact_entity_nll=.1,ambiguity_em=1.)
        self.assertIsNone(checkpoint_rank(metrics,1.,.15))
        self.assertIsNotNone(checkpoint_rank(metrics,1.,-1))
        metrics['replay_nll']=1.1
        self.assertEqual(checkpoint_rank(metrics,1.,.15)[0],.9)


if __name__=='__main__':unittest.main()
