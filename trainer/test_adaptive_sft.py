# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""检验目标完整性、动态权重边界、任务路由无答案泄露及原样本损失归一。"""
import json
from pathlib import Path
import unittest

import datasets
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from trainer.adaptive_sft_data import lossless_features, classify_prompt, quota_counts, prompt_of
from trainer.adaptive_sft_objective import TaskRouter, WeightController, window_loss
from trainer.sft_data import AssistantDataset, SFTCollator
from scripts.prepare_adaptive_sft import numeric_records

ROOT=Path(__file__).resolve().parent.parent


class AdaptiveSFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)

    def check_coverage(self,row,length=128,overlap=32):
        full=AssistantDataset([],self.tokenizer,2**31-1).encode(row)
        feature=lossless_features(row,self.tokenizer,length,overlap)
        targets=[]
        for window in feature['windows']:
            self.assertLessEqual(len(window['input_ids']),length)
            self.assertEqual(window['labels'][0],-100)
            for i,label in enumerate(window['labels']):
                if label!=-100:
                    original=window['start']+i
                    self.assertEqual(label,full['labels'][original])
                    targets.append(original)
        expected=[i for i,label in enumerate(full['labels']) if i>0 and label!=-100]
        self.assertEqual(targets,expected)
        self.assertEqual(len(targets),len(set(targets)))
        self.assertEqual(sum(w['eos_supervised'] for w in feature['windows']),full['eos_supervised'])
        return feature

    def test_long_answer_every_target_once_and_real_eos(self):
        feature=self.check_coverage(dict(conversations=[dict(role='user',content='写一段文字'),
            dict(role='assistant',content='春风山水明月。'*900)]))
        self.assertGreater(len(feature['windows']),10)
        self.assertTrue(feature['windowed'])

    def test_long_prompt_keeps_answer_instead_of_excluding(self):
        feature=self.check_coverage(dict(conversations=[dict(role='user',content='请阅读材料：'+'资料'*1500),
            dict(role='assistant',content='答案在这里。')]))
        self.assertGreater(feature['target_tokens'],0)
        self.assertLess(len(feature['windows']),5)

    def test_multiturn_targets_are_not_repeated(self):
        self.check_coverage(dict(conversations=[dict(role='user',content='问一'),
            dict(role='assistant',content='长答'*400),dict(role='user',content='问二'*300),
            dict(role='assistant',content='第二个回答。'*200)]))

    def test_tool_call_only_answer_and_tool_result_mask(self):
        calls=json.dumps([dict(type='function',function=dict(name='sum',arguments={'a':1,'b':2}))])
        feature=self.check_coverage(dict(conversations=[dict(role='user',content='计算1+2'),
            dict(role='assistant',content='',tool_calls=calls),dict(role='tool',content='TOOL_SECRET'),
            dict(role='assistant',content='3')]))
        text=self.tokenizer.decode([t for w in feature['windows'] for t in w['labels'] if t!=-100])
        self.assertIn('<tool_call>',text);self.assertNotIn('TOOL_SECRET',text)

    def test_prompt_router_input_excludes_answer(self):
        a=dict(conversations=[dict(role='user',content='这部书的作者是谁？'),dict(role='assistant',content='鲁迅')])
        b=dict(conversations=[dict(role='user',content='这部书的作者是谁？'),dict(role='assistant',content='完全不同的答案')])
        self.assertEqual(lossless_features(a,self.tokenizer)['prompt_ids'],lossless_features(b,self.tokenizer)['prompt_ids'])
        self.assertEqual(prompt_of(a),prompt_of(b))

    def test_padding_never_supervised(self):
        a=lossless_features(dict(conversations=[dict(role='user',content='问'),dict(role='assistant',content='答')]),self.tokenizer)
        b=lossless_features(dict(conversations=[dict(role='user',content='长问题'*30),dict(role='assistant',content='很长回答'*40)]),self.tokenizer)
        batch=SFTCollator(self.tokenizer.pad_token_id)([a['windows'][0],b['windows'][0]])
        self.assertTrue(batch['labels'][batch['attention_mask']==0].eq(-100).all())

    def test_overlap_must_provide_previous_token(self):
        with self.assertRaises(ValueError):lossless_features({},self.tokenizer,128,0)
        with self.assertRaises(ValueError):lossless_features({},self.tokenizer,128,128)

    def test_dynamic_weight_cannot_be_gamed_through_gradient(self):
        logits=torch.tensor([[10.,-10.,-10.],[-10.,10.,-10.]],requires_grad=True)
        controller=WeightController(alpha=10.,maximum=1.8,warmup=1)
        controller.router_macro_accuracy=1.
        weights,_=controller.weights(logits,torch.tensor([0,1]),torch.ones(2),torch.ones(2),10)
        self.assertFalse(weights.requires_grad)
        self.assertAlmostEqual(float(weights[0]),1.8,places=5)
        self.assertEqual(float(weights[1]),1.)

    def test_router_unreliable_falls_back_to_known_labels(self):
        controller=WeightController(alpha=.6)
        weights,details=controller.weights(torch.tensor([[-20.,20.,-20.]]),torch.tensor([0]),torch.ones(1),torch.ones(1),1000)
        self.assertEqual(details['router_mix'],0.)
        self.assertAlmostEqual(float(weights[0]),1.6,places=5)

    def test_difficulty_and_controller_resume(self):
        controller=WeightController();tasks=torch.tensor([0,1])
        controller.observe(torch.tensor([2.,3.]),tasks)
        controller.observe(torch.tensor([8.,3.]),tasks)
        other=WeightController();other.load_state_dict(controller.state_dict())
        args=(torch.tensor([[2.,0.,0.],[0.,2.,0.]]),tasks,torch.ones(2),torch.ones(2),101)
        self.assertTrue(torch.equal(controller.weights(*args)[0],other.weights(*args)[0]))
        self.assertGreater(controller.weights(*args)[1]['difficulty'],1.)

    def test_long_source_not_upweighted_by_more_windows(self):
        logits=torch.tensor([[[0.,2.],[0.,0.],[0.,0.]],[[2.,0.],[2.,0.],[0.,0.]],[[2.,0.],[2.,0.],[0.,0.]]],requires_grad=True)
        labels=torch.tensor([[-100,0,-100],[-100,0,0],[-100,0,0]])
        counts=torch.tensor([1,4]);weights=torch.tensor([2.,1.]);owners=torch.tensor([0,1,1])
        loss,_,_=window_loss(logits,labels,owners,counts,weights)
        short=F.cross_entropy(torch.tensor([[0.,2.]]),torch.tensor([0]))
        long=F.cross_entropy(torch.tensor([[2.,0.]]),torch.tensor([0]))
        self.assertAlmostEqual(float(loss.detach()),float((2*short+long)/3),places=6)
        first=window_loss(logits[:1],labels[:1],owners[:1],counts,weights)[0]
        rest=window_loss(logits[1:],labels[1:],owners[1:],counts,weights)[0]
        self.assertAlmostEqual(float((first+rest).detach()),float(loss.detach()),places=6)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_router_reads_last_real_prompt_position(self):
        router=TaskRouter(4)
        hidden=torch.randn(2,5,4,requires_grad=True)
        mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,1]])
        router(hidden,mask).sum().backward()
        self.assertEqual(float(hidden.grad[0,2:].abs().sum()),0.)
        self.assertGreater(float(hidden.grad[0,1].abs().sum()),0.)

    def test_task_rules_conservative_for_mixed_questions(self):
        self.assertEqual(classify_prompt('请背诵以下作品的原文，只输出正文。')[0],0)
        self.assertEqual(classify_prompt('创作一首关于春天的诗')[0],1)
        self.assertEqual(classify_prompt('作者是谁？再赏析这篇作品。')[0],2)
        self.assertEqual(classify_prompt('你好')[0],2)

    def test_quota_is_complete_and_never_empty(self):
        for count in (3,6,12,24):
            quota=quota_counts(count,[.35,.30,.35])
            self.assertEqual(sum(quota),count);self.assertTrue(all(n>0 for n in quota))
        self.assertEqual(quota_counts(100,[.35,.30,.35]),[35,30,35])

    def test_numeric_groups_are_disjoint_and_deterministic(self):
        left=list(numeric_records(20,5));right=list(numeric_records(20,5))
        self.assertEqual(left,right)
        train={group for split,_,group in left if split=='train'}
        valid={group for split,_,group in left if split=='valid'}
        self.assertFalse(train&valid);self.assertEqual(len(train),20);self.assertEqual(len(valid),5)

    def test_defaults_use_requested_pretrain_and_full_sampling_budget(self):
        from trainer.train_adaptive_sft import parse_args
        args=parse_args([])
        self.assertEqual(Path(args.init_weight_path).name,'pretrain_poetry_gujilab_768_moe_best.pth')
        self.assertIn('continue_pretrain',args.init_weight_path)
        self.assertEqual(args.max_updates,0);self.assertEqual(args.examples_per_update,64)
        self.assertIn('sft_t2t_with_poetry_clean_v1',args.manifest)

    def test_router_microbatches_preserve_update_normalization(self):
        logits=torch.randn(7,3,requires_grad=True);tasks=torch.tensor([0,1,2,0,1,2,0])
        confidence=torch.tensor([1.,.85,0.,.6,.85,.35,1.])
        expected=(F.cross_entropy(logits,tasks,reduction='none')*confidence).sum()/confidence.sum()
        actual=sum((F.cross_entropy(logits[i:i+2],tasks[i:i+2],reduction='none')*confidence[i:i+2]).sum()/confidence.sum() for i in range(0,7,2))
        self.assertAlmostEqual(float(expected.detach()),float(actual.detach()),places=6)
        grad1=torch.autograd.grad(expected,logits,retain_graph=True)[0]
        grad2=torch.autograd.grad(actual,logits)[0]
        self.assertTrue(torch.allclose(grad1,grad2,atol=1e-7))


if __name__=='__main__':unittest.main()
