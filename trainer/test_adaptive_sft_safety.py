# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""审计缺陷的回归测试：推理路径、评分、风险隔离、MoE 梯度及 padding。"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import random
import unittest

import datasets
import torch
from transformers import AutoTokenizer

from eval_llm import checkpoint_path, resolved_prompt_style
from model.model_mindchen import MindChenConfig, MOEFeedForward
from trainer.adaptive_sft_objective import WeightController
from trainer.adaptive_sft_safety import quarantine_reason, training_signature, stratified_indices
from trainer.train_adaptive_sft import answer_is_correct, materialize, creative_probes

ROOT=Path(__file__).resolve().parent.parent


def row(prompt,answer):
    return dict(conversations=[dict(role='user',content=prompt),dict(role='assistant',content=answer)])


class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)

    def test_explicit_checkpoint_controls_file_and_auto_template(self):
        args=SimpleNamespace(checkpoint='out/custom/adaptive_768_moe_best.pth',
            save_dir='unused',weight='pretrain',hidden_size=768,use_moe=1,
            prompt_style='auto',load_from='model')
        self.assertEqual(checkpoint_path(args),ROOT/'out/custom/adaptive_768_moe_best.pth')
        self.assertEqual(resolved_prompt_style(args),'chat')
        args.checkpoint='';args.save_dir='out/changed';args.weight='pretrain'
        self.assertEqual(checkpoint_path(args),ROOT/'out/changed/pretrain_768_moe.pth')
        self.assertEqual(resolved_prompt_style(args),'raw')

    def test_author_alias_does_not_make_other_tasks_loose(self):
        self.assertTrue(answer_is_correct('钱钟书','钱锺书',aliases=['钱锺书','钱钟书']))
        self.assertFalse(answer_is_correct('钱钟书或老舍','钱锺书',aliases=['钱锺书','钱钟书']))
        self.assertFalse(answer_is_correct('[1 2]','[1,2]',{'kind':'list_operation'}))
        self.assertFalse(answer_is_correct('1230','12:30',{'kind':'clock'}))

    def test_missing_evidence_quarantine_preserves_clarification_and_context(self):
        self.assertEqual(quarantine_reason(row('L这首诗的作者是谁？','唐代诗人李白。')),'missing_poem_clue')
        self.assertIsNone(quarantine_reason(row('这首诗的作者是谁？','请提供诗句。')))
        self.assertIsNone(quarantine_reason(row('《静夜思》的作者是谁？','李白')))
        multi=row('这首诗的作者是谁？','李白')
        multi['conversations'][:0]=[dict(role='user',content='静夜思'),dict(role='assistant',content='收到')]
        self.assertIsNone(quarantine_reason(multi))
        self.assertEqual(quarantine_reason(row('选择正确答案。\nA. $','答案是A')),'empty_last_option')

    def test_full_context_replaces_row_without_partial_supervision(self):
        rows=[row('材料'*300,'答案'),row('问题','短答案')]
        class Corpus:
            indices={'train_base_general':range(2)}
            def read(self,key,index):return rows[index],dict(task=2,index=index,channel=key)
        args=SimpleNamespace(seed=42,max_seq_len=128,context_overlap=32,context_policy='full_context')
        draws=[('train_base_general',0)];counts=Counter()
        samples=materialize(Corpus(),draws,self.tokenizer,args,counts)
        self.assertEqual(draws,[('train_base_general',1)])
        self.assertGreater(counts['context_exceeds_window'],0)
        self.assertLessEqual(samples[0]['full_tokens'],128)
        self.assertEqual(samples[0]['target_retention'],1.)

    def test_stratified_generation_covers_all_kinds(self):
        rows=[dict(task_meta=dict(kind=k)) for k in ('a','a','a','a','b','c')]
        corpus=SimpleNamespace(read=lambda key,i:(rows[i],{}))
        indices=stratified_indices(corpus,'valid_verified_fact',range(6),3,random.Random(42))
        self.assertEqual({rows[i]['task_meta']['kind'] for i in indices},{'a','b','c'})
        self.assertEqual(len(set(indices)),3)

    def test_confident_router_mistake_cannot_erase_fact_boost(self):
        c=WeightController(warmup=1);c.router_macro_accuracy=1.
        weights,_=c.weights(torch.tensor([[-30.,30.,-30.],[-30.,30.,-30.]]),
            torch.tensor([0,1]),torch.ones(2),torch.ones(2),100)
        self.assertGreaterEqual(float(weights[0]),1.29)
        self.assertEqual(float(weights[1]),1.)

    def test_signature_detects_chat_template_change(self):
        manifest=ROOT/'dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready/manifest.json'
        config=MindChenConfig(hidden_size=32,num_hidden_layers=1)
        one=training_signature(ROOT,self.tokenizer,config,manifest)
        old=self.tokenizer.chat_template
        try:
            self.tokenizer.chat_template=old+"{{ 'EXTRA' }}"
            two=training_signature(ROOT,self.tokenizer,config,manifest)
        finally:self.tokenizer.chat_template=old
        self.assertNotEqual(one['tokenizer'],two['tokenizer'])
        self.assertEqual(one['code'],two['code'])

    def test_creative_probes_are_reproducible_and_preserve_training_rng(self):
        class Model:
            def generate(self,inputs,**kwargs):
                return torch.cat([inputs,torch.randint(10,500,(1,4))],dim=1)
        args=SimpleNamespace(device='cpu',dtype='float32',seed=42,max_seq_len=512,
            open_generation_examples=1,open_generation_samples=2,open_generation_tokens=4)
        before=torch.random.get_rng_state().clone()
        one,_=creative_probes(Model(),self.tokenizer,args)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        two,_=creative_probes(Model(),self.tokenizer,args)
        self.assertEqual(one,two)
        self.assertEqual(len(one),2)
        self.assertNotEqual(one[0]['seed'],one[1]['seed'])

    def test_padding_does_not_change_masked_auxiliary(self):
        torch.manual_seed(42)
        model=MOEFeedForward(MindChenConfig(hidden_size=32,num_hidden_layers=1,use_moe=True,
            intermediate_size=64,moe_intermediate_size=32,moe_mask_padding=True)).train()
        x=torch.randn(1,4,32)
        baseline=model(x,torch.ones(1,4));aux=model.aux_loss.detach().clone()
        padded=torch.cat([x,torch.randn(1,40,32)],dim=1)
        out=model(padded,torch.tensor([[1]*4+[0]*40]))
        self.assertTrue(torch.allclose(baseline,out[:,:4],atol=1e-6))
        self.assertTrue(torch.allclose(aux,model.aux_loss,atol=1e-8))
        model(padded,torch.zeros(1,44))
        self.assertEqual(float(model.aux_loss.detach()),0.)

    def test_experimental_router_preserves_forward_and_has_lm_gradient(self):
        torch.manual_seed(8)
        config=MindChenConfig(hidden_size=32,num_hidden_layers=1,use_moe=True,
            intermediate_size=64,moe_intermediate_size=32,router_aux_loss_coef=0.)
        original=MOEFeedForward(config).train()
        experimental=MOEFeedForward(MindChenConfig(**{**config.to_dict(),
            'moe_router_gradient':'straight_through'})).train()
        experimental.load_state_dict(original.state_dict(),strict=True)
        x=torch.randn(2,7,32)
        a=original(x);b=experimental(x)
        self.assertTrue(torch.equal(a,b))
        a.square().mean().backward();b.square().mean().backward()
        self.assertLess(float(original.gate.weight.grad.norm()),1e-7)
        self.assertGreater(float(experimental.gate.weight.grad.norm()),1e-5)
        experimental.eval();self.assertTrue(torch.equal(experimental(x),a.detach()))


if __name__=='__main__':unittest.main()
