# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Behavioral checks for the selected copy's CISPO objective and native resume."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import datasets
import torch
from transformers import AutoTokenizer
from model.model_mindchen import MindChenConfig, MindChenForCausalLM
from model.checkpoint_config import load_checkpoint_config
from trainer import grpo_copy_runtime as runtime

ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('selected_grpo_copy',ROOT/'trainer/train_grpo - 副本.py')
selected=importlib.util.module_from_spec(spec);spec.loader.exec_module(selected)


class FakeReward:
    def __init__(self,*args,**kwargs):self.model=torch.nn.Linear(1,1)
    def get_score(self,messages,response):return (sum(map(ord,response))%31)/10.-1.5


class CopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
        cls.tokenizer.padding_side='left';torch.set_num_threads(2)
        (ROOT/'out/grpo_copy').mkdir(exist_ok=True)

    def test_cispo_upper_clip_is_detached_with_expected_corrective_gradient(self):
        current=torch.tensor([[-1.],[-1.]],requires_grad=True)
        old=current.detach()-torch.tensor([[math.log(6.)],[math.log(.5)]])
        args=SimpleNamespace(loss_type='cispo',epsilon_high=5.,epsilon=.2,beta=.1)
        loss,kl=selected.policy_objective(current,old,current.detach(),torch.tensor([1.,-1.]),torch.ones_like(current),args)
        loss.backward()
        torch.testing.assert_close(current.grad,torch.tensor([[-2.5],[.25]]))
        self.assertEqual(float(kl),0.)

    def test_grpo_clips_improving_out_of_bounds_ratios(self):
        current=torch.tensor([[-1.],[-1.]],requires_grad=True)
        old=current.detach()-torch.tensor([[math.log(6.)],[math.log(.5)]])
        args=SimpleNamespace(loss_type='grpo',epsilon_high=5.,epsilon=.2,beta=0.)
        loss,_=selected.policy_objective(current,old,current.detach(),torch.tensor([1.,-1.]),torch.ones_like(current),args)
        loss.backward()
        self.assertTrue(current.grad.eq(0).all())

    def test_original_reward_including_think_bonus_and_scorer_is_preserved(self):
        selected.args=SimpleNamespace(device='cpu',num_generations=2)
        prompt='<|im_start|>user\nExplain something.<|im_end|>\n'
        response='A clear and complete answer to the question.'
        thinking='abcdefghijklmnopqrstuvwxyz'+'</think>'+response
        class Zero:
            def get_score(self,messages,response):return 0.
        rewards=selected.calculate_rewards([prompt],[response,thinking],Zero())
        self.assertAlmostEqual(float(rewards[0]),.5)
        self.assertAlmostEqual(float(rewards[1]),1.75)

    def test_completion_eos_padding_is_excluded(self):
        mask=runtime.completion_mask(torch.tensor([[8,2,2,2],[8,9,10,11]]),2)
        self.assertEqual(mask.tolist(),[[True,True,False,False],[True]*4])

    def test_sampling_and_training_probability_are_aligned(self):
        config=MindChenConfig(hidden_size=32,num_hidden_layers=1,intermediate_size=64,
            use_moe=True,moe_intermediate_size=32,max_position_embeddings=512,norm_topk_prob=False)
        model=MindChenForCausalLM(config).eval()
        args=SimpleNamespace(device='cpu',num_generations=2,max_gen_len=4,temperature=.8,dtype='float32')
        prompt=self.tokenizer.apply_chat_template([dict(role='user',content='Hello')],tokenize=False,
            add_generation_prompt=True,open_thinking=False)
        with patch.object(model,'generate',wraps=model.generate) as generate:
            ids,mask,valid,_,_=runtime.rollout(model,self.tokenizer,[prompt],args)
            self.assertEqual(generate.call_args.kwargs['top_k'],0)
            self.assertEqual(generate.call_args.kwargs['top_p'],1.)
        with torch.no_grad():old=runtime.logps(model,ids,mask,valid.size(1),args)
        model.train()
        current,aux=runtime.logps(model,ids,mask,valid.size(1),args,with_aux=True)
        torch.testing.assert_close(old,current.detach(),atol=1e-5,rtol=1e-5)
        self.assertGreater(float(aux.detach()),0.)

    def write_data(self,folder):
        path=folder/'rlaif.jsonl'
        rows=[dict(conversations=[dict(role='user',content=f'Explain number {i}.'),
                                  dict(role='assistant',content='REFERENCE ANSWER MUST NOT LEAK')]) for i in range(100)]
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        return path

    def test_prompt_quarantine_dedup_and_rng_are_deterministic(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out/grpo_copy') as tmp:
            folder=Path(tmp);path=self.write_data(folder)
            with path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(conversations=[dict(role='user',content='Explain number 0.'),
                    dict(role='assistant',content='one'),dict(role='assistant',content='two')]))+'\n')
                stream.write(json.dumps(dict(conversations=[dict(role='user',content='corrupt\ufffd')]))+'\n')
            digest=runtime.file_digest(path)
            args=SimpleNamespace(validation_fraction=.05,seed=42,max_seq_len=128,thinking_ratio=.9)
            data=runtime.CopyPrompts(path,self.tokenizer,args)
            self.assertEqual(data.duplicates,1);self.assertEqual(data.removed_tails,1)
            self.assertEqual(len(data.rejected),1)
            self.assertFalse({r['id'] for r in data.train}&{r['id'] for r in data.valid})
            random.seed(10);prior=random.getstate()
            prompt=data.prompt(data.train[0],epoch=3)
            self.assertEqual(prompt,data.prompt(data.train[0],epoch=3))
            self.assertEqual(random.getstate(),prior)
            self.assertNotIn('REFERENCE ANSWER',prompt)
            self.assertEqual(runtime.file_digest(path),digest)

    def test_real_native_optimizer_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out/grpo_copy') as tmp:
            folder=Path(tmp);source=folder/'source';source.mkdir()
            path=self.write_data(folder)
            torch.manual_seed(17)
            model=MindChenForCausalLM(MindChenConfig(hidden_size=32,num_hidden_layers=1,use_moe=True,
                intermediate_size=64,moe_intermediate_size=32,max_position_embeddings=512,norm_topk_prob=False))
            weight=source/'tiny.pth';torch.save(model.state_dict(),weight)
            runtime.write_checkpoint_config(weight,model.config);original=runtime.file_digest(weight)
            common=['--checkpoint',str(weight),'--save_dir',str(folder/'runs'),'--data_path',str(path),
                '--device','cpu','--ref_device','cpu','--reward_device','cpu','--dtype','float32','--cpu_threads','2',
                '--num_generations','2','--max_seq_len','128','--max_gen_len','4','--accumulation_steps','2',
                '--max_steps','2','--save_interval','1','--eval_interval','1','--validation_samples','1']
            with patch.object(runtime,'ensure_reward_model',return_value=ROOT/'model'), \
                 patch.object(runtime,'LMForRewardModel',FakeReward),contextlib.redirect_stdout(io.StringIO()):
                runtime.main(selected,common+['--run_name','full'])
                runtime.main(selected,common+['--run_name','split','--stop_after_steps','1'])
                runtime.main(selected,['--resume',str(folder/'runs/split/resume.pth'),
                    '--device','cpu','--ref_device','cpu','--reward_device','cpu'])
            full=torch.load(folder/'runs/full/resume.pth',map_location='cpu',weights_only=True)
            split=torch.load(folder/'runs/split/resume.pth',map_location='cpu',weights_only=True)
            self.assertEqual(full['progress'],split['progress'])
            self.assertEqual(split['progress']['update'],2)
            self.assertTrue(all(torch.equal(value,split['model'][key]) for key,value in full['model'].items()))
            self.assertTrue(split['optimizer']['state'])
            self.assertEqual(full['scheduler'],split['scheduler'])
            self.assertEqual(runtime.file_digest(weight),original)
            config,_=load_checkpoint_config(folder/'runs/split/grpo_copy_32_moe.pth')
            self.assertFalse(config.norm_topk_prob)
            self.assertTrue(all(t.dtype==torch.float32 for t in split['model'].values() if t.is_floating_point()))
            with self.assertRaisesRegex(ValueError,'learning_rate'):
                runtime.main(selected,['--resume',str(folder/'runs/split/resume.pth'),'--learning_rate','.01'])


if __name__=='__main__':unittest.main()
