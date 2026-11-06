# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""概率门控的真实梯度、参数兼容性、配置绑定和两种推理入口回归。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import datasets
import torch
from model.model_mindchen import MindChenConfig, MindChenForCausalLM, MOEFeedForward
from model.checkpoint_config import (write_checkpoint_config,load_checkpoint_config,sidecar_path)
from trainer.train_adaptive_sft import parse_args,require_checkpoint_format


class ProbabilityGateTests(unittest.TestCase):
    def small_config(self,normalize=False):
        return MindChenConfig(hidden_size=32,num_hidden_layers=1,use_moe=True,
            intermediate_size=64,moe_intermediate_size=32,norm_topk_prob=normalize,
            router_aux_loss_coef=0.,max_position_embeddings=512)

    def test_new_training_defaults_to_probability_and_old_model_default_stays_compatible(self):
        self.assertEqual(parse_args([]).norm_topk_prob,0)
        self.assertEqual(parse_args([]).moe_router_gradient,'legacy')
        self.assertTrue(MindChenConfig().norm_topk_prob)
        with self.assertRaises(ValueError):
            require_checkpoint_format(dict(format_version='adaptive_sft_safety_v1'))

    def test_raw_probability_scales_selected_expert_and_gate_has_true_gradient(self):
        torch.manual_seed(7)
        model=MOEFeedForward(self.small_config()).train()
        x=torch.randn(1,5,32)
        actual=model(x)
        scores=model.gate(x).softmax(-1)
        probabilities,indices=scores.max(-1)
        expected=torch.stack([model.experts[int(indices[0,i])](x[0,i])*probabilities[0,i]
                              for i in range(x.shape[1])]).unsqueeze(0)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7))
        actual.square().mean().backward()
        self.assertGreater(float(model.gate.weight.grad.norm()),1e-5)

    def test_lm_gradient_reaches_gate_with_auxiliary_disabled(self):
        torch.manual_seed(4)
        model=MindChenForCausalLM(self.small_config()).train()
        ids=torch.randint(5,100,(1,12))
        out=model(ids,labels=ids)
        self.assertEqual(float(out.aux_loss.detach()),0.)
        out.loss.backward()
        self.assertGreater(float(model.model.layers[0].mlp.gate.weight.grad.norm()),1e-6)

    def test_probability_and_normalized_modes_share_state_shapes_and_strict_load(self):
        old=MindChenForCausalLM(self.small_config(True))
        new=MindChenForCausalLM(self.small_config(False))
        result=new.load_state_dict(old.state_dict(),strict=True)
        self.assertFalse(result.missing_keys or result.unexpected_keys)
        self.assertEqual({k:tuple(v.shape) for k,v in old.state_dict().items()},
                         {k:tuple(v.shape) for k,v in new.state_dict().items()})

    def test_bound_config_survives_roundtrip_and_rejects_conflicting_override(self):
        with tempfile.TemporaryDirectory() as d:
            weight=Path(d)/'adaptive_32_moe.pth';weight.write_bytes(b'weights')
            write_checkpoint_config(weight,self.small_config(False))
            config,info=load_checkpoint_config(weight,fallback=dict(hidden_size=768))
            self.assertFalse(config.norm_topk_prob)
            self.assertEqual(config.hidden_size,32)
            self.assertTrue(info['weight_bound'])
            with self.assertRaisesRegex(ValueError,'norm_topk_prob'):
                load_checkpoint_config(weight,norm_topk_prob=True)
            weight.write_bytes(b'different weights')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                load_checkpoint_config(weight)

    def test_legacy_fallback_does_not_inherit_unrelated_folder_config(self):
        with tempfile.TemporaryDirectory() as d:
            weight=Path(d)/'old.pth';weight.write_bytes(b'old')
            (weight.parent/'model_config.json').write_text(json.dumps(self.small_config(False).to_dict()))
            cfg,info=load_checkpoint_config(weight)
            self.assertTrue(cfg.norm_topk_prob)
            self.assertEqual(info['source'],'legacy_fallback')
            cfg,_=load_checkpoint_config(weight,norm_topk_prob=0)
            self.assertFalse(cfg.norm_topk_prob)

    def test_bound_config_rejects_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as d:
            weight=Path(d)/'model.pth';weight.write_bytes(b'a')
            write_checkpoint_config(weight,self.small_config())
            with self.assertRaisesRegex(ValueError,'hidden_size'):
                load_checkpoint_config(weight,expected_architecture=dict(hidden_size=768))

    def test_actual_chat_and_quality_loaders_use_saved_probability_config(self):
        from eval_llm import init_model
        import sys
        eval_dir=Path(__file__).resolve().parent.parent/'eval'
        sys.path.insert(0,str(eval_dir))
        from eval_mindchen_quality import load_model
        with tempfile.TemporaryDirectory() as d:
            weight=Path(d)/'adaptive_32_moe_best.pth'
            original=MindChenForCausalLM(self.small_config()).eval()
            torch.save(original.state_dict(),weight)
            write_checkpoint_config(weight,original.config)
            args=SimpleNamespace(load_from='model',checkpoint=str(weight),save_dir=d,
                hidden_size=768,num_hidden_layers=8,use_moe=1,inference_rope_scaling=False,
                lora_weight='None',device='cpu',norm_topk_prob=None,model_config='')
            chat,_=init_model(args)
            args.hidden_size=args.num_hidden_layers=args.use_moe=None
            quality,_,_=load_model(args)
            self.assertFalse(chat.config.norm_topk_prob or quality.config.norm_topk_prob)
            ids=torch.randint(5,100,(1,9))
            with torch.inference_mode():
                expected=original(ids).logits
                self.assertTrue(torch.equal(chat(ids).logits,expected))
                self.assertTrue(torch.equal(quality(ids).logits,expected))


if __name__=='__main__':unittest.main()
