# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Behavioral checks: frozen bases, GQA, merge parity, and exact resumed updates."""
import datasets  # Arrow before torch on Windows.
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import sys

import torch
from model.model_mindchen import MindChenConfig, MindChenForCausalLM
from model.lora_adapter import attach_lora, adapter_state, merged_state, load_bound_adapter, save_merged
from trainer.lora_training import ROOT, ChatData, collate, main, sha256
from transformers import AutoTokenizer


class LoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        (ROOT/'out').mkdir(exist_ok=True)
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT/'model', local_files_only=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='lora_unit_', dir=ROOT/'out')
        self.folder = Path(self.temp.name).resolve()
        assert self.folder.is_relative_to((ROOT/'out').resolve())
        self.addCleanup(self.temp.cleanup)
        torch.manual_seed(7)

    def tiny(self):
        return MindChenForCausalLM(MindChenConfig(hidden_size=32, num_hidden_layers=1, vocab_size=6400,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, intermediate_size=64,
            max_position_embeddings=512, use_moe=True, num_experts=2, moe_intermediate_size=48,
            norm_topk_prob=False))

    def base(self, model):
        path = self.folder/'base.pth'; torch.save(model.state_dict(), path)
        Path(str(path)+'.config.json').write_text(json.dumps(dict(schema='mindchen_native_config_v1',
             weight_sha256=sha256(path), model_config=model.config.to_dict())), encoding='utf8')
        return path

    def data(self):
        p = self.folder/'chat.jsonl'
        rows = [dict(conversations=[dict(role='system',content='你是 MindChen。'),dict(role='user',content=q),
                                    dict(role='assistant',content=a)]) for q,a in
                [('你叫什么名字？','我是 MindChen。'),('2 加 3 等于几？','等于 5。'),('Who are you?','I am MindChen.')]]
        p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
        return p

    def test_freeze_non_square_and_merge(self):
        model = self.tiny().eval(); base = {k:v.clone() for k,v in model.state_dict().items()}
        inputs = torch.tensor([[1,50,60,70,2]])
        reference = model(inputs).logits.detach()
        names = attach_lora(model, rank=2, alpha=4)
        self.assertTrue(any(n.endswith('v_proj') for n in names))
        self.assertEqual(model.model.layers[0].self_attn.v_proj.out_features,16)
        self.assertTrue(all('.lora_' in n for n,p in model.named_parameters() if p.requires_grad))
        torch.testing.assert_close(model(inputs).logits,reference,rtol=0,atol=0)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        for _ in range(2):
            optimizer.zero_grad(); model(inputs,labels=inputs).loss.backward(); optimizer.step()
        for name,value in base.items(): torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        self.assertTrue(any(torch.count_nonzero(v) for k,v in adapter_state(model).items() if 'lora_B' in k))
        merged = self.tiny().eval(); merged.load_state_dict(merged_state(model),strict=True)
        torch.testing.assert_close(model(inputs).logits,merged(inputs).logits,rtol=1e-4,atol=1e-5)
        with self.assertRaises(ValueError): attach_lora(model)

    def test_dataset_masks_only_assistant(self):
        data = ChatData(self.data(),self.tokenizer,self.tiny().config,max_length=128)
        item = data[0]; start = item['labels'].index(next(v for v in item['labels'] if v!=-100))
        self.assertTrue(all(v==-100 for v in item['labels'][:start]))
        decoded = self.tokenizer.decode([v for v in item['labels'] if v!=-100])
        self.assertIn('MindChen',decoded); self.assertNotIn('你叫什么名字',decoded)
        batch = collate([data[0],data[1]],self.tokenizer,'cpu')
        self.assertTrue(torch.all(batch['labels'][batch['attention_mask']==0]==-100))
        with self.assertRaisesRegex(ValueError,'No supervised'): ChatData(self.data(),self.tokenizer,self.tiny().config,4)[0]

    def test_merged_file_loads_for_further_sft(self):
        model=self.tiny();attach_lora(model,rank=2,alpha=4)
        for n,p in model.named_parameters():
            if 'lora_B' in n:p.data.fill_(.02)
        target=self.folder/'merged.pth';save_merged(model,target)
        from trainer.lora_training import load_native
        restored,_,_=load_native(target,'llm','cpu')
        self.assertFalse(restored.config.norm_topk_prob)
        ids=torch.tensor([[1,40,50,2]]);model.eval();restored.eval()
        torch.testing.assert_close(model(ids).logits,restored(ids).logits,rtol=1e-4,atol=1e-5)
        if not (ROOT/'model/model_vlm.py').exists():
            from trainer.trainer_utils import init_model
            sft, _=init_model(self.tiny().config,from_weight=str(target),device='cpu')
            self.assertFalse(sft.config.norm_topk_prob)
            sft.eval();torch.testing.assert_close(restored(ids).logits,sft(ids).logits,rtol=0,atol=0)

    def test_resume_matches_uninterrupted_and_bound_adapter(self):
        base = self.base(self.tiny()); data = self.data()
        entry = ROOT/'trainer'/('train_lora_vlm.py' if (ROOT/'model/model_vlm.py').exists() else 'train_lora_sft.py')
        # Use LLM mode in both projects; VLM has the same language config implementation.
        common = [str(entry),'--checkpoint',str(base),'--data_path',str(data),'--eval_data_path',str(data),
                  '--batch_size','2','--accumulation_steps','2','--rank','2','--alpha','4','--dropout','0.1',
                  '--max_seq_len','128','--save_interval','1','--eval_interval','1','--device','cpu','--dtype','float32']
        full, split = self.folder/'full',self.folder/'split'
        with patch.object(sys,'argv',common+['--run_dir',str(full),'--max_steps','3']): main('llm')
        with patch.object(sys,'argv',common+['--run_dir',str(split),'--max_steps','1']): main('llm')
        with patch.object(sys,'argv',common+['--resume',str(split/'resume.pth'),'--max_steps','3']): main('llm')
        a=torch.load(full/'adapter.pth',weights_only=True);b=torch.load(split/'adapter.pth',weights_only=True)
        for key in a['adapter']: torch.testing.assert_close(a['adapter'][key],b['adapter'][key],rtol=0,atol=0)
        wrong=self.folder/'wrong.pth';torch.save({'not':'the same base'},wrong)
        with self.assertRaisesRegex(ValueError,'different base'):load_bound_adapter(self.tiny(),split/'adapter.pth',wrong)
        with patch.object(sys,'argv',common+['--resume',str(split/'resume.pth'),'--max_steps','4','--rank','3']):
            with self.assertRaisesRegex(ValueError,'signature mismatch'):main('llm')

    @unittest.skipUnless((ROOT/'model/model_vlm.py').exists(),'VLM-only check')
    def test_mixed_image_batch_and_visual_merge_preservation(self):
        from model.model_vlm import MindChenVLM,VLMConfig
        from types import SimpleNamespace
        from PIL import Image
        from torch import nn
        class Vision(nn.Module):
            def __init__(self):
                super().__init__(); self.proj=nn.Linear(3,16,bias=False); nn.init.constant_(self.proj.weight,.01)
            def forward(self,pixel_values):
                return SimpleNamespace(last_hidden_state=self.proj(pixel_values.mean((2,3))).unsqueeze(1).repeat(1,4,1))
        class Processor:
            def __call__(self,images,return_tensors):return {'pixel_values':torch.ones(1,3,8,8)}
        with patch.object(MindChenVLM,'get_vision_model',side_effect=lambda p:(Vision(),Processor())):
            config=VLMConfig(hidden_size=32,num_hidden_layers=1,vocab_size=6400,num_attention_heads=4,
                num_key_value_heads=2,head_dim=8,intermediate_size=64,max_position_embeddings=512,
                image_hidden_size=16,image_token_len=4,norm_topk_prob=False)
            model=MindChenVLM(config); processor=model.processor
            p=self.data();rows=[json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
            Image.new('RGB',(8,8),'red').save(self.folder/'test.png'); rows[0]['image']='test.png'
            p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
            data=ChatData(p,self.tokenizer,config,128,'vlm',processor)
            old_fingerprint=data.fingerprint()
            Image.new('RGB',(8,8),'blue').save(self.folder/'test.png')
            self.assertNotEqual(old_fingerprint,data.fingerprint())
            batch=collate([data[0],data[1]],self.tokenizer,'cpu')
            frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('vision_proj.','vision_encoder.'))}
            names=attach_lora(model,rank=2,alpha=4)
            self.assertTrue(all(n.startswith('model.layers.') for n in names))
            optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
            optim.zero_grad();loss=model(**batch).loss;self.assertTrue(torch.isfinite(loss));loss.backward();optim.step()
            for k,v in frozen.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
            state=merged_state(model)
            self.assertTrue(any(k.startswith('vision_proj.') for k in state))
            self.assertFalse(any(k.startswith('vision_encoder.') or '.lora_' in k for k in state))
            merged=MindChenVLM(config);missing=merged.load_state_dict(state,strict=False).missing_keys
            self.assertTrue(all(k.startswith('vision_encoder.') for k in missing))
            model.eval();merged.eval();torch.testing.assert_close(model(**batch).logits,merged(**batch).logits,rtol=1e-4,atol=1e-5)
            saved=self.folder/'vlm_merged.pth';save_merged(model,saved)
            from trainer.trainer_utils import init_vlm_model
            fallback=config.to_dict();fallback['norm_topk_prob']=True
            continued,_,_=init_vlm_model(VLMConfig(**fallback),from_weight=str(saved),device='cpu',freeze_llm=1)
            self.assertFalse(continued.config.norm_topk_prob)
            continued.eval();torch.testing.assert_close(model(**batch).logits,continued(**batch).logits,rtol=1e-4,atol=1e-5)


if __name__=='__main__':unittest.main()
