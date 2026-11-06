# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Curriculum quality gates, supervised replay, and exact warmup resume."""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import datasets
import torch
from transformers import AutoTokenizer
from model.model_mindchen import MindChenConfig, MindChenForCausalLM
from trainer.grpo_resources import write_checkpoint_config
from trainer.grpo_replay import SupervisedReplay
from trainer.sft_data import AssistantDataset
from trainer.warmup_grpo_format import balanced_probes, quality_gate, repair_pair, main, ROOT


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
        torch.set_num_threads(2)

    def rows(self, prefix='train', per_shape=1):
        output=[]
        for lines,width in ((4,5),(4,7),(8,5),(8,7)):
            for i in range(per_shape):
                body=''.join(chr(0x4e00+(n+i*64)%1000) for n in range(lines*width))
                spec=dict(lines=lines,width=width,reference_body=body,preserve_chars=True,
                          punctuation=True,linebreaks=True,task='format')
                output.append(repair_pair(dict(id=f'{prefix}_{lines}_{width}_{i}',poetry_spec=spec,
                    messages=[dict(role='user',content=f'请为诗文恢复格式，共{lines}句，每句{width}字。'+body)])))
        return output

    def general(self):
        return [dict(conversations=[dict(role='user',content=f'Explain number {i}.'),
                    dict(role='assistant',content=f'The number is {i}.')]) for i in range(8)]

    def test_probe_selection_is_balanced_unique_and_deterministic(self):
        rows=self.rows(per_shape=9)
        probes=balanced_probes(rows,32)
        self.assertEqual(len({r['id'] for r in probes}),32)
        for shape in ((4,5),(4,7),(8,5),(8,7)):
            self.assertEqual(sum((r['spec']['lines'],r['spec']['width'])==shape for r in probes),8)
        self.assertEqual([r['id'] for r in probes],[r['id'] for r in balanced_probes(rows[::-1],32)])
        with self.assertRaises(ValueError):balanced_probes(rows,5)

    def test_gate_needs_restore_success_and_general_retention(self):
        baseline=dict(general_nll=1.)
        metrics=dict(format_pass_rate=.5,restore_exact_rate=.25,general_nll=1.05)
        self.assertTrue(quality_gate(metrics,baseline))
        for field,value in [('restore_exact_rate',0.),('format_pass_rate',.25),('general_nll',1.2)]:
            self.assertFalse(quality_gate(dict(metrics,**{field:value}),baseline))

    def test_replay_learns_and_sampling_does_not_consume_global_rng(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as tmp:
            path=Path(tmp)/'train.jsonl'
            rows=self.rows()+self.general()
            path.write_text(''.join(json.dumps(dict(r,split='train'),ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            replay=SupervisedReplay(path,self.tokenizer,512)
            random.seed(20);before=random.getstate()
            self.assertEqual(replay.examples(18),replay.examples(18))
            self.assertEqual(random.getstate(),before)
            model=MindChenForCausalLM(MindChenConfig(hidden_size=32,num_hidden_layers=1,
                intermediate_size=64,max_position_embeddings=512)).eval()
            optimizer=torch.optim.SGD(model.parameters(),lr=.01)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            prior=model.lm_head.weight.detach().clone()
            nll=replay.backward(model,scaler,'cpu',contextlib.nullcontext,18,.1)
            self.assertGreater(nll,0.)
            optimizer.step()
            self.assertFalse(torch.equal(prior,model.lm_head.weight))
            invalid=dict(rows[0],split='valid')
            path.write_text(json.dumps(invalid,ensure_ascii=False)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'training'):SupervisedReplay(path,self.tokenizer,512)

    def test_warmup_optimizer_resume_is_exact(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as tmp:
            base=Path(tmp);source=base/'source';source.mkdir()
            config=MindChenConfig(hidden_size=32,num_hidden_layers=1,use_moe=True,
                intermediate_size=64,moe_intermediate_size=32,max_position_embeddings=1024,norm_topk_prob=False)
            torch.manual_seed(51)
            weight=source/'tiny.pth';torch.save(MindChenForCausalLM(config).state_dict(),weight)
            write_checkpoint_config(weight,config)
            poems={s:self.rows(s) for s in ('train','valid')}
            general_rows={s:self.general() for s in ('train','valid')}
            encoder=AssistantDataset([],self.tokenizer,768)
            features={s:[encoder.encode(r) for r in poems[s]] for s in poems}
            general={s:[encoder.encode(r) for r in general_rows[s]] for s in poems}
            measured=dict(poetry_nll=1.,general_nll=1.,format_pass_rate=.5,restore_exact_rate=.5,predictions=[])
            common=['--checkpoint',str(weight),'--archive_workspace',str(base),'--updates','2',
                    '--device','cpu','--validation_probes','4','--eval_interval','1','--save_interval','1']
            with patch('trainer.warmup_grpo_format.build_data',return_value=(poems,features,general,general_rows)), \
                 patch('trainer.warmup_grpo_format.measure',return_value=measured),contextlib.redirect_stdout(io.StringIO()):
                main(common+['--run_dir',str(base/'full')])
                main(common+['--run_dir',str(base/'split'),'--stop_after_steps','1'])
                main(['--resume',str(base/'split/resume.pth'),'--device','cpu'])
            full=torch.load(base/'full/resume.pth',weights_only=True)
            split=torch.load(base/'split/resume.pth',weights_only=True)
            self.assertEqual(full['progress'],split['progress'])
            self.assertEqual(split['progress']['update'],2)
            self.assertTrue(all(torch.equal(value,split['model'][key]) for key,value in full['model'].items()))
            self.assertTrue(json.loads((base/'split/summary.json').read_text())['quality_gate_passed'])
