# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""普通 SFT 身份提示接入自适应训练的回归测试。"""
import copy
from types import SimpleNamespace
import random
import unittest

import datasets
from transformers import AutoTokenizer
from trainer.adaptive_sft_data import prepare_sft_row
from trainer.train_adaptive_sft import ROOT, materialize


def row(prompt='你好', answer='你好'):
    return dict(conversations=[dict(role='user',content=prompt),dict(role='assistant',content=answer)])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
        cls.args=SimpleNamespace(seed=42,max_seq_len=512,context_overlap=128,context_policy='full_context',add_system_ratio=1.)

    def test_system_is_deterministic_without_global_rng_or_input_mutation(self):
        original=row();before=copy.deepcopy(original);state=random.getstate()
        a=prepare_sft_row(original,42,1.)
        self.assertEqual(a,prepare_sft_row(original,42,1.))
        self.assertEqual(original,before);self.assertEqual(state,random.getstate())
        self.assertIn('chenfangyi',a['conversations'][0]['content'])
        self.assertEqual(a['conversations'][-1]['content'],before['conversations'][-1]['content'])

    def test_custom_system_tool_and_old_identity_answer_are_preserved(self):
        examples=[dict(conversations=[dict(role='system',content='只输出JSON'),*row()['conversations']]),
            dict(conversations=[dict(role='user',content='查天气'),dict(role='assistant',content='',tool_calls='[]')]),
            row('你是谁','我是mindchen，由chen开发。')]
        for value in examples:self.assertIs(prepare_sft_row(value,42,1.),value)

    def test_training_integration_and_validation_unchanged(self):
        source=row('你好','回答正文')
        class Corpus:
            indices={'train_base_general':range(1),'valid_base_general':range(1)}
            def read(self,key,index):return source,dict(task=2,channel=key,index=index)
        train=materialize(Corpus(),[('train_base_general',0)],self.tokenizer,self.args)[0]
        valid=materialize(Corpus(),[('valid_base_general',0)],self.tokenizer,self.args)[0]
        self.assertEqual(train['row']['conversations'][0]['role'],'system')
        self.assertEqual(valid['row'],source)
        learned=self.tokenizer.decode([t for w in train['windows'] for t in w['labels'] if t!=-100])
        self.assertIn('回答正文',learned);self.assertNotIn('chenfangyi',learned)
        self.assertEqual(train['target_retention'],1.)


if __name__=='__main__':unittest.main()
