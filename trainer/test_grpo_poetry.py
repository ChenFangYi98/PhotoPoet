# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""不修改 SFT 来源的快照、混合切分、奖励和真实梯度/断点回归。"""
import contextlib
import ast
import io
import json
import os
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import datasets  # noqa: F401
import torch
from transformers import AutoTokenizer
from model.model_mindchen import MindChenConfig, MindChenForCausalLM
from model.checkpoint_config import write_checkpoint_config, load_checkpoint_config, sidecar_path
from trainer.grpo_resources import ROOT, protect_source, snapshot_source, file_digest, anchor_read
from trainer.grpo_poetry_data import (DEFAULT_DATA, MixedPromptDataset, make_rows, THEMES)
from trainer.grpo_poetry_reward import (poetry_rule_score, rep_penalty, calculate_rewards,
    infer_spec, resolve_poetry_spec, legacy_general_rule_score)
from trainer.train_grpo import completion_mask, completion_ended, group_advantages, policy_objective, token_logps, main
from trainer.grpo_poetry_judge import LocalPoetryJudge
from trainer.grpo_control import StopController, StopRequested


class FakeContentReward:
    def __init__(self, *args, **kwargs):
        pass

    def get_score(self, messages, response):
        return sum(ord(c) for c in response) % 101 / 20 - 2.5


class PoetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'model'), local_files_only=True)
        cls.valid = '春风入小院，\n细雨润新芽。\n燕语惊幽梦，\n晨光照落花。'
        cls.messages = [dict(role='user', content='请写一首五言绝句，一句一行。')]
        cls.spec = dict(lines=4, width=5, linebreaks=True, punctuation=True, rhyme=True)

    def test_format_preference_and_no_repeat_hacking(self):
        good, details = poetry_rule_score(self.messages, self.valid, self.spec)
        self.assertEqual(details['format_exact'], 1.)
        for bad in ['', '月月月月月，\n月月月月月。\n月月月月月，\n月月月月月。',
                    self.valid.replace('小院', '院'), '下面是诗歌：\n'+self.valid,
                    self.valid.replace('\n', ''), self.valid+'\n这首诗描写了美丽的春天。']:
            score, _ = poetry_rule_score(self.messages, bad, self.spec)
            self.assertGreater(good, score, repr(bad))
        self.assertGreater(rep_penalty('春风吹山月春风吹山月春风吹山月'), 0)

    def test_general_answer_has_no_poem_constraints(self):
        _, details = poetry_rule_score([dict(role='user', content='什么是机器学习？')], '机器学习是一种从数据学习规律的方法。')
        self.assertNotIn('line_width', details)

    def test_requested_form_and_explicit_width(self):
        cases = [('我要写一个七言律诗', 8, 7), ('帮我写首五律', 8, 5),
                 ('写一首七言绝句', 4, 7), ('请创作五言绝句', 4, 5),
                 ('写一首诗，共四句，每句恰好七个汉字', 4, 7),
                 ('写一首诗，共8句，每句5字', 8, 5),
                 ('写一首7言四句藏头诗，每句7个汉字', 4, 7),
                 ('以“奉和御製重陽五七言詩其一”为题意，创作一首五言律诗，共8句，每句5字', 8, 5)]
        for text, lines, width in cases:
            with self.subTest(text=text):
                spec = infer_spec([dict(role='user', content=text)])
                self.assertEqual((spec['lines'], spec['width']), (lines, width))
        for text in ['解释七言律诗的含义', '七言律诗的写作特点是什么？']:
            self.assertFalse(infer_spec([dict(role='user', content=text)]))

    def test_conflicting_prompt_and_annotation_rejected(self):
        for text, spec in [('请写七言律诗，每句5字', None),
                           ('写一首五言绝句', dict(lines=4, width=7)),
                           ('写一首七言律诗', dict(form='七言律诗', lines=4, width=7))]:
            with self.subTest(text=text, spec=spec), self.assertRaises(ValueError):
                resolve_poetry_spec([dict(role='user', content=text)], spec)

    def test_character_count_is_per_verse_not_total_or_tokens(self):
        seven = '春风又绿江南岸，\n细雨轻沾陌上衣。\n燕影穿帘寻旧垒，\n桃花临水映朝晖。\n远山含黛连天碧，\n小径浮香伴客归。\n欲寄闲情随流水，\n一川烟柳正依依。'
        messages = [dict(role='user', content='我要写一个七言律诗，一句一行。')]
        _, exact = poetry_rule_score(messages, seven)
        _, short = poetry_rule_score(messages, seven.replace('春风又绿江南岸', '春风绿江南岸'))
        _, long = poetry_rule_score(messages, seven.replace('春风又绿江南岸', '春风又绿那江南岸'))
        self.assertEqual(exact['line_width'], 1.5)
        self.assertEqual(exact['format_exact'], 1.)
        self.assertAlmostEqual(short['line_width'], 1.5*7/8)
        self.assertAlmostEqual(long['line_width'], 1.5*7/8)
        _, five = poetry_rule_score(self.messages, self.valid)
        self.assertEqual(five['line_width'], 1.5)

    def test_legacy_helper_matches_archived_rules_but_active_scoring_sees_full_response(self):
        from trainer.grpo_poetry_reward import legacy_general_rule_score
        original_path = ROOT/'out/grpo/code_before_20261029/train_grpo.py'
        original = ast.parse(original_path.read_text(encoding='utf-8'))
        definitions = [n for n in original.body if isinstance(n, ast.FunctionDef)
                       and n.name in ('rep_penalty', 'calculate_rewards')]
        messages = [dict(role='user', content='前面的问题'), dict(role='assistant', content='前面的回答'),
                    dict(role='user', content='请写一首七言律诗')]
        class RecordingReward:
            def __init__(self, value):
                self.value, self.calls = value, []
            def get_score(self, context, answer):
                self.calls.append((context, answer))
                return self.value
        candidates = ['', '短回答', '这是一条符合原来长度要求的普通问答样本，用于核对奖励。', '长'*801,
                      '<think>'+'思考内容'*8+'</think>\n回答正文',
                      '<think>短</think>答案</think>多余标签',
                      'same words repeat same words repeat same words repeat']
        for raw in [-20., -1., 0., 1., 20.]:
            old_model, new_model = RecordingReward(max(-3., min(3., raw))), RecordingReward(raw)
            scope = dict(re=re, torch=torch,
                args=SimpleNamespace(device='cpu', num_generations=len(candidates)))
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(original_path), 'exec'), scope)
            prompt = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            expected = scope['calculate_rewards']([prompt], candidates, old_model)
            actual, details = calculate_rewards([dict(messages=messages, spec={}, channel='general')],
                candidates, new_model, len(candidates), model_weight=8., repetition_weight=0.)
            legacy = [legacy_general_rule_score(text)[0]+max(-3., min(3., raw)) for text in candidates]
            self.assertTrue(torch.allclose(torch.tensor(legacy), expected, atol=1e-6))
            self.assertTrue(all(d['reward_mode'] == 'general_reviewed_v4' for d in details))
            self.assertEqual([answer for _, answer in new_model.calls], candidates)
            self.assertTrue(all(d['model_bounded'] == max(-3., min(3., raw)) for d in details))
            self.assertLess(float(actual[4]), float(expected[4]))
            self.assertEqual(new_model.calls[0][0][0]['content'],
                'user: 前面的问题\nassistant: 前面的回答\n以上是对话历史。我的新问题是：\n请写一首七言律诗')

    def test_copy_and_format_repair_are_different_tasks(self):
        body = ''.join(c for c in self.valid if '\u4e00' <= c <= '\u9fff')
        creative, c = poetry_rule_score(self.messages, self.valid, dict(self.spec, reference_body=body, discourage_copy=True))
        repair, r = poetry_rule_score(self.messages, self.valid, dict(self.spec, reference_body=body, preserve_chars=True))
        self.assertEqual(c['copying'], -2.)
        self.assertEqual(r['preserve_text'], 1.)
        self.assertGreater(repair, creative)

    def test_acrostic_and_reward_alignment(self):
        spec = dict(self.spec, acrostic='春细燕晨')
        _, good = poetry_rule_score(self.messages, self.valid, spec)
        _, wrong = poetry_rule_score(self.messages, self.valid.replace('春风', '东风'), spec)
        self.assertGreater(good['acrostic'], wrong['acrostic'])
        rows = [dict(messages=self.messages, spec=self.spec), dict(messages=[dict(role='user',content='你好')], spec={})]
        totals, detail = calculate_rewards(rows, [self.valid, '', '你好', '您好'], FakeContentReward(), 2)
        self.assertEqual(len(totals), 4)
        self.assertIn('line_width', detail[0]['rules'])
        self.assertNotIn('line_width', detail[2]['rules'])
        self.assertTrue(torch.isfinite(totals).all())

    def test_eos_mask_and_group_standardization(self):
        mask = completion_mask(torch.tensor([[4, 2, 2, 2], [5, 6, 7, 8], [2, 2, 2, 2]]), 2)
        self.assertEqual(mask.tolist(), [[True, True, False, False], [True]*4, [True, False, False, False]])
        adv = group_advantages(torch.tensor([1., 1., 1., 3.]), 2)
        self.assertTrue(torch.equal(adv[:2], torch.zeros(2)))
        self.assertAlmostEqual(float(adv[2:].mean()), 0., places=6)

    def test_eos_before_padding_is_not_truncation(self):
        completions = torch.tensor([[7, 2, 0, 0], [7, 8, 9, 10], [2, 0, 0, 0]])
        self.assertEqual(completion_ended(completions, 2).tolist(), [True, False, True])
        self.assertEqual(completion_mask(completions, 2).tolist(),
                         [[True, True, False, False], [True]*4, [True, False, False, False]])

    def test_semantically_bad_poem_cannot_collect_format_reward(self):
        class SemanticReward(FakeContentReward):
            def poetry_rating(_, messages, response):
                good = response == self.valid
                return dict(coherence=4 if good else 0, quality=1. if good else 0., reason='fixture')
        row = dict(messages=self.messages, spec=self.spec, channel='poetry')
        # Same strict format, different semantic judgment: formatting alone cannot win.
        bad = self.valid.replace('春风入小院', '蹙湯仗山液')
        totals, details = calculate_rewards([row], [self.valid, bad], SemanticReward(), 2,
                                             completion_ended=[True, True])
        self.assertGreater(float(totals[0]), 0)
        self.assertLess(float(totals[1]), 0)
        self.assertEqual(details[1]['rules']['format_exact'], 1.)
        truncated, _ = calculate_rewards([row], [self.valid], SemanticReward(), 1, completion_ended=[False])
        self.assertAlmostEqual(float(totals[0]-truncated[0]), 1.5, places=5)

    def test_unchanged_unformatted_copy_has_no_preservation_bonus(self):
        body = ''.join(re.findall(r'[\u4e00-\u9fff]', self.valid))
        spec = dict(self.spec, reference_body=body, preserve_chars=True)
        raw, pieces = poetry_rule_score(self.messages, body, spec)
        formatted, good = poetry_rule_score(self.messages, self.valid, spec)
        self.assertEqual(pieces['preserve_text'], 0.)
        self.assertLess(raw, 0.)
        self.assertEqual(good['preserve_text'], 1.)
        self.assertGreater(formatted, raw)

    def test_local_judge_rejects_invalid_ratings(self):
        for value in [True, -1, 5, float('nan'), 2.5, '4', None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                LocalPoetryJudge.parse_rating(dict(coherence=value, relevance=4, reason='test'))
        self.assertEqual(LocalPoetryJudge.parse_rating(dict(coherence=4, relevance=1, reason='off topic'))['quality'], .25)

    def test_low_semantic_quality_also_blocks_positive_format_credit(self):
        class LowQuality(FakeContentReward):
            def poetry_rating(_, messages, response):
                return dict(coherence=1, quality=.25, reason='mostly incoherent')
        totals, _ = calculate_rewards([dict(messages=self.messages, spec=self.spec, channel='poetry')],
                                     [self.valid], LowQuality(), 1)
        self.assertLess(float(totals[0]), 0.)

    def test_format_repair_cannot_qualify_after_changing_reference_text(self):
        class Coherent(FakeContentReward):
            def poetry_rating(_, messages, response):
                return dict(coherence=4, quality=1., reason='coherent')
        body=''.join(re.findall(r'[\u4e00-\u9fff]',self.valid))
        row=dict(messages=self.messages,channel='poetry',spec=dict(self.spec,reference_body=body,preserve_chars=True))
        _,detail=calculate_rewards([row],[self.valid.replace('春风','东风')],Coherent(),1,completion_ended=[True])
        self.assertEqual(detail[0]['rules']['format_exact'],1.)
        self.assertFalse(detail[0]['qualified'])
        self.assertLess(detail[0]['total'],0.)

    def test_copying_reference_cannot_win_rewrite_reward(self):
        class Coherent(FakeContentReward):
            def poetry_rating(_, messages, response):
                return dict(coherence=4,quality=1.,reason='coherent copied poem')
        body=''.join(re.findall(r'[\u4e00-\u9fff]',self.valid))
        row=dict(messages=self.messages,channel='poetry',spec=dict(self.spec,reference_body=body,discourage_copy=True))
        rewards,details=calculate_rewards([row],[self.valid],Coherent(),1,completion_ended=[True])
        self.assertLess(float(rewards[0]),0.)
        self.assertFalse(details[0]['qualified'])

    def test_warmup_targets_are_real_format_repairs_with_eos(self):
        from trainer.warmup_grpo_format import repair_pair
        from trainer.sft_data import AssistantDataset
        body = ''.join(re.findall(r'[\u4e00-\u9fff]', self.valid))
        pair = repair_pair(dict(id='fixture', messages=[dict(role='user', content='恢复分句和标点')],
            poetry_spec=dict(self.spec, reference_body=body)))
        self.assertEqual(pair['conversations'][-1]['content'], self.valid)
        encoded = AssistantDataset([], self.tokenizer, 768).encode(pair)
        self.assertFalse(encoded['truncated'])
        self.assertGreater(encoded['eos_supervised'], 0)

    def test_stop_file_is_handled_without_leaving_stale_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with StopController(folder) as controller:
                self.assertFalse(controller.requested)
                (folder/'STOP').write_text('user requested stop')
                with self.assertRaises(StopRequested): controller.check()
            self.assertFalse((folder/'STOP').exists())
            self.assertEqual(len(list(folder.glob('STOP.handled.*'))), 1)

    def test_policy_gradient_reaches_router_and_reference_is_frozen(self):
        config = MindChenConfig(hidden_size=32, num_hidden_layers=1, use_moe=True, intermediate_size=64,
            moe_intermediate_size=32, max_position_embeddings=512, norm_topk_prob=False)
        model = MindChenForCausalLM(config).eval()
        ids = torch.randint(5, 100, (2, 12)); mask = torch.ones_like(ids)
        current = token_logps(model, ids, mask, 4)
        old = current.detach().clone(); reference = current.detach().clone()
        for kind in ['grpo', 'cispo']:
            model.zero_grad(set_to_none=True)
            current = token_logps(model, ids, mask, 4)
            loss, kl = policy_objective(current, old, reference, torch.tensor([-1., 1.]), torch.ones(2, 4),
                SimpleNamespace(loss_type=kind, epsilon=.2, epsilon_high=5., beta=.1))
            loss.backward()
            self.assertGreater(float(model.model.layers[0].mlp.gate.weight.grad.norm()), 0)
            self.assertAlmostEqual(float(kl), 0., places=6)
            self.assertFalse(reference.requires_grad)

    def test_cached_generation_logits_match_full_prefix(self):
        config=MindChenConfig(hidden_size=32,num_hidden_layers=2,use_moe=True,intermediate_size=64,
            moe_intermediate_size=32,max_position_embeddings=512,norm_topk_prob=False)
        torch.manual_seed(15);model=MindChenForCausalLM(config).eval()
        ids=torch.randint(5,100,(2,14));mask=torch.ones_like(ids)
        with torch.no_grad():
            prefix=model(ids[:,:10],attention_mask=mask[:,:10],use_cache=True)
            for index in range(10,14):
                step=model(ids[:,index:index+1],attention_mask=mask[:,:index+1],
                    past_key_values=prefix.past_key_values,use_cache=True)
                full=model(ids[:,:index+1],attention_mask=mask[:,:index+1],use_cache=False)
                torch.testing.assert_close(step.logits[:,-1],full.logits[:,-1],atol=2e-5,rtol=2e-5)
                prefix=step

    def test_snapshot_is_bound_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; source.mkdir(); run = root/'grpo'; run.mkdir()
            weight = source/'test.pth'; weight.write_bytes(b'consistent weight')
            write_checkpoint_config(weight, MindChenConfig(hidden_size=32, norm_topk_prob=False))
            before = (file_digest(weight), sidecar_path(weight).read_bytes())
            snapshot, config, provenance = snapshot_source(weight, run)
            self.assertEqual(file_digest(snapshot), before[0])
            self.assertFalse(config.norm_topk_prob)
            self.assertEqual((file_digest(weight), sidecar_path(weight).read_bytes()), before)
            with self.assertRaises(ValueError):
                protect_source(source/'grpo', weight)
            weight.write_bytes(b'updating concurrently')
            other = root/'other'; other.mkdir()
            with self.assertRaises(RuntimeError):
                snapshot_source(weight, other, retries=1)
            self.assertEqual(weight.read_bytes(), b'updating concurrently')

    def test_source_read_allows_training_to_atomically_replace_file(self):
        project_output = ROOT/'out/grpo'
        project_output.mkdir(parents=True, exist_ok=True)
        for base in [None, str(project_output)]:
            with tempfile.TemporaryDirectory(dir=base) as tmp:
                old, new = Path(tmp)/'weight.pth', Path(tmp)/'weight.tmp'
                old.write_bytes(b'old consistent checkpoint')
                new.write_bytes(b'new checkpoint from ongoing trainer')
                aliases = Path(tmp)/'grpo_aliases'
                with anchor_read(old, aliases) as opened:
                    os.replace(new, old)
                    self.assertEqual(opened.read(), b'old consistent checkpoint')
                self.assertEqual(old.read_bytes(), b'new checkpoint from ongoing trainer')

    def write_small_data(self, root):
        general = root/'rlaif.jsonl'
        rows = [dict(conversations=[dict(role='user', content=f'请介绍数字{i}的含义。'),
                                   dict(role='assistant', content='不能泄漏的参考答案'),
                                   dict(role='assistant', content='')]) for i in range(100)]
        general.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows+[rows[0]]),encoding='utf-8')
        poetry = root/'poetry'; poetry.mkdir()
        for split, themes in [('train', THEMES[:1]), ('valid', THEMES[40:41])]:
            (poetry/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'
                    for r in make_rows(themes, split)),encoding='utf-8')
        return general, poetry/'train.jsonl'

    def test_mixed_split_no_answer_leak_and_full_epoch_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            general, poetry = self.write_small_data(Path(tmp))
            before = file_digest(general)
            data = MixedPromptDataset(general, poetry, self.tokenizer)
            self.assertEqual(data.duplicates, 1)
            self.assertEqual(data.normalized_tails, 101)
            self.assertEqual(data.general_count+len(data.general_valid), 100)
            self.assertFalse({r['prompt'] for r in data.general}&{r['prompt'] for r in data.general_valid})
            self.assertTrue(all('不能泄漏' not in r['prompt'] for r in data.general))
            indices = data.epoch_indices(42, .5)
            self.assertEqual(set(indices), set(range(len(data))))
            self.assertEqual(indices, data.epoch_indices(42, .5))
            self.assertEqual(file_digest(general), before)
            self.assertEqual({r['channel'] for r in data.validation_rows(2)}, {'general', 'poetry'})

    def test_corrupt_and_empty_user_prompts_are_quarantined_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            general, poetry = self.write_small_data(Path(tmp))
            with general.open('a', encoding='utf-8') as stream:
                for text in ['', '坏\ufffd编码']:
                    stream.write(json.dumps(dict(conversations=[dict(role='user', content=text),
                        dict(role='assistant', content='')]), ensure_ascii=False)+'\n')
            before = file_digest(general)
            data = MixedPromptDataset(general, poetry, self.tokenizer)
            self.assertEqual(len(data.quarantine), 2)
            self.assertEqual({q['reason'] for q in data.quarantine},
                             {'empty_final_user', 'unicode_replacement_character'})
            self.assertEqual(data.general_count+len(data.general_valid), 100)
            self.assertEqual(file_digest(general), before)

    def test_real_optimizer_resume_matches_uninterrupted_training(self):
        self.verify_real_optimizer_resume(False)

    def test_supervised_replay_resume_matches_uninterrupted_training(self):
        self.verify_real_optimizer_resume(True)

    def verify_real_optimizer_resume(self, with_replay):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'sft'; source.mkdir()
            general, poetry = self.write_small_data(root)
            config = MindChenConfig(hidden_size=32, num_hidden_layers=1, use_moe=True, intermediate_size=64,
                moe_intermediate_size=32, max_position_embeddings=512, norm_topk_prob=False)
            torch.manual_seed(7)
            weight = source/'tiny.pth'; torch.save(MindChenForCausalLM(config).state_dict(), weight)
            write_checkpoint_config(weight, config)
            before = file_digest(weight)
            common = ['--checkpoint', str(weight), '--save_dir', str(root/'grpo'), '--data_path', str(general),
                '--poetry_data_path', str(poetry), '--device', 'cpu', '--dtype', 'float32', '--cpu_threads', '2',
                '--max_seq_len', '256', '--max_gen_len', '4', '--eval_max_gen_len', '4', '--num_generations', '2',
                '--accumulation_steps', '2', '--max_steps', '2', '--validation_per_channel', '1', '--save_interval', '1']
            if with_replay:
                replay=root/'replay_train.jsonl'
                replay_rows=[dict(conversations=self.messages+[dict(role='assistant',content=self.valid)],spec=self.spec),
                    dict(conversations=[dict(role='user',content='请介绍一个数字。'),dict(role='assistant',content='这是一个正整数。')])]
                replay.write_text(''.join(json.dumps(dict(row,split='train'),ensure_ascii=False)+'\n'
                                         for row in replay_rows),encoding='utf-8')
                common+=['--replay_data_path',str(replay),'--replay_weight','.1']
            with patch('trainer.train_grpo.ensure_reward_model', return_value=ROOT/'model/internlm2-1_8b-reward'), \
                 patch('trainer.train_grpo.PoetryRewardModel', FakeContentReward), contextlib.redirect_stdout(io.StringIO()):
                main(common+['--run_name', 'full'])
                main(common+['--run_name', 'split', '--stop_after_steps', '1'])
                main(['--resume', str(root/'grpo/split/resume.pth'), '--device', 'cpu', '--cpu_threads', '2'])
            full = torch.load(root/'grpo/full/resume.pth', weights_only=True)
            split = torch.load(root/'grpo/split/resume.pth', weights_only=True)
            self.assertEqual(full['progress']['update'], 2)
            self.assertEqual(full['progress'], split['progress'])
            self.assertTrue(all(torch.equal(v, split['model'][k]) for k,v in full['model'].items()))
            self.assertEqual(file_digest(weight), before)
            cfg, _ = load_checkpoint_config(root/'grpo/split/grpo_poetry_32_moe.pth')
            self.assertFalse(cfg.norm_topk_prob)

    def test_stop_during_validation_saves_resumable_zero_update_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'sft';source.mkdir()
            general,poetry=self.write_small_data(root)
            config=MindChenConfig(hidden_size=32,num_hidden_layers=1,use_moe=True,intermediate_size=64,
                moe_intermediate_size=32,max_position_embeddings=512,norm_topk_prob=False)
            weight=source/'tiny.pth';torch.save(MindChenForCausalLM(config).state_dict(),weight)
            write_checkpoint_config(weight,config);before=file_digest(weight)
            args=['--checkpoint',str(weight),'--save_dir',str(root/'grpo'),'--run_name','stop',
                '--data_path',str(general),'--poetry_data_path',str(poetry),'--device','cpu',
                '--max_seq_len','256','--max_gen_len','4','--eval_max_gen_len','4',
                '--num_generations','2','--max_steps','1','--validation_per_channel','1']
            with patch('trainer.train_grpo.ensure_reward_model',return_value=ROOT/'model/internlm2-1_8b-reward'), \
                 patch('trainer.train_grpo.PoetryRewardModel',FakeContentReward), \
                 patch('trainer.train_grpo.validate',side_effect=StopRequested), \
                 contextlib.redirect_stdout(io.StringIO()):
                main(args)
            folder=root/'grpo/stop'
            summary=json.loads((folder/'summary.json').read_text())
            self.assertEqual((summary['status'],summary['update']),('stopped',0))
            checkpoint=torch.load(folder/'resume.pth',weights_only=True)
            self.assertEqual(checkpoint['progress']['update'],0)
            self.assertEqual(file_digest(weight),before)
            with patch('trainer.train_grpo.ensure_reward_model',return_value=ROOT/'model/internlm2-1_8b-reward'), \
                 patch('trainer.train_grpo.PoetryRewardModel',FakeContentReward), \
                 contextlib.redirect_stdout(io.StringIO()):
                main(['--resume',str(folder/'resume.pth'),'--device','cpu'])
            self.assertEqual(json.loads((folder/'summary.json').read_text())['status'],'completed')


if __name__ == '__main__':
    unittest.main()
