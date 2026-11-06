# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Regression checks for the six code-review fixes, including Windows file locks."""
import errno
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import datasets  # Windows DLL initialization order
import torch

from trainer.grpo_io import atomic_json, atomic_save, replace_with_retry
from trainer.grpo_control import live_state
from trainer.grpo_poetry_reward import calculate_rewards
from trainer.grpo_repetition import follows_repeat_request, response_repetition, unlikelihood_loss
from trainer.train_grpo import selection_decision
import eval_llm


class ReviewFixTests(unittest.TestCase):
    def test_transient_lock_retries_and_failed_write_preserves_old_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'status.json'
            atomic_json(target, {'old': True})
            original = os.replace
            attempts = []
            def transient(source, destination):
                attempts.append(source)
                if len(attempts) < 3:
                    raise PermissionError(errno.EACCES, 'sharing violation')
                return original(source, destination)
            with patch('trainer.grpo_io.os.replace', side_effect=transient), patch('trainer.grpo_io.time.sleep'):
                atomic_json(target, {'new': True})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(target.read_text()), {'new': True})
            with patch('trainer.grpo_io.os.replace', side_effect=PermissionError(errno.EACCES, 'locked')), patch('trainer.grpo_io.time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_json(target, {'lost': True})
                with self.assertWarns(RuntimeWarning):
                    self.assertFalse(live_state(folder, 'running', dict(update=1, total_updates=2)))
                with self.assertRaises(PermissionError):
                    atomic_save(Path(folder)/'weights.pth', {'weight': torch.tensor([1.])})
            self.assertEqual(json.loads(target.read_text()), {'new': True})
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_nontransient_error_is_not_retried(self):
        with patch('trainer.grpo_io.os.replace', side_effect=OSError(errno.ENOSPC, 'disk full')) as replace:
            with self.assertRaises(OSError):
                replace_with_retry('source', 'target')
            self.assertEqual(replace.call_count, 1)

    @unittest.skipUnless(os.name == 'nt', 'real Windows sharing lock')
    def test_real_windows_read_lock_recovers_after_release(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'runtime.json'
            atomic_json(target, {'update': 0})
            locked = target.open('rb')  # CRT denies delete/replace while this handle is open.
            timer = threading.Timer(.2, locked.close)
            timer.start()
            try:
                atomic_json(target, {'update': 1})
                self.assertEqual(json.loads(target.read_text()), {'update': 1})
            finally:
                locked.close()
                timer.join()

    def test_thinking_garbage_has_no_bonus_and_is_visible_to_reward_model(self):
        answer = 'This answer gives the requested information clearly and directly.'
        garbage = ''.join(chr(0x4e00+i) for i in range(32))
        responses = [answer, garbage+'</think>'+answer, '<think>'+garbage+'</think>'+answer,
                     '<think></think>'+answer, '<think>'+garbage+answer]
        class Recorder:
            def __init__(self):
                self.answers = []
            def get_score(self, messages, text):
                self.answers.append(text)
                return 0.
        scorer = Recorder()
        row = dict(channel='general', messages=[dict(role='user', content='Explain the result.')])
        rewards, details = calculate_rewards([row], responses, scorer, len(responses))
        self.assertTrue((rewards[1:] < rewards[0]).all())
        self.assertEqual(scorer.answers, responses)
        self.assertTrue(all('thinking_length' not in d['rules'] for d in details))

    def test_quality_collapse_and_empty_shortcut_cannot_be_best(self):
        options = SimpleNamespace(general_regression_tolerance=.15, repetition_regression_tolerance=.05)
        progress = {}
        def result(reward=-2., repetition=.25, empty=0., semantic=.25):
            return dict(general=dict(model_score=-1., repetition_rate=repetition, empty_fraction=empty),
                        poetry=dict(reward=reward, repetition_rate=repetition, qualified_pass_rate=0.,
                                    semantic_quality=semantic, empty_fraction=empty))
        self.assertTrue(selection_decision(result(), progress, options)[2])
        self.assertFalse(selection_decision(result(-8., 0.), progress, options)[2])
        self.assertFalse(selection_decision(result(-2., 0., empty=.5), progress, options)[2])
        self.assertFalse(selection_decision(result(-2., 0., semantic=0.), progress, options)[2])
        self.assertTrue(selection_decision(result(-2., 0.), progress, options)[2])
        progress['best_quality'] = dict(progress['baseline_quality'], poetry_reward=0., score=0.)
        self.assertFalse(selection_decision(result(-2., 0.), progress, options)[2])

    def test_unlikelihood_gradient_survives_near_probability_one(self):
        logps = torch.tensor([[-1e-7, 0., -20.]], requires_grad=True)
        loss = unlikelihood_loss(logps, torch.ones_like(logps, dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logps.grad).all())
        self.assertTrue((logps.grad > 0).all())

    def test_explicit_repeat_task_must_match_phrase_count_and_format(self):
        tasks = [('Output hello on ten separate lines, exactly as requested.', '\n'.join(['hello']*10)),
                 ('Repeat "hello" ten times.', 'hello '*9+'hello'),
                 ('请把“你好”重复10次。', '你好'*10),
                 ('请逐行输出十次你好', '\n'.join(['你好']*10)),
                 ('请把“你好”重复十次，每行一次。', '\n'.join(['你好']*10))]
        for prompt, response in tasks:
            row = dict(channel='general', messages=[dict(role='user', content=prompt)])
            with self.subTest(prompt=prompt):
                self.assertTrue(follows_repeat_request(row, response))
                self.assertFalse(response_repetition(row, response)['excessive'])
                _, details = calculate_rewards([row], [response], None, 1)
                self.assertEqual(details[0]['rules']['repetition'], 0.)
                self.assertEqual(details[0]['rules']['repetition_penalty'], 0.)
                for bad in (response+' extra', response+response, ''):
                    self.assertFalse(follows_repeat_request(row, bad))
        row['messages'][-1]['content'] = 'Avoid repetition. Explain the word hello ten times more clearly.'
        self.assertFalse(follows_repeat_request(row, 'hello '*10))

    def test_evaluation_argument_precedence_and_training_defaults(self):
        args = eval_llm.parse_args(['--save_dir', 'out/chosen', '--weight', 'pretrain'])
        self.assertEqual(eval_llm.checkpoint_path(args), eval_llm.ROOT/'out/chosen/pretrain_768_moe.pth')
        args.checkpoint = 'out/exact.pth'
        self.assertEqual(eval_llm.checkpoint_path(args), eval_llm.ROOT/'out/exact.pth')
        args = eval_llm.parse_args([])
        self.assertEqual((args.open_thinking, args.max_new_tokens, args.top_p, args.dtype), (0, 256, 1., 'auto'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root/'out/grpo/current'
            run.mkdir(parents=True)
            control = root/'out/session_control'
            control.mkdir()
            (control/'grpo_session.json').write_text(json.dumps(dict(final_run_dir=str(run))))
            (run/'run_config.json').write_text(json.dumps(dict(args=dict(save_weight='chosen'),
                model_config=dict(hidden_size=32, use_moe=False), source={})))
            with patch.object(eval_llm, 'ROOT', root):
                self.assertEqual(eval_llm.checkpoint_path(args), run/'chosen_best_32.pth')


if __name__ == '__main__':
    unittest.main()
