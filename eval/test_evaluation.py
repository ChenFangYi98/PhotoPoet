# Modified by chenfangyi for the unified MindChen release (2026-11-06).
# -*- coding: utf-8 -*-
"""无需模型、无网络即可运行的评分器单元测试。"""
import unittest
import csv
import json
import tempfile
import io
import contextlib
from pathlib import Path
from unittest.mock import patch
from eval_mindchen_quality import (grade_answer, poetry_lines, strip_reasoning, format_prompt,
                                   aggregate, merge_manual, REVIEW_COLS, load_cases, HERE)
from capability_metrics import response_diagnostics, infer_architecture, select_cases
from literature_cases import build_literature_cases
from compare_runs import compatibility_errors, paired_deltas

class EvalTests(unittest.TestCase):
    def case(self, typ, **kwargs):
        return {'id':'test', 'rule': {'type': typ, **kwargs}}

    def test_exact(self):
        c = self.case('exact', answer='45')
        self.assertEqual(grade_answer(c, '答案：45')['auto_score'], 100)
        self.assertEqual(grade_answer(c, '46')['auto_score'], 0)

    def test_poem_format(self):
        c = self.case('poem', form='五言绝句', pattern=[5]*4,
                      concept_groups=[['春'],['雨']], rhyme_positions=[2,4])
        result = grade_answer(c, '春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。')
        self.assertEqual(result['details']['句数'], 100)
        self.assertEqual(result['details']['每句字数'], 100)
        self.assertEqual(len(result['details']['切分句子']), 4)

    def test_ci(self):
        c = self.case('ci', pattern=[6,6,5,6,2,2,6], concept_groups=[['雨']])
        response = '昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。知否？知否？应是绿肥红瘦。'
        result = grade_answer(c, response)
        self.assertEqual(result['details']['词牌字数模板'], 100)
        self.assertEqual(result['details']['句数'], 100)

    def test_wenyan(self):
        c = self.case('wenyan', source='今天下雨了', max_chars=20, concept_groups=[['雨'],['夜','今']])
        self.assertEqual(grade_answer(c, '今夜有雨')['details']['原意关键词代理'], 100)

    def test_remove_think(self):
        self.assertEqual(strip_reasoning('<think>内部分析</think>\n春风又绿江南岸'), '春风又绿江南岸')
        self.assertEqual(poetry_lines('《五言绝句》\n春眠不觉晓，处处闻啼鸟。'), ['春眠不觉晓','处处闻啼鸟'])

    def test_unclosed_reasoning_is_not_answer(self):
        self.assertEqual(strip_reasoning('<think>答案：45'), '')
        self.assertEqual(grade_answer(self.case('exact', answer='45'), '<think>45')['auto_score'], 0)

    def test_numeric_and_invalid_numbers(self):
        c = self.case('numeric', answer=1.5)
        self.assertEqual(grade_answer(c, '1.500')['auto_score'], 100)
        for bad in ('NaN', 'Infinity', '结果1.5也许2', '1.6', '1.5和1.6'):
            self.assertEqual(grade_answer(c, bad)['auto_score'], 0)

    def test_json_strict_semantics(self):
        c = self.case('json', expected={'姓名': '张生', '数量': 1})
        self.assertEqual(grade_answer(c, '{"数量":1,"姓名":"张生"}')['auto_score'], 100)
        for bad in ('{"姓名":"张生","数量":true}', '{"姓名":"张生","数量":1,"数量":1}',
                    '```json\n{"姓名":"张生","数量":1}\n```', '{"姓名":"张生","数量":NaN}'):
            self.assertEqual(grade_answer(c, bad)['auto_score'], 0)

    def test_constraints_all_required(self):
        c = self.case('constraints', checks={'line_count': 2, 'contains': ['春'], 'not_contains': ['秋']})
        self.assertTrue(grade_answer(c, '春山\n新雨')['details']['全约束通过'])
        self.assertFalse(grade_answer(c, '春山秋雨')['details']['全约束通过'])
        self.assertEqual(grade_answer(c, '')['auto_score'], 0)

    def test_choice_correctness_and_format_separate(self):
        c = self.case('choice', options={'A': '对', 'B': '错', 'C': '错2', 'D': '错3'}, answer='A')
        self.assertEqual(grade_answer(c, 'A')['auto_score'], 100)
        explained = grade_answer(c, '答案是A，因为题目如此。')
        self.assertEqual(explained['auto_score'], 0)
        self.assertEqual(explained['details']['选项内容正确'], 100)
        self.assertEqual(grade_answer(c, 'A或B')['details']['选项内容正确'], 0)

    def test_frozen_cases_and_reference_rules(self):
        from make_cases import cases as generated
        frozen = load_cases(HERE / 'eval_cases.jsonl', 'all')
        self.assertEqual(frozen, generated)
        self.assertEqual(len({c['id'] for c in frozen}), len(frozen))
        for c in frozen:
            r = c['rule']
            if r['type'] in ('exact', 'numeric', 'choice'):
                self.assertEqual(grade_answer(c, str(r.get('answer')))['auto_score'], 100, c['id'])
                self.assertEqual(grade_answer(c, '无关错误答案')['auto_score'], 0, c['id'])
            elif r['type'] == 'json':
                self.assertEqual(grade_answer(c, json.dumps(r['expected'], ensure_ascii=False))['auto_score'], 100)
            elif r['type'] == 'manual_only':
                self.assertIsNone(grade_answer(c, '任意文本')['auto_score'])

    def test_prompts_keep_history_and_hide_references(self):
        class Tokenizer:
            bos_token = '<bos>'
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages, ensure_ascii=False)
        case = next(c for c in build_literature_cases() if c.get('messages'))
        rendered = format_prompt(case, Tokenizer(), 'chat')
        self.assertIn('等我给出格式', rendered)
        self.assertIn('现在写五言绝句', rendered)
        case = dict(case, reference='SECRET_REFERENCE', rubric='SECRET_RUBRIC')
        for style in ('pretrain', 'chat'):
            self.assertNotIn('SECRET', format_prompt(case, Tokenizer(), style))

    def test_stratified_limit(self):
        cases = build_literature_cases()
        selected = select_cases(cases, limit=8, stratified=True)
        self.assertEqual(len({c['category'] for c in selected}), 8)

    def test_diagnostics(self):
        self.assertTrue(response_diagnostics('  ')['empty_response'])
        self.assertGreater(response_diagnostics('春雨' * 20)['repeated_char_4gram_ratio'], 0.8)

    def test_manual_translation_profile(self):
        row = dict(attempt_id='t#1', domain='poetry', category='古文今译', review_profile='translation', auto_score=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manual.csv'
            fields = ['attempt_id'] + REVIEW_COLS['translation']
            with path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(attempt_id='t#1', **{k: 4 for k in fields[1:]}))
            reviewed = merge_manual([row], path)[0]
            self.assertEqual(reviewed['manual_score'], 80)
            self.assertEqual(reviewed['combined_score'], 80)

    def test_errors_count_against_objective_accuracy(self):
        rows = [dict(id='x', status='ok', domain='poetry', category='阅读', rule_type='choice',
                     auto_score=100, details={'准确匹配': 100}),
                dict(id='y', status='error', domain='poetry', category='阅读', rule_type='choice', auto_score=None)]
        summary = aggregate(rows)
        self.assertEqual(summary['categories']['阅读']['objective_accuracy_including_errors_pct'], 50)
        self.assertEqual(summary['objective_category_macro_accuracy_pct'], 50)

    def test_compare_rejects_changed_rules_and_seed(self):
        cfg = dict(scorer_sha256='v1', selected_cases_sha256='cases1', seed=42)
        a = ('a', [], {}, cfg, set())
        b = ('b', [], {}, dict(cfg, seed=43, selected_cases_sha256='cases2'), set())
        self.assertTrue(any('seed' in s for s in compatibility_errors([a, b])))
        self.assertTrue(any('selected_cases' in s for s in compatibility_errors([a, b])))

    def test_pairing_averages_samples_before_bootstrap(self):
        a = [dict(id='a', category='阅读', status='ok', auto_score=s) for s in (0, 100)]
        b = [dict(id='a', category='阅读', status='ok', auto_score=100)]
        result = paired_deltas(a, b)[0]
        self.assertEqual(result['配对题数'], 1)
        self.assertEqual(result['候选减基线'], 50)
        self.assertIsNone(result['差值95%下界'])

    def test_infer_moe_architecture_from_shapes(self):
        class Tensor:
            def __init__(self, *shape):
                self.shape = shape
        state = {'model.embed_tokens.weight': Tensor(6400, 768),
                 'model.layers.0.self_attn.q_norm.weight': Tensor(96),
                 'model.layers.0.self_attn.q_proj.weight': Tensor(768, 768),
                 'model.layers.0.self_attn.k_proj.weight': Tensor(384, 768),
                 'model.layers.0.mlp.gate.weight': Tensor(4, 768),
                 'model.layers.0.mlp.experts.0.gate_proj.weight': Tensor(2432, 768)}
        cfg = infer_architecture(state)
        self.assertTrue(cfg['use_moe'])
        self.assertEqual(cfg['num_experts'], 4)
        self.assertEqual(cfg['num_key_value_heads'], 4)

    def test_cli_score_merge_compare_pipeline(self):
        from eval_mindchen_quality import main as evaluate
        from compare_runs import main as compare
        cases = [c for c in build_literature_cases() if c['id'] in ('lit_memory_01', 'lit_translate_01')]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            case_file = root / 'cases.jsonl'
            case_file.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases), encoding='utf-8')
            for name, memory in [('a', '疑是地上霜'), ('b', '错误回答')]:
                responses = root / f'{name}.jsonl'
                responses.write_text(json.dumps({'id': 'lit_memory_01', 'response': memory}) + '\n' +
                                     json.dumps({'id': 'lit_translate_01', 'response': '客人来了，主人请他进去并备茶接待。'}) + '\n', encoding='utf-8')
                with patch('sys.argv', ['eval', '--mode', 'score-only', '--cases', str(case_file),
                           '--responses', str(responses), '--output_root', str(root), '--run_name', name]):
                    evaluate()
                self.assertTrue((root / name / '评估报告.md').is_file())
            manual = root / 'a' / '人工评分表.csv'
            with manual.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                fields, reviews = reader.fieldnames, list(reader)
            for row in reviews:
                if row['id'] == 'lit_translate_01':
                    row.update({key: '4' for key in REVIEW_COLS['translation']})
            with manual.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(reviews)
            with patch('sys.argv', ['eval', '--mode', 'merge', '--input_results', str(root / 'a' / 'results.jsonl'),
                                   '--manual_csv', str(manual)]):
                evaluate()
            reviewed = [json.loads(s) for s in (root / 'a' / 'reviewed_results.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(next(r for r in reviewed if r['id'] == 'lit_translate_01')['manual_score'], 80)
            with patch('sys.argv', ['compare', str(root / 'a'), str(root / 'b'), '--output', str(root / 'comparison.csv')]):
                compare()
            with (root / 'comparison_配对.csv').open(encoding='utf-8-sig', newline='') as f:
                paired = list(csv.DictReader(f))
            self.assertEqual(float(paired[0]['候选减基线']), -100)

    def test_changed_prompt_is_not_paired(self):
        a = [dict(id='a', prompt='问题一', category='阅读', status='ok', auto_score=0)]
        b = [dict(id='a', prompt='问题二', category='阅读', status='ok', auto_score=100)]
        self.assertEqual(paired_deltas(a, b), [])

    def test_memory_span_is_only_proxy(self):
        c = dict(id='memory', category='名句背诵补全', rule=dict(type='exact', answer='疑是地上霜'))
        r = grade_answer(c, '床前明月光，疑是地上霜。')
        self.assertEqual(r['auto_score'], 0)
        self.assertTrue(r['details']['参考答案字面出现'])

if __name__ == '__main__':
    unittest.main()
