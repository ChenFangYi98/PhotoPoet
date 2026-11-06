# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""针对真实清洗风险的回归：正文保真、歧义、工具结构、第三方事实、身份与分割。"""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.clean_sft_dataset import (
    POETRY_SYSTEM, RECITE, ZH_IDENTITY, EN_IDENTITY, clean, transform,
    recitation_prompt, digest, build_recitation_catalog,
)


def dialogue(user, answer, system=None, **assistant_fields):
    messages = [dict(role='system', content=system)] if system else []
    messages += [dict(role='user', content=user), dict(role='assistant', content=answer, **assistant_fields)]
    return dict(conversations=messages)


class CleaningTests(unittest.TestCase):
    def test_recitation_keeps_exact_body_and_author(self):
        body = '生前喜作死規模，强把冠裳束幻軀。\n兒輩對人拋好語，吾家南極老人圖。'
        original = dialogue(RECITE+'\n《宋诗 / 身後寄顔》\n作者：葉茵\n首句提示：生前喜作死規模，', body, POETRY_SYSTEM)
        before = json.dumps(original, ensure_ascii=False)
        result, tags, reason = transform(original)
        self.assertIsNone(reason)
        self.assertEqual(result['conversations'][0]['content'], RECITE+'\n《身後寄顔》\n作者：葉茵')
        self.assertEqual(result['conversations'][1]['content'], body)
        self.assertEqual(json.dumps(original, ensure_ascii=False), before)

    def test_same_title_author_different_bodies_are_ambiguous(self):
        prompt = RECITE+'\n《宋词 / 词牌：浣溪沙》\n作者：甲\n首句提示：甲'
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for split, body in [('train', '甲甲甲'), ('valid', '甲乙乙')]:
                paths[split] = Path(folder)/(split+'.jsonl')
                paths[split].write_text(json.dumps(dialogue(prompt, body, POETRY_SYSTEM), ensure_ascii=False)+'\n', encoding='utf-8')
            db = sqlite3.connect(':memory:')
            db.execute('CREATE TABLE recitations(prompt BLOB PRIMARY KEY,answer BLOB,ambiguous INTEGER)')
            ambiguous = build_recitation_catalog(paths, db)
            self.assertEqual(len(ambiguous), 1)
            self.assertEqual(transform(dialogue(prompt, '甲甲甲', POETRY_SYSTEM), ambiguous)[2], 'ambiguous_title_author')

    def test_external_identity_rewritten_and_reasoning_cleared(self):
        row = dialogue('你是谁？', '我是ChatGPT，由OpenAI开发。', reasoning_content='我是OpenAI的模型。')
        result, _, reason = transform(row)
        self.assertIsNone(reason)
        self.assertEqual(result['conversations'][1]['content'], ZH_IDENTITY)
        self.assertEqual(result['conversations'][1]['reasoning_content'], '')

    def test_english_identity_case_and_creator_alias(self):
        row = dialogue('Hello', 'Hello! I am MINIMIND, developed by Jingyao Gong.')
        result, _, reason = transform(row)
        self.assertIsNone(reason)
        self.assertEqual(result['conversations'][1]['content'], 'Hello! I am mindchen, developed by chen.')
        result, _, _ = transform(dialogue('Who is your developer?', 'OpenAI'))
        self.assertEqual(result['conversations'][1]['content'], EN_IDENTITY)

    def test_third_party_facts_and_roleplay_not_identity(self):
        for user, answer in [('OpenAI是什么公司？', 'OpenAI是一家人工智能公司。'),
                             ('ChatGPT是什么？', 'ChatGPT是OpenAI开发的AI助手。'),
                             ('翻译这句话：I am ChatGPT.', '我是ChatGPT。'),
                             ('请扮演ChatGPT写一段故事。', '我是ChatGPT，这是故事的开头。')]:
            row = dialogue(user, answer)
            result, _, reason = transform(row)
            self.assertIsNone(reason)
            self.assertEqual(result, row)

    def test_code_urls_quarantined_not_corrupted(self):
        for answer in ['使用 `MiniMind` 类。', 'https://example.com/MiniMind/model']:
            row = dialogue('解释代码', answer)
            result, _, reason = transform(row)
            self.assertIsNone(result)
            self.assertEqual(reason, 'legacy_identity_in_protected_or_structured_text')
            self.assertEqual(row['conversations'][1]['content'], answer)

    def test_tool_fields_and_reasoning_preserved(self):
        row = dict(conversations=[dict(role='system', content='查询天气', tools='[{"name":"weather"}]'),
            dict(role='user', content='北京天气如何？'),
            dict(role='assistant', content='', tool_calls='[{"name":"weather","arguments":{}}]'),
            dict(role='tool', content='晴天'),
            dict(role='assistant', content='北京晴天。', reasoning_content='依据工具结果回答。')])
        result, _, reason = transform(row)
        self.assertIsNone(reason)
        self.assertEqual(result, row)

    def test_false_original_poetry_task_is_quarantined(self):
        row = dialogue('写诗：\n静夜思', '床前明月光，疑是地上霜。', POETRY_SYSTEM)
        self.assertEqual(transform(row)[2], 'unverified_poetry_instruction')

    def test_mixed_self_identity_quarantined_for_review(self):
        row = dialogue('给我解释光合作用。', '作为ChatGPT，我来解释光合作用。')
        self.assertEqual(transform(row)[2], 'external_identity_answer_or_reasoning')

    def test_tool_return_with_old_brand_is_not_rewritten(self):
        row = dict(conversations=[dict(role='user', content='查询模型'),
            dict(role='tool', content='MiniMind'), dict(role='assistant', content='已查询。')])
        self.assertEqual(transform(row)[2], 'legacy_identity_in_protected_or_structured_text')
        self.assertEqual(row['conversations'][1]['content'], 'MiniMind')

    def test_parenthesized_and_curly_apostrophe_self_identity(self):
        for answer in ['我（Qwen）可以帮助你。', 'I’m Claude. Let me explain.']:
            self.assertEqual(transform(dialogue('解释一下数学。',answer))[2], 'external_identity_answer_or_reasoning')

    def test_pipeline_blocks_validation_answers_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'input', root/'output'
            source.mkdir()
            valid = dialogue('问题甲', '受保护的答案')
            rows = dict(valid=[valid], train=[dialogue('另一个问题', '受保护的答案'),
                dialogue('你是谁？', '我是MiniMind，由jingyaogong创建。'),
                dialogue('你是谁？', '我是MiniMind，由jingyaogong创建。')])
            originals = {}
            for split, items in rows.items():
                path = source/f'sft_t2t_with_poetry_{split}.jsonl'
                originals[path] = '\n'.join(json.dumps(r, ensure_ascii=False) for r in items)+'\n'
                path.write_text(originals[path], encoding='utf-8')
            clean(source, output, None)
            report = json.loads((output/'clean_report.json').read_text(encoding='utf-8'))
            counts = report['outputs']['train']['counts']
            self.assertEqual(counts['output_rows'], 1)
            self.assertEqual(counts['rejected_protected_validation_answer'], 1)
            self.assertEqual(counts['rejected_duplicate_after_cleaning'], 1)
            for path, content in originals.items():
                self.assertEqual(path.read_text(encoding='utf-8'), content)


if __name__ == '__main__':
    unittest.main()
