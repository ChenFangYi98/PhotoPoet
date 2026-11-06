# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Regressions for the judge's deterministic token-repeat failure."""
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from trainer.grpo_poetry_judge import LocalPoetryJudge, JudgeServiceError, RATING_SCHEMA


class JudgeTransportTests(unittest.TestCase):
    def judge(self):
        judge = LocalPoetryJudge.__new__(LocalPoetryJudge)
        judge.model, judge.url, judge.timeout = 'fixture', 'http://127.0.0.1:11435', 5
        judge.cache, judge.audit_path, judge.provenance = {}, None, {'version': 'fixture'}
        return judge

    def test_schema_is_sent_and_score_is_validated_and_cached(self):
        judge = self.judge()
        response = dict(done_reason='stop', message=dict(content=json.dumps(dict(coherence=1, relevance=2, reason='incoherent'))))
        with patch.object(judge, 'request', return_value=response) as request:
            rating = judge.rate([dict(content='Restore the poem.')], 'candidate')
            self.assertEqual(rating['quality'], .25)
            payload = request.call_args.args[1]
            self.assertEqual(payload['format'], RATING_SCHEMA)
            self.assertEqual(payload['options']['temperature'], 0)
            self.assertEqual(payload['format']['properties']['coherence']['enum'], list(range(5)))
            judge.rate([dict(content='Restore the poem.')], 'candidate')
            self.assertEqual(request.call_count, 1)

    def test_repeat_limit_preserves_body_and_never_retries_same_generation(self):
        error = HTTPError('url', 500, 'server', {}, io.BytesIO(b'{"error":"prediction aborted, token repeat limit reached"}'))
        with patch('trainer.grpo_poetry_judge.urlopen', side_effect=error) as request:
            with self.assertRaisesRegex(JudgeServiceError, 'token repeat limit reached'):
                self.judge().request('/api/chat', {})
            self.assertEqual(request.call_count, 1)

    def test_temporary_http_error_retries_exact_payload(self):
        error = HTTPError('url', 503, 'loading', {}, io.BytesIO(b'{"error":"loading model"}'))
        response = io.BytesIO(b'{"done":true}')
        with patch('trainer.grpo_poetry_judge.urlopen', side_effect=[error, response]) as request, patch('trainer.grpo_poetry_judge.time.sleep'):
            self.assertEqual(self.judge().request('/api/chat', {'original': True}), {'done': True})
            self.assertEqual(request.call_count, 2)
            self.assertEqual(request.call_args_list[0].args[0].data, request.call_args_list[1].args[0].data)

    def test_transport_retries_are_bounded(self):
        with patch('trainer.grpo_poetry_judge.urlopen', side_effect=URLError('unavailable')) as request, patch('trainer.grpo_poetry_judge.time.sleep'):
            with self.assertRaisesRegex(JudgeServiceError, 'unavailable'):
                self.judge().request('/api/chat', {})
            self.assertEqual(request.call_count, 3)

    def test_invalid_or_truncated_rating_does_not_enter_cache(self):
        for result in [dict(done_reason='length'), dict(message=dict(content='{"coherence":100,"relevance":4,"reason":"bad"}'))]:
            judge = self.judge()
            with patch.object(judge, 'request', return_value=result):
                with self.assertRaises((RuntimeError, ValueError)):
                    judge.rate([dict(content='question')], 'candidate')
            self.assertEqual(judge.cache, {})
