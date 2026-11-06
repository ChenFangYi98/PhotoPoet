# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Supervisor recovery never changes model state or accepts an invalid reward."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('grpo_recovery', ROOT/'out/session_control/grpo_recovery.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
FAILURE = ['File "trainer/grpo_poetry_judge.py", line 43\n',
           'urllib.error.HTTPError: HTTP Error 500: Internal Server Error\n']


class RecoveryTests(unittest.TestCase):
    def run_case(self, folder, runner, **overrides):
        self.update, self.emit, self.sleep = Mock(), Mock(), Mock()
        return recovery.run_with_recovery(['original'], 'grpo', folder, 'python', ROOT,
            runner, self.update, self.emit, sleep=self.sleep, **overrides)

    def test_transient_failure_resumes_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'resume.pth'
            checkpoint.write_bytes(b'immutable checkpoint')
            runner = Mock(side_effect=[(1, FAILURE), (0, [])])
            self.assertEqual(self.run_case(tmp, runner), 0)
            command = runner.call_args_list[1].args[0]
            self.assertEqual(command[command.index('--resume')+1], str(checkpoint))
            self.assertEqual(checkpoint.read_bytes(), b'immutable checkpoint')
            self.sleep.assert_called_once_with(5)

    def test_retry_budget_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'resume.pth').touch()
            runner = Mock(return_value=(1, FAILURE))
            self.assertEqual(self.run_case(tmp, runner), 1)
            self.assertEqual(runner.call_count, 4)
            self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [5, 15, 45])

    def test_no_restart_without_checkpoint_or_with_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            for checkpoint, stop in [(False, False), (True, True)]:
                if checkpoint:
                    (Path(tmp)/'resume.pth').touch()
                if stop:
                    (Path(tmp)/'STOP').touch()
                runner = Mock(return_value=(1, FAILURE))
                self.assertEqual(self.run_case(tmp, runner), 1)
                self.assertEqual(runner.call_count, 1)

    def test_stop_during_backoff_prevents_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'resume.pth').touch()
            runner = Mock(return_value=(1, FAILURE))
            def sleep(_):
                (Path(tmp)/'STOP').touch()
            self.assertEqual(recovery.run_with_recovery(['original'], 'grpo', tmp, 'python', ROOT,
                runner, Mock(), Mock(), sleep=sleep), 1)
            self.assertEqual(runner.call_count, 1)

    def test_consumed_stop_marker_still_prevents_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'resume.pth').touch()
            def runner(command):
                (Path(tmp)/'STOP.handled.during_failure').touch()
                return 1, FAILURE
            mocked = Mock(side_effect=runner)
            self.assertEqual(self.run_case(tmp, mocked), 1)
            self.assertEqual(mocked.call_count, 1)

    def test_invalid_rating_signature_and_unrelated_errors_are_not_retried(self):
        for tail in [['grpo_poetry_judge.py\n', 'ValueError: Invalid local judge coherence rating'],
                     ['ValueError: code signature changed'],
                     ['other_service.py\n', 'HTTP Error 500: Internal Server Error'],
                     ['grpo_poetry_judge.py\n', 'HTTP Error 400: Bad Request']]:
            self.assertFalse(recovery.transient_judge_failure(tail))
        self.assertFalse(recovery.transient_judge_failure(FAILURE+['prediction aborted, token repeat limit reached']))
        self.assertTrue(recovery.transient_judge_failure(['grpo_poetry_judge.py\n', 'Local judge HTTP 503: temporarily unavailable']))


if __name__ == '__main__':
    unittest.main()
