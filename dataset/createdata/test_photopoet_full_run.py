# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Quota/recovery regression checks using a temporary dataset, not user data."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import requests
import photopoet_build as build
import photopoet_config as config

class FullRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = {**config.DEFAULTS, 'targets': {'pretrain': 1, 'sft': 40}, 'teacher_http_retries': 2}
        self.patches = [patch.object(build, 'ROOT', Path(self.temp.name)),
                        patch.object(build, 'settings', lambda: self.cfg),
                        patch.object(config, 'settings', lambda: self.cfg)]
        for item in self.patches: item.start()
        self.db = build.database()
    def tearDown(self):
        self.db.close()
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()
    def test_exact_stage_and_task_quotas_survive_reopen(self):
        for i in range(3):
            build.save_sample(self.db, {'id': str(i)}, 'pretrain', 'caption', 'prompt', chr(0x4e00+i), {})
        self.assertEqual(config.count(self.db, 'pretrain'), 1)
        total = 0
        for task, quota in config.task_quotas().items():
            for i in range(quota + 3):
                build.save_sample(self.db, {'id': f'{task}{i}'}, 'sft', task, 'prompt', chr(0x5000+total), {})
                total += 1
            n = self.db.execute('SELECT count(*) FROM samples WHERE task=?', (task,)).fetchone()[0]
            self.assertEqual(n, quota)
        self.assertEqual(config.count(self.db, 'sft'), 40)
        self.assertTrue(config.reached(self.db))
        self.db.close(); self.db = build.database()
        self.assertEqual(config.count(self.db, 'sft'), 40)
        self.db.execute("DELETE FROM samples WHERE id=(SELECT id FROM samples WHERE stage='sft' LIMIT 1)")
        self.db.commit()
        self.assertEqual(config.count(self.db, 'sft'), 39)
        self.assertFalse(config.reached(self.db))
    def test_transient_teacher_500_is_retried(self):
        self.db.execute('INSERT INTO assets(id,path,sha,phash,cluster_id,split,provenance,attempts) VALUES(?,?,?,?,?,?,?,?)', ('photo','unused','hash','0','cluster','train','{}',1)); self.db.commit()
        teacher = build.Teacher.__new__(build.Teacher)
        teacher.db = self.db; teacher.model = 'test'; teacher.digest = 'test'; teacher.endpoint = 'http://localhost'
        bad = Mock(); bad.raise_for_status.side_effect = requests.HTTPError('500')
        good = Mock(); good.raise_for_status.return_value = None
        good.json.return_value = {'done_reason': 'stop', 'message': {'content': '{"ok":true}'}}
        with patch.object(build.requests, 'post', side_effect=[bad,good]) as post, patch.object(build.time, 'sleep') as sleep:
            self.assertTrue(teacher.ask('photo','facts','prompt')['ok'])
            self.assertEqual(post.call_count, 2); sleep.assert_called_once()
        self.assertEqual(self.db.execute('SELECT count(*) FROM calls').fetchone()[0], 1)
    def test_unchanged_shard_is_reused_and_manifest_matches(self):
        from PIL import Image
        import photopoet_export as exporter
        root = Path(self.temp.name)
        path = root / 'image.jpg'; Image.new('RGB', (128,128), (10,20,30)).save(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.db.execute('INSERT INTO assets(id,path,sha,phash,cluster_id,split,provenance) VALUES(?,?,?,?,?,?,?)', ('photo','image.jpg',sha,'0','cluster','train','{}')); self.db.commit()
        build.save_sample(self.db, {'id':'photo'}, 'pretrain', 'caption', '<image>\nprompt', '测试图片', {})
        with patch.object(exporter, 'ROOT', root), patch.object(exporter, 'settings', lambda: self.cfg):
            exporter.export_dataset(self.db)
            parquet = root / '06_final/pretrain/train/photopoet_pretrain_00000.parquet'
            before = parquet.stat().st_mtime_ns
            exporter.export_dataset(self.db)
            after = parquet.stat().st_mtime_ns
            self.assertEqual(before, after)
            status = json.loads((root/'06_final/manifest/status.json').read_text())
            self.assertEqual(status['shards_written_this_export'], 0)
            exporter.verify_dataset()

if __name__ == '__main__': unittest.main()
