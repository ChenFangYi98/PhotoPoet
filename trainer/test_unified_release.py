"""Compatibility checks for the shared language/vision release."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import datasets
import torch
from torch import nn
from model.checkpoint_config import load_checkpoint_config, weight_sha256
from model.checkpoint_compat import SUPPORTED_CONFIG_SCHEMAS, LEGACY_PROJECT_DIR_NAMES
from model.model_mindchen import MindChenConfig
from trainer.project_paths import relocate_project_path
from trainer.trainer_utils import lm_checkpoint, vlm_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class UnifiedReleaseTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'out').mkdir(exist_ok=True)
        folder = tempfile.TemporaryDirectory(prefix='unified_unit_', dir=ROOT / 'out')
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_shared_saver_preserves_language_and_visual_resume_conventions(self):
        model = nn.Module()
        model.body = nn.Linear(3, 4)
        model.vision_encoder = nn.Linear(3, 4)
        optimizer = torch.optim.AdamW(model.parameters())
        config = SimpleNamespace(use_moe=False, hidden_size=4)
        for function, name in [(lm_checkpoint, 'llm'), (vlm_checkpoint, 'vlm')]:
            function(config, weight=name, model=model, optimizer=optimizer,
                     save_dir=self.folder, epoch=2, step=7, extra_signature='kept')
            inference = torch.load(self.folder / f'{name}_4.pth', weights_only=True)
            resume = torch.load(self.folder / f'{name}_4_resume.pth', weights_only=True)
            self.assertEqual(resume['step'], 7)
            self.assertEqual(resume['extra_signature'], 'kept')
            self.assertEqual(inference['body.weight'].dtype, torch.float16)
            self.assertIn('vision_encoder.weight', resume['model'])
            self.assertEqual('vision_encoder.weight' in inference, name == 'llm')
            self.assertEqual(resume['model']['body.weight'].dtype,
                             torch.float16 if name == 'llm' else torch.float32)

    def test_historical_sidecar_retains_routing_and_hash_binding(self):
        path = self.folder / 'old.pth'
        torch.save({'weight': torch.ones(2, 2)}, path)
        config = MindChenConfig(hidden_size=32, num_hidden_layers=1, norm_topk_prob=False)
        payload = dict(schema=SUPPORTED_CONFIG_SCHEMAS[-1], weight_sha256=weight_sha256(path),
                       model_config=config.to_dict())
        Path(str(path) + '.config.json').write_text(json.dumps(payload), encoding='utf8')
        loaded, metadata = load_checkpoint_config(path)
        self.assertFalse(loaded.norm_topk_prob)
        self.assertTrue(metadata['weight_bound'])
        torch.save({'weight': torch.zeros(2, 2)}, path)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            load_checkpoint_config(path)

    def test_historical_project_root_relocation_when_release_root_has_new_name(self):
        current = self.folder / 'renamed-project'
        for name in LEGACY_PROJECT_DIR_NAMES:
            source = f'Z:/example/{name}/out/base.pth'
            self.assertEqual(Path(relocate_project_path(source, current)), current.resolve() / 'out/base.pth')
        external = 'Z:/independent/data/train.jsonl'
        self.assertEqual(relocate_project_path(external, current), external)


if __name__ == '__main__':
    unittest.main()
