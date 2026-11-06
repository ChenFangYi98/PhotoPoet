# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Text or image-text LoRA SFT on the language backbone of an existing VLM."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.lora_training import main

if __name__ == '__main__':
    main('vlm')
