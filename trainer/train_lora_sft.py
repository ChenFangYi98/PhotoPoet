# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Short, resumable LoRA SFT on a native language checkpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.lora_training import main

if __name__ == '__main__':
    main('llm')
