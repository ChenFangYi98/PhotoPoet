# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""只读核验 GRPO 数据隔离、来源保护和下载资源，报告写入 out/grpo。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import json
from collections import Counter
from trainer.grpo_resources import atomic_json, file_digest, reward_files_ready, DEFAULT_REWARD
from trainer.grpo_poetry_data import DEFAULT_DATA

PROTECTED_BEFORE = {
    'dataset/rlaif.jsonl': '8c6634db971fa34b0217f7db4f7c30684f57d20bbb771eb717f1b5aeacb089ba',
    'model/model_mindchen.py': '28a3a890bba0b0e45a5c1889c981e6bcf9922a3e4d7679793cf2a757e453d32b',
    'model/checkpoint_config.py': '0dedc30ffb88476c412169c171c2db21acabedf66d1c6facfb1f77841b1a9c44',
    'dataset/lm_dataset.py': '18de2276f7b652c6c95a417cb604505e2f641020c4e3f84abc33a6a7b5c0c74f',
    'trainer/trainer_utils.py': '0bd4103fdf9601a0c605330cfe4bed0e7b5c810e8437c2cc99997d2707f0d99a',
    'trainer/train_adaptive_sft.py': 'dfd65b1412083cfa8272560836966d58f1cabe0077a6c80f10890aaf777ea3b7',
}


def main():
    report = dict(protected={})
    for name, expected in PROTECTED_BEFORE.items():
        actual = file_digest(ROOT/name)
        report['protected'][name] = dict(sha256=actual, unchanged=actual == expected)
    data = {split: [json.loads(line) for line in DEFAULT_DATA.with_name(split+'.jsonl').read_text(encoding='utf-8').splitlines()]
            for split in ['train', 'valid']}
    prompts, bodies, titles = {}, {}, {}
    for split, rows in data.items():
        prompts[split] = {json.dumps(r['messages'], ensure_ascii=False, sort_keys=True) for r in rows}
        bodies[split] = {r['poetry_spec']['reference_body'] for r in rows if r['poetry_spec'].get('reference_body')}
        titles[split] = {r['poetry_spec']['topic'] for r in rows if r['poetry_spec'].get('reference_body')}
        report[split] = dict(rows=len(rows), unique_prompts=len(prompts[split]), unique_source_poems=len(bodies[split]),
            forms=dict(Counter(r['poetry_spec']['form'] for r in rows)),
            tasks=dict(Counter(r['poetry_spec'].get('task', 'template') for r in rows)),
            sha256=file_digest(DEFAULT_DATA.with_name(split+'.jsonl')))
    report['overlap'] = dict(prompts=len(prompts['train'] & prompts['valid']),
        source_poems=len(bodies['train'] & bodies['valid']), source_titles=len(titles['train'] & titles['valid']))
    report['reward_model_complete'] = reward_files_ready(DEFAULT_REWARD)
    report['passed'] = (all(r['unchanged'] for r in report['protected'].values()) and
                        not any(report['overlap'].values()) and report['reward_model_complete'] and
                        all(report[s]['rows'] == report[s]['unique_prompts'] for s in data))
    destination = ROOT/'out/grpo/preparation_audit.json'
    destination.parent.mkdir(exist_ok=True, parents=True)
    atomic_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
