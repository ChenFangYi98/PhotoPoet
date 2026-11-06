# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""流式扫描SFT全库：角色、诗词任务模板及模型身份，不把大文件装入内存。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent.parent
POETRY_SYSTEM = '你是一位擅长中华古典文学的中文写作助手。请按照用户要求回答，引用和续写已有作品时保持原文，不将已有作品声称为原创。'
BRANDS = re.compile(r'MiniMind|jingyaogong|Jingyao\s+Gong|龚靖尧|mindchen|ChatGPT|OpenAI|GPT[- ]?[2345]|Claude|Anthropic|DeepSeek|深度求索|通义千问|Qwen|智谱|ChatGLM|文心一言|百度|阿里巴巴|腾讯混元|豆包', re.I)


def audit(paths, output):
    report = {}
    for split, path in paths.items():
        counts, brands, tasks, systems = Counter(), Counter(), Counter(), Counter()
        examples = defaultdict(list)
        started = time.perf_counter()
        with Path(path).open(encoding='utf-8-sig') as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    counts['blank_lines'] += 1
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counts['invalid_json'] += 1
                    continue
                counts['rows'] += 1
                messages = row.get('conversations', [])
                poetry = bool(messages and messages[0].get('content') == POETRY_SYSTEM)
                if poetry:
                    counts['fixed_poetry_system_rows'] += 1
                    user = next((m.get('content', '') for m in messages if m.get('role') == 'user'), '')
                    tasks[user.split('\n')[0]] += 1
                    counts['first_line_hints'] += '首句提示：' in user
                previous_user = ''
                for msg in messages:
                    role = msg.get('role', 'missing')
                    counts['role_' + role] += 1
                    if role == 'system':
                        systems[msg.get('content', '')[:500]] += 1
                    if role == 'user':
                        previous_user = msg.get('content', '')
                    for field in ('content', 'reasoning_content'):
                        value = msg.get(field)
                        if not isinstance(value, str):
                            continue
                        hits = set(m.group().casefold() for m in BRANDS.finditer(value))
                        for brand in hits:
                            key = f'{role}/{field}/{brand}'
                            brands[key] += 1
                            if len(examples[key]) < 6:
                                match = next(m for m in BRANDS.finditer(value) if m.group().casefold() == brand)
                                examples[key].append(dict(line=line_no, user=previous_user[:400], text=value[max(0, match.start()-150):match.end()+350]))
                if line_no % 250000 == 0:
                    print(f'{split}: {line_no:,} rows, {time.perf_counter()-started:.1f}s', flush=True)
        report[split] = dict(path=str(Path(path).resolve()), bytes=Path(path).stat().st_size,
            counts=dict(counts), brands=dict(brands), poetry_task_templates=dict(tasks),
            common_systems=systems.most_common(25), identity_examples=dict(examples), seconds=time.perf_counter()-started)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{split}: finished {counts["rows"]:,}', flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir', type=Path, default=ROOT/'dataset/sft_t2t_with_poetry')
    p.add_argument('--output', type=Path, default=ROOT/'dataset/sft_quality_audit.json')
    args = p.parse_args()
    audit({s: args.input_dir/f'sft_t2t_with_poetry_{s}.jsonl' for s in ('valid', 'train')}, args.output)
