# Modified by chenfangyi for the unified MindChen release (2026-11-06).
# -*- coding: utf-8 -*-
"""合并多轮 MindChen 评估结果；要求相同题目集合与评分设置。"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_run(path):
    path = Path(path).resolve()
    records_file = path / 'reviewed_results.jsonl'
    if not records_file.exists():
        records_file = path / 'results.jsonl'
    data = [json.loads(line) for line in records_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
    config_file = path / 'config.json'
    config = json.loads(config_file.read_text(encoding='utf-8')) if config_file.exists() else {}
    ids = {(x['attempt_id'], x.get('prompt', ''), json.dumps(x.get('messages'), ensure_ascii=False, sort_keys=True)) for x in data}
    if len({x['attempt_id'] for x in data}) != len(data):
        raise ValueError(f'{path} 中 attempt_id 重复')
    return path.name, data, summary, config, ids


def compatibility_errors(runs, allow_different_cases=False):
    errors = []
    if not allow_different_cases and any(r[4] != runs[0][4] for r in runs[1:]):
        errors.append('题目、提示、多轮历史或采样次数不同')
    keys = ['scorer_sha256', 'tokenizer_sha256', 'pypinyin_version', 'seed',
            'sample_general', 'sample_poetry', 'temperature', 'top_p',
            'max_input_tokens', 'max_general_tokens', 'max_poetry_tokens',
            'general_samples', 'poetry_samples', 'choice_likelihood', 'likelihood_only']
    if not allow_different_cases:
        keys.append('selected_cases_sha256')
    for key in keys:
        values = [r[3].get(key) for r in runs]
        if any(v != values[0] for v in values[1:]):
            errors.append(f'{key} 不一致（含缺失元数据）')
    if any(not r[3].get('scorer_sha256') or not r[3].get('selected_cases_sha256') for r in runs):
        errors.append('缺少评分器或冻结题库指纹，请用新版脚本重算')
    present = [r[2]['heldout_ppl'] for r in runs if r[2].get('heldout_ppl')]
    if len(present) > 1:
        for key in ('file_sha256', 'block_size', 'max_documents', 'evaluated_tokens'):
            if not all(key in val for val in present) or any(val[key] != present[0][key] for val in present[1:]):
                errors.append(f'PPL 的 {key} 不一致或缺失')
    return errors


def paired_deltas(baseline, candidate, metric='auto_score', bootstrap=2000):
    def collapse(data):
        groups = defaultdict(list)
        for row in data:
            value = row.get(metric)
            if metric == 'choice_likelihood':
                value = 100.0 * row.get(metric, {}).get('correct', False) if row.get('likelihood_requested') else None
            if metric == 'auto_score' and row['status'] == 'error' and row.get('rule_type') != 'manual_only':
                value = 0.0
            if value is not None:
                signature = row.get('case_sha256') or json.dumps(
                    [row.get('prompt'), row.get('messages')], ensure_ascii=False, sort_keys=True)
                groups[(row['category'], row['id'], signature)].append(value)
        return {k: mean(v) for k, v in groups.items()}
    a, b = collapse(baseline), collapse(candidate)
    groups = defaultdict(list)
    for key in sorted(a.keys() & b.keys()):
        groups[key[0]].append(b[key] - a[key])
    results = []
    for category, deltas in groups.items():
        n = len(deltas)
        rng = random.Random(42)
        if n > 1:
            boot = sorted(mean(rng.choices(deltas, k=n)) for _ in range(bootstrap))
            low, high = boot[int(0.025 * bootstrap)], boot[min(bootstrap - 1, int(0.975 * bootstrap))]
        else:
            low = high = None
        results.append({'分类': category, '指标': metric, '配对题数': n,
                        '候选减基线': round(mean(deltas), 2),
                        '差值95%下界': round(low, 2) if low is not None else None,
                        '差值95%上界': round(high, 2) if high is not None else None,
                        '提升题数': sum(d > 0 for d in deltas), '持平题数': sum(d == 0 for d in deltas),
                        '退化题数': sum(d < 0 for d in deltas)})
    return results


def write_table(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description='对比不同 checkpoint 的模型评估')
    p.add_argument('run_dirs', nargs='+', help='至少两个运行结果目录')
    p.add_argument('--output', default='model_comparison.csv')
    p.add_argument('--allow_different_cases', action='store_true',
                   help='允许不同题目集合（分数不可直接横向比较，默认禁止）')
    p.add_argument('--allow_incompatible', action='store_true', help='明确跳过一致性检查，仅作探索')
    args = p.parse_args()
    if len(args.run_dirs) < 2:
        p.error('至少要提供两个评估目录')
    runs = [load_run(s) for s in args.run_dirs]
    problems = compatibility_errors(runs, args.allow_different_cases)
    if problems and not args.allow_incompatible:
        p.error('；'.join(problems))
    warnings = list(problems)
    if any(not r[3].get('tokenizer_sha256') for r in runs):
        warnings.append('存在离线导入且缺少tokenizer/推理元数据的运行，只能核对已提供信息')
    if args.allow_different_cases:
        warnings.append('不同题集只作探索；配对表只保留共有题目')
    if len({r[3].get('prompt_style_resolved') for r in runs}) > 1:
        warnings.append('提示模板不同，分数同时受到模板与权重影响；可另用同一模板复测')
    columns = ['运行名称', 'checkpoint', '输入模式', '样本数量',
               '通用自动分', '五言自动分', '七言自动分', '宋词自动分', '文言自动分',
               '诗词自动分', '通用综合分', '诗词综合分', 'PPL', '平均速度', '客观分类宏平均%', '比较说明']
    rows = []
    for name, data, s, cfg, ids in runs:
        cats = s['categories']
        doms = s['domains']
        ppl = s.get('heldout_ppl', {}).get('ppl', None)
        allspeed = [x['tok_per_s'] for x in data if x.get('tok_per_s') is not None]
        row = dict(zip(columns, [name, cfg.get('checkpoint', cfg.get('load_from', '未知')),
                                 cfg.get('prompt_style_resolved', '未知'), len(data),
                                 doms.get('general', {}).get('auto_mean'),
                                 cats.get('五言绝句', {}).get('auto_mean'),
                                 cats.get('七言绝句', {}).get('auto_mean'),
                                 cats.get('宋词词牌', {}).get('auto_mean'),
                                 cats.get('文言文改写', {}).get('auto_mean'),
                                 doms.get('poetry', {}).get('auto_mean'),
                                 doms.get('general', {}).get('combined_mean'),
                                 doms.get('poetry', {}).get('combined_mean'),
                                 ppl, round(mean(allspeed), 2) if allspeed else None,
                                 s.get('objective_category_macro_accuracy_pct'), '；'.join(warnings)]))
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    category_rows, paired = [], []
    for name, data, summary, cfg, ids in runs:
        for category, s in summary['categories'].items():
            category_rows.append({'运行名称': name, '分类': category, '输出数': s['attempts'],
                '自动分': s['auto_mean'], '人工分': s.get('human_mean'), '人工已评': s.get('human_reviewed'),
                '综合分': s.get('combined_mean'), '客观准确率%（含错误）': s.get('objective_accuracy_including_errors_pct'),
                '名句参考片段出现率%': s.get('reference_span_present_rate_pct'),
                '概率判题准确率%': s.get('choice_likelihood_accuracy_pct'),
                '全约束通过率%': s.get('strict_pass_rate_pct'), '格式通过率%': s.get('poem_form_pass_rate_pct'),
                '错误率%': s.get('error_rate_pct'), '空回答率%': s.get('empty_response_rate_pct'),
                '长度上限率%': s.get('token_limit_rate_pct'), '字符4gram重复率': s.get('mean_repeated_char_4gram_ratio')})
    for run in runs[1:]:
        for metric in ('auto_score', 'choice_likelihood', 'manual_score', 'combined_score'):
            for row in paired_deltas(runs[0][1], run[1], metric):
                paired.append({'基线': runs[0][0], '候选': run[0], **row})
    write_table(output.with_name(output.stem + '_分类.csv'), category_rows)
    write_table(output.with_name(output.stem + '_配对.csv'), paired)
    notes = ['# 比较说明', '', '首个运行作为基线，差值=候选−基线。',
             '配对表先平均同题重复采样，再按题目自助采样2000次，估计差值95%区间。',
             '小题集及共享模板/题干可能违反独立性，区间仅供探索；单题不估计区间。',
             '人工分仅比较双方已完整评分的共有题，请检查配对题数和人工覆盖率。',
             '诗词自动分混合格式代理与理解客观题，不能解释为文学总能力，请优先看分类表。', '']
    notes += ['- ' + warning for warning in warnings]
    output.with_suffix('.md').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(f'模型对比结果：{output.resolve()}')
    for r in rows:
        print(r['运行名称'], '通用：', r['通用自动分'], '诗词：', r['诗词自动分'], 'PPL：', r['PPL'])
    print('注意：PPL 还要求相同的留出集、Tokenizer 和分块设置；提示模板不同也会影响指令测试可比性。')

if __name__ == '__main__':
    main()
