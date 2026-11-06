# Modified by chenfangyi for the unified MindChen release (2026-11-06).
# -*- coding: utf-8 -*-
"""MindChen 通用与诗词古文能力评估，离线运行。
从项目根目录运行 python eval/eval_mindchen_quality.py，参见 README_使用说明.md。
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCORER_VERSION = 'capability-v2'
REVIEW_COLS = {
    'general': ['事实正确_1to5', '指令完成_1to5', '表达清晰_1to5'],
    'poetry': ['主题契合_1to5', '文学表现_1to5', '语言流畅_1to5', '意象新颖_1to5', '整体质量_1to5'],
    'translation': ['语义忠实_1to5', '信息完整_1to5', '译写自然_1to5'],
    'analysis': ['文本依据_1to5', '分析准确_1to5', '解释充分_1to5'],
    'prose': ['内容完成_1to5', '文言自然_1to5', '结构连贯_1to5'],
    'couplet': ['对仗工整_1to5', '意境关联_1to5', '语言自然_1to5', '声律协调_1to5'],
}
ALL_REVIEW_COLS = list(dict.fromkeys(col for cols in REVIEW_COLS.values() for col in cols))
HANZI = re.compile(r'[\u3400-\u9fff]')


def review_profile(case):
    cat = case.get('category', '')
    if cat in ('古文今译', '今文古译', '文言文改写'):
        return 'translation'
    if cat == '诗词鉴赏':
        return 'analysis'
    if cat.startswith('古文') and case['rule']['type'] == 'manual_only':
        return 'prose'
    if cat == '对联创作':
        return 'couplet'
    return case['domain']


def scorer_fingerprint():
    h = hashlib.sha256()
    for path in (Path(__file__), HERE / 'capability_metrics.py'):
        h.update(path.read_bytes())
    return h.hexdigest()


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as ex:
                    raise ValueError(f'{path}:{n} 格式错误: {ex}') from ex


def cjk_length(text):
    return len(HANZI.findall(text))


def strip_reasoning(response):
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.S)
    if '</think>' in response:
        response = response.rsplit('</think>', 1)[1]
    if '<think>' in response:  # 推理未结束时，不能把推理正文当作正式答案
        response = response.split('<think>', 1)[0]
    return response.strip()


def clean_exact(text):
    text = strip_reasoning(text)
    text = re.sub(r'^\s*(?:答案|答)\s*[：:]\s*', '', text)
    return text.strip(' \n\t\r。.!！,，')


def poetry_lines(response):
    """按句读而非物理换行切分；不会把标题/大段讲解算作诗句。"""
    t = strip_reasoning(response)
    t = re.sub(r'```(?:\w+)?\s*|```', '', t)
    raw = re.split(r'[，,。！？!?；;\n\r]+', t)
    result = []
    for piece in raw:
        piece = piece.strip()
        piece = re.sub(r'^\s*(?:\d+[.)、]|[（(]\d+[）)])\s*', '', piece)
        piece = piece.strip('“”\'\" \t:：')
        if not piece or re.fullmatch(r'《[^》]{1,15}》', piece):
            continue
        result.append(piece)
    return result


def phrase_coverage(text, groups):
    """每个词组是一个概念，组内任一别名命中即得分。"""
    if not groups:
        return None, []
    hits = [any(str(alias) in text for alias in group) for group in groups]
    return 100.0 * sum(hits) / len(hits), hits


def rhyme_proxy(lines, rhyme_positions):
    """现代普通话韵母相等性代理：不是平水韵/词林正韵检测。"""
    if not rhyme_positions or max(rhyme_positions) > len(lines):
        return None, '句数不足，无法比较韵脚'
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return None, '未安装 pypinyin，韵脚指标未参与评分'
    tail_chars = []
    for p in rhyme_positions:
        chars = HANZI.findall(lines[p - 1])
        if not chars:
            return 0.0, '某个押韵句没有汉字'
        tail_chars.append(chars[-1])
    finals = [lazy_pinyin(x, style=Style.FINALS, strict=False)[0] for x in tail_chars]
    if not all(finals):
        return None, '存在无法判断的韵母'
    # 保留具体韵脚供人工复核；全部相等仅意味着普通话韵母相同。
    matching = sum(x == finals[0] for x in finals[1:])
    return round(100 * matching / max(1, len(finals) - 1), 2), '韵脚=' + '/'.join(tail_chars) + ', 现代韵母=' + '/'.join(finals)


def weighted_score(components):
    """每项 (分值或 None, 权重)。None 不视为0，调整有效权重并记录覆盖率。"""
    total = sum(w for _, w in components.values())
    valid = sum(w for v, w in components.values() if v is not None)
    if not valid:
        return None, 0
    value = sum(v * w for v, w in components.values() if v is not None) / valid
    return round(value, 2), round(valid / total * 100, 2)


def grade_answer(case, response):
    response = strip_reasoning(response)
    rule = case['rule']
    typ = rule['type']
    details = {}
    remarks = []
    if typ in ('numeric', 'json', 'constraints', 'choice'):
        from capability_metrics import grade_capability
        return grade_capability(case, response)
    if typ == 'exact':
        expected = [str(x) for x in rule.get('answers', [rule.get('answer', '')])]
        val = 100.0 if clean_exact(response) in [clean_exact(x) for x in expected] else 0.0
        details['准确匹配'] = val
        details['全约束通过'] = val == 100
        if case.get('category') == '名句背诵补全':
            details['参考答案字面出现'] = any(x and x in response for x in expected)
        score, cov = val, 100
    elif typ == 'manual_only':
        score, cov = None, 0
        remarks.append('开放式任务不适合基于关键词的自动评判；请填写人工评分表')
    elif typ == 'contains_all':
        val, hits = phrase_coverage(response, [[x] for x in rule['keywords']])
        details = {'关键词覆盖率': val, '关键词命中': dict(zip(rule['keywords'], hits))}
        score, cov = round(val, 2), 100
        remarks.append('关键词匹配只能反映命中，不能保证事实正确或语义一致')
    elif typ == 'contains_any':
        val = float(100 if any(x in response for x in rule['keywords']) else 0)
        details = {'词汇命中': val}
        score, cov = val, 100
        remarks.append('词汇命中不等于完整事实正确')
    elif typ in ('poem', 'ci'):
        lines = poetry_lines(response)
        pattern = rule.get('pattern') or ([5] * 4 if rule['form'] == '五言绝句' else [7] * 4)
        expected_count = len(pattern)
        count_score = max(0, 100 * (1 - abs(len(lines) - expected_count) / expected_count))
        correct = sum(cjk_length(line) == size for line, size in zip(lines, pattern))
        lengths_score = 100.0 * correct / expected_count
        theme_score, hits = phrase_coverage(response, rule.get('concept_groups', []))
        # 《如梦令》常有叠句，故不以重复惩罚词牌。
        repeat_score = 100 * len(set(lines)) / len(lines) if lines else 0.0
        if typ == 'poem':
            rhyme_score, info = rhyme_proxy(lines, rule.get('rhyme_positions', [2, 4]))
            components = {'句数': (count_score, 20), '每句字数': (lengths_score, 30),
                          '意象关键词': (theme_score, 15), '普通话韵脚代理': (rhyme_score, 20),
                          '句子不重复': (repeat_score, 15)}
            remarks.append(info)
            remarks.append('未自动评估历史平仄、平水韵、对仗、语义美感或创造性')
        else:
            components = {'句数': (count_score, 20), '词牌字数模板': (lengths_score, 50),
                          '意象关键词': (theme_score, 20), '句子不重复': (repeat_score, 10)}
            remarks.append('仅检查指定词牌的字数模板，未验证词谱平仄、换韵及词林正韵')
        score, cov = weighted_score(components)
        details = {k: (round(v, 2) if v is not None else None) for k, (v, w) in components.items()}
        details.update({'切分句子': lines, '实际字数': [cjk_length(x) for x in lines],
                        '期望字数': pattern, '意象命中': hits})
    elif typ == 'wenyan':
        cov_score, hits = phrase_coverage(response, rule['concept_groups'])
        target_len = int(rule.get('max_chars', 40))
        length = cjk_length(response)
        brief = 100.0 if 0 < length <= target_len else (max(0.0, 100 * target_len / max(1, length)))
        changed = 0.0 if not response or clean_exact(response) == clean_exact(rule['source']) else 100.0
        components = {'原意关键词代理': (cov_score, 60), '简洁长度': (brief, 20), '非原文照抄': (changed, 20)}
        score, cov = weighted_score(components)
        details = {k: round(v, 2) for k, (v, w) in components.items()}
        details.update({'实际汉字数': length, '概念命中': hits})
        remarks.append('此项仅评估字面概念覆盖与长度，文言风格和真实语义保留需人工审核')
    else:
        raise ValueError(f"题目 {case['id']} 的 rule.type 不支持：{typ}")
    return {'auto_score': score, 'measured_coverage_pct': cov, 'details': details,
            'notes': remarks}


def load_cases(path, selected):
    cases = list(load_jsonl(path))
    ids = [x['id'] for x in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('测试集 id 必须唯一')
    cases = [x for x in cases if selected == 'all' or x['domain'] == selected]
    if not cases:
        raise ValueError('未筛选到任何题目')
    for c in cases:
        if not all(k in c for k in ('id', 'domain', 'category', 'prompt', 'rule')):
            raise ValueError(f'题目缺少必要字段: {c}')
        if c['rule']['type'] == 'choice':
            if c['rule']['answer'] not in c['rule']['options']:
                raise ValueError(f"选择题答案不在选项中: {c['id']}")
    return cases


def load_model(args):
    # 仅在实际推理模式才导入 torch，score-only 和 merge 可独立运行。
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    source = str(ROOT / 'model') if args.load_from == 'model' else args.load_from
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    dev = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用：请使用 --device cpu 或先修复 PyTorch/CUDA 环境')
    if Path(source).resolve() == (ROOT / 'model').resolve():
        from model.model_mindchen import MindChenConfig, MindChenForCausalLM
        if not args.checkpoint:
            raise ValueError('原生权重模式必须设置 --checkpoint（支持 Windows 绝对路径）')
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'权重不存在: {checkpoint}')
        state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        from capability_metrics import infer_architecture
        inferred = infer_architecture(state)
        for key in ('hidden_size', 'num_hidden_layers', 'use_moe'):
            supplied = getattr(args, key)
            if supplied is not None and int(supplied) != int(inferred[key]):
                raise ValueError(f'{key}={supplied} 与权重推断值 {inferred[key]} 不符')
        from model.checkpoint_config import load_checkpoint_config
        cfg,config_info=load_checkpoint_config(checkpoint,fallback=inferred,
            config_path=getattr(args,'model_config',''),norm_topk_prob=getattr(args,'norm_topk_prob',None),
            expected_architecture=inferred,
            inference_rope_scaling=True if args.inference_rope_scaling else None)
        model = MindChenForCausalLM(cfg)
        model.load_state_dict(state, strict=True)
        print('权重架构：', inferred, flush=True)
        print('权重配置：',config_info,flush=True)
        if config_info['source']=='legacy_fallback':
            print('配置提示：裸权重沿用旧归一化默认值；新 SFT 请保留 .pth.config.json，或显式 --norm_topk_prob 0。',flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=False)
    model = model.to(dev)
    if dev.startswith('cuda'):
        model = model.half()
    else:
        model = model.float()  # CPU float16 往往有兼容或性能问题
    return model.eval(), tokenizer, dev


def format_prompt(case, tokenizer, style):
    prompt = case['prompt']
    if style == 'pretrain':
        if case.get('messages'):
            prompt = '\n'.join(('用户：' if m['role'] == 'user' else '助手：') + m['content']
                               for m in case['messages']) + '\n助手：'
        return (tokenizer.bos_token or '') + prompt
    return tokenizer.apply_chat_template(
        case.get('messages') or [{'role': 'user', 'content': prompt}], tokenize=False,
        add_generation_prompt=True, open_thinking=False)


def generate_one(model, tokenizer, dev, case, args, style, index):
    import torch
    seed = args.seed + 10000 * index + int(hashlib.sha256(case['id'].encode('utf-8')).hexdigest()[:8], 16) % 10000
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prompt = format_prompt(case, tokenizer, style)
    inputs = tokenizer(prompt, return_tensors='pt',
                       add_special_tokens=False).to(dev)
    if inputs['input_ids'].shape[1] > args.max_input_tokens:
        raise ValueError('输入超出 max_input_tokens；拒绝静默截断测试题，请增大限制')
    is_poetry = case['domain'] == 'poetry' and case['rule']['type'] in ('poem', 'ci', 'wenyan', 'manual_only', 'constraints')
    do_sample = args.sample_poetry if is_poetry else args.sample_general
    max_tokens = args.max_poetry_tokens if is_poetry else args.max_general_tokens
    cfg = dict(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
               max_new_tokens=max_tokens, do_sample=do_sample,
               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if do_sample:
        cfg.update(temperature=args.temperature, top_p=args.top_p)
    if dev.startswith('cuda'):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**cfg)
    if isinstance(output, dict):
        output = output['generated_ids']
    if dev.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    prefix_len = inputs['input_ids'].shape[1]
    generated = output[0, prefix_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    from capability_metrics import response_diagnostics
    diagnostics = response_diagnostics(response)
    diagnostics.update(input_tokens=prefix_len,
                       hit_token_limit=len(generated) >= max_tokens and int(generated[-1]) != tokenizer.eos_token_id)
    return response, len(generated), round(elapsed, 4), seed, diagnostics


def perplexity(model, tokenizer, dev, filepath, block_size=512, max_documents=200):
    """对从未训练过的文本逐块计算 token-weighted NLL，统计 PPL。
    互不重叠的块会重置上下文，每块首 token 不计分；跨模型必须使用同 tokenizer 与本方法。
    """
    import torch
    total_loss, total_tokens, docs = 0.0, 0, 0
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('{'):
                try:
                    d = json.loads(line)
                    line = d.get('text', d.get('content', ''))
                except json.JSONDecodeError:
                    pass
            if not isinstance(line, str) or not line:
                continue
            ids = tokenizer(line, add_special_tokens=False)['input_ids']
            if len(ids) < 2:
                continue
            docs += 1
            for i in range(0, len(ids), block_size):
                segment = ids[i:i + block_size]
                if len(segment) < 2:
                    continue
                x = torch.tensor([segment], dtype=torch.long, device=dev)
                with torch.inference_mode():
                    loss = model(input_ids=x, labels=x).loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError('PPL 测试产生非有限 loss，请检查权重/精度')
                n = len(segment) - 1
                total_loss += float(loss.item()) * n
                total_tokens += n
            if docs >= max_documents:
                break
    if not total_tokens:
        raise ValueError('PPL 文件没有可评估 token')
    avg = total_loss / total_tokens
    return {'documents': docs, 'evaluated_tokens': total_tokens,
            'file_sha256': hashlib.sha256(Path(filepath).read_bytes()).hexdigest(),
            'block_size': block_size, 'max_documents': max_documents,
            'mean_nll': round(avg, 6) if avg is not None else None,
            'ppl': round(math.exp(avg), 4) if avg is not None and avg < 700 else None,
            'note': '仅对独立留出集有效；相同Tokenizer与分块方法下才适合横向比较'}


def run_inference(args, cases, folder):
    model, tokenizer, dev = load_model(args)
    style = args.prompt_style
    if style == 'auto':
        native = args.load_from == 'model' or Path(args.load_from).resolve() == (ROOT / 'model').resolve()
        label = (Path(args.checkpoint).stem if native else args.load_from).lower()
        style = 'pretrain' if label.startswith('pretrain') else 'chat'
    import torch
    import transformers
    from importlib.metadata import version, PackageNotFoundError
    try:
        pinyin_version = version('pypinyin')
    except PackageNotFoundError:
        pinyin_version = None
    config = vars(args).copy()
    config.update({'device_resolved': dev, 'prompt_style_resolved': style,
                   'cases_sha256': hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
                   'selected_cases_sha256': hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                   'scorer_sha256': scorer_fingerprint(), 'scorer_version': SCORER_VERSION,
                   'tokenizer_sha256': hashlib.sha256((tokenizer.backend_tokenizer.to_str() +
                       json.dumps(tokenizer.special_tokens_map, sort_keys=True)).encode()).hexdigest(),
                   'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                   'pypinyin_version': pinyin_version, 'model_config': model.config.to_dict(),
                   'timestamp': datetime.now().isoformat(timespec='seconds')})
    (folder / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    write_jsonl(folder / 'selected_cases.jsonl', cases)
    output = folder / 'results.jsonl'
    def sample_count(case):
        return args.poetry_samples if case['rule']['type'] in ('poem', 'ci', 'wenyan', 'manual_only', 'constraints') and case['domain'] == 'poetry' else args.general_samples
    total = sum(sample_count(x) for x in cases)
    completed = 0
    with output.open('w', encoding='utf-8') as stream:
        for case in cases:
            n = sample_count(case)
            for i in range(n):
                completed += 1
                print(f'[{completed}/{total}] {case["id"]} 取样{i + 1} ...', flush=True)
                row = {'attempt_id': f'{case["id"]}#{i + 1}', 'id': case['id'],
                       'sample': i + 1, 'domain': case['domain'],
                       'category': case['category'], 'prompt': case['prompt'],
                       'rule_type': case['rule']['type'], 'review_profile': review_profile(case),
                       'reference': case.get('reference', ''), 'rubric': case.get('rubric', ''),
                       'messages': case.get('messages'),
                       'case_sha256': hashlib.sha256(json.dumps(case, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
                row['likelihood_requested'] = args.choice_likelihood and case['rule']['type'] == 'choice'
                try:
                    if not args.likelihood_only:
                        response, gen_tokens, latency, seed, diagnostics = generate_one(model, tokenizer, dev, case, args, style, i)
                        row.update({'response': response, 'tokens': gen_tokens, 'seconds': latency,
                                'tok_per_s': round(gen_tokens / max(latency, 1e-6), 2),
                                'seed': seed, 'status': 'ok', 'diagnostics': diagnostics})
                        row.update(grade_answer(case, response))
                        print(f'  自动得分：{row["auto_score"]}，输出：{response[:80]!r}', flush=True)
                    else:
                        row.update(response='', auto_score=None, measured_coverage_pct=0, status='ok')
                    if args.choice_likelihood and case['rule']['type'] == 'choice':
                        from capability_metrics import choice_likelihood
                        row['choice_likelihood'] = choice_likelihood(model, tokenizer, dev,
                            format_prompt(case, tokenizer, style), case, args.max_input_tokens)
                        print('  候选字母概率判题：', row['choice_likelihood']['predicted'], flush=True)
                except Exception as exc:
                    row.setdefault('response', '')
                    row.update({'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                                'auto_score': None, 'measured_coverage_pct': 0})
                    print('  出错：', row['error'], flush=True)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()  # 中断后已经评过的行不丢
    if args.ppl_file:
        val = perplexity(model, tokenizer, dev, args.ppl_file,
                         block_size=args.ppl_block_size, max_documents=args.ppl_max_docs)
        (folder / 'ppl.json').write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding='utf-8')
    return list(load_jsonl(output))


def score_only(args, cases, folder):
    answer_map = {c['id']: c for c in cases}
    result = []
    seen = set()
    for index, given in enumerate(load_jsonl(args.responses), 1):
        cid = given.get('id')
        if cid not in answer_map:
            print(f'跳过不存在的题目 ID: {cid}')
            continue
        c = answer_map[cid]
        response = given['response']
        if not isinstance(response, str):
            raise ValueError(f'{cid} 的 response 必须为字符串')
        sample = given.get('sample', 1)
        row = {'attempt_id': given.get('attempt_id', f'{cid}#{sample}'),
               'id': cid, 'sample': sample, 'domain': c['domain'], 'category': c['category'],
               'prompt': c['prompt'], 'response': response, 'status': given.get('status', 'ok')}
        if row['attempt_id'] in seen:
            raise ValueError(f'重复 attempt_id: {row["attempt_id"]}')
        seen.add(row['attempt_id'])
        from capability_metrics import response_diagnostics
        row.update(rule_type=c['rule']['type'], review_profile=review_profile(c),
                   reference=c.get('reference', ''), rubric=c.get('rubric', ''),
                   messages=c.get('messages'), diagnostics=response_diagnostics(response),
                   case_sha256=hashlib.sha256(json.dumps(c, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
        for key in ('tokens', 'seconds', 'tok_per_s', 'seed', 'error', 'likelihood_requested', 'choice_likelihood'):
            if key in given:
                row[key] = given[key]
        if 'diagnostics' in given and 'hit_token_limit' in given['diagnostics']:
            row['diagnostics']['hit_token_limit'] = given['diagnostics']['hit_token_limit']
        if row['status'] == 'error':
            row.update(auto_score=None, measured_coverage_pct=0)
        else:
            row.update(grade_answer(c, response))
        result.append(row)
    if not result:
        raise ValueError('没有与题库匹配的有效回答')
    source_config = Path(args.responses).parent / 'config.json'
    config = json.loads(source_config.read_text(encoding='utf-8')) if source_config.exists() else {}
    actual_ids = {row['id'] for row in result}
    cases = [c for c in cases if c['id'] in actual_ids]
    config.update(mode='score-only', source_responses=str(Path(args.responses).resolve()),
                  scorer_sha256=scorer_fingerprint(), scorer_version=SCORER_VERSION,
                  selected_cases_sha256=hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    (folder / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    write_jsonl(folder / 'selected_cases.jsonl', cases)
    write_jsonl(folder / 'results.jsonl', result)
    return result


def write_jsonl(filepath, rows):
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def make_review_template(rows, folder):
    filename = folder / '人工评分表.csv'
    header = ['attempt_id', 'id', 'domain', 'category', 'review_profile', 'prompt', 'response', 'reference', 'rubric',
              'auto_score'] + ALL_REVIEW_COLS + ['评语']
    with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for x in rows:
            writer.writerow({k: x.get(k, '') for k in header})
    return filename


def merge_manual(rows, manual_csv):
    with open(manual_csv, 'r', encoding='utf-8-sig', newline='') as f:
        reviews = list(csv.DictReader(f))
    review_map = {}
    for review in reviews:
        key = review.get('attempt_id')
        if key in review_map:
            raise ValueError(f'人工评分存在重复 attempt_id: {key}')
        review_map[key] = review
    merged = []
    for x in rows:
        x = dict(x)
        r = review_map.get(x['attempt_id'], {})
        fields = REVIEW_COLS[x.get('review_profile', x['domain'])]
        vals = []
        for field in fields:
            s = str(r.get(field) or '').strip()
            if not s:
                continue
            try:
                num = float(s)
            except ValueError as ex:
                raise ValueError(f'{x["attempt_id"]} 的 {field} 必须填写 1~5') from ex
            if not 1 <= num <= 5:
                raise ValueError(f'{x["attempt_id"]} 的 {field} 必须填写 1~5')
            vals.append(num)
        # 要求同一领域的全部人工项目填写完毕，避免部分评分抬高或压低均分
        human = round(statistics.mean(vals) * 20, 2) if len(vals) == len(fields) else None
        x['manual_score'] = human
        x['human_review_note'] = r.get('评语', '')
        if human is not None and x.get('auto_score') is not None:
            auto_weight = 0.7 if x['domain'] == 'general' else 0.4
            x['combined_score'] = round(auto_weight * x['auto_score'] + (1-auto_weight) * human, 2)
        elif human is not None:
            x['combined_score'] = human  # 纯主观题以人工分作为最终分，不虚构自动评分
        else:
            x['combined_score'] = None
        merged.append(x)
    return merged


def write_csv(rows, folder):
    path = folder / '逐题得分.csv'
    keys = ['attempt_id', 'domain', 'category', 'prompt', 'response', 'status',
            'auto_score', 'measured_coverage_pct', 'manual_score', 'combined_score',
            'tokens', 'seconds', 'tok_per_s', 'choice_likelihood', 'diagnostics', 'details', 'notes', 'error']
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if k in ('details', 'notes', 'diagnostics', 'choice_likelihood') and k in row
                             else row.get(k, '') for k in keys})


def aggregate(rows):
    summary = {'total_attempts': len(rows), 'success_attempts': sum(x['status'] == 'ok' for x in rows),
               'domains': {}, 'categories': {}, 'quality_score_definition':
               '无人工评分时只给自动规则分；人工完整评分后，通用=70%规则+30%人工，诗词=40%规则+60%人工'}
    groups = defaultdict(list)
    for x in rows:
        groups[('domain', x['domain'])].append(x)
        groups[('category', x['category'])].append(x)
    for (kind, name), subset in groups.items():
        scores = [x['auto_score'] for x in subset if x.get('auto_score') is not None]
        human = [x['manual_score'] for x in subset if x.get('manual_score') is not None]
        combined = [x['combined_score'] for x in subset if x.get('combined_score') is not None]
        tps = [x['tok_per_s'] for x in subset if isinstance(x.get('tok_per_s'), (float, int))]
        checked = [x['details']['准确匹配'] for x in subset
                   if isinstance(x.get('details'), dict) and '准确匹配' in x['details']]
        objective = [x for x in subset if x.get('rule_type') in ('exact', 'numeric', 'json', 'choice')
                     and (x.get('auto_score') is not None or x.get('status') == 'error')]
        likelihood = [x for x in subset if x.get('likelihood_requested')]
        strict = [x['details']['全约束通过'] for x in subset if '全约束通过' in x.get('details', {})]
        diagnostics = [x['diagnostics'] for x in subset if 'diagnostics' in x]
        content = [x['details']['选项内容正确'] for x in subset if '选项内容正确' in x.get('details', {})]
        recall = [x['details']['参考答案字面出现'] for x in subset if '参考答案字面出现' in x.get('details', {})]
        formatted = []
        for x in subset:
            d = x.get('details', {})
            if '句数' in d and ('每句字数' in d or '词牌字数模板' in d):
                formatted.append(d['句数'] == 100 and d.get('每句字数', d.get('词牌字数模板')) == 100)
        summary['domains' if kind == 'domain' else 'categories'][name] = {
            'attempts': len(subset), 'scored': len(scores), 'human_reviewed': len(human),
            'auto_mean': round(statistics.mean(scores), 2) if scores else None,
            'auto_std': round(statistics.stdev(scores), 2) if len(scores) > 1 else None,
            'human_mean': round(statistics.mean(human), 2) if human else None,
            'combined_mean': round(statistics.mean(combined), 2) if combined else None,
            'mean_tokens_per_second': round(statistics.mean(tps), 2) if tps else None,
            'objective_exact_accuracy_pct': round(statistics.mean(checked), 2) if checked else None,
            'objective_accuracy_including_errors_pct': round(sum(x.get('auto_score') or 0 for x in objective) / len(objective), 2) if objective else None,
            'objective_attempts': len(objective),
            'choice_content_accuracy_pct': round(statistics.mean(content), 2) if content else None,
            'reference_span_present_rate_pct': round(100 * statistics.mean(recall), 2) if recall else None,
            'choice_likelihood_accuracy_pct': round(100 * sum(x.get('choice_likelihood', {}).get('correct', False) for x in likelihood) / len(likelihood), 2) if likelihood else None,
            'likelihood_attempts': len(likelihood),
            'strict_pass_rate_pct': round(100 * statistics.mean(strict), 2) if strict else None,
            'error_rate_pct': round(100 * sum(x['status'] == 'error' for x in subset) / len(subset), 2),
            'empty_response_rate_pct': round(100 * statistics.mean(x['empty_response'] for x in diagnostics), 2) if diagnostics else None,
            'token_limit_rate_pct': round(100 * statistics.mean(x['hit_token_limit'] for x in diagnostics if 'hit_token_limit' in x), 2) if any('hit_token_limit' in x for x in diagnostics) else None,
            'mean_repeated_char_4gram_ratio': round(statistics.mean(x['repeated_char_4gram_ratio'] for x in diagnostics), 4) if diagnostics else None,
            'poem_form_pass_rate_pct': round(100 * sum(formatted) / len(formatted), 2) if formatted else None,
        }
    category_acc = [s['objective_accuracy_including_errors_pct'] for s in summary['categories'].values()
                    if s['objective_accuracy_including_errors_pct'] is not None]
    summary['objective_category_macro_accuracy_pct'] = round(statistics.mean(category_acc), 2) if category_acc else None
    summary['error_attempts'] = len(rows) - summary['success_attempts']
    summary['manual_pending_attempts'] = sum(x.get('rule_type') == 'manual_only' and x.get('manual_score') is None for x in rows)
    return summary


def report(rows, folder):
    summary = aggregate(rows)
    ppl_path = folder / 'ppl.json'
    if ppl_path.exists():
        summary['heldout_ppl'] = json.loads(ppl_path.read_text(encoding='utf-8'))
    (folder / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(rows, folder)
    lines = ['# MindChen 自动评估报告', '', f'生成时间：{datetime.now().isoformat(timespec="seconds")}',
             f'总输出数：{summary["total_attempts"]}；成功输出：{summary["success_attempts"]}',
             '', '## 分类统计', '', '| 分类 | 样本数 | 自动规则均分 | 标准差 | 客观题准确率% | 诗词格式通过率% | 人工已评 | 综合分 | tokens/s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: '未评估' if v is None else str(v)
    for name, s in {**summary['domains'], **summary['categories']}.items():
        lines.append(f'| {name} | {s["attempts"]} | {fmt(s["auto_mean"])} | {fmt(s["auto_std"])} | {fmt(s["objective_exact_accuracy_pct"])} | {fmt(s["poem_form_pass_rate_pct"])} | {s["human_reviewed"]} | {fmt(s["combined_mean"])} | {fmt(s["mean_tokens_per_second"])} |')
    lines += ['', '## 能力与输出诊断', '',
              f'客观题分类宏平均准确率（错误输出计错）：{fmt(summary["objective_category_macro_accuracy_pct"])}%；待人工评价的开放题输出：{summary["manual_pending_attempts"]}。',
              '', '| 分类 | 客观准确率%（含推理错误） | 选项内容正确% | 候选字母概率准确率% | 全约束通过% | 出错% | 空输出% | 到长度上限% | 字符4-gram重复率 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, s in summary['categories'].items():
        keys = ('objective_accuracy_including_errors_pct', 'choice_content_accuracy_pct',
                'choice_likelihood_accuracy_pct', 'strict_pass_rate_pct', 'error_rate_pct',
                'empty_response_rate_pct', 'token_limit_rate_pct', 'mean_repeated_char_4gram_ratio')
        lines.append('| ' + name + ' | ' + ' | '.join(fmt(s[k]) for k in keys) + ' |')
    memory = summary['categories'].get('名句背诵补全', {})
    if memory:
        lines += ['', f'名句补全：参考答案字面出现率 {fmt(memory.get("reference_span_present_rate_pct"))}%。这只是检查正确片段是否出现，可与“只输出下一句”的严格匹配区分；不验证片段的归属、顺序或否定关系。']
    if 'heldout_ppl' in summary:
        p = summary['heldout_ppl']
        lines += ['', '## 留出集困惑度', '', f'- PPL：{fmt(p["ppl"])}；token 数：{p["evaluated_tokens"]}；文档数：{p["documents"]}']
    lines += ['', '## 指标解释与限制', '',
              '- 自动规则分用于检查精确答案、关键概念、字数、句数和（可用时）现代普通话韵母；不直接等价于真实语言/文学质量。',
              '- 诗词平仄、古韵、意境、流畅度、原创性需人工评分或经过验证的独立评审模型。',
              '- 如未装 pypinyin，诗词韵脚权重被排除并重新归一，跨运行对比必须保持环境一致。',
              '- 诗词综合分=40%自动规则+60%人工；通用综合分=70%自动规则+30%人工。人工分未填齐时不计算综合分。',
              '- 此处提供的题库是功能演示集，并非标准化学术基准；正式实验请扩展并冻结独立未见题库。',
              '- 诗词领域均分混合了格式代理、理解客观题，不能当作文学总分；优先按分类比较。名句补全侧重记忆，不代表原创。',
              '- 候选字母概率准确率比较固定提示后各选项字母的总条件log概率；四选一随机参考值25%，与自由生成准确率分开解读。',
              '- 字符4-gram重复率、空输出、长度上限是退化诊断；重复率并非作品多样性或质量的直接评分。',
              '- 文言今译、今文古译、鉴赏、古文写作、对联使用各自人工维度，参考答案与要点不进入模型提示。',
              '', '## 得分最低的输出（方便排查）', '']
    ordered = sorted([x for x in rows if x.get('auto_score') is not None], key=lambda x: x['auto_score'])[:10]
    for x in ordered:
        txt = strip_reasoning(x.get('response', '')).replace('\n', ' / ')[:100]
        lines.append(f'- **{x["attempt_id"]}**（{x["category"]}，{x["auto_score"]}分）：{txt}')
    (folder / '评估报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description='MindChen 通用能力/古典诗词质量评估与保存')
    p.add_argument('--mode', choices=['infer', 'score-only', 'merge'], default='infer',
                   help='infer加载模型推理；score-only对现有答案打分；merge合并人工评分')
    p.add_argument('--cases', default=str(HERE / 'eval_cases.jsonl'))
    p.add_argument('--domain', choices=['all', 'general', 'poetry'], default='all')
    p.add_argument('--categories', default='', help='分类名，多个用英文逗号分隔')
    p.add_argument('--stratified', action='store_true', help='按分类轮流取题，使 --limit 覆盖多种能力')
    p.add_argument('--list_cases', action='store_true', help='列出题库分类数量，不加载模型')
    p.add_argument('--limit', type=int, default=0, help='调试时只运行前N个案例；0为全部')
    p.add_argument('--output_root', default=str(HERE / 'evaluation_results'))
    p.add_argument('--run_name', default='')
    p.add_argument('--responses', default='', help='score-only：JSONL，行中有 id,response')
    p.add_argument('--input_results', default='', help='merge：results.jsonl 的路径')
    p.add_argument('--manual_csv', default='', help='merge：人工评分表.csv 的路径')
    p.add_argument('--load_from', default='model')
    p.add_argument('--checkpoint', default=str(ROOT / 'out/continue_pretrain/poetry_gujilab_20261022_190751_427063/pretrain_poetry_gujilab_768_moe_best.pth'), help='原生 .pth 权重路径')
    p.add_argument('--model_config',default='',help='默认读取 .pth.config.json；可显式指定完整配置')
    p.add_argument('--norm_topk_prob',type=int,choices=(0,1),default=None,help='无配置权重的显式设置，有保存配置时禁止冲突')
    p.add_argument('--hidden_size', type=int, default=None, help='默认由权重推断；显式值用于核对')
    p.add_argument('--num_hidden_layers', type=int, default=None)
    p.add_argument('--use_moe', type=int, choices=[0, 1], default=None)
    p.add_argument('--inference_rope_scaling', action='store_true')
    p.add_argument('--device', default='auto')
    p.add_argument('--prompt_style', choices=['auto', 'pretrain', 'chat'], default='auto')
    p.add_argument('--choice_likelihood', action='store_true', help='额外计算选择题候选字母概率准确率')
    p.add_argument('--likelihood_only', action='store_true', help='只对选择题计算概率，不生成文本')
    p.add_argument('--general_samples', type=int, default=1)
    p.add_argument('--poetry_samples', type=int, default=1)
    p.add_argument('--sample_general', action='store_true', help='启用通用题随机采样（默认greedy）')
    p.add_argument('--no_sample_poetry', action='store_false', dest='sample_poetry', help='诗词改greedy')
    p.set_defaults(sample_poetry=True)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--max_input_tokens', type=int, default=1024)
    p.add_argument('--max_general_tokens', type=int, default=128)
    p.add_argument('--max_poetry_tokens', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ppl_file', default='', help='独立留出的纯文本语料；每行一个文档')
    p.add_argument('--ppl_block_size', type=int, default=512)
    p.add_argument('--ppl_max_docs', type=int, default=200)
    args = p.parse_args()
    if args.limit < 0 or args.max_input_tokens < 1 or min(args.max_general_tokens, args.max_poetry_tokens) < 1:
        p.error('limit 必须 >= 0，token 长度必须 >= 1')
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.ppl_block_size < 2 or args.ppl_max_docs < 1:
        p.error('temperature > 0、0 < top_p <= 1、ppl_block_size >= 2、ppl_max_docs >= 1')
    if args.likelihood_only:
        args.choice_likelihood = True
    if args.mode == 'merge':
        if not args.input_results or not args.manual_csv:
            p.error('merge 模式需要 --input_results 和 --manual_csv')
        original = Path(args.input_results)
        rows = merge_manual(list(load_jsonl(original)), args.manual_csv)
        folder = original.parent  # 合并到原运行目录，保留 config/ppl，不改原始 results.jsonl
        write_jsonl(folder / 'reviewed_results.jsonl', rows)
        summary = report(rows, folder)
    else:
        if args.mode == 'score-only' and not args.responses:
            p.error('score-only 需要 --responses')
        if args.general_samples < 1 or args.poetry_samples < 1:
            p.error('samples 必须 >= 1')
        cases = load_cases(args.cases, args.domain)
        from capability_metrics import select_cases
        if args.likelihood_only:
            cases = [c for c in cases if c['rule']['type'] == 'choice']
        cases = select_cases(cases, args.categories, args.limit, args.stratified)
        if not cases:
            p.error('筛选后没有题目')
        if args.list_cases:
            from collections import Counter
            for category, count in Counter(c['category'] for c in cases).items():
                print(f'{category}: {count}')
            print(f'总题数: {len(cases)}')
            return
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = args.run_name or (('pretrain' if 'pretrain' in args.checkpoint.lower() else 'model') + '_' + now)
        name = re.sub(r'[^\w.\-]+', '_', name).strip('._')
        folder = Path(args.output_root) / name
        if folder.exists():
            p.error(f'结果目录已存在，请更换 --run_name：{folder}')
        folder.mkdir(parents=True)
        if args.mode == 'infer':
            rows = run_inference(args, cases, folder)
        else:
            rows = score_only(args, cases, folder)
        make_review_template(rows, folder)
        summary = report(rows, folder)
    print('\n======== 测试汇总 ========')
    for domain, s in summary['domains'].items():
        print(f'{domain}: 自动规则均分={s["auto_mean"]} / 100, 人工综合分={s["combined_mean"]}, 有效输出数={s["scored"]}')
    print('报告目录：', folder.resolve())
    print('逐题答案 results.jsonl | 分数 逐题得分.csv | 汇总 评估报告.md | 人工表 人工评分表.csv')
    if rows and all(x['status'] == 'error' for x in rows):
        raise SystemExit('全部推理失败，请检查 results.jsonl 中的 error 字段')


if __name__ == '__main__':
    main()
