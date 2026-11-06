# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""可复现的SFT规则清洗：任务/答案一致、无提示背诵、身份统一、分割保护与去重。

默认读取原混合库，输出新目录，绝不就地覆盖原数据或启动训练。
这里只验证可计算的规则，不把自动清洗声称为全库事实/文学质量人工审核。
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.audit_sft_dataset import POETRY_SYSTEM

RECITE = '请背诵以下作品的原文，只输出正文。'
IDENTITY_SYSTEM = '你是mindchen，由chen开发的AI助手。请准确、清楚地回答用户问题。'
ZH_IDENTITY = '我是mindchen，由chen开发的AI助手。我可以帮助你解答问题、整理信息和进行中文写作，包括诗词古文相关任务。'
EN_IDENTITY = 'I am mindchen, an AI assistant developed by chen. I can help answer questions, organize information, and assist with writing.'
OLD_NAMES = re.compile(r'(?<![A-Za-z0-9_])mini[-_ ]?mind(?![A-Za-z0-9_])|jingyaogong|jingyao\s+gong|龚靖尧|龚敬尧', re.I)
NAME_ONLY = re.compile(r'(?<![A-Za-z0-9_])mini[-_ ]?mind(?![A-Za-z0-9_])', re.I)
PROTECTED = re.compile(r'```[\s\S]*?```|`[^`\n]+`|https?://[^\s<>）)]+')
VENDORS = r'(?:ChatGPT|OpenAI|GPT[- ]?[2345](?:\.\d)?|Claude|Anthropic|DeepSeek|深度求索|通义千问|Qwen|智谱|ChatGLM|文心一言|百度|阿里巴巴|腾讯|豆包|Google|Gemini|微软|Microsoft|清华大学)'
SELF_CLAIM = re.compile(r'(?:我是|我叫|我的(?:名字|名称|身份)是|作为(?:一[个款种])?|I\s*(?:am|\x27m|’m)|my name is|as an?)' + r'[^。！？!?.\n]{0,110}' + VENDORS + r'|(?:我.{0,15}由|I was|I am)[^。！？!?.\n]{0,80}' + VENDORS + r'|我[（(]'+VENDORS+r'[）)]|'+ VENDORS + r'[^。！？!?.\n]{0,35}(?:developed me|created me)', re.I)
SYSTEM_CLAIM = re.compile(r'(?:你是|你叫|You are)[^。！？!?.\n]{0,100}' + VENDORS, re.I)
IDENTITY_QUERY = re.compile(r'你是谁|你叫什么|你的(?:名字|名称|开发者|作者|创造者|身份)|你是什么(?:模型|AI|人工智能)|谁(?:开发|创造|创建|训练|设计)了?你|你(?:是|由).{0,35}(?:开发|创建|训练)|介绍.{0,4}你自己|自我介绍|你是(?:ChatGPT|MiniMind|mindchen|Claude|DeepSeek).{0,5}[吗么?？]|who are you|what(?:\x27s| is) your name|who.{0,30}(?:develop|creat|train|design).{0,20}you|who is your (?:developer|creator)|introduce yourself|are you (?:ChatGPT|MiniMind|mindchen|Claude|DeepSeek)|what (?:model|AI) are you', re.I)
OTHER_TASK = re.compile(r'扮演|假设|模仿|角色|故事|小说|翻译|改写|引用|这句话|这段话|role.?play|pretend|translate|rewrite|quotation', re.I)
KNOWN_TASKS = (
    RECITE,
    '请按原文填入【缺句】，只输出被省略的句子，保留原有标点。',
    '请按原文补全这篇作品的后半部分，只输出后续原文，不要另写新作。',
    '请按原文补全这一原文片段的后半部分，只输出后续原文，不要另写新作。',
    '将下面的白话文译成古文，只输出译文。',
    '将下面的古文译成现代汉语，只输出译文。',
    '请为下面的古文恢复标点，不增删或改换文字，只输出加标点后的原文。',
    '用文言文表达这句话：', '把这段白话改写为古文：',
    '将下列现代汉语翻译成古文：', '请用古文表达：',
)
OLD_MARKERS = ('minimind', 'mini-mind', 'mini_mind', 'mini mind', 'jingyaogong', 'jingyao', '龚靖尧', '龚敬尧')
VENDOR_MARKERS = ('chatgpt', 'openai', 'gpt', 'claude', 'anthropic', 'deepseek', '深度求索',
    '通义千问', 'qwen', '智谱', 'chatglm', '文心一言', '百度', '阿里巴巴', '腾讯', '豆包',
    'google', 'gemini', '微软', 'microsoft', '清华大学')


def contains_old_name(text):
    lowered = text.lower()
    return any(word in lowered for word in OLD_MARKERS) and bool(OLD_NAMES.search(text))


def contains_vendor(text):
    lowered = text.lower()
    return any(word in lowered for word in VENDOR_MARKERS)


def normalized(text):
    return unicodedata.normalize('NFC', text.replace('\r\n', '\n').replace('\r', '\n')).strip()


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).digest()


def dump(row):
    return json.dumps(row, ensure_ascii=False, separators=(',', ':'))


def conversation_key(row):
    # 仅比较时统一NFC/换行和空可选字段；输出诗词正文保持原字符串。
    messages = []
    for message in row['conversations']:
        messages.append({k: normalized(v) if isinstance(v, str) else v for k, v in sorted(message.items())
            if v not in (None, '', [], {}) or k in ('role', 'content')})
    return digest(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def answer_keys(row):
    return {digest(normalized(m['content'])) for m in row['conversations']
        if m['role'] == 'assistant' and m.get('content', '').strip()}


def is_poetry(row):
    messages = row.get('conversations', [])
    return bool(messages and messages[0].get('role') == 'system' and messages[0].get('content') == POETRY_SYSTEM)


def recitation_prompt(user):
    # 删除集合名（宋诗 / 等）与答案首句。作者不删，同名作品不能仅靠标题区分。
    lines = [line for line in user.splitlines() if not line.startswith('首句提示：')]
    return re.sub(r'(?m)^《[^\n]*? / (.*)》$', r'《\1》', '\n'.join(lines)).strip()


def replace_own_names(text):
    """替换自然语言中的旧身份；代码、URL不盲改，剩余旧名的样本进入隔离区。"""
    if not contains_old_name(text):
        return text
    spans = [(m.start(), m.end()) for m in PROTECTED.finditer(text)]
    def substitute(match):
        if any(a <= match.start() < b for a, b in spans):
            return match.group()
        return 'mindchen' if NAME_ONLY.fullmatch(match.group()) else 'chen'
    return OLD_NAMES.sub(substitute, text)


def is_identity_question(text):
    return len(text) <= 250 and bool(IDENTITY_QUERY.search(text)) and not OTHER_TASK.search(text)


def transform(row, ambiguous=frozenset()):
    """返回新样本、修改标签、排除原因；不修改传入对象。"""
    tags = []
    messages = row.get('conversations')
    if not isinstance(messages, list) or not messages:
        return None, tags, 'invalid_conversations'
    if any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant', 'tool')
           or not isinstance(m.get('content'), str) for m in messages):
        return None, tags, 'invalid_message_schema'
    if not any(m['role'] == 'user' for m in messages) or not any(m['role'] == 'assistant' and
           (m['content'].strip() or m.get('tool_calls') or m.get('reasoning_content')) for m in messages):
        return None, tags, 'missing_user_or_answer'
    result = {**row, 'conversations': [dict(m) for m in messages]}
    messages = result['conversations']
    if is_poetry(row):
        if len(messages) != 3 or messages[1]['role'] != 'user' or messages[2]['role'] != 'assistant':
            return None, tags, 'unexpected_poetry_schema'
        user = messages[1]['content']
        if user.split('\n')[0] not in KNOWN_TASKS:
            # 旧库含“写诗/模仿创作”的提示配已有作品；没有可靠作者元数据时不臆造出处。
            return None, tags, 'unverified_poetry_instruction'
        if user.startswith(RECITE):
            prompt = recitation_prompt(user)
            if digest(prompt) in ambiguous:
                return None, tags, 'ambiguous_title_author'
            hints = [line.split('：', 1)[1] for line in user.splitlines() if line.startswith('首句提示：')]
            if hints and not messages[2]['content'].startswith(hints[0]):
                return None, tags, 'recitation_hint_answer_mismatch'
            messages[1]['content'] = prompt
            tags.append('recitation_remove_hint_and_collection')
        # 具体指令已在user中，去掉重复的领域system，避免只在此system下才会完成任务。
        messages.pop(0)
        tags.append('remove_fixed_poetry_system')
        # 已审计的文学模板只改prompt；原文不能因“百度”等古汉语词汇触发身份改写。
        if not any(contains_old_name(m.get(field, '') or '') for m in messages for field in ('content','reasoning_content')):
            return result, sorted(set(tags)), None
    previous_user = ''
    for message in messages:
        role = message['role']
        if role == 'tool':
            continue  # 工具返回也是结构的一部分；如含旧名，整行隔离，不伪造工具输出。
        if role == 'user':
            previous_user = message['content']
        if role == 'assistant' and is_identity_question(previous_user) and not message.get('tool_calls'):
            english = not re.search(r'[\u3400-\u9fff]', previous_user)
            answer = EN_IDENTITY if english else ZH_IDENTITY
            if message['content'] != answer or message.get('reasoning_content'):
                message['content'] = answer
                if 'reasoning_content' in message:
                    message['reasoning_content'] = ''
                tags.append('canonical_identity_answer')
        for field in ('content', 'reasoning_content'):
            value = message.get(field)
            if not isinstance(value, str):
                continue
            new = replace_own_names(value)
            if value != new:
                message[field] = new
                tags.append('replace_legacy_identity')
            if role == 'system' and contains_vendor(new) and SYSTEM_CLAIM.search(new):
                # 混有外部身份与复杂约束的system不擅自整体改写，排除待审。
                return None, tags, 'external_identity_system'
            if role == 'assistant' and contains_vendor(new) and not OTHER_TASK.search(previous_user) and SELF_CLAIM.search(new):
                return None, tags, 'external_identity_answer_or_reasoning'
    # 确保训练语料没有旧名残留，包括工具字符串/代码/URL；隔离而不是破坏结构。
    if contains_old_name(dump(result)):
        return None, tags, 'legacy_identity_in_protected_or_structured_text'
    return result, sorted(set(tags)), None


def read_rows(path, limit=0):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if limit and line_no > limit:
                break
            try:
                yield line_no, json.loads(line)
            except (ValueError, TypeError):
                yield line_no, None


def build_recitation_catalog(paths, db, limit=0):
    """先全库找同题同作者多答案，再删首句，防止把原本可区分的样本变成冲突标签。"""
    for split, path in paths.items():
        count = 0
        # 绝大部分通用样本不是背诵；字符串预筛后才解析JSON，减少14GB扫描开销。
        with path.open(encoding='utf-8-sig') as stream:
            for line_no, line in enumerate(stream, 1):
                if limit and line_no > limit:
                    break
                if RECITE not in line:
                    continue
                row = json.loads(line)
                if not is_poetry(row) or len(row['conversations']) != 3:
                    continue
                user, answer = row['conversations'][1:]
                if not user['content'].startswith(RECITE):
                    continue
                db.execute('INSERT INTO recitations VALUES(?,?,0) ON CONFLICT(prompt) DO UPDATE SET ambiguous=MAX(ambiguous,answer!=excluded.answer)',
                    (digest(recitation_prompt(user['content'])), digest(normalized(answer['content']))))
                count += 1
        db.commit()
        print(f'catalog {split}: {count:,} recitations', flush=True)
    return {r[0] for r in db.execute('SELECT prompt FROM recitations WHERE ambiguous=1')}


def identity_rows(identity_path):
    for _, row in read_rows(identity_path):
        if row:
            yield row
    # 少量不同问法，不大量复制同一句，且一半不依赖system提示。
    questions = ['你是谁？', '你叫什么名字？', '你的开发者是谁？', '请做一个简短的自我介绍。',
        '你是什么模型？', '谁创建了你？', '你的名字和开发者分别是什么？', '你是MiniMind吗？',
        '你是ChatGPT吗？', '介绍一下你自己。', '谁开发了你？', '你的身份是什么？',
        'Who are you?', 'What is your name?', 'Who developed you?', 'Who created you?',
        'Please introduce yourself.', 'Are you ChatGPT?', 'Are you MiniMind?', 'Who is your developer?']
    for i, question in enumerate(questions):
        messages = [] if i % 2 else [dict(role='system', content=IDENTITY_SYSTEM)]
        messages.extend([dict(role='user', content=question), dict(role='assistant',
            content=EN_IDENTITY if not re.search(r'[\u3400-\u9fff]', question) else ZH_IDENTITY)])
        yield dict(conversations=messages)


def clean(input_dir, output_dir, identity_path, limit=0):
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = {s: input_dir/f'sft_t2t_with_poetry_{s}.jsonl' for s in ('valid', 'train')}
    sources = {s: dict(path=str(p.resolve()), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for s, p in paths.items()}
    db = sqlite3.connect(output_dir/'clean_index.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-131072;
        CREATE TABLE recitations(prompt BLOB PRIMARY KEY,answer BLOB NOT NULL,ambiguous INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE seen(key BLOB PRIMARY KEY,split TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE protected_answers(key BLOB PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE valid_prompts(key BLOB PRIMARY KEY) WITHOUT ROWID;''')
    ambiguous = build_recitation_catalog(paths, db, limit)
    protected, valid_prompts = set(), set()
    stats, previews = {}, defaultdict(list)
    report = dict(status='running', sources=sources, identity_source=str(identity_path) if identity_path else None,
        cleaner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limited_input_rows=limit,
        ambiguous_recitation_prompts=len(ambiguous), policy=dict(
            poetry='Exact known tasks retained; remove fixed system; unhinted recitation retains author; ambiguous title-author groups excluded.',
            identity='Legacy names replaced in natural text; direct identity answers canonicalized; conflicting self identity and protected-text remnants quarantined.',
            split='Existing membership retained; validation processed first; transformed dialogue dedup; identical validation assistant answer and prompt blocked from train.',
            length='No truncation or token-length filtering in this cleaner. Length audit is separate.',
            scope='Rule validation, not full semantic/historical verification. No source code/framework renaming.'))
    with (output_dir/'rejected_rows.jsonl').open('w', encoding='utf-8') as rejects, \
         (output_dir/'identity_clean.jsonl').open('w', encoding='utf-8') as identities:
        for split, path in paths.items():
            counts = Counter()
            out_path = output_dir/f'sft_t2t_with_poetry_{split}.jsonl'
            out_hash = hashlib.sha256()
            before_hash = hashlib.sha256()
            with out_path.open('wb') as output, (output_dir/f'provenance_{split}.jsonl').open('w', encoding='utf-8') as provenance:
                def process(row, line_no, source):
                    counts['input_rows'] += 1
                    if row is None:
                        reason, result, tags = 'invalid_json', None, []
                    else:
                        before_hash.update((dump(row)+'\n').encode('utf-8'))
                        result, tags, reason = transform(row, ambiguous)
                    if result:
                        key = conversation_key(result)
                        prompt_key = digest(dump({'conversations': [m for m in result['conversations'] if m['role'] != 'system'][:-1]}))
                        keys = answer_keys(result)
                        if split == 'train' and keys & protected:
                            reason = 'protected_validation_answer'
                        elif split == 'train' and prompt_key in valid_prompts:
                            reason = 'protected_validation_prompt'
                        else:
                            inserted = db.execute('INSERT OR IGNORE INTO seen VALUES(?,?)', (key, split)).rowcount
                            if not inserted:
                                reason = 'duplicate_after_cleaning'
                    if reason:
                        counts['rejected_'+reason] += 1
                        # 记录原文和源行号，可以审查/恢复；不自动猜测诗作出处或改坏代码。
                        rejects.write(dump(dict(split=split, source=source, input_line=line_no, reason=reason, row=row))+'\n')
                        if len(previews['rejected_'+reason]) < 3:
                            previews['rejected_'+reason].append(dict(input_line=line_no, row=row))
                        return
                    output_line = counts['output_rows']
                    content = (dump(result)+'\n').encode('utf-8')
                    output.write(content)
                    out_hash.update(content)
                    counts['output_rows'] += 1
                    counts['poetry_rows' if row and is_poetry(row) else 'general_or_identity_rows'] += 1
                    for tag in tags:
                        counts['changed_'+tag] += 1
                        if len(previews[tag]) < 4:
                            previews[tag].append(dict(split=split, input_line=line_no, before=row, after=result))
                    provenance.write(dump(dict(output_line=output_line, source=source, input_line=line_no, changes=tags))+'\n')
                    if split == 'valid':
                        protected.update(keys)
                        valid_prompts.add(prompt_key)
                    if source == 'identity':
                        identities.write(dump(result)+'\n')
                for line_no, row in read_rows(path, limit):
                    process(row, line_no, split)
                    if line_no % 100000 == 0:
                        db.commit()
                        print(f'clean {split}: input={line_no:,} kept={counts["output_rows"]:,}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
                if split == 'train' and identity_path:
                    for line_no, row in enumerate(identity_rows(identity_path), 1):
                        process(row, line_no, 'identity')
                if split == 'valid':
                    db.executemany('INSERT OR IGNORE INTO protected_answers VALUES(?)', ((k,) for k in protected))
                    db.executemany('INSERT OR IGNORE INTO valid_prompts VALUES(?)', ((k,) for k in valid_prompts))
                db.commit()
            stats[split] = dict(counts=counts, path=str(out_path.resolve()), bytes=out_path.stat().st_size,
                sha256=out_hash.hexdigest(), input_canonical_sha256=before_hash.hexdigest())
            report['outputs'] = stats
            (output_dir/'clean_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    for split, path in paths.items():
        if path.stat().st_size != sources[split]['bytes'] or path.stat().st_mtime_ns != sources[split]['mtime_ns']:
            raise RuntimeError('Source changed during cleaning: '+str(path))
    report.update(status='cleaned_pending_independent_validation', seconds=time.perf_counter()-started,
        protected_validation_answers=len(protected), protected_validation_prompts=len(valid_prompts))
    (output_dir/'clean_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir/'changes_preview.json').write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir', type=Path, default=ROOT/'dataset/sft_t2t_with_poetry')
    p.add_argument('--output_dir', type=Path, default=ROOT/'dataset/sft_t2t_with_poetry_clean_v1')
    p.add_argument('--identity_path', type=Path, default=ROOT/'dataset/lora_identity.jsonl')
    p.add_argument('--limit', type=int, default=0, help='仅用于流程试跑；0为完整库')
    args = p.parse_args()
    clean(args.input_dir.resolve(), args.output_dir.resolve(), args.identity_path.resolve(), args.limit)
