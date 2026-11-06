# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Detect excessive repetition and train against repeated completion tokens."""
from collections import Counter, defaultdict
import re
import unicodedata

import torch


def text_units(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return re.findall(r'[\u3400-\u9fff]|[^\W\d_]+|\d+', text)


def loop_positions(sequence, max_period=16):
    """Mark third and later cycles, keeping two cycles as unpenalized context."""
    sequence = list(sequence)
    bad = set()
    for width in range(1, min(max_period, len(sequence)//3)+1):
        start = 0
        while start+3*width <= len(sequence):
            block = sequence[start:start+width]
            if block != sequence[start+width:start+2*width] or block != sequence[start+2*width:start+3*width]:
                start += 1
                continue
            end = start+3*width
            while end < len(sequence) and sequence[end] == sequence[end-width]:
                end += 1
            if end-start >= max(8, 3*width):
                bad.update(range(start+2*width, end))
            start = max(start+1, end-2*width)
    return bad


def repetition_diagnostics(text):
    units = text_units(text)
    count = len(units)
    loops = loop_positions(units, 32)
    recurring = set()
    for width in (4, 8):
        occurrences = defaultdict(list)
        for start in range(max(0, count-width+1)):
            occurrences[tuple(units[start:start+width])].append(start)
        for starts in occurrences.values():
            # Normal use of a term twice is not a degenerate loop.
            if len(starts) >= 3:
                for start in starts[2:]:
                    recurring.update(range(start, start+width))
    sentence_counts = Counter()
    sentence_bad = 0
    for sentence in re.split(r'[。！？.!?\n]+', text):
        key = tuple(text_units(sentence))
        sentence_counts[key] += 1
        if len(key) >= 12 and sentence_counts[key] > 1:
            sentence_bad += len(key)
    # Also catch an unbroken Latin string such as abcabcabc..., without splitting words normally.
    latin_loop = any(len(word) >= 16 and len(loop_positions(word, 16))/len(word) >= .5
                     for word in re.findall(r'[a-zA-Z]{16,}', text))
    loop_fraction = len(loops)/max(count, 1)
    recurring_fraction = len(recurring)/max(count, 1)
    sentence_fraction = min(1., sentence_bad/max(count, 1))
    excessive = ((len(loops) >= 3 and loop_fraction >= .15) or
                 (len(recurring) >= 12 and recurring_fraction >= .20) or
                 sentence_fraction >= .20 or latin_loop)
    fraction = max(loop_fraction, recurring_fraction, sentence_fraction, float(latin_loop))
    severity = min(1., fraction/.6) if excessive else 0.
    distinct = {}
    for width in (1, 2, 3):
        grams = [tuple(units[i:i+width]) for i in range(max(0, count-width+1))]
        distinct[f'distinct_{width}'] = len(set(grams))/max(len(grams), 1) if grams else 1.
    return dict(excessive=bool(excessive), severity=severity, repeated_fraction=fraction,
                loop_fraction=loop_fraction, repeated_sentence_fraction=sentence_fraction,
                units=count, **distinct)


def preserves_reference(row, response):
    spec = row.get('spec') or {}
    reference = spec.get('reference_body')
    return bool(spec.get('preserve_chars') and reference and
                re.fullmatch(r'[\u3400-\u9fff\s，。！？；,.!?;]+', response) and
                ''.join(re.findall(r'[\u3400-\u9fff]', response)) == reference)


def response_repetition(row, response):
    result = repetition_diagnostics(response)
    result['allowed_reference'] = preserves_reference(row, response)
    result['allowed_requested_repetition'] = follows_repeat_request(row, response)
    if result['allowed_reference'] or result['allowed_requested_repetition']:
        result.update(excessive=False, severity=0.)
    return result


def follows_repeat_request(row, response):
    """Exempt only a fully matched explicit phrase/count task and exact answer."""
    prompt = next((m['content'].strip() for m in reversed(row.get('messages', []))
                   if m['role'] == 'user'), '')
    numbers = dict(zip(('one two three four five six seven eight nine ten eleven twelve').split(), range(1, 13)))
    digits = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
              '六': 6, '七': 7, '八': 8, '九': 9}
    def number(value):
        if value.isdecimal():
            return int(value)
        if value.lower() in numbers:
            return numbers[value.lower()]
        if value == '十':
            return 10
        if '十' in value:
            left, right = value.split('十')
            return digits.get(left, 1)*10+digits.get(right, 0)
        return digits.get(value, 0)
    count = r'(\d{1,2}|[一二两三四五六七八九]?十[一二三四五六七八九]?|[一二两三四五六七八九]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)'
    quoted = r'[“"「](.{1,100}?)[”"」]'
    matches = [
        (re.fullmatch(r'(?:请)?(?:把|将)?'+quoted+r'(?:重复|输出|打印)\s*'+count+r'\s*次(?:[，, ]*(?:每行一次|逐行输出|不要添加其他内容|只输出结果))*[。.!！]?', prompt, re.I), False),
        (re.fullmatch(r'(?:请)?(?:逐行输出|逐行打印)\s*'+count+r'\s*次\s*(.{1,100}?)[。！]?', prompt), True),
        (re.fullmatch(r'(?:please\s+)?repeat\s+'+quoted+r'\s+'+count+r'\s+times(?:[,. ]*(?:one per line|on separate lines|exactly as requested|and nothing else))*[.!]?', prompt, re.I), False),
        (re.fullmatch(r'(?:please\s+)?output\s+(.{1,100}?)\s+on\s+'+count+r'\s+separate lines(?:,?\s+exactly as requested)?[.!]?', prompt, re.I), True)]
    for index, (match, lines) in enumerate(matches):
        if not match:
            continue
        phrase, value = match.groups()
        if index == 1:
            value, phrase = phrase, value
        n = number(value)
        if not 2 <= n <= 64 or not phrase.strip() or phrase != phrase.strip():
            return False
        lines = lines or bool(re.search(r'每行一次|逐行输出|one per line|on separate lines', prompt, re.I))
        if lines:
            return response.strip().splitlines() == [phrase]*n
        return bool(re.fullmatch(re.escape(phrase)+(r'\s*'+re.escape(phrase))*(n-1), response.strip()))
    return False


def repeated_token_mask(completions, completion_mask, exempt=None):
    """Only completion positions before EOS participate; prompt/padding are excluded."""
    result = torch.zeros_like(completion_mask, dtype=torch.bool)
    sequences = completions.detach().cpu().tolist()
    lengths = completion_mask.sum(1).detach().cpu().tolist()
    for index, (sequence, length) in enumerate(zip(sequences, lengths)):
        if exempt and exempt[index]:
            continue
        positions = loop_positions(sequence[:int(length)], 32)
        for width, threshold in ((4, 3), (8, 2)):
            seen = Counter()
            for start in range(max(0, int(length)-width+1)):
                gram = tuple(sequence[start:start+width])
                seen[gram] += 1
                if seen[gram] >= threshold:
                    positions.update(range(start, start+width))
        if positions:
            result[index, sorted(positions)] = True
    return result & completion_mask.bool()


def unlikelihood_loss(logps, repeated_mask):
    """-log(1-p(token)) on confirmed excessive repeats, averaged per candidate."""
    # expm1 preserves 1-p near p=1; additive smoothing retains a finite,
    # nonzero gradient there instead of a hard probability clamp's dead zone.
    logps = logps.float().clamp(max=0.)
    complement = -torch.expm1(logps)
    near_one = -torch.log((complement+1e-6)/(1.+1e-6))
    low_probability = -torch.log1p(-logps.exp()/(1.+1e-6))
    losses = torch.where(logps > -.693147, near_one, low_probability)
    mask = repeated_mask.to(losses.dtype)
    return ((losses*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()


def aggregate_repetition(records):
    if not records:
        return {}
    diagnoses = [r['reward']['repetition'] for r in records]
    return dict(repetition_rate=sum(d['excessive'] for d in diagnoses)/len(diagnoses),
                repeated_fraction=sum(d['repeated_fraction'] for d in diagnoses)/len(diagnoses),
                **{key: sum(d[key] for d in diagnoses)/len(diagnoses)
                   for key in ('distinct_1', 'distinct_2', 'distinct_3')})
