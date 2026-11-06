# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Resumable photo -> Chinese caption/poetry -> visual review -> Parquet pipeline.

Outputs are automatically screened candidates. Human quality calibration and
the original handbook's full acceptance gates remain explicitly unfulfilled.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import tarfile
import time

import numpy as np
from PIL import Image, ImageOps, ImageStat
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from photopoet_config import settings, count as sample_count, reached, task_quotas

ROOT = Path(__file__).resolve().parent / 'PhotoPoet-Data'
SCHEMA = pa.schema([('conversations', pa.string()), ('image_bytes', pa.binary())])
TASKS = {
    'qijue': '请根据照片创作一首七言绝句，四句，每句恰好七个汉字。只输出诗句，不要标题或说明。',
    'wujue': '请根据照片创作一首五言绝句，四句，每句恰好五个汉字。只输出诗句，不要标题或说明。',
    'modern': '请为照片写一首四行现代短诗，意象来自画面，语言清新克制，不要标题和解释。',
    'social': '请为照片写两行有诗意但自然的朋友圈配文，不要话题标签，不要标题或解释。',
    'mood': '请根据照片写一首四行现代小诗，风格温柔克制；情感仅作艺术表达，不猜测人物真实心理。',
    'keywords': '请根据照片写一首四行现代短诗，自然包含这些可见意象：{keywords}。只输出诗句。',
    'interpret': '请先用一句话客观解释照片中的两个主要意象，再另起四行创作现代短诗，不要标题。',
}
# Frequencies match the handbook's seven task proportions; average 2.5 tasks/photo.
TASK_CYCLE = ['qijue'] * 8 + ['wujue'] * 7 + ['modern'] * 9 + ['social'] * 6 + ['mood'] * 4 + ['keywords'] * 3 + ['interpret'] * 3

def jd(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def han(value):
    return ''.join(re.findall(r'[\u3400-\u9fff]', value))

def database():
    (ROOT / '02_metadata').mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / '02_metadata/index.sqlite3', timeout=120)
    db.row_factory = sqlite3.Row
    db.executescript('''
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS assets(
      id TEXT PRIMARY KEY, path TEXT NOT NULL, sha TEXT UNIQUE NOT NULL,
      phash TEXT NOT NULL, cluster_id TEXT NOT NULL, split TEXT NOT NULL,
      provenance TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
      facts TEXT, reason TEXT, attempts INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS signatures(bucket INTEGER, key INTEGER, asset TEXT,
      PRIMARY KEY(bucket,key,asset));
    CREATE TABLE IF NOT EXISTS calls(
      id INTEGER PRIMARY KEY, asset TEXT, stage TEXT, prompt TEXT, response TEXT,
      model TEXT, model_digest TEXT, seconds REAL, created TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS samples(
      id TEXT PRIMARY KEY, asset TEXT NOT NULL, stage TEXT NOT NULL, task TEXT NOT NULL,
      prompt TEXT NOT NULL, target TEXT NOT NULL, target_sha TEXT NOT NULL,
      review TEXT NOT NULL, UNIQUE(stage,target_sha));
    CREATE INDEX IF NOT EXISTS asset_samples ON samples(asset);
    CREATE TABLE IF NOT EXISTS ngrams(stage TEXT, gram TEXT, sample TEXT,
      PRIMARY KEY(stage,gram,sample));
    CREATE INDEX IF NOT EXISTS assets_state_id ON assets(state,id);
    CREATE INDEX IF NOT EXISTS assets_cluster ON assets(cluster_id);
    CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY,value INTEGER NOT NULL);
    CREATE TRIGGER IF NOT EXISTS samples_counter_insert AFTER INSERT ON samples BEGIN
      INSERT INTO counters VALUES('stage:' || NEW.stage,1) ON CONFLICT(name) DO UPDATE SET value=value+1;
      INSERT INTO counters VALUES('task:' || NEW.task,1) ON CONFLICT(name) DO UPDATE SET value=value+1;
    END;
    CREATE TRIGGER IF NOT EXISTS samples_counter_delete AFTER DELETE ON samples BEGIN
      UPDATE counters SET value=value-1 WHERE name='stage:' || OLD.stage;
      UPDATE counters SET value=value-1 WHERE name='task:' || OLD.task;
    END;
    ''')
    db.execute('BEGIN IMMEDIATE')
    if not db.execute("SELECT 1 FROM counters WHERE name='initialized'").fetchone():
        db.execute('DELETE FROM counters')
        db.execute("INSERT INTO counters SELECT 'stage:' || stage,count(*) FROM samples GROUP BY stage")
        db.execute("INSERT INTO counters SELECT 'task:' || task,count(*) FROM samples GROUP BY task")
        db.execute("INSERT INTO counters VALUES('initialized',1)")
    db.commit()
    return db

def phash(im):
    # Orthonormal DCT-II, matching the standard 32x32 -> 8x8 pHash construction.
    pixels = np.asarray(im.convert('L').resize((32, 32)), dtype=float)
    n = np.arange(32)
    basis = np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / 64)
    coeff = (basis @ pixels @ basis.T)[:8, :8]
    bits = coeff > np.median(coeff)
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result

def add_asset(db, key, path, cluster, provenance):
    if db.execute('SELECT 1 FROM assets WHERE id=?', (key,)).fetchone():
        return
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if db.execute('SELECT 1 FROM assets WHERE sha=?', (sha,)).fetchone():
        return
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        if min(im.size) < 128 or max(ImageStat.Stat(im.convert('RGB')).stddev) < 6:
            return
        h = phash(im)
    signatures = [(i, (h >> (i * 13)) & ((1 << min(13, 64 - i * 13)) - 1)) for i in range(5)]
    near = set()
    for i, k in signatures:
        for row in db.execute('SELECT asset FROM signatures WHERE bucket=? AND key=?', (i, k)):
            near.add(row[0])
    # Keep near duplicates in the same split and cluster. Review cluster boundaries separately.
    matched_clusters = {cluster}
    for other in sorted(near):
        row = db.execute('SELECT phash,cluster_id FROM assets WHERE id=?', (other,)).fetchone()
        if (h ^ int(row['phash'], 16)).bit_count() <= 4:
            matched_clusters.add(row['cluster_id'])
    cluster = min(matched_clusters)
    split_value = int(hashlib.sha256(cluster.encode()).hexdigest()[:8], 16) % 10000
    split = 'train' if split_value < 9800 else ('validation' if split_value < 9900 else 'test')
    for previous in matched_clusters - {cluster}:
        db.execute('UPDATE assets SET cluster_id=?,split=? WHERE cluster_id=?', (cluster, split, previous))
    db.execute('INSERT INTO assets(id,path,sha,phash,cluster_id,split,provenance) VALUES(?,?,?,?,?,?,?)',
               (key, str(path.relative_to(ROOT)), sha, f'{h:016x}', cluster, split, jd(provenance)))
    db.executemany('INSERT INTO signatures VALUES(?,?,?)', [(i, k, key) for i, k in signatures])
    db.commit()

def import_docci(db, extract_all=False):
    folder = ROOT / '00_sources/docci'
    descriptions = {r['example_id']: r for r in map(json.loads, (folder / 'docci_descriptions.jsonlines').open(encoding='utf-8')) if r['split'] == 'train'}
    clusters = {}
    with (folder / 'docci_metadata.jsonlines').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['example_id'] in descriptions:
                clusters[row['example_id']] = str(row['cluster_id'])
    target = ROOT / '01_images/docci'
    target.mkdir(parents=True, exist_ok=True)
    if extract_all:
        archive = folder / 'docci_images.tar.gz'
        with tarfile.open(archive, 'r|gz') as tar:
            for item in tar:
                name = Path(item.name).name
                key = Path(name).stem
                if item.isfile() and key in descriptions and name == descriptions[key]['image_file']:
                    dest = target / name
                    if not dest.exists():
                        with tar.extractfile(item) as src:
                            dest.write_bytes(src.read())
    for path in sorted(target.glob('*.jpg')):
        key = path.stem
        if key in descriptions:
            add_asset(db, 'docci:' + key, path, 'docci:' + clusters[key], {
                'dataset': 'DOCCI', 'original_id': key, 'original_split': 'train',
                'source_url': 'https://storage.googleapis.com/docci/data/docci_images.tar.gz',
                'license': 'CC-BY-4.0', 'attribution': 'Google LLC; Onoe et al., DOCCI, ECCV 2024',
                'original_description': descriptions[key]['description'],
            })
    print('INDEX', dict(db.execute('SELECT state,count(*) FROM assets GROUP BY state').fetchall()), flush=True)

class Teacher:
    def __init__(self, db, model, endpoint):
        self.db, self.model, self.endpoint = db, model, endpoint.rstrip('/')
        models = requests.get(self.endpoint + '/api/tags', timeout=10).json()['models']
        found = [r for r in models if r['name'] == model]
        if not found:
            raise RuntimeError(f'local teacher not installed: {model}')
        self.digest = found[0]['digest']
        if 'vision' not in found[0].get('capabilities', []):
            raise RuntimeError('selected teacher does not advertise vision support')

    def ask(self, asset, stage, prompt, image=None, temperature=0.2, tokens=900):
        msg = {'role': 'user', 'content': prompt}
        if image:
            with Image.open(image) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((896, 896))
                buf = io.BytesIO(); im.save(buf, 'JPEG', quality=90)
            msg['images'] = [base64.b64encode(buf.getvalue()).decode('ascii')]
        start = time.monotonic()
        attempt = self.db.execute('SELECT attempts FROM assets WHERE id=?', (asset,)).fetchone()[0]
        payload = {
            'model': self.model, 'messages': [msg], 'stream': False, 'think': False,
            'format': 'json', 'keep_alive': '10m',
            'options': {'temperature': temperature, 'num_predict': tokens, 'num_ctx': 8192, 'seed': int(hashlib.sha256((asset + stage + str(attempt)).encode()).hexdigest()[:7], 16)},
        }
        for retry in range(settings()['teacher_http_retries']):
            try:
                r = requests.post(self.endpoint + '/api/chat', json=payload, timeout=(20, 600))
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                if retry + 1 >= settings()['teacher_http_retries']: raise
                print('TEACHER_RETRY', stage, retry + 1, str(exc), flush=True)
                time.sleep(min(2 ** (retry + 1), 30))
        result = r.json()
        self.db.execute('INSERT INTO calls(asset,stage,prompt,response,model,model_digest,seconds) VALUES(?,?,?,?,?,?,?)',
            (asset, stage, prompt, jd(result), self.model, self.digest, time.monotonic() - start))
        self.db.commit()
        if result.get('done_reason') == 'length':
            raise ValueError('teacher output truncated')
        return json.loads(result['message']['content'])

def validate_text(task, text):
    if not isinstance(text, str) or not text.strip() or len(text) > 900:
        return 'empty_or_overlong'
    if '<image>' in text or '<think>' in text or '```' in text or '\ufffd' in text:
        return 'control_or_invalid_text'
    if re.search(r'[A-Za-z]', text):
        return 'english_in_chinese_target'
    if len(han(text)) < len(text.strip()) * .45:
        return 'not_chinese'
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if task in ('qijue', 'wujue'):
        parts = [p for p in re.split(r'[，。！？；、,.;!?\n]+', text) if p.strip()]
        size = 7 if task == 'qijue' else 5
        if len(parts) != 4 or any(len(han(p)) != size for p in parts):
            return 'verse_length'
        if any(re.sub(r'[\u3400-\u9fff\s]', '', p) for p in parts):
            return 'verse_non_han'
        if len(set(han(p) for p in parts)) != 4:
            return 'repeated_lines'
    elif task == 'social':
        if len(lines) != 2 or len(han(text)) > 100 or '#' in text:
            return 'social_format'
    elif task == 'interpret':
        if len(lines) != 5:
            return 'interpret_format'
    elif len(lines) != 4:
        return 'modern_format'
    elif len(set(len(han(s)) for s in lines)) == 1 and len(han(lines[0])) in (5, 7):
        return 'classical_instead_of_modern'
    return None

def save_sample(db, row, stage, task, prompt, target, review):
    if sample_count(db, stage) >= settings()['targets'][stage]: return False
    if stage == 'sft':
        current = db.execute('SELECT value FROM counters WHERE name=?', ('task:' + task,)).fetchone()
        if current and current[0] >= task_quotas()[task]: return False
    normalized = re.sub(r'\s+', '', target)
    h = hashlib.sha256(normalized.encode()).hexdigest()
    key = hashlib.sha256((row['id'] + stage + task).encode()).hexdigest()[:24]
    chars = han(target)
    grams = {chars[i:i+8] for i in range(max(0, len(chars)-7))}
    matches = Counter()
    informative = 0
    for gram in grams:
        hits = db.execute('SELECT sample FROM ngrams WHERE stage=? AND gram=? LIMIT 257', (stage, gram)).fetchall()
        if len(hits) <= 256:
            informative += 1
            matches.update(r[0] for r in hits)
    if matches and max(matches.values()) / max(1, informative) > .6:
        return
    goal = settings()['targets'][stage]
    task_key = 'task:' + task if stage == 'sft' else 'stage:' + stage
    task_goal = task_quotas()[task] if stage == 'sft' else goal
    inserted = db.execute('''INSERT OR IGNORE INTO samples
        SELECT ?,?,?,?,?,?,?,? WHERE COALESCE((SELECT value FROM counters WHERE name=?),0) < ?
        AND COALESCE((SELECT value FROM counters WHERE name=?),0) < ?''',
        (key, row['id'], stage, task, prompt, target, h, jd(review), 'stage:' + stage, goal, task_key, task_goal))
    if inserted.rowcount:
        db.executemany('INSERT OR IGNORE INTO ngrams VALUES(?,?,?)', [(stage, gram, key) for gram in grams])
    db.commit()
    return bool(inserted.rowcount)

def process(db, teacher, row, index):
    image = ROOT / row['path']
    source = json.loads(row['provenance'])
    reference = source.get('original_description', '')
    facts = json.loads(row['facts']) if row['facts'] else teacher.ask(row['id'], 'visual_facts', '''观察真实照片，只返回JSON：
{"is_photo":true,"unsafe_or_private":false,"ocr_dominant":false,"severely_blurry":false,"objects":["可见意象1","可见意象2"],"scene":"场景","lighting":"可见光线","uncertainties":[],"caption":"70至180个汉字的客观中文描述"}。
绘画、漫画、屏幕截图、广告合成图的is_photo为false。玩偶必须写为玩偶，不能写为活体。
描述只写清晰可见的实体、动作、位置、颜色、构图与光线。不要猜测具体时间地点、亲属身份、职业、人物心理和天气原因。
有可读私人联系方式、身份证件、明显隐私信息或不适宜内容时标unsafe_or_private=true。
caption只描述照片内容，不要出现关于隐私、安全、画质好坏、审核结果的评价句。不要识别品牌型号、精确拍摄时间地点，不抄录图中的文字，只说明有文字或标记；不要猜测灯是否点亮、光源类型、材质成分。照片不足以支持70字客观细节时宁可短写，不要编造。不要输出说明。
如下面提供了原始人工英文描述，以它为事实参考并对照照片，caption用中文忠实概述，不加入参考和照片均不支持的信息。
所有描述和意象名称都用中文，不出现英文字母或品牌名称，例如futon应译作“沙发床”，不要复制原始描述中的英文文字。原始描述：''' + reference, image, tokens=600)
    db.execute('UPDATE assets SET facts=? WHERE id=?', (jd(facts), row['id'])); db.commit()
    for flag in ('unsafe_or_private', 'ocr_dominant', 'severely_blurry'):
        if facts.get(flag) is not False:
            raise ValueError('image_reject:' + flag)
    if facts.get('is_photo') is not True:
        raise ValueError('image_reject:not_photo')
    caption = facts.get('caption', '')
    if not 70 <= len(han(caption)) <= 180:
        raise ValueError('caption_length')
    if re.search(r'[A-Za-z]', caption):
        raise ValueError('caption_english')
    if any(x in caption for x in ('隐私信息', '不当内容', '不适宜内容', '无法判断', '无明显模糊')):
        raise ValueError('caption_contains_qc_commentary')
    current_counts = {r[0][5:]: r[1] for r in db.execute("SELECT name,value FROM counters WHERE name LIKE 'task:%'")}
    weights = Counter(TASK_CYCLE)
    quotas = task_quotas()
    tasks = sorted((t for t in TASKS if current_counts.get(t, 0) < quotas[t]), key=lambda task: current_counts.get(task, 0) / weights[task])[:2 + index % 2]
    words = [w for w in facts.get('objects', []) if isinstance(w, str) and re.fullmatch(r'[\u3400-\u9fff]{1,8}', w)][:2]
    prompts = {t: TASKS[t].format(keywords='、'.join(words)) for t in tasks if t != 'keywords' or len(words) == 2}
    generation = teacher.ask(row['id'], 'poetry', '你是中文诗歌作者。只根据以下可核实视觉事实创作，不引用已有诗句，不套用名句，不添加画面没有的具体事物；艺术联想不得成为人物心理或地点身份的事实断言。\n视觉事实：' + jd(facts) + '\n任务：' + jd(prompts) + '\n每个任务给出两个不同候选。只返回JSON，格式为{"candidates":[{"task":"任务键","text":"候选正文，换行用\\n"}]}。五言七言必须四句逐句数汉字；不要标题、括号、序号、解释，interpret任务按指令输出一句解释和四行诗。', temperature=.8, tokens=1300)
    candidates = []
    for c in generation.get('candidates', []):
        if not isinstance(c, dict) or c.get('task') not in prompts:
            continue
        text = c.get('text', '').strip()
        if validate_text(c['task'], text):
            continue
        if c['task'] == 'keywords' and any(w not in text for w in words):
            continue
        if any(text == x['text'] for x in candidates):
            continue
        candidates.append({'task': c['task'], 'text': text})
    review = teacher.ask(row['id'], 'visual_review', '你是严格的数据审查员，重新直接检查照片，同时交叉核对原始人工英文描述。生成内容不能当作证据。只要一个具体物体、动作、空间关系、人物或时间事实没有被照片支持，就判supported=false；杯子存在不等于杯中有茶，包装糖果不等于瓜果，有灯不等于亮灯，室内灯光不等于暮色，旗杆下不能凭空出现人。\n拒绝引用已有名句、语言不通顺或词语生硬堆砌的诗，艺术情绪和明显比喻可以，但不能借此编造物体。仔细检查病句和怪词，不要给所有候选打高分。\n原始人工描述：' + reference + '\n待审中文描述：' + caption + '\n候选（按下标）：' + jd(candidates) + '\n只返回JSON：{"caption_ok":true,"caption_errors":[],"reviews":[{"index":0,"supported":true,"natural":true,"original":true,"style_ok":true,"score":85,"errors":[]}]}。score为0到100，图文一致40%、语言自然20%、原创15%、诗体格式15%、风格10%；逐条审查所有候选，不能漏项。', image, tokens=1000)
    if review.get('caption_ok') is not True:
        raise ValueError('caption_visual_review:' + jd(review.get('caption_errors')))
    common = {'status': 'automated_pass_pending_human', 'teacher': teacher.model, 'teacher_digest': teacher.digest, 'human_review': False, 'prosody_review': False}
    save_sample(db, row, 'pretrain', 'caption', '<image>\n请客观描述这张照片中的主要内容。', caption, {**common, 'visual_review': review.get('caption_errors', [])})
    passed = {}
    for item in review.get('reviews', []):
        i = item.get('index')
        if not isinstance(i, int) or not 0 <= i < len(candidates):
            continue
        c = candidates[i]
        if not all(item.get(k) is True for k in ('supported', 'natural', 'original', 'style_ok')) or item.get('score', 0) < 80:
            continue
        if c['task'] not in passed or item['score'] > passed[c['task']][1]['score']:
            passed[c['task']] = (c, item)
    saved = 0
    for task, (c, item) in passed.items():
        saved += bool(save_sample(db, row, 'sft', task, '<image>\n' + prompts[task], c['text'], {**common, 'review': item}))
    db.execute("UPDATE assets SET state='done',reason=NULL WHERE id=?", (row['id'],)); db.commit()
    return saved

def legacy_export(db):
    db.execute('BEGIN')
    final = ROOT / '06_final'
    manifest_dir = final / 'manifest'; manifest_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for stage in ('pretrain', 'sft'):
        for split in ('train', 'validation', 'test'):
            folder = final / stage / split; folder.mkdir(parents=True, exist_ok=True)
            rows = db.execute('SELECT s.*,a.path,a.sha,a.cluster_id,a.split,a.provenance FROM samples s JOIN assets a ON a.id=s.asset WHERE s.stage=? AND a.split=? ORDER BY s.id', (stage, split))
            writer = None; shard = -1; count = 0; batch = []
            mf = manifest_dir / f'{stage}_{split}.jsonl'
            with mf.with_suffix('.jsonl.tmp').open('w', encoding='utf-8') as meta:
                try:
                    for row in rows:
                        if count % 10000 == 0:
                            if writer:
                                if batch: writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA)); batch=[]
                                writer.close(); temp.replace(dest)
                            shard += 1
                            dest = folder / f'photopoet_{stage}_{shard:05d}.parquet'
                            temp = dest.with_suffix('.parquet.tmp')
                            writer = pq.ParquetWriter(temp, SCHEMA, compression='zstd')
                        conv = [{'role': 'user', 'content': row['prompt']}, {'role': 'assistant', 'content': row['target']}]
                        batch.append({'conversations': jd(conv), 'image_bytes': (ROOT / row['path']).read_bytes()})
                        meta.write(jd({'sample_id': row['id'], 'asset_id': row['asset'], 'sha256': row['sha'], 'cluster_id': row['cluster_id'], 'split': split, 'task_type': row['task'], 'source': json.loads(row['provenance']), 'qc': json.loads(row['review'])}) + '\n')
                        count += 1
                        if len(batch) >= 16:
                            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA)); batch=[]
                    if writer:
                        if batch: writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                        writer.close(); writer=None; temp.replace(dest)
                finally:
                    if writer: writer.close()
            mf.with_suffix('.jsonl.tmp').replace(mf)
            counts[f'{stage}_{split}'] = count
            # A cluster merge can move samples to another split. Remove stale
            # generated shards so no old partition can leak into training.
            for stale in folder.glob(f'photopoet_{stage}_*.parquet'):
                if int(stale.stem.rsplit('_', 1)[1]) > shard:
                    stale.unlink()
    info = {'status': 'automated_candidates_pending_human_review', 'targets': {'pretrain': 2000000, 'sft': 2000000}, 'counts': dict(counts), 'unique_images': db.execute('SELECT count(DISTINCT asset) FROM samples').fetchone()[0], 'assets': dict(db.execute('SELECT state,count(*) FROM assets GROUP BY state').fetchall()), 'tasks': dict(db.execute("SELECT task,count(*) FROM samples WHERE stage='sft' GROUP BY task").fetchall()), 'human_reviewed': 0, 'thresholds_calibrated': False, 'prosody_checked': False, 'updated_unix': time.time()}
    (manifest_dir / 'status.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    db.commit()
    print('EXPORTED', jd(info), flush=True)

def export(db):
    from photopoet_export import export_dataset
    export_dataset(db)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--extract-all', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--model', default='qwen3.5:latest')
    parser.add_argument('--endpoint', default='http://localhost:11434')
    parser.add_argument('--export-every', type=int, default=4)
    parser.add_argument('--watch', action='store_true', help='Wait for newly acquired photos and continue until limit or source exhaustion')
    args = parser.parse_args()
    db = database()
    if args.export_only:
        export(db); return
    if args.extract_all or not db.execute("SELECT 1 FROM assets WHERE id LIKE 'docci:%' LIMIT 1").fetchone():
        import_docci(db, args.extract_all)
    teacher = Teacher(db, args.model, args.endpoint)
    if args.watch:
        processed = 0
        marker = ROOT / '02_metadata/docci_import_complete.json'
        extracted = marker.exists()
        failures = 0
        while not reached(db) and (args.limit is None or processed < args.limit):
            if (ROOT / 'STOP').exists():
                export(db); print('STOP file detected; generation stopped gracefully', flush=True); return
            archive = ROOT / '00_sources/docci/docci_images.tar.gz'
            if not extracted and archive.exists():
                import_docci(db, True); extracted = True
                marker.write_text(jd({'complete': True, 'updated_unix': time.time()}), encoding='utf-8')
            pending_count = db.execute("SELECT count(*) FROM assets WHERE state='pending'").fetchone()[0]
            if pending_count:
                import subprocess, sys
                size = min(settings()['worker_batch'], (args.limit - processed) if args.limit else pending_count, pending_count)
                result = subprocess.run([sys.executable, '-u', __file__, '--limit', str(size), '--export-every', str(args.export_every), '--model', args.model, '--endpoint', args.endpoint])
                if result.returncode:
                    failures += 1
                    if failures >= settings()['process_retries']: raise RuntimeError('generation failed repeatedly; saved data retained')
                    print('WORKER_RESTART', failures, flush=True)
                    time.sleep(settings()['retry_delay_seconds'])
                else:
                    processed += size; failures = 0
            else:
                if extracted and (ROOT / '00_sources/openimages/acquisition_full_complete.json').exists():
                    export(db)
                    print('SOURCE EXHAUSTED: candidate pool processed; 2M+2M target not asserted', flush=True)
                    return
                print('WAITING for new source photos', flush=True)
                time.sleep(30)
        export(db)
        print('AUTOMATED SAMPLE TARGETS REACHED' if reached(db) else 'WORKER LIMIT REACHED', flush=True)
        return
    pending = db.execute("SELECT * FROM assets WHERE state='pending' ORDER BY id LIMIT ?", (args.limit or 64,)).fetchall()
    for i, row in enumerate(pending):
        if reached(db): break
        if (ROOT / 'STOP').exists():
            print('STOP file detected', flush=True); break
        print('PROCESS', i + 1, '/', len(pending), row['id'], flush=True)
        for attempt in range(3):
            try:
                db.execute('UPDATE assets SET attempts=attempts+1 WHERE id=?', (row['id'],)); db.commit()
                current = db.execute('SELECT * FROM assets WHERE id=?', (row['id'],)).fetchone()
                count = process(db, teacher, current, int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16) % 100000)
                print('PASS', row['id'], 'sft', count, flush=True)
                break
            except (ValueError, KeyError, TypeError) as exc:
                is_last = attempt == 2 or str(exc).startswith('image_reject:')
                db.execute('UPDATE assets SET state=?,reason=?,facts=NULL WHERE id=?', ('rejected' if is_last else 'pending', str(exc), row['id'])); db.commit()
                print('REJECT' if is_last else 'RETRY', row['id'], str(exc), flush=True)
                if is_last: break
            except requests.RequestException as exc:
                db.execute('UPDATE assets SET reason=? WHERE id=?', (str(exc), row['id'])); db.commit()
                print('TEACHER_ERROR', str(exc), flush=True)
                export(db)
                raise
        if (i + 1) % max(1, args.export_every) == 0:
            export(db)
    export(db)

if __name__ == '__main__':
    main()
