# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Stream shards and reuse unchanged completed shards on subsequent exports."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import pyarrow as pa
import pyarrow.parquet as pq
from photopoet_build import ROOT, SCHEMA, jd
from photopoet_config import settings, reached

def export_dataset(db):
    db.execute('BEGIN')
    final = ROOT / '06_final'; manifest = final / 'manifest'
    manifest.mkdir(parents=True, exist_ok=True)
    counts = Counter(); entries = []; changed = 0
    try:
        for stage in ('pretrain', 'sft'):
            for split in ('train', 'validation', 'test'):
                counts[f'{stage}_{split}'] = 0
                folder = final / stage / split; folder.mkdir(parents=True, exist_ok=True)
                cursor = db.execute('SELECT s.*,a.path,a.sha,a.cluster_id,a.split,a.provenance FROM samples s JOIN assets a ON a.id=s.asset WHERE s.stage=? AND a.split=? ORDER BY s.rowid', (stage, split))
                shard = 0; expected = set()
                while records := cursor.fetchmany(10000):
                    dest = folder / f'photopoet_{stage}_{shard:05d}.parquet'
                    metadata = manifest / f'{stage}_{split}_{shard:05d}.jsonl'
                    signature = metadata.with_suffix('.sha256')
                    fingerprint = hashlib.sha256(jd([dict(r) for r in records]).encode()).hexdigest()
                    cached = dest.exists() and metadata.exists() and signature.exists() and signature.read_text() == fingerprint
                    if not cached:
                        temp = dest.with_suffix('.parquet.tmp')
                        temp_meta = metadata.with_suffix('.jsonl.tmp')
                        with pq.ParquetWriter(temp, SCHEMA, compression='zstd') as writer, temp_meta.open('w', encoding='utf-8') as output:
                            batch = []
                            for row in records:
                                conv = [{'role': 'user', 'content': row['prompt']}, {'role': 'assistant', 'content': row['target']}]
                                batch.append({'conversations': jd(conv), 'image_bytes': (ROOT / row['path']).read_bytes()})
                                output.write(jd({'sample_id': row['id'], 'asset_id': row['asset'], 'sha256': row['sha'], 'cluster_id': row['cluster_id'], 'split': split, 'task_type': row['task'], 'source': json.loads(row['provenance']), 'qc': json.loads(row['review'])}) + '\n')
                                if len(batch) == 16:
                                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA)); batch.clear()
                            if batch: writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                        temp.replace(dest); temp_meta.replace(metadata)
                        signature.write_text(fingerprint)
                        changed += 1
                    entries.append({'stage': stage, 'split': split, 'rows': len(records),
                                    'parquet': dest.relative_to(final).as_posix(),
                                    'manifest': metadata.relative_to(final).as_posix(), 'fingerprint': fingerprint})
                    expected.add(dest.name); counts[f'{stage}_{split}'] += len(records); shard += 1
                for stale in folder.glob(f'photopoet_{stage}_*.parquet'):
                    if stale.name not in expected: stale.unlink()
                old = manifest / f'{stage}_{split}.jsonl'
                if old.exists():
                    archive = manifest / 'legacy'; archive.mkdir(exist_ok=True)
                    destination = archive / old.name
                    if destination.exists(): destination = archive / (str(time.time_ns()) + '_' + old.name)
                    old.replace(destination)
        index = {'version': 2, 'updated_unix': time.time(), 'shards': entries}
        temp_index = manifest / 'shards.json.tmp'
        temp_index.write_text(json.dumps(index, indent=2), encoding='utf-8'); temp_index.replace(manifest / 'shards.json')
        info = {'status': 'automated_targets_reached_pending_human_review' if reached(db) else 'automated_candidates_pending_human_review',
                'targets': settings()['targets'], 'counts': dict(counts), 'manifest_format': 'per_shard_jsonl_v2',
                'unique_images': db.execute('SELECT count(DISTINCT asset) FROM samples').fetchone()[0],
                'unique_images_by_stage': dict(db.execute('SELECT stage,count(DISTINCT asset) FROM samples GROUP BY stage').fetchall()),
                'assets': dict(db.execute('SELECT state,count(*) FROM assets GROUP BY state').fetchall()),
                'tasks': dict(db.execute("SELECT task,count(*) FROM samples WHERE stage='sft' GROUP BY task").fetchall()),
                'human_reviewed': 0, 'thresholds_calibrated': False, 'prosody_checked': False,
                'shards_written_this_export': changed, 'updated_unix': time.time()}
        temporary = manifest / 'status.json.tmp'
        temporary.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8'); temporary.replace(manifest / 'status.json')
        db.commit()
        print('EXPORTED', jd(info), flush=True)
    except Exception:
        db.rollback(); raise

def verify_dataset():
    import io
    import sqlite3
    from PIL import Image
    from photopoet_build import validate_text
    final = ROOT / '06_final'
    index = json.loads((final / 'manifest/shards.json').read_text(encoding='utf-8'))
    seen = sqlite3.connect(':memory:')
    seen.execute('CREATE TABLE clusters(id TEXT PRIMARY KEY,split TEXT)')
    counts = Counter(); errors = []
    for entry in index['shards']:
        path = final / entry['parquet']; count = 0
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(SCHEMA, check_metadata=False): errors.append(str(path) + ': schema mismatch')
        with (final / entry['manifest']).open(encoding='utf-8') as meta:
            for batch in parquet.iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    try:
                        item = json.loads(next(meta)); conv = json.loads(row['conversations'])
                        assert [t['role'] for t in conv] == ['user', 'assistant']
                        assert conv[0]['content'].count('<image>') == 1
                        assert hashlib.sha256(row['image_bytes']).hexdigest() == item['sha256']
                        assert item['split'] == entry['split']
                        with Image.open(io.BytesIO(row['image_bytes'])) as image:
                            image.load(); assert min(image.size) >= 128
                        if entry['stage'] == 'sft': assert validate_text(item['task_type'], conv[1]['content']) is None
                        prior = seen.execute('SELECT split FROM clusters WHERE id=?', (item['cluster_id'],)).fetchone()
                        assert not prior or prior[0] == entry['split'], 'cluster split overlap'
                        seen.execute('INSERT OR IGNORE INTO clusters VALUES(?,?)', (item['cluster_id'], entry['split']))
                    except (AssertionError, ValueError, KeyError, StopIteration, OSError) as exc:
                        errors.append(f'{path.name}:{count}: {exc}')
                    count += 1
            if next(meta, None) is not None: errors.append(str(path) + ': excess manifest rows')
        if count != entry['rows']: errors.append(str(path) + ': shard row count mismatch')
        counts[entry['stage'] + '_' + entry['split']] += count
    report = {'structural_pass': not errors, 'counts': dict(counts), 'errors': errors, 'human_quality_acceptance': False, 'verified_unix': time.time()}
    target = ROOT / '05_review'; target.mkdir(exist_ok=True)
    (target / 'structural_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors: raise SystemExit(1)
