# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Merge locally created SFT records into the training Parquet, keeping a backup."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def record_key(row):
    conversation = row['conversations']
    if isinstance(conversation, str):
        conversation = json.loads(conversation)
    canonical = json.dumps(conversation, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(canonical).digest() + hashlib.sha256(row['image_bytes']).digest()


def validate_created(row):
    conversation = json.loads(row['conversations'])
    if [turn['role'] for turn in conversation] != ['user', 'assistant']:
        raise ValueError('Created SFT record must contain user and assistant turns')
    if conversation[0]['content'].count('<image>') != 1:
        raise ValueError('Created SFT record must contain one image placeholder')
    if not all(isinstance(turn['content'], str) and turn['content'].strip()
               for turn in conversation):
        raise ValueError('Empty conversation content')
    with Image.open(io.BytesIO(row['image_bytes'])) as image:
        image.verify()


def database_extras(created):
    """Include accepted records not yet present in the last exported snapshot."""
    root = created / 'PhotoPoet-Data'
    database = root / '02_metadata/index.sqlite3'
    manifests = root / '06_final/manifest'
    if not database.exists() or not manifests.exists():
        return []
    exported_ids = set()
    for path in sorted(manifests.glob('sft_*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            exported_ids.update(json.loads(line)['sample_id'] for line in stream)
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('BEGIN')
        # A sequential scan avoids thousands of random reads of large SQLite pages.
        missing = [dict(row) for row in connection.execute(
            "SELECT id,asset,prompt,target FROM samples NOT INDEXED WHERE stage='sft'"
        ) if row['id'] not in exported_ids]
        for row in missing:
            asset = connection.execute('SELECT path,sha,split FROM assets WHERE id=?',
                                       (row['asset'],)).fetchone()
            row.update(dict(asset))
    finally:
        connection.close()
    for row in missing:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Image outside the source directory: {path}')
        image = path.read_bytes()
        if hashlib.sha256(image).hexdigest() != row['sha']:
            raise ValueError(f'Image hash mismatch: {path}')
        row['record'] = {
            'conversations': json.dumps([
                {'role': 'user', 'content': row['prompt']},
                {'role': 'assistant', 'content': row['target']},
            ], ensure_ascii=False, separators=(',', ':')),
            'image_bytes': image,
        }
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    parser.add_argument('--target', type=Path, default=repository / 'dataset/sft_i2t.parquet')
    parser.add_argument('--created', type=Path, default=repository / 'dataset/createdata')
    args = parser.parse_args()
    target, created = args.target.resolve(), args.created.resolve()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = target.with_name(f'{target.stem}.before_createdata_{stamp}.parquet.bak')
    temporary = target.with_name(f'{target.name}.{stamp}.tmp')
    report_path = target.with_name(f'{target.stem}.merge_createdata_{stamp}.json')
    started = time.time()
    sources = sorted(path for path in created.rglob('*.parquet')
                     if 'sft' in path.name.lower())
    if not sources:
        raise ValueError('No SFT Parquet sources found')
    source_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                    for path in sources}
    original_stat = (target.stat().st_size, target.stat().st_mtime_ns)
    candidates = set()
    per_source = []
    print('Checking created SFT records...', flush=True)
    for path in sources:
        counts = Counter()
        with pq.ParquetFile(path) as parquet:
            if set(parquet.schema_arrow.names) != {'conversations', 'image_bytes'}:
                raise ValueError(f'Unexpected source fields: {path}')
            for batch in parquet.iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    validate_created(row)
                    key = record_key(row)
                    counts['rows'] += 1
                    counts['duplicates_with_created_sources'] += key in candidates
                    candidates.add(key)
        per_source.append({'path': str(path), **counts})
        print(json.dumps(per_source[-1]), flush=True)
    extras = database_extras(created)
    for item in extras:
        validate_created(item['record'])
        candidates.add(record_key(item['record']))
    unique_created = len(candidates)
    print(f'Unique created records: {unique_created}; unexported database rows: {len(extras)}', flush=True)
    remaining = candidates.copy()
    original = pq.ParquetFile(target)
    schema = original.schema_arrow
    original_rows = original.metadata.num_rows
    written = 0
    appended = Counter()
    # Digests verify every logical value after writing, independent of compression.
    expected_digest = hashlib.sha256()
    try:
        with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
            for batch in original.iter_batches(batch_size=5000):
                for row in batch.to_pylist():
                    key = record_key(row)
                    expected_digest.update(key)
                    remaining.discard(key)
                writer.write_table(pa.Table.from_batches([batch], schema=schema))
                written += batch.num_rows
                if written % 250000 == 0:
                    print(f'Original rows copied: {written}/{original_rows}', flush=True)
            for path in sources:
                with pq.ParquetFile(path) as source:
                    for batch in source.iter_batches(batch_size=16):
                        accepted = []
                        for row in batch.to_pylist():
                            key = record_key(row)
                            if key in remaining:
                                remaining.remove(key)
                                expected_digest.update(key)
                                accepted.append(row)
                        if accepted:
                            writer.write_table(pa.Table.from_pylist(accepted, schema=schema))
                            appended[str(path)] += len(accepted)
            accepted = []
            for item in extras:
                row = item['record']
                key = record_key(row)
                if key in remaining:
                    remaining.remove(key)
                    expected_digest.update(key)
                    accepted.append(row)
            if accepted:
                writer.write_table(pa.Table.from_pylist(accepted, schema=schema))
                appended['database_unexported'] = len(accepted)
        original.close()
        added = sum(appended.values())
        if remaining:
            raise ValueError(f'{len(remaining)} created records were not written')
        print(f'Verifying merged file: {original_rows + added} rows...', flush=True)
        actual_digest = hashlib.sha256()
        with pq.ParquetFile(temporary) as merged:
            if merged.metadata.num_rows != original_rows + added:
                raise ValueError('Merged row count mismatch')
            if not merged.schema_arrow.equals(schema, check_metadata=True):
                raise ValueError('Merged schema mismatch')
            verified = 0
            for batch in merged.iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    actual_digest.update(record_key(row))
                verified += batch.num_rows
                if verified % 500000 < 64:
                    print(f'Verified rows: {verified}', flush=True)
        if actual_digest.digest() != expected_digest.digest():
            raise ValueError('Merged content digest mismatch')
        for path in sources:
            if source_stats[str(path)] != (path.stat().st_size, path.stat().st_mtime_ns):
                raise ValueError(f'Source changed during merge: {path}')
        if original_stat != (target.stat().st_size, target.stat().st_mtime_ns):
            raise ValueError('Target changed during merge')
        report = {
            'target': str(target), 'backup': str(backup),
            'original_rows': original_rows, 'added_rows': added,
            'total_rows': original_rows + added, 'sources': per_source,
            'database_unexported_rows': len(extras),
            'database_unexported_sample_ids': [item['id'] for item in extras],
            'unique_created_rows': unique_created,
            'already_in_target': unique_created - added,
            'added_by_source': dict(appended),
            'all_splits_included': ['train', 'validation', 'test'],
            'evaluation_note': 'Merged data includes held-out splits; use separate evaluation data.',
            'schema': str(schema), 'content_sha256': actual_digest.hexdigest(),
            'verified_all_rows': True, 'elapsed_seconds': round(time.time() - started, 2),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        target.replace(backup)
        try:
            temporary.replace(target)
        except BaseException:
            backup.replace(target)
            raise
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f'Report: {report_path}', flush=True)
    finally:
        original.close()
        if temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()
