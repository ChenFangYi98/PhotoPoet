# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Verify the emitted binary-image/string-conversations Parquet contract."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
from photopoet_build import ROOT, SCHEMA, validate_text

def main():
    if (ROOT / '06_final/manifest/shards.json').exists():
        from photopoet_export import verify_dataset
        verify_dataset()
        return
    counts = Counter(); clusters = defaultdict(set); errors = []
    manifests = {}
    for path in (ROOT / '06_final/manifest').glob('*.jsonl'):
        rows = [json.loads(line) for line in path.open(encoding='utf-8')]
        manifests[path.stem] = rows
        for row in rows:
            clusters[row['cluster_id']].add(row['split'])
    for path in sorted((ROOT / '06_final').rglob('*.parquet')):
        stage, split = path.parent.parent.name, path.parent.name
        f = pq.ParquetFile(path)
        if not f.schema_arrow.equals(SCHEMA, check_metadata=False):
            errors.append(f'{path}: schema mismatch')
        meta = manifests.get(stage + '_' + split, [])
        shard = int(path.stem.rsplit('_', 1)[1]); offset = shard * 10000
        for batch in f.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                try:
                    conv = json.loads(row['conversations'])
                    assert [x['role'] for x in conv] == ['user', 'assistant']
                    assert conv[0]['content'].count('<image>') == 1
                    assert isinstance(row['image_bytes'], bytes)
                    with Image.open(io.BytesIO(row['image_bytes'])) as im:
                        im.load(); assert min(im.size) >= 128
                    item = meta[offset]
                    assert hashlib.sha256(row['image_bytes']).hexdigest() == item['sha256']
                    if stage == 'sft':
                        assert validate_text(item['task_type'], conv[1]['content']) is None
                    counts[stage + '_' + split] += 1
                except (AssertionError, ValueError, KeyError, IndexError, OSError) as exc:
                    errors.append(f'{path.name}:{offset}: {type(exc).__name__} {exc}')
                finally:
                    offset += 1
    for name, rows in manifests.items():
        if counts[name] != len(rows): errors.append(f'{name}: manifest count mismatch')
    overlap = {key: sorted(value) for key, value in clusters.items() if len(value) != 1}
    if overlap: errors.append('cluster split overlap: ' + str(overlap))
    report = {'structural_pass': not errors, 'counts': dict(counts), 'cluster_split_overlap': overlap,
              'errors': errors, 'human_quality_acceptance': False,
              'notes': ['Visual factuality and literary quality require separate review.', 'Automated self-review scores are uncalibrated.', 'No claim of 2M+2M completion.']}
    target = ROOT / '05_review'; target.mkdir(parents=True, exist_ok=True)
    (target / 'structural_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors: raise SystemExit(1)

if __name__ == '__main__':
    main()
