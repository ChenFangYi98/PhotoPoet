# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Fetch new Open Images training candidates with per-image attribution.

The metadata file is retained as provenance. Only CC-BY entries are imported;
the visual teacher still filters photographs, privacy, and suitability later.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import requests

from photopoet_build import ROOT, database, add_asset, jd
from photopoet_download import download

URL = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'

def fetch(row):
    ident = row['ImageID']
    path = ROOT / '01_images/openimages' / ident[:2] / (ident + '.jpg')
    url = f'https://open-images-dataset.s3.amazonaws.com/train/{ident}.jpg'
    if path.exists():
        return row, path, url
    try:
        r = requests.get(url, timeout=(15, 40))
        r.raise_for_status()
        if len(r.content) > 20 * 1024**2:
            raise ValueError('oversize image')
        from PIL import Image
        with Image.open(io.BytesIO(r.content)) as image:
            image.verify()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix('.jpg.part').write_bytes(r.content)
        path.with_suffix('.jpg.part').replace(path)
        return row, path, url
    except (requests.RequestException, OSError, ValueError) as e:
        return row, None, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    folder = ROOT / '00_sources/openimages'; folder.mkdir(parents=True, exist_ok=True)
    source = folder / 'train-images-boxable-with-rotation.csv'
    download(URL, source)
    db = database()
    done = {r[0].split(':', 1)[1] for r in db.execute("SELECT id FROM assets WHERE id LIKE 'openimages:%'")}
    needed = max(0, args.limit - len(done))
    imported = failed = 0
    with source.open(encoding='utf-8', newline='') as f, (folder / 'download_log.jsonl').open('a', encoding='utf-8') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        selected = (row for row in csv.DictReader(f) if row['Subset'] == 'train' and '/licenses/by/' in row['License'] and row['ImageID'] not in done)
        while imported < needed:
            if (ROOT / 'STOP').exists():
                print('STOP file detected', flush=True); return
            if shutil.disk_usage(ROOT).free < 100 * 1024**3:
                raise RuntimeError('Less than 100 GiB free: stopped before filling the disk')
            batch = []
            for _ in range(min(64, needed - imported)):
                row = next(selected, None)
                if row is None: break
                batch.append(row)
            if not batch: break
            for row, path, location in pool.map(fetch, batch):
                log.write(jd({'original_id': row['ImageID'], 'metadata': row, 'ok': path is not None, 'download': location}) + '\n'); log.flush()
                if path is None:
                    failed += 1
                    continue
                before = db.total_changes
                add_asset(db, 'openimages:' + row['ImageID'], path, 'openimages:' + row['ImageID'], {
                    'dataset': 'OpenImages', 'original_id': row['ImageID'], 'original_split': 'train',
                    'source_url': row['OriginalLandingURL'], 'download_url': location,
                    'original_url': row['OriginalURL'], 'license': row['License'],
                    'author': row['Author'], 'author_url': row['AuthorProfileURL'], 'title': row['Title'],
                    'metadata_url': URL, 'usage_status': 'research_only_pending_source_review',
                })
                if db.total_changes > before: imported += 1
            print('OPENIMAGES', {'imported': imported, 'failed': failed}, flush=True)
    print('Open Images acquisition complete', imported, 'new assets', flush=True)
    (folder / 'acquisition_complete.json').write_text(json.dumps({'new_assets': imported, 'total_assets': imported + len(done), 'failed_downloads': failed, 'limit': args.limit}), encoding='utf-8')

if __name__ == '__main__':
    main()
