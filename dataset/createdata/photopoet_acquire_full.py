# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Continuously replenish a bounded queue from the official full OI train list."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlparse
import requests
from PIL import Image
from photopoet_build import ROOT, database, add_asset, jd
from photopoet_config import settings, reached
from photopoet_download import download

URL = 'https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv'

def fetch(row):
    ident = row['ImageID']
    if not re.fullmatch('[0-9a-f]{16}', ident): return row, None, 'invalid image ID'
    path = ROOT / '01_images/openimages' / ident[:2] / (ident + '.jpg')
    if path.exists(): return row, path, 'retained_existing_download'
    errors = []
    # The S3 mirror contains only the boxable subset. The full list needs the
    # publisher-supplied thumbnail/original URLs, not fabricated S3 addresses.
    urls = [row.get('Thumbnail300KURL'), row.get('OriginalURL')]
    for url in dict.fromkeys(u for u in urls if u):
        if urlparse(url).scheme not in ('http', 'https'): continue
        try:
            data = bytearray()
            with requests.get(url, stream=True, timeout=(15, 40)) as response:
                response.raise_for_status()
                if int(response.headers.get('Content-Length', 0)) > 20 * 1024**2:
                    raise ValueError('image exceeds 20 MiB')
                for chunk in response.iter_content(256 * 1024):
                    data.extend(chunk)
                    if len(data) > 20 * 1024**2: raise ValueError('image exceeds 20 MiB')
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix('.jpg.part')
            part.write_bytes(data); part.replace(path)
            return row, path, url
        except (requests.RequestException, OSError, ValueError) as exc:
            errors.append(str(exc))
    return row, None, '; '.join(errors)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='Optional debugging limit, not the full-run default')
    args = parser.parse_args()
    folder = ROOT / '00_sources/openimages'; folder.mkdir(parents=True, exist_ok=True)
    source = folder / 'oidv6-train-images-with-labels-with-rotation.csv'
    download(URL, source)
    db = database()
    progress_path = folder / 'acquisition_full_progress.json'
    progress = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {}
    last_row = progress.get('last_row', 0)
    imported = progress.get('imported_new_assets', 0)
    failed = progress.get('failed_downloads', 0)
    added_this_run = 0
    def update(state):
        progress.update(status=state, last_row=last_row, imported_new_assets=imported,
                        failed_downloads=failed, source_url=URL, updated_unix=time.time())
        temporary = progress_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(progress, indent=2), encoding='utf-8'); temporary.replace(progress_path)
    with source.open(encoding='utf-8', newline='') as stream, (folder / 'download_full_log.jsonl').open('a', encoding='utf-8') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = enumerate(csv.DictReader(stream), 1)
        for ordinal, row in rows:
            if ordinal > last_row: break
        else:
            row = None
        batch = []
        exhausted = row is None
        while not reached(db):
            if (ROOT / 'STOP').exists(): update('stopped'); return
            if args.limit is not None and added_this_run >= args.limit:
                update('debug_limit_reached_full_target_unmet'); return
            pending = db.execute("SELECT count(*) FROM assets WHERE state='pending'").fetchone()[0]
            if pending >= settings()['pending_buffer']:
                update('waiting_for_generation'); time.sleep(30); continue
            if shutil.disk_usage(ROOT).free < settings()['min_free_gib'] * 1024**3:
                update('waiting_for_disk_space'); time.sleep(30); continue
            batch_end = last_row
            while len(batch) < min(64, settings()['pending_buffer'] - pending) and not exhausted:
                batch_end = ordinal
                if row['Subset'] == 'train' and '/licenses/by/' in row['License'] and row.get('Rotation', '') in ('0', '0.0'):
                    if not db.execute('SELECT 1 FROM assets WHERE id=?', ('openimages:' + row['ImageID'],)).fetchone():
                        batch.append(row)
                try: ordinal, row = next(rows)
                except StopIteration: exhausted = True
            for item, path, location in pool.map(fetch, batch):
                ok = False
                if path is not None:
                    try:
                        add_asset(db, 'openimages:' + item['ImageID'], path, 'openimages:' + item['ImageID'], {
                            'dataset': 'OpenImages', 'original_id': item['ImageID'], 'original_split': 'train',
                            'source_url': item['OriginalLandingURL'], 'download_url': location,
                            'original_url': item['OriginalURL'], 'license': item['License'],
                            'author': item['Author'], 'author_url': item['AuthorProfileURL'], 'title': item['Title'],
                            'metadata_url': URL, 'rotation': item.get('Rotation'),
                            'usage_status': 'research_only_pending_source_review',
                        })
                        ok = bool(db.execute('SELECT 1 FROM assets WHERE id=?', ('openimages:' + item['ImageID'],)).fetchone())
                    except (ValueError, OSError) as exc: location = str(exc)
                if ok: imported += 1; added_this_run += 1
                elif path is None: failed += 1
                log.write(jd({'original_id': item['ImageID'], 'ok': ok, 'download': location, 'metadata': item}) + '\n')
            log.flush(); batch.clear(); last_row = batch_end
            update('running')
            print('FULL_SOURCE', jd(progress), flush=True)
            if exhausted:
                update('source_exhausted_full_target_unmet')
                (folder / 'acquisition_full_complete.json').write_text(jd(progress), encoding='utf-8')
                return
    update('automated_sample_targets_reached_pending_human_review')

if __name__ == '__main__':
    main()
