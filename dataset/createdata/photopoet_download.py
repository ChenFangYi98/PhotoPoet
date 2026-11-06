# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Download reproducible PhotoPoet pilot assets; existing training data is never read."""
from __future__ import annotations
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import time
import requests
import argparse

ROOT = Path(__file__).resolve().parent / 'PhotoPoet-Data'
MODEL = ROOT / '00_sources' / 'teacher' / 'Qwen3-VL-4B-Instruct'
REV = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'

def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = requests.head(url, allow_redirects=True, timeout=60)
    meta.raise_for_status()
    total = int(meta.headers.get('content-length', 0))
    if dest.exists() and total and dest.stat().st_size == total:
        return {'url': url, 'path': str(dest.relative_to(ROOT)), 'bytes': total}
    part = dest.with_name(dest.name + '.part')
    if total > 100 * 1024**2:
        # Keep any contiguous bytes from an earlier run, fetch the remainder
        # through bounded independent byte ranges, then concatenate in order.
        offset = part.stat().st_size if part.exists() else 0
        chunk_dir = dest.with_name(dest.name + '.chunks')
        chunk_dir.mkdir(exist_ok=True)
        chunk_size = 16 * 1024**2
        ranges = [(start, min(start + chunk_size, total) - 1) for start in range(offset, total, chunk_size)]
        def fetch_range(pair):
            start, end = pair
            chunk_path = chunk_dir / f'{start:014d}-{end:014d}'
            expected = end - start + 1
            if chunk_path.exists() and chunk_path.stat().st_size == expected:
                return chunk_path
            for retry in range(8):
                try:
                    with requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(30, 60)) as r:
                        r.raise_for_status()
                        if r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                            raise IOError('server did not honor byte range')
                        with chunk_path.open('wb') as out:
                            for data in r.iter_content(1024 * 1024):
                                out.write(data)
                    if chunk_path.stat().st_size != expected:
                        raise IOError('incomplete byte range')
                    return chunk_path
                except (requests.RequestException, OSError):
                    time.sleep(min(2**retry, 15))
            raise IOError(f'failed byte range {start}-{end}')
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch_range, ranges))
        with part.open('ab') as out:
            for start, end in ranges:
                chunk_path = chunk_dir / f'{start:014d}-{end:014d}'
                with chunk_path.open('rb') as src:
                    while data := src.read(4 * 1024**2):
                        out.write(data)
                chunk_path.unlink()
        if part.stat().st_size != total:
            raise IOError('assembled file size mismatch')
        part.replace(dest)
        chunk_dir.rmdir()
        print(f'DOWNLOADED {dest.name} {total}', flush=True)
        return {'url': url, 'path': str(dest.relative_to(ROOT)), 'bytes': total}
    for attempt in range(8):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(60, 120)) as r:
                r.raise_for_status()
                mode = 'ab' if offset and r.status_code == 206 else 'wb'
                with part.open(mode) as out:
                    for chunk in r.iter_content(4 * 1024 * 1024):
                        if chunk:
                            out.write(chunk)
            if total and part.stat().st_size != total:
                raise IOError(f'incomplete download: {part.stat().st_size}/{total}')
            part.replace(dest)
            print(f'DOWNLOADED {dest.name} {dest.stat().st_size}', flush=True)
            return {'url': url, 'path': str(dest.relative_to(ROOT)), 'bytes': dest.stat().st_size}
        except (requests.RequestException, OSError) as exc:
            print(f'RETRY {dest.name}: {exc}', flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f'download failed: {url}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', action='store_true', help='Optional Transformers teacher; default uses the existing local Ollama teacher')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    siblings = []
    if args.teacher:
        d = requests.get(f'https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct/revision/{REV}', timeout=60)
        d.raise_for_status()
        siblings = d.json()['siblings']
    for item in siblings:
        name = item['rfilename']
        if name.endswith(('.json', '.txt', '.safetensors', '.md')):
            jobs.append((f'https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/{REV}/{name}', MODEL / name))
    for name in ['docci_descriptions.jsonlines', 'docci_metadata.jsonlines', 'docci_images.tar.gz']:
        jobs.append(('https://storage.googleapis.com/docci/data/' + name, ROOT / '00_sources' / 'docci' / name))
    results = []
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(download, *job): job for job in jobs}
        while pending:
            done, _ = cf.wait(pending, timeout=30, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                results.append(fut.result())
                del pending[fut]
            active = {p.name: round(p.stat().st_size / 1024**3, 3) for p in ROOT.rglob('*.part')}
            print(json.dumps({'pending': len(pending), 'downloaded_GiB': active}), flush=True)
    for item in results:
        path = ROOT / item['path']
        with path.open('rb') as f:
            item['sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    (ROOT / '00_sources' / 'download_manifest.json').write_text(json.dumps({'teacher_revision': REV if args.teacher else None, 'assets': results}, indent=2), encoding='utf-8')
    print('DOWNLOAD COMPLETE', flush=True)

if __name__ == '__main__':
    main()
