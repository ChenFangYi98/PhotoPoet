# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Create a local adaptive-SFT manifest without modifying published metadata."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify_sha256', action='store_true', help='Read every source and verify its recorded SHA-256')
    args = parser.parse_args()
    original = args.manifest.resolve()
    root = args.data_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing file: {output}')
    manifest = json.loads(original.read_text(encoding='utf8'))
    for item in manifest['sources']:
        name = item['path'].replace('\\', '/')
        if '/dataset/' in name:
            name = name.split('/dataset/', 1)[1]
        elif name.startswith('dataset/'):
            name = name[len('dataset/'):]
        source = (root / name).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f'Source escapes data root: {source}')
        stat = source.stat()
        if stat.st_size != item['bytes']:
            raise ValueError(f'Source size mismatch: {source}')
        expected = item.get('expected_sha256') or item.get('sha256')
        if args.verify_sha256:
            if not expected:
                raise ValueError(f'Source has no recorded SHA-256: {source}')
            with source.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'Source SHA-256 mismatch: {source}')
        item['path'] = str(source)
        item['mtime_ns'] = stat.st_mtime_ns
    for item in manifest['indices'].values():
        index = (original.parent / item['file']).resolve()
        if not index.is_file():
            raise FileNotFoundError(index)
        item['file'] = str(index)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    mode = 'size and SHA-256' if args.verify_sha256 else 'size only; content SHA-256 not checked'
    print(f'Created {output}; source verification: {mode}')


if __name__ == '__main__':
    main()
