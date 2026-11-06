# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Atomic GRPO writes with bounded recovery from Windows sharing violations."""
import errno
import json
import os
import time
import uuid
from pathlib import Path


def replace_with_retry(source, target, attempts=9, delay=.05):
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            transient = (getattr(exc, 'winerror', None) in (5, 32, 33) or
                         exc.errno in (errno.EACCES, errno.EBUSY, errno.EPERM))
            if not transient or attempt == attempts-1:
                raise
            time.sleep(min(delay*2**attempt, .5))


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        writer(temporary)
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, payload):
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write(path, lambda temporary: temporary.write_text(data, encoding='utf-8'))


def atomic_save(path, payload):
    import torch
    _atomic_write(path, lambda temporary: torch.save(payload, temporary))
