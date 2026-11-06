# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""继续预训练的本地日志、原子保存与可复现数据游标。"""
import csv
import hashlib
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_info(path, checksum=False):
    path = Path(path).resolve()
    stat = path.stat()
    info = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if checksum:
        info['sha256'] = sha256_file(path)
    return info


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_torch_save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def make_logger(folder, name='continue_pretrain'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(folder / 'train.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


METRIC_FIELDS = ['timestamp', 'session', 'phase', 'epoch', 'batch', 'update', 'total_updates',
                 'lm_loss', 'aux_loss', 'total_loss', 'learning_rate', 'grad_norm',
                 'samples', 'valid_tokens', 'total_samples', 'total_valid_tokens',
                 'tokens_per_second', 'elapsed_seconds', 'eta_seconds',
                 'gpu_allocated_gb', 'gpu_reserved_gb', 'gpu_peak_gb',
                 'val_nll', 'val_ppl', 'val_tokens', 'best_val_nll']


class MetricsWriter:
    def __init__(self, folder, fieldnames=None):
        self.fields = fieldnames or METRIC_FIELDS
        path = folder / 'metrics.csv'
        exists = path.exists() and path.stat().st_size > 0
        self.csv_stream = path.open('a', encoding='utf-8-sig', newline='')
        self.json_stream = (folder / 'metrics.jsonl').open('a', encoding='utf-8')
        self.writer = csv.DictWriter(self.csv_stream, fieldnames=self.fields)
        if not exists:
            self.writer.writeheader()

    def write(self, row):
        self.json_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.writer.writerow({key: row.get(key) for key in self.fields})
        self.json_stream.flush()
        self.csv_stream.flush()

    def close(self):
        self.csv_stream.close()
        self.json_stream.close()


def rewind_metrics(folder, update, session, fieldnames=None):
    """保留已保存更新，回退未保存的日志；原日志另备份，避免混淆重复update。"""
    path = folder / 'metrics.jsonl'
    if not path.exists():
        return
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not any(row['update'] > update for row in rows):
        return
    for name in ('metrics.jsonl', 'metrics.csv'):
        original = folder / name
        if original.exists():
            original.rename(folder / f'{name}.before_resume_{session}')
    writer = MetricsWriter(folder, fieldnames=fieldnames)
    try:
        for row in rows:
            if row['update'] <= update:
                writer.write(row)
    finally:
        writer.close()


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class EpochBatchSampler(Sampler):
    """固定seed+epoch的排列，直接跳过已提交batch，不重新分词这些样本。"""
    def __init__(self, length, batch_size, seed, skip_batches=0):
        self.length, self.batch_size, self.seed, self.skip_batches = length, batch_size, seed, skip_batches

    def __iter__(self):
        indices = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed))
        for offset in range(self.skip_batches * self.batch_size, self.length, self.batch_size):
            yield indices[offset:offset + self.batch_size].tolist()

    def __len__(self):
        return max(0, math.ceil(self.length / self.batch_size) - self.skip_batches)
