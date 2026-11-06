# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Shared full-run quotas and bounded-resource settings."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent / 'PhotoPoet-Data'
DEFAULTS = {
    'targets': {'pretrain': 2000000, 'sft': 2000000},
    'pending_buffer': 40000, 'min_free_gib': 150, 'worker_batch': 32,
    'export_every': 32, 'teacher_http_retries': 5, 'process_retries': 8,
    'retry_delay_seconds': 30,
}
SETTINGS_PATH = ROOT / 'full_run_config.json'
def settings():
    values = dict(DEFAULTS)
    if SETTINGS_PATH.exists(): values.update(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    return values
def count(db, stage):
    row = db.execute('SELECT value FROM counters WHERE name=?', ('stage:' + stage,)).fetchone()
    return row[0] if row else 0
def reached(db):
    return all(count(db, stage) >= goal for stage, goal in settings()['targets'].items())
def task_quotas():
    total = settings()['targets']['sft']
    weights = {'qijue': 8, 'wujue': 7, 'modern': 9, 'social': 6, 'mood': 4, 'keywords': 3, 'interpret': 3}
    quotas = {task: total * weight // 40 for task, weight in weights.items()}
    quotas['modern'] += total - sum(quotas.values())
    return quotas
