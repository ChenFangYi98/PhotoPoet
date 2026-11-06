# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Supervise the 2M pretrain + 2M SFT run with resumable source acquisition."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from photopoet_build import ROOT, database
from photopoet_config import settings, count, reached

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'run.lock').open('a+b')
    if lock.tell() == 0: lock.write(b'0'); lock.flush()
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError: raise SystemExit('Another PhotoPoet supervisor is already running.')
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scripts = Path(__file__).resolve().parent
    logs = ROOT / 'logs'; logs.mkdir(exist_ok=True)
    db = database()
    if (ROOT / 'STOP').exists(): raise SystemExit('Remove PhotoPoet-Data/STOP to resume.')
    jobs = [
        ('openimages_full', [str(scripts / 'photopoet_acquire_full.py')]),
        ('generate', [str(scripts / 'photopoet_build.py'), '--watch', '--export-every', str(settings()['export_every'])]),
    ]
    if not all((ROOT / '00_sources/docci' / name).exists() for name in ['docci_images.tar.gz', 'docci_descriptions.jsonlines', 'docci_metadata.jsonlines']):
        jobs.insert(0, ('download', [str(scripts / 'photopoet_download.py')]))
    handles = {}; processes = {}; retries = {}; start_times = {}
    state_path = ROOT / 'run_state.json'
    state = {'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'status': 'running',
             'targets': settings()['targets'], 'scope': 'full official Open Images train metadata + DOCCI; stop at actual 2M+2M rows',
             'human_quality_acceptance': False, 'processes': {}, 'restarts': {}}
    baseline = {stage: count(db, stage) for stage in settings()['targets']}
    def launch(name, args):
        if name not in handles: handles[name] = (logs / (name + '.log')).open('a', encoding='utf-8')
        process = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', *args], cwd=scripts,
            stdout=handles[name], stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        processes[name] = process; start_times[name] = time.time()
        state['processes'][name] = process.pid
    def publish():
        state['counts_in_database'] = {stage: count(db, stage) for stage in settings()['targets']}
        state['remaining'] = {stage: max(0, goal - state['counts_in_database'][stage]) for stage, goal in settings()['targets'].items()}
        elapsed = max(1, time.time() - state['started_unix'])
        state['rows_per_hour_since_start'] = {stage: round((value - baseline[stage]) * 3600 / elapsed, 2) for stage, value in state['counts_in_database'].items()}
        state['updated_unix'] = time.time()
        tmp = state_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(state, indent=2), encoding='utf-8'); tmp.replace(state_path)
    try:
        for name, args in jobs: launch(name, args)
        publish()
        while True:
            stopping = (ROOT / 'STOP').exists()
            for name, args in jobs:
                code = processes[name].poll()
                if code not in (None, 0) and not stopping:
                    retries[name] = retries.get(name, 0) + 1
                    state['restarts'][name] = retries[name]
                    if retries[name] > settings()['process_retries']:
                        raise RuntimeError(f'{name} failed repeatedly; see logs/{name}.log')
                    state['status'] = 'recovering'; publish()
                    time.sleep(settings()['retry_delay_seconds'])
                    launch(name, args)
                    state['status'] = 'running'
            if stopping and processes['generate'].poll() is not None:
                state['status'] = 'stopped'; break
            if all(p.poll() == 0 for p in processes.values()):
                state['status'] = 'automated_targets_reached_pending_human_review' if reached(db) else 'sources_exhausted_targets_unmet'
                break
            publish(); time.sleep(10)
        state['finished_unix'] = time.time(); publish()
    except Exception as exc:
        state['status'] = 'error'; state['error'] = str(exc); publish()
        raise
    finally:
        for process in processes.values():
            if process.poll() is None: process.terminate()
        for handle in handles.values(): handle.close()
        db.close(); lock.close()

if __name__ == '__main__':
    main()
