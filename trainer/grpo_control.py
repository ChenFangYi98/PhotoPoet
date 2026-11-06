# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Stop only at optimizer boundaries; expose truthful live training state."""
import os
import signal
import warnings
from datetime import datetime
from pathlib import Path

from trainer.grpo_resources import atomic_json


class StopRequested(Exception):
    pass


class StopController:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.flag = False
        self.handlers = {}

    @property
    def requested(self):
        return self.flag or (self.run_dir / 'STOP').exists()

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.handlers[sig] = signal.signal(sig, lambda *_: setattr(self, 'flag', True))
        return self

    def check(self):
        if self.requested:
            raise StopRequested()

    def __exit__(self, *_):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
        stop_file = self.run_dir / 'STOP'
        if stop_file.exists():
            stop_file.rename(stop_file.with_name('STOP.handled.' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))


def live_state(run_dir, status, progress, **extra):
    try:
        atomic_json(Path(run_dir) / 'runtime.json', dict(
            status=status, pid=os.getpid(), timestamp=datetime.now().isoformat(timespec='seconds'),
            update=progress['update'], total_updates=progress['total_updates'], **extra))
        return True
    except OSError as exc:
        # Telemetry can be stale; a model/checkpoint write must still raise on failure.
        warnings.warn(f'Runtime telemetry unavailable (training continues): {exc}', RuntimeWarning)
        return False
