"""One background evaluation per dashboard process; no Streamlit calls in workers."""
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, RLock, Thread
from uuid import uuid4

from app.services.benchmark import run_benchmark
from app.services.llm_client import GeminiClient
from app.services.report_store import ReportWriter
from app.services.run_inputs import validate_run


@dataclass
class Job:
    id: str
    state: str = 'starting'
    cancel: Event = field(default_factory=Event)
    thread: Thread | None = None
    json_path: Path | None = None
    csv_path: Path | None = None
    report: dict | None = None
    error: str | None = None


class RunManager:
    def __init__(self, directory, provider_factory=GeminiClient):
        self.directory = Path(directory)
        self.provider_factory = provider_factory
        self.lock = RLock()
        self.jobs = {}
        self.active = None
        self.latest = None

    def start(self, cases, options, settings, prior=None):
        with self.lock:
            if self.active is not None:
                raise ValueError('An evaluation is already active. Cancel it or wait for it to finish.')
            configured, budget, reused = validate_run(cases, options, settings, prior)
            writer = ReportWriter(self.directory)
            job = Job(uuid4().hex, json_path=writer.json_path, csv_path=writer.csv_path)
            # Completed job history is bounded; reports remain available on disk.
            for key in list(self.jobs)[:-9]:
                self.jobs.pop(key)
            self.jobs[job.id] = job
            self.active = self.latest = job.id
            job.thread = Thread(target=self._work, args=(job, writer, deepcopy(cases),
                                options.model_copy(deep=True), configured.model_copy(deep=True), deepcopy(prior)),
                                name='evalora-evaluation', daemon=True)
            try:
                job.thread.start()
            except Exception:
                self.jobs.pop(job.id, None)
                self.active = None
                self.latest = None
                raise ValueError('Unable to start the evaluation worker.') from None
            return job.id

    def _work(self, job, writer, cases, options, settings, prior):
        def checkpoint(report):
            report['launch_source'] = 'dashboard'
            writer.checkpoint(report)
            with self.lock:
                job.report = deepcopy(report)
        with self.lock:
            job.state = 'cancelling' if job.cancel.is_set() else 'running'
        try:
            report = run_benchmark(cases, options.models, settings, self.provider_factory,
                                   request_interval=options.request_interval,
                                   model_intervals=options.model_intervals, prior_report=prior,
                                   on_checkpoint=checkpoint, max_retries=options.max_retries,
                                   max_retry_delay=options.max_retry_delay, cancel_event=job.cancel)
            writer.finish(report)
            with self.lock:
                job.state = 'finished'
        except KeyboardInterrupt:
            with self.lock:
                job.state = 'cancelled'
        except OSError:
            with self.lock:
                job.state, job.error = 'failed', 'Unable to save the report. The last valid checkpoint, if any, remains available.'
        except Exception:
            with self.lock:
                job.state, job.error = 'failed', 'The evaluation stopped unexpectedly. Inspect the saved checkpoint; provider details were not exposed.'
        finally:
            with self.lock:
                if self.active == job.id:
                    self.active = None

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or self.active != job_id or job.state not in {'starting', 'running', 'cancelling'}:
                return False
            job.cancel.set()
            job.state = 'cancelling'
            return True

    def snapshot(self, job_id=None):
        with self.lock:
            job = self.jobs.get(job_id or self.latest)
            if job is None:
                return None
            return {'id': job.id, 'state': job.state, 'report': deepcopy(job.report),
                    'json_path': str(job.json_path) if job.json_path else None,
                    'csv_path': str(job.csv_path) if job.csv_path else None,
                    'error': job.error, 'active': self.active == job.id}

    def wait(self, job_id, timeout=5):
        with self.lock:
            thread = self.jobs[job_id].thread
        thread.join(timeout)
        return not thread.is_alive()
