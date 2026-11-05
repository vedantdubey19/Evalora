import json
from pathlib import Path
from threading import Event

import pytest
from streamlit.testing.v1 import AppTest

from app.core.config import Settings
from app.services.benchmark import run_benchmark, validate_cases
from app.services.run_inputs import RunOptions, parse_dataset, read_checkpoint, validate_run
from app.services.run_manager import RunManager

SCORES = json.dumps({'correctness': 5, 'relevance': 5, 'clarity': 5,
                     'instruction_following': 5, 'completeness': 5,
                     'reasoning': 'Correct.', 'limitations': []})


def settings():
    return Settings(_env_file=None, gemini_api_key='unit-test-key', gemini_judge_model='judge')


def cases():
    return validate_cases([{'id': str(i), 'question': f'Q{i}', 'reference': 'R'} for i in range(2)])


def options(**kwargs):
    return RunOptions(models=['model'], judge_model='judge', rubric='quality-v2',
                      request_interval=0, max_retries=0, **kwargs)


class Fake:
    def __init__(self, config):
        self.settings = config
    def generate(self, prompt, judge=False, schema=None):
        return (SCORES if judge else 'Answer'), None


def test_input_parser_rejects_invalid_json_and_reports_rows():
    for payload in [b'{', b'{}', b'[]', b'\xff', b' ' * 2_000_001]:
        with pytest.raises(ValueError):
            parse_dataset(payload)
    with pytest.raises(ValueError, match='Row 2, question'):
        parse_dataset(b'[{"id":"a","question":"Q","reference":"R"},{"id":"b","question":"","reference":"R"}]')
    with pytest.raises(ValueError, match='unique'):
        parse_dataset(b'[{"id":"a","question":"Q","reference":"R"},{"id":"a","question":"Q","reference":"R"}]')


def test_budget_and_missing_key_are_checked_before_run():
    cfg, budget, reused = validate_run(cases(), options(), settings())
    assert budget == 4 and reused == 0
    with pytest.raises(ValueError, match='GEMINI_API_KEY'):
        validate_run(cases(), options(), Settings(_env_file=None, gemini_api_key=''))
    bad = RunOptions(models=['same', 'same'], judge_model='judge')
    with pytest.raises(ValueError, match='unique'):
        validate_run(cases(), bad, settings())


def test_manager_finishes_and_download_files_are_valid(tmp_path):
    manager = RunManager(tmp_path, Fake)
    job = manager.start(cases(), options(), settings())
    assert manager.wait(job)
    snap = manager.snapshot(job)
    assert snap['state'] == 'finished' and not snap['active']
    assert snap['report']['summary'][0]['evaluated'] == 2
    assert snap['report']['launch_source'] == 'dashboard'
    assert Path(snap['json_path']).is_file() and Path(snap['csv_path']).is_file()
    assert 'unit-test-key' not in json.dumps(snap)


def test_duplicate_start_blocked_and_cancel_waits_for_inflight_request(tmp_path):
    entered, release = Event(), Event()
    calls = []
    class Blocking(Fake):
        def generate(self, prompt, **kwargs):
            calls.append(kwargs.get('judge', False))
            entered.set()
            assert release.wait(3)
            return super().generate(prompt, **kwargs)
    manager = RunManager(tmp_path, Blocking)
    job = manager.start(cases(), options(), settings())
    assert entered.wait(3)
    with pytest.raises(ValueError, match='already active'):
        manager.start(cases(), options(), settings())
    assert manager.cancel(job)
    assert manager.snapshot(job)['state'] == 'cancelling'
    release.set()
    assert manager.wait(job)
    snap = manager.snapshot(job)
    assert snap['state'] == 'cancelled'
    assert calls == [False]
    assert snap['report']['run_state'] == 'interrupted'
    assert snap['report']['results'][0]['evaluation_attempts'] == 0


def test_pre_cancelled_engine_calls_no_provider():
    event = Event()
    event.set()
    snapshots = []
    def never(_):
        pytest.fail('Cancellation before construction must prevent provider access')
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(cases(), ['model'], settings(), never, cancel_event=event, on_checkpoint=snapshots.append)
    assert snapshots[-1]['run_state'] == 'interrupted'
    assert all(row['status'] == 'pending' for row in snapshots[-1]['results'])


def test_resume_reconstructs_uploaded_cases_and_preserves_pacing(tmp_path):
    manager = RunManager(tmp_path, Fake)
    opts = options(model_intervals={'model': 0})
    job = manager.start(cases(), opts, settings())
    assert manager.wait(job)
    original = manager.snapshot(job)
    restored, saved_options, prior = read_checkpoint(Path(original['json_path']), tmp_path)
    assert saved_options.model_intervals == {'model': 0}
    _, budget, reused = validate_run(restored, saved_options, settings(), prior)
    assert budget == 0 and reused == 2
    resumed = manager.start(restored, saved_options, settings(), prior)
    assert manager.wait(resumed)
    assert manager.snapshot(resumed)['report']['summary'][0]['provider_attempts'] == 0
    assert manager.snapshot(resumed)['json_path'] != original['json_path']


def test_checkpoint_outside_results_folder_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='results folder'):
        read_checkpoint(tmp_path.parent / 'eval-outside.json', tmp_path)


def test_worker_failure_never_exposes_raw_error_or_key(tmp_path):
    class Broken(Fake):
        def generate(self, *args, **kwargs):
            raise RuntimeError('private provider URL and unit-test-key')
    manager = RunManager(tmp_path, Broken)
    job = manager.start(cases(), options(), settings())
    assert manager.wait(job)
    snap = manager.snapshot(job)
    assert snap['state'] == 'failed' and not snap['active']
    assert 'unit-test-key' not in json.dumps(snap)
    assert 'private provider URL' not in json.dumps(snap)


def render(tmp_path, monkeypatch, manager):
    import dashboard.run_controls as ui
    monkeypatch.setattr(ui, 'get_manager', lambda _: manager)
    monkeypatch.setattr(ui, 'Settings', settings)
    return AppTest.from_string(f'from pathlib import Path\nfrom dashboard.run_controls import render_run_controls\nrender_run_controls(Path({str(tmp_path)!r}))').run(timeout=15)


def test_dashboard_open_does_not_start_work_and_budget_is_visible(tmp_path, monkeypatch):
    calls = []
    class Count(Fake):
        def generate(self, *a, **kw):
            calls.append(1)
            return super().generate(*a, **kw)
    manager = RunManager(tmp_path, Count)
    at = render(tmp_path, monkeypatch, manager)
    assert not at.exception
    assert manager.snapshot() is None and calls == []
    assert any('Provider-call budget' in item.value for item in at.info)
    assert at.button[0].label == 'Start evaluation (uses Gemini)'


def test_dashboard_can_start_fake_run_and_show_downloads(tmp_path, monkeypatch):
    manager = RunManager(tmp_path, Fake)
    at = render(tmp_path, monkeypatch, manager)
    # Zero pacing in this offline test; production retains the visible default.
    at.number_input[1].set_value(0.0).run()
    at.button[0].click().run()
    job = manager.latest
    assert job is not None and manager.wait(job)
    at.run()
    assert not at.exception
    assert manager.snapshot(job)['state'] == 'finished'
    assert len(at.get('download_button')) == 2


def test_cancellation_interrupts_retry_wait_without_new_calls(tmp_path):
    from app.services.llm_client import ProviderError
    entered, release = Event(), Event()
    calls = []
    class Flaky(Fake):
        def generate(self, *args, **kwargs):
            calls.append(1)
            entered.set()
            assert release.wait(3)
            raise ProviderError(reason='provider_unavailable')
    manager = RunManager(tmp_path, Flaky)
    opts = options().model_copy(update={'max_retries': 2})
    job = manager.start(cases(), opts, settings())
    assert entered.wait(3)
    manager.cancel(job)
    release.set()
    assert manager.wait(job)
    assert calls == [1]
    assert manager.snapshot(job)['state'] == 'cancelled'


def test_cancellation_interrupts_pacing_before_judging():
    calls, snapshots = [], []
    class CancelOnWait:
        stopped = False
        def is_set(self):
            return self.stopped
        def wait(self, seconds):
            assert seconds > 0
            self.stopped = True
            return True
    class Count(Fake):
        def generate(self, *args, **kwargs):
            calls.append(1)
            return super().generate(*args, **kwargs)
    cfg = settings().model_copy(update={'gemini_judge_model': 'same'})
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(cases(), ['same'], cfg, Count, request_interval=60,
                      cancel_event=CancelOnWait(), on_checkpoint=snapshots.append)
    assert calls == [1]
    assert snapshots[-1]['run_state'] == 'interrupted'
    assert snapshots[-1]['results'][0]['evaluation_attempts'] == 0


def test_upload_selection_requires_file_before_start(tmp_path, monkeypatch):
    manager = RunManager(tmp_path, Fake)
    at = render(tmp_path, monkeypatch, manager)
    at.selectbox[0].set_value('Upload JSON').run()
    assert not at.exception
    assert at.button[0].disabled
    assert manager.snapshot() is None


def test_dashboard_resumes_saved_completed_run_without_provider_calls(tmp_path, monkeypatch):
    manager = RunManager(tmp_path, Fake)
    job = manager.start(cases(), options(), settings())
    assert manager.wait(job)
    at = render(tmp_path, monkeypatch, manager)
    at.radio[0].set_value('Resume saved run').run()
    assert not at.exception
    assert any('up to 0 stage attempts' in item.value for item in at.info)
    at.button[0].click().run()
    resumed = manager.latest
    assert manager.wait(resumed)
    assert manager.snapshot(resumed)['report']['resumed_rows'] == 2
    assert manager.snapshot(resumed)['report']['summary'][0]['provider_attempts'] == 0


def test_running_checkpoint_requires_stopped_original_acknowledgement(tmp_path, monkeypatch):
    from app.services.report_store import ReportWriter
    report = run_benchmark(cases(), ['model'], settings(), Fake)
    report['run_state'] = 'running'
    ReportWriter(tmp_path).checkpoint(report)
    manager = RunManager(tmp_path, Fake)
    at = render(tmp_path, monkeypatch, manager)
    at.radio[0].set_value('Resume saved run').run()
    assert not at.exception
    assert at.button[0].disabled
    at.checkbox[0].set_value(True).run()
    assert not at.button[0].disabled



@pytest.mark.parametrize('state', ['finished', 'cancelled', 'failed'])
def test_late_cancel_never_overwrites_terminal_state(tmp_path, state):
    from app.services.run_manager import Job
    manager = RunManager(tmp_path, Fake)
    job = Job('terminal', state=state)
    manager.jobs[job.id] = job
    # Simulate the narrow window before the worker clears the active pointer.
    manager.active = job.id
    assert manager.cancel(job.id) is False
    assert job.state == state
    assert not job.cancel.is_set()
