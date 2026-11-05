import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings
from app.services.benchmark import run_benchmark, validate_cases
from app.services.llm_client import ProviderError
from app.services.provider_failures import classify_failure
from app.services.report_store import ReportWriter, atomic_text


SCORES = '{"correctness":5,"relevance":5,"clarity":5,"reasoning":"Matches."}'


def cases(count=3):
    return validate_cases([{'id': str(i), 'question': f'Q{i}', 'reference': 'R'} for i in range(count)])


def config():
    return Settings(_env_file=None, gemini_judge_model='judge')


class Healthy:
    def __init__(self, settings):
        self.settings = settings
    def generate(self, prompt, judge=False, schema=None):
        return (SCORES if judge else 'Answer'), SimpleNamespace(total_token_count=3)


def no_wait(monkeypatch):
    import app.services.benchmark as module
    clock, waits = [0.0], []
    monkeypatch.setattr(module, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(module, 'uniform', lambda a, b: 0)
    def pause(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(module, 'sleep', pause)
    return clock, waits


def test_checkpoint_is_written_before_calls_and_after_each_case(tmp_path):
    writer = ReportWriter(tmp_path)
    seen = []
    class Check(Healthy):
        def generate(self, prompt, **kwargs):
            saved = json.loads(writer.json_path.read_text())
            seen.append(saved['processed_cases'])
            return super().generate(prompt, **kwargs)
    report = run_benchmark(cases(2), ['model'], config(), Check, on_checkpoint=writer.checkpoint)
    saved = json.loads(writer.json_path.read_text())
    assert seen == [0, 0, 1, 1]
    assert saved == report
    assert saved['run_state'] == 'finished'
    assert saved['summary'][0]['provider_attempts'] == 4


def test_interrupt_keeps_valid_checkpoint_and_resumes_only_successes(tmp_path):
    writer = ReportWriter(tmp_path)
    events = []
    def interrupt(row):
        events.append(row)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(cases(), ['model'], config(), Healthy,
                      on_checkpoint=writer.checkpoint, on_result=interrupt)
    saved = json.loads(writer.json_path.read_text())
    assert saved['run_state'] == 'interrupted'
    assert [r['status'] for r in saved['results']] == ['ok', 'pending', 'pending']
    resumed = run_benchmark(cases(), ['model'], config(), Healthy, prior_report=saved)
    assert resumed['resumed_rows'] == 1
    assert resumed['summary'][0]['provider_attempts'] == 4
    assert resumed['summary'][0]['evaluated'] == 3


def test_initial_checkpoint_failure_prevents_provider_construction():
    def fail(_):
        raise OSError('disk unavailable')
    def factory(_):
        pytest.fail('No provider should be constructed before a durable checkpoint')
    with pytest.raises(OSError):
        run_benchmark(cases(), ['model'], config(), factory, on_checkpoint=fail)


def test_atomic_write_failure_preserves_old_file_and_cleans_temp(tmp_path, monkeypatch):
    import app.services.report_store as module
    path = tmp_path / 'report.json'
    path.write_text('{"old":true}')
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises(OSError):
        atomic_text(path, '{"new":true}')
    assert json.loads(path.read_text()) == {'old': True}
    assert list(tmp_path.iterdir()) == [path]


def test_transient_failure_recovers_with_bounded_retry_and_honest_latency(monkeypatch):
    clock, waits = no_wait(monkeypatch)
    calls = []
    class Flaky(Healthy):
        def generate(self, prompt, **kwargs):
            calls.append(1)
            clock[0] += .5
            if len(calls) < 3:
                raise ProviderError(reason='provider_unavailable')
            return super().generate(prompt, **kwargs)
    report = run_benchmark(cases(1), ['model'], config(), Flaky, max_retries=2)
    row = report['results'][0]
    assert row['status'] == 'ok'
    assert row['generation_attempts'] == 3 and row['evaluation_attempts'] == 1
    assert row['retries'] == 2
    assert row['generation_ms'] == 1500
    assert waits == [1, 2]


def test_retry_exhaustion_counts_one_failure_then_skips(monkeypatch):
    no_wait(monkeypatch)
    calls = []
    class Down(Healthy):
        def generate(self, *args, **kwargs):
            calls.append(1)
            raise ProviderError(reason='timeout')
    report = run_benchmark(cases(), ['model'], config(), Down, max_retries=2)
    assert len(calls) == 3
    assert [r['status'] for r in report['results']] == ['generation_error', 'skipped', 'skipped']
    assert report['summary'][0]['provider_attempts'] == 3
    assert report['summary'][0]['failures'] == 1
    assert report['summary'][0]['skipped'] == 2


@pytest.mark.parametrize('reason,retry_after', [('daily_quota', 1), ('rate_limited', None), ('rate_limited', 100), ('provider_unavailable', 100), ('model_unavailable', None)])
def test_nonretryable_limits_do_not_sleep_or_retry(monkeypatch, reason, retry_after):
    _, waits = no_wait(monkeypatch)
    class Limited(Healthy):
        def generate(self, *args, **kwargs):
            raise ProviderError(reason=reason, retry_after=retry_after)
    report = run_benchmark(cases(), ['model'], config(), Limited, max_retries=3)
    assert waits == []
    assert report['summary'][0]['provider_attempts'] == 1
    assert report['summary'][0]['skipped'] == 2


def test_short_rate_limit_respects_retry_after(monkeypatch):
    _, waits = no_wait(monkeypatch)
    calls = []
    class Limited(Healthy):
        def generate(self, prompt, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ProviderError(reason='rate_limited', retry_after=5)
            return super().generate(prompt, **kwargs)
    report = run_benchmark(cases(1), ['model'], config(), Limited, max_retries=1)
    assert report['results'][0]['status'] == 'ok'
    assert waits == [5]


def test_daily_quota_classification_uses_structured_metadata():
    error = SimpleNamespace(code=429, details={'error': {'details': [
        {'violations': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]},
        {'retryDelay': '41845s'}]}}, response=None)
    assert classify_failure(error) == ('daily_quota', 41845)
    assert classify_failure(SimpleNamespace(code=429, details={}, response=None)) == ('rate_limited', None)


@pytest.mark.parametrize('error,reason', [
    (httpx.ReadTimeout('private URL or key'), 'timeout'),
    (httpx.ConnectError('private URL or key'), 'network_error'),
    (SimpleNamespace(code=503, details={}, response=None), 'provider_unavailable'),
])
def test_safe_provider_error_classification(error, reason):
    assert classify_failure(error)[0] == reason


def test_judge_retry_exhaustion_skips_new_generation(monkeypatch):
    no_wait(monkeypatch)
    calls = []
    class JudgeDown(Healthy):
        def generate(self, prompt, judge=False, schema=None):
            calls.append(judge)
            if judge:
                raise ProviderError(reason='provider_unavailable')
            return super().generate(prompt, judge=judge, schema=schema)
    report = run_benchmark(cases(), ['model', 'other'], config(), JudgeDown, max_retries=2)
    assert calls == [False, True, True, True]
    assert report['summary'][0]['failures'] == 1
    assert sum(s['skipped'] for s in report['summary']) == 5
    assert report['results'][0]['evaluation_attempts'] == 3


def cli_setup(tmp_path, monkeypatch):
    import experiments.run_eval as cli
    dataset = tmp_path / 'cases.json'
    dataset.write_text(json.dumps([c.model_dump() for c in cases()]))
    output = tmp_path / 'output'
    monkeypatch.setattr('sys.argv', ['run_eval', '--dataset', str(dataset), '--models', 'model',
                                    '--output-dir', str(output), '--request-interval', '0', '--max-retries', '0'])
    monkeypatch.setattr(cli, 'Settings', lambda: Settings(_env_file=None, gemini_api_key='unit-test-key', gemini_judge_model='judge'))
    return cli, output


def test_cli_interrupt_returns_130_and_keeps_resumable_json(tmp_path, monkeypatch, capsys):
    import app.services.benchmark as engine
    cli, output = cli_setup(tmp_path, monkeypatch)
    calls = []
    class Interrupted(Healthy):
        def generate(self, prompt, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt()
            return super().generate(prompt, **kwargs)
    original = engine.run_benchmark
    monkeypatch.setattr(cli, 'run_benchmark', lambda *a, **kw: original(*a, provider_factory=Interrupted, **kw))
    assert cli.main() == 130
    reports = list(output.glob('eval-*.json'))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text())
    assert saved['run_state'] == 'interrupted'
    assert [r['status'] for r in saved['results']] == ['ok', 'interrupted', 'pending']
    assert saved['results'][1]['generation_attempts'] == 1
    assert not list(output.glob('*.csv'))
    assert 'Resume' in capsys.readouterr().out


def test_checkpoint_failure_stops_new_calls_and_preserves_previous_case(tmp_path, monkeypatch):
    import app.services.benchmark as engine
    cli, output = cli_setup(tmp_path, monkeypatch)
    calls, writes = [], []
    class Count(Healthy):
        def generate(self, prompt, **kwargs):
            calls.append(1)
            return super().generate(prompt, **kwargs)
    original_run = engine.run_benchmark
    original_write = ReportWriter.checkpoint
    def write(self, report):
        writes.append(1)
        if len(writes) == 3:
            raise OSError('disk full')
        original_write(self, report)
    monkeypatch.setattr(ReportWriter, 'checkpoint', write)
    monkeypatch.setattr(cli, 'run_benchmark', lambda *a, **kw: original_run(*a, provider_factory=Count, **kw))
    assert cli.main() == 1
    assert len(calls) == 4
    saved = json.loads(next(output.glob('eval-*.json')).read_text())
    assert [r['status'] for r in saved['results']] == ['ok', 'pending', 'pending']


def test_dashboard_displays_pending_skipped_and_attempted_failures(tmp_path):
    from streamlit.testing.v1 import AppTest
    writer = ReportWriter(tmp_path)
    class Down(Healthy):
        def generate(self, *args, **kwargs):
            raise ProviderError(reason='daily_quota')
    run_benchmark(cases(), ['model'], config(), Down, on_checkpoint=writer.checkpoint)
    at = AppTest.from_string(f'from dashboard.app import render_dashboard\nrender_dashboard({str(tmp_path)!r})').run(timeout=15)
    assert not at.exception
    assert at.metric[3].value == '1'
    assert any('Skipped: 2' in item.value for item in at.caption)
    assert any(item.value == 'Why cases stopped' for item in at.subheader)
    assert at.dataframe[0].value.iloc[0]['Skipped'] == 2


def test_dashboard_accepts_initial_running_checkpoint(tmp_path):
    from streamlit.testing.v1 import AppTest
    writer = ReportWriter(tmp_path)
    class Stop(Healthy):
        def __init__(self, settings):
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(cases(), ['model'], config(), Stop, on_checkpoint=writer.checkpoint)
    at = AppTest.from_string(f'from dashboard.app import render_dashboard\nrender_dashboard({str(tmp_path)!r})').run(timeout=15)
    assert not at.exception
    assert at.metric[0].value == '0 / 3'
    assert any('Pending: 3' in item.value for item in at.caption)



def test_http_date_retry_after_is_honored():
    from datetime import datetime, timezone
    error = SimpleNamespace(code=503, details={}, response=SimpleNamespace(headers=httpx.Headers({
        'Retry-After': 'Thu, 05 Nov 2026 12:10:00 GMT'})))
    reason, delay = classify_failure(error, now=datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc))
    assert reason == 'provider_unavailable'
    assert delay == 600



def test_cancel_before_initial_checkpoint_does_not_claim_saved_progress(tmp_path, monkeypatch, capsys):
    cli, output = cli_setup(tmp_path, monkeypatch)
    def cancel(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(ReportWriter, 'checkpoint', cancel)
    assert cli.main() == 130
    assert not list(output.glob('*.json'))
    assert 'before a progress file' in capsys.readouterr().out


def test_cancel_during_csv_export_keeps_completed_json(tmp_path, monkeypatch, capsys):
    import app.services.benchmark as engine
    cli, output = cli_setup(tmp_path, monkeypatch)
    original = engine.run_benchmark
    monkeypatch.setattr(cli, 'run_benchmark', lambda *a, **kw: original(*a, provider_factory=Healthy, **kw))
    def cancel(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(ReportWriter, 'finish', cancel)
    assert cli.main() == 130
    saved = json.loads(next(output.glob('*.json')).read_text())
    assert saved['run_state'] == 'finished'
    assert saved['summary'][0]['evaluated'] == 3
    assert 'completed JSON checkpoint' in capsys.readouterr().out


def test_legacy_dashboard_does_not_claim_known_attempt_accounting(tmp_path):
    from streamlit.testing.v1 import AppTest
    report = run_benchmark(cases(1), ['model'], config(), Healthy)
    report.pop('accounting_version')
    for row in report['results']:
        row.pop('generation_attempts')
        row.pop('evaluation_attempts')
        row.pop('retries')
    (tmp_path / 'eval-legacy.json').write_text(json.dumps(report))
    at = AppTest.from_string(f'from dashboard.app import render_dashboard\nrender_dashboard({str(tmp_path)!r})').run(timeout=15)
    assert not at.exception
    assert at.metric[3].label == 'Failures / legacy errors'
    assert any('Historical reports' in item.value for item in at.info)
