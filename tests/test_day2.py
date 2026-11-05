import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.core.config import Settings
from app.services.benchmark import run_benchmark, validate_cases
from app.services.cache_benchmark import run_cache_benchmark, validate_pairs
from dashboard.cache_view import load_cache_reports
from experiments.run_cache_eval import save_cache_report


ROOT = Path(__file__).resolve().parents[1]


def test_day2_datasets_are_balanced_and_valid():
    cases = validate_cases(json.loads((ROOT / 'experiments/day2_cases.json').read_text()))
    assert len(cases) == 24
    assert {c.category for c in cases} == {'reasoning', 'factual', 'coding', 'instruction_following'}
    pairs = validate_pairs(json.loads((ROOT / 'experiments/cache_pairs.json').read_text()))
    assert sum(p.expected_hit for p in pairs) == 12
    assert sum(not p.expected_hit for p in pairs) == 12


def test_cache_confusion_matrix_and_no_hit_precision(tmp_path):
    pairs = validate_pairs([
        {'id': 'tp', 'category': 'same', 'source': 'a', 'query': 'a', 'expected_hit': True},
        {'id': 'fp', 'category': 'changed', 'source': 'a', 'query': 'b', 'expected_hit': False},
        {'id': 'fn', 'category': 'same', 'source': 'a', 'query': 'c', 'expected_hit': True},
        {'id': 'tn', 'category': 'changed', 'source': 'a', 'query': 'c', 'expected_hit': False},
    ])
    vectors = {'a': [1, 0], 'b': [.99, .01], 'c': [0, 1]}
    report = run_cache_benchmark(pairs, vectors.get, 'fake', [.95])
    summary = report['summary'][0]
    assert [summary[k] for k in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']] == [1, 1, 1, 1]
    assert summary['precision'] == summary['recall'] == summary['accuracy'] == .5
    paths = save_cache_report(report, tmp_path)
    reports, skipped = load_cache_reports(tmp_path)
    assert len(reports) == 1 and not skipped
    assert paths[1].exists()
    no_hits = run_cache_benchmark([pairs[-1]], vectors.get, 'fake', [1.0])
    assert no_hits['summary'][0]['precision'] is None
    assert no_hits['summary'][0]['recall'] is None


def test_embedding_failure_excluded_from_rates():
    pairs = validate_pairs([{'id': 'a', 'category': 'same', 'source': 'a', 'query': 'b', 'expected_hit': True}])
    def broken(_):
        raise RuntimeError('private diagnostic')
    report = run_cache_benchmark(pairs, broken, 'fake', [.95])
    assert report['summary'][0]['errors'] == 1
    assert report['summary'][0]['accuracy'] is None
    assert 'private diagnostic' not in json.dumps(report)


@pytest.mark.parametrize('thresholds', [[], [float('nan')], [1.1], [.9, .9]])
def test_threshold_validation(thresholds):
    with pytest.raises(ValueError):
        run_cache_benchmark([], lambda _: [1, 0], 'fake', thresholds)


def test_benchmark_metadata_and_progress():
    configs, events = [], []
    class Fake:
        def __init__(self, config):
            self.settings = config
            configs.append(config)
        def generate(self, text, judge=False, schema=None):
            return ('{"correctness":5,"relevance":5,"clarity":5,"reasoning":"Correct."}' if judge else '4'), None
    cases = validate_cases([{'id': 'sum', 'category': 'reasoning', 'question': '2+2?', 'reference': '4'}])
    report = run_benchmark(cases, ['a', 'b'], Settings(_env_file=None), Fake, events.append)
    assert len(events) == 2
    assert all(c.generation_temperature == 0 for c in configs)
    assert report['results'][0]['category'] == 'reasoning'
    assert len(report['dataset_sha256']) == 64
    changed = run_benchmark(validate_cases([{'id': 'sum', 'question': '3+3?', 'reference': '6'}]), ['a'], Settings(_env_file=None), Fake)
    assert changed['dataset_sha256'] != report['dataset_sha256']


def test_cache_dashboard_with_saved_report(tmp_path):
    pairs = validate_pairs([{'id': 'a', 'category': 'same', 'source': 'a', 'query': 'a', 'expected_hit': True}])
    report = run_cache_benchmark(pairs, lambda _: [1, 0], 'fake', [.95])
    save_cache_report(report, tmp_path)
    at = AppTest.from_string(f'from dashboard.app import render_dashboard\nrender_dashboard({str(tmp_path)!r})').run(timeout=15)
    at.sidebar.radio[0].set_value('Cache accuracy').run()
    assert not at.exception
    assert at.subheader[0].value == 'Cache accuracy'
    assert at.dataframe[0].value.iloc[0]['precision'] == 1.0



@pytest.mark.parametrize('reason', ['rate_limited', 'model_unavailable', 'provider_unavailable'])
def test_rate_limit_stops_calls_for_the_affected_model(reason):
    from app.services.llm_client import ProviderError
    calls = []
    class Limited:
        def __init__(self, settings):
            self.settings = settings
        def generate(self, *args, **kwargs):
            calls.append(1)
            raise ProviderError(reason=reason)
    cases = validate_cases([{'id': str(i), 'question': 'Q', 'reference': 'R'} for i in range(3)])
    report = run_benchmark(cases, ['limited'], Settings(_env_file=None), Limited)
    assert len(calls) == 1
    assert report['summary'][0]['failures'] == 1
    assert report['summary'][0]['skipped'] == 2
    assert all(r['error_reason'] == reason for r in report['results'])


def test_pacing_does_not_inflate_generation_latency(monkeypatch):
    import app.services.benchmark as module
    clock = [0.0]
    waits = []
    monkeypatch.setattr(module, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module, 'perf_counter', lambda: clock[0])
    def pause(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(module, 'sleep', pause)
    class Fake:
        def __init__(self, settings):
            self.settings = settings
        def generate(self, prompt, judge=False, schema=None):
            clock[0] += 2
            return ('{"correctness":5,"relevance":5,"clarity":5,"reasoning":"Correct."}' if judge else '4'), None
    cases = validate_cases([{'id': str(i), 'question': 'Q', 'reference': 'R'} for i in range(2)])
    report = run_benchmark(cases, ['same'], Settings(_env_file=None, gemini_judge_model='same'), Fake, request_interval=10)
    assert waits == [8, 8, 8]
    assert all(r['generation_ms'] == 2000 and r['evaluation_ms'] == 2000 for r in report['results'])


def test_resume_reuses_successful_rows_without_provider_calls():
    calls = []
    class Fake:
        def __init__(self, settings):
            self.settings = settings
        def generate(self, prompt, judge=False, schema=None):
            calls.append(1)
            return ('{"correctness":5,"relevance":5,"clarity":5,"reasoning":"Correct."}' if judge else '4'), None
    cases = validate_cases([{'id': 'sum', 'question': '2+2?', 'reference': '4'}])
    settings = Settings(_env_file=None)
    first = run_benchmark(cases, ['a'], settings, Fake)
    assert len(calls) == 2
    resumed = run_benchmark(cases, ['a'], settings, Fake, prior_report=first)
    assert len(calls) == 2
    assert resumed['resumed_rows'] == 1
    assert resumed['results'][0]['reused'] is True
    assert resumed['summary'][0]['mean_score'] == 5


@pytest.mark.parametrize('field,value', [('dataset_sha256', 'wrong'), ('judge_model', 'wrong'), ('generation_temperature', .7)])
def test_resume_rejects_incompatible_experiments_before_calls(field, value):
    cases = validate_cases([{'id': 'sum', 'question': '2+2?', 'reference': '4'}])
    report = {'dataset_sha256': 'wrong', 'judge_model': 'gemini-2.5-flash', 'generation_temperature': 0, 'cache_enabled': False, 'results': []}
    from app.services.benchmark import dataset_fingerprint
    report['dataset_sha256'] = dataset_fingerprint(cases)
    report[field] = value
    def no_calls(_):
        pytest.fail('Provider must not be constructed for an incompatible resume report')
    with pytest.raises(ValueError):
        run_benchmark(cases, ['a'], Settings(_env_file=None), no_calls, prior_report=report)



def test_unavailable_judge_stops_further_generation():
    from app.services.llm_client import ProviderError
    calls = []
    class Fake:
        def __init__(self, settings):
            self.settings = settings
        def generate(self, prompt, judge=False, schema=None):
            calls.append(judge)
            if judge:
                raise ProviderError(reason='provider_unavailable')
            return 'Answer', None
    cases = validate_cases([{'id': str(i), 'question': 'Q', 'reference': 'R'} for i in range(2)])
    report = run_benchmark(cases, ['a', 'b'], Settings(_env_file=None), Fake)
    assert calls == [False, True]
    assert report['results'][1]['error_reason'] == 'judge_provider_unavailable'
    assert sum(s['failures'] for s in report['summary']) == 1
    assert sum(s['skipped'] for s in report['summary']) == 3


def test_resume_retries_only_failed_cases():
    from app.services.llm_client import ProviderError
    calls = []
    class Fake:
        fail = True
        def __init__(self, settings):
            self.settings = settings
        def generate(self, prompt, judge=False, schema=None):
            calls.append(1)
            if not judge and prompt == 'bad' and self.fail:
                raise ProviderError()
            return ('{"correctness":5,"relevance":5,"clarity":5,"reasoning":"Correct."}' if judge else '4'), None
    cases = validate_cases([{'id': name, 'question': name, 'reference': '4'} for name in ['good', 'bad']])
    settings = Settings(_env_file=None)
    first = run_benchmark(cases, ['a'], settings, Fake)
    assert len(calls) == 3
    Fake.fail = False
    resumed = run_benchmark(cases, ['a'], settings, Fake, prior_report=first)
    assert len(calls) == 5
    assert resumed['resumed_rows'] == 1
    assert resumed['summary'][0]['evaluated'] == 2



@pytest.mark.parametrize('usage', [None, {}])
def test_resume_rejects_incomplete_usage_before_calls(usage):
    from app.services.benchmark import dataset_fingerprint
    cases = validate_cases([{'id': 'sum', 'question': '2+2?', 'reference': '4'}])
    settings = Settings(_env_file=None)
    row = {'case_id': 'sum', 'category': 'general', 'question': '2+2?', 'reference': '4',
           'model': 'a', 'judge_model': settings.gemini_judge_model, 'status': 'ok',
           'answer': '4', 'generation_ms': 10, 'evaluation_ms': 10, 'usage': usage,
           'evaluation': {'correctness': 5, 'relevance': 5, 'clarity': 5, 'overall': 5, 'reasoning': 'Correct.'}}
    report = {'dataset_sha256': dataset_fingerprint(cases), 'judge_model': settings.gemini_judge_model,
              'generation_temperature': 0, 'cache_enabled': False, 'results': [row]}
    def no_calls(_):
        pytest.fail('Invalid resume usage must be rejected before provider calls')
    with pytest.raises(ValueError, match='usage'):
        run_benchmark(cases, ['a'], settings, no_calls, prior_report=report)


def test_unknown_model_interval_rejected_before_calls(tmp_path, monkeypatch, capsys):
    import experiments.run_eval as cli
    dataset = tmp_path / 'cases.json'
    dataset.write_text('[{"id":"a","question":"Q","reference":"R"}]')
    monkeypatch.setattr('sys.argv', ['run_eval', '--dataset', str(dataset), '--models', 'a', '--model-interval', 'typo=13'])
    monkeypatch.setattr(cli, 'Settings', lambda: Settings(_env_file=None, gemini_api_key='unit-test-key'))
    monkeypatch.setattr(cli, 'run_benchmark', lambda *a, **k: pytest.fail('No calls expected'))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert 'selected generation model' in capsys.readouterr().err
