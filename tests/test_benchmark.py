import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.benchmark import run_benchmark, validate_cases
from app.services.llm_client import ProviderError
from experiments.run_eval import save_report


def test_dataset_rejects_duplicate_and_empty_cases():
    case = {'id': 'one', 'question': 'Q', 'reference': 'R'}
    for data in [[], {}, [case, case], [{'id': 'one', 'question': ' ', 'reference': 'R'}]]:
        with pytest.raises(ValueError):
            validate_cases(data)


def test_benchmark_model_separation_partial_failure_and_exports(tmp_path):
    class Provider:
        def __init__(self, settings):
            self.settings = settings

        def generate(self, text, judge=False, schema=None):
            if self.settings.gemini_model == 'broken':
                raise ProviderError('private provider diagnostic')
            if judge:
                return '{"correctness":5,"relevance":4,"clarity":3,"reasoning":"Good."}', None
            return 'Answer', SimpleNamespace(total_token_count=12)

    settings = Settings(_env_file=None, gemini_api_key='fake-secret', gemini_judge_model='judge')
    cases = validate_cases([{'id': 'one', 'question': 'Q', 'reference': 'R'}])
    report = run_benchmark(cases, ['good', 'broken'], settings, Provider)
    assert report['summary'][0]['mean_score'] == 4
    assert report['summary'][0]['generation_total_tokens'] == 12
    assert report['summary'][1]['mean_score'] is None
    assert report['summary'][1]['failures'] == 1
    assert report['results'][1]['status'] == 'generation_error'
    paths = save_report(report, tmp_path)
    saved = paths[0].read_text()
    assert json.loads(saved) == report
    assert 'fake-secret' not in saved
    assert 'private provider diagnostic' not in saved
    assert 'correctness' in paths[1].read_text()


def test_judge_failure_retains_generated_answer():
    class Provider:
        def __init__(self, settings):
            self.settings = settings
        def generate(self, text, judge=False, schema=None):
            if judge:
                raise ProviderError()
            return 'Answer', None
    cases = validate_cases([{'id': 'one', 'question': 'Q', 'reference': 'R'}])
    report = run_benchmark(cases, ['model'], Settings(_env_file=None), Provider)
    assert report['results'][0]['answer'] == 'Answer'
    assert report['results'][0]['status'] == 'evaluation_error'
    assert report['summary'][0]['mean_score'] is None
