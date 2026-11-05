import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.cache import SemanticCache
from app.services.cache_benchmark import run_cache_benchmark, validate_pairs
from app.services.cache_policy import allows_semantic_reuse


@pytest.mark.parametrize('source,query', [
    ('Which Python types are mutable?', 'Which Python types are not mutable?'),
    ('Sort in ascending order.', 'Sort in descending order.'),
    ('Convert Celsius to Fahrenheit.', 'Convert Fahrenheit to Celsius.'),
    ('Select ages greater than 18.', 'Select ages less than 18.'),
    ('What is 17 * 6?', 'What is 17 * 7?'),
    ('What is 17 * 6?', 'What is 17 / 6?'),
    ('What is the capital of France?', 'What is the capital of Germany?'),
    ('Does Alice follow Bob?', 'Does Bob follow Alice?'),
    ('What is x >= 10?', 'What is x > 10?'),
    ('Return "Hello".', 'Return "hello".'),
    ('What does foo return?', 'What does Foo return?'),
    ('Is 2 ≤ 3?', 'Is 2 ≥ 3?'),
    ('What does $10 buy?', 'What does £10 buy?'),
    ('How do I reset my password?', 'Can I reset my password?'),
    ('What is type([1, 2])?', 'What is type((1, 2))?'),
    ('What does foo1 return?', 'What does foo 1 return?'),
    ('What is Mexico City?', 'What is Mexico?'),
    ('Explain: if ready:\n    run()\nstop()', 'Explain: if ready:\n    run()\n    stop()'),
])
def test_guard_blocks_meaning_changes_even_with_identical_embeddings(source, query):
    assert not allows_semantic_reuse(source, query)
    cache = SemanticCache(lambda _: [1, 0], match_mode='guarded')
    cache.put(source, 'model', 'answer')
    assert cache.get(query, 'model') is None


def test_guard_keeps_known_low_risk_paraphrase():
    cache = SemanticCache(lambda _: [1, 0], match_mode='guarded')
    cache.put('What is the capital of France?', 'model', 'Paris')
    assert cache.get('What is the capital city of France?', 'model') == ('Paris', 1.0)


def test_guard_filters_before_nearest_neighbor_selection():
    vectors = {'What is the capital of France?': [.99, .1],
               'What is the capital of Germany?': [1, 0],
               'What is the capital city of France?': [1, 0]}
    cache = SemanticCache(vectors.get, match_mode='guarded')
    cache.put('What is the capital of France?', 'model', 'Paris')
    cache.put('What is the capital of Germany?', 'model', 'Berlin')
    assert cache.get('What is the capital city of France?', 'model')[0] == 'Paris'


def test_exact_default_works_without_embeddings():
    def broken(_):
        pytest.fail('Exact cache must not load embeddings')
    cache = SemanticCache(broken)
    cache.put('What is 2+2?', 'model', '4')
    assert cache.get('What is 2+2?', 'model') == ('4', 1.0)
    assert cache.get('What is 2 + 2?', 'model') is None
    assert cache.get('What is 2+2?', 'other') is None


def test_runtime_rejects_unguarded_cosine_mode():
    with pytest.raises(ValueError):
        Settings(_env_file=None, cache_match_mode='cosine')


def test_exact_benchmark_does_not_require_encoder():
    pairs = validate_pairs([{'id': 'same', 'category': 'exact', 'source': 'Q', 'query': 'Q', 'expected_hit': True}])
    def broken(_):
        pytest.fail('No embeddings for exact benchmark')
    report = run_cache_benchmark(pairs, broken, 'unused', [.95], policies=['exact'])
    assert report['summary'][0]['precision'] == 1
    assert report['results'][0]['decision'] == 'exact_hit'
    assert report['results'][0]['similarity'] is None


def test_api_reports_exact_default_and_preserves_cache():
    class Provider:
        def __init__(self):
            self.calls = 0
        def generate(self, prompt):
            self.calls += 1
            return '4', None
    provider = Provider()
    client = TestClient(create_app(Settings(_env_file=None), provider))
    first = client.post('/generate', json={'question': '2+2?', 'use_cache': True}).json()
    second = client.post('/generate', json={'question': '2+2?', 'use_cache': True}).json()
    assert first['cache_match_mode'] == 'exact'
    assert second['cache_status'] == 'hit'
    assert provider.calls == 1



def test_labeled_negative_pairs_block_even_with_identical_embeddings():
    root = Path(__file__).resolve().parents[1]
    for name in ['cache_pairs.json', 'cache_challenge_pairs.json']:
        pairs = validate_pairs(json.loads((root / 'experiments' / name).read_text()))
        for pair in pairs:
            if not pair.expected_hit:
                cache = SemanticCache(lambda _: [1, 0], threshold=0, match_mode='guarded')
                cache.put(pair.source, 'model', 'wrong answer')
                assert cache.get(pair.query, 'model') is None, pair.id


def test_dashboard_filters_policy_and_supports_exact_null_similarity(tmp_path):
    from streamlit.testing.v1 import AppTest
    from experiments.run_cache_eval import save_cache_report
    pairs = validate_pairs([
        {'id': 'same', 'category': 'exact', 'source': 'Q', 'query': 'Q', 'expected_hit': True},
        {'id': 'different', 'category': 'changed', 'source': 'A', 'query': 'B', 'expected_hit': False},
    ])
    report = run_cache_benchmark(pairs, lambda _: [1, 0], 'fake', [.95], policies=['cosine', 'guarded', 'exact'])
    save_cache_report(report, tmp_path)
    at = AppTest.from_string(f'from dashboard.app import render_dashboard\nrender_dashboard({str(tmp_path)!r})').run(timeout=15)
    at.sidebar.radio[0].set_value('Cache accuracy').run()
    assert not at.exception
    assert at.selectbox[1].value == 'exact'
    at.selectbox[1].set_value('guarded').run()
    at.checkbox[0].set_value(False).run()
    assert not at.exception
    assert set(at.dataframe[1].value['policy']) == {'guarded'}


def test_legacy_cache_reports_load_as_cosine_baseline(tmp_path):
    from dashboard.cache_view import load_cache_reports
    pairs = validate_pairs([{'id': 'same', 'category': 'exact', 'source': 'Q', 'query': 'Q', 'expected_hit': True}])
    report = run_cache_benchmark(pairs, lambda _: [1, 0], 'fake', [.95])
    for row in report['results']:
        row.pop('policy')
        row.pop('decision')
    for row in report['summary']:
        row.pop('policy')
    (tmp_path / 'cache-legacy.json').write_text(json.dumps(report))
    reports, skipped = load_cache_reports(tmp_path)
    assert not skipped
    assert reports['cache-legacy'].results[0].policy == 'cosine'



def test_guarded_benchmark_does_not_embed_vetoed_query(tmp_path):
    from dashboard.cache_view import load_cache_reports
    from experiments.run_cache_eval import save_cache_report
    source, query = 'Is 2 ≤ 3?', 'Is 2 ≥ 3?'
    calls = []
    def encoder(text):
        calls.append(text)
        if text == query:
            raise RuntimeError('This vetoed query must not be embedded')
        return [1, 0]
    pairs = validate_pairs([{'id': 'unicode', 'category': 'operator', 'source': source, 'query': query, 'expected_hit': False}])
    report = run_cache_benchmark(pairs, encoder, 'fake', [.95], policies=['guarded'])
    assert calls == [source]
    assert report['summary'][0]['errors'] == 0
    assert report['results'][0]['similarity'] is None
    assert report['results'][0]['decision'] == 'guard_veto'
    save_cache_report(report, tmp_path)
    loaded, skipped = load_cache_reports(tmp_path)
    assert len(loaded) == 1 and not skipped
