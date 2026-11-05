from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.cache import SemanticCache
from app.services.llm_client import GeminiClient, ProviderError


def settings(**kwargs):
    return Settings(_env_file=None, gemini_api_key='', **kwargs)


class FakeProvider:
    def __init__(self):
        self.settings = settings()
        self.calls = 0
        self.failure = False
        self.judgment = '{"correctness":5,"relevance":4,"clarity":3,"reasoning":"Matches reference."}'

    def generate(self, prompt, *, judge=False, schema=None):
        self.calls += 1
        if self.failure:
            raise ProviderError('secret-provider-detail')
        return (self.judgment if judge else 'A useful answer.'), SimpleNamespace(prompt_token_count=4, candidates_token_count=6, total_token_count=10)


def test_validation_and_missing_key():
    client = TestClient(create_app(settings()))
    assert client.get('/health').json()['gemini_configured'] is False
    for question in ['', '   ', 'x' * 20001]:
        assert client.post('/generate', json={'question': question}).status_code == 422
    assert client.post('/generate', json={'question': 'Hello'}).status_code == 503
    assert client.post('/evaluate', json={'question': 'Q', 'answer': 'A'}).status_code == 503


def test_cache_hit_and_bypass():
    provider = FakeProvider()
    cache = SemanticCache(lambda _: [1, 0], match_mode='guarded')
    client = TestClient(create_app(settings(), provider, cache))
    payload = {'question': 'What is the capital of France?', 'use_cache': True}
    first = client.post('/generate', json=payload).json()
    assert first['cache_status'] == 'miss'
    assert first['usage']['total_tokens'] == 10
    second = client.post('/generate', json={'question': 'What is the capital city of France?', 'use_cache': True}).json()
    assert second['cache_status'] == 'hit'
    assert second['usage'] is None
    assert provider.calls == 1
    assert client.post('/generate', json={'question': 'Question?'}).json()['cache_status'] == 'disabled'
    assert provider.calls == 2


def test_errors_not_cached_or_leaked():
    provider = FakeProvider()
    provider.failure = True
    cache = SemanticCache(lambda _: [1, 0])
    client = TestClient(create_app(settings(), provider, cache))
    response = client.post('/generate', json={'question': 'Q', 'use_cache': True})
    assert response.status_code == 502
    assert 'secret-provider-detail' not in response.text
    assert not cache.entries


def test_cache_failure_falls_back():
    def broken(_):
        raise RuntimeError('embedding unavailable')
    client = TestClient(create_app(settings(), FakeProvider(), SemanticCache(broken, match_mode='guarded')))
    result = client.post('/generate', json={'question': 'Q', 'use_cache': True})
    assert result.status_code == 200
    assert result.json()['cache_status'] == 'unavailable'


def test_judge_and_malformed_response():
    provider = FakeProvider()
    client = TestClient(create_app(settings(), provider))
    payload = {'question': 'Q', 'answer': 'A', 'reference': 'R'}
    response = client.post('/evaluate', json=payload)
    assert response.status_code == 200
    assert response.json()['overall'] == 4
    assert response.json()['reference_provided'] is True
    provider.judgment = '{"correctness":6}'
    assert client.post('/evaluate', json=payload).status_code == 502


def test_cache_expiry_capacity_model_and_threshold():
    now = [0]
    vectors = {'a': [1, 0], 'b': [0, 1], 'c': [1, 1]}
    cache = SemanticCache(vectors.get, threshold=.95, ttl=10, capacity=1, clock=lambda: now[0])
    cache.put('a', 'model1', 'answer a')
    assert cache.get('a', 'model2') is None
    assert cache.get('b', 'model1') is None
    cache.put('b', 'model1', 'answer b')
    assert cache.get('a', 'model1') is None
    assert cache.get('b', 'model1') == ('answer b', 1.0)
    now[0] = 10
    assert cache.get('b', 'model1') is None


def test_sdk_call_and_error_sanitization():
    config = settings()
    config.gemini_api_key = 'test-key'
    with patch('app.services.llm_client.genai.Client') as factory:
        sdk = factory.return_value.__enter__.return_value
        sdk.models.generate_content.return_value = SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason='STOP')], text='Hello', usage_metadata=None)
        assert GeminiClient(config).generate('Q')[0] == 'Hello'
        call = sdk.models.generate_content.call_args
        assert call.kwargs['model'] == 'gemini-2.5-flash'
        assert factory.call_args.kwargs['http_options'].timeout == 60000
        sdk.models.generate_content.side_effect = RuntimeError('test-key')
        with pytest.raises(ProviderError) as error:
            GeminiClient(config).generate('Q')
        assert 'test-key' not in str(error.value)


@pytest.mark.parametrize('finish,text', [('MAX_TOKENS', 'partial'), ('SAFETY', ''), ('STOP', '')])
def test_reject_incomplete_provider_results(finish, text):
    config = settings()
    config.gemini_api_key = 'test-key'
    with patch('app.services.llm_client.genai.Client') as factory:
        sdk = factory.return_value.__enter__.return_value
        sdk.models.generate_content.return_value = SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason=finish)], text=text, usage_metadata=None)
        with pytest.raises(ProviderError):
            GeminiClient(config).generate('Q')


@pytest.mark.parametrize('key', ['', '   ', 'your_actual_api_key', ' your_actual_api_key '])
def test_missing_and_placeholder_keys_share_configuration_state(key):
    from app.services.llm_client import ConfigurationError
    config = Settings(_env_file=None, gemini_api_key=key)
    assert config.has_api_key is False
    client = TestClient(create_app(config))
    assert client.get('/health').json()['gemini_configured'] is False
    assert client.post('/generate', json={'question': 'Q'}).status_code == 503
    with pytest.raises(ConfigurationError):
        GeminiClient(config).generate('Q')


def test_real_key_configuration_state():
    config = Settings(_env_file=None, gemini_api_key='unit-test-key')
    assert config.has_api_key is True
    client = TestClient(create_app(config))
    assert client.get('/health').json()['gemini_configured'] is True
