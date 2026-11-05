import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app
from app.services.limits import LimitExceeded, LimitStore
from app.services.llm_client import GeminiClient, LocalLimitError
from app.services.report_repository import ReportRepository
from dashboard.security import gateway_authenticated

TOKEN = 'a' * 40
GATEWAY = 'b' * 40


def production(tmp_path, **kwargs):
    values = dict(_env_file=None, deployment_mode='production', api_access_token=TOKEN,
                  gateway_secret=GATEWAY, allowed_hosts=['testserver', 'localhost'],
                  reports_dir=tmp_path / 'reports', state_dir=tmp_path / 'state',
                  gemini_api_key='fake-key', cache_match_mode='exact')
    values.update(kwargs)
    return Settings(**values)


def test_production_fails_closed(tmp_path):
    for update in [dict(api_access_token=''), dict(gateway_secret='short'),
                   dict(gateway_secret=TOKEN), dict(allowed_hosts=['*']),
                   dict(reports_dir=Path('relative'))]:
        with pytest.raises(ValidationError):
            production(tmp_path, **update)


def test_all_sensitive_routes_need_token_and_health_is_minimal(tmp_path):
    client = TestClient(create_app(production(tmp_path)))
    for path in ['/', '/docs', '/openapi.json', '/reports', '/reports/eval-test.json']:
        response = client.get(path)
        assert response.status_code == 401
        assert response.headers['cache-control'] == 'no-store'
    for path in ['/generate', '/evaluate', '/evaluate/quality', '/check']:
        assert client.post(path, json={}).status_code == 401
    assert client.get('/health').json() == {'status': 'ok'}
    assert client.get('/health', headers={'Host': 'evil.example'}).status_code == 400
    assert client.get('/reports', headers={'Authorization': 'Bearer wrong'}).status_code == 401
    assert client.get('/reports', headers=[('Authorization', 'Bearer '+TOKEN), ('Authorization', 'Bearer '+TOKEN)]).status_code == 401
    auth = {'Authorization': 'Bearer '+TOKEN}
    assert client.get('/reports', headers=auth).json() == {'reports': []}
    assert client.get('/docs', headers=auth).status_code == 404


def test_rate_limit_shared_across_apps_and_restarts(tmp_path, monkeypatch):
    original_init = LimitStore.__init__
    monkeypatch.setattr(LimitStore, "__init__", lambda self, directory: original_init(self, directory, clock=lambda: 100))
    settings = production(tmp_path, api_requests_per_minute=1)
    auth = {'Authorization': 'Bearer '+TOKEN}
    with TestClient(create_app(settings)) as first:
        assert first.get('/reports', headers=auth).status_code == 200
    with TestClient(create_app(settings)) as second:
        response = second.get('/reports', headers=auth)
        assert response.status_code == 429 and int(response.headers['Retry-After']) > 0


def test_dashboard_requires_unforgeable_gateway_header(tmp_path):
    settings = production(tmp_path)
    assert not gateway_authenticated(settings, {})
    assert not gateway_authenticated(settings, {'X-Evalora-Gateway': 'wrong'})
    assert gateway_authenticated(settings, {'X-Evalora-Gateway': GATEWAY})
    assert gateway_authenticated(Settings(_env_file=None), {})


def test_chunked_body_cannot_bypass_limit(tmp_path):
    app = create_app(production(tmp_path, max_request_bytes=1000))
    messages = iter([{'type': 'http.request', 'body': b'x'*700, 'more_body': True},
                     {'type': 'http.request', 'body': b'x'*700, 'more_body': False}])
    sent = []
    async def receive():
        return next(messages)
    async def send(message):
        sent.append(message)
    asyncio.run(app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                     'method': 'POST', 'scheme': 'http', 'path': '/check', 'query_string': b'',
                     'headers': [(b'host', b'testserver'), (b'authorization', ('Bearer '+TOKEN).encode())],
                     'client': ('127.0.0.1', 42), 'server': ('testserver', 80)}, receive, send))
    assert sent[0]['status'] == 413


def test_atomic_limits_and_window_expiry(tmp_path):
    now = [100.0]
    store = LimitStore(tmp_path, clock=lambda: now[0])
    def consume(_):
        try:
            store.consume('test', 3)
            return True
        except LimitExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(consume, range(20))) == 3
    assert not consume(None)
    now[0] = 120
    assert consume(None)


def test_provider_budget_and_leases_survive_store_recreation(tmp_path):
    now = [100.0]
    first = LimitStore(tmp_path, clock=lambda: now[0])
    second = LimitStore(tmp_path, clock=lambda: now[0])
    with first.provider_slot(2, 1, 60):
        with pytest.raises(LimitExceeded):
            with second.provider_slot(2, 1, 60):
                pytest.fail('Concurrent call admitted')
    with second.provider_slot(2, 1, 60):
        pass
    with pytest.raises(LimitExceeded) as error:
        with first.provider_slot(2, 1, 60):
            pass
    assert error.value.reason == 'daily_quota'
    now[0] = 86400
    with first.provider_slot(2, 1, 60):
        pass


def test_provider_limit_does_not_call_sdk_or_leak_secrets(tmp_path):
    settings = production(tmp_path, provider_calls_per_day=1)
    with LimitStore(settings.state_dir).provider_slot(1, 1, 60):
        pass
    with patch('app.services.llm_client.genai.Client') as sdk:
        with pytest.raises(LocalLimitError) as error:
            GeminiClient(settings).generate('private question')
        assert sdk.return_value.__enter__.return_value.models.generate_content.call_count == 0
        assert error.value.reason == 'daily_quota' and error.value.retry_after is None
    with TestClient(create_app(settings)) as client:
        response = client.post('/generate', headers={'Authorization': 'Bearer '+TOKEN}, json={'question': 'Q'})
    assert response.status_code == 429
    assert TOKEN not in response.text and 'fake-key' not in response.text


def test_persistent_report_download_and_path_isolation(tmp_path):
    settings = production(tmp_path)
    settings.reports_dir.mkdir()
    name = 'eval-integration.json'
    payload = json.dumps({'results': [], 'safe': True})
    (settings.reports_dir / name).write_text(payload)
    outside = tmp_path / 'private.json'
    outside.write_text('private')
    (settings.reports_dir / 'eval-link.json').symlink_to(outside)
    auth = {'Authorization': 'Bearer '+TOKEN}
    for _ in range(2):
        with TestClient(create_app(settings)) as client:
            assert client.get('/reports', headers=auth).json()['reports'] == [name]
            response = client.get('/reports/'+name, headers=auth)
            assert response.text == payload
            assert response.headers['content-disposition'].startswith('attachment;')
            assert client.get('/reports/eval-link.json', headers=auth).status_code == 404
    repo = ReportRepository(settings.reports_dir)
    for name in ['../private.json', '/etc/passwd', 'eval-x.json\r\nInjected: true']:
        with pytest.raises(ValueError):
            repo.read(name)


def test_crashed_provider_lease_expires_and_failed_calls_count(tmp_path):
    now = [100.0]
    store = LimitStore(tmp_path, clock=lambda: now[0])
    with pytest.raises(RuntimeError):
        with store.provider_slot(3, 1, 5):
            raise RuntimeError('simulated failure')
    # Failed attempt still consumed one unit; the lease is released.
    with store.provider_slot(3, 1, 5):
        pass
    import sqlite3
    with sqlite3.connect(store.path) as connection:
        connection.execute('INSERT INTO leases VALUES (?, ?)', ('crashed-worker', 110))
    with pytest.raises(LimitExceeded):
        with store.provider_slot(3, 1, 5):
            pass
    now[0] = 111
    with store.provider_slot(3, 1, 5):
        pass
    with pytest.raises(LimitExceeded):
        with store.provider_slot(3, 1, 5):
            pass


def test_broken_limit_store_fails_closed(tmp_path):
    import sqlite3
    settings = production(tmp_path)
    client = TestClient(create_app(settings))
    # Initialize middleware, then simulate a runtime failure rather than reset counters.
    assert client.get('/health').status_code == 200
    with patch.object(LimitStore, 'consume', side_effect=sqlite3.OperationalError('private detail')):
        response = client.get('/reports', headers={'Authorization': 'Bearer '+TOKEN})
    assert response.status_code == 503 and 'private detail' not in response.text


def test_private_backup_contains_reports_and_consistent_limits(tmp_path):
    import sqlite3
    import tarfile
    from deploy.backup import backup
    settings = production(tmp_path)
    settings.reports_dir.mkdir()
    (settings.reports_dir / 'eval-test.json').write_text('{"results": []}')
    LimitStore(settings.state_dir).consume('test', 5)
    output = tmp_path / 'backup.tar.gz'
    backup(settings, output)
    assert output.stat().st_mode & 0o777 == 0o600
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == {'reports/eval-test.json', 'state/limits.sqlite3'}
        restored = tmp_path / 'restored.sqlite3'
        restored.write_bytes(archive.extractfile('state/limits.sqlite3').read())
    with sqlite3.connect(restored) as database:
        assert database.execute('SELECT count FROM counters WHERE bucket="test"').fetchone()[0] == 1
    with pytest.raises(FileExistsError):
        backup(settings, output)


def test_unauthorized_dashboard_stops_before_loading_private_data(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    settings = production(tmp_path)
    monkeypatch.setattr('app.core.config.Settings', lambda: settings)
    def forbidden(*args, **kwargs):
        pytest.fail('Private reports were read before gateway authentication')
    monkeypatch.setattr('dashboard.reports.load_reports', forbidden)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'dashboard/app.py').run()
    assert not app.exception
    assert app.error[0].value == 'Open this dashboard through the authenticated HTTPS gateway.'
    assert not app.sidebar.radio


def test_deployment_generator_keeps_secrets_private_and_refuses_overwrite(tmp_path, capsys):
    from types import SimpleNamespace
    from deploy.configure import configure
    generated_hash = '$2b$14$' + 'a'*53
    with patch('deploy.configure.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=generated_hash)) as docker:
        environment, credentials = configure('example.com', 'owner@example.com', 'private-test-key', tmp_path)
    assert not capsys.readouterr().out
    assert environment.stat().st_mode & 0o777 == 0o600
    assert credentials.stat().st_mode & 0o777 == 0o600
    assert 'private-test-key' in environment.read_text()
    assert 'private-test-key' not in credentials.read_text()
    invocation = docker.call_args
    assert 'private-test-key' not in str(invocation.args)
    assert invocation.kwargs['input'].strip() not in str(invocation.args)
    with pytest.raises(ValueError, match='already exist'):
        configure('example.com', 'owner@example.com', 'new-key', tmp_path)


def test_deployment_generator_hash_failure_leaves_no_secret_files(tmp_path):
    from types import SimpleNamespace
    from deploy.configure import configure
    with patch('deploy.configure.subprocess.run', return_value=SimpleNamespace(returncode=1, stdout='')):
        with pytest.raises(ValueError, match='hashing failed'):
            configure('example.com', 'owner@example.com', 'private-test-key', tmp_path)
    assert not list(tmp_path.iterdir())


def test_dashboard_limit_storage_error_is_sanitized(tmp_path):
    import sqlite3
    from app.services.run_manager import RunManager
    from app.services.run_inputs import RunOptions
    from app.services.benchmark import validate_cases
    settings = production(tmp_path)
    cases = validate_cases([{'id': 'one', 'question': 'Q', 'reference': 'R'}])
    options = RunOptions(models=['model'], judge_model='judge', rubric='quality-v2')
    with patch.object(LimitStore, 'consume', side_effect=sqlite3.OperationalError('secret path')):
        with pytest.raises(ValueError, match='Run limits are temporarily unavailable') as error:
            RunManager(settings.reports_dir).start(cases, options, settings)
    assert 'secret path' not in str(error.value)
    assert not settings.reports_dir.exists()


@pytest.mark.parametrize('failure,reason', [(TimeoutError('private'), 'timeout'),
                                           (ConnectionError('private'), 'network_error')])
def test_sdk_network_errors_retain_reason(tmp_path, failure, reason):
    from app.services.llm_client import ProviderError
    with patch('app.services.llm_client.genai.Client') as sdk:
        sdk.return_value.__enter__.return_value.models.generate_content.side_effect = failure
        with pytest.raises(ProviderError) as error:
            GeminiClient(production(tmp_path)).generate('private question')
    assert error.value.reason == reason and 'private' not in str(error.value)
