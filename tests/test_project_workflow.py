"""Exercise the release workflow across API, worker, files, resume and dashboard."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.run_inputs import RunOptions, parse_dataset, read_checkpoint
from app.services.run_manager import RunManager
from dashboard.reports import load_reports


class OfflineProvider:
    calls = 0

    def __init__(self, settings):
        self.settings = settings

    def generate(self, prompt, *, judge=False, schema=None):
        type(self).calls += 1
        scores = dict(correctness=5, relevance=5, clarity=5,
                      instruction_following=5, completeness=5,
                      reasoning='Synthetic integration-test judgment.', limitations=[])
        return (json.dumps(scores) if judge else '4'), None


def test_complete_local_workflow_and_resume_without_more_calls(tmp_path):
    OfflineProvider.calls = 0
    config = Settings(_env_file=None, gemini_api_key='offline-test-secret',
                      evaluation_rubric='quality-v2', cache_match_mode='exact')
    with TestClient(create_app(config, OfflineProvider(config))) as client:
        generated = client.post('/generate', json={'question': 'What is 2 + 2?', 'use_cache': True})
        assert generated.status_code == 200
        assert generated.json()['answer'] == '4'
        assert client.post('/generate', json={'question': 'What is 2 + 2?', 'use_cache': True}).json()['cache_status'] == 'hit'
        scored = client.post('/evaluate/quality', json={
            'question': 'What is 2 + 2?', 'answer': '4', 'reference': '4',
            'checks': {'expected_exact': '4'}})
        assert scored.status_code == 200
        assert scored.json()['checks_passed'] is True
    cases = parse_dataset(json.dumps([dict(id='arithmetic', question='What is 2 + 2?',
                         reference='4', checks={'expected_exact': '4'})]).encode())
    options = RunOptions(models=['offline-model'], judge_model='offline-judge',
                         rubric='quality-v2', request_interval=0, max_retries=0)
    manager = RunManager(tmp_path, OfflineProvider)
    job = manager.start(cases, options, config)
    assert manager.wait(job)
    snapshot = manager.snapshot(job)
    assert snapshot['state'] == 'finished'
    assert snapshot['report']['results'][0]['evaluation']['checks_passed'] is True
    records, skipped = load_reports(tmp_path)
    assert not skipped and len(records) == 1
    assert 'offline-test-secret' not in Path(snapshot['json_path']).read_text()
    assert Path(snapshot['csv_path']).read_text().count('\n') == 2
    restored_cases, restored_options, prior = read_checkpoint(Path(snapshot['json_path']), tmp_path)
    before_resume = OfflineProvider.calls
    resumed = manager.start(restored_cases, restored_options, config, prior)
    assert manager.wait(resumed)
    assert manager.snapshot(resumed)['report']['results'][0]['reused'] is True
    assert OfflineProvider.calls == before_resume
