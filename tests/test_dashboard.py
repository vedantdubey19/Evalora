import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.reports import load_reports, summarize


def make_report(directory, name='eval-test.json', status='ok'):
    row = {'case_id': 'arithmetic', 'model': 'gemini-test', 'judge_model': 'judge-test',
           'question': '2+2?', 'reference': '4', 'answer': '4', 'status': status,
           'generation_ms': 100, 'evaluation_ms': 150, 'usage': {'total_tokens': 20},
           'evaluation': {'correctness': 5, 'relevance': 5, 'clarity': 5, 'overall': 5, 'reasoning': 'Correct.'} if status == 'ok' else None}
    (directory / name).write_text(json.dumps({'created_at': '2026-11-05T00:00:00Z', 'judge_model': 'judge-test', 'results': [row]}))


def render(directory):
    return AppTest.from_string(f'from dashboard.app import render_dashboard\nrender_dashboard({str(directory)!r})').run(timeout=15)


def test_empty_dashboard(tmp_path):
    at = render(tmp_path)
    assert not at.exception
    assert 'No evaluation reports' in at.info[0].value


def test_dashboard_renders_filters_and_case_details(tmp_path):
    make_report(tmp_path)
    at = render(tmp_path)
    assert not at.exception
    assert at.metric[0].value == '1 / 1'
    assert at.metric[1].value == '5.00 / 5'
    assert any(item.value == 'Correct.' for item in at.text)
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception
    assert any('Select at least one' in item.value for item in at.info)


def test_dashboard_invalid_file_and_all_failures(tmp_path):
    make_report(tmp_path, status='generation_error')
    (tmp_path / 'eval-broken.json').write_text('{')
    at = render(tmp_path)
    assert not at.exception
    assert at.metric[0].value == '0 / 1'
    assert at.metric[1].value == '—'
    assert 'Skipped 1' in at.warning[0].value
    frame, skipped = load_reports(tmp_path)
    summary = summarize(frame)
    assert summary.iloc[0]['Failures'] == 1
    assert summary.iloc[0]['Mean score'] is None
    assert skipped == ['eval-broken.json']
