"""Local dashboard launch form and a polling view; workers never touch UI state."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.config import BASE_DIR, Settings
from app.services.run_inputs import RunOptions, parse_dataset, read_checkpoint, validate_run
from app.services.run_manager import RunManager


@st.cache_resource(show_spinner=False)
def get_manager(directory):
    return RunManager(directory)


@st.fragment(run_every=2)
def show_job(manager):
    job = manager.snapshot()
    if job is None:
        return
    observed = st.session_state.get('last_dashboard_job_state')
    st.session_state['last_dashboard_job_state'] = (job['id'], job['active'])
    if observed and observed[0] == job['id'] and observed[1] and not job['active']:
        st.rerun()
    st.subheader('Latest dashboard run')
    st.write('State: ' + job['state'])
    if job['error']:
        st.error(job['error'])
    report = job['report']
    if report:
        total = report['planned_cases']
        done = report['processed_cases']
        st.progress(done / total, text=f'Processed {done} of {total} cases')
        st.dataframe(pd.DataFrame(report['summary']), hide_index=True, width='stretch')
    if job['active']:
        if st.button('Cancel run', key='cancel_dashboard_run', disabled=job['state'] == 'cancelling'):
            if manager.cancel(job['id']):
                st.info('Cancellation requested. An in-flight provider request must return or time out first.')
            else:
                st.info('This run has already finished.')
    for kind in ['json', 'csv']:
        path = Path(job[kind + '_path']) if job[kind + '_path'] else None
        if path is not None and path.is_file():
            st.download_button('Download ' + ('saved progress JSON' if kind == 'json' else 'results CSV'),
                               path.read_bytes(), file_name=path.name,
                               mime='application/json' if kind == 'json' else 'text/csv', key='job_' + kind)
    st.caption('The worker continues if this tab closes. Restarting the dashboard stops its workers; resume from a saved JSON file after restart.')


def render_run_controls(directory):
    directory = Path(directory)
    manager = get_manager(str(directory.resolve()))
    st.subheader('Run an evaluation')
    st.caption('Choose a dataset and scoring policy. Nothing is sent to Gemini until you press Start.')
    source = st.radio('Run type', ['New evaluation', 'Resume saved run'])
    prior, cases, options, acknowledged = None, None, None, True
    try:
        settings = Settings()
    except ValueError:
        st.error('Local environment settings are invalid. Check the .env configuration.')
        show_job(manager)
        return
    if source == 'New evaluation':
        dataset = st.selectbox('Dataset', ['Sample questions', 'Quality and output requirements', 'Upload JSON'])
        if dataset == 'Upload JSON':
            upload = st.file_uploader('Evaluation dataset (JSON, up to 2 MB)', type=['json'])
            if upload is not None:
                try:
                    cases = parse_dataset(upload.getvalue())
                except ValueError as error:
                    st.error(str(error))
        else:
            name = 'sample_cases.json' if dataset == 'Sample questions' else 'quality_cases.json'
            try:
                cases = parse_dataset((BASE_DIR / 'experiments' / name).read_bytes())
            except (OSError, ValueError):
                st.error('The selected built-in dataset could not be loaded.')
        model_text = st.text_input('Generation model IDs (comma-separated)', value=settings.gemini_model)
        judge = st.text_input('Judge model ID', value=settings.gemini_judge_model)
        rubric = st.selectbox('Scoring rubric', ['quality-v2', 'legacy-v1'])
        retries = st.number_input('Additional retries per stage', min_value=0, max_value=3, value=2)
        interval = st.number_input('Minimum seconds between requests to each model', min_value=0.0, max_value=60.0, value=13.0)
        try:
            options = RunOptions(models=[m.strip() for m in model_text.split(',')], judge_model=judge,
                                 rubric=rubric, max_retries=retries, request_interval=interval)
        except ValueError:
            st.error('Use 1–4 unique model IDs with letters, digits, dots, slashes, underscores or hyphens.')
    else:
        reports = sorted((p for p in directory.glob('eval-*.json') if p.is_file()), reverse=True)
        if not reports:
            st.info('No saved runs are available in this results folder.')
        else:
            selected = st.selectbox('Saved progress', reports, format_func=lambda p: p.name)
            try:
                cases, options, prior = read_checkpoint(selected, directory)
                st.write('Models: ' + ', '.join(options.models))
                st.write('Judge: ' + options.judge_model + ' · Rubric: ' + options.rubric)
                if prior.get('run_state') == 'running':
                    acknowledged = st.checkbox('I have stopped the original run before resuming it')
                    st.caption('A running checkpoint may still have a live worker or CLI process. Stop that process first to avoid duplicate requests.')
            except (OSError, ValueError) as error:
                st.error(str(error) if isinstance(error, ValueError) else 'The saved progress file could not be read.')
                cases, options = None, None
    valid, budget, reused = False, 0, 0
    if cases and options:
        if prior is None:
            limit = st.number_input('Number of questions', min_value=1, max_value=len(cases), value=len(cases))
            cases = cases[:limit]
        st.write(f'{len(cases)} questions · {len(options.models)} generation model(s)')
        st.dataframe(pd.DataFrame([{'ID': c.id, 'Category': c.category, 'Question': c.question,
                                    'Reference': c.reference, 'Output requirements': json.dumps(c.checks.model_dump(exclude_defaults=True))}
                                   for c in cases[:5]]), hide_index=True, width='stretch')
        try:
            _, budget, reused = validate_run(cases, options, settings, prior)
            valid = True
        except ValueError as error:
            st.error(str(error))
        if valid:
            st.info(f'Provider-call budget: up to {budget} stage attempts, including retries. Reused completed evaluations: {reused}.')
        st.caption('Starting sends questions, answers and reference material to Gemini and may incur charges. This is a call-count bound, not a price estimate. Model access and quotas are checked by the provider when requests run. API keys stay in the local .env file.')
    active = manager.snapshot()
    busy = bool(active and active['active'])
    if st.button('Start evaluation (uses Gemini)', disabled=not valid or not acknowledged or busy):
        try:
            manager.start(cases, options, settings, prior)
            st.success('Evaluation started. Saved progress appears below.')
        except (ValueError, OSError) as error:
            st.error(str(error) if isinstance(error, ValueError) else 'The results folder could not be prepared.')
    show_job(manager)
