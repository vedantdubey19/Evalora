"""Launch with: python -m streamlit run dashboard/app.py"""
from pathlib import Path
import sys

# Streamlit executes this as a script; make project imports independent of cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from dashboard.reports import load_reports, summarize
from dashboard.cache_view import render_cache_view


def render_dashboard(results_dir=None):
    st.set_page_config(page_title='Evalora · Results', page_icon='📊', layout='wide')
    st.title('Evalora')
    st.caption('LLM evaluation · Compare quality, speed, and failures')
    view = st.sidebar.radio('View', ['Model evaluations', 'Cache accuracy'])
    directory = Path(results_dir) if results_dir else ROOT / 'results'
    if view == 'Cache accuracy':
        render_cache_view(directory)
        return
    st.sidebar.header('Saved evaluations')
    st.sidebar.button('Refresh reports')
    frame, skipped = load_reports(Path(results_dir) if results_dir else ROOT / 'results')
    if skipped:
        st.warning(f'Skipped {len(skipped)} unreadable or invalid report(s): ' + ', '.join(skipped))
    if frame.empty:
        st.info('No evaluation reports yet. Run the sample evaluation to get started.')
        st.code('python -m experiments.run_eval', language='bash')
        return

    runs = list(frame.run.unique())
    selected_runs = st.sidebar.multiselect('Runs', runs, default=runs[:1])
    available = frame[frame.run.isin(selected_runs)]
    models = list(available.model.unique())
    selected_models = st.sidebar.multiselect('Models', models, default=models)
    filtered = available[available.model.isin(selected_models)]
    categories = list(filtered.category.unique())
    selected_categories = st.sidebar.multiselect('Categories', categories, default=categories)
    filtered = filtered[filtered.category.isin(selected_categories)]
    if filtered.empty:
        st.info('Select at least one run, model, and category to view results.')
        return

    st.caption('Report times are UTC. Scores are judge estimates on a 1–5 scale; inspect references and reasoning before drawing conclusions.')
    if filtered.judge_model.nunique() > 1:
        st.warning('These results use different judges. Score differences may reflect the judge as well as the generation model.')
    case_sets = []
    for _, group in filtered.groupby(['run', 'model']):
        case_sets.append(frozenset(zip(group.case_id, group.question, group.reference)))
    if len(set(case_sets)) > 1:
        st.warning('Selected runs or models have different test cases. Their average scores are not directly comparable.')

    if (filtered.status != 'ok').any():
        st.warning('This comparison is incomplete. Scores include successful evaluations only and may reflect different subsets of questions.')
    success = filtered[filtered.status == 'ok']
    metrics = st.columns(4)
    metrics[0].metric('Evaluated', f'{len(success)} / {len(filtered)}')
    metrics[1].metric('Mean judge score', f'{success.overall.mean():.2f} / 5' if len(success) else '—')
    latency = pd.to_numeric(filtered.generation_ms, errors='coerce').mean()
    metrics[2].metric('Mean generation time', f'{latency / 1000:.2f} s' if pd.notna(latency) else '—')
    failure_states = ['generation_error', 'evaluation_error', 'configuration_error', 'interrupted']
    metrics[3].metric('Failures / legacy errors' if (filtered.accounting_version == 'legacy').any() else 'Failed attempts',
                      int(filtered.status.isin(failure_states).sum()))
    processed = int((filtered.status != 'pending').sum())
    st.progress(processed / len(filtered), text=f'Processed {processed} of {len(filtered)} cases')
    st.caption(f'Skipped: {int((filtered.status == "skipped").sum())} · Pending: {int((filtered.status == "pending").sum())} · Reused: {int(filtered.reused.sum())}')
    if (filtered.accounting_version == 'legacy').any():
        st.info('Historical reports may combine skipped cases with failures; their provider-attempt counts are unknown.')
    states = ', '.join(sorted(filtered.run_state.unique()))
    st.caption(f'Run state: {states}. A running checkpoint can be refreshed to see saved progress; it may also belong to a process that stopped unexpectedly.')

    st.subheader('Run comparison')
    summary = summarize(filtered)
    st.dataframe(summary, hide_index=True, width='stretch')
    st.caption('Generation token totals describe saved answers, including reused answers; judge usage is excluded. Provider-attempt and retry counts cover this run only and are not billing totals. Latency includes provider attempts but excludes pacing and retry waits. Failed or skipped evaluations are excluded from mean scores.')
    if len(summary):
        chart = summary.copy()
        chart['Run / model'] = chart['Run'] + ' · ' + chart['Model']
        left, right = st.columns(2)
        with left:
            st.write('Mean judge score')
            scores = chart.dropna(subset=['Mean score'])
            if len(scores):
                st.bar_chart(scores.set_index('Run / model')[['Mean score']], color='#4f46e5')
            else:
                st.info('No successful evaluations to chart.')
        with right:
            st.write('Mean generation time (seconds)')
            chart['Seconds'] = pd.to_numeric(chart['Generation ms'], errors='coerce') / 1000
            times = chart.dropna(subset=['Seconds'])
            if len(times):
                st.bar_chart(times.set_index('Run / model')[['Seconds']], color='#0d9488')
            else:
                st.info('No completed generations to chart.')
    stopped = filtered[filtered.status.isin(['generation_error', 'evaluation_error', 'configuration_error', 'interrupted', 'skipped'])]
    if len(stopped):
        st.subheader('Why cases stopped')
        reasons = stopped.groupby(['model', 'status', 'error_reason'], dropna=False).size().reset_index(name='Cases')
        reasons = reasons.rename(columns={'model': 'Model', 'status': 'Outcome', 'error_reason': 'Reason'})
        st.dataframe(reasons, hide_index=True, width='stretch')
        st.caption('Skipped cases did not make a new provider call. Failure reasons are safe labels; provider messages and URLs are not stored.')
    st.download_button('Download comparison CSV', summary.to_csv(index=False),
                       file_name='evalora-comparison.csv', mime='text/csv')

    if len(success):
        st.subheader('Scores by category')
        category_scores = success.groupby(['category', 'model'], as_index=False).agg(
            evaluated=('overall', 'count'), mean_score=('overall', 'mean'))
        st.dataframe(category_scores, hide_index=True, width='stretch')
        st.caption('Category scores pool successful evaluations across the selected runs.')

    st.subheader('Inspect a case')
    statuses = st.multiselect('Case status', list(filtered.status.unique()), default=list(filtered.status.unique()))
    cases = filtered[filtered.status.isin(statuses)]
    if cases.empty:
        st.info('No cases match the selected status.')
        return
    index = st.selectbox('Case', list(cases.index),
                         format_func=lambda i: f'{cases.loc[i, "case_id"]} · {cases.loc[i, "model"]} · {cases.loc[i, "run"]}')
    row = cases.loc[index]
    st.caption(f'Status: {row.status} · Judge: {row.judge_model}')
    error_labels = {
        'rate_limited': 'The provider rate limit was reached.',
        'judge_rate_limited': 'The judge rate limit was reached; further generation was skipped.',
        'model_unavailable': 'This model was unavailable for generation.',
        'provider_unavailable': 'The provider was temporarily unavailable.',
        'judge_provider_unavailable': 'The judge was unavailable; further generation was skipped.',
        'judge_model_unavailable': 'The configured judge model was unavailable.',
        'daily_quota': 'The daily quota was reached. This run will not retry it.',
        'judge_daily_quota': 'The judge daily quota was reached; further generation was skipped.',
        'timeout': 'The provider request timed out after bounded retry attempts.',
        'network_error': 'The provider could not be reached after bounded retry attempts.',
        'server_error': 'The provider returned server errors after bounded retries.',
        'interrupted': 'This case was interrupted; completed earlier cases remain saved.',
        'configuration_error': 'Provider configuration is missing or invalid.',
    }
    if row.error_reason in error_labels:
        st.warning(error_labels[row.error_reason])
    st.write('Question')
    st.text(row.question)
    left, right = st.columns(2)
    with left:
        st.write('Generated answer')
        st.text(row.answer if row.answer is not None else 'No answer generated.')
    with right:
        st.write('Reference answer')
        st.text(row.reference)
    if row.status == 'ok':
        st.write('Judge scores')
        st.dataframe(pd.DataFrame([{'Correctness': row.correctness, 'Relevance': row.relevance,
                                    'Clarity': row.clarity, 'Overall': row.overall}]), hide_index=True)
        st.write('Judge reasoning')
        st.text(row.reasoning)
    else:
        st.info('This case did not complete evaluation. See its status above; raw provider errors are not stored.')


if __name__ == '__main__':
    render_dashboard()
