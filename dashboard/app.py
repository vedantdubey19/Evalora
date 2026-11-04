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


def render_dashboard(results_dir=None):
    st.set_page_config(page_title='Evalora · Results', page_icon='📊', layout='wide')
    st.title('Evalora')
    st.caption('LLM evaluation · Compare quality, speed, and failures')
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
    selected_runs = st.sidebar.multiselect('Runs', runs, default=runs[:2])
    available = frame[frame.run.isin(selected_runs)]
    models = list(available.model.unique())
    selected_models = st.sidebar.multiselect('Models', models, default=models)
    filtered = available[available.model.isin(selected_models)]
    if filtered.empty:
        st.info('Select at least one run and model to view results.')
        return

    st.caption('Report times are UTC. Scores are judge estimates on a 1–5 scale; inspect references and reasoning before drawing conclusions.')
    if filtered.judge_model.nunique() > 1:
        st.warning('These results use different judges. Score differences may reflect the judge as well as the generation model.')
    case_sets = []
    for _, group in filtered.groupby(['run', 'model']):
        case_sets.append(frozenset(zip(group.case_id, group.question, group.reference)))
    if len(set(case_sets)) > 1:
        st.warning('Selected runs or models have different test cases. Their average scores are not directly comparable.')

    success = filtered[filtered.status == 'ok']
    metrics = st.columns(4)
    metrics[0].metric('Evaluated', f'{len(success)} / {len(filtered)}')
    metrics[1].metric('Mean judge score', f'{success.overall.mean():.2f} / 5' if len(success) else '—')
    latency = pd.to_numeric(filtered.generation_ms, errors='coerce').mean()
    metrics[2].metric('Mean generation time', f'{latency / 1000:.2f} s' if pd.notna(latency) else '—')
    metrics[3].metric('Failed cases', int((filtered.status != 'ok').sum()))

    st.subheader('Run comparison')
    summary = summarize(filtered)
    st.dataframe(summary, hide_index=True, width='stretch')
    st.caption('Token totals cover generation only, including provider-reported thinking tokens. Latency excludes judging. Failed evaluations are excluded from mean scores.')
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
    st.download_button('Download comparison CSV', summary.to_csv(index=False),
                       file_name='evalora-comparison.csv', mime='text/csv')

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
