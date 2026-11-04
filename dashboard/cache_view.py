"""Read-only view of cache quality reports."""
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, StrictBool, model_validator
import streamlit as st


class CacheSummary(BaseModel):
    threshold: float = Field(ge=0, le=1)
    pairs: int = Field(ge=1)
    evaluated: int = Field(ge=0)
    errors: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    false_positive_rate: float | None = Field(default=None, ge=0, le=1)


class CacheLookup(BaseModel):
    id: str
    category: str
    source: str
    query: str
    expected_hit: StrictBool
    actual_hit: StrictBool | None
    similarity: float | None = Field(ge=-1, le=1)
    threshold: float = Field(ge=0, le=1)
    status: Literal['ok', 'embedding_error']

    @model_validator(mode='after')
    def check_success(self):
        if self.status == 'ok' and (self.actual_hit is None or self.similarity is None):
            raise ValueError('Successful lookups need an outcome and similarity.')
        return self


class CacheReport(BaseModel):
    created_at: str
    embedding_model: str
    summary: list[CacheSummary] = Field(min_length=1)
    results: list[CacheLookup] = Field(min_length=1)


def load_cache_reports(directory: Path):
    reports, skipped = {}, []
    for path in sorted(directory.glob('cache-*.json'), reverse=True):
        try:
            if path.stat().st_size > 10_000_000:
                raise ValueError('Report too large')
            reports[path.stem] = CacheReport.model_validate_json(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            skipped.append(path.name)
    return reports, skipped


def render_cache_view(directory):
    reports, skipped = load_cache_reports(directory)
    if skipped:
        st.warning(f'Skipped {len(skipped)} invalid cache report(s).')
    st.subheader('Cache accuracy')
    st.caption('Can a saved answer safely be reused? Each labeled pair is tested independently, without LLM calls.')
    if not reports:
        st.info('No cache accuracy reports yet.')
        st.code('python -m experiments.run_cache_eval', language='bash')
        return
    selected = st.selectbox('Cache run', list(reports))
    report = reports[selected]
    st.caption(f'Embedding model: {report.embedding_model} · Report time: {report.created_at}')
    frame = pd.DataFrame([s.model_dump() for s in report.summary])
    st.dataframe(frame, hide_index=True, width='stretch')
    st.line_chart(frame.set_index('threshold')[['precision', 'recall', 'false_positive_rate']])
    st.caption('Precision measures how often a hit is appropriate. Recall measures how many reusable answers were found. False positives reuse an answer for a different question; lower is safer. Missing rates mean the denominator was zero. Errors are excluded from rates.')
    threshold = st.selectbox('Inspect threshold', list(frame.threshold), index=min(2, len(frame) - 1))
    rows = pd.DataFrame([r.model_dump() for r in report.results if r.threshold == threshold])
    errors = int((rows.status != 'ok').sum()) if len(rows) else 0
    if errors:
        st.warning(f'{errors} pair(s) could not be embedded at this threshold.')
    mistakes_only = st.checkbox('Show mismatches only', value=True)
    if mistakes_only and len(rows):
        rows = rows[(rows.status == 'ok') & (rows.expected_hit != rows.actual_hit)]
    st.dataframe(rows, hide_index=True, width='stretch')
    st.download_button('Download cache pairs CSV', rows.to_csv(index=False),
                       file_name='evalora-cache-pairs.csv', mime='text/csv')
    st.info('These small, hand-labeled pairs measure matching behavior, not generated-answer quality or production safety. Thresholds were tested on this same dataset; use separate held-out pairs before changing defaults.')
