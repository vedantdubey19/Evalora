"""Load saved evaluation reports without reading application configuration."""
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


class Evaluation(BaseModel):
    correctness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    overall: float = Field(ge=1, le=5)
    reasoning: str


class Usage(BaseModel):
    total_tokens: int | None = Field(default=None, ge=0)


class Result(BaseModel):
    case_id: str
    category: str = 'general'
    model: str
    judge_model: str
    question: str
    reference: str
    error_reason: str | None = None
    status: Literal['ok', 'generation_error', 'configuration_error', 'evaluation_error']
    answer: str | None = None
    generation_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    evaluation_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    usage: Usage | None = None
    evaluation: Evaluation | None = None


class Report(BaseModel):
    created_at: str
    judge_model: str
    results: list[Result] = Field(min_length=1)


def load_reports(directory: Path):
    records, skipped = [], []
    for path in sorted(directory.glob('eval-*.json'), reverse=True):
        try:
            if path.stat().st_size > 10_000_000:
                raise ValueError('Report too large')
            report = Report.model_validate(json.loads(path.read_text(encoding='utf-8')))
            for row in report.results:
                if row.status == 'ok' and (row.evaluation is None or row.answer is None):
                    raise ValueError('Incomplete successful result')
            for row in report.results:
                item = row.model_dump(exclude={'usage', 'evaluation'})
                item.update(run=path.stem, created_at=report.created_at,
                            total_tokens=row.usage.total_tokens if row.usage else None,
                            correctness=None, relevance=None, clarity=None, overall=None, reasoning=None)
                if row.evaluation is not None and row.status == 'ok':
                    item.update(row.evaluation.model_dump())
                records.append(item)
        except (OSError, ValueError, ValidationError):
            skipped.append(path.name)
    return pd.DataFrame(records), skipped


def summarize(frame):
    rows = []
    for (run, model, judge), group in frame.groupby(['run', 'model', 'judge_model'], sort=False):
        success = group[group.status == 'ok']
        rows.append({'Run': run, 'Model': model, 'Judge': judge,
                     'Cases': len(group), 'Evaluated': len(success),
                     'Failures': int((group.status != 'ok').sum()),
                     'Mean score': success.overall.mean() if len(success) else None,
                     'Generation ms': group.generation_ms.mean(),
                     'Generation tokens': group.total_tokens.sum(min_count=1)})
    return pd.DataFrame(rows)
