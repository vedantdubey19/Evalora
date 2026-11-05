"""Load saved evaluation reports without reading application configuration."""
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, StrictBool, ValidationError


class CheckOutcome(BaseModel):
    check: str
    passed: StrictBool
    detail: str


class Evaluation(BaseModel):
    rubric_version: Literal['legacy-v1', 'quality-v2'] = 'legacy-v1'
    instruction_following: int | None = Field(default=None, ge=1, le=5)
    completeness: int | None = Field(default=None, ge=1, le=5)
    checks_passed: StrictBool | None = None
    deterministic_checks: list[CheckOutcome] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
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
    checks: dict = Field(default_factory=dict)
    model: str
    judge_model: str
    question: str
    reference: str
    error_reason: str | None = None
    status: Literal['ok', 'generation_error', 'configuration_error', 'evaluation_error', 'pending', 'skipped', 'interrupted']
    reused: bool = False
    generation_attempts: int | None = Field(default=None, ge=0)
    evaluation_attempts: int | None = Field(default=None, ge=0)
    retries: int | None = Field(default=None, ge=0)
    answer: str | None = None
    generation_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    evaluation_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    usage: Usage | None = None
    evaluation: Evaluation | None = None


class Report(BaseModel):
    rubric_version: Literal['legacy-v1', 'quality-v2'] = 'legacy-v1'
    run_state: Literal['running', 'finished', 'interrupted'] = 'finished'
    accounting_version: str = 'legacy'
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
                if row.status == 'ok' and row.evaluation.rubric_version != report.rubric_version:
                    raise ValueError('Inconsistent report rubric')
                if row.status == 'ok' and report.rubric_version == 'quality-v2':
                    if row.evaluation.rubric_version != 'quality-v2' or row.evaluation.instruction_following is None or row.evaluation.completeness is None:
                        raise ValueError('Incomplete quality-v2 result')
            for row in report.results:
                item = row.model_dump(exclude={'usage', 'evaluation'})
                item.update(run=path.stem, created_at=report.created_at, run_state=report.run_state,
                            accounting_version=report.accounting_version, rubric_version=report.rubric_version,
                            total_tokens=row.usage.total_tokens if row.usage else None,
                            correctness=None, relevance=None, clarity=None, instruction_following=None, completeness=None,
                            checks_passed=None, deterministic_checks=[], limitations=[], overall=None, reasoning=None)
                if row.evaluation is not None and row.status == 'ok':
                    item.update(row.evaluation.model_dump())
                records.append(item)
        except (OSError, ValueError, ValidationError):
            skipped.append(path.name)
    return pd.DataFrame(records), skipped


def summarize(frame):
    rows = []
    for (run, model, judge, rubric), group in frame.groupby(['run', 'model', 'judge_model', 'rubric_version'], sort=False):
        success = group[group.status == 'ok']
        rows.append({'Run': run, 'Model': model, 'Judge': judge, 'Rubric': rubric,
                     'Cases': len(group), 'Evaluated': len(success),
                     'Failures': int(group.status.isin(['generation_error', 'evaluation_error', 'configuration_error', 'interrupted']).sum()),
                     'Skipped': int((group.status == 'skipped').sum()),
                     'Pending': int((group.status == 'pending').sum()),
                     'Reused': int(group.reused.sum()),
                     'Provider attempts': (group.generation_attempts + group.evaluation_attempts).sum(min_count=1),
                     'Retries': group.retries.sum(min_count=1),
                     'Checks failed': int((success.checks_passed == False).sum()),
                     'Checked cases': int(success.checks_passed.notna().sum()),
                     'Mean score': success.overall.mean() if len(success) else None,
                     'Generation ms': pd.to_numeric(group.generation_ms, errors='coerce').mean(),
                     'Generation tokens': group.total_tokens.sum(min_count=1)})
    return pd.DataFrame(rows)
