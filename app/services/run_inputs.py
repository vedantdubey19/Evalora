"""Validated inputs shared by dashboard starts and checkpoint resumption."""
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from app.services.benchmark import EvalCase, completed_rows, validate_cases

ModelID = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120,
                                          pattern=r'^[A-Za-z0-9][A-Za-z0-9._/\-]*$')]


class RunOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    models: list[ModelID] = Field(min_length=1, max_length=4)
    judge_model: ModelID
    rubric: Literal['legacy-v1', 'quality-v2'] = 'quality-v2'
    max_retries: int = Field(default=2, ge=0, le=3, strict=True)
    request_interval: float = Field(default=4, ge=0, le=60, allow_inf_nan=False)
    max_retry_delay: float = Field(default=30, ge=0, le=60, allow_inf_nan=False)
    model_intervals: dict[ModelID, Annotated[float, Field(ge=0, le=60, allow_inf_nan=False)]] = Field(default_factory=dict)


def parse_dataset(payload: bytes):
    if len(payload) > 2_000_000:
        raise ValueError('Dataset exceeds the 2 MB upload limit.')
    try:
        raw = json.loads(payload.decode('utf-8'))
    except (UnicodeError, ValueError, RecursionError):
        raise ValueError('Upload a UTF-8 JSON array of evaluation cases.') from None
    if not isinstance(raw, list) or not 1 <= len(raw) <= 100:
        raise ValueError('Dataset must contain between 1 and 100 cases.')
    errors = []
    for index, item in enumerate(raw):
        try:
            EvalCase.model_validate(item)
        except ValidationError as error:
            for detail in error.errors(include_input=False, include_url=False):
                field = '.'.join(str(x) for x in detail['loc']) or 'case'
                errors.append(f'Row {index + 1}, {field}: {detail["msg"]}')
    if errors:
        raise ValueError('\n'.join(errors[:8]))
    return validate_cases(raw)


def settings_for_run(settings, options):
    if len(set(options.models)) != len(options.models):
        raise ValueError('Generation model IDs must be unique.')
    return settings.model_copy(update={'gemini_judge_model': options.judge_model,
                                       'evaluation_rubric': options.rubric})


def validate_run(cases, options, settings, prior=None):
    configured = settings_for_run(settings, options)
    if set(options.model_intervals) - set(options.models) - {options.judge_model}:
        raise ValueError('Pacing overrides must match a selected model or the judge.')
    if not configured.has_api_key:
        raise ValueError('Set GEMINI_API_KEY in the local .env file before starting a run.')
    if options.rubric == 'legacy-v1' and any(c.checks.model_dump(exclude_defaults=True) for c in cases):
        raise ValueError('This dataset has output requirements; choose quality-v2.')
    reused = completed_rows(prior, cases, options.models, configured)
    remaining = len(cases) * len(options.models) - len(reused)
    return configured, remaining * 2 * (1 + options.max_retries), len(reused)


def read_checkpoint(path: Path, directory: Path):
    path, directory = path.resolve(), directory.resolve()
    if path.parent != directory or not path.name.startswith('eval-') or path.suffix != '.json':
        raise ValueError('Select a report from this dashboard’s results folder.')
    if path.stat().st_size > 10_000_000:
        raise ValueError('Checkpoint exceeds the 10 MB limit.')
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(report, dict) or not isinstance(report.get('results'), list) or not report['results']:
            raise ValueError()
        unique, models = {}, []
        for row in report['results']:
            case = EvalCase.model_validate({'id': row['case_id'], 'question': row['question'],
                                           'reference': row['reference'], 'category': row.get('category', 'general'),
                                           'checks': row.get('checks', {})})
            if case.id in unique and unique[case.id].model_dump() != case.model_dump():
                raise ValueError()
            unique[case.id] = case
            if row['model'] not in models:
                models.append(row['model'])
        cases = validate_cases([c.model_dump() for c in unique.values()])
        options = RunOptions(models=models, judge_model=report['judge_model'],
                             rubric=report.get('rubric_version', 'legacy-v1'),
                             max_retries=report.get('max_retries', 2),
                             request_interval=report.get('request_interval_seconds', 4),
                             max_retry_delay=report.get('max_retry_delay', 30),
                             model_intervals=report.get('model_intervals_seconds', {}))
    except (ValueError, KeyError, TypeError, RecursionError):
        raise ValueError('Checkpoint is invalid or cannot reconstruct its original dataset.') from None
    return cases, options, report
