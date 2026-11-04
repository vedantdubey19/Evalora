"""Small sequential evaluation runs with per-case failure isolation."""
from datetime import datetime, timezone
import hashlib
import json
import math
from copy import deepcopy
from statistics import mean
from time import perf_counter, monotonic, sleep
from typing import Annotated, Callable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.core.config import Settings
from app.services.evaluation import Scores, evaluate
from app.services.llm_client import ConfigurationError, GeminiClient, ProviderError

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]


class EvalCase(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    category: str = Field(default='general', min_length=1, max_length=100)
    question: Text
    reference: Text


def validate_cases(data):
    if not isinstance(data, list) or not 1 <= len(data) <= 100:
        raise ValueError('Dataset must contain between 1 and 100 cases.')
    cases = [EvalCase.model_validate(item) for item in data]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError('Dataset IDs must be unique.')
    return cases


def dataset_fingerprint(cases):
    return hashlib.sha256(json.dumps([c.model_dump() for c in cases], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def completed_rows(prior, cases, models, settings):
    if prior is None:
        return {}
    if not isinstance(prior, dict) or (
        prior.get('dataset_sha256') != dataset_fingerprint(cases)
        or prior.get('judge_model') != settings.gemini_judge_model
        or prior.get('generation_temperature') != 0
        or prior.get('cache_enabled') is not False
        or not isinstance(prior.get('results'), list)
    ):
        raise ValueError('Resume report does not match this dataset, judge, or generation settings.')
    lookup = {c.id: c for c in cases}
    completed = {}
    for row in prior['results']:
        if not isinstance(row, dict):
            raise ValueError('Invalid resume row.')
        if row.get('status') != 'ok' or row.get('model') not in models:
            continue
        case = lookup.get(row.get('case_id'))
        if case is None or any(row.get(k) != getattr(case, k) for k in ['question', 'reference', 'category']):
            raise ValueError('Resume row does not match the dataset.')
        if row.get('judge_model') != settings.gemini_judge_model or not isinstance(row.get('answer'), str) or not row['answer'].strip():
            raise ValueError('Invalid completed resume row.')
        evaluation = row.get('evaluation')
        if not isinstance(evaluation, dict):
            raise ValueError('Resume row has no valid evaluation.')
        scores = Scores.model_validate(evaluation)
        expected = round((scores.correctness + scores.relevance + scores.clarity) / 3, 2)
        if evaluation.get('overall') != expected:
            raise ValueError('Inconsistent resumed score.')
        for field in ['generation_ms', 'evaluation_ms']:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid resumed latency.')
        usage = row.get('usage')
        if not isinstance(usage, dict) or 'total_tokens' not in usage or any(v is not None and (not isinstance(v, int) or v < 0) for v in usage.values()):
            raise ValueError('Invalid resumed usage.')
        key = (case.id, row['model'])
        if key in completed:
            raise ValueError('Duplicate completed resume row.')
        completed[key] = deepcopy(row)
    return completed


def run_benchmark(cases: list[EvalCase], models: list[str], settings: Settings,
                  provider_factory: Callable = GeminiClient, on_result: Callable | None = None,
                  request_interval: float = 0, model_intervals: dict | None = None, prior_report=None):
    if not models or len(set(models)) != len(models) or any(not m.strip() for m in models):
        raise ValueError('Provide unique, nonempty model IDs.')
    reused = completed_rows(prior_report, cases, models, settings)
    rows = []
    next_request = {}
    blocked_models = {}
    intervals = model_intervals or {}

    class PacedProvider:
        def __init__(self, config):
            self.settings = config
            self.client = provider_factory(config)
            self.last_request_ms = None

        def generate(self, prompt, *, judge=False, schema=None):
            selected = self.settings.gemini_judge_model if judge else self.settings.gemini_model
            if selected in blocked_models:
                raise ProviderError(reason=blocked_models[selected])
            wait = next_request.get(selected, 0) - monotonic()
            if wait > 0:
                sleep(wait)
            next_request[selected] = monotonic() + intervals.get(selected, request_interval)
            started_call = perf_counter()
            try:
                return self.client.generate(prompt, judge=judge, schema=schema)
            except ProviderError as error:
                if error.reason in {'rate_limited', 'model_unavailable', 'provider_unavailable'}:
                    blocked_models[selected] = error.reason
                raise
            finally:
                self.last_request_ms = round((perf_counter() - started_call) * 1000, 2)
    for model in models:
        config = settings.model_copy(update={'gemini_model': model, 'generation_temperature': 0})
        provider = PacedProvider(config)
        for case in cases:
            if (case.id, model) in reused:
                row = reused[(case.id, model)]
                row['reused'] = True
                rows.append(row)
                if on_result:
                    on_result(row)
                continue
            row = {'case_id': case.id, 'category': case.category, 'model': model, 'judge_model': settings.gemini_judge_model,
                   'question': case.question, 'reference': case.reference,
                   'status': 'generation_error', 'error_reason': None, 'answer': None, 'generation_ms': None,
                   'evaluation_ms': None, 'usage': None, 'evaluation': None}
            if settings.gemini_judge_model in blocked_models:
                row['error_reason'] = 'judge_' + blocked_models[settings.gemini_judge_model]
                rows.append(row)
                if on_result:
                    on_result(row)
                continue
            try:
                answer, usage = provider.generate(case.question)
            except ConfigurationError:
                row['status'] = 'configuration_error'
                rows.append(row)
                if on_result:
                    on_result(row)
                continue
            except ProviderError as error:
                row['error_reason'] = error.reason
                rows.append(row)
                if on_result:
                    on_result(row)
                continue
            row.update(answer=answer, generation_ms=provider.last_request_ms,
                       usage={'input_tokens': getattr(usage, 'prompt_token_count', None),
                              'output_tokens': getattr(usage, 'candidates_token_count', None),
                              'total_tokens': getattr(usage, 'total_token_count', None)})
            try:
                row['evaluation'] = evaluate(provider, case.question, answer, case.reference)
                row['status'] = 'ok'
            except (ProviderError, ConfigurationError) as error:
                row['status'] = 'evaluation_error'
                row['error_reason'] = getattr(error, 'reason', 'configuration_error')
            row['evaluation_ms'] = provider.last_request_ms
            rows.append(row)
            if on_result:
                on_result(row)
    summaries = []
    for model in models:
        subset = [r for r in rows if r['model'] == model]
        success = [r for r in subset if r['status'] == 'ok']
        generated = [r for r in subset if r['generation_ms'] is not None]
        summaries.append({
            'model': model, 'cases': len(subset), 'evaluated': len(success),
            'failures': len(subset) - len(success),
            'mean_score': round(mean(r['evaluation']['overall'] for r in success), 2) if success else None,
            'mean_generation_ms': round(mean(r['generation_ms'] for r in generated), 2) if generated else None,
            'generation_total_tokens': sum(r['usage']['total_tokens'] or 0 for r in generated),
        })
    return {'created_at': datetime.now(timezone.utc).isoformat(),
            'judge_model': settings.gemini_judge_model,
            'dataset_sha256': dataset_fingerprint(cases),
            'resumed_rows': len(reused),
            'resumed_from_created_at': prior_report.get('created_at') if prior_report else None,
            'generation_temperature': 0, 'request_interval_seconds': request_interval,
            'model_intervals_seconds': intervals, 'cache_enabled': False, 'summary': summaries, 'results': rows}
