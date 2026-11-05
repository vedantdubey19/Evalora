"""Small sequential evaluation runs with per-case failure isolation."""
from datetime import datetime, timezone
import hashlib
import json
import math
from copy import deepcopy
from statistics import mean
from time import perf_counter, monotonic, sleep
from random import uniform
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
        if not isinstance(row.get('case_id'), str):
            raise ValueError('Invalid resumed case ID.')
        case = lookup.get(row['case_id'])
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


FAILURE_STATES = {'generation_error', 'evaluation_error', 'configuration_error', 'interrupted'}
TRANSIENT_REASONS = {'provider_unavailable', 'server_error', 'timeout', 'network_error'}
BLOCKING_REASONS = TRANSIENT_REASONS | {'daily_quota', 'rate_limited', 'model_unavailable'}


def run_benchmark(cases: list[EvalCase], models: list[str], settings: Settings,
                  provider_factory: Callable = GeminiClient, on_result: Callable | None = None,
                  request_interval: float = 0, model_intervals: dict | None = None,
                  prior_report=None, on_checkpoint: Callable | None = None,
                  max_retries: int = 0, max_retry_delay: float = 30):
    if not models or len(set(models)) != len(models) or any(not m.strip() for m in models):
        raise ValueError('Provide unique, nonempty model IDs.')
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 3:
        raise ValueError('Retry budget must be between 0 and 3.')
    if not math.isfinite(max_retry_delay) or not 0 <= max_retry_delay <= 60:
        raise ValueError('Retry delay cap must be between 0 and 60.')
    intervals = model_intervals or {}
    if set(intervals) - set(models) - {settings.gemini_judge_model}:
        raise ValueError('Unknown model interval.')
    if any(not math.isfinite(v) or not 0 <= v <= 60 for v in [request_interval, *intervals.values()]):
        raise ValueError('Intervals must be between 0 and 60.')
    reused = completed_rows(prior_report, cases, models, settings)
    rows = []
    for model in models:
        for case in cases:
            if (case.id, model) in reused:
                row = reused[(case.id, model)]
                row.update(reused=True, generation_attempts=0, evaluation_attempts=0, retries=0)
            else:
                row = {'case_id': case.id, 'category': case.category, 'model': model,
                       'judge_model': settings.gemini_judge_model, 'question': case.question,
                       'reference': case.reference, 'status': 'pending', 'error_reason': None,
                       'answer': None, 'generation_ms': None, 'evaluation_ms': None,
                       'usage': None, 'evaluation': None, 'reused': False,
                       'generation_attempts': 0, 'evaluation_attempts': 0, 'retries': 0}
            rows.append(row)
    created_at = datetime.now(timezone.utc).isoformat()
    next_request, blocked_models = {}, {}
    configuration_blocked = False

    def snapshot(state):
        summaries = []
        for model in models:
            subset = [r for r in rows if r['model'] == model]
            success = [r for r in subset if r['status'] == 'ok']
            generated = [r for r in subset if r['answer'] is not None]
            summaries.append({
                'model': model, 'cases': len(subset), 'evaluated': len(success),
                'failures': sum(r['status'] in FAILURE_STATES for r in subset),
                'skipped': sum(r['status'] == 'skipped' for r in subset),
                'pending': sum(r['status'] == 'pending' for r in subset),
                'reused': sum(r['reused'] for r in subset),
                'provider_attempts': sum(r['generation_attempts'] + r['evaluation_attempts'] for r in subset),
                'retries': sum(r['retries'] for r in subset),
                'mean_score': round(mean(r['evaluation']['overall'] for r in success), 2) if success else None,
                'mean_generation_ms': round(mean(r['generation_ms'] for r in generated), 2) if generated else None,
                'generation_total_tokens': sum((r['usage'] or {}).get('total_tokens') or 0 for r in generated),
            })
        return {'created_at': created_at, 'updated_at': datetime.now(timezone.utc).isoformat(),
                'run_state': state, 'schema_version': 2, 'accounting_version': 'day4-v1',
                'planned_cases': len(rows), 'processed_cases': sum(r['status'] != 'pending' for r in rows),
                'judge_model': settings.gemini_judge_model, 'dataset_sha256': dataset_fingerprint(cases),
                'resumed_rows': len(reused),
                'resumed_from_created_at': prior_report.get('created_at') if prior_report else None,
                'generation_temperature': 0, 'request_interval_seconds': request_interval,
                'model_intervals_seconds': intervals, 'max_retries': max_retries,
                'max_retry_delay': max_retry_delay, 'cache_enabled': False,
                'summary': summaries, 'results': deepcopy(rows)}

    def checkpoint(state='running'):
        report = snapshot(state)
        if on_checkpoint:
            on_checkpoint(report)
        return report

    def publish(row):
        checkpoint()
        if on_result:
            on_result(row)

    class PacedProvider:
        def __init__(self, config):
            self.settings = config
            self.client = provider_factory(config)
            self.last_request_ms = None
            self.last_attempts = 0

        def generate(self, prompt, *, judge=False, schema=None):
            selected = self.settings.gemini_judge_model if judge else self.settings.gemini_model
            self.last_attempts, self.last_request_ms = 0, 0.0
            if selected in blocked_models:
                raise ProviderError(reason=blocked_models[selected])
            for attempt in range(max_retries + 1):
                wait = next_request.get(selected, 0) - monotonic()
                if wait > 0:
                    sleep(wait)
                next_request[selected] = monotonic() + intervals.get(selected, request_interval)
                started = perf_counter()
                self.last_attempts += 1
                delay = None
                try:
                    return self.client.generate(prompt, judge=judge, schema=schema)
                except ConfigurationError:
                    self.last_attempts -= 1
                    raise
                except ProviderError as error:
                    transient = error.reason in TRANSIENT_REASONS
                    # A 429 without a server-supplied retry delay is not assumed temporary.
                    if error.reason == 'rate_limited' and error.retry_after is not None:
                        transient = True
                    if transient and attempt < max_retries:
                        delay = min(2 ** attempt + uniform(0, .25), max_retry_delay)
                        if error.retry_after is not None:
                            if not math.isfinite(error.retry_after) or error.retry_after > max_retry_delay:
                                delay = None
                            else:
                                delay = max(delay, error.retry_after)
                    if delay is None:
                        if error.reason in BLOCKING_REASONS:
                            blocked_models[selected] = error.reason
                        raise
                finally:
                    self.last_request_ms += round((perf_counter() - started) * 1000, 2)
                if delay is not None:
                    sleep(delay)

    # Durable initial state is written before a provider is constructed or called.
    checkpoint()
    try:
        for model in models:
            config = settings.model_copy(update={'gemini_model': model, 'generation_temperature': 0})
            provider = PacedProvider(config)
            for row in [r for r in rows if r['model'] == model]:
                if row['reused']:
                    publish(row)
                    continue
                reason = None
                if configuration_blocked:
                    reason = 'configuration_error'
                elif settings.gemini_judge_model in blocked_models:
                    reason = 'judge_' + blocked_models[settings.gemini_judge_model]
                elif model in blocked_models:
                    reason = blocked_models[model]
                if reason:
                    row.update(status='skipped', error_reason=reason)
                    publish(row)
                    continue
                try:
                    answer, usage = provider.generate(row['question'])
                except ConfigurationError:
                    row.update(status='configuration_error', error_reason='configuration_error')
                    configuration_blocked = True
                except ProviderError as error:
                    row.update(status='generation_error', error_reason=error.reason)
                except KeyboardInterrupt:
                    row.update(status='interrupted', error_reason='interrupted')
                    raise
                finally:
                    row['generation_attempts'] = provider.last_attempts
                    row['retries'] = max(0, provider.last_attempts - 1)
                if row['status'] != 'pending':
                    publish(row)
                    continue
                row.update(answer=answer, generation_ms=provider.last_request_ms,
                           usage={'input_tokens': getattr(usage, 'prompt_token_count', None),
                                  'output_tokens': getattr(usage, 'candidates_token_count', None),
                                  'total_tokens': getattr(usage, 'total_token_count', None)})
                try:
                    row['evaluation'] = evaluate(provider, row['question'], answer, row['reference'])
                    row['status'] = 'ok'
                except (ProviderError, ConfigurationError) as error:
                    row.update(status='evaluation_error', error_reason=getattr(error, 'reason', 'configuration_error'))
                    if isinstance(error, ConfigurationError):
                        configuration_blocked = True
                except KeyboardInterrupt:
                    row.update(status='interrupted', error_reason='interrupted')
                    raise
                finally:
                    row['evaluation_ms'] = provider.last_request_ms
                    row['evaluation_attempts'] = provider.last_attempts
                    row['retries'] += max(0, provider.last_attempts - 1)
                publish(row)
    except KeyboardInterrupt:
        checkpoint('interrupted')
        raise
    return checkpoint('finished')
