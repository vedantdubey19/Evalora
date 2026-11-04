"""Small sequential evaluation runs with per-case failure isolation."""
from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from typing import Annotated, Callable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.core.config import Settings
from app.services.evaluation import evaluate
from app.services.llm_client import ConfigurationError, GeminiClient, ProviderError

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]


class EvalCase(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    question: Text
    reference: Text


def validate_cases(data):
    if not isinstance(data, list) or not 1 <= len(data) <= 100:
        raise ValueError('Dataset must contain between 1 and 100 cases.')
    cases = [EvalCase.model_validate(item) for item in data]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError('Dataset IDs must be unique.')
    return cases


def run_benchmark(cases: list[EvalCase], models: list[str], settings: Settings,
                  provider_factory: Callable = GeminiClient):
    if not models or len(set(models)) != len(models) or any(not m.strip() for m in models):
        raise ValueError('Provide unique, nonempty model IDs.')
    rows = []
    for model in models:
        config = settings.model_copy(update={'gemini_model': model})
        provider = provider_factory(config)
        for case in cases:
            row = {'case_id': case.id, 'model': model, 'judge_model': settings.gemini_judge_model,
                   'question': case.question, 'reference': case.reference,
                   'status': 'generation_error', 'answer': None, 'generation_ms': None,
                   'evaluation_ms': None, 'usage': None, 'evaluation': None}
            started = perf_counter()
            try:
                answer, usage = provider.generate(case.question)
            except ConfigurationError:
                row['status'] = 'configuration_error'
                rows.append(row)
                continue
            except ProviderError:
                rows.append(row)
                continue
            row.update(answer=answer, generation_ms=round((perf_counter() - started) * 1000, 2),
                       usage={'input_tokens': getattr(usage, 'prompt_token_count', None),
                              'output_tokens': getattr(usage, 'candidates_token_count', None),
                              'total_tokens': getattr(usage, 'total_token_count', None)})
            started = perf_counter()
            try:
                row['evaluation'] = evaluate(provider, case.question, answer, case.reference)
                row['status'] = 'ok'
            except (ProviderError, ConfigurationError):
                row['status'] = 'evaluation_error'
            row['evaluation_ms'] = round((perf_counter() - started) * 1000, 2)
            rows.append(row)
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
            'cache_enabled': False, 'summary': summaries, 'results': rows}
