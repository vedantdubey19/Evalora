from time import perf_counter

from fastapi import FastAPI, HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from app.security.http import AdmissionMiddleware
from app.services.limits import LimitExceeded
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated

from app.core.config import Settings
from app.services.cache import SemanticCache, SentenceEncoder
from app.services.evaluation import evaluate
from app.services.answer_checks import AnswerChecks, check_answer
from app.services.llm_client import ConfigurationError, GeminiClient, ProviderError, LocalLimitError

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: Text
    use_cache: bool = False


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: Text
    answer: Text
    reference: Text | None = None


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer: str = Field(min_length=1, max_length=20000)
    checks: AnswerChecks = Field(default_factory=AnswerChecks)


class QualityEvaluationRequest(CheckRequest):
    question: Text
    reference: Text | None = None


def create_app(settings=None, provider=None, cache=None):
    settings = settings or Settings()
    provider = provider or GeminiClient(settings)
    cache = cache or SemanticCache(
        SentenceEncoder(settings.embedding_model), settings.cache_similarity_threshold,
        settings.cache_ttl_seconds, settings.cache_max_entries, match_mode=settings.cache_match_mode,
    )
    production = settings.deployment_mode == 'production'
    app = FastAPI(title='Evalora', version='0.2.0',
                  docs_url=None if production else '/docs',
                  redoc_url=None if production else '/redoc',
                  openapi_url=None if production else '/openapi.json')
    app.add_middleware(AdmissionMiddleware, settings=settings)
    if production:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    @app.exception_handler(LimitExceeded)
    async def limit_exceeded(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': str(error)}, status_code=429,
                            headers={'Retry-After': str(error.retry_after)})


    @app.get('/')
    def home():
        return {'message': 'Evalora API is running'}

    @app.get('/health')
    def health():
        return {'status': 'ok'} if production else {'status': 'ok', 'gemini_configured': settings.has_api_key}

    @app.post('/generate')
    def generate(request: QuestionRequest):
        started = perf_counter()
        hit = None
        cache_status = 'disabled'
        if request.use_cache:
            try:
                hit = cache.get(request.question, settings.gemini_model)
                cache_status = 'hit' if hit else 'miss'
            except Exception:
                cache_status = 'unavailable'
        usage = None
        if hit:
            answer = hit[0]
        else:
            try:
                answer, usage = provider.generate(request.question)
            except LocalLimitError as error:
                raise HTTPException(429, str(error), headers={'Retry-After': str(error.limit_retry_after)}) from None
            except ConfigurationError:
                raise HTTPException(503, 'Set GEMINI_API_KEY in the project .env file.') from None
            except ProviderError:
                raise HTTPException(502, 'Gemini request failed. Check API access, quota, and model configuration.') from None
            if request.use_cache and cache_status != 'unavailable':
                try:
                    cache.put(request.question, settings.gemini_model, answer)
                except Exception:
                    cache_status = 'unavailable'
        return {
            'question': request.question, 'model': settings.gemini_model, 'answer': answer,
            'cache_status': cache_status, 'cache_match_mode': cache.match_mode, 'similarity': hit[1] if hit else None,
            'latency_ms': round((perf_counter() - started) * 1000, 2),
            'usage': {'input_tokens': getattr(usage, 'prompt_token_count', None),
                      'output_tokens': getattr(usage, 'candidates_token_count', None),
                      'total_tokens': getattr(usage, 'total_token_count', None)} if usage else None,
        }

    @app.post('/evaluate')
    def judge(request: EvaluationRequest):
        try:
            return evaluate(provider, request.question, request.answer, request.reference)
        except LocalLimitError as error:
            raise HTTPException(429, str(error), headers={'Retry-After': str(error.limit_retry_after)}) from None
        except ConfigurationError:
            raise HTTPException(503, 'Set GEMINI_API_KEY in the project .env file.') from None
        except ProviderError:
            raise HTTPException(502, 'Gemini evaluation failed.') from None

    @app.post('/check')
    def check(request: CheckRequest):
        return check_answer(request.answer, request.checks)

    @app.post('/evaluate/quality')
    def quality(request: QualityEvaluationRequest):
        try:
            return evaluate(provider, request.question, request.answer, request.reference,
                            rubric='quality-v2', checks=request.checks)
        except LocalLimitError as error:
            raise HTTPException(429, str(error), headers={'Retry-After': str(error.limit_retry_after)}) from None
        except ConfigurationError:
            raise HTTPException(503, 'Set GEMINI_API_KEY in the project .env file.') from None
        except ProviderError:
            raise HTTPException(502, 'Gemini evaluation failed.') from None

    from app.services.report_repository import ReportRepository
    repository = ReportRepository(settings.reports_dir)

    @app.get('/reports')
    def reports():
        return {'reports': repository.list_reports()}

    @app.get('/reports/{name}')
    def download_report(name: str):
        from fastapi.responses import Response
        try:
            data, media_type = repository.read(name)
        except (OSError, ValueError):
            raise HTTPException(404, 'Report not found.') from None
        return Response(data, media_type=media_type,
                        headers={'Content-Disposition': f'attachment; filename="{name}"'})

    return app


app = create_app()
