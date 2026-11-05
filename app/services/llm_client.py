from contextlib import nullcontext
import sqlite3

from google import genai
from google.genai import types

from app.core.config import Settings
from app.services.limits import LimitExceeded, LimitStore
from app.services.provider_failures import classify_failure
from app.services.rubrics import QualityScores, judge_instructions


class ConfigurationError(Exception):
    pass


class ProviderError(Exception):
    def __init__(self, message='Gemini request failed.', *, reason='provider_error', retry_after=None):
        super().__init__(message)
        self.reason = reason
        self.retry_after = retry_after


class LocalLimitError(ProviderError):
    def __init__(self, *, reason, retry_after):
        ProviderError.__init__(self, 'Application provider-call limit reached.', reason=reason)
        self.limit_retry_after = retry_after


class GeminiClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, prompt: str, *, judge: bool = False, schema=None):
        key = self.settings.gemini_api_key.get_secret_value().strip()
        if not self.settings.has_api_key:
            raise ConfigurationError('Set GEMINI_API_KEY in the project .env file.')
        model = self.settings.gemini_judge_model if judge else self.settings.gemini_model
        config = types.GenerateContentConfig(
            temperature=0 if judge else self.settings.generation_temperature,
            max_output_tokens=4096,
            response_mime_type='application/json' if schema else 'text/plain',
            response_schema=schema,
            system_instruction=judge_instructions(schema is QualityScores) if judge else None,
        )
        try:
            admission = nullcontext()
            if self.settings.deployment_mode == 'production':
                try:
                    store = LimitStore(self.settings.state_dir)
                except OSError:
                    raise ProviderError(reason='provider_unavailable') from None
                admission = store.provider_slot(
                    self.settings.provider_calls_per_day, self.settings.provider_max_concurrent,
                    self.settings.provider_timeout_ms / 1000)
            with admission, genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=self.settings.provider_timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            )) as client:
                response = client.models.generate_content(model=model, contents=prompt, config=config)
            # Do not return truncated, blocked, or empty answers, or cache them.
            if not response.candidates or str(response.candidates[0].finish_reason).split('.')[-1] != 'STOP':
                raise ProviderError()
            if not response.text or not response.text.strip():
                raise ProviderError()
            return response.text, response.usage_metadata
        except LimitExceeded as error:
            # RetryInfo is deliberately absent: benchmarks stop rather than waiting out local limits.
            raise LocalLimitError(reason=error.reason, retry_after=error.retry_after) from None
        except sqlite3.Error:
            raise ProviderError(reason='provider_unavailable') from None
        except ProviderError:
            raise
        except Exception as error:
            # Provider exceptions can contain request URLs, keys, or user content.
            reason, retry_after = classify_failure(error)
            raise ProviderError(reason=reason, retry_after=retry_after) from None
