from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / '.env', extra='ignore', validate_assignment=True, hide_input_in_errors=True)
    gemini_api_key: SecretStr = SecretStr('')
    gemini_model: str = 'gemini-2.5-flash'
    gemini_judge_model: str = 'gemini-2.5-flash'
    evaluation_rubric: Literal['legacy-v1', 'quality-v2'] = 'legacy-v1'
    generation_temperature: float = Field(default=0.7, ge=0, le=2)
    provider_timeout_ms: int = Field(default=60000, ge=1000, le=300000)
    cache_match_mode: Literal['exact', 'guarded'] = 'exact'
    cache_similarity_threshold: float = Field(default=0.95, ge=0, le=1)
    cache_ttl_seconds: int = Field(default=600, ge=1)
    cache_max_entries: int = Field(default=256, ge=1, le=10000)
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'

    deployment_mode: Literal['local', 'production'] = 'local'
    api_access_token: SecretStr = SecretStr('')
    gateway_secret: SecretStr = SecretStr('')
    allowed_hosts: list[str] = Field(default_factory=lambda: ['localhost', '127.0.0.1', 'testserver'])
    reports_dir: Path = BASE_DIR / 'results'
    state_dir: Path = BASE_DIR / 'data'
    api_requests_per_minute: int = Field(default=30, ge=1, le=10000)
    provider_calls_per_day: int = Field(default=200, ge=1, le=100000)
    provider_max_concurrent: int = Field(default=2, ge=1, le=20)
    max_request_bytes: int = Field(default=1000000, ge=1000, le=2000000)

    @model_validator(mode='after')
    def production_requirements(self):
        if self.deployment_mode == 'production':
            if len(self.api_access_token.get_secret_value()) < 32 or len(self.gateway_secret.get_secret_value()) < 32:
                raise ValueError('Production requires independent API_ACCESS_TOKEN and GATEWAY_SECRET of at least 32 characters.')
            if self.api_access_token == self.gateway_secret:
                raise ValueError('Production secrets must be independent.')
            if not self.allowed_hosts or any('*' in host or not host.strip() for host in self.allowed_hosts):
                raise ValueError('Production requires explicit ALLOWED_HOSTS without wildcards.')
            if not self.reports_dir.is_absolute() or not self.state_dir.is_absolute():
                raise ValueError('Production storage paths must be absolute.')
        return self

    @property
    def has_api_key(self) -> bool:
        key = self.gemini_api_key.get_secret_value().strip()
        return bool(key) and key != 'your_actual_api_key'
