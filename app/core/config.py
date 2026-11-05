from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / '.env', extra='ignore', validate_assignment=True)
    gemini_api_key: SecretStr = SecretStr('')
    gemini_model: str = 'gemini-2.5-flash'
    gemini_judge_model: str = 'gemini-2.5-flash'
    generation_temperature: float = Field(default=0.7, ge=0, le=2)
    provider_timeout_ms: int = Field(default=60000, ge=1000, le=300000)
    cache_match_mode: Literal['exact', 'guarded'] = 'exact'
    cache_similarity_threshold: float = Field(default=0.95, ge=0, le=1)
    cache_ttl_seconds: int = Field(default=600, ge=1)
    cache_max_entries: int = Field(default=256, ge=1, le=10000)
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'

    @property
    def has_api_key(self) -> bool:
        key = self.gemini_api_key.get_secret_value().strip()
        return bool(key) and key != 'your_actual_api_key'
