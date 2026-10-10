from functools import lru_cache
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_JWT_SECRET = "change-me-to-a-long-random-string"  # noqa: S105


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    environment: Literal["development", "test", "demo"] = "development"
    jwt_secret: str
    jwt_expiry_hours: int = 24
    mongo_uri: str = "mongodb://localhost:27017/careflow"
    use_mock_db: bool = False
    bcrypt_rounds: int = 12
    rate_limit_auth: str = "10/minute"
    rate_limit_chat: str = "20/minute"
    llm_provider: Literal["gemini", "ollama", "mock"] = "mock"
    gemini_api_key: str = ""
    gemini_text_model: str = ""
    gemini_tts_model: str = ""
    # Voice. The safe defaults need no model; real runs set these in .env
    stt_provider: Literal["faster_whisper", "mock"] = "mock"
    tts_provider: Literal["gemini", "piper", "mock"] = "mock"
    piper_voice: str = "en_US-lessac-medium"
    vad_provider: Literal["silero", "energy"] = "energy"
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_cpu_threads: int = 0  # 0 lets CTranslate2 choose (4 at most)
    vad_silence_ms: int = 500
    tts_chunking: Literal["sentence", "reply"] = "sentence"
    rate_limit_voice_sessions: str = "5/minute"
    seed_default_password: str = ""
    frontend_build_dir: str = "../frontend/build"
    # Empty switches the embedding stage of the intent classifier off
    intent_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    @model_validator(mode="after")
    def _check_runtime(self) -> Self:
        if self.use_mock_db and self.environment != "test":
            raise ValueError("USE_MOCK_DB=true is only allowed when ENVIRONMENT=test")
        if self.environment == "demo" and self.jwt_secret == PLACEHOLDER_JWT_SECRET:
            raise ValueError("JWT_SECRET must be changed from the placeholder for the demo")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
