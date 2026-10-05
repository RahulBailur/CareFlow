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
    seed_default_password: str = ""
    frontend_build_dir: str = "../frontend/build"

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
