from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = "development"
    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    TAVILY_API_KEY: str = ""
    API_KEY: str = "dev-api-key"
    RATE_LIMIT: str = "10/minute"
    AGENT_TIMEOUT: int = 90
    LOG_LEVEL: str | None = None

    @model_validator(mode="after")
    def set_default_log_level(self) -> "Settings":
        """Set default LOG_LEVEL based on ENVIRONMENT if not explicitly set."""
        if self.LOG_LEVEL is None:
            self.LOG_LEVEL = "DEBUG" if self.ENVIRONMENT == "development" else "INFO"
        return self


settings = Settings()
