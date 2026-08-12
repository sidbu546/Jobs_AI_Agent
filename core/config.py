from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    database_url: str = "sqlite:///./jobs.db"
    chroma_persist_dir: str = "./.chroma"

    usajobs_api_key: str = ""
    usajobs_user_agent: str = "siddhank@bu.edu"

    # Normalized score (0–1): 0.65+ strong, 0.40–0.65 medium, <0.40 skip
    match_score_threshold: float = 0.40
    max_jobs_per_run: int = 200

    slack_webhook_url: str = ""


settings = Settings()
