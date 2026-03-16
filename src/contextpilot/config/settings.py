from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = Field(default="ContextPilot", alias="PROJECT_NAME")

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    vector_store_dir: str = Field(default="./data/vector_store/faiss", alias="VECTOR_STORE_DIR")
    raw_data_dir: str = Field(default="./data/raw", alias="RAW_DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def vector_store_path(self) -> Path:
        return Path(self.vector_store_dir)

    @property
    def raw_data_path(self) -> Path:
        return Path(self.raw_data_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()