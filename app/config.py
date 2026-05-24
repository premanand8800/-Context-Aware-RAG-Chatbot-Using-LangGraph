from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., alias="QDRANT_API_KEY")
    qdrant_collection: str = Field("research_papers", alias="QDRANT_COLLECTION")
    embedding_model: str = Field("gemini-embedding-2", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(768, alias="EMBEDDING_DIMENSIONS")
    gemini_chat_model: str = Field("gemini-2.5-flash-lite", alias="GEMINI_CHAT_MODEL")
    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8000, alias="APP_PORT")
    data_dir: Path = Path("data")
    demo_document_id: str = Field("", alias="DEMO_DOCUMENT_ID")
    demo_document_filename: str = Field("", alias="DEMO_DOCUMENT_FILENAME")
    demo_document_chunks: int = Field(0, alias="DEMO_DOCUMENT_CHUNKS")
    # Optional model fallback settings (comma-separated for additional models).
    # Example: FALLBACK_FIRST_MODEL="groq:openai/gpt-oss-120b"
    # and FALLBACK_ADDITIONAL_MODELS="groq:llama-3.3-70b-versatile"
    fallback_first_model: str = Field("", alias="FALLBACK_FIRST_MODEL")
    fallback_additional_models: str = Field("", alias="FALLBACK_ADDITIONAL_MODELS")
    groq_api_key: str = Field("", alias="GROQ_API_KEY")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "converted").mkdir(parents=True, exist_ok=True)
    return settings
