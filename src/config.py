from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, validator
from pydantic import Field, SecretStr, validator
from typing import List


class AppConfig(BaseSettings):
    # LLM settings
    default_llm_model: str = Field("llama3:latest", description="Default LLM model to use")

    # API keys
    sec_api_key: SecretStr = Field(..., description="SEC API key")
    serper_api_key: SecretStr = Field(..., description="Serper API key")

    # Logging settings
    log_level: str = Field("INFO", description="Logging level")
    log_file_size: int = Field(10_485_760, description="Log file size in bytes")  # 10MB
    log_backup_count: int = Field(5, description="Number of log files to keep")

    # Search settings
    search_result_limit: int = Field(
        100, description="Number of characters to log from search results"
    )

    # SEC Tools settings
    sec_form_types: List[str] = Field(
        ["10-Q", "10-K"], description="SEC form types to search"
    )

    # Embedding settings
    embedding_model: str = Field("llama2", description="Embedding model to use")
    embedding_chunk_size: int = Field(1000, ge=1, description="Chunk size for embeddings")
    embedding_chunk_overlap: int = Field(200, ge=0, description="Chunk overlap for embeddings")

    # Crew settings
    default_crew_process: str = Field(
        "sequential", description="Default process for crew execution"
    )

    @validator("log_level")
    def log_level_must_be_valid(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config() -> AppConfig:
    return AppConfig()


# Global config object
config = load_config()
