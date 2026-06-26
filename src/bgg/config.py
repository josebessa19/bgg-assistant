from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/bgg/config.py -> project root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: Path | str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (PROJECT_ROOT / resolved).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    bgg_data_dir: Path = Path("data")
    rulebooks_dir: Path = Path("rulebooks")
    mlflow_tracking_uri: str = "http://localhost:5000"
    chroma_host: str = "http://localhost:8001"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"
    mlflow_model_name: str = "bgg-recommender"
    mlflow_model_stage: str = "Production"
    spark_master: str = "local[*]"
    fastapi_url: str = "http://localhost:8000"

    @field_validator("bgg_data_dir", "rulebooks_dir", mode="before")
    @classmethod
    def _resolve_dir(cls, value: Path | str) -> Path:
        return resolve_project_path(value)

    @property
    def processed_dir(self) -> Path:
        return self.bgg_data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        return self.bgg_data_dir / "raw"


settings = Settings()
