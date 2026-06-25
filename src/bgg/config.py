from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    bgg_data_dir: Path = Path("./data")
    rulebooks_dir: Path = Path("./rulebooks")
    mlflow_tracking_uri: str = "http://localhost:5000"
    chroma_host: str = "http://localhost:8001"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"
    mlflow_model_name: str = "bgg-recommender"
    mlflow_model_stage: str = "Production"
    spark_master: str = "local[*]"
    fastapi_url: str = "http://localhost:8000"

    @property
    def processed_dir(self) -> Path:
        return self.bgg_data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        return self.bgg_data_dir / "raw"


settings = Settings()
