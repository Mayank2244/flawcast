"""FlowCast AI application configuration."""
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # Database — SQLite (zero-config)
    database_path: str = "database/flowcast.db"

    # MapMyIndia (Mappls) API — competition partner
    mapmyindia_client_id: str = ""
    mapmyindia_client_secret: str = ""

    # Optional external APIs
    openweather_api_key: str = ""
    gnews_api_key: str = ""

    # App settings
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    dataset_path: str = "data/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    models_dir: str = "models"

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def resolved_models_dir(self) -> str:
        p = Path(self.models_dir)
        return str(p if p.is_absolute() else PROJECT_ROOT / p)

    @property
    def resolved_dataset_path(self) -> Path:
        p = Path(self.dataset_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def has_mapmyindia(self) -> bool:
        return bool(self.mapmyindia_client_id and self.mapmyindia_client_secret)

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
