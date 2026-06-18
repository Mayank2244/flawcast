"""FlowCast AI application configuration."""
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "flowcast123"
    mysql_database: str = "flowcast_ai"
    openweather_api_key: str = ""
    gnews_api_key: str = ""
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    dataset_path: str = "dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
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
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4"
        )

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
