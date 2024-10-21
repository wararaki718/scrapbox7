from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class SampleSettings(BaseSettings):
    custom_project_name: str = Field("", alias="CUSTOM_PROJECT_NAME")


class SampleConfig(BaseModel):
    env: SampleSettings = SampleSettings()
    csv_path: Path

    @classmethod
    def load(cls, config_path: Path) -> "SampleConfig":
        return cls(csv_path=config_path)
