from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.6
    service_name: str = "text-classifier-api"
    version: str = "v1"

settings = Settings()