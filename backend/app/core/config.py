from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://bpmn_user:bpmn_pass@db:5432/bpmn_platform"
    REDIS_URL: str = "redis://redis:6379/0"
    SECRET_KEY: str = "thesis-bpmn-secret-key-change-in-prod"
    ENVIRONMENT: str = "development"
    ANTHROPIC_API_KEY: str = ""
    CHROMA_DIR: str = "/app/data/chroma"
    RUN_LOG_DIR: str = "/app/data/runs"
    UPLOAD_DIR: str = "/app/data/uploads"

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
