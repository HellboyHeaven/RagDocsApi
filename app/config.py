from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    database_url: str = "postgresql+asyncpg://chat:chat@localhost:5432/chat"
    max_context_tokens: int = 6000
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()