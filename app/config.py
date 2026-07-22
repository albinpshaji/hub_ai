from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Application
    app_name: str = "CixioHub AI Service"
    debug: bool = False

    # AI / LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_vision_model: str = "qwen3-vl:2b"
    enable_vision_rag: bool = True
    enable_web_search: bool = False
    tavily_api_key: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "user_documents"
    enable_reranker: bool = False
    reranker_model: str = "mixedbread-ai/mxbai-rerank-base-v1"
    reranker_device: str = "cuda"
    ai_service_url: str = "http://localhost:8003"
    ai_api_key: str | None = None



settings = Settings()
