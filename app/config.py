"""
CixioHub AI Service — Configuration.

Loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Ollama ────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"

    # ── ChromaDB ──────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8002
    chroma_collection: str = "cixiohub_documents"

    # ── Chunking ──────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Embedding ─────────────────────────────────
    embed_concurrency: int = 5


settings = Settings()
