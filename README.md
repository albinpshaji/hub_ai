# CixioHub AI Microservice (`hub_ai`)

This is the standalone FastAPI microservice for CixioHub that handles AI/ML tasks including LLM chat streaming, document processing, vision RAG, vector embedding generation, and web search integrations.

## Features
- **FastAPI** microservice on port `8003`
- **Ollama Integration** for local LLM inference and embeddings
- **Qdrant Vector Database** integration for vector storage and semantic search
- **Vision RAG & PDF Processing** using PyMuPDF and vision models
- **Reranking Support** via `sentence-transformers` / HuggingFace models

---

## Prerequisites
- **Python 3.10+**
- **Ollama** (Required for LLMs and embedding generation)
- **Qdrant** (Required for vector storage)

---

## Setup & Running Instructions

### 1. Navigate to the Directory
```bash
cd hub_ai
```

### 2. Create and Activate Virtual Environment

* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* **Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .\.venv\Scripts\Activate.ps1
  ```

---

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:

* **Linux / macOS:**
  ```bash
  cp .env.example .env
  ```

* **Windows (CMD / PowerShell):**
  ```cmd
  copy .env.example .env
  ```

Ensure your `.env` has the correct service URLs:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_VISION_MODEL=qwen3-vl:2b
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=user_documents
```

---

### 5. Start Dependent Local Services

Make sure Ollama and Qdrant are running:

1. **Ollama:**
   ```bash
   ollama serve
   ```
   *Pull required models:*
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ollama pull qwen3-vl:2b
   ```

2. **Qdrant (via Docker):**
   ```bash
   docker run -d --name local-qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

---

### 6. Start the AI Microservice

Run the Uvicorn development server on port **8003**:

* **Linux / macOS / Windows:**
  ```bash
  uvicorn app.main:app --port 8003 --reload
  ```

---

## Alternative Setup: Using Docker

To run the AI service using Docker Compose:

```bash
docker compose up --build
```

---

## API Documentation & Verification

Once running, access the interactive Swagger API documentation at:
- **API Docs:** [http://localhost:8003/docs](http://localhost:8003/docs)
- **Health Check:** [http://localhost:8003/openapi.json](http://localhost:8003/openapi.json)
