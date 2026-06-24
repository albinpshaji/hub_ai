# CixioHub AI Router

This repository houses the intelligent AI routing and streaming engine for CixioHub. It orchestrates user prompts across multiple specialized Local LLMs (powered by Ollama) and streams responses back to the backend platform natively using Server-Sent Events (SSE).

## 🚀 Features
- **FastAPI Core**: High-performance asynchronous REST API.
- **Dynamic Routing**: Intelligently routes queries to specialized LLMs (General, Coding, Reasoning, Vision) based on confidence matrices.
- **Real-Time Streaming**: Native SSE streaming (`data: {"delta": "token"}`) ensuring lightning-fast perceived response times in the frontend.
- **RAG Embeddings**: Provides dedicated vector embedding generation for semantic search.
- **Containerized Ecosystem**: Fully Dockerized environment connecting the FastAPI service with an Ollama sidecar container.

---

## 🛠️ Getting Started

### Prerequisites
- Docker & Docker Compose V2
- NVIDIA Container Toolkit (Optional but highly recommended for GPU acceleration)
- At least 8GB of System RAM / VRAM (for running `llama3.2:3b` and `qwen2.5-coder:7b`).

### Running the Stack
To start the AI Router and its models, simply run the bootstrapping script:

```bash
./start.sh
```

Or run Docker Compose directly:
```bash
docker compose up --build -d
```

> **Note**: On the very first boot, the `ollama-pull` sidecar container will automatically pull all required models (llama3.2:3b, qwen3.5:latest, qwen2.5-coder:7b, etc.). This may take several minutes depending on your internet connection.

---

## 📡 API Documentation

Once running, the interactive OpenAPI documentation is available at:
👉 **[http://localhost:8003/docs](http://localhost:8003/docs)**

### 1. Chat Stream
Generate AI completions dynamically via SSE streaming.

**Endpoint**: `POST /api/v1/chat/stream`

**Payload**:
```json
{
  "messages": [
    {"role": "user", "content": "Write a quicksort in python"}
  ],
  "user_id": "optional-user-id",
  "use_rag": false
}
```

**Curl Example**:
```bash
curl -N -X POST "http://localhost:8003/api/v1/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Write a quicksort in python"}], "use_rag": false}'
```

### 2. Embeddings
Generate vector embeddings for document storage or RAG retrieval.

**Endpoint**: `POST /api/v1/embed`

**Payload**:
```json
{
  "text": "The meaning of life is 42."
}
```

**Curl Example**:
```bash
curl -X POST "http://localhost:8003/api/v1/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": "The meaning of life is 42."}'
```

### 3. Health Check
Check if the router is alive.

**Endpoint**: `GET /api/v1/health`

**Curl Example**:
```bash
curl http://localhost:8003/api/v1/health
```

---

## 📂 Project Structure

```text
hub_ai/
├── ai_router/
│   ├── app/
│   │   ├── core/      # LLM clients, configurations
│   │   ├── router/    # Routing logic, prompt injection
│   │   └── main.py    # FastAPI entrypoint
│   ├── tests/         # Pytest API integration tests
│   └── Dockerfile     # FastAPI container definition
├── docker-compose.yml # Orchestrates Ollama and the AI Router
├── .env               # Environment configuration
└── start.sh           # Convenience bootstrapper
```

---

## 🧪 Testing

The API includes a test suite built with `pytest` and FastAPI's `TestClient`.

To run the tests inside the running container:
```bash
docker exec cixiohub-ai_router /bin/sh -c "pip install pytest pytest-mock httpx && pytest tests/test_api.py -v"
```
