# LangGraph Research RAG Agent

Context-aware RAG chatbot for research paper PDFs. Users upload a PDF, the app extracts text with OpenDataLoader PDF, embeds chunks with Gemini Embedding 2, stores vectors in Qdrant, and answers questions through a LangGraph workflow.

## Features

- PDF upload and ingestion
- arXiv-paper validation before embedding
- OpenDataLoader PDF extraction
- Gemini `gemini-embedding-2` embeddings with 768 dimensions
- Qdrant vector search
- LangChain `create_agent` research agent with LangGraph short-term memory and a RAG tool
- FastAPI backend
- Streaming chat endpoint with graph progress events
- Basic web UI
- Dockerized runtime using `uv`

## Setup

```bash
uv sync
cp .env.example .env
```

Fill `.env`:

```env
GEMINI_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=research_papers
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMENSIONS=768
GEMINI_CHAT_MODEL=gemini-2.5-flash-lite
FALLBACK_FIRST_MODEL=groq:openai/gpt-oss-120b
FALLBACK_ADDITIONAL_MODELS=
GROQ_API_KEY=
DEMO_DOCUMENT_ID=
DEMO_DOCUMENT_FILENAME=
DEMO_DOCUMENT_CHUNKS=0
```

OpenDataLoader PDF requires Java 11+. On Ubuntu/Debian:

```bash
sudo apt install openjdk-21-jdk
```

Run locally:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Security note:

- Do not commit real API keys to the repository. Add secrets to `.env` locally or use a secret manager.
- If an API key is accidentally committed, rotate it immediately.

Open `http://localhost:8000`.

To let reviewers test without uploading and re-embedding a PDF, set the
`DEMO_DOCUMENT_*` variables to a document that is already indexed in your vector
store. The UI will list that paper immediately and users can start chatting with
it directly.

## API

The chat layer is built with `langchain.agents.create_agent`. The agent has one tool, `rag_search`, that retrieves selected arXiv paper chunks from Qdrant. It calls that tool for paper questions and answers memory/model questions directly from the conversation state.

```bash
curl http://localhost:8000/api/health
```

Upload:

```bash
curl -F "file=@paper.pdf" http://localhost:8000/api/documents/upload
```

Only arXiv research paper PDFs are accepted for ingestion. Non-arXiv PDFs are rejected before embedding.

Chat:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main contribution?","document_id":"DOCUMENT_ID","session_id":"demo"}'
```

Streaming chat:

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main contribution?","document_id":"DOCUMENT_ID","session_id":"demo"}'
```

## Docker

Build:

```bash
docker build -t your-dockerhub-user/rag-research-agent:latest .
```

Run:

```bash
docker run --env-file .env -p 8000:8000 your-dockerhub-user/rag-research-agent:latest
```

Push:

```bash
docker push your-dockerhub-user/rag-research-agent:latest
```

Submission should include:

- GitHub repository URL
- Docker image URL
- Docker run command
- Required environment variables
- Demo video showing upload, ingestion, and context-aware chat
