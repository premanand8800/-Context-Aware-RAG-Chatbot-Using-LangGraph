# Context-Aware RAG Chatbot Using LangGraph

I built this project as a context-aware RAG chatbot for research paper PDFs. It lets me upload or select an already indexed paper, ask questions through a web UI, and get answers grounded in the retrieved document chunks with citations.

## What I built

- FastAPI backend for upload, retrieval, and chat
- LangGraph agent with short-term conversational memory
- RAG tool backed by Qdrant vector search
- Gemini embeddings for document chunking and retrieval
- Streaming chat endpoint for incremental responses
- Simple web UI for document selection and chat
- Dockerized runtime with `uv`
- Optional fallback model support through Groq

## How it works

1. I ingest a PDF and convert it to markdown with OpenDataLoader PDF.
2. I split the document into overlapping chunks.
3. I embed the chunks with `gemini-embedding-2` and store them in Qdrant.
4. I route chat requests through a LangGraph agent.
5. When the question is about the selected paper, the agent calls `rag_search` and answers from retrieved context.
6. The UI can also use a pre-indexed demo paper, so I can test the app without uploading again.

## Requirements

- Python 3.11+
- `uv`
- Java 11+ for OpenDataLoader PDF
- Gemini API key
- Qdrant instance
- Optional Groq API key for fallback models

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env`:

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

If I want to show a pre-indexed paper in the UI, I set the `DEMO_DOCUMENT_*` values to a document already stored in Qdrant.

## Run Locally

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

## API

Health check:

```bash
curl http://localhost:8000/api/health
```

List documents:

```bash
curl http://localhost:8000/api/documents
```

Upload a paper:

```bash
curl -F "file=@paper.pdf" http://localhost:8000/api/documents/upload
```

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

Build the image:

```bash
docker build -t premanandpathak/rag-research-agent:latest .
```

Run the container:

```bash
docker run --env-file .env -p 8000:8000 premanandpathak/rag-research-agent:latest
```

Docker Hub image:

```text
https://hub.docker.com/r/premanandpathak/rag-research-agent
```

## Demo Video

I included a demo video with the repository here:

```text
Rag_Agent.mp4
```

## Notes

- I kept `.env`, runtime data, caches, and session folders out of Git.
- I use a pre-indexed paper in the UI for testing when I want to avoid another upload and re-embedding pass.
- The default chat model is `gemini-2.5-flash-lite` to keep request usage lower.
