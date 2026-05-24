import json

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.documents import DocumentRegistry
from app.embeddings import GeminiEmbedder
from app.graph import ResearchGraph
from app.ingestion import PDFIngestionService
from app.llm import GeminiChat
from app.schemas import ChatRequest, ChatResponse, DocumentInfo, UploadResponse
from app.session_store import SessionStore
from app.vectorstore import QdrantStore

app = FastAPI(title="LangGraph Research RAG Agent")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

settings = get_settings()
embedder = GeminiEmbedder(settings)
vectorstore = QdrantStore(settings)
demo_documents = []
if settings.demo_document_id and settings.demo_document_filename:
    demo_documents.append(
        DocumentInfo(
            document_id=settings.demo_document_id,
            filename=settings.demo_document_filename,
            chunks=settings.demo_document_chunks,
        )
    )
registry = DocumentRegistry(settings.data_dir, seed_documents=demo_documents)
ingestion = PDFIngestionService(settings, embedder, vectorstore, registry)
chat_graph = ResearchGraph(embedder, vectorstore, GeminiChat(settings))
sessions = SessionStore()


def get_app_settings() -> Settings:
    return settings


@app.get("/")
def index():
    return FileResponse("app/static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents():
    return registry.list()


@app.post("/api/documents/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...), _: Settings = Depends(get_app_settings)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    try:
        saved_path = ingestion.save_upload(file)
        info = ingestion.ingest_pdf(saved_path, file.filename)
        return UploadResponse(**info.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        history = sessions.history(request.session_id)
        result = chat_graph.invoke(
            question=request.question,
            history=history,
            session_id=request.session_id,
            document_id=request.document_id,
        )
        sessions.append(request.session_id, "user", request.question)
        sessions.append(request.session_id, "assistant", result["answer"])
        return ChatResponse(
            answer=result["answer"],
            session_id=request.session_id,
            citations=result["citations"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    def event(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def generate():
        try:
            history = sessions.history(request.session_id)
            yield event({"type": "status", "message": "Routing question"})
            final_answer = ""
            final_citations = []
            for chunk in chat_graph.stream(
                question=request.question,
                history=history,
                session_id=request.session_id,
                document_id=request.document_id,
            ):
                node_name, update = next(iter(chunk.items()))
                if "status" in update:
                    yield event({"type": "status", "message": update["status"]})
                elif node_name == "agent" and "answer" in update:
                    final_answer = update.get("answer", "")
                    final_citations = [
                        citation.model_dump() if hasattr(citation, "model_dump") else citation
                        for citation in update.get("citations", [])
                    ]
                    yield event({"type": "answer", "answer": final_answer})

            sessions.append(request.session_id, "user", request.question)
            sessions.append(request.session_id, "assistant", final_answer)
            yield event(
                {
                    "type": "done",
                    "answer": final_answer,
                    "session_id": request.session_id,
                    "citations": final_citations,
                }
            )
        except Exception as exc:
            yield event({"type": "error", "message": f"Chat failed: {exc}"})

    return StreamingResponse(generate(), media_type="text/event-stream")
