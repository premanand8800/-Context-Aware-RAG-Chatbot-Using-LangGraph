from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int


class UploadResponse(DocumentInfo):
    status: str = "ingested"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    document_id: str | None = None
    session_id: str = "default"


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float | None = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    citations: list[Citation] = []

