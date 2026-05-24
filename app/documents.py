import json
from pathlib import Path

from app.schemas import DocumentInfo


class DocumentRegistry:
    def __init__(self, data_dir: Path, seed_documents: list[DocumentInfo] | None = None):
        self.path = data_dir / "documents.json"
        self.seed_documents = seed_documents or []

    def list(self) -> list[DocumentInfo]:
        documents = {item.document_id: item for item in self.seed_documents}
        if not self.path.exists():
            return list(documents.values())
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for item in raw:
            document = DocumentInfo(**item)
            documents[document.document_id] = document
        return list(documents.values())

    def save(self, document: DocumentInfo) -> None:
        documents = {item.document_id: item for item in self.list()}
        documents[document.document_id] = document
        self.path.write_text(
            json.dumps([item.model_dump() for item in documents.values()], indent=2),
            encoding="utf-8",
        )
