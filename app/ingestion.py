import hashlib
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import opendataloader_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct

from app.config import Settings
from app.documents import DocumentRegistry
from app.embeddings import GeminiEmbedder
from app.schemas import DocumentInfo
from app.vectorstore import QdrantStore


class PDFIngestionService:
    def __init__(
        self,
        settings: Settings,
        embedder: GeminiEmbedder,
        vectorstore: QdrantStore,
        registry: DocumentRegistry,
    ):
        self.settings = settings
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.registry = registry
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest_pdf(self, source_path: Path, original_filename: str) -> DocumentInfo:
        document_id = self._document_id(source_path, original_filename)
        markdown = self._extract_markdown(source_path)
        self._validate_arxiv_paper(markdown, original_filename)
        chunks = self.splitter.split_text(markdown)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        embeddings = self.embedder.embed_documents(chunks, title=original_filename)

        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True)):
            point_id = self._point_id(document_id, index)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "filename": original_filename,
                        "chunk_index": index,
                        "text": chunk,
                    },
                )
            )
        self.vectorstore.upsert_chunks(points)

        info = DocumentInfo(
            document_id=document_id,
            filename=original_filename,
            chunks=len(chunks),
        )
        self.registry.save(info)
        return info

    def _extract_markdown(self, source_path: Path) -> str:
        output_root = self.settings.data_dir / "converted"
        with tempfile.TemporaryDirectory(dir=output_root) as tmp:
            output_dir = Path(tmp)
            opendataloader_pdf.convert(
                input_path=[str(source_path)],
                output_dir=str(output_dir),
                format="markdown",
                quiet=True,
            )
            markdown_files = sorted(output_dir.rglob("*.md"))
            if not markdown_files:
                raise RuntimeError("OpenDataLoader did not produce markdown output")
            return "\n\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in markdown_files)

    def _validate_arxiv_paper(self, markdown: str, filename: str) -> None:
        arxiv_id = re.compile(r"(arxiv:|arxiv\.org|arxiv\s+preprint|\b\d{4}\.\d{4,5}v?\d*\b)", re.I)
        if arxiv_id.search(markdown) or arxiv_id.search(filename):
            return
        raise ValueError("Only arXiv research paper PDFs are allowed for ingestion.")

    def save_upload(self, upload_file) -> Path:
        suffix = Path(upload_file.filename or "paper.pdf").suffix or ".pdf"
        target = self.settings.data_dir / "uploads" / f"{uuid.uuid4().hex}{suffix}"
        with target.open("wb") as handle:
            shutil.copyfileobj(upload_file.file, handle)
        return target

    def _document_id(self, source_path: Path, filename: str) -> str:
        digest = hashlib.sha256()
        digest.update(filename.encode("utf-8"))
        digest.update(source_path.read_bytes())
        return digest.hexdigest()[:16]

    def _point_id(self, document_id: str, chunk_index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}"))
