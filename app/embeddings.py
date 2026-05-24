import time

from google import genai
from google.genai import types

from app.config import Settings


class GeminiEmbedder:
    batch_size = 100

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def _document_text(self, text: str, title: str | None = None) -> str:
        title_value = title or "none"
        return f"title: {title_value} | text: {text}"

    def _query_text(self, query: str) -> str:
        return f"task: question answering | query: {query}"

    def embed_documents(self, texts: list[str], title: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            if start > 0:
                time.sleep(65)
            batch = texts[start : start + self.batch_size]
            contents = [
                types.Content(parts=[types.Part.from_text(text=self._document_text(text, title))])
                for text in batch
            ]
            result = self.client.models.embed_content(
                model=self.settings.embedding_model,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.settings.embedding_dimensions
                ),
            )
            embeddings.extend(embedding.values for embedding in result.embeddings)
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.settings.embedding_model,
            contents=self._query_text(query),
            config=types.EmbedContentConfig(
                output_dimensionality=self.settings.embedding_dimensions
            ),
        )
        return result.embeddings[0].values
