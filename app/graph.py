import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.embeddings import GeminiEmbedder
from app.llm import GeminiChat
from app.schemas import Citation
from app.vectorstore import QdrantStore


class ResearchGraph:
    def __init__(self, embedder: GeminiEmbedder, vectorstore: QdrantStore, llm: GeminiChat):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        self.agent = self._build_agent()

    def invoke(
        self,
        question: str,
        history: list[dict[str, str]],
        session_id: str,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": self._user_content(question, document_id)}]},
            config={"configurable": {"thread_id": session_id}},
        )
        answer = self._last_ai_text(result)
        return {
            "answer": answer,
            "citations": self._citations_from_messages(result),
        }

    def stream(
        self,
        question: str,
        history: list[dict[str, str]],
        session_id: str,
        document_id: str | None = None,
    ):
        yield {"agent": {"status": "Running research agent"}}
        messages = []
        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": self._user_content(question, document_id)}]},
            config={"configurable": {"thread_id": session_id}},
            stream_mode="updates",
        ):
            if "model" in chunk:
                model_messages = chunk["model"].get("messages", [])
                messages.extend(model_messages)
                if model_messages and getattr(model_messages[-1], "tool_calls", None):
                    yield {"agent": {"status": "Calling RAG tool"}}
            if "tools" in chunk:
                messages.extend(chunk["tools"].get("messages", []))
                yield {"tools": {"status": "Retrieved paper context"}}

        if not messages:
            yield {"agent": {"answer": "I could not generate an answer.", "citations": []}}
            return

        state = {"messages": messages}
        yield {
            "agent": {
                "answer": self._last_ai_text(state),
                "citations": self._citations_from_messages(state),
            }
        }

    def _build_agent(self):
        @tool("rag_search")
        def rag_search(query: str, document_id: str) -> str:
            """Search the selected arXiv paper for evidence relevant to a research question.

            Args:
                query: The standalone research question to search for.
                document_id: The selected ingested arXiv paper document ID.
            """
            query_vector = self.embedder.embed_query(query)
            hits = self.vectorstore.search(query_vector, document_id=document_id, limit=6)
            results = []
            for index, item in enumerate(hits, start=1):
                payload = item["payload"]
                results.append(
                    {
                        "citation_id": index,
                        "document_id": payload.get("document_id", ""),
                        "filename": payload.get("filename", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                        "score": item.get("score"),
                        "text": payload.get("text", ""),
                    }
                )
            return json.dumps({"results": results})

        # Use the LangChain model factory to get primary and fallback model specs
        from app.llm import get_langchain_model_and_fallbacks

        primary_model, fallback_first, additional_fallbacks = get_langchain_model_and_fallbacks(
            self.llm.settings
        )

        middleware = []
        if fallback_first:
            middleware.append(ModelFallbackMiddleware(fallback_first, *additional_fallbacks))

        if primary_model is None and not middleware:
            raise RuntimeError(
                "No primary LangChain model available and no fallbacks configured. "
                "Install `langchain_google_genai` or set FALLBACK_FIRST_MODEL / FALLBACK_ADDITIONAL_MODELS in the environment."
            )

        return create_agent(
            primary_model,
            tools=[rag_search],
            checkpointer=InMemorySaver(),
            middleware=middleware or None,
            system_prompt=(
                "You are a research assistant for uploaded arXiv papers. "
                "Use the rag_search tool only when the user asks about the selected paper. "
                "Do not use rag_search for conversation-memory questions, the user's name, "
                "or questions about what model you are. "
                f"When asked what model you are, answer that you are configured to use "
                f"{self.llm.settings.gemini_chat_model}. "
                "When you use rag_search, answer only from tool results and cite chunks inline "
                "using [1], [2], etc. If the tool results do not support the answer, say the "
                "paper context does not contain enough information."
            ),
        )

    def _user_content(self, question: str, document_id: str | None) -> str:
        if document_id:
            return f"Selected document_id: {document_id}\n\nUser question: {question}"
        return f"No document is selected.\n\nUser question: {question}"

    def _last_ai_text(self, result: dict[str, Any]) -> str:
        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return self._content_to_text(message.content)
        return "I could not generate an answer."

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part for part in parts if part).strip()
        return str(content)

    def _citations_from_messages(self, result: dict[str, Any]) -> list[Citation]:
        citations: list[Citation] = []
        for message in result.get("messages", []):
            if not isinstance(message, ToolMessage):
                continue
            try:
                payload = json.loads(str(message.content))
            except json.JSONDecodeError:
                continue
            for item in payload.get("results", []):
                citations.append(
                    Citation(
                        document_id=item.get("document_id", ""),
                        filename=item.get("filename", ""),
                        chunk_index=item.get("chunk_index", 0),
                        score=item.get("score"),
                        text=item.get("text", "")[:500],
                    )
                )
        return citations[:6]
