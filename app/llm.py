import os

from google import genai

from app.config import Settings


class GeminiChat:
    """Simple wrapper around Google GenAI client for backwards compatibility.

    The rest of the app uses LangChain models for agent behavior. This class
    remains available for direct generation use elsewhere in the codebase.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.settings.gemini_chat_model,
            contents=prompt,
        )
        return response.text or "I could not generate an answer from the available context."


def get_langchain_model_and_fallbacks(settings: Settings):
    """Return a tuple (primary_model_obj, fallback_first, additional_fallbacks)

    - primary_model_obj: a LangChain chat model instance (ChatGoogleGenerativeAI)
    - fallback_first: a string or model spec for the first fallback accepted by ModelFallbackMiddleware
    - additional_fallbacks: a list of additional model specs/objects

    Note: We return model spec strings for fallbacks which the middleware will accept.
    """
    # Import here to avoid a hard dependency at module import time
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:
        ChatGoogleGenerativeAI = None

    primary = None
    if ChatGoogleGenerativeAI is not None:
        primary = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            api_key=settings.gemini_api_key,
            temperature=0.2,
        )

    first_fallback = settings.fallback_first_model or ""
    additional = [m.strip() for m in settings.fallback_additional_models.split(",") if m.strip()]
    if settings.groq_api_key:
        os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)

    return primary, first_fallback or None, additional
