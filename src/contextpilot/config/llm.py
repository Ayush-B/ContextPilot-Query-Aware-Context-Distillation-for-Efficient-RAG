from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from contextpilot.config.settings import get_settings


def get_chat_model():
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "openai":
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def get_embedding_model() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )