from __future__ import annotations

from contextpilot.config.llm import get_chat_model
from contextpilot.config.settings import get_settings


class Generator:
    """
    Wrapper around the configured chat model.

    All LLM calls pass through this class so the rest of the
    pipeline is independent of the specific model provider.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = get_chat_model()

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        try:
            response = self.llm.invoke(prompt)

        except Exception as exc:
            raise RuntimeError(
                f"LLM request failed using provider '{self.settings.llm_provider}'. "
                "Check API credentials, quota, or network connectivity."
            ) from exc

        if hasattr(response, "content"):
            return response.content

        return str(response)