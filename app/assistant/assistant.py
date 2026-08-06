"""
assistant.py

API publique de l'assistant, utilisee par app/views/assistant.py. C'est le
seul point d'entree que l'interface utilisateur doit connaitre — elle n'a
jamais besoin d'importer rag.py, retriever.py ou providers/ directement,
ce qui permet de faire evoluer l'architecture interne sans toucher a l'UI.
"""

from __future__ import annotations

import importlib
from .rag import AssistantAnswer


def ask(
    question: str,
    language: str = "fr",
    llm_provider: str | None = None,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
) -> AssistantAnswer:
    """Pose une question a l'assistant.

    Les arguments explicites permettent a l'UI de transmettre des reglages
    modifies en session sans coupler ce module a Streamlit.
    """
    from config import config
    rag = importlib.import_module(".rag", __package__)
    return rag.answer_question(
        question=question,
        language=language,
        llm_provider=llm_provider or config.llm_provider,
        ollama_base_url=ollama_base_url or config.ollama_base_url,
        ollama_model=ollama_model or config.ollama_model,
    )
