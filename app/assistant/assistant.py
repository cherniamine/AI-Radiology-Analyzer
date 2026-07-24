"""
assistant.py

API publique de l'assistant, utilisee par app/views/assistant.py. C'est le
seul point d'entree que l'interface utilisateur doit connaitre — elle n'a
jamais besoin d'importer rag.py, retriever.py ou providers/ directement,
ce qui permet de faire evoluer l'architecture interne sans toucher a l'UI.
"""

from __future__ import annotations

from .rag import AssistantAnswer, answer_question


def ask(question: str, language: str = "fr") -> AssistantAnswer:
    """Pose une question a l'assistant, avec la configuration actuelle de l'application."""
    from config import config
    return answer_question(
        question=question,
        language=language,
        llm_provider=config.llm_provider,
        ollama_base_url=config.ollama_base_url,
        ollama_model=config.ollama_model,
    )
