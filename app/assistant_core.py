"""
assistant.py

API publique de l'assistant, utilisee par app/views/assistant.py. C'est le
seul point d'entree que l'interface utilisateur doit connaitre — elle n'a
jamais besoin d'importer rag.py, retriever.py ou providers/ directement,
ce qui permet de faire evoluer l'architecture interne sans toucher a l'UI.
"""

from __future__ import annotations

from .rag import AssistantAnswer, answer_question


def ask(
    question: str,
    language: str = "fr",
    llm_provider: str | None = None,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
) -> AssistantAnswer:
    """Pose une question a l'assistant. Par defaut, utilise la configuration
    chargee depuis .env (config.py) ; un appelant qui a des reglages modifies
    pour la session (voir app/settings_store.py) peut les passer explicitement
    ici plutot que ce module n'importe settings_store lui-meme — il resterait
    sinon couple a Streamlit, ce qui casserait sa testabilite rapide sans
    ScriptRunContext (voir tests/test_assistant_rag.py)."""
    from config import config
    return answer_question(
        question=question,
        language=language,
        llm_provider=llm_provider or config.llm_provider,
        ollama_base_url=ollama_base_url or config.ollama_base_url,
        ollama_model=ollama_model or config.ollama_model,
    )
