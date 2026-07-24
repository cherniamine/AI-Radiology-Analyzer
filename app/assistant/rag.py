"""
rag.py

Orchestration retrieval-augmented generation :
  1. Garde-fou medical (safety.py) — s'execute EN PREMIER, avant tout appel
     LLM. Si la question demande un diagnostic/traitement/avis medical,
     on renvoie le message de refus traduit sans jamais contacter le LLM.
  2. Recuperation des passages pertinents dans la documentation du projet
     (retriever.py, TF-IDF local, aucun appel reseau).
  3. Construction des prompts (prompts.py).
  4. Generation via le fournisseur LLM configure (providers/, par defaut
     Ollama en local).

Ce module ne depend d'aucun framework RAG externe (pas de LangChain, etc.)
— l'ensemble tient dans quelques fonctions simples, ce qui le rend facile a
tester et a auditer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .prompts import build_system_prompt, build_user_prompt, REFUSAL_MESSAGES, OLLAMA_UNAVAILABLE_MESSAGES
from .providers import get_provider
from .providers.base import ProviderUnavailableError
from .retriever import get_retriever, RetrievedChunk
from .safety import check_medical_advice_request


@dataclass
class AssistantAnswer:
    text: str
    refused: bool = False
    sources: List[str] = field(default_factory=list)
    provider_error: bool = False


def answer_question(
    question: str,
    language: str = "fr",
    llm_provider: str = "ollama",
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "qwen3:8b",
    top_k: int = 3,
) -> AssistantAnswer:
    """
    Point d'entree principal. Ne leve jamais d'exception vers l'appelant :
    toute erreur (refus, LLM injoignable) est encodee dans AssistantAnswer
    pour que l'UI puisse l'afficher proprement.
    """
    # 1. Garde-fou medical — avant toute chose.
    safety_result = check_medical_advice_request(question, language=language)
    if safety_result.is_medical_request:
        return AssistantAnswer(
            text=REFUSAL_MESSAGES.get(language, REFUSAL_MESSAGES["fr"]),
            refused=True,
        )

    # 2. Recuperation de contexte dans la documentation du projet.
    retrieved: List[RetrievedChunk] = get_retriever().retrieve(question, top_k=top_k)

    # 3. Construction des prompts.
    system_prompt = build_system_prompt(language=language)
    user_prompt = build_user_prompt(question, retrieved)

    # 4. Generation via le fournisseur configure.
    try:
        provider = get_provider(
            llm_provider,
            base_url=ollama_base_url,
            model=ollama_model,
        )
        generated_text = provider.generate(system_prompt, user_prompt)
        return AssistantAnswer(
            text=generated_text,
            refused=False,
            sources=sorted({r.chunk.source for r in retrieved}),
        )
    except ProviderUnavailableError:
        message = OLLAMA_UNAVAILABLE_MESSAGES.get(language, OLLAMA_UNAVAILABLE_MESSAGES["fr"])
        return AssistantAnswer(
            text=message.format(base_url=ollama_base_url, model=ollama_model),
            refused=False,
            provider_error=True,
        )
