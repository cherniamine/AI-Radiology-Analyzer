"""
providers/__init__.py

Fabrique de fournisseurs LLM. Ajouter un nouveau fournisseur (OpenAI,
Gemini, Claude...) = creer providers/xxx_provider.py implementant
LLMProvider, puis l'ajouter a _PROVIDERS ci-dessous. Rien d'autre dans
l'application (rag.py, l'UI) n'a besoin d'etre modifie.
"""

from __future__ import annotations

from .base import LLMProvider, ProviderUnavailableError
from .ollama_provider import OllamaProvider

_PROVIDERS = {
    "ollama": OllamaProvider,
    # "openai": OpenAIProvider,   # a ajouter le jour ou ce fournisseur existe
    # "gemini": GeminiProvider,
    # "claude": ClaudeProvider,
}


def get_provider(provider_name: str, **kwargs) -> LLMProvider:
    """
    Instancie le fournisseur configure (LLM_PROVIDER dans .env). Leve une
    ValueError explicite si le fournisseur demande n'est pas implemente,
    plutot que de retomber silencieusement sur un autre.
    """
    provider_cls = _PROVIDERS.get(provider_name.lower())
    if provider_cls is None:
        raise ValueError(
            f"Fournisseur LLM inconnu : '{provider_name}'. "
            f"Disponibles : {', '.join(_PROVIDERS.keys())}."
        )
    return provider_cls(**kwargs)
