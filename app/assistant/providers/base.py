"""
providers/base.py

Interface abstraite pour un fournisseur de generation LLM. Toute nouvelle
integration (OpenAI, Gemini, Claude...) doit implementer cette interface et
etre enregistree dans providers/__init__.get_provider() — rag.py et
l'interface utilisateur n'ont alors jamais besoin d'etre modifies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderUnavailableError(Exception):
    """Levee quand le fournisseur LLM configure n'est pas joignable (ex: Ollama non demarre)."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Genere une reponse a partir d'un prompt systeme et d'un prompt utilisateur.

        Doit lever ProviderUnavailableError (pas une exception generique) si
        le service n'est pas joignable, pour que l'UI puisse afficher un
        message clair et traduit plutot qu'une trace d'erreur brute.
        """
        raise NotImplementedError
