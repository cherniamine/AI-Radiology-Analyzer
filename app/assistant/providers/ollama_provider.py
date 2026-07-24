"""
providers/ollama_provider.py

Fournisseur LLM local via Ollama (https://ollama.com), aucune API payante.
Configure via OLLAMA_BASE_URL et OLLAMA_MODEL (voir .env.example et
app/config.py).

IMPORTANT — non teste avec un vrai serveur Ollama : l'environnement dans
lequel ce code a ete ecrit n'a pas Ollama installe/accessible. La
construction de la requete HTTP et le parsing de la reponse sont testes
avec un serveur HTTP simule (voir tests/test_assistant_ollama_provider.py),
mais le comportement reel du modele qwen3:8b (ou tout autre modele
configure) n'a pas pu etre verifie. A valider par vous-meme avec un Ollama
reellement lance.
"""

from __future__ import annotations

import requests

from .base import LLMProvider, ProviderUnavailableError


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ProviderUnavailableError(f"Ollama injoignable sur {self.base_url}") from e
        except requests.exceptions.Timeout as e:
            raise ProviderUnavailableError(f"Ollama n'a pas répondu dans le délai imparti ({self.timeout_seconds}s)") from e
        except requests.exceptions.HTTPError as e:
            raise ProviderUnavailableError(f"Ollama a renvoyé une erreur HTTP : {e}") from e

        try:
            data = response.json()
            return data["message"]["content"]
        except (ValueError, KeyError) as e:
            raise ProviderUnavailableError(f"Réponse Ollama inattendue : impossible d'extraire le texte ({e})") from e

    def is_reachable(self) -> bool:
        """Verifie rapidement si le serveur Ollama repond, sans generer de texte (pour un diagnostic UI)."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False
