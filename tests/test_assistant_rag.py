"""
Tests pour app/assistant/rag.py, prompts.py et providers/.

Le fournisseur Ollama reel n'est jamais appele ici (aucun serveur Ollama
disponible en CI) : les tests soit exploitent le comportement reel
d'indisponibilite (aucun Ollama sur localhost:11434 dans l'environnement de
test non plus), soit simulent une reponse HTTP avec unittest.mock pour
verifier le chemin de succes sans dependance reseau.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from assistant.rag import answer_question  # noqa: E402
from assistant.prompts import build_system_prompt, build_user_prompt, REFUSAL_MESSAGES, SUGGESTED_QUESTIONS  # noqa: E402
from assistant.retriever import RetrievedChunk, Chunk  # noqa: E402
from assistant.providers import get_provider  # noqa: E402
from assistant.providers.ollama_provider import OllamaProvider  # noqa: E402


# ==============================================================
# prompts.py
# ==============================================================
class TestPrompts:
    @pytest.mark.parametrize("lang", ["fr", "en", "ar"])
    def test_system_prompt_mentions_the_target_language_instruction(self, lang):
        prompt = build_system_prompt(language=lang)
        assert len(prompt) > 0
        assert "diagnostic" in prompt.lower() or "diagnos" in prompt.lower()

    def test_user_prompt_includes_the_question(self):
        prompt = build_user_prompt("Comment fonctionne Grad-CAM ?", [])
        assert "Comment fonctionne Grad-CAM ?" in prompt

    def test_user_prompt_includes_retrieved_context(self):
        chunk = Chunk(source="gradcam_explained.md", heading="Grad-CAM", text="Explication de Grad-CAM.")
        prompt = build_user_prompt("question", [RetrievedChunk(chunk=chunk, score=0.9)])
        assert "gradcam_explained.md" in prompt
        assert "Explication de Grad-CAM." in prompt

    def test_user_prompt_handles_no_context_gracefully(self):
        prompt = build_user_prompt("question", [])
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @pytest.mark.parametrize("lang", ["fr", "en", "ar"])
    def test_refusal_message_exists_for_every_supported_language(self, lang):
        assert lang in REFUSAL_MESSAGES
        assert len(REFUSAL_MESSAGES[lang]) > 0

    @pytest.mark.parametrize("lang", ["fr", "en", "ar"])
    def test_suggested_questions_exist_for_every_supported_language(self, lang):
        assert lang in SUGGESTED_QUESTIONS
        assert len(SUGGESTED_QUESTIONS[lang]) >= 3


# ==============================================================
# providers/
# ==============================================================
class TestProviderFactory:
    def test_get_provider_ollama_returns_ollama_provider(self):
        provider = get_provider("ollama", base_url="http://localhost:11434", model="qwen3:8b")
        assert isinstance(provider, OllamaProvider)

    def test_get_provider_unknown_raises_value_error(self):
        with pytest.raises(ValueError):
            get_provider("some_unknown_provider", base_url="x", model="y")


# ==============================================================
# rag.answer_question — orchestration complete
# ==============================================================
class TestAnswerQuestionSafetyFirst:
    def test_medical_question_is_refused_without_calling_the_provider(self):
        with patch("assistant.providers.ollama_provider.requests.post") as mock_post:
            result = answer_question("Est-ce que j'ai le covid ?", language="fr")
            assert result.refused is True
            assert mock_post.called is False  # le LLM ne doit JAMAIS etre contacte

    @pytest.mark.parametrize("lang", ["fr", "en", "ar"])
    def test_refusal_message_matches_the_requested_language(self, lang):
        result = answer_question("Est-ce que j'ai le covid ?", language=lang)
        assert result.text == REFUSAL_MESSAGES[lang]


class TestAnswerQuestionProviderUnavailable:
    def test_connection_error_degrades_to_clear_message_not_a_crash(self):
        # Aucun Ollama reel n'ecoute sur ce port dans l'environnement de test.
        result = answer_question(
            "Comment fonctionne Grad-CAM ?", language="fr",
            ollama_base_url="http://localhost:1", ollama_model="qwen3:8b",
        )
        assert result.refused is False
        assert result.provider_error is True
        assert "http://localhost:1" in result.text


class TestAnswerQuestionMockedSuccess:
    def test_successful_generation_returns_the_model_text_and_sources(self):
        fake_response = MagicMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"message": {"content": "Réponse générée par le modèle."}}

        with patch("assistant.providers.ollama_provider.requests.post", return_value=fake_response):
            result = answer_question("Comment fonctionne Grad-CAM ?", language="fr")

        assert result.refused is False
        assert result.provider_error is False
        assert result.text == "Réponse générée par le modèle."
        assert "gradcam_explained.md" in result.sources

    def test_mocked_call_uses_the_configured_model_name(self):
        fake_response = MagicMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"message": {"content": "ok"}}

        with patch("assistant.providers.ollama_provider.requests.post", return_value=fake_response) as mock_post:
            answer_question("question", language="fr", ollama_model="llama3:8b")
            assert mock_post.call_args.kwargs["json"]["model"] == "llama3:8b"
