"""
Tests pour app/assistant/safety.py.

Aucune dependance externe (regex pures) : tres rapide.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from assistant.safety import check_medical_advice_request  # noqa: E402


class TestFrenchMedicalRequests:
    @pytest.mark.parametrize("message", [
        "Est-ce que j'ai le covid ?",
        "Quel traitement dois-je prendre ?",
        "Dois-je aller à l'hôpital ?",
        "Quelle est la posologie recommandée ?",
        "Est-ce grave docteur ?",
    ])
    def test_detects_medical_request(self, message):
        result = check_medical_advice_request(message, language="fr")
        assert result.is_medical_request is True

    @pytest.mark.parametrize("message", [
        "Comment fonctionne Grad-CAM ?",
        "Quelle est l'exactitude du modèle ?",
        "Comment lancer l'application avec Docker ?",
        "Quelles sont les classes reconnues par le modèle ?",
    ])
    def test_does_not_flag_legitimate_questions(self, message):
        result = check_medical_advice_request(message, language="fr")
        assert result.is_medical_request is False


class TestEnglishMedicalRequests:
    @pytest.mark.parametrize("message", [
        "Do I have covid?",
        "What treatment should I take?",
        "Should I go to the hospital?",
        "Is this serious?",
    ])
    def test_detects_medical_request(self, message):
        result = check_medical_advice_request(message, language="en")
        assert result.is_medical_request is True

    @pytest.mark.parametrize("message", [
        "How does Grad-CAM work?",
        "What is the model's accuracy?",
        "How do I run this with Docker?",
    ])
    def test_does_not_flag_legitimate_questions(self, message):
        result = check_medical_advice_request(message, language="en")
        assert result.is_medical_request is False


class TestArabicMedicalRequests:
    def test_detects_medical_request(self):
        result = check_medical_advice_request("هل عندي كوفيد؟", language="ar")
        assert result.is_medical_request is True

    def test_does_not_flag_legitimate_question(self):
        result = check_medical_advice_request("كيف تعمل تقنية Grad-CAM؟", language="ar")
        assert result.is_medical_request is False


class TestCrossLanguageFallback:
    def test_french_pattern_detected_even_when_interface_language_is_arabic(self):
        # L'utilisateur peut ecrire en francais alors que l'interface est en arabe
        result = check_medical_advice_request("Est-ce que j'ai le covid ?", language="ar")
        assert result.is_medical_request is True


class TestEdgeCases:
    def test_empty_message_is_not_medical(self):
        assert check_medical_advice_request("", language="fr").is_medical_request is False

    def test_whitespace_only_message_is_not_medical(self):
        assert check_medical_advice_request("   ", language="fr").is_medical_request is False

    def test_case_insensitive_detection(self):
        result = check_medical_advice_request("DOIS-JE ALLER À L'HÔPITAL ?", language="fr")
        assert result.is_medical_request is True
