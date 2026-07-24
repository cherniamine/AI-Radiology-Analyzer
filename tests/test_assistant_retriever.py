"""
Tests pour app/assistant/retriever.py.

Depend de scikit-learn mais pas de TensorFlow ni d'Ollama : rapide a
executer en CI.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from assistant.retriever import Retriever, get_retriever  # noqa: E402


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


class TestKnowledgeBaseLoading:
    def test_knowledge_base_has_documents(self, retriever):
        assert retriever.document_count > 0

    def test_knowledge_base_has_the_expected_topics(self, retriever):
        sources = {c.source for c in retriever._chunks}
        expected = {
            "overview.md", "supported_classes.md", "model_limitations.md",
            "gradcam_explained.md", "confidence_explained.md", "app_usage.md",
            "docker_deployment.md", "faq.md",
        }
        assert expected.issubset(sources)

    def test_get_retriever_returns_a_singleton(self):
        assert get_retriever() is get_retriever()


class TestRetrieve:
    def test_gradcam_question_retrieves_gradcam_document(self, retriever):
        results = retriever.retrieve("Comment fonctionne Grad-CAM ?", top_k=3)
        assert len(results) > 0
        assert any(r.chunk.source == "gradcam_explained.md" for r in results)

    def test_docker_question_retrieves_docker_document(self, retriever):
        results = retriever.retrieve("Comment lancer l'application avec Docker ?", top_k=3)
        assert len(results) > 0
        assert any(r.chunk.source == "docker_deployment.md" for r in results)

    def test_confidence_question_retrieves_confidence_document(self, retriever):
        results = retriever.retrieve("Que signifie le score de confiance ?", top_k=3)
        assert len(results) > 0
        assert any(r.chunk.source == "confidence_explained.md" for r in results)

    def test_results_are_sorted_by_descending_score(self, retriever):
        results = retriever.retrieve("modèle CNN classes limites", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self, retriever):
        results = retriever.retrieve("modèle", top_k=2)
        assert len(results) <= 2

    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.retrieve("", top_k=3) == []

    def test_irrelevant_query_returns_nothing_or_low_scores(self, retriever):
        # Une question totalement hors-sujet ne doit renvoyer que des
        # scores tres faibles (filtres par min_score), pas un resultat
        # a haute confiance choisi au hasard.
        results = retriever.retrieve("recette de cuisine pour un gâteau au chocolat", top_k=3)
        for r in results:
            assert r.score < 0.3
