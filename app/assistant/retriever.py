"""
retriever.py

Recherche par pertinence (TF-IDF + similarite cosinus) sur les documents de
app/assistant/knowledge_base/. Aucune dependance a un LLM ou a Ollama : ce
module est purement local et rapide a tester (voir
tests/test_assistant_retriever.py).

Chaque fichier .md de knowledge_base/ est decoupe en "chunks" (un chunk par
section ## ou par paragraphe si le fichier n'a pas de sections), et
retrieve(query, top_k) renvoie les chunks les plus proches de la question
posee, par similarite cosinus sur une representation TF-IDF.

C'est volontairement un retriever lexical (mots-cles ponderes), pas un
retriever a embeddings semantiques : il ne necessite aucun modele
supplementaire (pas d'appel a Ollama pour l'indexation), fonctionne hors
ligne, et suffit pour une base de connaissances de la taille de ce projet
(quelques documents).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")


@dataclass
class Chunk:
    source: str      # nom du fichier d'origine (ex: "gradcam_explained.md")
    heading: str      # titre de la section (ou nom du fichier si pas de heading)
    text: str          # contenu du chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def _split_into_chunks(filename: str, content: str) -> List[Chunk]:
    """Decoupe un document markdown en chunks par section (## ...)."""
    chunks: List[Chunk] = []
    sections = re.split(r"\n(?=#{1,2} )", content.strip())
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        heading = lines[0].lstrip("#").strip() if lines[0].startswith("#") else filename
        body = "\n".join(lines[1:]).strip() if lines[0].startswith("#") else section
        if body:
            chunks.append(Chunk(source=filename, heading=heading, text=body))
    return chunks


@lru_cache(maxsize=1)
def _load_all_chunks() -> tuple:
    """Charge et decoupe tous les documents de knowledge_base/. Mis en cache (contenu statique)."""
    chunks: List[Chunk] = []
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        return tuple(chunks)
    for filename in sorted(os.listdir(KNOWLEDGE_BASE_DIR)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        chunks.extend(_split_into_chunks(filename, content))
    return tuple(chunks)


class Retriever:
    """Index TF-IDF construit une fois (a la premiere requete), reutilise ensuite."""

    def __init__(self):
        self._chunks: List[Chunk] = list(_load_all_chunks())
        self._vectorizer = None
        self._matrix = None
        if self._chunks:
            texts = [f"{c.heading} {c.text}" for c in self._chunks]
            self._vectorizer = TfidfVectorizer(stop_words=None, max_df=0.95)
            self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[RetrievedChunk]:
        """
        Renvoie les `top_k` chunks les plus pertinents pour `query`, triés par
        score decroissant. Ignore les chunks dont le score est sous
        `min_score` (evite de renvoyer du contexte hors-sujet quand la
        question ne correspond a rien dans la base).
        """
        if not self._chunks or not query or not query.strip():
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedChunk(chunk=chunk, score=float(score))
            for chunk, score in ranked[:top_k]
            if score >= min_score
        ]

    @property
    def document_count(self) -> int:
        return len({c.source for c in self._chunks})

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


_retriever_instance: Retriever = None


def get_retriever() -> Retriever:
    """Singleton — construit l'index une seule fois par process."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
