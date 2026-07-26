"""
persistence.py

Couche de persistance legere (SQLite, fichier local) pour l'historique des
analyses. Sans dependance a Streamlit ni TensorFlow : peut etre importee et
testee independamment (voir tests/test_persistence.py), ce qui garde la
suite de tests rapide en CI.

Utilisation typique :

    from persistence import get_store

    store = get_store()
    record_id = store.save(
        image_name="scan.png", predicted_class="NORMAL", confidence=91.2,
        class_probabilities={...}, findings="...", impression="...",
        recommendation="...", inference_ms=340.0,
        original_image=img_rgb, overlay_image=overlay_rgb,
    )
    records = store.list(limit=50)
    stats = store.stats()
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import cv2

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    image_name TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    class_probabilities TEXT NOT NULL,
    findings TEXT NOT NULL DEFAULT '',
    impression TEXT NOT NULL DEFAULT '',
    recommendation TEXT NOT NULL DEFAULT '',
    inference_ms REAL NOT NULL DEFAULT 0,
    original_png BLOB,
    overlay_png BLOB,
    heatmap_png BLOB
);

CREATE TABLE IF NOT EXISTS assistant_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources TEXT  -- ✅ Ajout : stockage JSON des sources
);
"""


def _encode_png(img_rgb: Optional[np.ndarray]) -> Optional[bytes]:
    """Encode une image RGB (numpy) en PNG pour stockage en BLOB. None si l'image est absente."""
    if img_rgb is None:
        return None
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    return bytes(buffer) if ok else None


def _decode_png(data: Optional[bytes]) -> Optional[np.ndarray]:
    """Decode un BLOB PNG stocke en image RGB (numpy). None si le BLOB est absent/invalide."""
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


@dataclass
class AnalysisRecord:
    """Une ligne de l'historique. Les images ne sont decodees que si on les demande explicitement."""

    id: Optional[int]
    created_at: str
    image_name: str
    predicted_class: str
    confidence: float
    class_probabilities: dict
    findings: str
    impression: str
    recommendation: str
    inference_ms: float
    _original_png: Optional[bytes] = None
    _overlay_png: Optional[bytes] = None
    _heatmap_png: Optional[bytes] = None

    def original_image(self) -> Optional[np.ndarray]:
        return _decode_png(self._original_png)

    def overlay_image(self) -> Optional[np.ndarray]:
        return _decode_png(self._overlay_png)

    def heatmap_image(self) -> Optional[np.ndarray]:
        """Carte Grad-CAM pure (sans fusion avec l'image originale). Absente pour les
        analyses enregistrees avant l'ajout de cette colonne (migration sans retro-remplissage)."""
        return _decode_png(self._heatmap_png)

    def has_images(self) -> bool:
        return bool(self._original_png and self._overlay_png)

    def has_heatmap(self) -> bool:
        return bool(self._heatmap_png)

    def to_dict(self) -> dict:
        """Vue serialisable (sans les images) — utilisee pour l'export JSON depuis l'historique."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "image_name": self.image_name,
            "prediction": self.predicted_class,
            "confidence": round(self.confidence, 2),
            "class_probabilities": self.class_probabilities,
            "findings": self.findings,
            "impression": self.impression,
            "recommendation": self.recommendation,
            "inference_ms": round(self.inference_ms, 1),
        }


def _row_to_record(row: sqlite3.Row) -> AnalysisRecord:
    return AnalysisRecord(
        id=row["id"],
        created_at=row["created_at"],
        image_name=row["image_name"],
        predicted_class=row["predicted_class"],
        confidence=row["confidence"],
        class_probabilities=json.loads(row["class_probabilities"]),
        findings=row["findings"],
        impression=row["impression"],
        recommendation=row["recommendation"],
        inference_ms=row["inference_ms"],
        _original_png=row["original_png"],
        _overlay_png=row["overlay_png"],
        _heatmap_png=row["heatmap_png"],
    )


class HistoryStore:
    """Acces a l'historique des analyses stocke en SQLite (un fichier local)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Migrations : ajouter les colonnes qui n'existaient pas dans les
            # versions anterieures du schema, sans casser les bases deja
            # deployees (ALTER TABLE ... ADD COLUMN est idempotent ici via
            # le try/except, plutot qu'une verification de version formelle).
            try:
                conn.execute("ALTER TABLE assistant_messages ADD COLUMN sources TEXT")
            except sqlite3.OperationalError:
                pass  # La colonne existe déjà
            try:
                conn.execute("ALTER TABLE analyses ADD COLUMN heatmap_png BLOB")
            except sqlite3.OperationalError:
                pass  # La colonne existe déjà

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(
        self,
        *,
        image_name: str,
        predicted_class: str,
        confidence: float,
        class_probabilities: dict,
        findings: str = "",
        impression: str = "",
        recommendation: str = "",
        inference_ms: float = 0.0,
        original_image: Optional[np.ndarray] = None,
        overlay_image: Optional[np.ndarray] = None,
        heatmap_image: Optional[np.ndarray] = None,
    ) -> int:
        """Enregistre une analyse. Retourne l'id genere.

        heatmap_image : carte Grad-CAM pure (avant fusion avec l'image
        originale) — distincte de overlay_image. Optionnelle : les pages
        Historique/Rapports retombent sur une mise en page PDF a 2 images
        (original + overlay) si elle est absente."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO analyses
                   (created_at, image_name, predicted_class, confidence, class_probabilities,
                    findings, impression, recommendation, inference_ms, original_png, overlay_png,
                    heatmap_png)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    image_name,
                    predicted_class,
                    float(confidence),
                    json.dumps(class_probabilities),
                    findings,
                    impression,
                    recommendation,
                    float(inference_ms),
                    _encode_png(original_image),
                    _encode_png(overlay_image),
                    _encode_png(heatmap_image),
                ),
            )
            return cur.lastrowid

    def list(
        self,
        *,
        predicted_class: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 200,
    ) -> list[AnalysisRecord]:
        """Liste les analyses (plus recentes en premier), avec filtre optionnel par classe et recherche par nom de fichier."""
        query = "SELECT * FROM analyses"
        clauses, params = [], []
        if predicted_class:
            clauses.append("predicted_class = ?")
            params.append(predicted_class)
        if search:
            clauses.append("image_name LIKE ?")
            params.append(f"%{search}%")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_record(r) for r in rows]

    def get(self, record_id: int) -> Optional[AnalysisRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE id = ?", (record_id,)).fetchone()
        return _row_to_record(row) if row else None

    def delete(self, record_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM analyses WHERE id = ?", (record_id,))
        return cur.rowcount > 0

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()["c"]

    # === Conversation de l'assistant IA ===

    def add_assistant_message(self, role: str, content: str, sources: Optional[list] = None) -> int:
        """
        Ajoute un message a la conversation persistee.
        role doit etre 'user' ou 'assistant'.
        sources : liste de noms de fichiers sources (optionnel)
        """
        sources_json = json.dumps(sources) if sources else None
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO assistant_messages (created_at, role, content, sources) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), role, content, sources_json),
            )
            return cur.lastrowid

    def get_assistant_conversation(self) -> list[dict]:
        """
        Retourne la conversation complete, dans l'ordre chronologique.
        Chaque message contient 'role', 'content' et optionnellement 'sources'.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, sources FROM assistant_messages ORDER BY id ASC"
            ).fetchall()
        
        conversation = []
        for row in rows:
            message = {"role": row["role"], "content": row["content"]}
            if row["sources"]:
                try:
                    message["sources"] = json.loads(row["sources"])
                except json.JSONDecodeError:
                    pass
            conversation.append(message)
        return conversation

    def clear_assistant_conversation(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM assistant_messages")

    def count_assistant_messages(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) AS c FROM assistant_messages").fetchone()["c"]

    def stats(self) -> dict:
        """
        Agrege les KPI reels pour le Dashboard a partir des donnees stockees.
        Ne retourne QUE ce que les donnees permettent : les moyennes sont
        None (pas 0) si la table est vide, pour que l'UI affiche "N/A"
        plutot qu'un faux zero.
        """
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()["c"]
            per_class = {
                row["predicted_class"]: row["c"]
                for row in conn.execute(
                    "SELECT predicted_class, COUNT(*) AS c FROM analyses GROUP BY predicted_class"
                )
            }
            avg_conf = conn.execute("SELECT AVG(confidence) AS a FROM analyses").fetchone()["a"]
            avg_inf = conn.execute("SELECT AVG(inference_ms) AS a FROM analyses").fetchone()["a"]
            timeline = [
                {"day": row["day"], "count": row["c"]}
                for row in conn.execute(
                    "SELECT date(created_at) AS day, COUNT(*) AS c FROM analyses GROUP BY day ORDER BY day"
                )
            ]
        return {
            "total": total,
            "per_class": per_class,
            "avg_confidence": avg_conf,
            "avg_inference_ms": avg_inf,
            "timeline": timeline,
        }


_store_instance: Optional[HistoryStore] = None


def get_store() -> HistoryStore:
    """Singleton importable partout dans l'application (une seule connexion configuree par process)."""
    global _store_instance
    if _store_instance is None:
        from config import config
        _store_instance = HistoryStore(config.history_db_path)
    return _store_instance