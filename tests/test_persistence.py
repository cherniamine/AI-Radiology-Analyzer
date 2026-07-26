"""
Tests pour app/persistence.py.

Utilise un fichier SQLite temporaire par test (jamais le fichier reel de
l'application) — rapide, sans dependance a Streamlit ni TensorFlow.
"""

import json
import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from persistence import HistoryStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "history_test.db")
    return HistoryStore(db_path)


def make_fake_image(size=32):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


SAMPLE_PROBS = {"COVID": 5.0, "Lung_Opacity": 3.0, "NORMAL": 90.0, "Viral Pneumonia": 2.0}


class TestSaveAndGet:
    def test_save_returns_a_positive_id(self, store):
        record_id = store.save(
            image_name="scan.png", predicted_class="NORMAL", confidence=90.0,
            class_probabilities=SAMPLE_PROBS,
        )
        assert record_id > 0

    def test_get_returns_the_saved_record(self, store):
        record_id = store.save(
            image_name="scan.png", predicted_class="COVID", confidence=77.5,
            class_probabilities=SAMPLE_PROBS, findings="f", impression="i", recommendation="r",
            inference_ms=123.4,
        )
        record = store.get(record_id)
        assert record is not None
        assert record.image_name == "scan.png"
        assert record.predicted_class == "COVID"
        assert record.confidence == pytest.approx(77.5)
        assert record.inference_ms == pytest.approx(123.4)
        assert record.class_probabilities == SAMPLE_PROBS

    def test_get_unknown_id_returns_none(self, store):
        assert store.get(9999) is None

    def test_save_with_images_round_trips_correctly(self, store):
        original = make_fake_image()
        overlay = make_fake_image()
        record_id = store.save(
            image_name="scan.png", predicted_class="NORMAL", confidence=90.0,
            class_probabilities=SAMPLE_PROBS, original_image=original, overlay_image=overlay,
        )
        record = store.get(record_id)
        assert record.has_images() is True
        decoded_original = record.original_image()
        assert decoded_original is not None
        assert decoded_original.shape == original.shape

    def test_save_without_images_has_no_images(self, store):
        record_id = store.save(
            image_name="scan.png", predicted_class="NORMAL", confidence=90.0,
            class_probabilities=SAMPLE_PROBS,
        )
        record = store.get(record_id)
        assert record.has_images() is False
        assert record.original_image() is None

    def test_save_with_heatmap_round_trips_correctly(self, store):
        heatmap = make_fake_image()
        record_id = store.save(
            image_name="scan.png", predicted_class="COVID", confidence=90.0,
            class_probabilities=SAMPLE_PROBS,
            original_image=make_fake_image(), overlay_image=make_fake_image(),
            heatmap_image=heatmap,
        )
        record = store.get(record_id)
        assert record.has_heatmap() is True
        decoded = record.heatmap_image()
        assert decoded is not None
        assert decoded.shape == heatmap.shape

    def test_save_without_heatmap_has_no_heatmap(self, store):
        record_id = store.save(
            image_name="scan.png", predicted_class="NORMAL", confidence=90.0,
            class_probabilities=SAMPLE_PROBS,
            original_image=make_fake_image(), overlay_image=make_fake_image(),
        )
        record = store.get(record_id)
        assert record.has_heatmap() is False
        assert record.heatmap_image() is None

    def test_existing_db_without_heatmap_column_is_migrated(self, tmp_path):
        # Meme principe que la migration de assistant_messages.sources : une
        # base creee par une version anterieure (sans heatmap_png) doit etre
        # migree a l'ouverture, pas planter au premier save() avec heatmap.
        db_path = str(tmp_path / "legacy_analyses.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE analyses (
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
                overlay_png BLOB
            )
        """)
        conn.commit()
        conn.close()

        migrated = HistoryStore(db_path)
        record_id = migrated.save(
            image_name="scan.png", predicted_class="NORMAL", confidence=90.0,
            class_probabilities=SAMPLE_PROBS, heatmap_image=make_fake_image(),
        )
        assert migrated.get(record_id).has_heatmap() is True


class TestList:
    def test_list_returns_most_recent_first(self, store):
        id1 = store.save(image_name="a.png", predicted_class="NORMAL", confidence=1, class_probabilities=SAMPLE_PROBS)
        id2 = store.save(image_name="b.png", predicted_class="NORMAL", confidence=2, class_probabilities=SAMPLE_PROBS)
        records = store.list()
        assert records[0].id == id2
        assert records[1].id == id1

    def test_list_filters_by_class(self, store):
        store.save(image_name="a.png", predicted_class="COVID", confidence=1, class_probabilities=SAMPLE_PROBS)
        store.save(image_name="b.png", predicted_class="NORMAL", confidence=2, class_probabilities=SAMPLE_PROBS)
        records = store.list(predicted_class="COVID")
        assert len(records) == 1
        assert records[0].predicted_class == "COVID"

    def test_list_filters_by_search(self, store):
        store.save(image_name="chest_scan.png", predicted_class="NORMAL", confidence=1, class_probabilities=SAMPLE_PROBS)
        store.save(image_name="other.png", predicted_class="NORMAL", confidence=2, class_probabilities=SAMPLE_PROBS)
        records = store.list(search="chest")
        assert len(records) == 1
        assert records[0].image_name == "chest_scan.png"

    def test_list_respects_limit(self, store):
        for i in range(5):
            store.save(image_name=f"{i}.png", predicted_class="NORMAL", confidence=1, class_probabilities=SAMPLE_PROBS)
        records = store.list(limit=2)
        assert len(records) == 2

    def test_empty_store_lists_nothing(self, store):
        assert store.list() == []


class TestDelete:
    def test_delete_removes_the_record(self, store):
        record_id = store.save(image_name="a.png", predicted_class="NORMAL", confidence=1, class_probabilities=SAMPLE_PROBS)
        assert store.delete(record_id) is True
        assert store.get(record_id) is None

    def test_delete_unknown_id_returns_false(self, store):
        assert store.delete(9999) is False


class TestStats:
    def test_stats_on_empty_store_returns_none_averages_not_zero(self, store):
        stats = store.stats()
        assert stats["total"] == 0
        assert stats["avg_confidence"] is None
        assert stats["avg_inference_ms"] is None
        assert stats["per_class"] == {}
        assert stats["timeline"] == []

    def test_stats_total_matches_number_of_records(self, store):
        for i in range(3):
            store.save(image_name=f"{i}.png", predicted_class="NORMAL", confidence=80, class_probabilities=SAMPLE_PROBS)
        assert store.stats()["total"] == 3

    def test_stats_per_class_counts_are_correct(self, store):
        store.save(image_name="a.png", predicted_class="COVID", confidence=80, class_probabilities=SAMPLE_PROBS)
        store.save(image_name="b.png", predicted_class="COVID", confidence=80, class_probabilities=SAMPLE_PROBS)
        store.save(image_name="c.png", predicted_class="NORMAL", confidence=80, class_probabilities=SAMPLE_PROBS)
        stats = store.stats()
        assert stats["per_class"]["COVID"] == 2
        assert stats["per_class"]["NORMAL"] == 1

    def test_stats_average_confidence_is_correct(self, store):
        store.save(image_name="a.png", predicted_class="NORMAL", confidence=80, class_probabilities=SAMPLE_PROBS)
        store.save(image_name="b.png", predicted_class="NORMAL", confidence=90, class_probabilities=SAMPLE_PROBS)
        assert store.stats()["avg_confidence"] == pytest.approx(85.0)

    def test_count_matches_stats_total(self, store):
        store.save(image_name="a.png", predicted_class="NORMAL", confidence=80, class_probabilities=SAMPLE_PROBS)
        assert store.count() == store.stats()["total"] == 1


class TestRecordToDict:
    def test_to_dict_is_json_serializable(self, store):
        record_id = store.save(
            image_name="scan.png", predicted_class="NORMAL", confidence=91.234,
            class_probabilities=SAMPLE_PROBS, findings="f", impression="i", recommendation="r",
        )
        record = store.get(record_id)
        serialized = json.dumps(record.to_dict())
        reloaded = json.loads(serialized)
        assert reloaded["prediction"] == "NORMAL"
        assert reloaded["confidence"] == pytest.approx(91.23, abs=0.01)

    def test_to_dict_does_not_include_raw_image_bytes(self, store):
        record_id = store.save(
            image_name="scan.png", predicted_class="NORMAL", confidence=91.0,
            class_probabilities=SAMPLE_PROBS, original_image=make_fake_image(),
        )
        payload = store.get(record_id).to_dict()
        assert "original_png" not in payload
        assert "_original_png" not in payload


class TestAssistantConversation:
    def test_empty_conversation_by_default(self, store):
        assert store.get_assistant_conversation() == []
        assert store.count_assistant_messages() == 0

    def test_add_message_returns_a_positive_id(self, store):
        message_id = store.add_assistant_message("user", "Comment fonctionne Grad-CAM ?")
        assert message_id > 0

    def test_conversation_preserves_chronological_order(self, store):
        store.add_assistant_message("user", "Question 1")
        store.add_assistant_message("assistant", "Réponse 1")
        store.add_assistant_message("user", "Question 2")
        conversation = store.get_assistant_conversation()
        assert [m["content"] for m in conversation] == ["Question 1", "Réponse 1", "Question 2"]
        assert [m["role"] for m in conversation] == ["user", "assistant", "user"]

    def test_conversation_survives_a_new_store_instance(self, tmp_path):
        # Simule un rechargement de page : une nouvelle instance de
        # HistoryStore pointant vers le meme fichier doit retrouver la
        # conversation, contrairement a st.session_state qui serait vide.
        db_path = str(tmp_path / "shared.db")
        first = HistoryStore(db_path)
        first.add_assistant_message("user", "Une question")

        second = HistoryStore(db_path)
        assert second.get_assistant_conversation() == [{"role": "user", "content": "Une question"}]

    def test_clear_conversation_empties_the_table(self, store):
        store.add_assistant_message("user", "Question")
        store.add_assistant_message("assistant", "Réponse")
        store.clear_assistant_conversation()
        assert store.get_assistant_conversation() == []
        assert store.count_assistant_messages() == 0

    def test_clear_conversation_does_not_touch_analyses(self, store):
        store.save(image_name="a.png", predicted_class="NORMAL", confidence=80, class_probabilities=SAMPLE_PROBS)
        store.add_assistant_message("user", "Question")
        store.clear_assistant_conversation()
        assert store.count() == 1

    def test_message_without_sources_has_no_sources_key(self, store):
        store.add_assistant_message("user", "Question sans sources")
        message = store.get_assistant_conversation()[0]
        assert "sources" not in message

    def test_message_with_sources_round_trips(self, store):
        store.add_assistant_message("assistant", "Réponse", sources=["faq.md", "overview.md"])
        message = store.get_assistant_conversation()[0]
        assert message["sources"] == ["faq.md", "overview.md"]

    def test_empty_sources_list_is_treated_as_no_sources(self, store):
        # sources=[] est "faux" en Python -> traite comme "pas de sources",
        # coherent avec le "if sources:" cote UI (components.py) qui
        # n'affiche jamais de bloc Sources vide.
        store.add_assistant_message("assistant", "Réponse", sources=[])
        message = store.get_assistant_conversation()[0]
        assert "sources" not in message

    def test_existing_db_without_sources_column_is_migrated(self, tmp_path):
        # Simule une base creee par une version anterieure du code, avant
        # l'ajout de la colonne `sources` : HistoryStore doit la migrer
        # automatiquement a l'ouverture plutot que planter au premier
        # message avec sources.
        db_path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE assistant_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        migrated = HistoryStore(db_path)
        migrated.add_assistant_message("assistant", "Réponse", sources=["faq.md"])
        assert migrated.get_assistant_conversation()[0]["sources"] == ["faq.md"]

    def test_invalid_role_is_rejected(self, store):
        with pytest.raises(sqlite3.IntegrityError):
            store.add_assistant_message("system", "Devrait échouer")