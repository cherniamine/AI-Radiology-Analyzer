"""
Tests pour app/persistence.py.

Utilise un fichier SQLite temporaire par test (jamais le fichier reel de
l'application) — rapide, sans dependance a Streamlit ni TensorFlow.
"""

import json
import os
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
