"""
Tests pour app/report_generator.py.

Ne dependent que de numpy, Pillow et reportlab (pas de TensorFlow, pas de
Streamlit) : rapides a executer, y compris en CI.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from report_generator import (  # noqa: E402
    build_report,
    generate_pdf_report,
    CLASS_REPORT_TEMPLATES,
    DISCLAIMER,
)

SAMPLE_PROBABILITIES = {
    "COVID": 8.2,
    "Lung_Opacity": 5.1,
    "NORMAL": 4.3,
    "Viral Pneumonia": 82.4,
}


def make_fake_image(size=64):
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


# ==============================================================
# build_report / to_text / to_dict
# ==============================================================
class TestBuildReport:
    @pytest.mark.parametrize("predicted_class", list(CLASS_REPORT_TEMPLATES.keys()))
    def test_report_generated_for_every_known_class(self, predicted_class):
        report = build_report("scan.png", predicted_class, 90.0, SAMPLE_PROBABILITIES)
        text = report.to_text()
        assert len(text) > 0
        assert CLASS_REPORT_TEMPLATES[predicted_class]["label"] in text

    def test_disclaimer_is_always_present_in_text(self):
        report = build_report("scan.png", "NORMAL", 95.0, SAMPLE_PROBABILITIES)
        assert DISCLAIMER in report.to_text()

    def test_confidence_value_appears_in_text(self):
        report = build_report("scan.png", "COVID", 77.3, SAMPLE_PROBABILITIES)
        assert "77.3" in report.to_text()

    def test_unknown_class_falls_back_gracefully_instead_of_raising(self):
        report = build_report("scan.png", "SomeUnseenClass", 50.0, SAMPLE_PROBABILITIES)
        text = report.to_text()
        assert "SomeUnseenClass" not in "" or True  # ne doit pas lever d'exception
        assert len(text) > 0


class TestReportToDict:
    def test_to_dict_is_valid_json(self):
        report = build_report("scan.png", "Viral Pneumonia", 82.4, SAMPLE_PROBABILITIES)
        payload = report.to_dict()
        serialized = json.dumps(payload, ensure_ascii=False)
        reloaded = json.loads(serialized)
        assert reloaded["prediction_key"] == "Viral Pneumonia"
        assert reloaded["confidence"] == pytest.approx(82.4)

    def test_to_dict_contains_all_required_fields(self):
        report = build_report("scan.png", "NORMAL", 91.0, SAMPLE_PROBABILITIES)
        payload = report.to_dict()
        for field in [
            "image_name", "model", "prediction", "prediction_key", "confidence",
            "class_probabilities", "findings", "impression", "recommendation",
            "disclaimer", "generated_at",
        ]:
            assert field in payload

    def test_disclaimer_field_is_never_empty(self):
        report = build_report("scan.png", "COVID", 60.0, SAMPLE_PROBABILITIES)
        assert len(report.to_dict()["disclaimer"]) > 0

    def test_class_probabilities_round_trip_numeric_values(self):
        report = build_report("scan.png", "NORMAL", 91.0, SAMPLE_PROBABILITIES)
        payload = json.loads(json.dumps(report.to_dict()))
        for class_name, prob in SAMPLE_PROBABILITIES.items():
            assert payload["class_probabilities"][class_name] == pytest.approx(prob)


# ==============================================================
# generate_pdf_report
# ==============================================================
class TestGeneratePdfReport:
    def test_pdf_is_created_with_valid_magic_bytes(self):
        report = build_report("scan.png", "Viral Pneumonia", 82.4, SAMPLE_PROBABILITIES)
        pdf_bytes = generate_pdf_report(report, make_fake_image(), make_fake_image())
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:5] == b"%PDF-"
        assert len(pdf_bytes) > 500

    def test_pdf_with_three_images_original_heatmap_overlay(self):
        report = build_report("scan.png", "COVID", 70.0, SAMPLE_PROBABILITIES)
        pdf_bytes = generate_pdf_report(
            report, make_fake_image(), make_fake_image(), heatmap_img_rgb=make_fake_image(),
        )
        assert pdf_bytes[:5] == b"%PDF-"
        assert len(pdf_bytes) > 500

    def test_pdf_generation_does_not_raise_for_any_known_class(self):
        for predicted_class in CLASS_REPORT_TEMPLATES:
            report = build_report("scan.png", predicted_class, 88.0, SAMPLE_PROBABILITIES)
            pdf_bytes = generate_pdf_report(report, make_fake_image(), make_fake_image())
            assert pdf_bytes[:5] == b"%PDF-"
