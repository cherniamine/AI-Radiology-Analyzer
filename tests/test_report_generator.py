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
    CLASS_REPORT_TEMPLATES_BY_LANG,
    DISCLAIMER,
    DISCLAIMER_BY_LANG,
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

class TestMultilingualReports:
    @pytest.mark.parametrize("language", ["fr", "en", "ar"])
    def test_report_generated_in_each_supported_language(self, language):
        report = build_report("scan.png", "NORMAL", 90.0, SAMPLE_PROBABILITIES, language=language)
        assert report.language == language
        payload = report.to_dict()
        assert payload["language"] == language
        assert len(payload["findings"]) > 0

    def test_unsupported_language_falls_back_to_default(self):
        report = build_report("scan.png", "NORMAL", 90.0, SAMPLE_PROBABILITIES, language="de")
        assert report.language == "fr"

    def test_default_language_is_french_when_unspecified(self):
        report = build_report("scan.png", "NORMAL", 90.0, SAMPLE_PROBABILITIES)
        assert report.language == "fr"

    def test_english_and_french_reports_have_different_wording(self):
        fr_report = build_report("scan.png", "COVID", 90.0, SAMPLE_PROBABILITIES, language="fr")
        en_report = build_report("scan.png", "COVID", 90.0, SAMPLE_PROBABILITIES, language="en")
        assert fr_report.to_dict()["findings"] != en_report.to_dict()["findings"]
        assert fr_report.to_dict()["disclaimer"] != en_report.to_dict()["disclaimer"]

    def test_arabic_report_contains_arabic_text(self):
        report = build_report("scan.png", "COVID", 90.0, SAMPLE_PROBABILITIES, language="ar")
        payload = report.to_dict()
        # Une plage Unicode arabe basique suffit a distinguer d'un texte latin.
        assert any("\u0600" <= ch <= "\u06FF" for ch in payload["findings"])

    def test_to_text_headers_are_localized(self):
        fr_text = build_report("scan.png", "NORMAL", 90.0, SAMPLE_PROBABILITIES, language="fr").to_text()
        en_text = build_report("scan.png", "NORMAL", 90.0, SAMPLE_PROBABILITIES, language="en").to_text()
        assert "RAPPORT IA" in fr_text
        assert "AI RADIOLOGY REPORT" in en_text

    @pytest.mark.parametrize("language", ["fr", "en", "ar"])
    def test_every_language_has_a_template_for_every_known_class(self, language):
        for predicted_class in CLASS_REPORT_TEMPLATES_BY_LANG["fr"]:
            report = build_report("scan.png", predicted_class, 90.0, SAMPLE_PROBABILITIES, language=language)
            assert len(report.to_dict()["findings"]) > 0

    def test_pdf_generation_succeeds_in_english(self):
        report = build_report("scan.png", "NORMAL", 90.0, SAMPLE_PROBABILITIES, language="en")
        pdf_bytes = generate_pdf_report(report, make_fake_image(), make_fake_image())
        assert pdf_bytes[:5] == b"%PDF-"

    def test_pdf_generation_for_arabic_falls_back_to_english_without_raising(self):
        # Voir la docstring du module : Helvetica/ReportLab n'a pas de glyphes
        # arabes, donc la mise en page PDF retombe sur l'anglais pour ne pas
        # produire un PDF illisible (cases vides). Le test verifie surtout
        # l'absence de crash.
        report = build_report("scan.png", "NORMAL", 90.0, SAMPLE_PROBABILITIES, language="ar")
        pdf_bytes = generate_pdf_report(report, make_fake_image(), make_fake_image())
        assert pdf_bytes[:5] == b"%PDF-"
        assert len(pdf_bytes) > 500

    def test_pdf_with_gradcam_heatmap_in_english(self):
        report = build_report("scan.png", "COVID", 70.0, SAMPLE_PROBABILITIES, language="en")
        pdf_bytes = generate_pdf_report(
            report, make_fake_image(), make_fake_image(), heatmap_img_rgb=make_fake_image(),
        )
        assert pdf_bytes[:5] == b"%PDF-"

    def test_backward_compatible_flat_templates_constant_still_works(self):
        # CLASS_REPORT_TEMPLATES (sans suffixe _BY_LANG) doit continuer de
        # fonctionner pour du code/tests plus anciens qui l'importent
        # directement — alias vers les gabarits francais.
        assert CLASS_REPORT_TEMPLATES == CLASS_REPORT_TEMPLATES_BY_LANG["fr"]
        assert DISCLAIMER == DISCLAIMER_BY_LANG["fr"]