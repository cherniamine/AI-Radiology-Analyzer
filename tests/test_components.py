"""
Tests pour app/components.py.

Teste les fonctions pures `_build_xxx_html` (pas les wrappers qui appellent
st.markdown) : verifie que le HTML genere contient bien les bonnes classes
CSS, les bonnes valeurs, et la bonne couleur selon le statut — sans avoir
besoin d'un navigateur ni d'un contexte Streamlit.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from components import (  # noqa: E402
    _build_section_title_html,
    _build_metric_card_html,
    _build_status_badge_html,
    _build_glass_card_html,
    _build_empty_state_html,
    _build_confidence_gauge_html,
    _confidence_color,
    _build_probability_bars_html,
    _build_user_message_html,
    _build_assistant_message_html,
    _build_footer_html,
)


class TestSectionTitle:
    def test_includes_title_and_icon(self):
        html = _build_section_title_html("bar-chart-3", "Dashboard")
        assert "Dashboard" in html
        assert "<svg" in html

    def test_subtitle_included_when_given(self):
        html = _build_section_title_html("layout-dashboard", "Titre", subtitle="Sous-titre")
        assert "Sous-titre" in html

    def test_no_subtitle_paragraph_when_omitted(self):
        html = _build_section_title_html("layout-dashboard", "Titre")
        assert "subtitle" not in html or '<p class="subtitle">' not in html


class TestMetricCard:
    def test_includes_value_and_label(self):
        html = _build_metric_card_html("42", "Analyses totales")
        assert "42" in html
        assert "Analyses totales" in html
        assert "metric-card" in html

    def test_custom_color_is_applied(self):
        html = _build_metric_card_html("7", "COVID-19", color="#EF4444")
        assert "#EF4444" in html

    def test_no_color_style_when_omitted(self):
        html = _build_metric_card_html("7", "Normal")
        assert "style=" not in html.split("metric-value")[1].split(">")[0]


class TestStatusBadge:
    @pytest.mark.parametrize("variant", ["success", "warning", "danger", "info"])
    def test_valid_variant_produces_matching_class(self, variant):
        html = _build_status_badge_html("Texte", variant=variant)
        assert f"badge-{variant}" in html

    def test_invalid_variant_falls_back_to_info(self):
        html = _build_status_badge_html("Texte", variant="not_a_real_variant")
        assert "badge-info" in html

    def test_icon_is_included_when_given(self):
        html = _build_status_badge_html("Actif", variant="success", icon_name="check-circle")
        assert "<svg" in html


class TestGlassCard:
    def test_wraps_content_in_card_class(self):
        html = _build_glass_card_html("<p>Contenu</p>")
        assert "card" in html
        assert "<p>Contenu</p>" in html

    def test_extra_style_is_applied(self):
        html = _build_glass_card_html("content", extra_style="text-align:center;")
        assert "text-align:center;" in html


class TestEmptyState:
    def test_includes_title_and_description(self):
        html = _build_empty_state_html("Rien ici", "Description explicative.")
        assert "Rien ici" in html
        assert "Description explicative." in html


class TestConfidenceGauge:
    @pytest.mark.parametrize("value,expected_color", [
        (95.0, "var(--success-text)"),
        (80.0, "var(--success-text)"),
        (70.0, "var(--warning-text)"),
        (60.0, "var(--warning-text)"),
        (45.0, "var(--danger-text)"),
        (0.0, "var(--danger-text)"),
    ])
    def test_color_thresholds_match_the_rest_of_the_app(self, value, expected_color):
        assert _confidence_color(value) == expected_color

    def test_gauge_html_includes_the_percentage(self):
        html = _build_confidence_gauge_html(87.3, label="Confiance")
        assert "87.3%" in html

    def test_gauge_html_includes_an_svg(self):
        html = _build_confidence_gauge_html(50.0)
        assert "<svg" in html
        assert "<circle" in html

    def test_value_is_clamped_to_valid_range(self):
        html_over = _build_confidence_gauge_html(150.0)
        assert "100.0%" in html_over
        html_under = _build_confidence_gauge_html(-20.0)
        assert "0.0%" in html_under


class TestProbabilityBars:
    def test_all_classes_are_rendered(self):
        def fake_color_fn(name):
            return ("#EF4444", "soft", name, "bar-chart-3")

        probs = {"COVID": 70.0, "NORMAL": 20.0, "Lung_Opacity": 10.0}
        html = _build_probability_bars_html(probs, fake_color_fn)
        for name in probs:
            assert name in html

    def test_bars_are_sorted_by_descending_probability(self):
        def fake_color_fn(name):
            return ("#000000", "soft", name, "bar-chart-3")

        probs = {"A": 10.0, "B": 90.0, "C": 50.0}
        html = _build_probability_bars_html(probs, fake_color_fn)
        # B (90%) doit apparaitre avant C (50%) qui doit apparaitre avant A (10%)
        assert html.index("B") < html.index("C") < html.index("A")

    def test_bar_width_reflects_probability(self):
        def fake_color_fn(name):
            return ("#000000", "soft", name, "bar-chart-3")

        html = _build_probability_bars_html({"X": 42.5}, fake_color_fn)
        assert "width:42.5%" in html


class TestChatBubbles:
    def test_user_message_contains_text(self):
        html = _build_user_message_html("Bonjour")
        assert "Bonjour" in html

    def test_assistant_message_contains_text(self):
        html = _build_assistant_message_html("Réponse de l'assistant")
        assert "Réponse de l'assistant" in html

    def test_assistant_message_includes_sources_when_given(self):
        html = _build_assistant_message_html("Réponse", sources=["faq.md", "overview.md"])
        assert "faq.md" in html
        assert "overview.md" in html

    def test_assistant_message_omits_sources_block_when_none(self):
        html = _build_assistant_message_html("Réponse", sources=None)
        assert "Sources" not in html

    def test_user_and_assistant_bubbles_align_differently(self):
        user_html = _build_user_message_html("test")
        assistant_html = _build_assistant_message_html("test")
        assert "flex-end" in user_html
        assert "flex-start" in assistant_html


class TestFooter:
    def test_includes_all_provided_information(self):
        html = _build_footer_html("AI Radiology Analyzer", "1.0.0", "MIT", ["TensorFlow", "Streamlit"])
        assert "AI Radiology Analyzer" in html
        assert "1.0.0" in html
        assert "MIT" in html
        assert "TensorFlow" in html
        assert "Streamlit" in html
