"""
Tests pour app/translator.py.

Depend de Streamlit (session_state / query_params) mais pas de TensorFlow.
Streamlit tolere ces appels hors d'un vrai script run (avec un simple
avertissement), ce qui permet de tester translator.py directement avec
pytest sans lancer un vrai serveur Streamlit.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import translator as tr  # noqa: E402


@pytest.fixture(autouse=True)
def reset_language():
    """Revient au francais avant/apres chaque test pour eviter les fuites entre tests."""
    tr.set_language("fr")
    yield
    tr.set_language("fr")


class TestSupportedLanguages:
    def test_all_three_required_languages_are_present(self):
        assert set(tr.SUPPORTED_LANGUAGES.keys()) == {"fr", "en", "ar"}

    def test_arabic_is_marked_rtl(self):
        assert tr.SUPPORTED_LANGUAGES["ar"]["rtl"] is True

    def test_french_and_english_are_not_rtl(self):
        assert tr.SUPPORTED_LANGUAGES["fr"]["rtl"] is False
        assert tr.SUPPORTED_LANGUAGES["en"]["rtl"] is False


class TestLocaleFilesLoadCorrectly:
    @pytest.mark.parametrize("lang", ["fr", "en", "ar"])
    def test_locale_file_loads_without_error(self, lang):
        data = tr._load_locale(lang)
        assert isinstance(data, dict)
        assert len(data) > 0

    @pytest.mark.parametrize("lang", ["fr", "en", "ar"])
    def test_locale_file_has_common_nav_keys(self, lang):
        data = tr._load_locale(lang)
        nav = data.get("common", {}).get("nav", {})
        for key in ["analysis", "dashboard", "reports", "history", "assistant", "settings", "about"]:
            assert key in nav
            assert len(nav[key]) > 0


class TestTranslate:
    def test_translate_known_key_in_french(self):
        tr.set_language("fr")
        assert tr.t("common.nav.analysis") == "Nouvelle analyse"

    def test_translate_known_key_in_english(self):
        tr.set_language("en")
        assert tr.t("common.nav.analysis") == "New Analysis"

    def test_translate_known_key_in_arabic(self):
        tr.set_language("ar")
        assert tr.t("common.nav.analysis") == "تحليل جديد"

    def test_missing_key_falls_back_to_the_key_itself_not_a_crash(self):
        tr.set_language("fr")
        result = tr.t("this.key.does.not.exist.anywhere")
        assert result == "this.key.does.not.exist.anywhere"

    def test_format_kwargs_are_substituted(self):
        tr.set_language("en")
        result = tr.t("about.version_line", version="2.0.0")
        assert "2.0.0" in result

    def test_format_with_missing_kwarg_does_not_raise(self):
        tr.set_language("en")
        # "about.version_line" expects {version}; omitting it must not crash
        result = tr.t("about.version_line")
        assert isinstance(result, str)

    def test_list_values_are_returned_as_is(self):
        tr.set_language("en")
        limitations = tr.t("about.limitations")
        assert isinstance(limitations, list)
        assert len(limitations) == 3


class TestLanguageSwitching:
    def test_set_language_updates_get_language(self):
        tr.set_language("en")
        assert tr.get_language() == "en"

    def test_set_language_ignores_unsupported_language(self):
        tr.set_language("fr")
        tr.set_language("xx")  # langue non supportee
        assert tr.get_language() == "fr"

    def test_is_rtl_follows_current_language(self):
        tr.set_language("ar")
        assert tr.is_rtl() is True
        tr.set_language("en")
        assert tr.is_rtl() is False
