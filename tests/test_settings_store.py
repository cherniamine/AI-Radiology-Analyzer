"""
Tests pour app/settings_store.py.

Depend de Streamlit (session_state) mais pas de TensorFlow. Streamlit
tolere ces appels hors d'un vrai script run (avec un simple avertissement),
ce qui permet de tester ce module directement avec pytest — meme
convention que tests/test_translator.py.
"""

import os
import sys

import pytest
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import settings_store as ss  # noqa: E402
from config import config  # noqa: E402


@pytest.fixture(autouse=True)
def reset_overrides():
    """Repart d'une session propre avant/apres chaque test pour eviter les fuites."""
    ss.reset_all()
    yield
    ss.reset_all()


class TestEditableKeys:
    def test_language_and_theme_are_not_editable_here(self):
        # Ont deja leur propre persistance dediee (translator.py / predict.py) —
        # les dupliquer ici creerait un second controle concurrent.
        assert "default_language" not in ss.EDITABLE_KEYS
        assert "default_theme" not in ss.EDITABLE_KEYS

    def test_every_editable_key_exists_on_appconfig(self):
        # Protege contre une cle fantome (faute de frappe) qui ferait planter
        # get_setting() silencieusement a l'usage plutot qu'a l'import.
        for key in ss.EDITABLE_KEYS:
            assert hasattr(config, key), f"'{key}' n'existe pas sur AppConfig"


class TestGetSetting:
    def test_returns_config_value_by_default(self):
        assert ss.get_setting("default_gradcam_alpha") == config.default_gradcam_alpha
        assert ss.get_setting("llm_provider") == config.llm_provider

    def test_returns_override_after_set_setting(self):
        ss.set_setting("default_gradcam_alpha", 0.75)
        assert ss.get_setting("default_gradcam_alpha") == 0.75

    def test_works_for_non_editable_keys_too_read_only(self):
        # Pas d'override possible, mais la lecture uniforme reste utile.
        assert ss.get_setting("app_title") == config.app_title


class TestSetSetting:
    def test_rejects_a_key_outside_editable_keys(self):
        with pytest.raises(ValueError):
            ss.set_setting("app_title", "Titre pirate")

    def test_does_not_mutate_the_global_config_object(self):
        # L'override est purement cote session_state ; l'objet config (singleton
        # partage par toute l'app) ne doit jamais etre modifie en place.
        original = config.default_gradcam_alpha
        ss.set_setting("default_gradcam_alpha", 0.75)
        assert config.default_gradcam_alpha == original

    def test_boolean_settings_round_trip_correctly(self):
        ss.set_setting("enable_assistant", False)
        assert ss.get_setting("enable_assistant") is False


class TestIsOverridden:
    def test_false_before_any_override(self):
        assert ss.is_overridden("default_gradcam_alpha") is False

    def test_true_after_set_setting(self):
        ss.set_setting("max_upload_size_mb", 25)
        assert ss.is_overridden("max_upload_size_mb") is True

    def test_false_for_a_key_never_touched(self):
        ss.set_setting("max_upload_size_mb", 25)
        assert ss.is_overridden("ollama_model") is False


class TestResetAll:
    def test_clears_every_override(self):
        ss.set_setting("default_gradcam_alpha", 0.75)
        ss.set_setting("enable_history", False)
        ss.reset_all()
        assert ss.get_setting("default_gradcam_alpha") == config.default_gradcam_alpha
        assert ss.get_setting("enable_history") == config.enable_history
        assert not any(ss.is_overridden(k) for k in ss.EDITABLE_KEYS)

    def test_reset_on_a_fresh_session_does_not_raise(self):
        ss.reset_all()  # aucun override existant — ne doit pas lever
