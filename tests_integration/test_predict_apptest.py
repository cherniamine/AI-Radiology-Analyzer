"""
tests_integration/test_predict_apptest.py

Test d'integration complet, utilisant streamlit.testing.v1.AppTest (l'API
de test officielle de Streamlit). Contrairement a une execution "a blanc"
(bare mode, `python3 predict.py`), AppTest simule un vrai contexte de
script Streamlit — ce qui s'est revele essentiel : deux bugs reels
n'avaient JAMAIS ete detectes par l'execution a blanc ni par des tests
serveur "curl sur /" :

1. `st.navigation(pages)` levait une StreamlitAPIException a CHAQUE
   chargement de l'application, car les sept pages (views/*.py) exposent
   toutes une fonction nommee `render`, et Streamlit deduit le chemin
   d'URL du nom du callable par defaut -> collision. Corrige en passant
   url_path="..." explicitement a chaque st.Page() dans predict.py.
2. `NameError: name 'json' is not defined` dans la section export de
   analysis.py (import manquant), qui ne se declenchait qu'en cas
   d'upload REEL suivi d'un export JSON — un chemin de code que le mode
   bare (sans fichier uploade) ni un simple curl sur `/` n'exercent
   jamais, puisque curl sur `/` ne recupere que la coquille HTML/JS
   statique et n'execute pas le script cote serveur tant qu'aucune
   session websocket reelle ne s'etablit.

CE FICHIER N'EST PAS EXECUTE PAR LA CI RAPIDE (voir .github/workflows/ci.yml
et requirements-dev.txt) : il necessite TensorFlow et le modele entraine
(models/simple_cnn_model.h5), ce qui va a l'encontre de l'objectif de
rapidite de la CI. A executer localement avant tout changement touchant
predict.py ou app/views/analysis.py :

    pip install -r requirements.txt pytest
    pytest tests_integration/ -v
"""

import glob
import os
import sys

import pytest

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

pytest.importorskip("tensorflow", reason="Necessite TensorFlow (voir docstring du module)")
from streamlit.testing.v1 import AppTest  # noqa: E402
from streamlit.util import calc_hash  # noqa: E402


def goto_page(at: AppTest, url_path: str) -> AppTest:
    """
    Navigue vers une page enregistree via st.Page(callable, url_path=...).

    IMPORTANT : `AppTest.switch_page(path)` NE FONCTIONNE PAS pour les pages
    definies par callable (notre cas, voir predict.py) — decouverte faite
    pendant cette session. `switch_page` verifie seulement que le chemin
    donne correspond a un FICHIER existant sur disque (`Path.is_file()`),
    ce qui est vrai pour "views/history.py" par exemple (le fichier existe
    reellement), donc aucune exception n'est levee — mais il ne calcule PAS
    le meme hash que celui utilise en interne par Streamlit pour les pages
    a base de callable, donc la navigation echoue silencieusement et
    l'application reste sur la page par defaut. Verifie empiriquement :
    apres switch_page("views/about.py"), le contenu rendu etait
    strictement identique (meme longueur, memes octets) a celui de la page
    par defaut (Analyse), et ne contenait aucun texte propre a la page
    About ("MIT" absent, texte de zone d'upload de l'Analyse present).

    Le vrai mecanisme (retrouve en lisant le source de Streamlit,
    streamlit/navigation/page.py) : chaque StreamlitPage expose
    `_script_hash = calc_hash(self._url_path)`, et c'est ce hash que
    l'AppTest doit placer dans son attribut prive `_page_hash` pour que la
    page correspondante soit rendue au prochain `.run()`. C'est pourquoi
    predict.py declare un `url_path` explicite pour chaque page.
    """
    at._page_hash = calc_hash(url_path)
    return at


def _sample_image_bytes(class_dir="COVID"):
    files = glob.glob(os.path.join(DATA_DIR, class_dir, "images", "*"))
    if not files:
        pytest.skip(f"Aucune image d'exemple trouvee dans data/{class_dir}/images/")
    with open(files[0], "rb") as f:
        return f.read()


@pytest.fixture(autouse=True, scope="session")
def _cleanup_real_history_db():
    """
    Les tests qui n'utilisent pas app_with_seeded_history (donc pas de
    HISTORY_DB_PATH surcharge) ecrivent dans le vrai results/history.db de
    ce depot, via le chemin par defaut de app/config.py. Ce fichier est
    gitignore et purement local, mais on le nettoie tout de meme apres la
    session de tests pour ne rien laisser trainer.
    """
    yield
    real_db = os.path.join(os.path.dirname(APP_DIR), "results", "history.db")
    if os.path.exists(real_db):
        os.remove(real_db)


@pytest.fixture
def app():
    at = AppTest.from_file(os.path.join(APP_DIR, "predict.py"), default_timeout=120)
    at.run()
    return at


class TestAppLoadsWithoutError:
    def test_default_page_loads_with_no_exception(self, app):
        assert list(app.exception) == []

    def test_sidebar_has_language_selector(self, app):
        # Le selecteur de langue est le premier selectbox de la sidebar
        assert len(app.sidebar.selectbox) >= 1


class TestRealFileUpload:
    def test_uploading_a_real_image_produces_no_exception(self, app):
        content = _sample_image_bytes("COVID")
        app.file_uploader[0].upload("test_covid.png", content, mime_type="image/png")
        app.run()
        assert list(app.exception) == []

    def test_uploading_a_real_image_completes_the_pipeline(self, app):
        content = _sample_image_bytes("NORMAL")
        app.file_uploader[0].upload("test_normal.png", content, mime_type="image/png")
        app.run()
        assert len(app.status) == 1
        assert app.status[0].state == "complete"
        assert len(app.error) == 0

    def test_status_label_reflects_a_real_prediction(self, app):
        content = _sample_image_bytes("NORMAL")
        app.file_uploader[0].upload("test_normal.png", content, mime_type="image/png")
        app.run()
        label = app.status[0].label
        # Le libelle final contient la classe predite, la confiance et un temps reel (ms)
        assert "%" in label
        assert "ms" in label

    def test_export_section_renders_without_nameerror(self, app):
        # Reproduit precisement le bug NameError('json') trouve dans cette session :
        # il ne se declenchait qu'apres un upload reel, au rendu de la section export.
        content = _sample_image_bytes("COVID")
        app.file_uploader[0].upload("test_covid.png", content, mime_type="image/png")
        app.run()
        assert not any("json" in str(e.value).lower() and "not defined" in str(e.value).lower()
                       for e in app.exception)
        assert len(app.download_button) >= 3  # ZIP, CSV, JSON au minimum

    def test_multiple_uploads_each_produce_a_status(self, app):
        content_covid = _sample_image_bytes("COVID")
        content_normal = _sample_image_bytes("NORMAL")
        app.file_uploader[0].upload("a_covid.png", content_covid, mime_type="image/png")
        app.file_uploader[0].upload("b_normal.png", content_normal, mime_type="image/png")
        app.run()
        assert list(app.exception) == []
        assert len(app.status) == 2
        for s in app.status:
            assert s.state == "complete"


class TestRejectedImage:
    def test_a_tiny_image_is_rejected_not_crashed(self, app):
        import numpy as np
        import cv2
        tiny = np.full((10, 10, 3), 200, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", tiny)
        assert ok
        app.file_uploader[0].upload("too_small.png", bytes(buf), mime_type="image/png")
        app.run()
        assert list(app.exception) == []
        # Une image rejetee ne produit pas de resultat "complete" dans le pipeline
        if app.status:
            assert app.status[0].state != "complete"


class TestDarkModeAndLanguage:
    """Verifie le mode sombre et l'i18n avec un vrai contexte Streamlit (pas juste
    curl sur '/', qui ne recupere que la coquille HTML statique sans executer le
    script cote serveur)."""

    def test_dark_mode_toggle_flips_session_state_without_error(self):
        at = AppTest.from_file(os.path.join(APP_DIR, "predict.py"), default_timeout=60)
        at.run()
        assert ("_dark_mode" not in at.session_state) or (at.session_state["_dark_mode"] is False)
        at.sidebar.button[0].click().run()
        assert list(at.exception) == []
        assert at.session_state["_dark_mode"] is True

    def test_arabic_language_via_query_param_loads_without_error(self):
        at = AppTest.from_file(os.path.join(APP_DIR, "predict.py"), default_timeout=60)
        at.query_params["lang"] = "ar"
        at.run()
        assert list(at.exception) == []

    def test_english_language_via_query_param_loads_without_error(self):
        at = AppTest.from_file(os.path.join(APP_DIR, "predict.py"), default_timeout=60)
        at.query_params["lang"] = "en"
        at.run()
        assert list(at.exception) == []

    def test_every_page_renders_without_error_in_dark_mode(self):
        """Angle mort classique : le mode sombre est teste sur la page par
        defaut, la navigation est testee en mode clair — mais jamais les
        deux combines avant ce test."""
        at = AppTest.from_file(os.path.join(APP_DIR, "predict.py"), default_timeout=60)
        at.run()
        at.sidebar.button[0].click().run()
        assert at.session_state["_dark_mode"] is True

        for path in ["dashboard", "historique", "assistant", "parametres", "a-propos", "rapports"]:
            goto_page(at, path).run()
            assert list(at.exception) == [], f"Page '{path}' leve une exception en mode sombre"


class TestPageNavigation:
    """Verifie chaque page via une vraie navigation (voir goto_page() ci-dessus
    pour pourquoi AppTest.switch_page() ne peut pas etre utilise ici)."""

    def test_dashboard_page_renders_distinct_content(self, app):
        goto_page(app, "dashboard").run()
        assert list(app.exception) == []
        md = " ".join(m.value for m in app.markdown)
        assert "Dashboard" in md
        assert "Charger" not in md  # texte de la zone d'upload de l'Analyse

    def test_history_page_renders_distinct_content(self, app):
        goto_page(app, "historique").run()
        assert list(app.exception) == []
        md = " ".join(m.value for m in app.markdown)
        assert "Historique" in md
        assert "Charger" not in md

    def test_settings_page_renders_distinct_content(self, app):
        goto_page(app, "parametres").run()
        assert list(app.exception) == []
        md = " ".join(m.value for m in app.markdown)
        assert "Charger" not in md

    def test_assistant_page_renders_distinct_content_and_chat_input(self, app):
        goto_page(app, "assistant").run()
        assert list(app.exception) == []
        assert len(app.chat_input) == 1
        md = " ".join(m.value for m in app.markdown)
        assert "Charger" not in md

    def test_reports_page_renders_distinct_content(self, app):
        goto_page(app, "rapports").run()
        assert list(app.exception) == []
        md = " ".join(m.value for m in app.markdown)
        assert "Charger" not in md

    def test_about_page_renders_distinct_content(self, app):
        goto_page(app, "a-propos").run()
        assert list(app.exception) == []
        md = " ".join(m.value for m in app.markdown)
        assert "MIT" in md
        assert "Charger" not in md

    def test_each_page_renders_genuinely_different_content(self, app):
        """Garde-fou contre une regression silencieuse du mecanisme de
        navigation lui-meme (celle qui a fait echouer switch_page)."""
        seen = set()
        for path in ["dashboard", "historique", "assistant", "parametres", "a-propos", "rapports"]:
            goto_page(app, path).run()
            md = " ".join(m.value for m in app.markdown)
            assert md not in seen, f"La page '{path}' rend un contenu identique a une page precedente"
            seen.add(md)


class TestAssistantChatInteraction:
    def test_asking_a_legitimate_question_returns_an_answer_in_chat(self, app):
        goto_page(app, "assistant").run()
        app.chat_input[0].set_value("Comment fonctionne Grad-CAM ?").run()
        assert list(app.exception) == []
        md = " ".join(m.value for m in app.markdown)
        assert "Grad-CAM" in md

    def test_asking_a_medical_question_triggers_the_refusal(self, app):
        goto_page(app, "assistant").run()
        app.chat_input[0].set_value("Est-ce que j'ai le covid ?").run()
        assert list(app.exception) == []
        md = " ".join(m.value for m in app.markdown)
        assert "diagnostic" in md.lower()


class TestDashboardAndHistoryWithRealData:
    """Seede une vraie analyse dans la base SQLite avant de naviguer, pour
    verifier l'etat "avec donnees" (pas seulement l'etat vide) de ces deux pages."""

    @pytest.fixture
    def app_with_seeded_history(self, tmp_path, monkeypatch):
        import numpy as np

        db_path = str(tmp_path / "seeded_history.db")
        monkeypatch.setenv("HISTORY_DB_PATH", db_path)

        # Reinitialise les singletons config/persistence pour qu'ils relisent HISTORY_DB_PATH
        import config as config_module
        import persistence as persistence_module
        config_module.config = config_module.AppConfig.from_env()
        persistence_module._store_instance = None

        from persistence import get_store
        store = get_store()
        img = (np.random.rand(48, 48, 3) * 255).astype(np.uint8)
        probs = {"COVID": 5.0, "Lung_Opacity": 3.0, "NORMAL": 88.0, "Viral Pneumonia": 4.0}
        store.save(
            image_name="seed_analysis.png", predicted_class="NORMAL", confidence=88.0,
            class_probabilities=probs, findings="f", impression="i", recommendation="r",
            inference_ms=210.0, original_image=img, overlay_image=img,
        )

        at = AppTest.from_file(os.path.join(APP_DIR, "predict.py"), default_timeout=120)
        at.run()
        return at

    def test_dashboard_shows_real_seeded_kpi(self, app_with_seeded_history):
        at = app_with_seeded_history
        goto_page(at, "dashboard").run()
        assert list(at.exception) == []
        md = " ".join(m.value for m in at.markdown)
        assert "Analyses totales" in md

    def test_history_shows_the_seeded_record_and_export_buttons(self, app_with_seeded_history):
        at = app_with_seeded_history
        goto_page(at, "historique").run()
        assert list(at.exception) == []
        md = " ".join(m.value for m in at.markdown)
        assert "seed_analysis.png" in md
        assert len(at.download_button) >= 1
