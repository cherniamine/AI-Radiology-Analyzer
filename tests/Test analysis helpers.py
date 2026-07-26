"""
tests_integration/test_analysis_helpers.py

Teste les fonctions pures ajoutees a app/views/analysis.py :
- _files_fingerprint : identifiant d'un lot de fichiers uploades, utilise
  pour eviter de relancer l'inference CNN (et de dupliquer l'historique
  SQLite) a chaque rerun declenche par un widget sans rapport (theme,
  langue) — voir le commentaire au-dessus de `results = []` dans render().
- _md : retire l'indentation d'un bloc HTML avant st.markdown(), pour eviter
  que Markdown ne le traite comme un bloc de code indente (meme bug deja
  rencontre et corrige dans components.py).

Ce fichier vit dans tests_integration/ (pas tests/) parce que analysis.py
importe TensorFlow au niveau module — l'importer, meme pour tester deux
fonctions pures sans rapport avec le modele, tirerait TensorFlow dans la
suite rapide et casserait sa promesse de vitesse (voir README, section
Tests). Ces deux fonctions n'ont elles-memes aucune dependance a TensorFlow
ni a un vrai contexte Streamlit.
"""

import os
import sys

import pytest

_APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from views.analysis import _files_fingerprint, _md  # noqa: E402


class _FakeUploadedFile:
    """Simule l'API minimale de streamlit.runtime.uploaded_file_manager.UploadedFile
    utilisee par _files_fingerprint (seuls .name et .size sont lus)."""

    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size


class TestFilesFingerprint:
    def test_same_files_produce_the_same_fingerprint(self):
        batch_a = [_FakeUploadedFile("scan1.png", 1024), _FakeUploadedFile("scan2.png", 2048)]
        batch_b = [_FakeUploadedFile("scan1.png", 1024), _FakeUploadedFile("scan2.png", 2048)]
        assert _files_fingerprint(batch_a) == _files_fingerprint(batch_b)

    def test_different_file_name_changes_the_fingerprint(self):
        original = [_FakeUploadedFile("scan1.png", 1024)]
        renamed = [_FakeUploadedFile("scan1_v2.png", 1024)]
        assert _files_fingerprint(original) != _files_fingerprint(renamed)

    def test_different_file_size_changes_the_fingerprint(self):
        # Meme nom, contenu different (taille differente) : doit etre traite
        # comme un nouvel upload, pas comme un rerun du meme lot.
        v1 = [_FakeUploadedFile("scan.png", 1024)]
        v2 = [_FakeUploadedFile("scan.png", 2048)]
        assert _files_fingerprint(v1) != _files_fingerprint(v2)

    def test_order_of_files_matters(self):
        # Comportement volontairement strict : un upload dans un ordre
        # different est traite comme un nouveau lot plutot que de risquer un
        # faux "cache hit" avec des resultats desalignes.
        batch_a = [_FakeUploadedFile("a.png", 100), _FakeUploadedFile("b.png", 200)]
        batch_b = [_FakeUploadedFile("b.png", 200), _FakeUploadedFile("a.png", 100)]
        assert _files_fingerprint(batch_a) != _files_fingerprint(batch_b)

    def test_empty_batch_is_a_stable_empty_tuple(self):
        assert _files_fingerprint([]) == ()

    def test_fingerprint_is_hashable_for_session_state_comparison(self):
        # Doit pouvoir etre stocke/compare via st.session_state.get(...) ==
        # fingerprint sans lever d'exception (tuple de tuples, donc hashable).
        fp = _files_fingerprint([_FakeUploadedFile("scan.png", 1024)])
        hash(fp)  # ne doit pas lever


class TestMarkdownIndentationFix:
    def test_strips_leading_whitespace_from_every_line(self):
        html = """
        <div>
            <h2>Titre</h2>
        </div>
        """
        result = _md(html)
        assert all(not line.startswith(" ") for line in result.split("\n") if line)

    def test_preserves_content_and_order(self):
        html = """
        <div class="card">
            <p>Texte important</p>
        </div>
        """
        result = _md(html)
        assert '<div class="card">' in result
        assert "<p>Texte important</p>" in result
        assert result.index("<div") < result.index("<p>") < result.index("</div>")

    def test_handles_a_blank_line_in_the_middle_without_reintroducing_indentation(self):
        # Reproduit precisement le scenario a risque : une ligne vide suivie
        # d'une ligne indentee (le declencheur du bug CommonMark deja
        # rencontre dans components.py).
        html = """
        <div>
            <p>Premier paragraphe.</p>

            <p>Second paragraphe apres une ligne vide.</p>
        </div>
        """
        result = _md(html)
        non_blank_lines = [line for line in result.split("\n") if line]
        assert all(not line.startswith(" ") for line in non_blank_lines)

    def test_idempotent_on_already_flat_html(self):
        html = "<div><p>deja plat</p></div>"
        assert _md(html) == html


class TestCachedResultsSurviveAnUnrelatedRerun:
    """
    Reproduit le bug rapporte : sur la page Analyse, changer de theme ou de
    langue (widgets de la sidebar, sans rapport avec cette page) provoquait
    un st.rerun() apres lequel st.file_uploader() ne renvoyait plus les
    fichiers precedemment uploades — et comme tout l'affichage des
    resultats etait conditionne par `if uploaded_files and results:`, les
    resultats disparaissaient entierement, meme s'ils avaient deja ete
    calcules. Le correctif : `results` est desormais restaure depuis
    st.session_state (`elif "_analysis_results" in st.session_state:`) et
    la condition d'affichage est `if results:` (sans dependre de la valeur
    *courante* de uploaded_files).

    Plutot que de dependre du comportement de st.file_uploader dans
    AppTest (incertain et hors du perimetre de ce test), on pre-remplit
    directement st.session_state comme le ferait un run precedent, puis on
    verifie que le rendu affiche bien la section RESULTATS (pas l'etat
    d'accueil HOME STATE) alors qu'aucun fichier n'est "uploade" dans ce run.
    """

    def _seed_fake_analysis(self, at):
        import numpy as np
        from report_generator import build_report

        fake_image = np.zeros((32, 32, 3), dtype=np.uint8)
        fake_report = build_report(
            image_name="scan_test.png", predicted_class="NORMAL", confidence=91.0,
            class_probabilities={"NORMAL": 91.0, "COVID": 3.0, "Lung_Opacity": 3.0, "Viral Pneumonia": 3.0},
        )
        fake_result = {
            "Image": "scan_test.png",
            "Predicted Class": "Normal",
            "Confidence": 91.0,
            "Overlay": fake_image,
            "Original": fake_image,
            "Heatmap": np.zeros((7, 7), dtype=np.float32),
            "Report": fake_report,
            "Prob_NORMAL": 91.0, "Prob_COVID": 3.0, "Prob_Lung_Opacity": 3.0, "Prob_Viral Pneumonia": 3.0,
        }
        at.session_state["_analysis_fingerprint"] = (("scan_test.png", 12345),)
        at.session_state["_analysis_results"] = [fake_result]
        at.session_state["_analysis_rejected"] = []
        at.session_state["_analysis_zip_buffer"] = __import__("io").BytesIO()
        return at

    def test_results_section_renders_without_a_current_upload(self):
        from streamlit.testing.v1 import AppTest
        from streamlit.util import calc_hash

        predict_path = os.path.join(_APP_DIR, "predict.py")
        at = AppTest.from_file(predict_path, default_timeout=60)
        at.session_state["_analysis_fingerprint"] = None  # force la branche elif, pas cache_hit
        at = self._seed_fake_analysis(at)
        at.run()
        at._page_hash = calc_hash("analyse")
        at.run()

        assert at.exception == []
        page_text = " ".join(m.value for m in at.main.markdown)
        # La section RESULTATS (bandeau recapitulatif) doit apparaitre...
        assert "traitée" in page_text or "traité" in page_text or "Confidence" in page_text or "scan_test" in page_text
        # ...et l'etat d'accueil (affiche uniquement quand `results` est vide) ne doit PAS apparaitre.
        assert "Comment ça fonctionne" not in page_text