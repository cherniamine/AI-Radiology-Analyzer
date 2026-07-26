"""
views/analysis.py

Page "Nouvelle analyse" : upload de radiographies, inference CNN, Grad-CAM,
rapport IA et export (CSV/ZIP/PDF/JSON). C'est la fonctionnalite historique
de l'application, migree ici telle quelle depuis l'ancien predict.py
monolithique (Phase 1 de la refonte SaaS) : aucun comportement n'a change,
seul l'emplacement du code a bouge.
"""

import base64
import json
import os
import time
import warnings
import zipfile
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow import keras

from config import config, CLASS_NAMES, CLASS_META, DATASET_SIZE, class_color, load_real_metrics
from translator import get_language
from icons import icon as render_icon
from components import confidence_gauge, probability_bars, confidence_color
from image_utils import (
    preprocess_image,
    validate_xray_image,
    load_and_preprocess_image,
    colorize_heatmap,
    overlay_heatmap_cv,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from report_generator import build_report, generate_pdf_report
except ImportError:
    class DummyReport:
        def __init__(self, **kwargs):
            self._dict = kwargs

        def to_dict(self):
            return self._dict

    def build_report(image_name, predicted_class, confidence, class_probabilities):
        findings_map = {
            "COVID": "Opacités en verre dépoli bilatérales et périphériques, prédominance dans les lobes inférieurs.",
            "Lung_Opacity": "Opacité interstitielle ou alvéolaire, à corréler avec le contexte clinique.",
            "NORMAL": "Pas d'opacité anormale visualisée. Architecture pulmonaire normale.",
            "Viral Pneumonia": "Opacités multifocales, parfois confluentes, avec distribution péribronchique.",
        }
        impression_map = {
            "COVID": "Suspicion de COVID-19. Corrélation clinique et biologique recommandée.",
            "Lung_Opacity": "Opacité pulmonaire identifiée. Une imagerie complémentaire peut être utile.",
            "NORMAL": "Radiographie thoracique normale. Pas de signe de pneumopathie.",
            "Viral Pneumonia": "Évoque une pneumonie virale. Corrélation avec les symptômes.",
        }
        recommendation_map = {
            "COVID": "Confirmation par PCR/antigénique, suivi clinique et isolement si indiqué.",
            "Lung_Opacity": "Bilan complémentaire : scanner thoracique si contexte clinique.",
            "NORMAL": "Pas d'examen complémentaire urgent. Suivi si symptômes persistants.",
            "Viral Pneumonia": "Surveillance clinique, traitement symptomatique, réévaluation si aggravation.",
        }
        return DummyReport(
            image_name=image_name,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            findings=findings_map.get(predicted_class, "Anomalie identifiée. Corrélation clinique recommandée."),
            impression=impression_map.get(predicted_class, "Interprétation radiologique à confirmer par un expert."),
            recommendation=recommendation_map.get(predicted_class, "Avis médical spécialisé recommandé."),
        )

    def generate_pdf_report(report, original_img, overlay_img, heatmap_img):
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Rapport d'analyse radiologique")
            c.setFont("Helvetica", 11)
            y = height - 80
            c.drawString(50, y, f"Image: {report.to_dict().get('image_name', 'N/A')}")
            y -= 20
            c.drawString(50, y, f"Classe prédite: {report.to_dict().get('predicted_class', 'N/A')}")
            y -= 20
            c.drawString(50, y, f"Confiance: {report.to_dict().get('confidence', 0):.1f}%")
            c.save()
            return buffer.getvalue()
        except Exception:
            import json as _json
            return _json.dumps(report.to_dict()).encode("utf-8")


# Chemins/constantes specifiques a cette page (le modele et les metriques
# sont geres via config.py, voir app/config.py)
MODEL_PATH = config.model_path
GRADCAM_DIR = os.path.join(os.path.dirname(config.report_output), "gradcam")
os.makedirs(GRADCAM_DIR, exist_ok=True)


# ==============================================================
def _files_fingerprint(uploaded_files) -> tuple:
    """Identifiant stable d'un lot de fichiers uploades, base sur (nom, taille) de
    chacun — pas sur un hash du contenu binaire (trop couteux a recalculer a
    chaque rerun juste pour verifier que rien n'a change). Utilise pour
    detecter un VRAI nouvel upload et distinguer ce cas d'un simple rerun
    declenche par un widget sans rapport (theme, langue) — voir le commentaire
    au-dessus de `results = []` plus bas dans render()."""
    return tuple((f.name, f.size) for f in uploaded_files)


# ==============================================================
def _md(html: str) -> str:
    """A appeler autour de CHAQUE bloc HTML multi-lignes passe a st.markdown() dans
    ce fichier — retire l'indentation de chaque ligne avant l'envoi.

    Pourquoi : le moteur Markdown de Streamlit (st.markdown) traite toute ligne
    indentee de 4 espaces ou plus comme un BLOC DE CODE indente (regle
    CommonMark), pas comme du HTML a injecter — meme bug deja rencontre et
    corrige dans components.py (voir le commentaire au-dessus de
    _build_assistant_message_html). Ici, chaque bloc HTML est ecrit en Python
    avec une indentation qui suit la profondeur d'imbrication du code (8, 12,
    16 espaces...), ce qui declenche exactement ce piege des qu'un des appels
    imbriques (ex. render_icon(), une f-string sur plusieurs lignes) laisse une
    ligne vide avant un bloc indente. Plutot que corriger un site d'appel a la
    fois des qu'un rapport de bug arrive, tous les appels st.markdown(f\"\"\"...\"\"\")
    de ce fichier passent desormais par cette fonction (34 sites au moment ou
    ce commentaire est ecrit — voir git blame / historique de conversation)."""
    return "\n".join(line.lstrip() for line in html.strip("\n").split("\n"))


# ==============================================================
def get_model_input_shape(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) == 4:
        return input_shape[1:3]
    elif len(input_shape) == 3:
        return input_shape[0:2]
    return (128, 128)


def find_last_conv_layer(model):
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer.name)
        elif hasattr(layer, "layers"):
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    conv_layers.append(f"{layer.name}/{sublayer.name}")
    return conv_layers[-1] if conv_layers else None


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"{render_icon('x-circle', size=16, color='var(--danger-text)')} Modèle introuvable : {MODEL_PATH}")
        return None
    try:
        with st.spinner(f"{render_icon('loader', size=16)} Chargement du modèle en cours..."):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            input_shape = model.input_shape
            if input_shape is not None:
                if isinstance(input_shape, list):
                    dummy_shape = (1,) + tuple(input_shape[0][1:])
                else:
                    dummy_shape = (1,) + tuple(input_shape[1:])
                if len(dummy_shape) == 3:
                    dummy_shape = dummy_shape + (3,)
                elif dummy_shape[-1] == 1:
                    dummy_shape = dummy_shape[:-1] + (3,)
                _ = model(tf.zeros(dummy_shape))
        return model
    except Exception as e:
        st.error(f"{render_icon('x-circle', size=16, color='var(--danger-text)')} Erreur lors du chargement du modèle : {str(e)}")
        return None


def gradcam(model, img_array, last_conv_layer_name):
    if last_conv_layer_name is None:
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        dummy_heatmap = np.zeros((img_array.shape[1] // 8, img_array.shape[2] // 8))
        return dummy_heatmap, int(predicted_class), predictions[0]

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy(), int(predicted_class), predictions[0].numpy()
    except Exception:
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        dummy_heatmap = np.zeros((img_array.shape[1] // 8, img_array.shape[2] // 8))
        return dummy_heatmap, int(predicted_class), predictions[0]


def plotly_light_layout(fig, height=320):
    """Applique un style clair et moderne aux graphiques Plotly avec vérification"""
    if fig is None:
        return fig
    try:
        fig.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Inter", size=12, color="#475569"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=height,
            legend=dict(font=dict(color="#475569")),
            hovermode='x unified',
            xaxis=dict(
                gridcolor="#E2E8F0",
                gridwidth=1,
                linecolor="#E2E8F0",
                linewidth=1,
                tickfont=dict(color="#64748B"),
            ),
            yaxis=dict(
                gridcolor="#E2E8F0",
                gridwidth=1,
                linecolor="#E2E8F0",
                linewidth=1,
                tickfont=dict(color="#64748B"),
            ),
        )
        return fig
    except Exception as e:
        st.warning(f"{render_icon('alert-triangle', size=14, color='var(--warning-text)')} Erreur de mise en page du graphique : {str(e)}")
        return fig


def _colorized_heatmap_for_storage(original_shape, raw_heatmap, colormap: str) -> np.ndarray:
    """Convertit la carte d'activation Grad-CAM brute (2D, valeurs 0-1) en image RGB
    uint8 coloree, prete a etre encodee en PNG pour la persistance (voir
    persistence.HistoryStore.save(heatmap_image=...)). Meme normalisation que celle
    deja appliquee au re-rendu "live" par carte plus bas dans ce fichier."""
    colored = colorize_heatmap(original_shape, raw_heatmap, colormap)
    if colored.dtype != np.uint8:
        colored = (colored * 255).astype(np.uint8) if colored.max() <= 1.0 else colored.astype(np.uint8)
    return colored


def render() -> None:
    # ==============================================================
    # MODEL LOAD
    # ==============================================================
    METRICS = load_real_metrics()
    OVERALL_ACCURACY = METRICS["accuracy"] * 100 if METRICS else None
    model = load_model()
    if model is None:
        st.stop()

    IMG_SIZE = get_model_input_shape(model)
    LAST_CONV_LAYER = find_last_conv_layer(model)

    # ==============================================================
    # HEADER AVEC ICÔNES SVG
    # ==============================================================
    st.markdown(_md(f"""
    <div class="main-header fade-in">
        <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
            <div style="flex: 1;">
                <h1>{render_icon('hospital', size=28, color='var(--accent-primary)')} AI Radiology Analyzer</h1>
                <p class="subtitle">Classification de radiographies pulmonaires assistée par IA avec visualisation Grad-CAM</p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="badge badge-info">{render_icon('search', size=14)} Recherche</span>
                <span class="badge badge-success">{render_icon('cpu', size=14)} CNN</span>
                <span class="badge badge-warning">{render_icon('layers', size=14)} 4 classes</span>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    accuracy_display = f"{OVERALL_ACCURACY:.1f}%" if OVERALL_ACCURACY else "N/A"

    st.markdown(_md(f"""
    <div class="readout-strip fade-in">
        <div class="readout-item">
            <div class="readout-label">{render_icon('brain-circuit', size=13)} Modèle</div>
            <div class="readout-value">CNN • 4 classes</div>
        </div>
        <div class="readout-item">
            <div class="readout-label">{render_icon('maximize', size=13)} Résolution</div>
            <div class="readout-value">{IMG_SIZE[0]}×{IMG_SIZE[1]}</div>
        </div>
        <div class="readout-item">
            <div class="readout-label">{render_icon('target', size=13)} Exactitude</div>
            <div class="readout-value" style="color: {confidence_color(OVERALL_ACCURACY) if OVERALL_ACCURACY else 'var(--danger-text)'};">{accuracy_display}</div>
        </div>
        <div class="readout-item">
            <div class="readout-label">{render_icon('database', size=13)} Images entraînement</div>
            <div class="readout-value">{DATASET_SIZE:,}</div>
        </div>
    </div>
    """).replace(",", "\u2009"), unsafe_allow_html=True)

    # ==============================================================
    # SIDEBAR
    # ==============================================================
    with st.sidebar:
        st.markdown(_md(f"""#### {render_icon('tag', size=16)} Classes détectées"""), unsafe_allow_html=True)
        for cname in CLASS_NAMES:
            color, soft, label, icon = class_color(cname)
            st.markdown(_md(f"""
            <div class="legend-item" style="--legend-color:{color};">
                <div class="legend-dot" style="background-color:{color};"></div>
                <span class="legend-text">{render_icon(icon, size=14)} {label}</span>
                <span class="legend-sub">{CLASS_META[cname]['description']}</span>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(_md(f"""#### {render_icon('palette', size=16)} Visualisation Grad-CAM"""), unsafe_allow_html=True)
    
        heatmap_alpha = st.slider(
            "Intensité",
            0.1, 0.9, 0.5, 0.05,
            help="Opacité de la superposition Grad-CAM",
            key="heatmap_slider",
        )
    
        colormap = st.selectbox(
            "Palette",
            ["JET", "HOT", "PLASMA", "VIRIDIS", "INFERNO"],
            index=0,
            help="Palette de couleurs utilisée pour dessiner la carte Grad-CAM",
            key="colormap_select",
        )

        if METRICS:
            st.markdown("---")
            st.markdown(_md(f"""#### {render_icon('bar-chart-3', size=16)} Performance par classe"""), unsafe_allow_html=True)
            for cname in CLASS_NAMES:
                cm = METRICS["classes"].get(cname)
                if not cm:
                    continue
                color, soft, label, icon = class_color(cname)
                st.markdown(_md(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px;">
                        <span style="color:var(--text-primary);">{render_icon(icon, size=13)} {label}</span>
                        <span style="color:{color}; font-weight:600;">F1 {cm['f1_score']*100:.1f}%</span>
                    </div>
                    <div style="height:4px; background:var(--bg-input); border-radius:2px; overflow:hidden; margin-top:4px;">
                        <div style="width:{cm['f1_score']*100:.1f}%; height:100%; background:{color}; border-radius:2px;"></div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(_md(f"""
        <div class="disclaimer">
            {render_icon('alert-triangle', size=16)} <b>Prototype académique</b><br>
            Les prédictions n'ont pas de valeur diagnostique et ne remplacent pas l'avis d'un radiologue ou d'un médecin.
        </div>
        """), unsafe_allow_html=True)

    # ==============================================================
    # UPLOAD SECTION
    # ==============================================================
    st.markdown(_md(f"""
    <div class="card fade-in" style="margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            {render_icon('upload-cloud', size=24, color='var(--accent-primary)')}
            <h3 style="margin: 0;">Charger des radiographies</h3>
        </div>
        <p style="font-size: 14px; margin: 0; color: var(--text-secondary);">
        Déposez une ou plusieurs radiographies pulmonaires (PNG, JPG). Le modèle estime la classe
        la plus probable et génère une carte Grad-CAM des zones ayant influencé la décision.
        </p>
    </div>
    """), unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Sélectionner des images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # ==============================================================
    # PROCESSING
    # ==============================================================
    #
    # IMPORTANT (cache par lot de fichiers) : sans ce cache, TOUTE la boucle
    # ci-dessous (inference CNN, Grad-CAM, ET ecriture dans l'historique
    # SQLite via get_store().save()) se relancait a CHAQUE rerun du script —
    # y compris ceux declenches par des widgets totalement etrangers a cette
    # page (bouton clair/sombre et selecteur de langue dans la sidebar de
    # predict.py, qui appellent st.rerun()). Consequence concrete : chaque
    # changement de theme ou de langue dupliquait silencieusement les memes
    # analyses dans persistence.HistoryStore, en plus de re-executer un
    # modele TensorFlow inutilement (lent).
    #
    # CORRECTION (constat verifie en conditions reelles, pas seulement en
    # theorie) : contrairement a l'hypothese initiale, st.file_uploader NE
    # conserve PAS de facon fiable les fichiers uploades a travers un rerun
    # declenche par un widget d'une autre partie de la page (ici, la sidebar
    # de predict.py) — `uploaded_files` redevient vide sur ce rerun. Le cache
    # ci-dessous ne suffit donc pas a lui seul : il faut aussi POUVOIR
    # reafficher les resultats caches meme quand `uploaded_files` est vide
    # sur le run courant (voir le bloc `elif` juste apres, et le changement
    # de condition d'affichage plus bas : `if results:` au lieu de
    # `if uploaded_files and results:`).
    #
    # _files_fingerprint identifie un lot de fichiers par (nom, taille) —
    # suffisant pour detecter un nouvel upload sans avoir a hacher le
    # contenu binaire de chaque image a chaque rerun.
    results = []
    rejected = []
    zip_buffer = BytesIO()
    if uploaded_files:
        fingerprint = _files_fingerprint(uploaded_files)
        cache_hit = (
            st.session_state.get("_analysis_fingerprint") == fingerprint
            and "_analysis_results" in st.session_state
        )

        if cache_hit:
            results = st.session_state["_analysis_results"]
            rejected = st.session_state["_analysis_rejected"]
            zip_buffer = st.session_state["_analysis_zip_buffer"]
        else:
            st.markdown(_md(f"""
            <div class="card fade-in" style="display:flex; align-items:center; justify-content:space-between; padding: 16px 24px;">
                <div>
                    <span style="font-size: 18px; font-weight: 600; color: var(--text-primary);">
                        {render_icon('chart-line', size=16, color='var(--accent-primary)')} Analyse en cours
                    </span>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: var(--text-secondary);">
                        {len(uploaded_files)} image(s) sélectionnée(s)
                    </p>
                </div>
                <span class="badge badge-info pulse"><span class="icon-spin">{render_icon('loader', size=13)}</span> EN COURS</span>
            </div>
            """), unsafe_allow_html=True)

            zip_buffer = BytesIO()
            progress_bar = st.progress(0)
            rejected = []

            for i, uploaded_file in enumerate(uploaded_files):
                with st.status(f"Analyse de {uploaded_file.name}...", expanded=(i == 0), state="running") as status:
                    try:
                        step_start = time.perf_counter()
                        img_rgb, img_array, reject_reason = load_and_preprocess_image(uploaded_file, IMG_SIZE)
                        validation_ms = (time.perf_counter() - step_start) * 1000
                        if img_rgb is None or img_array is None:
                            rejected.append((uploaded_file.name, reject_reason or "Image non valide."))
                            status.update(label=f"{uploaded_file.name} — rejetée", state="error")
                            st.markdown(f"{render_icon('x-circle', size=14, color='var(--danger-text)')} Validation échouée ({validation_ms:.0f} ms) : {reject_reason or 'Image non valide.'}", unsafe_allow_html=True)
                            continue
                        st.markdown(f"{render_icon('check-circle', size=14, color='var(--success-text)')} Validation &amp; prétraitement ({validation_ms:.0f} ms)", unsafe_allow_html=True)

                        inference_start = time.perf_counter()
                        heatmap, pred_class, preds_all = gradcam(model, img_array, LAST_CONV_LAYER)
                        inference_ms = (time.perf_counter() - inference_start) * 1000
                        st.markdown(f"{render_icon('brain-circuit', size=14, color='var(--success-text)')} Prédiction CNN &amp; Grad-CAM ({inference_ms:.0f} ms)", unsafe_allow_html=True)

                        if heatmap is not None:
                            overlay_start = time.perf_counter()
                            overlay_img = overlay_heatmap_cv(img_rgb, heatmap, alpha=heatmap_alpha, colormap=colormap)

                            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                                if is_success:
                                    safe_name = uploaded_file.name.replace(' ', '_').replace('-', '_')
                                    base_name = os.path.splitext(safe_name)[0]
                                    timestamp = datetime.now().strftime('%H%M%S_%f')[:10]
                                    zip_file.writestr(f"gradcam_{base_name}_{timestamp}.png", BytesIO(buffer).getvalue())
                            overlay_ms = (time.perf_counter() - overlay_start) * 1000
                            st.markdown(f"{render_icon('flame', size=14, color='var(--success-text)')} Carte de chaleur &amp; export ZIP ({overlay_ms:.0f} ms)", unsafe_allow_html=True)

                            confidence = round(float(preds_all[pred_class]) * 100, 2)
                            class_probabilities = {name: round(float(prob) * 100, 2) for name, prob in zip(CLASS_NAMES, preds_all)}

                            report_start = time.perf_counter()
                            report = build_report(
                                image_name=uploaded_file.name,
                                predicted_class=CLASS_NAMES[pred_class],
                                confidence=confidence,
                                class_probabilities=class_probabilities,
                                language=get_language(),
                            )
                            report_ms = (time.perf_counter() - report_start) * 1000
                            st.markdown(f"{render_icon('file-text', size=14, color='var(--success-text)')} Rapport IA généré ({report_ms:.0f} ms)", unsafe_allow_html=True)

                            if config.enable_history:
                                try:
                                    from persistence import get_store
                                    history_start = time.perf_counter()
                                    report_dict = report.to_dict()
                                    get_store().save(
                                        image_name=uploaded_file.name,
                                        predicted_class=CLASS_NAMES[pred_class],
                                        confidence=confidence,
                                        class_probabilities=class_probabilities,
                                        findings=report_dict["findings"],
                                        impression=report_dict["impression"],
                                        recommendation=report_dict["recommendation"],
                                        inference_ms=inference_ms,
                                        original_image=img_rgb,
                                        overlay_image=overlay_img,
                                        heatmap_image=_colorized_heatmap_for_storage(img_rgb.shape, heatmap, colormap),
                                    )
                                    history_ms = (time.perf_counter() - history_start) * 1000
                                    st.markdown(f"{render_icon('database', size=14, color='var(--success-text)')} Enregistré dans l'historique ({history_ms:.0f} ms)", unsafe_allow_html=True)
                                except Exception as e:
                                    st.warning(f"{render_icon('alert-triangle', size=14, color='var(--warning-text)')} Analyse effectuée mais non enregistrée dans l'historique : {str(e)}")

                            result_dict = {
                                "Image": uploaded_file.name,
                                "Predicted Class": CLASS_NAMES[pred_class],
                                "Confidence": confidence,
                                "Overlay": overlay_img,
                                "Original": img_rgb,
                                "Heatmap": heatmap,
                                "Report": report,
                            }
                            for name, prob in class_probabilities.items():
                                result_dict[f"Prob_{name}"] = prob
                            results.append(result_dict)

                            color, _soft, label, _icon = class_color(CLASS_NAMES[pred_class])
                            total_ms = validation_ms + inference_ms + overlay_ms + report_ms
                            status.update(
                                label=f"{uploaded_file.name} — {label} ({confidence:.1f}%) · {total_ms:.0f} ms",
                                state="complete",
                            )

                    except Exception as e:
                        status.update(label=f"{uploaded_file.name} — erreur", state="error")
                        st.error(f"{render_icon('x-circle', size=14, color='var(--danger-text)')} Erreur avec {uploaded_file.name} : {str(e)}")
                        continue

                progress_bar.progress((i + 1) / len(uploaded_files))

            progress_bar.empty()

            st.session_state["_analysis_fingerprint"] = fingerprint
            st.session_state["_analysis_results"] = results
            st.session_state["_analysis_rejected"] = rejected
            st.session_state["_analysis_zip_buffer"] = zip_buffer

        if rejected:
            rejected_rows = "".join(
                f"""<div style='display:flex; justify-content:space-between; gap:16px; padding:8px 0; border-bottom:1px solid var(--border-color); font-size:12.5px;'>
                    <span style='color:var(--text-primary);'>{render_icon('image', size=14)} {name}</span>
                    <span style='color:var(--warning-text); text-align:right;'>{render_icon('alert-circle', size=14)} {reason}</span>
                </div>"""
                for name, reason in rejected
            )
            st.markdown(_md(f"""
            <div class='card' style='border-color: rgba(253, 126, 20, 0.3);'>
                <h4 style='margin:0 0 10px 0; color:var(--warning-text);'>{render_icon('alert-triangle', size=16)} {len(rejected)} image(s) écartée(s)</h4>
                {rejected_rows}
            </div>
            """), unsafe_allow_html=True)

        if not results:
            st.markdown(_md(f"""
            <div class='card' style='border-color: rgba(253, 126, 20, 0.3);'>
                <h4 style='margin:0; color:var(--warning-text);'>{render_icon('alert-triangle', size=16)} Aucune analyse valide</h4>
                <p style='margin:8px 0 0 0; font-size:13px; color:var(--text-secondary);'>
                Aucune image exploitable n'a été trouvée. Vérifiez qu'il s'agit bien de radiographies en niveaux de gris, au format PNG/JPG.
                </p>
            </div>
            """), unsafe_allow_html=True)
            st.stop()
    elif "_analysis_results" in st.session_state:
        # Pas de fichier renvoye par st.file_uploader sur CE run precis, mais
        # une analyse precedente existe dans cette session : on la reaffiche
        # plutot que de tout vider. C'est le coeur du correctif — voir le
        # commentaire plus haut au-dessus de `results = []`.
        results = st.session_state["_analysis_results"]
        rejected = st.session_state.get("_analysis_rejected", [])
        zip_buffer = st.session_state.get("_analysis_zip_buffer", zip_buffer)

    # ==============================================================
    # RESULTS
    # ==============================================================
    if results:
        df_results = pd.DataFrame(results)
        df_sorted = df_results.sort_values("Confidence", ascending=False)

        st.markdown(_md(f"""
        <div style='margin: 36px 0 16px 0;'>
            <h2>{render_icon('chart-pie', size=16, color='var(--accent-primary)')} Résultats de l'analyse</h2>
        </div>
        """), unsafe_allow_html=True)

        # Metrics Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(_md(f"""
            <div class="metric-card">
                <div class="metric-value">{len(results)}</div>
                <div class="metric-label">{render_icon('layers', size=14)} Images traitées</div>
            </div>
            """), unsafe_allow_html=True)

        with col2:
            avg_conf = df_results["Confidence"].mean() if len(df_results) > 0 else 0
            conf_color = confidence_color(avg_conf)
            st.markdown(_md(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{conf_color};">{avg_conf:.1f}%</div>
                <div class="metric-label">{render_icon('gauge', size=14)} Confiance moyenne</div>
            </div>
            """), unsafe_allow_html=True)

        with col3:
            most_common = df_results["Predicted Class"].mode()[0] if len(df_results) > 0 else "N/A"
            color, soft, label, icon = class_color(most_common) if most_common != "N/A" else ("#64748B", "rgba(100,116,139,0.10)", "N/A", "bar-chart-3")
            st.markdown(_md(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{render_icon(icon, size=20, color=color)} {label}</div>
                <div class="metric-label">{render_icon('stethoscope', size=14)} Diagnostic dominant</div>
            </div>
            """), unsafe_allow_html=True)

        with col4:
            covid_count = len(df_results[df_results["Predicted Class"] == "COVID"])
            covid_percent = (covid_count / len(results)) * 100 if len(results) > 0 else 0
            st.markdown(_md(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:var(--danger-text);">{covid_count}</div>
                <div class="metric-label">{render_icon('shield-check', size=14)} Cas COVID-19 ({covid_percent:.0f}%)</div>
            </div>
            """), unsafe_allow_html=True)

        # Charts
        st.markdown(_md(f"""
        <div style='margin: 32px 0 16px 0;'>
            <h3>{render_icon('bar-chart-3', size=16)} Visualisation des résultats</h3>
        </div>
        """), unsafe_allow_html=True)
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            with st.container(border=True):
                st.markdown(_md(f"""
                <h4 style='margin-bottom:14px;'>{render_icon('chart-pie', size=14)} Distribution par classe</h4>
                """), unsafe_allow_html=True)
                class_dist = df_results["Predicted Class"].value_counts()
                if len(class_dist) > 0:
                    colors_map = {c: class_color(c)[0] for c in CLASS_NAMES}
                    fig_dist = px.pie(
                        values=class_dist.values,
                        names=class_dist.index,
                        color=class_dist.index,
                        color_discrete_map=colors_map,
                        hole=0.50,
                    )
                    fig_dist.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                        insidetextfont=dict(size=12, color="#FFFFFF", weight=600),
                        marker=dict(line=dict(color="#FFFFFF", width=2)),
                    )
                    fig_dist = plotly_light_layout(fig_dist)
                    st.plotly_chart(fig_dist, width='stretch')
                else:
                    st.info("Aucune donnée à afficher pour la distribution par classe.")

        with col_chart2:
            with st.container(border=True):
                st.markdown(_md(f"""
                <h4 style='margin-bottom:14px;'>{render_icon('bar-chart-3', size=14)} Distribution des scores de confiance</h4>
                """), unsafe_allow_html=True)
                if len(df_results) > 0:
                    counts, bin_edges = np.histogram(df_results["Confidence"], bins=15, range=(0, 100))
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bar_colors = [
                        "#047857" if c >= 80 else "#92400E" if c >= 60 else "#B91C1C"
                        for c in bin_centers
                    ]
                    fig_conf = go.Figure(data=[go.Bar(
                        x=bin_centers,
                        y=counts,
                        marker=dict(color=bar_colors, line=dict(width=0)),
                        width=(bin_edges[1] - bin_edges[0]) * 0.9,
                        opacity=0.85,
                    )])
                    avg_conf = df_results["Confidence"].mean()
                    fig_conf.add_vline(
                        x=avg_conf, line_dash="dash", line_color="#0F172A", line_width=2,
                        annotation_text=f"moyenne {avg_conf:.1f}%",
                        annotation_position="top right",
                        annotation_font_size=11,
                        annotation_font_color="#0F172A",
                    )
                    fig_conf.update_layout(
                        xaxis_title="Confiance (%)",
                        yaxis_title="Nombre d'images",
                        bargap=0.12,
                    )
                    fig_conf = plotly_light_layout(fig_conf)
                    st.plotly_chart(fig_conf, width='stretch')
                else:
                    st.info("Aucune donnée à afficher pour la distribution des confiances.")

        # Per-class breakdown
        st.markdown(_md(f"""
        <div style='margin: 32px 0 16px 0;'>
            <h3>{render_icon('list', size=16)} Détail par classe</h3>
        </div>
        """), unsafe_allow_html=True)
        class_cols = st.columns(4)
        for idx, cname in enumerate(CLASS_NAMES):
            class_data = df_results[df_results["Predicted Class"] == cname]
            count = len(class_data)
            percent = (count / len(df_results) * 100) if len(df_results) > 0 else 0
            avg = class_data["Confidence"].mean() if count > 0 else 0
            color, soft, label, icon = class_color(cname)

            with class_cols[idx]:
                st.markdown(_md(f"""
                <div class='card' style='height:100%; border-left: 4px solid {color};'>
                    <p style='margin:0 0 12px 0; font-weight:600; color:{color}; font-size:14px;'>
                        {render_icon(icon, size=15, color=color)} {label}
                    </p>
                    <div style='display:flex; justify-content:space-between; font-size:13px; margin-bottom:6px;'>
                        <span style='color:var(--text-muted);'>Cas détectés</span>
                        <span style='color:{color}; font-weight:600;'>{count}</span>
                    </div>
                    <div style='height:4px; background:var(--bg-input); border-radius:2px; margin-bottom:12px; overflow:hidden;'>
                        <div style='width:{percent}%; height:100%; background:{color}; border-radius:2px;'></div>
                    </div>
                    <div style='display:flex; justify-content:space-between; font-size:13px;'>
                        <span style='color:var(--text-muted);'>Confiance moy.</span>
                        <span style='color:var(--text-primary);'>{avg:.1f}%</span>
                    </div>
                </div>
                """), unsafe_allow_html=True)

        # Grad-CAM Gallery
        st.markdown(_md(f"""
        <div style='margin: 40px 0 16px 0;'>
            <h2>{render_icon('flame', size=16, color='var(--accent-primary)')} Cartes Grad-CAM</h2>
        </div>
        """), unsafe_allow_html=True)

        st.markdown(_md(f"""
        <div class="disclaimer" style="margin-top:0;">
            {render_icon('info', size=16)} <b>Interprétation.</b> La heatmap met en évidence les régions de l'image ayant le plus
            influencé la prédiction du modèle. Elle ne constitue pas une segmentation de la lésion
            et ne doit pas être interprétée comme une localisation clinique exacte.
        </div>
        """), unsafe_allow_html=True)

        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            color, soft, label, icon = class_color(row["Predicted Class"])
            fname = row["Image"] if len(row["Image"]) <= 40 else row["Image"][:37] + "..."
            confidence = row["Confidence"]

            st.markdown(_md(f"""
            <div class='card' style='padding:16px 20px 4px 20px; margin-bottom:12px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>
                    <span style='font-size:14px; font-weight:500; color:var(--text-primary);'>
                        {render_icon('image', size=14)} {fname}
                    </span>
                    <span class="badge badge-info">{label} • {confidence:.1f}%</span>
                </div>
            </div>
            """), unsafe_allow_html=True)

            card_alpha = st.slider(
                "Intensité", 0.0, 1.0, float(heatmap_alpha), 0.05,
                key=f"alpha_{idx}", label_visibility="collapsed",
                help="Ajuste l'opacité de la fusion Grad-CAM pour cette image uniquement",
            )
            heatmap_only = colorize_heatmap(row["Original"].shape, row["Heatmap"], colormap)
            live_overlay = overlay_heatmap_cv(row["Original"], row["Heatmap"], alpha=card_alpha, colormap=colormap)

            if live_overlay.dtype != np.uint8:
                live_overlay = (live_overlay * 255).astype(np.uint8) if live_overlay.max() <= 1.0 else live_overlay.astype(np.uint8)
            if heatmap_only.dtype != np.uint8:
                heatmap_only = (heatmap_only * 255).astype(np.uint8) if heatmap_only.max() <= 1.0 else heatmap_only.astype(np.uint8)

            # Images avec icônes au lieu d'émojis
            img_col1, img_col2, img_col3 = st.columns(3)
            with img_col1:
                st.image(row["Original"], width='stretch', caption=f"{render_icon('image', size=12)} Original", output_format="auto")
            with img_col2:
                st.image(heatmap_only, width='stretch', caption=f"{render_icon('flame', size=12)} Grad-CAM", output_format="auto")
            with img_col3:
                st.image(live_overlay, width='stretch', caption=f"{render_icon('layers', size=12)} Fusion", output_format="auto")

            row_probabilities = {name: row.get(f"Prob_{name}", 0.0) for name in CLASS_NAMES}
            gauge_col, bars_col = st.columns([1, 2])
            with gauge_col:
                confidence_gauge(confidence, label="Confiance")
            with bars_col:
                st.markdown("<div style='padding-top:8px;'>", unsafe_allow_html=True)
                probability_bars(row_probabilities, class_color)
                st.markdown("</div>", unsafe_allow_html=True)

            target_layer_display = LAST_CONV_LAYER if LAST_CONV_LAYER else "non détectée"
            st.markdown(_md(f"""
            <div class="readout-strip" style="margin: 8px 0 16px 0;">
                <div class="readout-item">
                    <div class="readout-label">{render_icon('target', size=13)} Classe ciblée</div>
                    <div class="readout-value" style="font-size:14px; color:{color};">{label}</div>
                </div>
                <div class="readout-item">
                    <div class="readout-label">{render_icon('gauge', size=13)} Confiance</div>
                    <div class="readout-value" style="font-size:14px;">{confidence:.1f}%</div>
                </div>
                <div class="readout-item">
                    <div class="readout-label">{render_icon('layers', size=13)} Couche cible</div>
                    <div class="readout-value" style="font-size:13px; font-weight:400; color:var(--text-secondary);">{target_layer_display}</div>
                </div>
                <div class="readout-item">
                    <div class="readout-label">{render_icon('brain-circuit', size=13)} Méthode</div>
                    <div class="readout-value" style="font-size:14px;">Grad-CAM</div>
                </div>
            </div>
            """), unsafe_allow_html=True)

            safe_name = os.path.splitext(row["Image"])[0].replace(" ", "_").replace("-", "_")
            png_col1, png_col2, png_col3 = st.columns(3)
            for col, img, suffix, dl_label in [
                (png_col1, row["Original"], "original", "Original"),
                (png_col2, heatmap_only, "heatmap", "Heatmap"),
                (png_col3, live_overlay, "overlay", "Fusion"),
            ]:
                with col:
                    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if ok:
                        st.markdown(_md(f"""
                        <div style="display: flex; justify-content: center;">
                            <a href="data:image/png;base64,{base64.b64encode(buf).decode()}" download="{safe_name}_{suffix}.png" style="text-decoration: none;">
                                <button style="background: var(--accent-gradient); color: var(--on-accent); border: none; border-radius: 8px; padding: 8px 16px; font-weight: 600; font-size: 13px; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; box-shadow: 0 2px 8px rgba(194,121,12,0.2); transition: all 0.2s;">
                                    {render_icon('download', size=14)} {dl_label}
                                </button>
                            </a>
                        </div>
                        """), unsafe_allow_html=True)

            # Report avec icône
            with st.expander(f"{render_icon('file-text', size=14)} Rapport IA détaillé"):
                report = row["Report"]
                report_dict = report.to_dict() if hasattr(report, 'to_dict') else report
            
                st.markdown(_md(f"""
                <div style="display:grid; gap:12px;">
                    <div style="background:var(--bg-input); padding:12px 16px; border-radius:var(--radius-sm); border-left: 4px solid #2563EB;">
                        <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;">
                            {render_icon('eye', size=14)} Observations
                        </div>
                        <div style="font-size:14px; color:var(--text-primary); margin-top:4px;">{report_dict.get('findings', 'N/A')}</div>
                    </div>
                    <div style="background:var(--bg-input); padding:12px 16px; border-radius:var(--radius-sm); border-left: 4px solid #60A5FA;">
                        <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;">
                            {render_icon('message-circle', size=14)} Impression
                        </div>
                        <div style="font-size:14px; color:var(--text-primary); margin-top:4px;">{report_dict.get('impression', 'N/A')}</div>
                    </div>
                    <div style="background:var(--bg-input); padding:12px 16px; border-radius:var(--radius-sm); border-left: 4px solid #10B981;">
                        <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;">
                            {render_icon('shield-check', size=14)} Recommandation
                        </div>
                        <div style="font-size:14px; color:var(--text-primary); margin-top:4px;">{report_dict.get('recommendation', 'N/A')}</div>
                    </div>
                </div>
                """), unsafe_allow_html=True)

                pdf_bytes = generate_pdf_report(report, row["Original"], live_overlay, heatmap_only)
                st.download_button(
                    label=f"{render_icon('file-text', size=14)} Télécharger le rapport (PDF)",
                    data=pdf_bytes,
                    file_name=f"rapport_{safe_name}.pdf",
                    mime="application/pdf",
                    help="Rapport complet pour cette image : original, Grad-CAM, fusion et observations",
                    key=f"pdf_dl_{idx}",
                    width='stretch',
                )

            st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

        # Export Section - avec icônes
        st.markdown(_md(f"""
        <div style='margin: 40px 0 16px 0;'>
            <h2>{render_icon('archive', size=16, color='var(--accent-primary)')} Exporter les résultats</h2>
        </div>
        """), unsafe_allow_html=True)
    
        col_dl1, col_dl2, col_dl3 = st.columns(3)
    
        with col_dl1:
            st.download_button(
                label=f"{render_icon('archive', size=14)} Visualisations (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"radiology_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                help="Toutes les radiographies avec superposition Grad-CAM",
                width='stretch',
            )
    
        with col_dl2:
            csv_columns = [col for col in df_results.columns if col not in ["Overlay", "Original", "Heatmap", "Report"]]
            csv_data = df_results[csv_columns].to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label=f"{render_icon('bar-chart-3', size=14)} Rapport CSV",
                data=csv_data.encode("utf-8-sig"),
                file_name=f"radiology_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Prédictions détaillées avec probabilités par classe",
                width='stretch',
            )
    
        with col_dl3:
            all_reports_json = json.dumps(
                [row["Report"].to_dict() if hasattr(row["Report"], 'to_dict') else row["Report"] 
                 for _, row in df_results.iterrows()],
                indent=2, ensure_ascii=False,
            )
            st.download_button(
                label=f"{render_icon('file-text', size=14)} Rapports JSON",
                data=all_reports_json.encode("utf-8"),
                file_name=f"radiology_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Rapports structurés pour chaque image",
                width='stretch',
            )

        # Message de succès avec icône professionnelle
        st.markdown(_md(f"""
        <div class="card" style="border-left: 4px solid var(--success-text); margin-top: 24px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                {render_icon('check-circle', size=24, color='var(--success-text)')}
                <div>
                    <strong style="color: var(--success-text); font-size: 16px;">Analyse terminée</strong>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: var(--text-secondary);">
                        {len(results)} radiographie(s) traitée(s) avec succès
                    </p>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    else:
        # ==============================================================
        # HOME STATE
        # ==============================================================
        col_welcome1, col_welcome2 = st.columns([2, 1])

        with col_welcome1:
            st.markdown(_md(f"""
            <div class="card" style="height:100%;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                    {render_icon('rocket', size=28, color='var(--accent-primary)')}
                    <h3 style="margin:0;">Comment ça fonctionne</h3>
                </div>
                <p style="font-size:14px; line-height:1.8; color:var(--text-secondary);">
                Ce prototype classe des radiographies pulmonaires en quatre catégories à l'aide d'un
                réseau de neurones convolutif entraîné sur un jeu de données public de radiographies
                thoraciques. Pour chaque image, le modèle produit une probabilité par classe et une
                carte Grad-CAM localisant les régions ayant motivé sa prédiction.
                </p>
                <div style="display:grid; gap:8px; margin-top:12px;">
                    <div style="display:flex; align-items:center; gap:12px; padding:8px 12px; background:var(--bg-input); border-radius:var(--radius-sm);">
                        <span style="font-size:18px; font-weight:700; color:var(--accent-primary); width: 28px; text-align: center;">1</span>
                        <span style="color:var(--text-primary);">{render_icon('upload-cloud', size=14)} Déposer une ou plusieurs radiographies (PNG/JPG)</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px; padding:8px 12px; background:var(--bg-input); border-radius:var(--radius-sm);">
                        <span style="font-size:18px; font-weight:700; color:var(--accent-primary); width: 28px; text-align: center;">2</span>
                        <span style="color:var(--text-primary);">{render_icon('calculator', size=14)} Le modèle calcule les probabilités par classe</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px; padding:8px 12px; background:var(--bg-input); border-radius:var(--radius-sm);">
                        <span style="font-size:18px; font-weight:700; color:var(--accent-primary); width: 28px; text-align: center;">3</span>
                        <span style="color:var(--text-primary);">{render_icon('flame', size=14)} Une carte Grad-CAM localise les zones d'intérêt</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px; padding:8px 12px; background:var(--bg-input); border-radius:var(--radius-sm);">
                        <span style="font-size:18px; font-weight:700; color:var(--accent-primary); width: 28px; text-align: center;">4</span>
                        <span style="color:var(--text-primary);">{render_icon('download', size=14)} Export du rapport (CSV) et des visualisations (ZIP)</span>
                    </div>
                </div>
            </div>
            """), unsafe_allow_html=True)

        with col_welcome2:
            acc_line = f"{OVERALL_ACCURACY:.1f}%" if OVERALL_ACCURACY else "N/A"
            st.markdown(_md(f"""
            <div class="card" style="height:100%; text-align:center;">
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; padding:16px 0;">
                    {render_icon('target', size=42, color='var(--accent-primary)')}
                    <p style="margin:8px 0 4px 0; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted);">
                        Exactitude mesurée
                    </p>
                    <p style="margin:0; font-size:38px; font-weight:700; background:var(--accent-gradient); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                        {acc_line}
                    </p>
                    <p style="font-size:12px; color:var(--text-muted); margin:4px 0 0 0;">
                        {render_icon('database', size=14)} {DATASET_SIZE:,} radiographies utilisées
                    </p>
                </div>
            </div>
            """).replace(",", "\u2009"), unsafe_allow_html=True)

        st.markdown(_md(f"""
        <div style='margin: 32px 0 16px 0;'>
            <h3>{render_icon('sparkles', size=16)} Fonctionnalités</h3>
        </div>
        """), unsafe_allow_html=True)
    
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        features = [
            (f"{render_icon('brain-circuit', size=14)} Classification 4 classes", "COVID-19, opacité pulmonaire, pneumonie virale, ou normal, avec un score de confiance par classe."),
            (f"{render_icon('search', size=14)} Explicabilité Grad-CAM", "Cartes de chaleur superposées à la radiographie originale, pour visualiser les zones jugées pertinentes par le modèle."),
            (f"{render_icon('download', size=14)} Rapports exportables", "Export CSV des prédictions détaillées et ZIP des visualisations, pour analyse ou archivage."),
        ]
        for col, (title, desc) in zip([feat_col1, feat_col2, feat_col3], features):
            with col:
                st.markdown(_md(f"""
                <div class="card" style="height:100%;">
                    <h4 style="font-size:15px; margin-bottom:8px;">{title}</h4>
                    <p style="font-size:13px; margin:0; color:var(--text-secondary);">{desc}</p>
                </div>
                """), unsafe_allow_html=True)