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
        st.error(f"❌ Modèle introuvable : {MODEL_PATH}")
        return None
    try:
        with st.spinner("🔄 Chargement du modèle en cours..."):
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
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
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
        st.warning(f"Erreur de mise en page du graphique : {e}")
        return fig




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
    # HEADER AVEC ICÔNES FONT AWESOME
    # ==============================================================
    st.markdown(f"""
    <div class="main-header fade-in">
        <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
            <div style="flex: 1;">
                <h1><span style="margin-right: 8px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 3v6"/><path d="M9 6h6"/><path d="M2 22V9a2 2 0 0 1 2-2h4v15"/><path d="M22 22V9a2 2 0 0 0-2-2h-4v15"/><path d="M2 22h20"/><path d="M9 14h6"/><path d="M12 11v6"/></svg></span> AI Radiology Analyzer</h1>
                <p class="subtitle">Classification de radiographies pulmonaires assistée par IA avec visualisation Grad-CAM</p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="badge badge-info"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg> Recherche</span>
                <span class="badge badge-success"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2"/><path d="M15 2v2"/><path d="M9 20v2"/><path d="M15 20v2"/><path d="M2 9h2"/><path d="M2 15h2"/><path d="M20 9h2"/><path d="M20 15h2"/></svg> CNN</span>
                <span class="badge badge-warning"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg> 4 classes</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    accuracy_display = f"{OVERALL_ACCURACY:.1f}%" if OVERALL_ACCURACY else "N/A"

    st.markdown(f"""
    <div class="readout-strip fade-in">
        <div class="readout-item">
            <div class="readout-label"><span style="margin-right: 6px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/></svg></span> Modèle</div>
            <div class="readout-value">CNN • 4 classes</div>
        </div>
        <div class="readout-item">
            <div class="readout-label"><span style="margin-right: 6px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg></span> Résolution</div>
            <div class="readout-value">{IMG_SIZE[0]}×{IMG_SIZE[1]}</div>
        </div>
        <div class="readout-item">
            <div class="readout-label"><span style="margin-right: 6px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></svg></span> Exactitude</div>
            <div class="readout-value" style="color: {confidence_color(OVERALL_ACCURACY) if OVERALL_ACCURACY else 'var(--danger-text)'};">{accuracy_display}</div>
        </div>
        <div class="readout-item">
            <div class="readout-label"><span style="margin-right: 6px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg></span> Images entraînement</div>
            <div class="readout-value">{DATASET_SIZE:,}</div>
        </div>
    </div>
    """.replace(",", "\u2009"), unsafe_allow_html=True)

    # ==============================================================
    # SIDEBAR AVEC ICÔNES FONT AWESOME
    # ==============================================================
    with st.sidebar:
        st.markdown("""#### <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 2H2v10l9.3 9.3a1 1 0 0 0 1.4 0l8.6-8.6a1 1 0 0 0 0-1.4L12 2Z"/><circle cx="7" cy="7" r="1"/></svg> Classes détectées""", unsafe_allow_html=True)
        for cname in CLASS_NAMES:
            color, soft, label, icon = class_color(cname)
            st.markdown(f"""
            <div class="legend-item" style="--legend-color:{color};">
                <div class="legend-dot" style="background-color:{color};"></div>
                <span class="legend-text">{render_icon(icon, size=14)} {label}</span>
                <span class="legend-sub">{CLASS_META[cname]['description']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""#### <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="13.5" cy="6.5" r=".5"/><circle cx="17.5" cy="10.5" r=".5"/><circle cx="8.5" cy="7.5" r=".5"/><circle cx="6.5" cy="12.5" r=".5"/><path d="M12 2a10 10 0 1 0 0 20 2.5 2.5 0 0 0 2-4 2.5 2.5 0 0 1 2-4h1.5a2.5 2.5 0 0 0 2.5-2.5C20 6 16.4 2 12 2Z"/></svg> Visualisation Grad-CAM""", unsafe_allow_html=True)
    
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
            st.markdown("""#### <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg> Performance par classe""", unsafe_allow_html=True)
            for cname in CLASS_NAMES:
                cm = METRICS["classes"].get(cname)
                if not cm:
                    continue
                color, soft, label, icon = class_color(cname)
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px;">
                        <span style="color:var(--text-primary);">{render_icon(icon, size=13)} {label}</span>
                        <span style="color:{color}; font-weight:600;">F1 {cm['f1_score']*100:.1f}%</span>
                    </div>
                    <div style="height:4px; background:var(--bg-input); border-radius:2px; overflow:hidden; margin-top:4px;">
                        <div style="width:{cm['f1_score']*100:.1f}%; height:100%; background:{color}; border-radius:2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="disclaimer">
            <span style="margin-right: 6px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="m21.7 18-8-14.3a2 2 0 0 0-3.4 0L2.3 18a2 2 0 0 0 1.7 3h16a2 2 0 0 0 1.7-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg></span> <b>Prototype académique</b><br>
            Les prédictions n'ont pas de valeur diagnostique et ne remplacent pas l'avis d'un radiologue ou d'un médecin.
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================
    # UPLOAD SECTION
    # ==============================================================
    st.markdown("""
    <div class="card fade-in" style="margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M16 16l-4-4-4 4"/><path d="M12 12v9"/><path d="M20.4 17.6A5 5 0 0 0 18 8h-1.3A7 7 0 1 0 4 15.3"/></svg>
            <h3 style="margin: 0;">Charger des radiographies</h3>
        </div>
        <p style="font-size: 14px; margin: 0; color: var(--text-secondary);">
        Déposez une ou plusieurs radiographies pulmonaires (PNG, JPG). Le modèle estime la classe
        la plus probable et génère une carte Grad-CAM des zones ayant influencé la décision.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Sélectionner des images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # ==============================================================
    # PROCESSING
    # ==============================================================
    results = []
    if uploaded_files:
        st.markdown(f"""
        <div class="card fade-in" style="display:flex; align-items:center; justify-content:space-between; padding: 16px 24px;">
            <div>
                <span style="font-size: 18px; font-weight: 600; color: var(--text-primary);">
                    <span style="margin-right: 8px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg></span> Analyse en cours
                </span>
                <p style="margin: 4px 0 0 0; font-size: 13px; color: var(--text-secondary);">
                    {len(uploaded_files)} image(s) sélectionnée(s)
                </p>
            </div>
            <span class="badge badge-info pulse"><span class="icon-spin"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.9" y1="4.9" x2="7.8" y2="7.8"/><line x1="16.2" y1="16.2" x2="19.1" y2="19.1"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.9" y1="19.1" x2="7.8" y2="16.2"/><line x1="16.2" y1="7.8" x2="19.1" y2="4.9"/></svg></span> EN COURS</span>
        </div>
        """, unsafe_allow_html=True)

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
                                )
                                history_ms = (time.perf_counter() - history_start) * 1000
                                st.markdown(f"{render_icon('database', size=14, color='var(--success-text)')} Enregistré dans l'historique ({history_ms:.0f} ms)", unsafe_allow_html=True)
                            except Exception as e:
                                # L'historique est une fonctionnalite secondaire : un souci de
                                # persistance ne doit jamais faire echouer l'analyse elle-meme.
                                st.warning(f"⚠️ Analyse effectuée mais non enregistrée dans l'historique : {str(e)}")

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
                    st.error(f"❌ Erreur avec {uploaded_file.name} : {str(e)}")
                    continue

            progress_bar.progress((i + 1) / len(uploaded_files))

        progress_bar.empty()

        if rejected:
            rejected_rows = "".join(
                f"""<div style='display:flex; justify-content:space-between; gap:16px; padding:8px 0; border-bottom:1px solid var(--border-color); font-size:12.5px;'>
                    <span style='color:var(--text-primary);'><span style="margin-right: 6px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/></svg></span>{name}</span>
                    <span style='color:var(--warning-text); text-align:right;'><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="12" r="9"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg> {reason}</span>
                </div>"""
                for name, reason in rejected
            )
            st.markdown(f"""
            <div class='card' style='border-color: rgba(253, 126, 20, 0.3);'>
                <h4 style='margin:0 0 10px 0; color:var(--warning-text);'><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="m21.7 18-8-14.3a2 2 0 0 0-3.4 0L2.3 18a2 2 0 0 0 1.7 3h16a2 2 0 0 0 1.7-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg> {len(rejected)} image(s) écartée(s)</h4>
                {rejected_rows}
            </div>
            """, unsafe_allow_html=True)

        if not results:
            st.markdown("""
            <div class='card' style='border-color: rgba(253, 126, 20, 0.3);'>
                <h4 style='margin:0; color:var(--warning-text);'><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="m21.7 18-8-14.3a2 2 0 0 0-3.4 0L2.3 18a2 2 0 0 0 1.7 3h16a2 2 0 0 0 1.7-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg> Aucune analyse valide</h4>
                <p style='margin:8px 0 0 0; font-size:13px; color:var(--text-secondary);'>
                Aucune image exploitable n'a été trouvée. Vérifiez qu'il s'agit bien de radiographies en niveaux de gris, au format PNG/JPG.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

    # ==============================================================
    # RESULTS
    # ==============================================================
    if uploaded_files and results:
        df_results = pd.DataFrame(results)
        df_sorted = df_results.sort_values("Confidence", ascending=False)

        st.markdown("""<div style='margin: 36px 0 16px 0;'><h2><span style="margin-right: 10px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg></span> Résultats de l'analyse</h2></div>""", unsafe_allow_html=True)

        # Metrics Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(results)}</div>
                <div class="metric-label"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="m12 2 9 5-9 5-9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></svg> Images traitées</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            avg_conf = df_results["Confidence"].mean() if len(df_results) > 0 else 0
            conf_color = confidence_color(avg_conf)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{conf_color};">{avg_conf:.1f}%</div>
                <div class="metric-label"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M21.8 10A10 10 0 1 1 17 3.34"/><path d="m9 11 3 3L22 4"/></svg> Confiance moyenne</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            most_common = df_results["Predicted Class"].mode()[0] if len(df_results) > 0 else "N/A"
            color, soft, label, icon = class_color(most_common) if most_common != "N/A" else ("#64748B", "rgba(100,116,139,0.10)", "N/A", "bar-chart-3")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{render_icon(icon, size=20, color=color)} {label}</div>
                <div class="metric-label"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/></svg> Diagnostic dominant</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            covid_count = len(df_results[df_results["Predicted Class"] == "COVID"])
            covid_percent = (covid_count / len(results)) * 100 if len(results) > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:var(--danger-text);">{covid_count}</div>
                <div class="metric-label"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5Z"/></svg> Cas COVID-19 ({covid_percent:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        # Charts
        st.markdown("""<div style='margin: 32px 0 16px 0;'><h3><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg> Visualisation des résultats</h3></div>""", unsafe_allow_html=True)
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            with st.container(border=True):
                st.markdown("""<h4 style='margin-bottom:14px;'><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg> Distribution par classe</h4>""", unsafe_allow_html=True)
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
                st.markdown("""<h4 style='margin-bottom:14px;'><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg> Distribution des scores de confiance</h4>""", unsafe_allow_html=True)
                if len(df_results) > 0:
                    # Couleur par tranche de confiance (rouge < 60 %, orange 60-80 %, vert >= 80 %) :
                    # coherent avec le code couleur deja utilise pour "Confiance moyenne" plus haut.
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
        st.markdown("""<div style='margin: 32px 0 16px 0;'><h3><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg> Détail par classe</h3></div>""", unsafe_allow_html=True)
        class_cols = st.columns(4)
        for idx, cname in enumerate(CLASS_NAMES):
            class_data = df_results[df_results["Predicted Class"] == cname]
            count = len(class_data)
            percent = (count / len(df_results) * 100) if len(df_results) > 0 else 0
            avg = class_data["Confidence"].mean() if count > 0 else 0
            color, soft, label, icon = class_color(cname)

            with class_cols[idx]:
                st.markdown(f"""
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
                """, unsafe_allow_html=True)

        # Grad-CAM Gallery
        st.markdown("""<div style='margin: 40px 0 16px 0;'><h2><span style="margin-right: 10px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5Z"/></svg></span> Cartes Grad-CAM</h2></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer" style="margin-top:0;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg> <b>Interprétation.</b> La heatmap met en évidence les régions de l'image ayant le plus
            influencé la prédiction du modèle. Elle ne constitue pas une segmentation de la lésion
            et ne doit pas être interprétée comme une localisation clinique exacte.
        </div>
        """, unsafe_allow_html=True)

        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            color, soft, label, icon = class_color(row["Predicted Class"])
            fname = row["Image"] if len(row["Image"]) <= 40 else row["Image"][:37] + "..."
            confidence = row["Confidence"]

            st.markdown(f"""
            <div class='card' style='padding:16px 20px 4px 20px; margin-bottom:12px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>
                    <span style='font-size:14px; font-weight:500; color:var(--text-primary);'>
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/></svg> {fname}
                    </span>
                    <span class="badge badge-info">{label} • {confidence:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            card_alpha = st.slider(
                "Intensité", 0.0, 1.0, float(heatmap_alpha), 0.05,
                key=f"alpha_{idx}", label_visibility="collapsed",
                help="Ajuste l'opacité de la fusion Grad-CAM pour cette image uniquement",
            )
            live_overlay = overlay_heatmap_cv(row["Original"], row["Heatmap"], alpha=card_alpha, colormap=colormap)

            # Vérification que les images sont bien en RGB uint8
            if live_overlay.dtype != np.uint8:
                live_overlay = (live_overlay * 255).astype(np.uint8) if live_overlay.max() <= 1.0 else live_overlay.astype(np.uint8)

            # Affichage de 2 colonnes : Original et Grad-CAM (fusion renommée)
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(row["Original"], width='stretch', caption="🖼️ Original", output_format="auto")
            with img_col2:
                st.image(live_overlay, width='stretch', caption="🧬 Grad-CAM", output_format="auto")

            # Jauge de confiance + barres de probabilite par classe
            row_probabilities = {name: row.get(f"Prob_{name}", 0.0) for name in CLASS_NAMES}
            gauge_col, bars_col = st.columns([1, 2])
            with gauge_col:
                confidence_gauge(confidence, label="Confiance")
            with bars_col:
                st.markdown("<div style='padding-top:8px;'>", unsafe_allow_html=True)
                probability_bars(row_probabilities, class_color)
                st.markdown("</div>", unsafe_allow_html=True)

            target_layer_display = LAST_CONV_LAYER if LAST_CONV_LAYER else "non détectée"
            st.markdown(f"""
            <div class="readout-strip" style="margin: 8px 0 16px 0;">
                <div class="readout-item">
                    <div class="readout-label"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></svg> Classe ciblée</div>
                    <div class="readout-value" style="font-size:14px; color:{color};">{label}</div>
                </div>
                <div class="readout-item">
                    <div class="readout-label"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M21.8 10A10 10 0 1 1 17 3.34"/><path d="m9 11 3 3L22 4"/></svg> Confiance</div>
                    <div class="readout-value" style="font-size:14px;">{confidence:.1f}%</div>
                </div>
                <div class="readout-item">
                    <div class="readout-label"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="m12 2 9 5-9 5-9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></svg> Couche cible</div>
                    <div class="readout-value" style="font-size:13px; font-weight:400; color:var(--text-secondary);">{target_layer_display}</div>
                </div>
                <div class="readout-item">
                    <div class="readout-label"><svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/></svg> Méthode</div>
                    <div class="readout-value" style="font-size:14px;">Grad-CAM</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # PNG Downloads - Boutons améliorés avec espacement contrôlé
            safe_name = os.path.splitext(row["Image"])[0].replace(" ", "_").replace("-", "_")
            
            # Conteneur pour les boutons de téléchargement avec espacement contrôlé
            st.markdown("""
            <style>
            .download-buttons-container {
                display: flex;
                gap: 16px;
                justify-content: center;
                margin: 8px 0;
                flex-wrap: wrap;
            }
            .download-btn {
                background: var(--accent-gradient);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 2px 8px rgba(79,70,229,0.2);
                transition: all 0.2s;
                text-decoration: none;
                white-space: nowrap;
            }
            .download-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(79,70,229,0.3);
            }
            </style>
            """, unsafe_allow_html=True)
            
            png_col1, png_col2 = st.columns(2)
            
            # Bouton Original
            with png_col1:
                ok, buf = cv2.imencode(".png", cv2.cvtColor(row["Original"], cv2.COLOR_RGB2BGR))
                if ok:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: center;">
                        <a href="data:image/png;base64,{base64.b64encode(buf).decode()}" download="{safe_name}_original.png" style="text-decoration: none;">
                            <button class="download-btn">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg> Original
                            </button>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Bouton Grad-CAM
            with png_col2:
                ok, buf = cv2.imencode(".png", cv2.cvtColor(live_overlay, cv2.COLOR_RGB2BGR))
                if ok:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: center;">
                        <a href="data:image/png;base64,{base64.b64encode(buf).decode()}" download="{safe_name}_gradcam.png" style="text-decoration: none;">
                            <button class="download-btn">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg> Grad-CAM
                            </button>
                        </a>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)

            # Report
            with st.expander("📄 Rapport IA détaillé"):
                report = row["Report"]
                report_dict = report.to_dict() if hasattr(report, 'to_dict') else report
            
                st.markdown(f"""
                <div style="display:grid; gap:12px;">
                    <div style="background:var(--bg-input); padding:12px 16px; border-radius:var(--radius-sm); border-left: 4px solid #2563EB;">
                        <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg> Observations</div>
                        <div style="font-size:14px; color:var(--text-primary); margin-top:4px;">{report_dict.get('findings', 'N/A')}</div>
                    </div>
                    <div style="background:var(--bg-input); padding:12px 16px; border-radius:var(--radius-sm); border-left: 4px solid #60A5FA;">
                        <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5Z"/></svg> Impression</div>
                        <div style="font-size:14px; color:var(--text-primary); margin-top:4px;">{report_dict.get('impression', 'N/A')}</div>
                    </div>
                    <div style="background:var(--bg-input); padding:12px 16px; border-radius:var(--radius-sm); border-left: 4px solid #10B981;">
                        <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M21.8 10A10 10 0 1 1 17 3.34"/><path d="m9 11 3 3L22 4"/></svg> Recommandation</div>
                        <div style="font-size:14px; color:var(--text-primary); margin-top:4px;">{report_dict.get('recommendation', 'N/A')}</div>
                    </div>
                </div>
                <br>
                """, unsafe_allow_html=True)

                pdf_bytes = generate_pdf_report(report, row["Original"], live_overlay, live_overlay)
                st.download_button(
                    label="📄 Télécharger le rapport (PDF)",
                    data=pdf_bytes,
                    file_name=f"rapport_{safe_name}.pdf",
                    mime="application/pdf",
                    help="Rapport complet pour cette image : original, Grad-CAM et observations",
                    key=f"pdf_dl_{idx}",
                    width='stretch',
                )

            st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

        # Export Section - Boutons améliorés avec couleurs distinctes
        st.markdown("""<div style='margin: 40px 0 16px 0;'><h2><span style="margin-right: 10px; display:inline-flex; vertical-align:middle;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><rect x="2" y="4" width="20" height="5" rx="1"/><path d="M4 9v9a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9"/><path d="M10 13h4"/></svg></span> Exporter les résultats</h2></div>""", unsafe_allow_html=True)
    
        col_dl1, col_dl2, col_dl3 = st.columns(3)
    
        with col_dl1:
            st.download_button(
                label="📦 Visualisations (ZIP)",
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
                label="📊 Rapport CSV",
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
                label="📄 Rapports JSON",
                data=all_reports_json.encode("utf-8"),
                file_name=f"radiology_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Rapports structurés pour chaque image",
                width='stretch',
            )

        # Message de succès avec icône professionnelle (remplace ✅)
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid var(--success-text); margin-top: 24px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success-text);"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                <div>
                    <strong style="color: var(--success-text);">Analyse terminée</strong>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: var(--text-secondary);">
                        {len(results)} radiographie(s) traitée(s) avec succès
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ==============================================================
        # HOME STATE
        # ==============================================================
        col_welcome1, col_welcome2 = st.columns([2, 1])

        with col_welcome1:
            st.markdown("""
            <div class="card" style="height:100%;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09Z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2Z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>
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
                        <span style="font-size:18px; font-weight:700; color:#2563EB; width: 28px; text-align: center;">1</span>
                        <span style="color:var(--text-primary);"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M16 16l-4-4-4 4"/><path d="M12 12v9"/><path d="M20.4 17.6A5 5 0 0 0 18 8h-1.3A7 7 0 1 0 4 15.3"/></svg> Déposer une ou plusieurs radiographies (PNG/JPG)</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px; padding:8px 12px; background:var(--bg-input); border-radius:var(--radius-sm);">
                        <span style="font-size:18px; font-weight:700; color:#2563EB; width: 28px; text-align: center;">2</span>
                        <span style="color:var(--text-primary);"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><rect x="4" y="2" width="16" height="20" rx="2"/><line x1="8" y1="6" x2="16" y2="6"/><line x1="8" y1="10" x2="8" y2="10"/><line x1="12" y1="10" x2="12" y2="10"/><line x1="16" y1="10" x2="16" y2="10"/><line x1="8" y1="14" x2="8" y2="14"/><line x1="12" y1="14" x2="12" y2="14"/><line x1="16" y1="14" x2="16" y2="18"/><line x1="8" y1="18" x2="8" y2="18"/><line x1="12" y1="18" x2="12" y2="18"/></svg> Le modèle calcule les probabilités par classe</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px; padding:8px 12px; background:var(--bg-input); border-radius:var(--radius-sm);">
                        <span style="font-size:18px; font-weight:700; color:#2563EB; width: 28px; text-align: center;">3</span>
                        <span style="color:var(--text-primary);"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5Z"/></svg> Une carte Grad-CAM localise les zones d'intérêt</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px; padding:8px 12px; background:var(--bg-input); border-radius:var(--radius-sm);">
                        <span style="font-size:18px; font-weight:700; color:#2563EB; width: 28px; text-align: center;">4</span>
                        <span style="color:var(--text-primary);"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg> Export du rapport (CSV) et des visualisations (ZIP)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_welcome2:
            acc_line = f"{OVERALL_ACCURACY:.1f}%" if OVERALL_ACCURACY else "N/A"
            st.markdown(f"""
            <div class="card" style="height:100%; text-align:center;">
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; padding:16px 0;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></svg>
                    <p style="margin:8px 0 4px 0; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted);">
                        Exactitude mesurée
                    </p>
                    <p style="margin:0; font-size:38px; font-weight:700; background:var(--accent-gradient); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                        {acc_line}
                    </p>
                    <p style="font-size:12px; color:var(--text-muted); margin:4px 0 0 0;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg> {DATASET_SIZE:,} radiographies utilisées
                    </p>
                </div>
            </div>
            """.replace(",", "\u2009"), unsafe_allow_html=True)

        st.markdown("""<div style='margin: 32px 0 16px 0;'><h3><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="m12 2 3.1 6.3 6.9 1-5 4.9 1.2 6.9-6.2-3.3-6.2 3.3 1.2-6.9-5-4.9 6.9-1Z"/></svg> Fonctionnalités</h3></div>""", unsafe_allow_html=True)
    
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        features = [
            ("""<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/></svg> Classification 4 classes""", "COVID-19, opacité pulmonaire, pneumonie virale, ou normal, avec un score de confiance par classe."),
            ("""<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg> Explicabilité Grad-CAM""", "Cartes de chaleur superposées à la radiographie originale, pour visualiser les zones jugées pertinentes par le modèle."),
            ("""<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; display:inline-block;"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg> Rapports exportables""", "Export CSV des prédictions détaillées et ZIP des visualisations, pour analyse ou archivage."),
        ]
        for col, (title, desc) in zip([feat_col1, feat_col2, feat_col3], features):
            with col:
                st.markdown(f"""
                <div class="card" style="height:100%;">
                    <h4 style="font-size:15px; margin-bottom:8px;">{title}</h4>
                    <p style="font-size:13px; margin:0; color:var(--text-secondary);">{desc}</p>
                </div>
                """, unsafe_allow_html=True)