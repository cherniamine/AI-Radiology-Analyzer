"""
views/about.py — Page "A propos" : vue d'ensemble reelle du projet
(pas un stub — contenu statique, ne depend d'aucune autre phase).

Refactorise pour utiliser app/components.py et app/icons.py (plus d'emoji,
plus de HTML de carte duplique — remplace par section_title/glass_card/
metric_card/footer).
"""

import streamlit as st

from config import config, load_real_metrics, DATASET_SIZE
from translator import t
from components import section_title, glass_card, metric_card, footer
from icons import icon as render_icon


def render() -> None:
    section_title("hospital", t("common.app_title"), t("about.version_line", version=config.app_version))

    st.markdown(f"""
    <div class="disclaimer" style="margin-top:0;">
        {render_icon("alert-triangle", size=15)} {t('about.disclaimer')}
    </div>
    """, unsafe_allow_html=True)

    metrics = load_real_metrics()
    accuracy = f"{metrics['accuracy'] * 100:.1f}%" if metrics else "N/A"
    dataset_display = f"{DATASET_SIZE:,}".replace(",", "\u2009")

    glass_card(f"<h3>{render_icon('info', size=18)} {t('about.overview_title')}</h3>"
               f"<p style='font-size:14px; line-height:1.7;'>{t('about.overview_text')}</p>")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    stat_cols = st.columns(3)
    with stat_cols[0]:
        metric_card(accuracy, t("about.accuracy_label"))
    with stat_cols[1]:
        metric_card(dataset_display, t("about.training_images_label"))
    with stat_cols[2]:
        metric_card("4", t("about.classes_label"))
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    glass_card(f"""
    <h3>{render_icon('workflow', size=18)} {t('about.architecture_title')}</h3>
    <p style="font-size:13.5px; line-height:1.8;">
    <b>app/predict.py</b> — point d'entrée, navigation multi-pages<br>
    <b>app/config.py</b> — configuration centralisée (.env)<br>
    <b>app/theme.py</b> — design system partagé<br>
    <b>app/components.py</b> — bibliothèque de composants UI réutilisables<br>
    <b>app/icons.py</b> — jeu d'icônes SVG autonome (style Lucide/Feather)<br>
    <b>app/translator.py</b> — internationalisation (fr/en/ar)<br>
    <b>app/persistence.py</b> — historique des analyses (SQLite)<br>
    <b>app/assistant/</b> — assistant IA (RAG local + Ollama)<br>
    <b>app/views/</b> — une page par fonctionnalité (Dashboard, Analyse, Rapports, Historique, Assistant, Paramètres, À propos)<br>
    <b>app/image_utils.py</b> — validation d'image, prétraitement, Grad-CAM (post-traitement)<br>
    <b>app/report_generator.py</b> — génération du rapport structuré + export PDF/JSON<br>
    <b>src/</b> — entraînement et évaluation du modèle (hors application)
    </p>
    """)

    tech_badges = "".join(
        f'<span class="badge badge-info">{name}</span>'
        for name in ["TensorFlow / Keras", "Streamlit", "OpenCV", "Plotly", "ReportLab",
                     "SQLite", "Ollama", "Docker", "GitHub Actions"]
    )
    glass_card(f"""
    <h3>{render_icon('layers', size=18)} {t('about.technologies_title')}</h3>
    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">{tech_badges}</div>
    """)

    limitations_html = "".join(f"<li>{item}</li>" for item in t("about.limitations"))
    glass_card(f"""
    <h3>{render_icon('alert-triangle', size=18)} {t('about.limitations_title')}</h3>
    <ul style="font-size:13.5px; line-height:1.8; padding-left:20px;">{limitations_html}</ul>
    """)

    glass_card(f"""
    <h3>{render_icon('sparkles', size=18)} {t('about.roadmap_title')}</h3>
    <p style="font-size:13.5px; line-height:1.8;">
    {t('about.roadmap_done')}<br>
    {t('about.roadmap_pending')}
    </p>
    """)

    footer(
        config.app_title, config.app_version, "MIT",
        ["TensorFlow", "Streamlit", "Plotly", "ReportLab", "Ollama"],
    )
