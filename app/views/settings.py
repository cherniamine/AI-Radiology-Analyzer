"""
views/settings.py — Parametres (edition pas encore implementee, Phase 6).

En attendant l'edition, cette page affiche la configuration REELLE
actuellement chargee depuis .env / les valeurs par defaut — pas de valeurs
inventees, ce sont les memes que celles utilisees par l'application.

Refactorise pour utiliser app/components.py et app/icons.py (plus d'emoji,
plus de HTML de carte duplique).
"""

import streamlit as st

from config import config
from translator import t
from components import section_title, status_badge, glass_card
from icons import icon as render_icon


def _settings_group_html(title_icon: str, title: str, rows: list) -> str:
    rows_html = "".join(
        f"""<div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border-color); font-size:13.5px;">
            <span style="color:var(--text-secondary);">{label}</span>
            <span style="color:var(--text-primary); font-weight:600;">{value}</span>
        </div>"""
        for label, value in rows
    )
    return f"""
    <h4 style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
        {render_icon(title_icon, size=16, color="var(--accent-primary)")} {title}
    </h4>
    <div>{rows_html}</div>
    """


def render() -> None:
    section_title("settings", t("settings.title"), t("settings.subtitle"))
    status_badge(t("settings.edit_badge"), variant="warning", icon_name="alert-triangle")
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    yes, no = t("common.yes"), t("common.no")

    general_rows = [
        (t("settings.rows.default_language"), config.default_language),
        (t("settings.rows.default_theme"), config.default_theme),
    ]
    analysis_rows = [
        (t("settings.rows.gradcam_alpha"), f"{config.default_gradcam_alpha:.2f}"),
        (t("settings.rows.max_upload"), f"{config.max_upload_size_mb} Mo"),
    ]
    features_rows = [
        (t("settings.rows.assistant_enabled"), yes if config.enable_assistant else no),
        (t("settings.rows.history_enabled"), yes if config.enable_history else no),
        (t("settings.rows.pdf_export_enabled"), yes if config.enable_pdf_export else no),
        (t("settings.rows.json_export_enabled"), yes if config.enable_json_export else no),
    ]
    assistant_rows = [
        (t("settings.rows.llm_provider"), config.llm_provider),
        (t("settings.rows.ollama_model"), config.ollama_model),
    ]

    col1, col2 = st.columns(2)
    with col1:
        glass_card(_settings_group_html("languages", "Général", general_rows))
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        glass_card(_settings_group_html("gauge", "Analyse", analysis_rows))
    with col2:
        glass_card(_settings_group_html("layers", "Fonctionnalités", features_rows))
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        glass_card(_settings_group_html("bot", "Assistant IA", assistant_rows))

    st.caption(t("settings.note"))
