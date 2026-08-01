"""Entrée Streamlit de l'application. Configure le thème, la langue et la
navigation multi-pages."""

import streamlit as st

from config import config
from theme import inject_theme
from translator import t, get_language, set_language, is_rtl, SUPPORTED_LANGUAGES
from views import analysis, dashboard, reports, history, assistant, settings, about

st.set_page_config(
    page_title=t("common.app_title"),
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

current_lang = get_language()

if "_dark_mode" not in st.session_state:
    st.session_state._dark_mode = (config.default_theme == "dark")

inject_theme(rtl=is_rtl(), dark=st.session_state._dark_mode)

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-header">
        <div class="brand-mark">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/></svg>
        </div>
        <div class="brand-text">
            <h3>{t('common.app_title')}</h3>
            <p>{t('common.app_subtitle')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    top_cols = st.columns([5, 1])
    with top_cols[0]:
        lang_codes = list(SUPPORTED_LANGUAGES.keys())
        selector_key = f"_lang_selector_{current_lang}"
        selected = st.selectbox(
            t("common.language_selector_label"),
            options=lang_codes,
            index=lang_codes.index(current_lang),
            format_func=lambda code: f"{SUPPORTED_LANGUAGES[code]['flag']} {SUPPORTED_LANGUAGES[code]['label']}",
            key=selector_key,
            label_visibility="collapsed",
        )
        if selected != current_lang:
            set_language(selected)
            st.rerun()
    with top_cols[1]:
        theme_icon = "☀️" if st.session_state._dark_mode else "🌙"
        if st.button("", icon=theme_icon, key="_theme_toggle", help=t("common.theme_toggle_help")):
            st.session_state._dark_mode = not st.session_state._dark_mode
            st.rerun()

# Navigation avec icones Material Symbols (":material/nom:") — rendu
# coherent entre systemes d'exploitation, contrairement aux emojis
# (qui varient de style selon la plateforme). Le rendu de ces icones
# depend de span/label en CSS ; voir theme.py pour le correctif qui
# empeche ce rendu d'etre casse par les regles typographiques globales.
#
# IMPORTANT : url_path est explicite pour chaque page. Sans cela, Streamlit
# deduit le chemin d'URL du NOM DE LA FONCTION callable (ici, "render" pour
# les sept pages, puisque chaque module views/*.py expose une fonction
# render()) -> collision de chemin -> st.navigation() levait une
# StreamlitAPIException a CHAQUE chargement de l'application. Ce bug est
# passe inapercu lors des tests d'execution "a blanc" (bare mode, sans vrai
# contexte Streamlit) : st.navigation() n'y valide pas les chemins de la
# meme facon. Il n'a ete detecte qu'avec streamlit.testing.v1.AppTest, qui
# simule un vrai contexte de script. Voir tests/test_predict_apptest.py.
pages = [
    st.Page(analysis.render, title=t("common.nav.analysis"), icon=":material/biotech:", default=True, url_path="analyse"),
    st.Page(dashboard.render, title=t("common.nav.dashboard"), icon=":material/monitoring:", url_path="dashboard"),
    st.Page(reports.render, title=t("common.nav.reports"), icon=":material/description:", url_path="rapports"),
    st.Page(history.render, title=t("common.nav.history"), icon=":material/history:", url_path="historique"),
    st.Page(assistant.render, title=t("common.nav.assistant"), icon=":material/smart_toy:", url_path="assistant"),
    st.Page(settings.render, title=t("common.nav.settings"), icon=":material/tune:", url_path="parametres"),
    st.Page(about.render, title=t("common.nav.about"), icon=":material/info:", url_path="a-propos"),
]

nav = st.navigation(pages)
nav.run()