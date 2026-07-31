"""
views/settings.py — Parametres, modifiables pour la session en cours.

Chaque widget est lie a settings_store.get_setting()/set_setting() : la
modification prend effet immediatement (les autres pages relisent
get_setting() a chaque rerun), mais n'est jamais ecrite dans .env ni sur
le disque — un redemarrage de l'application repart des valeurs de .env.

La langue et le theme ne sont PAS geres ici : ils ont deja leur propre
controle dedie dans la barre laterale (predict.py), avec sa propre
persistance (session + URL). Les dupliquer ici creerait un second
controle concurrent et pretant a confusion.

IMPORTANT — piege Streamlit corrige ici : un widget avec une `key=`
explicite conserve SA PROPRE valeur dans st.session_state[key], qui prend
le pas sur le parametre `value=` a chaque rerun suivant. Reinitialiser
uniquement l'override cote settings_store (session_state["_setting_override_xxx"])
ne suffit donc pas : le widget re-affirme aussitot son ancienne valeur et
re-ecrase l'override. Chaque widget ci-dessous recoit une cle explicite
stable (_widget_key), et le bouton de reinitialisation efface AUSSI ces
cles-la avant d'appeler reset_all().
"""

import streamlit as st

from config import config
from translator import t
from components import section_title, status_badge, glass_card
from icons import icon as render_icon
from settings_store import get_setting, set_setting, is_overridden, reset_all, EDITABLE_KEYS


def _widget_key(key: str) -> str:
    return f"widget_{key}"


def _overridden_marker(key: str) -> str:
    if is_overridden(key):
        return f" *({t('settings.overridden_badge')})*"
    return ""


def render() -> None:
    section_title("settings", t("settings.title"), t("settings.subtitle"))
    status_badge(t("settings.edit_badge"), variant="info", icon_name="check-circle")
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        glass_card(f"""
        <h4 style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            {render_icon('languages', size=16, color='var(--accent-primary)')} {t('settings.sections.general')}
        </h4>
        <p style="font-size:12px; color:var(--text-muted); margin-bottom:10px;">{t('settings.language_theme_note')}</p>
        <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--border-color); font-size:13.5px;">
            <span style="color:var(--text-secondary);">{t('settings.rows.default_language')}</span>
            <span style="color:var(--text-primary); font-weight:600;">{config.default_language}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:6px 0; font-size:13.5px;">
            <span style="color:var(--text-secondary);">{t('settings.rows.default_theme')}</span>
            <span style="color:var(--text-primary); font-weight:600;">{config.default_theme}</span>
        </div>
        """)

        st.markdown(f"""
        <div class="card" style="margin-top:14px;">
            <h4 style="display:flex; align-items:center; gap:8px;">
                {render_icon('gauge', size=16, color='var(--accent-primary)')} {t('settings.sections.analysis')}
            </h4>
        </div>
        """, unsafe_allow_html=True)

        new_alpha = st.slider(
            f"{t('settings.rows.gradcam_alpha')}{_overridden_marker('default_gradcam_alpha')}",
            0.1, 0.9, min(0.9, max(0.1, get_setting("default_gradcam_alpha"))), 0.05,
            help=t("settings.help.gradcam_alpha"),
            key=_widget_key("default_gradcam_alpha"),
        )
        if abs(new_alpha - get_setting("default_gradcam_alpha")) > 1e-9:
            set_setting("default_gradcam_alpha", new_alpha)
            st.rerun()

        new_upload = st.number_input(
            f"{t('settings.rows.max_upload')}{_overridden_marker('max_upload_size_mb')}",
            min_value=1, max_value=500, value=int(get_setting("max_upload_size_mb")), step=1,
            help=t("settings.help.max_upload"),
            key=_widget_key("max_upload_size_mb"),
        )
        if new_upload != get_setting("max_upload_size_mb"):
            set_setting("max_upload_size_mb", new_upload)
            st.rerun()

    with col2:
        st.markdown(f"""
        <div class="card">
            <h4 style="display:flex; align-items:center; gap:8px;">
                {render_icon('layers', size=16, color='var(--accent-primary)')} {t('settings.sections.features')}
            </h4>
        </div>
        """, unsafe_allow_html=True)

        for key, label_key in [
            ("enable_assistant", "assistant_enabled"),
            ("enable_history", "history_enabled"),
            ("enable_pdf_export", "pdf_export_enabled"),
            ("enable_json_export", "json_export_enabled"),
        ]:
            new_value = st.toggle(
                f"{t(f'settings.rows.{label_key}')}{_overridden_marker(key)}",
                value=bool(get_setting(key)),
                key=_widget_key(key),
            )
            if new_value != get_setting(key):
                set_setting(key, new_value)
                st.rerun()

        st.markdown(f"""
        <div class="card" style="margin-top:14px;">
            <h4 style="display:flex; align-items:center; gap:8px;">
                {render_icon('bot', size=16, color='var(--accent-primary)')} {t('settings.sections.assistant')}
            </h4>
        </div>
        """, unsafe_allow_html=True)

        new_provider = st.selectbox(
            f"{t('settings.rows.llm_provider')}{_overridden_marker('llm_provider')}",
            options=["ollama"],
            index=0,
            help=t("settings.help.llm_provider"),
            key=_widget_key("llm_provider"),
        )
        if new_provider != get_setting("llm_provider"):
            set_setting("llm_provider", new_provider)
            st.rerun()

        new_base_url = st.text_input(
            f"OLLAMA_BASE_URL{_overridden_marker('ollama_base_url')}",
            value=get_setting("ollama_base_url"),
            help=t("settings.help.ollama_base_url"),
            key=_widget_key("ollama_base_url"),
        )
        if new_base_url != get_setting("ollama_base_url"):
            set_setting("ollama_base_url", new_base_url)
            st.session_state.pop(f"ollama_reachable::{get_setting('ollama_base_url')}::{get_setting('ollama_model')}", None)
            st.rerun()

        new_model = st.text_input(
            f"{t('settings.rows.ollama_model')}{_overridden_marker('ollama_model')}",
            value=get_setting("ollama_model"),
            help=t("settings.help.ollama_model"),
            key=_widget_key("ollama_model"),
        )
        if new_model != get_setting("ollama_model"):
            set_setting("ollama_model", new_model)
            st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if any(is_overridden(k) for k in EDITABLE_KEYS):
        if st.button(t("settings.reset_button"), help=t("settings.help.reset_button")):
            # Efface d'abord l'etat propre de chaque widget (sinon il re-ecrase
            # aussitot l'override qu'on vient de reinitialiser — voir docstring).
            for key in EDITABLE_KEYS:
                st.session_state.pop(_widget_key(key), None)
            reset_all()
            st.success(t("settings.reset_success"))
            st.rerun()

    st.caption(t("settings.note"))
