"""Page de paramètres de session. Les valeurs modifiées sont appliquées
immédiatement mais ne sont pas écrites sur le disque ni dans .env."""

import streamlit as st

from translator import t, get_language, set_language, SUPPORTED_LANGUAGES
from components import section_title, status_badge
from icons import icon as render_icon
from settings_store import get_setting, set_setting, is_overridden, reset_all, EDITABLE_KEYS


def _widget_key(key: str) -> str:
    return f"widget_{key}"


def _overridden_marker(key: str) -> str:
    if is_overridden(key):
        return f" *({t('settings.overridden_badge')})*"
    return ""


def _card_heading(icon_name: str, label: str) -> None:
    """En-tete de bloc homogene (icone + titre), reutilise par les 4 sections
    de cette page — a appeler a l'interieur d'un `with st.container(border=True):`."""
    st.markdown(
        f"<h4 style='display:flex; align-items:center; gap:8px; margin-bottom:4px;'>"
        f"{render_icon(icon_name, size=16, color='var(--accent-primary)')} {label}</h4>",
        unsafe_allow_html=True,
    )


def _kv_row(label: str, value: str, border: bool = True) -> None:
    """Ligne label/valeur statique (bloc General) — factorisee plutot que dupliquee."""
    border_style = "border-bottom:1px solid var(--border-color);" if border else ""
    st.markdown(
        f"<div style='display:flex; justify-content:space-between; padding:6px 0; "
        f"{border_style} font-size:13.5px;'>"
        f"<span style='color:var(--text-secondary);'>{label}</span>"
        f"<span style='color:var(--text-primary); font-weight:600;'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render() -> None:
    section_title("settings", t("settings.title"), t("settings.subtitle"))
    status_badge(t("settings.edit_badge"), variant="info", icon_name="check-circle")
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    # === COLONNE 1 ===
    with col1:
        # ========= Box 1 : General ==========
        with st.container(border=True):
            _card_heading("languages", t("settings.sections.general"))
            st.markdown(
                f"<p style='font-size:12px; color:var(--text-muted); margin-bottom:10px;'>{t('settings.language_theme_note')}</p>",
                unsafe_allow_html=True,
            )

            lang_codes = list(SUPPORTED_LANGUAGES.keys())
            current_lang = get_language()
            selector_key = f"_lang_selector_settings_{current_lang}"
            selected_lang = st.selectbox(
                t("common.language_selector_label"),
                options=lang_codes,
                index=lang_codes.index(current_lang),
                format_func=lambda code: f"{SUPPORTED_LANGUAGES[code]['flag']} {SUPPORTED_LANGUAGES[code]['label']}",
                help=t("settings.help.language_selector"),
                key=selector_key,
            )
            if selected_lang != current_lang:
                set_language(selected_lang)
                st.rerun()

            # Toggle theme clair/sombre — meme etat de session que le bouton
            # de la sidebar (st.session_state._dark_mode), toujours synchronise.
            theme_widget_key = f"widget_dark_mode_{st.session_state._dark_mode}"
            dark_mode_new = st.toggle(
                f"{t('settings.rows.default_theme')}{' 🌙' if st.session_state._dark_mode else ' ☀️'}",
                value=st.session_state._dark_mode,
                help=t("settings.help.theme_toggle"),
                key=theme_widget_key,
            )
            if dark_mode_new != st.session_state._dark_mode:
                st.session_state._dark_mode = dark_mode_new
                st.rerun()

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # ========= Box 2 : Analysis ==========
        with st.container(border=True):
            _card_heading("gauge", t("settings.sections.analysis"))

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

    # === COLONNE 2 ===
    with col2:
        # ========= Box 3 : Features ==========
        with st.container(border=True):
            _card_heading("layers", t("settings.sections.features"))

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

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # ========= Box 4 : AI Assistant ==========
        with st.container(border=True):
            _card_heading("bot", t("settings.sections.assistant"))

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
                st.session_state.pop(
                    f"ollama_reachable::{get_setting('ollama_base_url')}::{get_setting('ollama_model')}",
                    None
                )
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

    # === Reset button ===
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
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