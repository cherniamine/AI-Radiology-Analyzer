"""
views/assistant.py — Assistant IA.

Interface de chat branchee sur app/assistant/ (RAG local + Ollama). Cette
page reste en francais pour ses libelles fixes (l'i18n de l'assistant lui-
meme couvre les reponses generees et les questions suggerees, dans les 3
langues, via app/assistant/prompts.py) — voir README pour le perimetre i18n
exact.

IMPORTANT : la generation via Ollama n'a pas pu etre testee de bout en
bout avec un vrai serveur Ollama dans l'environnement de developpement
(aucun Ollama installe/accessible ici). Le code degrade proprement vers un
message d'erreur clair si Ollama n'est pas joignable — verifie vous-meme
avec un Ollama reellement lance.
"""

import os
import sys
from typing import Optional
import streamlit as st

_ASSISTANT_PARENT = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.join(_ASSISTANT_PARENT, "..")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from translator import get_language, t
from settings_store import get_setting
from assistant import ask
from assistant.prompts import SUGGESTED_QUESTIONS
from components import section_title, user_message, assistant_message
from icons import icon as render_icon


def render() -> None:
    section_title(
        "bot", t("assistant.title"), t("assistant.subtitle"),
    )

    if not get_setting("enable_assistant"):
        st.info(t("assistant.disabled"))
        return

    st.markdown(f"""
    <div class="disclaimer" style="margin-bottom:16px;">
        {render_icon("alert-triangle", size=15)}
        {t("assistant.disclaimer")}
    </div>
    """, unsafe_allow_html=True)

    language = get_language()

    # Persistance : rattachee au meme interrupteur que l'historique des
    # analyses (ENABLE_HISTORY) — coherent avec le reste de l'app, qui n'a
    # qu'une seule notion de "conserver des donnees en local ou non". Si
    # desactive, on retombe sur l'ancien comportement (session_state
    # seulement, perdu au rechargement de page).
    persist_chat = get_setting("enable_history")
    if persist_chat:
        from persistence import get_store
        store = get_store()

    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = store.get_assistant_conversation() if persist_chat else []

    def _append(role: str, content: str, sources: Optional[list] = None) -> None:
        """Ajoute un message à l'historique avec ses sources."""
        message = {"role": role, "content": content}
        if sources:
            message["sources"] = sources
        st.session_state.assistant_history.append(message)
        if persist_chat:
            store.add_assistant_message(role, content, sources=sources)

    # Etat du serveur Ollama, verifie une fois par (url, modele) — pas juste une
    # fois par session — pour qu'un changement dans Parametres declenche une
    # nouvelle verification plutot que de garder le resultat de l'ancienne
    # configuration. Meme convention de cle que celle que Parametres invalide
    # deja lui-meme au moment de la sauvegarde (voir views/settings.py).
    ollama_base_url = get_setting("ollama_base_url")
    ollama_model = get_setting("ollama_model")
    reachable_key = f"ollama_reachable::{ollama_base_url}::{ollama_model}"
    if reachable_key not in st.session_state:
        try:
            from assistant.providers.ollama_provider import OllamaProvider
            probe = OllamaProvider(base_url=ollama_base_url, model=ollama_model)
            st.session_state[reachable_key] = probe.is_reachable()
        except Exception:
            st.session_state[reachable_key] = False

    if not st.session_state[reachable_key]:
        st.warning(t("assistant.unavailable", base_url=ollama_base_url, model=ollama_model))

    # Questions suggerees (uniquement avant le premier message)
    if not st.session_state.assistant_history:
        st.markdown(f"<p style='font-size:13px; color:var(--text-muted);'>{t('assistant.suggestions')}</p>", unsafe_allow_html=True)
        suggestions = SUGGESTED_QUESTIONS.get(language, SUGGESTED_QUESTIONS["fr"])
        cols = st.columns(len(suggestions))
        for col, question in zip(cols, suggestions):
            with col:
                if st.button(question, key=f"suggest_{question}", width='stretch'):
                    _append("user", question)
                    st.rerun()

    # ✅ Affichage de l'historique avec sources persistantes
    for message in st.session_state.assistant_history:
        if message["role"] == "user":
            user_message(message["content"])
        else:
            # On récupère les sources si elles existent
            sources = message.get("sources")
            assistant_message(message["content"], sources=sources)

    # Si le dernier message est de l'utilisateur sans reponse, la generer maintenant
    if st.session_state.assistant_history and st.session_state.assistant_history[-1]["role"] == "user":
        with st.spinner(t("assistant.thinking")):
            result = ask(
                st.session_state.assistant_history[-1]["content"], language=language,
                llm_provider=get_setting("llm_provider"),
                ollama_base_url=ollama_base_url, ollama_model=ollama_model,
            )
        
        # ✅ Stockage et affichage avec sources
        sources = result.sources if (result.sources and not result.refused) else None
        _append("assistant", result.text, sources=sources)
        assistant_message(result.text, sources=sources)

    user_input = st.chat_input(t("assistant.input_placeholder"))
    if user_input:
        _append("user", user_input)
        st.rerun()

    if st.session_state.assistant_history:
        if st.button(
                    t("assistant.clear"),
                    icon=":material/delete:",
                    help=t("assistant.clear_help")
                ):
            st.session_state.assistant_history = []
            if persist_chat:
                store.clear_assistant_conversation()
            st.rerun()
