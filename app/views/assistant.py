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

import streamlit as st

_ASSISTANT_PARENT = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.join(_ASSISTANT_PARENT, "..")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from config import config
from translator import get_language
from assistant import ask
from assistant.prompts import SUGGESTED_QUESTIONS
from components import section_title, user_message, assistant_message
from icons import icon as render_icon


def render() -> None:
    section_title(
        "bot", "Assistant IA",
        "Posez des questions sur le projet, le modèle, Grad-CAM, la confiance, Docker...",
    )

    if not config.enable_assistant:
        st.info("L'assistant IA est désactivé (`ENABLE_ASSISTANT=false` dans `.env`).")
        return

    st.markdown(f"""
    <div class="disclaimer" style="margin-bottom:16px;">
        {render_icon("alert-triangle", size=15)}
        Cet assistant explique le projet et sa documentation. Il ne fournit
        <b>jamais</b> de diagnostic, de traitement, de prescription ou d'avis médical
        personnel — pour toute question de santé, consultez un professionnel.
    </div>
    """, unsafe_allow_html=True)

    language = get_language()

    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = []

    # Etat du serveur Ollama, verifie une fois par session (pas a chaque rerun)
    if "ollama_reachable" not in st.session_state:
        try:
            from assistant.providers.ollama_provider import OllamaProvider
            probe = OllamaProvider(base_url=config.ollama_base_url, model=config.ollama_model)
            st.session_state.ollama_reachable = probe.is_reachable()
        except Exception:
            st.session_state.ollama_reachable = False

    if not st.session_state.ollama_reachable:
        st.warning(
            f"⚠️ Serveur Ollama non détecté sur `{config.ollama_base_url}`. "
            f"L'assistant répondra avec un message d'erreur tant qu'Ollama "
            f"n'est pas lancé (`ollama serve`) avec le modèle `{config.ollama_model}` "
            f"téléchargé (`ollama pull {config.ollama_model}`)."
        )

    # Questions suggerees (uniquement avant le premier message)
    if not st.session_state.assistant_history:
        st.markdown("<p style='font-size:13px; color:var(--text-muted);'>Suggestions :</p>", unsafe_allow_html=True)
        suggestions = SUGGESTED_QUESTIONS.get(language, SUGGESTED_QUESTIONS["fr"])
        cols = st.columns(len(suggestions))
        for col, question in zip(cols, suggestions):
            with col:
                if st.button(question, key=f"suggest_{question}", width='stretch'):
                    st.session_state.assistant_history.append({"role": "user", "content": question})
                    st.rerun()

    for message in st.session_state.assistant_history:
        if message["role"] == "user":
            user_message(message["content"])
        else:
            assistant_message(message["content"])

    # Si le dernier message est de l'utilisateur sans reponse, la generer maintenant
    if st.session_state.assistant_history and st.session_state.assistant_history[-1]["role"] == "user":
        with st.spinner("Réflexion..."):
            result = ask(st.session_state.assistant_history[-1]["content"], language=language)
        assistant_message(result.text, sources=result.sources if (result.sources and not result.refused) else None)
        st.session_state.assistant_history.append({"role": "assistant", "content": result.text})

    user_input = st.chat_input("Posez votre question...")
    if user_input:
        st.session_state.assistant_history.append({"role": "user", "content": user_input})
        st.rerun()

    if st.session_state.assistant_history:
        if st.button("🗑️ Effacer la conversation", help="Supprime l'historique de cette conversation (ne supprime rien dans l'Historique des analyses)"):
            st.session_state.assistant_history = []
            st.rerun()
