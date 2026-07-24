"""
views/_stub.py

Rendu partage pour les pages pas encore implementees. Affiche clairement
"a venir" plutot que des donnees inventees ou une page vide qui ressemble
a un bug — voir la feuille de route (README, section Roadmap) pour le
phasage.

Refactorise pour utiliser app/components.py et app/icons.py.
"""

from components import section_title, status_badge, glass_card
import streamlit as st


def render_stub(icon_name: str, title: str, description: str, phase_note: str) -> None:
    section_title(icon_name, title)
    glass_card(
        f"""
        <div style="text-align:center; padding: 24px 8px;">
            <p style="color:var(--text-secondary); font-size:15px; max-width:520px; margin:0 auto 16px auto;">
                {description}
            </p>
        </div>
        """,
    )
    status_badge("Pas encore implémenté", variant="warning", icon_name="alert-triangle")
    st.markdown(
        f'<p style="color:var(--text-muted); font-size:12.5px; margin-top:12px;">{phase_note}</p>',
        unsafe_allow_html=True,
    )
