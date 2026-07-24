"""
views/reports.py — Rapports.

Complementaire a l'Historique (qui gere recherche/filtres/suppression et le
telechargement) : cette page se concentre sur l'APERCU d'un rapport deja
enregistre, sans telechargement necessaire.

- Apercu JSON interactif (st.json, arborescence repliable) + bloc copiable
  (st.code affiche une icone de copie native, aucun JavaScript necessaire).
- Apercu PDF inline, integre en base64 dans un <iframe> (technique standard
  cote navigateur — necessite un lecteur PDF natif dans le navigateur ;
  non garanti sur tous les navigateurs mobiles, voir avertissement affiche).
- Impression directe : bouton qui declenche window.print() du navigateur
  (aucune bibliotheque externe), imprime la page actuelle.

Reutilise persistence.HistoryStore et report_generator.build_report /
generate_pdf_report — aucune logique dupliquee avec history.py ou
analysis.py.
"""

import base64
import json

import streamlit as st

from config import CLASS_NAMES, class_color, config
from persistence import get_store
from report_generator import build_report, generate_pdf_report
from components import section_title, empty_state
from icons import icon as render_icon


def _print_button_html() -> str:
    return """
    <button onclick="window.print()" style="
        background: var(--accent-gradient); color: #FFFFFF; border: none;
        border-radius: var(--radius-sm); padding: 8px 16px; font-size: 13px;
        font-weight: 600; cursor: pointer; transition: var(--transition);
    ">🖨️ Imprimer cette page</button>
    """


def render() -> None:
    section_title("file-text", "Rapports", "Aperçu inline d'un rapport déjà enregistré — copie et impression")

    store = get_store()

    if store.count() == 0:
        empty_state(
            "Aucun rapport disponible",
            "Les analyses effectuées dans <b>🔬 Nouvelle analyse</b> apparaîtront ici pour aperçu.",
        )
        return

    records = store.list(limit=200)

    def _option_label(r):
        color, soft, label, icon = class_color(r.predicted_class)
        return f"{r.image_name} — {label} ({r.confidence:.1f}%) — {r.created_at[:16]}"

    options = {_option_label(r): r for r in records}
    choice = st.selectbox("Choisir une analyse à prévisualiser", options=list(options.keys()))
    record = options[choice]

    color, soft, label, icon_name = class_color(record.predicted_class)

    st.markdown(f"""
    <div class="card" style="margin-top:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="mono" style="font-weight:600;">{record.image_name}</span>
            <span class="badge badge-info" style="color:{color}; background:{soft};">{render_icon(icon_name, size=13, color=color)} {label}</span>
        </div>
        <p style="font-size:12px; color:var(--text-muted); margin:6px 0 0 0;">
            Analysé le {record.created_at[:19].replace('T', ' ')} · Confiance {record.confidence:.1f}% · {record.inference_ms:.0f} ms
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_json, tab_pdf = st.tabs(["📄 Aperçu JSON", "🖨️ Aperçu PDF"])

    with tab_json:
        payload = record.to_dict()

        st.markdown("<p style='font-size:13px; color:var(--text-muted); margin-bottom:4px;'>Arborescence interactive :</p>", unsafe_allow_html=True)
        st.json(payload)

        st.markdown("<p style='font-size:13px; color:var(--text-muted); margin:12px 0 4px 0;'>Texte copiable (icône en haut à droite du bloc) :</p>", unsafe_allow_html=True)
        json_str = json.dumps(payload, indent=2, ensure_ascii=False)
        st.code(json_str, language="json")

    with tab_pdf:
        if not (record.has_images() and config.enable_pdf_export):
            st.info(
                "Aperçu PDF indisponible : images non enregistrées pour cette analyse "
                "(historique désactivé au moment de l'analyse) ou export PDF désactivé."
            )
        else:
            report = build_report(
                image_name=record.image_name,
                predicted_class=record.predicted_class,
                confidence=record.confidence,
                class_probabilities=record.class_probabilities,
            )
            pdf_bytes = generate_pdf_report(report, record.original_image(), record.overlay_image())
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

            st.markdown(_print_button_html(), unsafe_allow_html=True)
            st.caption(
                "Ouvre la boîte de dialogue d'impression de votre navigateur pour la page actuelle. "
                "Pour n'imprimer que le rapport, utilisez plutôt le bouton de téléchargement PDF ci-dessous "
                "puis imprimez le fichier."
            )

            st.markdown(
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                f'width="100%" height="600" style="border:1px solid var(--border-color); border-radius:var(--radius-sm);" '
                f'type="application/pdf"></iframe>',
                unsafe_allow_html=True,
            )
            st.caption(
                "⚠️ L'aperçu inline nécessite un lecteur PDF intégré au navigateur "
                "(fonctionne sur la plupart des navigateurs de bureau ; peut être indisponible sur certains navigateurs mobiles)."
            )

            st.download_button(
                "📄 Télécharger ce PDF",
                data=pdf_bytes,
                file_name=f"rapport_{record.id}.pdf",
                mime="application/pdf",
                help="Télécharger le fichier pour l'imprimer ou l'archiver séparément",
                width='stretch',
            )