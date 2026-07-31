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
from translator import get_language, t
from persistence import get_store
from report_generator import build_report, generate_pdf_report
from components import section_title, empty_state
from icons import icon as render_icon


def _json_payload(record, language: str) -> dict:
    """Voir history._json_payload : meme raisonnement (regenerer le rapport dans
    la langue courante plutot que servir le texte fige en francais enregistre a
    l'analyse). Duplique volontairement plutot que factorise entre les deux pages
    — 8 lignes, pas assez pour justifier un import croise entre deux views/*.py
    qui restent par ailleurs independantes (voir docstring du module)."""
    report = build_report(
        image_name=record.image_name,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        class_probabilities=record.class_probabilities,
        language=language,
    )
    payload = report.to_dict()
    payload["id"] = record.id
    payload["created_at"] = record.created_at
    payload["inference_ms"] = round(record.inference_ms, 1)
    return payload


def _print_button_html() -> str:
    return f"""
    <button onclick="window.print()" style="
        background: var(--accent-gradient); color: var(--on-accent); border: none;
        border-radius: var(--radius-sm); padding: 10px 20px; font-size: 14px;
        font-weight: 600; cursor: pointer; transition: var(--transition);
        display: inline-flex; align-items: center; gap: 8px;
        box-shadow: 0 2px 10px rgba(194, 121, 12, 0.25);
    ">{render_icon('printer', size=16, color='var(--on-accent)')} {t('reports.print_button')}</button>
    """


def render() -> None:
    section_title("file-text", t("reports.title"), t("reports.subtitle"))

    store = get_store()
    language = get_language()

    if store.count() == 0:
        empty_state(
            t("reports.empty_title"), t("reports.empty_text"),
        )
        return

    records = store.list(limit=200)

    def _option_label(r):
        color, soft, label, icon = class_color(r.predicted_class)
        return f"{r.image_name} — {label} ({r.confidence:.1f}%) — {r.created_at[:16]}"

    options = {_option_label(r): r for r in records}
    choice = st.selectbox(t("reports.select_label"), options=list(options.keys()))
    record = options[choice]

    color, soft, label, icon_name = class_color(record.predicted_class)

    st.markdown(f"""
    <div class="card" style="margin-top:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="mono" style="font-weight:600;">{render_icon('image', size=14)} {record.image_name}</span>
            <span class="badge badge-info" style="color:{color}; background:{soft};">{render_icon(icon_name, size=13, color=color)} {label}</span>
        </div>
        <p style="font-size:12px; color:var(--text-muted); margin:6px 0 0 0;">
            {render_icon('clock', size=12)} {t('reports.analyzed_on', date=record.created_at[:19].replace('T', ' '))} · {render_icon('gauge', size=12)} {t('reports.confidence', value=f'{record.confidence:.1f}')} · {render_icon('cpu', size=12)} {record.inference_ms:.0f} ms
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Onglets sans icônes dans le label (car st.tabs() n'accepte pas le HTML)
    tab_json, tab_pdf = st.tabs([
        t("reports.tab_json"),
        t("reports.tab_pdf")
    ])

    with tab_json:
        payload = _json_payload(record, language)
        
        # Ajouter l'icône à l'intérieur de l'onglet
        st.markdown(f"{render_icon('file-text', size=14)} **{t('reports.tab_json')}**", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:13px; color:var(--text-muted); margin-bottom:4px;'>{render_icon('layers', size=12)} {t('reports.json_tree_label')}</p>", unsafe_allow_html=True)
        st.json(payload)
        
        st.markdown(f"<p style='font-size:13px; color:var(--text-muted); margin:12px 0 4px 0;'>{render_icon('copy', size=12)} {t('reports.json_copy_label')}</p>", unsafe_allow_html=True)
        json_str = json.dumps(payload, indent=2, ensure_ascii=False)
        st.code(json_str, language="json")

    with tab_pdf:
        # Ajouter l'icône à l'intérieur de l'onglet
        st.markdown(f"{render_icon('file-text', size=14)} **{t('reports.tab_pdf')}**", unsafe_allow_html=True)

        if not (record.has_images() and config.enable_pdf_export):
            st.info(
                f"{render_icon('alert-circle', size=16)} {t('reports.pdf_unavailable')}"
            )
        else:
            report = build_report(
                image_name=record.image_name,
                predicted_class=record.predicted_class,
                confidence=record.confidence,
                class_probabilities=record.class_probabilities,
                language=language,
            )
            pdf_bytes = generate_pdf_report(
                report, record.original_image(), record.overlay_image(),
                heatmap_img_rgb=record.heatmap_image(),
            )
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

            st.markdown(_print_button_html(), unsafe_allow_html=True)
            st.markdown(
                f"{render_icon('info', size=12)} *{t('reports.print_caption')}*",
                unsafe_allow_html=True
            )

            st.markdown(
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                f'width="100%" height="600" style="border:1px solid var(--border-color); border-radius:var(--radius-sm);" '
                f'type="application/pdf"></iframe>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"{render_icon('alert-triangle', size=12)} *{t('reports.iframe_caption')}*",
                unsafe_allow_html=True
            )

            st.download_button(
                label=t("reports.download_pdf"),
                data=pdf_bytes,
                file_name=f"rapport_{record.id}.pdf",
                mime="application/pdf",
                help=t("reports.download_pdf_help"),
                width='stretch',
            )
