"""
views/history.py — Historique des analyses.

Lit/ecrit directement dans persistence.HistoryStore. Les images (originale
et overlay) sont stockees en base pour permettre de regenerer un PDF a la
demande, sans jamais re-executer le modele.
"""

import json

import streamlit as st

from config import CLASS_NAMES, class_color, config
from translator import get_language, t
from persistence import get_store
from report_generator import build_report, generate_pdf_report
from components import section_title, empty_state
from icons import icon as render_icon


def _json_payload(record, language: str) -> dict:
    """Rapport JSON regenere dans la langue courante de l'interface, plutot que le
    texte fige en francais enregistre au moment de l'analyse (voir report_generator.py
    pour le detail multilingue). Les champs propres a l'enregistrement (id, date,
    temps d'inference) restent ceux de persistance.HistoryStore."""
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


def render() -> None:
    section_title("history", t("history.title"), t("history.subtitle"))

    store = get_store()
    language = get_language()

    if store.count() == 0:
        empty_state(
            t("history.empty_title"), t("history.empty_text"),
        )
        return

    # ---- Recherche et filtres ----
    filter_cols = st.columns([2, 1])
    with filter_cols[0]:
        st.markdown(f"{render_icon('search', size=14)} **{t('history.search_label')}**", unsafe_allow_html=True)
        search = st.text_input(
            t("history.search_label"),
            value="",
            placeholder=t("history.search_placeholder"),
            label_visibility="collapsed"  
        )
    with filter_cols[1]:
        class_options = [t("history.all_classes")] + [class_color(c)[2] for c in CLASS_NAMES]
        class_choice = st.selectbox(t("history.filter_by_class"), class_options)

    predicted_class_filter = None
    if class_choice != t("history.all_classes"):
        for cname in CLASS_NAMES:
            if class_color(cname)[2] == class_choice:
                predicted_class_filter = cname
                break

    records = store.list(predicted_class=predicted_class_filter, search=search or None, limit=200)

    st.caption(t("history.results_count", count=len(records)))

    if not records:
        st.info(f"{render_icon('info', size=14)} {t('history.no_results')}")
        return

    for record in records:
        color, soft, label, icon = class_color(record.predicted_class)

        with st.container(border=True):
            header_cols = st.columns([3, 1, 1, 1])
            with header_cols[0]:
                st.markdown(f"""
                <span class="mono" style="font-weight:600;">{render_icon('image', size=14)} {record.image_name}</span><br>
                <span style="font-size:12px; color:var(--text-muted);">{render_icon('clock', size=12)} {record.created_at}</span>
                """, unsafe_allow_html=True)
            with header_cols[1]:
                st.markdown(f'<span class="badge badge-info" style="color:{color}; background:{soft};">{render_icon(icon, size=13, color=color)} {label}</span>', unsafe_allow_html=True)
            with header_cols[2]:
                st.markdown(f"<span class='mono'>{render_icon('gauge', size=12)} {record.confidence:.1f}%</span>", unsafe_allow_html=True)
            with header_cols[3]:
                st.markdown(f"<span style='font-size:12px; color:var(--text-muted);'>{render_icon('cpu', size=12)} {record.inference_ms:.0f} ms</span>", unsafe_allow_html=True)

            with st.expander(t("history.details_export")):
                st.markdown(f"{render_icon('file-text', size=14)} **{t('history.details_export')}**", unsafe_allow_html=True)
                
                st.markdown(f"**{render_icon('eye', size=14)} {t('history.findings')}** — {record.findings}", unsafe_allow_html=True)
                st.markdown(f"**{render_icon('message-circle', size=14)} {t('history.impression')}** — {record.impression}", unsafe_allow_html=True)
                st.markdown(f"**{render_icon('shield-check', size=14)} {t('history.recommendation')}** — {record.recommendation}", unsafe_allow_html=True)

                # === BOUTONS AVEC ICÔNES MATERIAL ===
                # Les icônes Material sont en currentColor, elles héritent
                # automatiquement de la couleur du bouton définie dans theme.py
                action_cols = st.columns(3)
                
                with action_cols[0]:
                    st.download_button(
                        label=t("history.download_json"),
                        data=json.dumps(_json_payload(record, language), indent=2, ensure_ascii=False).encode("utf-8"),
                        file_name=f"analyse_{record.id}.json",
                        mime="application/json",
                        help=t("history.download_json_help"),
                        key=f"json_{record.id}",
                        width='stretch',
                        disabled=not config.enable_json_export,
                        icon=":material/data_object:",
                    )
                    
                with action_cols[1]:
                    if record.has_images() and config.enable_pdf_export:
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
                        st.download_button(
                            label=t("history.download_pdf"),
                            data=pdf_bytes,
                            file_name=f"rapport_{record.id}.pdf",
                            mime="application/pdf",
                            help=t("history.download_pdf_help"),
                            key=f"pdf_{record.id}",
                            width='stretch',
                            icon=":material/picture_as_pdf:",
                        )
                    else:
                        st.button(
                            label=t("history.pdf_unavailable"),
                            disabled=True,
                            key=f"pdf_disabled_{record.id}",
                            width='stretch',
                            help=t("history.pdf_unavailable_help"),
                            icon=":material/block:",
                        )
                        
                with action_cols[2]:
                    if st.button(
                        label=t("history.delete"),
                        key=f"delete_{record.id}",
                        width='stretch',
                        help=t("history.delete_help"),
                        icon=":material/delete:",
                    ):
                        store.delete(record.id)
                        st.rerun()
