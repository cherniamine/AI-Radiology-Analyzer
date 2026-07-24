"""
views/dashboard.py — Tableau de bord.

Toutes les valeurs affichees viennent de persistence.HistoryStore.stats() —
aucun chiffre n'est invente. Si aucune analyse n'a encore ete enregistree,
la page l'indique explicitement plutot que d'afficher des zeros qui
pourraient passer pour un vrai resultat.

Refactorise pour utiliser app/components.py (SectionTitle, MetricCard,
EmptyState) au lieu de blocs HTML dupliques inline.
"""

import plotly.graph_objects as go
import streamlit as st

from config import CLASS_NAMES, class_color
from persistence import get_store
from components import section_title, metric_card, empty_state
from icons import icon as render_icon


def render() -> None:
    section_title(
        "bar-chart-3", "Dashboard",
        "Vue d'ensemble des analyses effectuées dans cette application",
    )

    store = get_store()
    stats = store.stats()

    if stats["total"] == 0:
        empty_state(
            "Aucune analyse enregistrée pour l'instant",
            "Les statistiques apparaîtront ici après votre première analyse. "
            "Rendez-vous dans <b>🔬 Nouvelle analyse</b> pour commencer.",
        )
        return

    # ---- KPI : total + une carte par classe ----
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        metric_card(str(stats["total"]), "Analyses totales")

    for i, cname in enumerate(CLASS_NAMES):
        color, soft, label, icon = class_color(cname)
        count = stats["per_class"].get(cname, 0)
        with kpi_cols[i + 1]:
            metric_card(str(count), f'{render_icon(icon, size=13, color=color)} {label}', color=color)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    kpi_cols2 = st.columns(2)
    with kpi_cols2[0]:
        avg_conf_display = f"{stats['avg_confidence']:.1f}%" if stats["avg_confidence"] is not None else "N/A"
        metric_card(avg_conf_display, "Confiance moyenne")
    with kpi_cols2[1]:
        avg_inf_display = f"{stats['avg_inference_ms']:.0f} ms" if stats["avg_inference_ms"] is not None else "N/A"
        metric_card(avg_inf_display, "Temps d'inférence moyen")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ---- Graphiques ----
    chart_cols = st.columns(2)

    with chart_cols[0]:
        with st.container(border=True):
            st.markdown(f"<h4>{render_icon('chart-pie', size=16)} Répartition par classe</h4>", unsafe_allow_html=True)
            per_class = stats["per_class"]
            if per_class:
                colors_map = {c: class_color(c)[0] for c in CLASS_NAMES}
                labels = [class_color(c)[2] for c in per_class.keys()]
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=list(per_class.values()),
                    hole=0.5,
                    marker=dict(colors=[colors_map.get(c, "#64748B") for c in per_class.keys()],
                                line=dict(color="#FFFFFF", width=2)),
                    textinfo="percent+label",
                )])
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                                   font=dict(family="Inter", size=12, color="#475569"),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width='stretch')

    with chart_cols[1]:
        with st.container(border=True):
            st.markdown(f"<h4>{render_icon('chart-line', size=16)} Analyses dans le temps</h4>", unsafe_allow_html=True)
            timeline = stats["timeline"]
            if timeline:
                days = [row["day"] for row in timeline]
                counts = [row["count"] for row in timeline]
                fig2 = go.Figure(data=[go.Scatter(
                    x=days, y=counts, mode="lines+markers",
                    line=dict(color="#2563EB", width=2.5),
                    marker=dict(size=7, color="#2563EB"),
                    fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.08)",
                )])
                fig2.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10), height=320,
                    font=dict(family="Inter", size=12, color="#475569"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Date", yaxis_title="Analyses",
                )
                st.plotly_chart(fig2, width='stretch')
            else:
                st.info("Pas assez de données pour une évolution temporelle.")

    st.caption(
        f"Basé sur {stats['total']} analyse(s) réellement enregistrée(s) — "
        "aucune donnée simulée. Désactivable via ENABLE_HISTORY dans .env."
    )
