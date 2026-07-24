"""
components.py

Bibliotheque de composants UI reutilisables, pour eviter la duplication de
HTML inline qui existait auparavant dans chaque page de views/ (le meme
bloc "metric-card", "empty state", etc. copie-colle a plusieurs endroits).

Chaque composant est scinde en deux :
  - une fonction pure `_build_xxx_html(...)` qui construit et renvoie une
    chaine HTML, sans dependance a Streamlit — testable directement (voir
    tests/test_components.py) sans navigateur ni ScriptRunContext ;
  - une fonction `xxx(...)` qui appelle `st.markdown(..., unsafe_allow_html=True)`
    avec ce HTML — c'est celle-ci que les pages (views/*.py) utilisent.

Toutes les couleurs viennent des variables CSS definies dans theme.py
(var(--success), var(--accent-primary), etc.) plutot que d'etre codees en
dur, pour rester coherentes si la palette change.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from icons import icon as render_icon

# ==============================================================
# SectionTitle — en-tete de page (remplace le bloc "main-header" duplique
# au debut de chaque views/*.py)
# ==============================================================
def _build_section_title_html(icon_name: str, title: str, subtitle: Optional[str] = None) -> str:
    subtitle_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    icon_html = render_icon(icon_name, size=26, color="var(--accent-primary)")
    return f"""
    <div class="main-header fade-in">
        <h1>{icon_html} {title}</h1>
        {subtitle_html}
    </div>
    """


def section_title(icon_name: str, title: str, subtitle: Optional[str] = None) -> None:
    st.markdown(_build_section_title_html(icon_name, title, subtitle), unsafe_allow_html=True)


# ==============================================================
# MetricCard
# ==============================================================
def _build_metric_card_html(value: str, label: str, color: Optional[str] = None) -> str:
    color_style = f'style="color:{color};"' if color else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value" {color_style}>{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def metric_card(value: str, label: str, color: Optional[str] = None) -> None:
    st.markdown(_build_metric_card_html(value, label, color), unsafe_allow_html=True)


# ==============================================================
# StatusBadge
# ==============================================================
_BADGE_VARIANTS = {"success", "warning", "danger", "info"}


def _build_status_badge_html(text: str, variant: str = "info", icon_name: Optional[str] = None) -> str:
    variant = variant if variant in _BADGE_VARIANTS else "info"
    icon_html = f'{render_icon(icon_name, size=12)} ' if icon_name else ""
    return f'<span class="badge badge-{variant}">{icon_html}{text}</span>'


def status_badge(text: str, variant: str = "info", icon_name: Optional[str] = None) -> None:
    st.markdown(_build_status_badge_html(text, variant, icon_name), unsafe_allow_html=True)


# ==============================================================
# GlassCard — conteneur generique (`.card`), pour eviter de retaper
# `<div class="card">...</div>` a chaque fois
# ==============================================================
def _build_glass_card_html(content_html: str, extra_style: str = "") -> str:
    style_attr = f' style="{extra_style}"' if extra_style else ""
    return f'<div class="card fade-in"{style_attr}>{content_html}</div>'


def glass_card(content_html: str, extra_style: str = "") -> None:
    st.markdown(_build_glass_card_html(content_html, extra_style), unsafe_allow_html=True)


# ==============================================================
# EmptyState — remplace le bloc "Aucune analyse..." duplique dans
# dashboard.py et history.py
# ==============================================================
def _build_empty_state_html(title: str, description: str) -> str:
    return f"""
    <div class="card fade-in" style="text-align:center; padding: 48px 32px;">
        <h3 style="margin-bottom:8px;">{title}</h3>
        <p style="color:var(--text-secondary); font-size:14px; max-width:480px; margin:0 auto;">
            {description}
        </p>
    </div>
    """


def empty_state(title: str, description: str) -> None:
    st.markdown(_build_empty_state_html(title, description), unsafe_allow_html=True)


# ==============================================================
# ConfidenceGauge — jauge circulaire SVG (remplace le pourcentage brut)
# ==============================================================
def confidence_color(value_percent: float) -> str:
    """Couleur de confiance (variante texte/icone, contraste WCAG verifie) selon les
    memes seuils que partout ailleurs dans l'application : >=80% succes, >=60% avertissement, sinon danger."""
    if value_percent >= 80:
        return "var(--success-text)"
    if value_percent >= 60:
        return "var(--warning-text)"
    return "var(--danger-text)"


# Alias retro-compatible (nom prive utilise historiquement dans ce module)
_confidence_color = confidence_color


def _build_confidence_gauge_html(value_percent: float, label: str = "Confiance", size: int = 140) -> str:
    value_percent = max(0.0, min(100.0, value_percent))
    radius = size / 2 - 10
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - value_percent / 100)
    color = _confidence_color(value_percent)
    center = size / 2
    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="transform:rotate(-90deg);">
            <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
                    stroke="var(--border-color)" stroke-width="10" />
            <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
                    stroke="{color}" stroke-width="10" stroke-linecap="round"
                    stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"
                    style="transition: stroke-dashoffset 0.6s ease;" />
        </svg>
        <div style="margin-top:-{size / 2 + 14}px; font-size:{size * 0.19:.0f}px; font-weight:700; color:{color};">
            {value_percent:.1f}%
        </div>
        <div style="margin-top:{size / 2 - 8}px; font-size:12px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em;">
            {label}
        </div>
    </div>
    """


def confidence_gauge(value_percent: float, label: str = "Confiance", size: int = 140) -> None:
    st.markdown(_build_confidence_gauge_html(value_percent, label, size), unsafe_allow_html=True)


# ==============================================================
# ProbabilityBars — barres de probabilite par classe (remplace le texte brut)
# ==============================================================
def _build_probability_bars_html(class_probabilities: Dict[str, float], color_fn) -> str:
    rows = []
    for name, prob in sorted(class_probabilities.items(), key=lambda kv: -kv[1]):
        color, _soft, label, icon_name = color_fn(name)
        rows.append(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                <span style="color:var(--text-secondary);">{render_icon(icon_name, size=14, color=color)} {label}</span>
                <span style="font-weight:600; color:{color};">{prob:.1f}%</span>
            </div>
            <div style="height:8px; background:var(--bg-input); border-radius:6px; overflow:hidden;">
                <div style="width:{prob:.1f}%; height:100%; background:{color}; border-radius:6px;
                            transition: width 0.6s ease;"></div>
            </div>
        </div>
        """)
    return "\n".join(rows)


def probability_bars(class_probabilities: Dict[str, float], color_fn) -> None:
    st.markdown(_build_probability_bars_html(class_probabilities, color_fn), unsafe_allow_html=True)


# ==============================================================
# Chat bubbles — AssistantMessage / UserMessage (page Assistant IA)
# ==============================================================
def _build_user_message_html(text: str) -> str:
    return f"""
    <div style="display:flex; justify-content:flex-end; margin-bottom:10px;">
        <div style="max-width:80%; background:var(--accent-gradient); color:#FFFFFF;
                    padding:10px 16px; border-radius:16px 16px 4px 16px; font-size:14px; line-height:1.5;">
            {text}
        </div>
    </div>
    """


def _build_assistant_message_html(text: str, sources: Optional[List[str]] = None) -> str:
    sources_html = ""
    if sources:
        sources_html = (
            '<div style="margin-top:6px; font-size:11px; color:var(--text-muted);">'
            f'Sources : {", ".join(sources)}</div>'
        )
    return f"""
    <div style="display:flex; justify-content:flex-start; margin-bottom:10px;">
        <div style="max-width:80%; background:var(--bg-input); color:var(--text-primary);
                    padding:10px 16px; border-radius:16px 16px 16px 4px; font-size:14px; line-height:1.5;">
            {text}
            {sources_html}
        </div>
    </div>
    """


def user_message(text: str) -> None:
    st.markdown(_build_user_message_html(text), unsafe_allow_html=True)


def assistant_message(text: str, sources: Optional[List[str]] = None) -> None:
    st.markdown(_build_assistant_message_html(text, sources), unsafe_allow_html=True)


# ==============================================================
# Footer
# ==============================================================
def _build_footer_html(app_title: str, version: str, license_name: str, tech_list: List[str]) -> str:
    tech_str = ", ".join(tech_list)
    return f"""
    <div class="footer">
        <p>{app_title} &middot; Version {version} &middot; Licence {license_name}</p>
        <p style="font-size:11px;">Construit avec {tech_str}.</p>
    </div>
    """


def footer(app_title: str, version: str, license_name: str, tech_list: List[str]) -> None:
    st.markdown(_build_footer_html(app_title, version, license_name, tech_list), unsafe_allow_html=True)
