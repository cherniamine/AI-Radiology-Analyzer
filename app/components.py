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
import os  
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
        icon_svg = render_icon(icon_name, size=14, color=color)
        rows.append(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                <span style="color:var(--text-secondary);display:flex;align-items:center;gap:6px;">{icon_svg} {label}</span>
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
#
# IMPORTANT (bug corrige) : ces deux fonctions renvoient volontairement du
# HTML SANS retour a la ligne ni indentation dans le balisage que NOUS
# ecrivons (seul `text`/`source_items`, fournis par l'appelant, peuvent
# contenir des retours a la ligne). Raison : le moteur Markdown de
# Streamlit (st.markdown) traite toute ligne indentee de 4 espaces ou plus
# et precedee d'une ligne vide comme un BLOC DE CODE indente (regle
# CommonMark), pas comme du HTML a injecter. `text` est la reponse
# generee par le LLM et contient souvent des sauts de paragraphe (lignes
# vides) ; des lors qu'un tel saut de paragraphe apparaissait dans `text`
# AVANT le bloc `sources_html` (qui, lui, etait indente de 8 espaces dans
# le code source), le parseur sortait du mode "bloc HTML brut" et
# affichait tout le reste litteralement — c'est le bug observe en
# production (voir capture d'ecran : la balise `<div class="sidebar-header">`
# et les balises suivantes s'affichaient telles quelles au lieu d'etre
# rendues). Ecrire le balisage sur une seule ligne rend le bug impossible
# a reproduire, quel que soit le contenu de `text`.
def _build_user_message_html(text: str) -> str:
    return (
        '<div style="display:flex; justify-content:flex-end; margin-bottom:10px;">'
        '<div style="max-width:80%; background:var(--accent-gradient); color:#FFFFFF; '
        'padding:10px 16px; border-radius:16px 16px 4px 16px; font-size:14px; line-height:1.5;">'
        f'{text}'
        '</div>'
        '</div>'
    )


def _build_assistant_message_html(text: str, sources: Optional[List[str]] = None) -> str:
    """Construit le HTML pour un message de l'assistant avec ses sources."""
    sources_html = ""
    if sources:
        # Nettoyer et formater les noms de sources
        clean_sources = []
        for s in sources:
            # Si c'est un chemin de fichier, ne garder que le nom du fichier
            if isinstance(s, str):
                # Enlever le chemin complet
                clean_name = os.path.basename(s)
                # Enlever l'extension
                clean_name = os.path.splitext(clean_name)[0]
                # Remplacer les underscores par des espaces
                clean_name = clean_name.replace("_", " ").replace("-", " ")
                # Mettre en titre (première lettre majuscule)
                clean_name = clean_name.title()
                clean_sources.append(clean_name)
            else:
                clean_sources.append(str(s))

        # Éviter les doublons
        clean_sources = list(dict.fromkeys(clean_sources))

        # Limiter à 5 sources maximum pour ne pas surcharger
        if len(clean_sources) > 5:
            clean_sources = clean_sources[:5]
            clean_sources.append("...")

        source_items = "".join(
            f'<span class="badge badge-info" style="font-size:10px; margin:2px 4px 2px 0; display:inline-block; background:var(--bg-input); color:var(--text-secondary); border:1px solid var(--border-color); padding:2px 8px; border-radius:12px;">{render_icon("file-text", size=10)} {s}</span>'
            for s in clean_sources
        )
        sources_html = (
            '<div style="margin-top:8px; padding-top:6px; border-top:1px solid var(--border-color);">'
            f'<span style="font-size:11px; color:var(--text-muted);">{render_icon("paperclip", size=12)} Sources :</span>'
            f'<div style="margin-top:4px; display:flex; flex-wrap:wrap; gap:4px;">{source_items}</div>'
            '</div>'
        )
    return (
        '<div style="display:flex; justify-content:flex-start; margin-bottom:10px;">'
        '<div style="max-width:80%; background:var(--bg-input); color:var(--text-primary); '
        'padding:10px 16px; border-radius:16px 16px 16px 4px; font-size:14px; line-height:1.5;">'
        f'{text}{sources_html}'
        '</div>'
        '</div>'
    )



def user_message(text: str) -> None:
    st.markdown(_build_user_message_html(text), unsafe_allow_html=True)


def assistant_message(text: str, sources: Optional[List[str]] = None) -> None:
    st.markdown(_build_assistant_message_html(text, sources), unsafe_allow_html=True)


# ==============================================================
# Footer
# ==============================================================
def _build_footer_html(
    app_title: str, 
    version: str, 
    license_name: str, 
    tech_list: List[str],
    built_with_text: Optional[str] = None
) -> str:
    """Construit le footer avec texte traduit possible."""
    if built_with_text is None:
        tech_str = ", ".join(tech_list)
        built_with_text = f"Construit avec {tech_str}."
    return f"""
    <div class="footer">
        <p>{app_title} &middot; Version {version} &middot; Licence {license_name}</p>
        <p style="font-size:11px;">{built_with_text}</p>
    </div>
    """


def footer(
    app_title: str, 
    version: str, 
    license_name: str, 
    tech_list: List[str],
    built_with_text: Optional[str] = None
) -> None:
    """Affiche le footer avec texte traduit possible."""
    st.markdown(_build_footer_html(app_title, version, license_name, tech_list, built_with_text), unsafe_allow_html=True)