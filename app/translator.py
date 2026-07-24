"""
translator.py

Internationalisation (i18n) minimale et honnete :

- Dictionnaires de traduction charges depuis app/locales/{lang}.json
- t("cle.pointee") renvoie la traduction ; si la cle manque dans la langue
  courante, repli sur l'anglais, puis en dernier recours renvoie la cle
  elle-meme (visible, donc detectable) plutot que d'inventer un texte ou
  de faire planter la page.
- La langue courante persiste via l'URL (?lang=fr), pas seulement
  st.session_state : un lien partage ou un rechargement de page conservent
  donc la langue choisie, ce qu'une simple variable de session ne permet
  pas.

Ce module ne couvre aujourd'hui que la navigation, la page "A propos" et la
page "Parametres" (voir README, section Pistes d'amelioration, pour le
perimetre exact) — les autres pages restent en francais tant qu'elles n'ont
pas ete migrees vers t().
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Optional

import streamlit as st

LOCALES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")

SUPPORTED_LANGUAGES: dict[str, dict[str, Any]] = {
    "fr": {"label": "Français", "flag": "🇫🇷", "rtl": False},
    "en": {"label": "English", "flag": "🇺🇸", "rtl": False},
    "ar": {"label": "العربية", "flag": "🇸🇦", "rtl": True},
}

FALLBACK_LANGUAGE = "en"


@lru_cache(maxsize=None)
def _load_locale(lang: str) -> dict:
    """Charge et met en cache le dictionnaire de traduction d'une langue. {} si le fichier est absent/invalide."""
    path = os.path.join(LOCALES_DIR, f"{lang}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _lookup(data: dict, dotted_key: str) -> Optional[Any]:
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def get_language() -> str:
    """
    Langue courante, dans cet ordre de priorite :
    1. Parametre d'URL ?lang= (permet de partager un lien dans une langue donnee)
    2. Choix precedent dans la session (st.session_state)
    3. DEFAULT_LANGUAGE de la configuration (.env)
    4. Repli final : francais
    """
    query_lang = st.query_params.get("lang")
    if query_lang in SUPPORTED_LANGUAGES:
        st.session_state["_lang"] = query_lang
        return query_lang

    if st.session_state.get("_lang") in SUPPORTED_LANGUAGES:
        return st.session_state["_lang"]

    try:
        from config import config
        default = config.default_language
    except Exception:
        default = "fr"

    if default not in SUPPORTED_LANGUAGES:
        default = "fr"

    st.session_state["_lang"] = default
    return default


def set_language(lang: str) -> None:
    """Change la langue courante et la reflete dans l'URL pour qu'elle persiste au rechargement."""
    if lang not in SUPPORTED_LANGUAGES:
        return
    st.session_state["_lang"] = lang
    st.query_params["lang"] = lang


def is_rtl() -> bool:
    return SUPPORTED_LANGUAGES.get(get_language(), {}).get("rtl", False)


def t(key: str, **kwargs) -> str:
    """
    Traduit `key` (notation pointee, ex. "settings.title") dans la langue
    courante. Ne renvoie jamais une exception ni un texte invente : repli
    sur l'anglais, puis sur la cle brute si vraiment introuvable.
    """
    lang = get_language()
    value = _lookup(_load_locale(lang), key)

    if value is None and lang != FALLBACK_LANGUAGE:
        value = _lookup(_load_locale(FALLBACK_LANGUAGE), key)

    if value is None:
        return key  # cle manquante : visible et grep-able, jamais un texte fabrique

    if isinstance(value, str) and kwargs:
        try:
            return value.format(**kwargs)
        except Exception:
            return value

    return value
