"""
settings_store.py

Reglages modifiables depuis la page Parametres (views/settings.py), pour la
duree de la session en cours SEULEMENT — jamais ecrits dans .env ni sur le
disque. Redemarrer l'application (ou ouvrir une nouvelle session/onglet
prive) repart des valeurs de .env, exactement comme avant l'ajout de cette
page. C'est un choix delibere : editer .env depuis l'UI impliquerait
d'ecrire sur le systeme de fichiers du conteneur, ce qui n'a pas sa place
dans une session utilisateur (droits, concurrence entre sessions, valeurs
secretes potentiellement affichees/modifiables sans controle).

La langue (`?lang=`) et le theme (`_dark_mode`) NE PASSENT PAS par ce
module : ils ont deja leur propre persistance dediee (session + URL, voir
translator.py et predict.py) et n'ont pas de sens a etre "reinitialises"
avec le reste des reglages.

Utilisation :

    from settings_store import get_setting, set_setting, is_overridden, reset_all

    alpha = get_setting("default_gradcam_alpha")   # override de session, sinon config.py
    set_setting("default_gradcam_alpha", 0.7)      # ecrit l'override
    is_overridden("default_gradcam_alpha")          # True/False, pour un badge "modifie"
    reset_all()                                      # revient a config.py pour tout

Toutes les pages qui veulent que leurs reglages soient modifiables depuis
Parametres DOIVENT lire via get_setting(cle), pas directement config.cle —
sinon un changement dans Parametres n'a aucun effet ailleurs (voir README,
section reglages modifiables, pour la liste des endroits deja migres).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from config import config

# Les seuls reglages qu'on autorise a modifier depuis l'UI — deliberement
# une liste explicite plutot que "tous les champs de AppConfig", pour ne
# jamais exposer par erreur un futur champ sensible (cle API, chemin
# systeme...) sans y avoir reflechi. La langue/le theme sont exclus (voir
# docstring du module).
EDITABLE_KEYS = [
    "default_gradcam_alpha",
    "max_upload_size_mb",
    "enable_assistant",
    "enable_history",
    "enable_pdf_export",
    "enable_json_export",
    "llm_provider",
    "ollama_base_url",
    "ollama_model",
]

_OVERRIDES_KEY = "_setting_overrides"


def _overrides() -> dict:
    if _OVERRIDES_KEY not in st.session_state:
        st.session_state[_OVERRIDES_KEY] = {}
    return st.session_state[_OVERRIDES_KEY]


def get_setting(key: str) -> Any:
    """Valeur effective d'un reglage : l'override de cette session s'il existe,
    sinon la valeur chargee depuis .env / les defauts (config.py). Fonctionne
    aussi pour des cles hors EDITABLE_KEYS (lecture seule dans ce cas, puisqu'on
    ne peut jamais y avoir ecrit d'override) — pratique pour lire n'importe quel
    champ de AppConfig de facon uniforme sans se demander s'il est modifiable."""
    overrides = _overrides()
    if key in overrides:
        return overrides[key]
    return getattr(config, key)


def set_setting(key: str, value: Any) -> None:
    """Enregistre un override de session. Leve ValueError si `key` n'est pas
    dans EDITABLE_KEYS — erreur de programmation (mauvaise cle cote appelant),
    pas une entree utilisateur a valider silencieusement."""
    if key not in EDITABLE_KEYS:
        raise ValueError(f"'{key}' n'est pas un reglage modifiable (voir settings_store.EDITABLE_KEYS).")
    _overrides()[key] = value


def is_overridden(key: str) -> bool:
    """True si `key` a ete modifie dans cette session (donc differe potentiellement
    de .env) — utilise par l'UI pour afficher un badge "modifie"."""
    return key in _overrides()


def reset_all() -> None:
    """Efface tous les overrides de la session : chaque get_setting() retombe
    immediatement sur la valeur de .env / config.py."""
    st.session_state[_OVERRIDES_KEY] = {}
