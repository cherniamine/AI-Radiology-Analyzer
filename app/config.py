"""
config.py

Configuration centralisee de l'application, chargee depuis les variables
d'environnement (via un fichier .env local, voir .env.example a la racine
du depot). Aucune valeur de configuration ne doit etre codee en dur ailleurs
dans le code — importer `config` depuis ce module a la place.

Exemple :
    from config import config
    model_path = config.model_path
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# app/config.py -> app/ -> racine du depot
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# Charge .env s'il existe (sinon les valeurs par defaut ci-dessous s'appliquent).
# N'ecrase jamais une variable deja presente dans l'environnement (utile en
# conteneur Docker, ou les variables sont souvent injectees directement).
load_dotenv(REPO_ROOT / ".env", override=False)


def _get_bool(name: str, default: bool) -> bool:
    """Parse une variable d'environnement booleenne ('true'/'1'/'yes' -> True)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"true", "1", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_path(raw: str) -> str:
    """Resout un chemin relatif au dossier app/ (comportement historique de l'app)."""
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((APP_DIR / raw).resolve())


@dataclass(frozen=True)
class AppConfig:
    """Configuration immuable de l'application, construite une fois au demarrage."""

    # Chemins
    model_path: str
    metrics_path: str
    report_output: str
    history_db_path: str

    # Application
    app_title: str
    app_version: str
    default_language: str
    default_theme: str
    default_gradcam_alpha: float
    max_upload_size_mb: int

    # Fonctionnalites
    enable_assistant: bool
    enable_history: bool
    enable_pdf_export: bool
    enable_json_export: bool

    # Assistant IA (Ollama par defaut — voir Phase 7 de la feuille de route)
    llm_provider: str
    ollama_base_url: str
    ollama_model: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            model_path=_resolve_path(os.getenv("MODEL_PATH", "../models/simple_cnn_model.h5")),
            metrics_path=_resolve_path(os.getenv("METRICS_PATH", "../results/metrics.json")),
            report_output=_resolve_path(os.getenv("REPORT_OUTPUT", "../results/reports")),
            history_db_path=_resolve_path(os.getenv("HISTORY_DB_PATH", "../results/history.db")),
            app_title=os.getenv("APP_TITLE", "AI Radiology Analyzer"),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            default_language=os.getenv("DEFAULT_LANGUAGE", "fr"),
            default_theme=os.getenv("DEFAULT_THEME", "light"),
            default_gradcam_alpha=_get_float("DEFAULT_GRADCAM_ALPHA", 0.5),
            max_upload_size_mb=_get_int("MAX_UPLOAD_SIZE_MB", 10),
            enable_assistant=_get_bool("ENABLE_ASSISTANT", True),
            enable_history=_get_bool("ENABLE_HISTORY", True),
            enable_pdf_export=_get_bool("ENABLE_PDF_EXPORT", True),
            enable_json_export=_get_bool("ENABLE_JSON_EXPORT", True),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        )


# Singleton importable partout dans l'application.
config: AppConfig = AppConfig.from_env()


# ==============================================================
# Constantes du domaine (pas des reglages d'environnement, mais
# des faits sur le modele/les donnees — centralisees ici pour
# eviter la duplication entre les pages).
# ==============================================================
CLASS_NAMES = ["COVID", "Lung_Opacity", "NORMAL", "Viral Pneumonia"]

CLASS_META = {
    "COVID": {
        "label": "COVID-19",
        "color": "#DC3545",
        "soft": "rgba(220, 53, 69, 0.10)",
        "icon": "flame",
        "description": "Infection à SARS-CoV-2",
    },
    "Lung_Opacity": {
        "label": "Opacité pulmonaire",
        "color": "#FD7E14",
        "soft": "rgba(253, 126, 20, 0.10)",
        "icon": "search",
        "description": "Anomalie interstitielle ou alvéolaire",
    },
    "NORMAL": {
        "label": "Normal",
        "color": "#28A745",
        "soft": "rgba(40, 167, 69, 0.10)",
        "icon": "shield-check",
        "description": "Radiographie normale",
    },
    "Viral Pneumonia": {
        "label": "Pneumonie virale",
        "color": "#007BFF",
        "soft": "rgba(0, 123, 255, 0.10)",
        "icon": "hospital",
        "description": "Infection virale pulmonaire",
    },
}

# Taille reelle du jeu de donnees d'entrainement (voir README, section Jeu de donnees)
DATASET_SIZE = 42330


def class_color(name: str) -> tuple[str, str, str, str]:
    """Retourne (couleur, couleur_douce, libelle, nom_icone) pour une classe donnee.
    Le nom d'icone est une cle de app/icons.py (icons.icon(nom)), pas un emoji."""
    meta = CLASS_META.get(
        name, {"color": "#64748B", "soft": "rgba(100, 116, 139, 0.10)", "label": name, "icon": "bar-chart-3"}
    )
    return meta["color"], meta["soft"], meta["label"], meta.get("icon", "bar-chart-3")


def load_real_metrics() -> dict | None:
    """
    Charge les metriques d'evaluation reelles produites par src/evaluate.py.
    Retourne None si le fichier est absent — les pages appelantes doivent
    afficher "N/A" plutot que d'inventer une valeur par defaut.
    """
    import json

    try:
        with open(config.metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

