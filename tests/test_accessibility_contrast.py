"""
Tests pour la conformite de contraste WCAG AA des tokens de couleur definis
dans app/theme.py.

Calcule reellement le ratio de contraste (formule WCAG 2.1) pour chaque
paire texte/fond utilisee dans l'application, plutot que de se fier a une
impression visuelle. Empeche la regression des corrections d'accessibilite
apportees (variantes -text pour success/warning/danger/accent, text-muted
assombri) — voir README, section Accessibilite, pour le detail de la
demarche et des valeurs choisies.

Seuils WCAG 2.1 niveau AA :
- Texte normal (< ~18pt ou < ~14pt gras) : 4.5:1
- Texte large (>= 18pt, ou >= 14pt gras) et composants graphiques/icones
  (SC 1.4.11 Non-text Contrast) : 3:1
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

AA_NORMAL_TEXT = 4.5
AA_LARGE_TEXT_OR_GRAPHIC = 3.0


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def relative_luminance(rgb):
    def chan(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def contrast_ratio(rgb1, rgb2):
    l1, l2 = relative_luminance(rgb1), relative_luminance(rgb2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def composite(fg_rgb, alpha, bg_rgb):
    """Couleur reellement affichee quand `fg_rgb` est applique avec `alpha`
    de transparence par-dessus `bg_rgb` (ex: fond teinte rgba(...) d'un badge)."""
    return tuple(round(fg_rgb[i] * alpha + bg_rgb[i] * (1 - alpha)) for i in range(3))


def ratio(fg_hex, bg_hex):
    return contrast_ratio(hex_to_rgb(fg_hex), hex_to_rgb(bg_hex))


# ==============================================================
# MODE CLAIR
# ==============================================================
WHITE = (255, 255, 255)

LIGHT_TEXT_ON_CARD = [
    ("text-primary", "#0F172A"),
    ("text-secondary", "#475569"),
    ("text-muted", "#64748B"),
]

LIGHT_BADGE_TEXT = [
    # (nom, couleur_texte, couleur_base_du_badge, alpha_du_fond)
    ("badge-success", "#047857", (16, 185, 129), 0.12),
    ("badge-warning", "#92400E", (245, 158, 11), 0.12),
    ("badge-danger", "#B91C1C", (239, 68, 68), 0.12),
    ("badge-info", "#1D4ED8", (37, 99, 235), 0.12),
]

LIGHT_ICON_ON_CARD = [
    ("icone succes (pipeline)", "#047857"),
    ("icone avertissement", "#92400E"),
    ("icone danger", "#B91C1C"),
    ("icone accent (section_title)", "#2563EB"),
]


class TestLightModeContrast:
    @pytest.mark.parametrize("name,fg", LIGHT_TEXT_ON_CARD)
    def test_body_text_passes_normal_text_threshold(self, name, fg):
        r = ratio(fg, "#FFFFFF")
        assert r >= AA_NORMAL_TEXT, f"{name} ({fg}) sur carte blanche : {r:.2f}:1, sous le seuil {AA_NORMAL_TEXT}:1"

    @pytest.mark.parametrize("name,fg,base_rgb,alpha", LIGHT_BADGE_TEXT)
    def test_badge_text_passes_normal_text_threshold(self, name, fg, base_rgb, alpha):
        actual_bg = composite(base_rgb, alpha, WHITE)
        r = contrast_ratio(hex_to_rgb(fg), actual_bg)
        assert r >= AA_NORMAL_TEXT, f"{name} : {r:.2f}:1 sur fond reel {actual_bg}, sous le seuil {AA_NORMAL_TEXT}:1"

    @pytest.mark.parametrize("name,fg", LIGHT_ICON_ON_CARD)
    def test_icons_pass_graphic_contrast_threshold(self, name, fg):
        r = ratio(fg, "#FFFFFF")
        assert r >= AA_LARGE_TEXT_OR_GRAPHIC, f"{name} ({fg}) sur carte blanche : {r:.2f}:1, sous le seuil {AA_LARGE_TEXT_OR_GRAPHIC}:1"


# ==============================================================
# MODE SOMBRE
# ==============================================================
DARK_CARD = hex_to_rgb("#161F36")

DARK_TEXT_ON_CARD = [
    ("text-primary (dark)", "#F1F5F9"),
    ("text-secondary (dark)", "#CBD5E1"),
    ("text-muted (dark)", "#7C89A6"),
]

DARK_BADGE_TEXT = [
    ("badge-success (dark)", "#34D399", (16, 185, 129), 0.12),
    ("badge-warning (dark)", "#FCD34D", (245, 158, 11), 0.12),
    ("badge-danger (dark)", "#F87171", (239, 68, 68), 0.12),
    ("badge-info (dark)", "#93C5FD", (37, 99, 235), 0.12),
]


class TestDarkModeContrast:
    @pytest.mark.parametrize("name,fg", DARK_TEXT_ON_CARD)
    def test_body_text_passes_normal_text_threshold(self, name, fg):
        r = contrast_ratio(hex_to_rgb(fg), DARK_CARD)
        assert r >= AA_NORMAL_TEXT, f"{name} ({fg}) sur carte sombre : {r:.2f}:1, sous le seuil {AA_NORMAL_TEXT}:1"

    @pytest.mark.parametrize("name,fg,base_rgb,alpha", DARK_BADGE_TEXT)
    def test_badge_text_passes_normal_text_threshold(self, name, fg, base_rgb, alpha):
        actual_bg = composite(base_rgb, alpha, DARK_CARD)
        r = contrast_ratio(hex_to_rgb(fg), actual_bg)
        assert r >= AA_NORMAL_TEXT, f"{name} : {r:.2f}:1 sur fond reel {actual_bg}, sous le seuil {AA_NORMAL_TEXT}:1"


class TestComponentsUseAccessibleVariants:
    """Garde-fou structurel : verifie que le code utilise bien les variantes
    -text/accessibles la ou c'est necessaire, pas les tokens de marque bruts
    qui echouent le contraste dans un contexte texte/icone."""

    def test_confidence_color_returns_text_variants_not_raw_brand_tokens(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
        from components import confidence_color
        for value in [95.0, 75.0, 40.0]:
            result = confidence_color(value)
            assert result.endswith("-text)"), f"confidence_color({value}) renvoie '{result}', pas une variante -text"
