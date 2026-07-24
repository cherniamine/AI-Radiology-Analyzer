"""
Tests pour app/icons.py.

Verifie que chaque icone produit du SVG valide (XML bien forme), et que le
repli pour un nom inconnu ne leve jamais d'exception.
"""

import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from icons import icon, available_icons  # noqa: E402
from config import CLASS_META, class_color  # noqa: E402


class TestIconValidity:
    @pytest.mark.parametrize("name", available_icons())
    def test_every_icon_produces_valid_svg(self, name):
        svg = icon(name)
        assert svg.startswith("<svg")
        ET.fromstring(svg)  # leve ET.ParseError si le XML est mal forme

    def test_unknown_icon_falls_back_without_raising(self):
        svg = icon("un_nom_qui_n_existe_vraiment_pas")
        ET.fromstring(svg)
        assert svg.startswith("<svg")

    def test_size_parameter_is_applied(self):
        svg = icon("info", size=32)
        assert 'width="32"' in svg
        assert 'height="32"' in svg

    def test_color_parameter_is_applied(self):
        svg = icon("info", color="#EF4444")
        assert 'stroke="#EF4444"' in svg

    def test_default_color_is_currentcolor(self):
        svg = icon("info")
        assert 'stroke="currentColor"' in svg


class TestClassMetaIconsResolveToRealIcons:
    """Les icones utilisees par CLASS_META (config.py) doivent exister dans icons.py,
    sinon les pages Analyse/Dashboard/Historique afficheraient le repli generique."""

    @pytest.mark.parametrize("class_name", list(CLASS_META.keys()))
    def test_class_icon_is_a_known_icon_name(self, class_name):
        icon_name = CLASS_META[class_name]["icon"]
        assert icon_name in available_icons(), (
            f"L'icone '{icon_name}' utilisee par la classe '{class_name}' "
            f"n'existe pas dans app/icons.py — elle afficherait le repli generique."
        )

    def test_class_color_returns_a_valid_icon_name(self):
        for class_name in CLASS_META:
            _, _, _, icon_name = class_color(class_name)
            svg = icon(icon_name)
            assert svg.startswith("<svg")

    def test_no_emoji_remains_in_class_meta_icons(self):
        # Les emoji sont des caracteres au-dela du plan Unicode de base (> 0x2FFF
        # environ) ; un nom d'icone valide (ex: "flame") ne contient que de l'ASCII.
        for class_name, meta in CLASS_META.items():
            icon_name = meta["icon"]
            assert icon_name.isascii(), f"'{icon_name}' pour {class_name} ressemble a un emoji, pas a un nom d'icone."
