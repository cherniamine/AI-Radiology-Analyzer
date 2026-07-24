"""
Tests pour app/image_utils.py.

Ne dependent que de numpy et opencv (pas de TensorFlow, pas de Streamlit) :
ils s'executent en quelques secondes, y compris en CI. Le calcul du gradient
Grad-CAM lui-meme (qui necessite le modele Keras charge) n'est pas couvert
ici par choix — voir la note dans .github/workflows/ci.yml.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from image_utils import (  # noqa: E402
    validate_xray_image,
    preprocess_image,
    colorize_heatmap,
    overlay_heatmap_cv,
    MIN_VALID_DIMENSION,
)


def make_grayscale_xray_like(size=256):
    """Image en niveaux de gris avec du bruit, pour imiter une radiographie."""
    rng = np.random.default_rng(42)
    gray = rng.integers(20, 220, size=(size, size), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def make_color_photo(size=256):
    """Image fortement saturee (rouge), pour imiter une photo couleur non medicale."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = 220   # canal rouge dominant
    img[:, :, 1] = 30
    img[:, :, 2] = 30
    return img


def make_uniform_image(size=256, value=200):
    """Image quasi unie, pour imiter une capture vide ou un fichier corrompu."""
    return np.full((size, size, 3), value, dtype=np.uint8)


# ==============================================================
# validate_xray_image
# ==============================================================
class TestValidateXrayImage:
    def test_valid_grayscale_xray_like_image_is_accepted(self):
        img = make_grayscale_xray_like()
        is_valid, reason = validate_xray_image(img)
        assert is_valid is True
        assert reason is None

    def test_image_too_small_is_rejected(self):
        img = make_grayscale_xray_like(size=MIN_VALID_DIMENSION - 1)
        is_valid, reason = validate_xray_image(img)
        assert is_valid is False
        assert "resolution" in reason.lower() or "faible" in reason.lower()

    def test_uniform_blank_image_is_rejected(self):
        img = make_uniform_image()
        is_valid, reason = validate_xray_image(img)
        assert is_valid is False
        assert "uniforme" in reason.lower()

    def test_saturated_color_photo_is_rejected(self):
        img = make_color_photo()
        is_valid, reason = validate_xray_image(img)
        assert is_valid is False
        assert "saturation" in reason.lower() or "couleur" in reason.lower()

    def test_minimum_valid_dimension_boundary_is_accepted(self):
        img = make_grayscale_xray_like(size=MIN_VALID_DIMENSION)
        is_valid, _ = validate_xray_image(img)
        assert is_valid is True


# ==============================================================
# preprocess_image
# ==============================================================
class TestPreprocessImage:
    def test_output_shape_matches_target_size(self):
        img_bgr = cv2.cvtColor(make_grayscale_xray_like(), cv2.COLOR_RGB2BGR)
        result = preprocess_image(img_bgr, (128, 128))
        assert result is not None
        assert result.shape == (1, 128, 128, 3)

    def test_output_is_normalized_between_0_and_1(self):
        img_bgr = cv2.cvtColor(make_grayscale_xray_like(), cv2.COLOR_RGB2BGR)
        result = preprocess_image(img_bgr, (128, 128))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_invalid_target_size_returns_none_instead_of_raising(self):
        img_bgr = cv2.cvtColor(make_grayscale_xray_like(), cv2.COLOR_RGB2BGR)
        result = preprocess_image(img_bgr, None)
        assert result is None


# ==============================================================
# Grad-CAM post-traitement : colorize_heatmap / overlay_heatmap_cv
# (a partir d'une heatmap simulee — pas d'inference reelle du modele)
# ==============================================================
@pytest.fixture
def fake_heatmap():
    """Heatmap Grad-CAM simulee (petite resolution, comme en sortie de la derniere Conv2D)."""
    rng = np.random.default_rng(0)
    return rng.random((16, 16)).astype(np.float32)


class TestColorizeHeatmap:
    def test_output_has_same_spatial_dimensions_as_original(self, fake_heatmap):
        original = make_grayscale_xray_like(size=200)
        result = colorize_heatmap(original.shape, fake_heatmap, colormap="JET")
        assert result.shape == (200, 200, 3)

    def test_output_is_a_valid_uint8_rgb_image(self, fake_heatmap):
        original = make_grayscale_xray_like(size=100)
        result = colorize_heatmap(original.shape, fake_heatmap, colormap="JET")
        assert result.dtype == np.uint8
        assert result.min() >= 0 and result.max() <= 255

    @pytest.mark.parametrize("colormap", ["JET", "HOT", "PLASMA", "VIRIDIS", "INFERNO", "UNKNOWN_FALLBACK"])
    def test_all_supported_colormaps_run_without_error(self, fake_heatmap, colormap):
        original = make_grayscale_xray_like(size=100)
        result = colorize_heatmap(original.shape, fake_heatmap, colormap=colormap)
        assert result.shape == (100, 100, 3)


class TestOverlayHeatmapCv:
    def test_output_has_same_dimensions_as_original(self, fake_heatmap):
        original = make_grayscale_xray_like(size=200)
        result = overlay_heatmap_cv(original, fake_heatmap, alpha=0.5, colormap="JET")
        assert result.shape == original.shape

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_alpha_range_runs_without_error(self, fake_heatmap, alpha):
        original = make_grayscale_xray_like(size=100)
        result = overlay_heatmap_cv(original, fake_heatmap, alpha=alpha, colormap="JET")
        assert result.shape == original.shape
        assert result.dtype == np.uint8

    def test_png_encoding_of_overlay_succeeds(self, fake_heatmap):
        original = make_grayscale_xray_like(size=100)
        overlay = overlay_heatmap_cv(original, fake_heatmap, alpha=0.5, colormap="JET")
        ok, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        assert ok is True
        assert len(buffer) > 0
        # Un PNG valide commence toujours par cette signature d'octets
        assert bytes(buffer[:8]) == b"\x89PNG\r\n\x1a\n"
