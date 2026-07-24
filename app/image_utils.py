"""
image_utils.py

Logique de traitement d'image pure, sans dependance a TensorFlow ni a
Streamlit : validation des entrees, pretraitement, colorisation et fusion
des cartes Grad-CAM. Extrait de app/predict.py pour etre testable
unitairement (voir tests/test_image_utils.py) sans avoir a charger le
modele entraine ni l'environnement Streamlit — ce qui garde la suite de
tests et la pipeline CI rapides.

Le calcul du gradient Grad-CAM lui-meme (qui necessite le modele Keras)
reste dans app/predict.py ; ce module se limite a ce qui s'applique a une
heatmap ou une image deja obtenue.
"""

import numpy as np
import cv2

# ==============================================================
# VALIDATION DES ENTREES
# ==============================================================
MIN_VALID_DIMENSION = 64   # en dessous, une radiographie n'a plus de contenu exploitable
MAX_MEAN_SATURATION = 60   # 0-255 ; les radiographies sont quasi grises, une photo couleur est bien au-dessus
MIN_PIXEL_STD = 8          # une image quasi unie (page blanche, capture d'erreur) a un ecart-type tres faible

COLORMAP_DICT = {
    "JET": cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "INFERNO": cv2.COLORMAP_INFERNO,
}


def validate_xray_image(img_rgb):
    """
    Sanity checks avant inference. Ce sont des heuristiques, pas une detection
    garantie : elles filtrent les cas evidents (image corrompue, capture d'ecran
    unie, photo couleur) plutot que de faire tourner le modele sur une entree
    qui n'a aucune chance de ressembler a une radiographie.
    Retourne (is_valid: bool, reason: str | None).
    """
    h, w = img_rgb.shape[:2]
    if h < MIN_VALID_DIMENSION or w < MIN_VALID_DIMENSION:
        return False, f"Resolution trop faible ({w}x{h}px, minimum {MIN_VALID_DIMENSION}x{MIN_VALID_DIMENSION}px)."

    if img_rgb.std() < MIN_PIXEL_STD:
        return False, "Image quasi uniforme (pas de contenu detectable) : capture d'ecran vide ou fichier corrompu ?"

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mean_saturation = hsv[:, :, 1].mean()
    if mean_saturation > MAX_MEAN_SATURATION:
        return False, (
            f"Saturation colorimetrique elevee ({mean_saturation:.0f}/255) : "
            "cette image ne ressemble pas a une radiographie en niveaux de gris."
        )

    return True, None


def load_and_preprocess_image(uploaded_file, target_size):
    """Decode + valide + redimensionne un fichier uploade. Retourne (img_rgb, img_array, reason)."""
    try:
        raw_bytes = uploaded_file.read()
        if not raw_bytes:
            return None, None, "Fichier vide."

        file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        uploaded_file.seek(0)

        if img_cv is None:
            return None, None, "Format d'image illisible ou fichier corrompu."

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        is_valid, reason = validate_xray_image(img_rgb)
        if not is_valid:
            return None, None, reason

        img_array = preprocess_image(img_cv, target_size)
        if img_array is None:
            return None, None, "Echec du pretraitement (redimensionnement impossible)."

        return img_rgb, img_array, None
    except Exception as e:
        return None, None, f"Erreur inattendue : {str(e)}"


def preprocess_image(img_cv, target_size):
    """Redimensionne et normalise une image (BGR, uint8) pour l'entree du modele."""
    try:
        img_resized = cv2.resize(img_cv, target_size)
        img_array = img_resized.astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception:
        return None


# ==============================================================
# GRAD-CAM : COLORISATION ET FUSION (post-traitement d'une heatmap deja calculee)
# ==============================================================
def colorize_heatmap(original_shape, heatmap, colormap="JET"):
    """Retourne la heatmap Grad-CAM colorisee seule (sans fusion avec l'image d'origine)."""
    try:
        heatmap_resized = cv2.resize(heatmap, (original_shape[1], original_shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), COLORMAP_DICT.get(colormap, cv2.COLORMAP_JET)
        )
        return cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)


def overlay_heatmap_cv(original_img, heatmap, alpha=0.5, colormap="JET"):
    """Fusionne la heatmap colorisee avec l'image d'origine (RGB, uint8)."""
    try:
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), COLORMAP_DICT.get(colormap, cv2.COLORMAP_JET)
        )
        if len(original_img.shape) == 2:
            original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        elif original_img.shape[2] == 3:
            original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        else:
            original_img_bgr = original_img
        overlay = cv2.addWeighted(original_img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    except Exception:
        return original_img
