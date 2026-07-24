"""
report_generator.py

Genere un rapport structure a partir de la prediction du modele de
classification (classe + probabilites par classe) utilise par l'application
Streamlit (app/predict.py).

Portee et limites importantes (a lire avant toute modification) :

- Le modele fait UNIQUEMENT de la classification d'image entiere sur 4
  classes (COVID, Lung_Opacity, NORMAL, Viral Pneumonia). Il ne localise
  aucune anomalie de facon validee : la carte Grad-CAM indique une zone
  d'attention du reseau, pas une localisation anatomique fiable (pas de
  "lobe inferieur droit", pas de mesure de taille, etc.).
- Le texte genere ici est un gabarit (template) fixe par classe : seules les
  valeurs numeriques (confiance, probabilites) varient d'une image a
  l'autre. Ce n'est pas un LLM et il ne doit jamais affirmer une observation
  que le modele n'a pas reellement produite.
- Le disclaimer fait partie integrante du rapport (texte, PDF et JSON) et ne
  doit jamais etre retire ou rendu moins visible.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable,
)


# ==============================================================
# GABARITS PAR CLASSE
# ==============================================================
# Texte generique associe a chaque classe telle que reconnue par le CNN.
# Volontairement non specifique a une localisation anatomique : le modele
# ne produit pas ce niveau de detail, donc le rapport ne doit pas l'inventer.
CLASS_REPORT_TEMPLATES = {
    "NORMAL": {
        "label": "Normal",
        "findings": (
            "Le modele ne detecte pas de pattern d'opacite associe aux trois autres classes "
            "sur cette radiographie. Aucune anomalie n'est signalee par le classifieur."
        ),
        "impression": "Aucune anomalie cardio-pulmonaire detectee par le modele.",
        "recommendation": (
            "Aucune action complementaire suggeree par le modele. En cas de symptomes cliniques "
            "persistants, un avis medical reste recommande independamment de ce resultat."
        ),
    },
    "Lung_Opacity": {
        "label": "Opacite pulmonaire",
        "findings": (
            "Le modele associe cette image a un pattern d'opacite pulmonaire, categorie non "
            "specifique pouvant correspondre a plusieurs origines (infectieuse, inflammatoire, ou "
            "autre). Aucune localisation anatomique precise n'est determinee par le systeme."
        ),
        "impression": "Pattern compatible avec une opacite pulmonaire (classe non specifique).",
        "recommendation": (
            "Correlation clinique et relecture par un radiologue recommandees pour preciser "
            "l'origine de l'opacite."
        ),
    },
    "COVID": {
        "label": "COVID-19",
        "findings": (
            "Le modele associe cette image au pattern qu'il a appris a reconnaitre parmi les cas "
            "etiquetes COVID-19 du jeu de donnees d'entrainement. Ceci ne constitue pas une "
            "confirmation virologique."
        ),
        "impression": (
            "Pattern compatible avec une pneumopathie de type COVID-19, selon la classification "
            "d'image uniquement."
        ),
        "recommendation": (
            "Un test biologique (RT-PCR ou antigenique) reste necessaire pour confirmer ou "
            "infirmer une infection a SARS-CoV-2. Correlation clinique et relecture par un "
            "radiologue recommandees."
        ),
    },
    "Viral Pneumonia": {
        "label": "Pneumonie virale",
        "findings": (
            "Le modele associe cette image au pattern qu'il a appris a reconnaitre parmi les cas "
            "etiquetes pneumonie virale du jeu de donnees d'entrainement."
        ),
        "impression": "Pattern compatible avec une pneumonie virale, selon la classification d'image uniquement.",
        "recommendation": (
            "Correlation clinique et relecture par un radiologue recommandees avant toute decision "
            "therapeutique."
        ),
    },
}

DISCLAIMER = (
    "Ce rapport est genere automatiquement par un prototype academique de classification "
    "d'images. Il ne constitue pas un diagnostic medical, n'a pas ete valide cliniquement et ne "
    "doit en aucun cas remplacer l'avis d'un radiologue ou d'un medecin qualifie."
)

MODEL_NAME = "CNN 4 classes (simple_cnn_model.h5)"


def _template_for(predicted_class: str) -> Dict[str, str]:
    return CLASS_REPORT_TEMPLATES.get(
        predicted_class,
        {
            "label": predicted_class,
            "findings": "Aucun gabarit disponible pour cette classe.",
            "impression": "Non disponible.",
            "recommendation": "Correlation clinique recommandee.",
        },
    )


@dataclass
class RadiologyReport:
    image_name: str
    predicted_class: str
    confidence: float                       # 0-100
    class_probabilities: Dict[str, float]    # 0-100 par classe
    model_name: str = MODEL_NAME
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        t = _template_for(self.predicted_class)
        return {
            "image_name": self.image_name,
            "model": self.model_name,
            "prediction": t["label"],
            "prediction_key": self.predicted_class,
            "confidence": round(self.confidence, 2),
            "class_probabilities": {k: round(v, 2) for k, v in self.class_probabilities.items()},
            "findings": t["findings"],
            "impression": t["impression"],
            "recommendation": t["recommendation"],
            "disclaimer": DISCLAIMER,
            "generated_at": self.generated_at,
        }

    def to_text(self) -> str:
        t = _template_for(self.predicted_class)
        sorted_probs = sorted(self.class_probabilities.items(), key=lambda x: -x[1])
        probs_lines = "\n".join(f"  {k:<18} {v:5.1f}%" for k, v in sorted_probs)
        return f"""RAPPORT IA - RADIOLOGIE
========================

Image analysee : {self.image_name}
Modele : {self.model_name}
Genere le : {self.generated_at}

RESULTATS
---------
Classe predite : {t['label']}
Confiance : {self.confidence:.1f}%

Probabilites par classe :
{probs_lines}

OBSERVATIONS (gabarit associe a la classe predite, pas une lecture radiologique)
---------------------------------------------------------------------------
{t['findings']}

IMPRESSION
----------
{t['impression']}

RECOMMANDATION
--------------
{t['recommendation']}

AVERTISSEMENT
-------------
{DISCLAIMER}
"""


def build_report(image_name: str, predicted_class: str, confidence: float,
                  class_probabilities: Dict[str, float]) -> RadiologyReport:
    return RadiologyReport(
        image_name=image_name,
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
    )


# ==============================================================
# EXPORT PDF
# ==============================================================
def _np_to_image_reader(img_rgb: np.ndarray, max_width_px: int = 700) -> ImageReader:
    """Convert a numpy RGB array to a reportlab ImageReader, downscaled for a lighter PDF."""
    pil_img = PILImage.fromarray(img_rgb.astype(np.uint8))
    if pil_img.width > max_width_px:
        ratio = max_width_px / pil_img.width
        pil_img = pil_img.resize((max_width_px, int(pil_img.height * ratio)))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def generate_pdf_report(report: RadiologyReport, original_img_rgb: np.ndarray,
                         overlay_img_rgb: np.ndarray, heatmap_img_rgb: np.ndarray = None,
                         class_color: str = "#22B8A6") -> bytes:
    """Build a one-page PDF: original image, Grad-CAM heatmap, overlay, structured report, disclaimer."""
    t = _template_for(report.predicted_class)
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=17, spaceAfter=2)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], textColor=colors.HexColor("#5B6472"),
                                     fontSize=9.5, spaceAfter=10)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=11.5,
                                    textColor=colors.HexColor("#111318"), spaceBefore=12, spaceAfter=4)
    body_style = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=9.7, leading=14)
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8.3,
                                       textColor=colors.HexColor("#8A6A1F"), leading=12)

    story = []
    story.append(Paragraph("AI Radiology Analyzer &mdash; Rapport genere par IA", title_style))
    story.append(Paragraph(f"Genere le {report.generated_at} &middot; {report.model_name}", subtitle_style))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#E5E7EB"), thickness=0.7))

    # Prediction summary table
    story.append(Paragraph("Resultat de la classification", section_style))
    summary_data = [
        ["Image analysee", report.image_name],
        ["Classe predite", t["label"]],
        ["Confiance", f"{report.confidence:.1f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[55 * mm, 110 * mm])
    summary_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5B6472")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#111318")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#E5E7EB")),
    ]))
    story.append(summary_table)

    # Class probabilities
    story.append(Paragraph("Probabilites par classe", section_style))
    sorted_probs = sorted(report.class_probabilities.items(), key=lambda x: -x[1])
    prob_data = [[k, f"{v:.1f}%"] for k, v in sorted_probs]
    prob_table = Table(prob_data, colWidths=[80 * mm, 30 * mm])
    prob_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.3),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#333A45")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#EFF1F4")),
    ]))
    story.append(prob_table)

    # Images side by side
    story.append(Paragraph("Radiographie, carte Grad-CAM et fusion", section_style))
    try:
        img_w = 56 * mm if heatmap_img_rgb is not None else 78 * mm
        img1 = RLImage(_np_to_image_reader(original_img_rgb), width=img_w, height=img_w, kind="proportional")
        img3 = RLImage(_np_to_image_reader(overlay_img_rgb), width=img_w, height=img_w, kind="proportional")
        if heatmap_img_rgb is not None:
            img2 = RLImage(_np_to_image_reader(heatmap_img_rgb), width=img_w, height=img_w, kind="proportional")
            img_table = Table(
                [[img1, img2, img3], ["Original", "Grad-CAM", "Fusion (overlay)"]],
                colWidths=[58 * mm, 58 * mm, 58 * mm],
            )
        else:
            img_table = Table(
                [[img1, img3], ["Image originale", "Carte Grad-CAM (zones d'attention du modele)"]],
                colWidths=[85 * mm, 85 * mm],
            )
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, 1), 8),
            ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#5B6472")),
            ("TOPPADDING", (0, 1), (-1, 1), 3),
        ]))
        story.append(img_table)
    except Exception:
        story.append(Paragraph("(Images non disponibles pour ce rapport.)", body_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>La heatmap Grad-CAM met en evidence les regions ayant le plus influence la prediction. "
        "Elle ne constitue pas une segmentation de lesion ni une localisation clinique exacte.</i>",
        ParagraphStyle("GradcamNote", parent=body_style, fontSize=8.3, textColor=colors.HexColor("#5B6472")),
    ))

    # Findings / impression / recommendation
    story.append(Paragraph("Observations (gabarit associe a la classe predite)", section_style))
    story.append(Paragraph(t["findings"], body_style))
    story.append(Paragraph("Impression", section_style))
    story.append(Paragraph(t["impression"], body_style))
    story.append(Paragraph("Recommandation", section_style))
    story.append(Paragraph(t["recommendation"], body_style))

    # Disclaimer
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#E5E7EB"), thickness=0.7))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Avertissement.</b> {DISCLAIMER}", disclaimer_style))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
