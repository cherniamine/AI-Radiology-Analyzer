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

Multilingue (fr/en/ar) :

- Tous les gabarits cliniques (CLASS_REPORT_TEMPLATES_BY_LANG), le
  disclaimer (DISCLAIMER_BY_LANG) et les libelles fixes du PDF/texte
  (REPORT_STRINGS) existent dans les 3 langues de l'application.
- Le JSON (to_dict) est genere fidelement dans la langue demandee, y
  compris l'arabe : ce n'est que du texte Unicode, aucune contrainte de
  rendu.
- Le PDF (generate_pdf_report), lui, utilise ReportLab avec la police
  standard Helvetica, qui ne contient AUCUN glyphe arabe et ne fait pas de
  reshaping/bidi. Generer un PDF "arabe" avec cette police produirait des
  cases vides illisibles plutot qu'un vrai rendu degrade. Plutot que de
  livrer silencieusement un PDF casse, generate_pdf_report retombe sur
  l'anglais pour la mise en page PDF quand la langue demandee est "ar", et
  l'indique explicitement dans le PDF genere. Corriger cela proprement
  necessiterait d'embarquer une police arabe (ex. Noto Sans Arabic) et les
  bibliotheques arabic-reshaper/python-bidi — hors perimetre de cette
  passe. Voir le README pour ce point.
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


SUPPORTED_REPORT_LANGUAGES = ("fr", "en", "ar")
DEFAULT_REPORT_LANGUAGE = "fr"
# Langue de repli pour la mise en page PDF uniquement (voir docstring du
# module) : ReportLab/Helvetica ne peut pas afficher l'arabe correctement.
_PDF_FALLBACK_LANGUAGE = "en"
_PDF_SUPPORTED_LANGUAGES = ("fr", "en")


# ==============================================================
# GABARITS PAR CLASSE (fr / en / ar)
# ==============================================================
# Texte generique associe a chaque classe telle que reconnue par le CNN.
# Volontairement non specifique a une localisation anatomique : le modele
# ne produit pas ce niveau de detail, donc le rapport ne doit pas l'inventer.
CLASS_REPORT_TEMPLATES_BY_LANG = {
    "fr": {
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
    },
    "en": {
        "NORMAL": {
            "label": "Normal",
            "findings": (
                "The model does not detect an opacity pattern associated with the three other classes "
                "on this radiograph. No abnormality is reported by the classifier."
            ),
            "impression": "No cardiopulmonary abnormality detected by the model.",
            "recommendation": (
                "No further action suggested by the model. If clinical symptoms persist, medical "
                "advice remains recommended regardless of this result."
            ),
        },
        "Lung_Opacity": {
            "label": "Lung Opacity",
            "findings": (
                "The model associates this image with a lung opacity pattern, a non-specific category "
                "that may correspond to several origins (infectious, inflammatory, or other). No precise "
                "anatomical localization is determined by the system."
            ),
            "impression": "Pattern consistent with lung opacity (non-specific class).",
            "recommendation": (
                "Clinical correlation and review by a radiologist are recommended to clarify the origin "
                "of the opacity."
            ),
        },
        "COVID": {
            "label": "COVID-19",
            "findings": (
                "The model associates this image with the pattern it learned to recognize among cases "
                "labeled COVID-19 in the training dataset. This does not constitute virological "
                "confirmation."
            ),
            "impression": "Pattern consistent with COVID-19-type pneumonia, based on image classification only.",
            "recommendation": (
                "A biological test (RT-PCR or antigen) remains necessary to confirm or rule out "
                "SARS-CoV-2 infection. Clinical correlation and review by a radiologist are recommended."
            ),
        },
        "Viral Pneumonia": {
            "label": "Viral Pneumonia",
            "findings": (
                "The model associates this image with the pattern it learned to recognize among cases "
                "labeled viral pneumonia in the training dataset."
            ),
            "impression": "Pattern consistent with viral pneumonia, based on image classification only.",
            "recommendation": (
                "Clinical correlation and review by a radiologist are recommended before any "
                "therapeutic decision."
            ),
        },
    },
    "ar": {
        "NORMAL": {
            "label": "طبيعي",
            "findings": (
                "لا يكتشف النموذج أي نمط عتامة مرتبط بالفئات الثلاث الأخرى في هذه الصورة الشعاعية. "
                "لم يُبلّغ المصنّف عن أي شذوذ."
            ),
            "impression": "لم يكتشف النموذج أي شذوذ قلبي رئوي.",
            "recommendation": (
                "لا يقترح النموذج أي إجراء إضافي. في حال استمرار الأعراض السريرية، يُنصح باستشارة "
                "طبية بغض النظر عن هذه النتيجة."
            ),
        },
        "Lung_Opacity": {
            "label": "عتامة رئوية",
            "findings": (
                "يربط النموذج هذه الصورة بنمط عتامة رئوية، وهي فئة غير نوعية قد تعود لعدة أسباب "
                "(معدية أو التهابية أو غيرها). لا يحدد النظام أي موقع تشريحي دقيق."
            ),
            "impression": "نمط متوافق مع عتامة رئوية (فئة غير نوعية).",
            "recommendation": "يُنصح بالمقارنة السريرية ومراجعة طبيب أشعة لتحديد سبب العتامة.",
        },
        "COVID": {
            "label": "كوفيد-19",
            "findings": (
                "يربط النموذج هذه الصورة بالنمط الذي تعلّم التعرف عليه من الحالات المصنّفة كوفيد-19 "
                "في بيانات التدريب. لا يشكّل هذا تأكيدًا فيروسيًا."
            ),
            "impression": "نمط متوافق مع التهاب رئوي من نوع كوفيد-19، استنادًا إلى تصنيف الصورة فقط.",
            "recommendation": (
                "يبقى إجراء اختبار بيولوجي (RT-PCR أو مستضد) ضروريًا لتأكيد أو استبعاد الإصابة "
                "بفيروس SARS-CoV-2. يُنصح بالمقارنة السريرية ومراجعة طبيب أشعة."
            ),
        },
        "Viral Pneumonia": {
            "label": "التهاب رئوي فيروسي",
            "findings": (
                "يربط النموذج هذه الصورة بالنمط الذي تعلّم التعرف عليه من الحالات المصنّفة التهابًا "
                "رئويًا فيروسيًا في بيانات التدريب."
            ),
            "impression": "نمط متوافق مع التهاب رئوي فيروسي، استنادًا إلى تصنيف الصورة فقط.",
            "recommendation": "يُنصح بالمقارنة السريرية ومراجعة طبيب أشعة قبل اتخاذ أي قرار علاجي.",
        },
    },
}

# Alias retro-compatible : du code (ou des tests) important
# CLASS_REPORT_TEMPLATES directement continue de fonctionner, avec les
# gabarits francais (comportement identique a avant l'ajout du multilingue).
CLASS_REPORT_TEMPLATES = CLASS_REPORT_TEMPLATES_BY_LANG[DEFAULT_REPORT_LANGUAGE]

DISCLAIMER_BY_LANG = {
    "fr": (
        "Ce rapport est genere automatiquement par un prototype academique de classification "
        "d'images. Il ne constitue pas un diagnostic medical, n'a pas ete valide cliniquement et ne "
        "doit en aucun cas remplacer l'avis d'un radiologue ou d'un medecin qualifie."
    ),
    "en": (
        "This report is automatically generated by an academic image-classification prototype. It "
        "does not constitute a medical diagnosis, has not been clinically validated, and must never "
        "replace the opinion of a qualified radiologist or physician."
    ),
    "ar": (
        "يُنشأ هذا التقرير تلقائيًا بواسطة نموذج أكاديمي أولي لتصنيف الصور. لا يشكّل تشخيصًا طبيًا، "
        "ولم يخضع للتحقق السريري، ولا يجب أن يحل بأي حال من الأحوال محل رأي طبيب أشعة أو طبيب مؤهل."
    ),
}
# Alias retro-compatible (voir CLASS_REPORT_TEMPLATES ci-dessus).
DISCLAIMER = DISCLAIMER_BY_LANG[DEFAULT_REPORT_LANGUAGE]

MODEL_NAME = "CNN 4 classes (simple_cnn_model.h5)"

# Libelles fixes (pas de contenu clinique) : titres de section, en-tetes du
# PDF et du format texte, legendes d'images.
REPORT_STRINGS = {
    "fr": {
        "title": "AI Radiology Analyzer &mdash; Rapport genere par IA",
        "generated_on": "Genere le {date} &middot; {model}",
        "section_result": "Resultat de la classification",
        "row_image": "Image analysee",
        "row_class": "Classe predite",
        "row_confidence": "Confiance",
        "section_probabilities": "Probabilites par classe",
        "section_images": "Radiographie, carte Grad-CAM et fusion",
        "caption_original": "Original",
        "caption_gradcam": "Grad-CAM",
        "caption_overlay": "Fusion (overlay)",
        "caption_original_only": "Image originale",
        "caption_overlay_only": "Carte Grad-CAM (zones d'attention du modele)",
        "gradcam_note": (
            "La heatmap Grad-CAM met en evidence les regions ayant le plus influence la prediction. "
            "Elle ne constitue pas une segmentation de lesion ni une localisation clinique exacte."
        ),
        "images_unavailable": "(Images non disponibles pour ce rapport.)",
        "section_findings": "Observations (gabarit associe a la classe predite)",
        "section_impression": "Impression",
        "section_recommendation": "Recommandation",
        "disclaimer_label": "Avertissement.",
        "unknown_class_findings": "Aucun gabarit disponible pour cette classe.",
        "unknown_class_impression": "Non disponible.",
        "unknown_class_recommendation": "Correlation clinique recommandee.",
        "text_title": "RAPPORT IA - RADIOLOGIE",
        "text_image": "Image analysee",
        "text_model": "Modele",
        "text_generated": "Genere le",
        "text_results": "RESULTATS",
        "text_class": "Classe predite",
        "text_confidence": "Confiance",
        "text_probs_header": "Probabilites par classe :",
        "text_observations": "OBSERVATIONS (gabarit associe a la classe predite, pas une lecture radiologique)",
        "text_impression": "IMPRESSION",
        "text_recommendation": "RECOMMANDATION",
        "text_disclaimer": "AVERTISSEMENT",
    },
    "en": {
        "title": "AI Radiology Analyzer &mdash; AI-Generated Report",
        "generated_on": "Generated on {date} &middot; {model}",
        "section_result": "Classification Result",
        "row_image": "Analyzed image",
        "row_class": "Predicted class",
        "row_confidence": "Confidence",
        "section_probabilities": "Class Probabilities",
        "section_images": "Radiograph, Grad-CAM Map and Overlay",
        "caption_original": "Original",
        "caption_gradcam": "Grad-CAM",
        "caption_overlay": "Overlay (fusion)",
        "caption_original_only": "Original image",
        "caption_overlay_only": "Grad-CAM map (model attention areas)",
        "gradcam_note": (
            "The Grad-CAM heatmap highlights the regions that most influenced the prediction. It does "
            "not constitute a lesion segmentation or an exact clinical localization."
        ),
        "images_unavailable": "(Images not available for this report.)",
        "section_findings": "Findings (template associated with the predicted class)",
        "section_impression": "Impression",
        "section_recommendation": "Recommendation",
        "disclaimer_label": "Disclaimer.",
        "unknown_class_findings": "No template available for this class.",
        "unknown_class_impression": "Not available.",
        "unknown_class_recommendation": "Clinical correlation recommended.",
        "text_title": "AI RADIOLOGY REPORT",
        "text_image": "Analyzed image",
        "text_model": "Model",
        "text_generated": "Generated on",
        "text_results": "RESULTS",
        "text_class": "Predicted class",
        "text_confidence": "Confidence",
        "text_probs_header": "Class probabilities:",
        "text_observations": "FINDINGS (template associated with the predicted class, not a radiological reading)",
        "text_impression": "IMPRESSION",
        "text_recommendation": "RECOMMENDATION",
        "text_disclaimer": "DISCLAIMER",
    },
    "ar": {
        "title": "محلل الأشعة بالذكاء الاصطناعي &mdash; تقرير مُنشأ آليًا",
        "generated_on": "أُنشئ في {date} &middot; {model}",
        "section_result": "نتيجة التصنيف",
        "row_image": "الصورة المحللة",
        "row_class": "الفئة المتوقعة",
        "row_confidence": "نسبة الثقة",
        "section_probabilities": "احتمالات كل فئة",
        "section_images": "الصورة الشعاعية وخريطة Grad-CAM والدمج",
        "caption_original": "الأصلية",
        "caption_gradcam": "Grad-CAM",
        "caption_overlay": "الدمج",
        "caption_original_only": "الصورة الأصلية",
        "caption_overlay_only": "خريطة Grad-CAM (مناطق اهتمام النموذج)",
        "gradcam_note": (
            "تُبرز خريطة Grad-CAM الحرارية المناطق الأكثر تأثيرًا في التنبؤ. لا تشكّل تجزئة للآفة "
            "ولا تحديدًا سريريًا دقيقًا للموقع."
        ),
        "images_unavailable": "(الصور غير متوفرة لهذا التقرير.)",
        "section_findings": "الملاحظات (نموذج مرتبط بالفئة المتوقعة)",
        "section_impression": "الانطباع",
        "section_recommendation": "التوصية",
        "disclaimer_label": "تنويه.",
        "unknown_class_findings": "لا يوجد نموذج متاح لهذه الفئة.",
        "unknown_class_impression": "غير متوفر.",
        "unknown_class_recommendation": "يُنصح بالمقارنة السريرية.",
        "text_title": "تقرير الأشعة بالذكاء الاصطناعي",
        "text_image": "الصورة المحللة",
        "text_model": "النموذج",
        "text_generated": "أُنشئ في",
        "text_results": "النتائج",
        "text_class": "الفئة المتوقعة",
        "text_confidence": "نسبة الثقة",
        "text_probs_header": "احتمالات كل فئة:",
        "text_observations": "الملاحظات (نموذج مرتبط بالفئة المتوقعة، وليس قراءة أشعة)",
        "text_impression": "الانطباع",
        "text_recommendation": "التوصية",
        "text_disclaimer": "تنويه",
    },
}


def _templates_for_language(language: str) -> Dict[str, Dict[str, str]]:
    return CLASS_REPORT_TEMPLATES_BY_LANG.get(language, CLASS_REPORT_TEMPLATES_BY_LANG[DEFAULT_REPORT_LANGUAGE])


def _strings_for(language: str) -> Dict[str, str]:
    return REPORT_STRINGS.get(language, REPORT_STRINGS[DEFAULT_REPORT_LANGUAGE])


def _template_for(predicted_class: str, language: str = DEFAULT_REPORT_LANGUAGE) -> Dict[str, str]:
    strings = _strings_for(language)
    return _templates_for_language(language).get(
        predicted_class,
        {
            "label": predicted_class,
            "findings": strings["unknown_class_findings"],
            "impression": strings["unknown_class_impression"],
            "recommendation": strings["unknown_class_recommendation"],
        },
    )


@dataclass
class RadiologyReport:
    image_name: str
    predicted_class: str
    confidence: float                       # 0-100
    class_probabilities: Dict[str, float]    # 0-100 par classe
    model_name: str = MODEL_NAME
    language: str = DEFAULT_REPORT_LANGUAGE
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        t = _template_for(self.predicted_class, self.language)
        return {
            "language": self.language,
            "image_name": self.image_name,
            "model": self.model_name,
            "prediction": t["label"],
            "prediction_key": self.predicted_class,
            "confidence": round(self.confidence, 2),
            "class_probabilities": {k: round(v, 2) for k, v in self.class_probabilities.items()},
            "findings": t["findings"],
            "impression": t["impression"],
            "recommendation": t["recommendation"],
            "disclaimer": DISCLAIMER_BY_LANG.get(self.language, DISCLAIMER_BY_LANG[DEFAULT_REPORT_LANGUAGE]),
            "generated_at": self.generated_at,
        }

    def to_text(self) -> str:
        t = _template_for(self.predicted_class, self.language)
        s = _strings_for(self.language)
        sorted_probs = sorted(self.class_probabilities.items(), key=lambda x: -x[1])
        probs_lines = "\n".join(f"  {k:<18} {v:5.1f}%" for k, v in sorted_probs)
        return f"""{s['text_title']}
{'=' * len(s['text_title'])}

{s['text_image']} : {self.image_name}
{s['text_model']} : {self.model_name}
{s['text_generated']} : {self.generated_at}

{s['text_results']}
{'-' * len(s['text_results'])}
{s['text_class']} : {t['label']}
{s['text_confidence']} : {self.confidence:.1f}%

{s['text_probs_header']}
{probs_lines}

{s['text_observations']}
{'-' * len(s['text_observations'])}
{t['findings']}

{s['text_impression']}
{'-' * len(s['text_impression'])}
{t['impression']}

{s['text_recommendation']}
{'-' * len(s['text_recommendation'])}
{t['recommendation']}

{s['text_disclaimer']}
{'-' * len(s['text_disclaimer'])}
{DISCLAIMER_BY_LANG.get(self.language, DISCLAIMER_BY_LANG[DEFAULT_REPORT_LANGUAGE])}
"""


def build_report(image_name: str, predicted_class: str, confidence: float,
                  class_probabilities: Dict[str, float],
                  language: str = DEFAULT_REPORT_LANGUAGE) -> RadiologyReport:
    return RadiologyReport(
        image_name=image_name,
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
        language=language if language in SUPPORTED_REPORT_LANGUAGES else DEFAULT_REPORT_LANGUAGE,
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
    """Build a one-page PDF: original image, Grad-CAM heatmap, overlay, structured report, disclaimer.

    La mise en page suit `report.language`, sauf pour l'arabe qui retombe
    sur l'anglais (voir docstring du module — Helvetica ne supporte pas les
    glyphes arabes)."""
    pdf_language = report.language if report.language in _PDF_SUPPORTED_LANGUAGES else _PDF_FALLBACK_LANGUAGE
    fell_back = pdf_language != report.language

    t = _template_for(report.predicted_class, pdf_language)
    s = _strings_for(pdf_language)
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
    fallback_note_style = ParagraphStyle("FallbackNote", parent=styles["Normal"], fontSize=7.8,
                                          textColor=colors.HexColor("#8A6A1F"), leading=11, spaceAfter=8)

    story = []
    story.append(Paragraph(s["title"], title_style))
    story.append(Paragraph(
        s["generated_on"].format(date=report.generated_at, model=report.model_name), subtitle_style,
    ))
    if fell_back:
        story.append(Paragraph(
            "[This PDF is shown in English \u2014 Arabic PDF typesetting is not supported yet. "
            "The JSON export of this report is fully available in Arabic.]",
            fallback_note_style,
        ))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#E5E7EB"), thickness=0.7))

    # Prediction summary table
    story.append(Paragraph(s["section_result"], section_style))
    summary_data = [
        [s["row_image"], report.image_name],
        [s["row_class"], t["label"]],
        [s["row_confidence"], f"{report.confidence:.1f}%"],
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
    story.append(Paragraph(s["section_probabilities"], section_style))
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
    story.append(Paragraph(s["section_images"], section_style))
    try:
        img_w = 56 * mm if heatmap_img_rgb is not None else 78 * mm
        img1 = RLImage(_np_to_image_reader(original_img_rgb), width=img_w, height=img_w, kind="proportional")
        img3 = RLImage(_np_to_image_reader(overlay_img_rgb), width=img_w, height=img_w, kind="proportional")
        if heatmap_img_rgb is not None:
            img2 = RLImage(_np_to_image_reader(heatmap_img_rgb), width=img_w, height=img_w, kind="proportional")
            img_table = Table(
                [[img1, img2, img3], [s["caption_original"], s["caption_gradcam"], s["caption_overlay"]]],
                colWidths=[58 * mm, 58 * mm, 58 * mm],
            )
        else:
            img_table = Table(
                [[img1, img3], [s["caption_original_only"], s["caption_overlay_only"]]],
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
        story.append(Paragraph(s["images_unavailable"], body_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"<i>{s['gradcam_note']}</i>",
        ParagraphStyle("GradcamNote", parent=body_style, fontSize=8.3, textColor=colors.HexColor("#5B6472")),
    ))

    # Findings / impression / recommendation
    story.append(Paragraph(s["section_findings"], section_style))
    story.append(Paragraph(t["findings"], body_style))
    story.append(Paragraph(s["section_impression"], section_style))
    story.append(Paragraph(t["impression"], body_style))
    story.append(Paragraph(s["section_recommendation"], section_style))
    story.append(Paragraph(t["recommendation"], body_style))

    # Disclaimer
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#E5E7EB"), thickness=0.7))
    story.append(Spacer(1, 6))
    disclaimer_text = DISCLAIMER_BY_LANG.get(pdf_language, DISCLAIMER_BY_LANG[DEFAULT_REPORT_LANGUAGE])
    story.append(Paragraph(f"<b>{s['disclaimer_label']}</b> {disclaimer_text}", disclaimer_style))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()