"""
safety.py

Garde-fou execute AVANT tout appel au LLM : detecte si la question demande
un diagnostic, un traitement, une prescription, une hospitalisation ou un
avis medical personnel, et refuse poliment sans jamais atteindre le modele
de langage.

C'est une defense en profondeur : le prompt systeme (voir prompts.py)
demande deja au LLM de refuser ce type de question, mais un modele local
peut se tromper ou halluciner une reponse malgre l'instruction. Ce filtre
lexical, execute cote application, garantit le refus independamment du
comportement du LLM.

Volontairement un filtre a base de motifs (regex), pas un modele : rapide,
deterministe, testable sans dependance externe, et le taux de faux positifs
(refuser une question legitime a tort) est prefere au risque inverse
(laisser passer une demande de diagnostic).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Motifs signalant une demande de diagnostic/traitement/avis medical personnel.
# Volontairement larges : mieux vaut un refus a tort qu'un conseil medical a tort.
_MEDICAL_ADVICE_PATTERNS = {
    "fr": [
        r"\b(suis[- ]je|ai[- ]je|est[- ]ce que j'ai)\b.{0,30}\b(covid|pneumonie|malade|infect|cancer|tuberculose)",
        r"\b(mon|ma|mes)\b.{0,20}\b(radio|scan|analyse|résultat)s?\b.{0,30}\b(veut dire|signifie|montre)",
        r"\bdois[- ]je\b.{0,30}\b(prendre|consulter|aller|traitement|médicament|hôpital|urgence)",
        r"\b(quel|quelle)\b.{0,20}\btraitement\b",
        r"\b(prescri|ordonnance|posologie|dosage)\w*",
        r"\bdiagnosti\w*.{0,20}\b(moi|ma|mon|patient)",
        r"\best[- ]ce (grave|dangereux|sérieux)\b",
        r"\bdois[- ]je (m'inquiéter|paniquer)\b",
    ],
    "en": [
        r"\b(do i have|am i sick|could i have)\b.{0,30}\b(covid|pneumonia|cancer|tuberculosis|infection)",
        r"\bwhat (treatment|medication|prescription|dosage)\b",
        r"\bshould i\b.{0,30}\b(take|see a doctor|go to|the hospital|the er|worry)",
        r"\bdiagnos\w*.{0,20}\b(me|my|patient)",
        r"\bis (it|this) (serious|dangerous|life[- ]threatening)\b",
        r"\bprescri\w*",
    ],
    "ar": [
        r"هل (عندي|أعاني من|مصاب ب)",
        r"ما (هو )?العلاج",
        r"هل يجب أن (آخذ|أذهب إلى الطبيب|أذهب إلى المستشفى)",
        r"شخّص لي",
        r"هل هذا خطير",
        r"وصفة طبية",
    ],
}


@dataclass
class SafetyCheckResult:
    is_medical_request: bool
    matched_pattern: str = ""


def check_medical_advice_request(message: str, language: str = "fr") -> SafetyCheckResult:
    """
    Verifie si `message` demande un diagnostic/traitement/avis medical
    personnel. Teste les motifs de la langue demandee ET du francais/anglais
    en repli (un utilisateur peut ecrire en francais alors que l'interface
    est en arabe, par exemple).
    """
    if not message or not message.strip():
        return SafetyCheckResult(is_medical_request=False)

    text = message.lower()
    languages_to_check = [language] + [lg for lg in ("fr", "en") if lg != language]

    for lang in languages_to_check:
        for pattern in _MEDICAL_ADVICE_PATTERNS.get(lang, []):
            if re.search(pattern, text, flags=re.IGNORECASE):
                return SafetyCheckResult(is_medical_request=True, matched_pattern=pattern)

    return SafetyCheckResult(is_medical_request=False)
