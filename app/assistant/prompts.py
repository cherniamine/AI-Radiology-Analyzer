"""
prompts.py

Gabarits de prompts pour l'assistant. Aucune dependance a Ollama ici :
ce module ne fait que construire des chaines de caracteres, ce qui le rend
testable independamment de la disponibilite d'un serveur LLM.
"""

from __future__ import annotations

from typing import List

from .retriever import RetrievedChunk

_LANGUAGE_NAMES = {"fr": "français", "en": "English", "ar": "العربية"}


def build_system_prompt(language: str = "fr") -> str:
    """
    Prompt systeme : cadre le role de l'assistant, ses limites strictes, et
    la langue de reponse attendue. Le refus medical est aussi impose ici en
    defense en profondeur, en plus du filtre programmatique (safety.py) qui
    s'execute avant meme d'atteindre le LLM.
    """
    language_name = _LANGUAGE_NAMES.get(language, "français")
    return (
        "Tu es l'assistant documentaire d'AI Radiology Analyzer, un prototype "
        "académique de classification de radiographies pulmonaires par IA. "
        "Ton rôle est d'expliquer le projet, son modèle CNN, Grad-CAM, les "
        "scores de confiance, les limites du modèle, l'utilisation de "
        "l'application, Docker, et l'architecture — en te basant UNIQUEMENT "
        "sur le contexte fourni ci-dessous, extrait de la documentation du "
        "projet. Si le contexte ne contient pas la réponse, dis-le "
        "clairement plutôt que d'inventer une information.\n\n"
        "RÈGLE ABSOLUE : tu ne dois JAMAIS fournir de diagnostic médical, de "
        "traitement, de prescription, de conseil d'hospitalisation, ni "
        "aucun avis médical personnel — même si on te le demande "
        "explicitement ou de façon détournée. Si une question relève du "
        "médical personnel, refuse poliment et explique que cette "
        "application est un outil éducatif/de recherche qui ne remplace "
        "pas un professionnel de santé.\n\n"
        f"Réponds toujours en {language_name}, quelle que soit la langue de "
        "la question, de façon concise et factuelle."
    )


def build_user_prompt(question: str, retrieved: List[RetrievedChunk]) -> str:
    """Construit le prompt utilisateur final : question + contexte recupere par le retriever."""
    if not retrieved:
        context_block = "(Aucun passage pertinent trouvé dans la documentation du projet.)"
    else:
        context_block = "\n\n".join(
            f"[Source : {r.chunk.source} — {r.chunk.heading}]\n{r.chunk.text}"
            for r in retrieved
        )
    return (
        f"Contexte extrait de la documentation du projet :\n\n{context_block}\n\n"
        f"---\n\nQuestion de l'utilisateur : {question}"
    )


REFUSAL_MESSAGES = {
    "fr": (
        "Je ne peux pas répondre à cette question : elle relève d'un "
        "diagnostic, d'un traitement ou d'un avis médical personnel. "
        "AI Radiology Analyzer est un outil éducatif/de recherche — il ne "
        "remplace pas un professionnel de santé. Si vous avez une "
        "préoccupation médicale, merci de consulter un médecin ou un "
        "radiologue."
    ),
    "en": (
        "I can't answer that — it involves a diagnosis, treatment, or "
        "personal medical advice. AI Radiology Analyzer is an educational/"
        "research tool and does not replace a healthcare professional. If "
        "you have a medical concern, please consult a doctor or "
        "radiologist."
    ),
    "ar": (
        "لا يمكنني الإجابة عن هذا السؤال لأنه يتعلق بتشخيص أو علاج أو "
        "استشارة طبية شخصية. AI Radiology Analyzer أداة تعليمية/بحثية ولا "
        "تغني عن استشارة أخصائي رعاية صحية. إذا كان لديك قلق طبي، يُرجى "
        "استشارة طبيب أو أخصائي أشعة."
    ),
}

SUGGESTED_QUESTIONS = {
    "fr": [
        "Comment fonctionne Grad-CAM ?",
        "Quelles maladies le modèle reconnaît-il ?",
        "Que signifie le score de confiance ?",
        "Quelles sont les limites du modèle ?",
        "Comment lancer l'application avec Docker ?",
    ],
    "en": [
        "How does Grad-CAM work?",
        "Which diseases does the model recognize?",
        "What does the confidence score mean?",
        "What are the model's limitations?",
        "How do I run the app with Docker?",
    ],
    "ar": [
        "كيف تعمل تقنية Grad-CAM؟",
        "ما هي الأمراض التي يتعرف عليها النموذج؟",
        "ماذا يعني درجة الثقة؟",
        "ما هي حدود النموذج؟",
        "كيف أشغّل التطبيق باستخدام Docker؟",
    ],
}

OLLAMA_UNAVAILABLE_MESSAGES = {
    "fr": (
        "⚠️ Impossible de contacter le serveur Ollama local ({base_url}). "
        "Vérifiez qu'Ollama est installé et lancé (`ollama serve`), et que "
        "le modèle `{model}` est bien téléchargé (`ollama pull {model}`)."
    ),
    "en": (
        "⚠️ Could not reach the local Ollama server ({base_url}). Check that "
        "Ollama is installed and running (`ollama serve`), and that the "
        "`{model}` model has been pulled (`ollama pull {model}`)."
    ),
    "ar": (
        "⚠️ تعذّر الوصول إلى خادم Ollama المحلي ({base_url}). تحقق من أن "
        "Ollama مثبّت وقيد التشغيل (`ollama serve`)، وأن نموذج `{model}` "
        "تم تنزيله (`ollama pull {model}`)."
    ),
}
