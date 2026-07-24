# Le score de confiance

Pour chaque radiographie analysée, le modèle produit une probabilité pour
chacune des quatre classes (COVID-19, Opacité pulmonaire, Normal, Pneumonie
virale). Ces probabilités somment à 100 %. La "confiance" affichée est la
probabilité de la classe prédite (celle avec le score le plus élevé).

Ce score de confiance est une sortie statistique du réseau de neurones — il
ne mesure pas une certitude médicale. Un score élevé (par exemple 95 %)
signifie que le modèle est très cohérent avec les patterns qu'il a appris
sur son jeu d'entraînement pour cette classe, pas que le diagnostic est
correct à 95 %.

Codes couleur utilisés dans l'application pour la confiance :
- Vert : confiance supérieure à 80 %.
- Orange : confiance entre 60 % et 80 %.
- Rouge : confiance inférieure à 60 %.

Une confiance faible ou moyenne peut indiquer une image ambiguë, un cas
atypique par rapport au jeu d'entraînement, ou les limites générales du
modèle (voir le document sur les limites du modèle).
