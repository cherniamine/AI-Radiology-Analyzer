# Classes reconnues par le modèle

Le modèle classe chaque radiographie dans l'une de ces quatre classes,
apprises sur un jeu de données public de radiographies thoraciques
(42 330 images au total) :

- **COVID-19** : 7 232 images d'entraînement. Précision 85.6 %, rappel 77.5 %,
  F1 81.3 % sur le jeu de test (1 446 images de test).
- **Opacité pulmonaire** (Lung_Opacity) : 12 024 images d'entraînement.
  Précision 80.3 %, rappel 86.4 %, F1 83.2 % (2 404 images de test).
- **Normal** : 20 384 images d'entraînement. Précision 89.0 %, rappel 88.1 %,
  F1 88.6 % (4 076 images de test).
- **Pneumonie virale** (Viral Pneumonia) : 2 690 images d'entraînement.
  Précision 90.0 %, rappel 88.5 %, F1 89.2 % (538 images de test).

Exactitude globale du modèle sur le jeu de test : 85.9 %.

Le modèle NE reconnaît QUE ces quatre classes. Il ne peut pas détecter une
tuberculose, un cancer du poumon, un pneumothorax, ou toute autre pathologie
absente de cette liste, quelle que soit l'image fournie.

Le jeu de données est déséquilibré (la classe Normal représente près de la
moitié des images, la Pneumonie virale moins de 7 %), ce qui explique en
partie l'écart de performance entre classes — en particulier le rappel plus
faible sur COVID-19 (77.5 %), le point le plus sensible du modèle dans un
contexte clinique.
