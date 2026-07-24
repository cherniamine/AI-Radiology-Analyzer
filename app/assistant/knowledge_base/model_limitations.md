# Limites du modèle

- Le modèle est un CNN simple (3 blocs Conv2D + MaxPooling, puis Flatten,
  Dropout, et deux couches Dense), entraîné from scratch sur des images
  redimensionnées en 128×128. Ce n'est pas un modèle de imagerie médicale
  de qualité clinique.

- Le modèle classe l'IMAGE ENTIÈRE. Il ne localise aucune anomalie de façon
  fiable : Grad-CAM montre une zone d'attention du réseau, pas une
  localisation anatomique validée (pas de lobe précis, pas de mesure de
  taille de lésion).

- Le jeu de données d'entraînement est un jeu de données PUBLIC, non annoté
  par un radiologue dans le cadre de ce projet. Aucune validation clinique
  n'a été réalisée.

- Le jeu de données est déséquilibré entre les quatre classes, ce qui affecte
  la performance de façon inégale (voir le fichier sur les classes
  reconnues pour les chiffres exacts par classe).

- Le modèle n'a jamais été testé sur des données provenant d'un autre
  hôpital, d'un autre pays, ou d'un autre type d'appareil de radiographie
  (biais de distribution possible, non mesuré).

- Il existe une architecture alternative (EfficientNetB0, transfer
  learning) définie dans le code source du projet mais elle n'a jamais été
  entraînée ni évaluée — le modèle réellement utilisé par l'application est
  le CNN simple décrit ci-dessus.

- Ce projet est un prototype de recherche / académique. Il ne constitue
  JAMAIS un diagnostic médical et ne remplace jamais l'avis d'un
  radiologue ou d'un médecin.
