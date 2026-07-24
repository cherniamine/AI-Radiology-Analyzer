# Grad-CAM : explicabilité du modèle

Grad-CAM (Gradient-weighted Class Activation Mapping) est une technique qui
génère une carte de chaleur ("heatmap") montrant quelles régions d'une
image ont le plus influencé la prédiction du réseau de neurones.

Dans cette application, Grad-CAM est calculé sur la dernière couche
convolutive du modèle (conv2d_2). Pour chaque radiographie analysée,
l'interface affiche trois vues côte à côte : l'image originale, la heatmap
seule, et la fusion des deux (overlay), avec une intensité de superposition
réglable et un choix de palette de couleurs (JET, HOT, PLASMA, VIRIDIS,
INFERNO).

Points importants sur ce que Grad-CAM NE fait PAS :
- Ce n'est PAS une segmentation de lésion : la heatmap ne délimite pas les
  contours précis d'une anomalie.
- Ce n'est PAS une localisation anatomique validée : on ne peut pas dire
  "l'anomalie est dans le lobe inférieur droit" à partir de la heatmap
  seule.
- Une zone chaude dans la heatmap signifie seulement que cette région a eu
  du poids dans le calcul du réseau pour la classe prédite — ce n'est pas
  une preuve clinique d'anomalie à cet endroit.

Grad-CAM sert à la transparence et au débogage du modèle, pas à guider une
décision médicale.
