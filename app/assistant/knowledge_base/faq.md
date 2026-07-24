# Questions fréquentes

**Cette application peut-elle diagnostiquer une maladie ?**
Non. C'est un prototype académique de classification d'images. Il ne
remplace jamais l'avis d'un radiologue ou d'un médecin, et ne doit jamais
être utilisé pour prendre une décision médicale.

**Pourquoi le modèle se trompe-t-il parfois ?**
Le modèle a une exactitude globale de 85.9 % sur son jeu de test — il se
trompe donc dans une proportion significative de cas. Le rappel est en
particulier plus faible sur la classe COVID-19 (77.5 %). Voir le document
sur les limites du modèle.

**Le modèle peut-il détecter une tuberculose, un cancer, un pneumothorax ?**
Non. Le modèle ne reconnaît que quatre classes : COVID-19, opacité
pulmonaire, pneumonie virale, et normal. Toute autre pathologie est hors de
son périmètre, quelle que soit l'image fournie.

**Qu'est-ce que la carte Grad-CAM montre exactement ?**
Une zone d'attention du réseau de neurones, pas une localisation clinique
précise ni une segmentation de lésion. Voir le document dédié à Grad-CAM.

**Mes données (images, historique) sont-elles envoyées quelque part ?**
Les analyses sont traitées et stockées localement (base SQLite locale). Si
l'application est déployée avec l'assistant IA basé sur Ollama, les
échanges avec l'assistant sont également traités localement (aucune API
cloud payante par défaut).

**Puis-je faire confiance à 100 % à une prédiction ?**
Non. Aucun score de confiance affiché par cette application ne doit être
interprété comme une certitude médicale. Voir le document sur le score de
confiance.
