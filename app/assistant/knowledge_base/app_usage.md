# Utilisation de l'application

L'application a une navigation multi-pages dans la barre latérale :

- **Nouvelle analyse** : déposez une ou plusieurs radiographies (PNG/JPG).
  Pour chaque image, l'application affiche la classe prédite, le score de
  confiance par classe, la carte Grad-CAM (original / heatmap / fusion, avec
  intensité réglable), un rapport structuré (observations, impression,
  recommandation), et permet d'exporter en CSV, PDF (par image), JSON, ou
  ZIP (toutes les visualisations).

- **Dashboard** : statistiques réelles calculées à partir des analyses déjà
  effectuées (total, répartition par classe, confiance moyenne, temps
  d'inférence moyen, évolution dans le temps). Vide tant qu'aucune analyse
  n'a été faite.

- **Historique** : liste des analyses passées, avec recherche par nom de
  fichier, filtre par classe, suppression, et ré-export JSON/PDF d'une
  analyse déjà réalisée.

- **Rapports** : aperçu centralisé des rapports (fonctionnalité en cours de
  développement — le téléchargement existe déjà via Analyse et Historique).

- **Assistant IA** : ce chat, qui répond aux questions sur le projet, le
  modèle, Grad-CAM, la confiance, Docker, etc. — jamais de diagnostic.

- **Paramètres** : affiche la configuration actuelle de l'application
  (lecture seule pour l'instant).

- **À propos** : présentation du projet, architecture, limites, feuille de
  route.

Le sélecteur de langue (français / anglais / arabe) est disponible en haut
de la barre latérale et s'applique à la navigation ainsi qu'aux pages
À propos et Paramètres.
