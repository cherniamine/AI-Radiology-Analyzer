# Docker et déploiement

L'application peut se lancer avec Docker :

```bash
git clone https://github.com/cherniamine/AI-Radiology-Analyzer.git
cd AI-Radiology-Analyzer
docker compose up --build
```

L'application est alors disponible sur http://localhost:8501.

L'image Docker ne contient que le code de l'application, le modèle entraîné
(models/simple_cnn_model.h5, environ 38 Mo) et les métriques d'évaluation
(results/metrics.json) — pas les 900 Mo d'images d'entraînement du dossier
data/, qui ne sont pas nécessaires pour servir des prédictions.

Un volume Docker nommé (history-data) est monté sur /app/results pour que
l'historique des analyses (base SQLite) survive à un redémarrage du
conteneur — sans ce volume, le système de fichiers du conteneur serait
éphémère.

Sans Docker, l'application se lance avec :

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/predict.py
```

Une pipeline d'intégration continue (GitHub Actions, `.github/workflows/ci.yml`)
vérifie la syntaxe et exécute la suite de tests automatiquement à chaque
push ou pull request, sans charger le modèle TensorFlow complet (pour rester
rapide).
