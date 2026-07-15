# AI Radiology Analyzer

Classification de radiographies pulmonaires (COVID-19, opacité pulmonaire, pneumonie virale, normal) par réseau de neurones convolutif, avec cartes d'attention Grad-CAM et interface Streamlit interactive.

> **Avertissement.** Ce projet est un prototype académique à but pédagogique. Il ne fournit aucune valeur diagnostique et ne doit en aucun cas remplacer l'avis d'un radiologue ou d'un médecin.

---

## Sommaire

- [Aperçu](#aperçu)
- [Résultats](#résultats)
- [Architecture du modèle](#architecture-du-modèle)
- [Jeu de données](#jeu-de-données)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Limites connues](#limites-connues)
- [Pistes d'amélioration](#pistes-damélioration)
- [Auteur](#auteur)

---

## Aperçu

Le projet couvre l'ensemble du pipeline de classification d'images médicales :

- **Prétraitement** des radiographies (redimensionnement, normalisation, augmentation de données)
- **Entraînement** d'un CNN sur quatre classes de pathologies pulmonaires
- **Évaluation** quantitative (précision, rappel, F1-score, matrice de confusion)
- **Explicabilité** via Grad-CAM, pour visualiser les régions de l'image ayant motivé la prédiction
- **Interface applicative** (Streamlit) permettant de déposer des radiographies et d'obtenir un rapport exportable (CSV) et des visualisations (ZIP)

## Résultats

Mesures obtenues sur le jeu de test (`results/metrics.json`), à jour au dernier entraînement disponible dans ce dépôt :

**Exactitude globale : 85.9 %**

| Classe            | Précision | Rappel | F1-score | Échantillons (test) |
|-------------------|:---------:|:------:|:--------:|:--------------------:|
| COVID-19           | 85.6 %    | 77.5 % | 81.3 %   | 1 446 |
| Opacité pulmonaire | 80.3 %    | 86.4 % | 83.2 %   | 2 404 |
| Normal             | 89.0 %    | 88.1 % | 88.6 %   | 4 076 |
| Pneumonie virale   | 90.0 %    | 88.5 % | 89.2 %   | 538 |

La matrice de confusion et les courbes d'entraînement complètes sont disponibles dans `results/confusion_matrix.png` et `results/curves/`.

Le rappel plus faible sur la classe COVID-19 (77.5 %) est le point le plus sensible du modèle dans un contexte clinique — voir [Limites connues](#limites-connues).

## Architecture du modèle

Le dépôt contient deux définitions de modèle :

- **`models/simple_cnn_model.h5`** — le modèle effectivement utilisé par l'application (`app/predict.py`) et évalué ci-dessus. Il s'agit d'un CNN simple : 3 blocs `Conv2D` + `MaxPooling2D`, suivis d'un `Flatten`, d'un `Dropout` et de deux couches `Dense` (128×128×3 en entrée, 4 classes en sortie).
- **`src/model.py`** — une architecture alternative basée sur **EfficientNetB0** (transfer learning, poids ImageNet gelés, tête de classification dédiée), prévue pour une itération future mais non utilisée par le modèle livré.

L'explicabilité est assurée par **Grad-CAM**, calculé sur la dernière couche convolutive du modèle, avec superposition colorée (colormap configurable : JET, HOT, PLASMA, VIRIDIS, INFERNO) sur la radiographie d'origine.

## Jeu de données

Radiographies thoraciques réparties en 4 classes, pour un total de **42 330 images** :

| Classe             | Images |
|--------------------|-------:|
| Normal             | 20 384 |
| Opacité pulmonaire | 12 024 |
| COVID-19           |  7 232 |
| Pneumonie virale   |  2 690 |

Le jeu de données est **déséquilibré** (la classe *Normal* représente près de la moitié des images, la *Pneumonie virale* moins de 7 %), ce qui explique en partie l'écart de performance entre classes observé plus haut.

Répartition entraînement / validation / test gérée par `src/preprocessing.py` (80/20 avec `ImageDataGenerator`, augmentation par rotation, zoom et retournement horizontal sur l'ensemble d'entraînement uniquement).

## Structure du projet

```
AI-Radiology-Analyzer/
├── app/
│   └── predict.py            # Application Streamlit (upload, prédiction, Grad-CAM, export)
├── src/
│   ├── preprocessing.py       # Générateurs de données (train/val/test)
│   ├── model.py                # Définition du modèle (EfficientNetB0)
│   ├── train.py                # Boucle d'entraînement
│   └── evaluate.py             # Évaluation (rapport de classification, matrice de confusion)
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_gradcam.ipynb
├── models/
│   └── simple_cnn_model.h5    # Modèle entraîné utilisé par l'application
├── results/
│   ├── metrics.json           # Métriques d'évaluation par classe
│   ├── model_info.json        # Détail de l'architecture du modèle livré
│   ├── confusion_matrix.png
│   └── curves/                 # Courbes d'entraînement
├── data/                       # Images sources, organisées par classe
└── requirements.txt
```

> Le fichier `05 Evaluation Gradcam.ipynb` semble être une copie de travail de `04_evaluation_gradcam.ipynb` ; à vérifier et supprimer si redondant.

## Installation

Prérequis : Python 3.9+.

```bash
git clone https://github.com/cherniamine/AI-Radiology-Analyzer.git
cd AI-Radiology-Analyzer
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

**Lancer l'application :**

```bash
streamlit run app/predict.py
```

L'application s'ouvre sur `http://localhost:8501`. Déposez une ou plusieurs radiographies (PNG/JPG) pour obtenir, par image : la classe prédite, le score de confiance par classe, la carte Grad-CAM, puis exportez le rapport (CSV) et les visualisations (ZIP).

**Ré-entraîner ou évaluer le modèle :**

```bash
python src/train.py       # entraînement (lit data/, écrit models/simple_cnn_model.h5)
python src/evaluate.py    # évaluation sur le jeu de test
```

Les notebooks du dossier `notebooks/` retracent, dans l'ordre, l'exploration des données, le prétraitement, l'entraînement et l'évaluation avec Grad-CAM.

## Limites connues

- **Rappel COVID-19 limité (77.5 %)** : dans un cas d'usage réel, ce taux de faux négatifs serait inacceptable sans confirmation par un second examen.
- **Déséquilibre des classes** dans le jeu de données, non compensé par pondération de classe ou sur-échantillonnage.
- **Modèle simple CNN** entraîné from scratch avec peu de régularisation ; l'architecture EfficientNetB0 fournie dans `src/model.py` n'a pas été évaluée dans ce dépôt.
- **Jeu de données non annoté par un radiologue** dans ce projet (dataset public) ; aucune validation clinique n'a été réalisée.
- Le dépôt contient les images sources et les poids du modèle (~900 Mo au total) directement versionnés dans Git plutôt que via Git LFS ou un stockage externe.

## Pistes d'amélioration

- Pondération de classe ou ré-échantillonnage pour réduire l'écart de rappel entre classes.
- Évaluation et comparaison du modèle EfficientNetB0 par rapport au CNN actuel.
- Validation croisée plutôt qu'un split unique train/val/test.
- Déplacer `data/` et `models/*.h5` hors du contrôle de version (Git LFS ou stockage cloud) et les exclure via `.gitignore`.

## Auteur

**Cherni Mohamed Amine**
Étudiant ingénieur en IA & Data Science — Université Centrale Tunisie
