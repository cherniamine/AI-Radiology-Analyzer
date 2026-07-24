# AI Radiology Analyzer — image Docker pour l'application Streamlit
#
# Ne construit QUE ce dont l'application a besoin a l'execution :
# app/, le modele entraine (models/simple_cnn_model.h5) et les metriques
# d'evaluation (results/metrics.json). Le dossier data/ (~900 Mo d'images
# d'entrainement) n'est pas necessaire pour servir des predictions et
# n'est donc jamais copie dans l'image (voir .dockerignore).

FROM python:3.10-slim

# Dependances systeme requises par opencv-python (libGL) et curl pour le healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dependances Python en premier pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application
COPY app/ ./app/

# Artefacts necessaires a l'execution uniquement (pas tout results/, pas data/)
COPY models/simple_cnn_model.h5 ./models/simple_cnn_model.h5
COPY results/metrics.json ./results/metrics.json

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/predict.py", \
    "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
