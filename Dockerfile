# Image de service : modèle spaCy (léger, CPU) + API. Les variantes Flair / CamemBERT
# s'installent avec les extras correspondants (voir README).
#
#   docker build --build-arg MODEL_DIR=models/spacy -t pseudo-ner:<version> .
#   <version> = champ "version" de models/current.json (pseudo-ner registry current)
FROM python:3.12-slim

ARG MODEL=spacy
ARG MODEL_DIR=models/spacy
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PSEUDO_NER_MODEL=${MODEL} PSEUDO_NER_MODEL_DIR=${MODEL_DIR}
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# modèle, registre et modèle promu : copiés au build (ou montés en volume en production)
COPY ${MODEL_DIR} ./${MODEL_DIR}
COPY models/registry.json ./models/registry.json

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"
CMD ["sh", "-c", "pseudo-ner serve $PSEUDO_NER_MODEL --model-dir $PSEUDO_NER_MODEL_DIR --host 0.0.0.0 --port 8000"]
