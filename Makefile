.PHONY: install lint test check data train-spacy train-flair train-flair-offline train-camembert evaluate-all judilibre evaluate-real train-mixed train-real-only adapt-camembert train-mixed-camembert train-mixed-flair promote image serve index up down

install:
	pip install -e ".[dev,flair,camembert,search]"

lint:
	ruff check . && ruff format --check . && mypy

test:
	pytest --cov=pseudo_ner --cov-report=term-missing --ignore=tests/test_regression.py

check: lint test                 ## ce que fait la CI

data:                            ## corpus synthétique (train/dev : valeurs connues ; test : valeurs jamais vues)
	pseudo-ner generate --n 600

train-spacy:
	pseudo-ner train spacy --out models/spacy --epochs 12
	pseudo-ner evaluate spacy --model-dir models/spacy --report reports/spacy.json --register

train-flair:                     ## architecture du moteur de la Cour (BiLSTM-CRF + embeddings fr-forward/backward)
	pseudo-ner train flair --out models/flair --epochs 10 --embeddings flair-fr
	pseudo-ner evaluate flair --model-dir models/flair --report reports/flair.json --register

train-flair-offline:             ## même architecture, embeddings one-hot mot + forme (sans téléchargement)
	pseudo-ner train flair --out models/flair-onehot --epochs 8 --embeddings onehot
	pseudo-ner evaluate flair --model-dir models/flair-onehot --report reports/flair-onehot.json

train-camembert:                 ## Transformer (CamemBERT) fine-tuné en classification de tokens
	pseudo-ner train camembert --out models/camembert --epochs 2
	pseudo-ner evaluate camembert --model-dir models/camembert --report reports/camembert.json --register

evaluate-all: train-spacy train-flair train-camembert
	python scripts/results_table.py

judilibre:                       ## décisions réelles (PISTE_CLIENT_ID / PISTE_CLIENT_SECRET) : export, exploration, ré-injection
	pseudo-ner judilibre fetch --n 500 --production
	pseudo-ner judilibre stats
	pseudo-ner judilibre reinject

evaluate-real:                   ## les modèles sur les décisions réelles ré-injectées (catégories masquées)
	pseudo-ner evaluate spacy --model-dir models/spacy --file data/judilibre/reinjected.jsonl --labels reinjected --report reports/spacy-judilibre.json
	pseudo-ner evaluate flair --model-dir models/flair --file data/judilibre/reinjected.jsonl --labels reinjected --report reports/flair-judilibre.json
	pseudo-ner evaluate camembert --model-dir models/camembert --file data/judilibre/reinjected.jsonl --labels reinjected --report reports/camembert-judilibre.json
	python scripts/results_table.py

train-mixed:                     ## réentraîne avec des décisions réelles ré-injectées, test sur des décisions réelles jamais vues
	pseudo-ner judilibre split
	pseudo-ner evaluate spacy --model-dir models/spacy --file data/mixed/test.jsonl --labels reinjected --report reports/spacy-realtest.json
	pseudo-ner train spacy --data data/mixed --out models/spacy-mixed --epochs 12
	pseudo-ner evaluate spacy --model-dir models/spacy-mixed --file data/mixed/test.jsonl --labels reinjected --report reports/spacy-mixed-realtest.json
	pseudo-ner evaluate spacy --model-dir models/spacy-mixed --report reports/spacy-mixed-synthetic.json
	python scripts/results_table.py

train-real-only:                 ## ablation : réel ré-injecté seul, même test réel que train-mixed
	pseudo-ner judilibre split --no-synthetic --out data/real
	pseudo-ner train spacy --data data/real --out models/spacy-real --epochs 12
	pseudo-ner evaluate spacy --model-dir models/spacy-real --file data/mixed/test.jsonl --labels reinjected --report reports/spacy-real-realtest.json
	pseudo-ner evaluate spacy --model-dir models/spacy-real --report reports/spacy-real-synthetic.json
	python scripts/results_table.py

adapt-camembert:                 ## pré-entraînement adapté au domaine (MLM) sur le texte du train mixte ; ~30 min CPU
	pseudo-ner adapt camembert --data data/mixed --out models/camembert-judilibre --epochs 1

train-mixed-camembert: adapt-camembert ## ablation : CamemBERT générique vs adapté, affinés sur le mixte, même test réel (plusieurs heures CPU)
	pseudo-ner train camembert --data data/mixed --out models/camembert-mixed --epochs 2
	pseudo-ner evaluate camembert --model-dir models/camembert-mixed --file data/mixed/test.jsonl --labels reinjected --report reports/camembert-mixed-realtest.json
	pseudo-ner train camembert --data data/mixed --out models/camembert-adapted-mixed --epochs 2 --base models/camembert-judilibre
	pseudo-ner evaluate camembert --model-dir models/camembert-adapted-mixed --file data/mixed/test.jsonl --labels reinjected --report reports/camembert-adapted-mixed-realtest.json
	python scripts/results_table.py

train-mixed-flair:               ## idem avec Flair fr-forward/backward (long : plusieurs heures CPU)
	pseudo-ner train flair --data data/mixed --out models/flair-mixed --epochs 3 --embeddings flair-fr
	pseudo-ner evaluate flair --model-dir models/flair-mixed --file data/mixed/test.jsonl --labels reinjected --report reports/flair-mixed-realtest.json

promote:                         ## désigne le modèle servi (models/current.json) : empreinte et seuil vérifiés
	pseudo-ner registry promote spacy
	pseudo-ner registry list

image: promote                   ## image Docker étiquetée avec la version du modèle promu
	docker build -t pseudo-ner:$$(python -c "import json; print(json.load(open('models/current.json'))['version'])") .

serve:                           ## sert le modèle promu (models/current.json) ; sinon : pseudo-ner serve spacy --model-dir models/spacy
	pseudo-ner serve

index:                           ## traite le jeu de test, stocke dans MongoDB, indexe dans ElasticSearch
	pseudo-ner index spacy --model-dir models/spacy --mongo mongodb://localhost:27017 --es http://localhost:9200 --query "licenciement"

up:
	docker compose up -d --build

down:
	docker compose down
