# Journal des versions

## 0.1.0 — 2026-11-10

- Résultats sur le test réel : CamemBERT mixte 0,962 [0,95 – 0,97], 69 fuites ; adapté 0,955 (pas de gain) ; spaCy mixte 0,877.
- CamemBERT : fenêtres de mots entiers à l'entraînement et à la prédiction (les en-têtes de décision dépassaient
  192 tokens et étaient tronqués : adresses et localités perdues).
- `pseudo-ner adapt camembert` : pré-entraînement adapté au domaine (MLM sur le texte du train) ; `train camembert --base` ;
  cible `train-mixed-camembert` (générique vs adapté, même test réel).
- Pseudonymes : une seule suite de lettres pour prénoms, noms et professionnels (« [A] [B] », jamais « [A] [A] »).
- Mise en production : version lisible par modèle, `registry promote` / `rollback` / `list` (`models/current.json`),
  `/health` avec la version servie, compteurs de relecture et de volume dans `/metrics`, image Docker étiquetée par
  version, manifestes Kubernetes, `release.yml`, CI qui construit et démarre l'image.
- Intervalle de confiance à 95 % du F1 micro (bootstrap par document) dans les rapports et le tableau comparatif.
- Corpus synthétique d'arrêts de cour d'appel, annoté par construction, pools de valeurs disjoints entre
  entraînement et test, bruit de surface.
- Trois modèles NER derrière une interface commune : spaCy, Flair BiLSTM-CRF (embeddings fr-forward/backward ou
  one-hot mot + forme hors ligne), CamemBERT token-classification. Correctif du rechargement Flair
  (ordre des sous-embeddings).
- Post-traitements d'après le moteur public de la Cour de cassation : propagation des noms, regex des numéros,
  checklist de relecture (`check_needed`).
- Pseudonymisation cohérente par décision, table de correspondance.
- Évaluation stricte P/R/F1 par catégorie, fuites, registre des modèles avec empreinte et seuil, test de non-régression.
- API FastAPI au format `pseudonymisation-api` (`/ner`, `/health`, `/metrics`).
- MongoDB (résultats, relecture) et ElasticSearch (recherche, `more_like_this`) avec implémentations en mémoire.
- Décisions réelles : client API Judilibre (export paginé, quota), exploration du corpus (marques, contextes),
  ré-injection des marques en valeurs synthétiques cohérentes → jeu d'évaluation réaliste (`--labels reinjected`),
  annotation faible des professionnels par règles, jeu mixte synthétique + réel pour réentraîner (`make train-mixed`),
  pools de valeurs étendus et disjoints train / dev / test (`pools.py`) contre la mémorisation des noms.
- Docker, docker-compose (api + mongo + elasticsearch), CI, 47 tests, docs (état de l'art, exploitation).
