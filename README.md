# pseudonymisation-ner

Pseudonymisation automatisée de décisions de justice : **reconnaissance d'entités nommées** (trois
architectures comparées — spaCy, Flair BiLSTM-CRF, CamemBERT), **post-traitements déterministes**, **remplacement
cohérent**, **évaluation stricte avec tests de non-régression**, **API** au format de celle de la Cour de
cassation, **MongoDB** pour les résultats et relectures, **ElasticSearch** pour la recherche sur le texte
pseudonymisé, et **évaluation sur décisions réelles** (API Judilibre, marques ré-injectées).

Le dépôt reprend volontairement les catégories, le format de sortie et la logique de post-traitement du moteur
public de la Cour (`Cour-de-cassation/moteurNER`, `pseudonymisation-api`) pour que les résultats soient
comparables et le code directement lisible par l'équipe. Il tourne sur un portable, sans GPU.

```
 décision ──▶ tokenizer spaCy ──▶ modèle NER ──▶ post-traitement ──▶ pseudonymisation ──▶ API /ner
   (texte)      (phrases, tokens)   spaCy | Flair   propagation des     [A], [Adresse 1]…    entities,
                                    | CamemBERT     noms, regex,        cohérent par         check_needed,
                                                    checklist           décision             checklist
                                          │                                     │
                                   évaluation stricte                   MongoDB (résultats, relecture)
                                   P/R/F1 par catégorie,                ElasticSearch (texte pseudonymisé,
                                   fuites, non-régression               recherche, similarité)
```

## Résultats

Jeu de test : 90 décisions synthétiques, **3 192 entités**, dont les noms, villes, voies et sociétés
**n'apparaissent jamais dans l'entraînement**, avec bruit de surface (noms en majuscules, accents perdus,
civilité absente). Entraînement : 420 décisions, CPU (MacBook Pro), de 2 min (spaCy) à 1 h 30 (Flair fr).

| Modèle | F1 micro (modèle seul) | F1 micro (avec post-traitement) | Rappel | Fuites | s / décision | entraînement |
|---|---|---|---|---|---|---|
| Flair BiLSTM-CRF, embeddings fr-forward / fr-backward (architecture de la Cour) | 1,000 | 1,000 | 1,000 | 0 | 1,482 | 90 min, 10 époques |
| CamemBERT-base, classification de tokens | 0,9998 | 1,000 | 1,000 | 0 | 0,502 | 25 min, 2 époques |
| Flair BiLSTM-CRF, embeddings one-hot mot + forme (hors ligne) | 1,000 | 1,000 | 1,000 | 0 | 0,049 | 8 min |
| spaCy (CNN, features de forme) | 0,985 | 0,985 | 0,985 | 0 | 0,012 | 2 min |

Flair fr-forward / fr-backward atteint 0,9986 de F1 en dev dès la première époque et 1,000 à la deuxième ; les
huit suivantes n'apportent rien. Sur le test, aucune erreur sur les 3 126 entités. CamemBERT en manque une seule :
le nom **« Meunier »**, qui est aussi un nom commun — la propagation des noms du post-traitement le rattrape
(la même personne est nommée ailleurs dans la décision), d'où 1,000 après post-traitement.

**Lecture honnête.** Le corpus est synthétique, donc régulier : des scores proches de 1 sont un *plafond*,
pas une performance attendue sur de vraies décisions (la Cour publie 0,94 de F1 micro sur son corpus réel).
Ce que la comparaison montre vraiment :

- l'erreur systématique de spaCy est instructive : **« La Rochelle »** (localité à article, jamais vue) est
  étiquetée « Rochelle » — bornes fausses, entité comptée manquée *et* fausse alerte ; le CRF de Flair, qui
  modélise la séquence, ne fait pas cette erreur ;
- la **fuite** (entité à masquer non couverte) est à 0 pour les quatre : c'est la métrique qui compte pour la
  publication, elle est testée en non-régression ;
- le **post-traitement** n'est pas décoratif : il corrige la seule erreur de CamemBERT (« Meunier ») et prend les
  66 numéros ; c'est le même rôle que dans le moteur de la Cour ;
- les **numéros normalisés** (sécurité sociale, téléphone, immatriculation, SIREN) sont pris par les règles,
  pas par le modèle : 66/66 ;
- le **temps** : les embeddings de caractères contextuels (fr-forward / fr-backward) coûtent 1,5 s par décision
  sur CPU, 3 × CamemBERT, 30 × la variante one-hot et 120 × spaCy. Sur 2 M de décisions, c'est le vrai choix à faire :
  ce qu'ils apportent sur du texte réel (noms rares, fautes d'OCR) contre le débit — et c'est ce que la Cour a
  tranché en gardant Flair, avec un modèle de langage ré-entraîné sur ses décisions.

Tableau complet par catégorie : `reports/*.json`, `python scripts/results_table.py`.

## Décisions réelles : Judilibre et ré-injection

Les décisions publiées par la Cour (API Judilibre, portail PISTE) sont **déjà pseudonymisées** :

```
Madame [E] [A]-[M]
[Adresse 1]
[Localité 1]
née le [Date naissance 1] 1975 à [Localité 2]
Représentée par Me Thibault ROUX, avocat au barreau de LYON
```

On ne peut donc pas entraîner un NER dessus. `judilibre.py` en tire deux choses :

- **exploration** (`pseudo-ner judilibre stats`) : tailles, nombre de mentions par marque, mots qui précèdent chaque
  marque (« née le », « demeurant », « domicilié ») — ce qui manque au corpus synthétique se lit dans ce rapport ;
- **ré-injection** (`pseudo-ner judilibre reinject`) : chaque marque est remplacée par une valeur synthétique,
  cohérente dans la décision (même lettre → même nom ; « [E] [A]-[M] » → prénom puis nom composé ; civilité
  respectée ; l'année de naissance, publiée en clair, est gardée dans l'entité). On obtient de **vraies décisions**
  — vraies phrases, vraie mise en page, avocats et magistrats en clair — annotées aux positions exactes des valeurs
  injectées, pour les catégories que la Cour masque.

```bash
export PISTE_CLIENT_ID=… PISTE_CLIENT_SECRET=…   # identifiants OAuth de l'application PISTE (gratuits), API Judilibre
make judilibre               # 500 décisions de cours d'appel → data/judilibre/, reports/judilibre_stats.json
make evaluate-real           # les trois modèles sur ces décisions, catégories masquées uniquement
```

L'évaluation sur ce jeu (`--labels reinjected`) ne compte que parties, adresses, localités et dates de naissance.
Les avocats et magistrats, en clair dans le texte publié, reçoivent une annotation **faible, par règles**
(« Me Prénom NOM », « Prénom NOM, conseillère », « Greffier : Mme … ») qui sert à l'entraînement, jamais au score.
Deux artefacts à garder en tête en lisant les chiffres : les villes en clair (siège de la cour, barreau) ne sont pas
annotées, donc la *précision* en localité n'est pas mesurable, seul le rappel l'est ; et une date d'audience
étiquetée « date de naissance » compte comme fausse alerte — c'est bien une erreur, du sur-masquage.

**Réentraîner avec du réel** (`make train-mixed`, ablation `make train-real-only`) : partage des décisions en
train / dev / test *par décision*, train = synthétique + 360 décisions réelles ré-injectées, test = 100 décisions
réelles jamais vues. Le même modèle, évalué sur le même test avant et après, mesure ce que le réel apporte. C'est le
chiffre à mettre en face des 0,94 publiés par la Cour, pas celui du tableau synthétique.

| Même test réel (100 décisions, 5 026 entités), avec post-traitement | F1 micro [IC 95 %] | F1 noms | F1 prénoms | F1 adresse | F1 localité | Fuites | s / décision |
|---|---|---|---|---|---|---|---|
| spaCy, synthétique seul (`reports/spacy-realtest.json`) | 0,457 [0,40 – 0,51] | 0,600 | 0,562 | 0,365 | 0,086 | 1 153 | 0,13 |
| spaCy, réel ré-injecté seul (`spacy-real-realtest`) | 0,845 [0,82 – 0,87] | 0,888 | 0,815 | 0,934 | 0,617 | 695 | 0,12 |
| spaCy, synthétique + réel (`spacy-mixed-realtest`) | 0,877 [0,85 – 0,91] | 0,938 | 0,859 | 0,876 | 0,583 | 266 | 0,12 |
| CamemBERT, synthétique + réel (`camembert-mixed-realtest`) | **0,962** [0,95 – 0,97] | **0,965** | 0,959 | **0,963** | **0,963** | **69** | 2,8 |
| CamemBERT adapté au domaine, synthétique + réel (`camembert-adapted-mixed-realtest`) | 0,955 [0,93 – 0,97] | 0,953 | **0,965** | 0,950 | 0,961 | 61 | 2,8 |
| moteur de la Cour, chiffres publiés (183 k phrases annotées à la main) | 0,941 | 0,974 | 0,944 | 0,865 | — | — | — |

Attention : le test est lui aussi ré-injecté. En dé-identification médicale, Yeniterzi et al. (JAMIA 2010) mesurent
qu'un modèle entraîné et testé sur des identifiants resynthétisés surestime sa performance sur du texte brut (F 0,98
contre 0,73) ; ces chiffres sont des majorants, seul un test annoté à la main tranche. Le 0,962 de CamemBERT n'est
donc pas « au-dessus » des 0,941 de la Cour : il est du même ordre, sur un test plus facile.

Ce que l'ablation spaCy établit : le réel fait l'essentiel (les intervalles « synthétique seul » et « mixte » ne se
recouvrent pas) ; le synthétique ajouté au réel apporte 3 points, mais les intervalles « réel seul » et « mixte » se
recouvrent — non tranché sur 100 décisions ; ce qui l'est : 2,6 fois moins de fuites (266 contre 695) et les dates de
naissance, rares dans les décisions publiées (14 dans le test), que le modèle réel seul ne trouve jamais.

Ce que CamemBERT établit : à données égales, le Transformer bat spaCy sans ambiguïté (intervalles disjoints, 4 fois
moins de fuites, adresses et localités de 0,6–0,9 à 0,96), pour 23 fois plus de temps par décision. Le point faible
commun reste la date de naissance (14 cas : des dates d'audience prises pour des dates de naissance).

**Deuxième étage : pré-entraînement adapté au domaine** (`make train-mixed-camembert`). Avant d'affiner CamemBERT
pour le NER, `pseudo-ner adapt camembert` continue son pré-entraînement (masquage de mots, sans étiquette) sur le
texte du jeu d'entraînement — « don't stop pretraining » (Gururangan et al., ACL 2020) ; à petite échelle, ce que
fait la Cour avec son modèle de langage ré-entraîné sur 2 M de décisions, et ce que mesure Niklaus et al. (2023) sur
les arrêts du Tribunal fédéral suisse (« plus de 5 % » de F1). Résultat ici : la perte de masquage passe de 2,74 à 1,32 en
une époque (32 067 paragraphes, 1 h 25 CPU), mais le F1 ne bouge pas (0,955 contre 0,962, intervalles confondus ;
61 fuites contre 69). Comme pour JuriBERT sur la classification des mémoires, l'adaptation ne paie pas à cette
échelle (780 décisions) ; à celle de la Cour (2 M), c'est une autre question.

**Deuxième piège, trouvé en lisant les scores par catégorie** : la première évaluation de CamemBERT donnait adresses
0,64 et localités 0,66 alors que les noms étaient à 0,94. Cause : la prédiction découpait en phrases puis tronquait
chaque phrase à 192 tokens ; l'en-tête d'une décision (parties, adresses, sans ponctuation) est une seule « phrase »
de plusieurs centaines de mots, dont la fin n'était jamais lue. Correctif : fenêtres de mots entiers, à la
prédiction et à l'entraînement ; sans réentraîner, adresses 0,64 → 0,96, fuites 507 → 69.

**Piège rencontré, et corrigé** (`pools.py`) : une décision réelle porte en moyenne 45 mentions de personnes. Avec
un pool de 25 noms, chaque nom revenait des centaines de fois dans l'entraînement et spaCy apprenait *la liste*
plutôt que *le contexte* — dev à 0,99 (mêmes noms), test à 0,13 de rappel sur les noms (noms nouveaux). Les valeurs
ré-injectées viennent désormais de pools de plusieurs centaines de noms, prénoms, communes et voies, partagés en
trois parts disjointes (train / dev / test, le test excluant aussi tout ce que le synthétique a vu) : un nom
n'apparaît que dans une ou deux décisions, le contexte redevient le seul signal fiable. Effet sur le même test
réel : F1 noms 0,22 → 0,94, F1 micro 0,42 → 0,88, fuites 2 889 → 266. Les données brutes restent hors du dépôt ; leur réutilisation suit les
[CGU de la Cour](https://www.courdecassation.fr/conditions-generales-dutilisation-pour-la-reutilisation-des-donnees-issues-des-decisions-de-justice).

## Ce que fait la chaîne

**Catégories** (`categories.py`) : `personnePhysiqueNom`, `personnePhysiquePrenom`, `dateDeNaissance`,
`personneMorale`, `adresse`, `localite`, `professionnelAvocat`, `professionnelMagistratGreffier` (apprises) et
`numeroIdentifiant` (règles). Par défaut sont masqués les parties, leurs dates de naissance, adresses et numéros ;
professionnels et localités sont conservés, comme dans la pratique de la Cour — paramétrable par requête.

**Post-traitements** (`postprocess.py`), d'après le moteur public : propagation d'un nom vu ≥ 2 fois à toutes ses
occurrences ; regex pour les numéros ; **checklist** pour les relecteurs — nom très court, prénoms proches
(« Thibaut » / « Thibault »), même mention étiquetée partie *et* professionnel — qui met `check_needed` à vrai.

**Pseudonymisation** (`pseudonymise.py`) : même valeur → même pseudonyme dans la décision (`[A]`, `[B]`,
`[Adresse 1]`, `[Date naissance 1]`, `[Société A]`), table de correspondance conservée hors publication.

**Évaluation** (`evaluate.py`) : P / R / F1 par catégorie, bornes et catégorie exactes ; liste des manqués et des
fausses alertes ; **fuites** = mentions de référence à masquer non couvertes par une entité masquée ; **intervalle
à 95 %** du F1 micro par bootstrap sur les *documents* (les mentions d'une même décision ne sont pas indépendantes :
un bootstrap sur les mentions donnerait un intervalle faussement étroit).

**Registre, promotion, non-régression** (`registry.py`, `tests/test_regression.py`) : chaque modèle évalué est
enregistré avec l'empreinte SHA-256 de ses fichiers, une version lisible (`spacy-2026-11-10-3f2a9c1e`), ses scores
et un seuil (F1 − 2 points) ; le test recharge le modèle, vérifie l'empreinte, le seuil et l'absence de fuite.
`pseudo-ner registry promote <modèle>` désigne le modèle servi (`models/current.json`, empreinte et seuil
revérifiés) ; `registry rollback` remet le précédent ; `registry list` montre lequel est servi.

**API** (`api.py`) : `POST /ner` `{idDocument, text, categories?}` → `{entities, check_needed, checklist,
pseudonymised_text, replaced}` ; `/health` renvoie la version du modèle servi (vérification après déploiement) ;
`/metrics` Prometheus (requêtes, erreurs, temps, entités, décisions envoyées en relecture, caractères traités).

**Déploiement** : image Docker étiquetée avec la version du modèle promu (`make image`), `docker-compose.yml`,
manifestes Kubernetes (`deploy/k8s/` : mise à jour progressive, sondes sur `/health`, conteneur en lecture seule
sans privilège), `release.yml` qui publie l'image à chaque étiquette `model-<version>`. Pas à pas et retour arrière
dans [`docs/exploitation.md`](docs/exploitation.md).

**Stockage et recherche** (`search.py`) : MongoDB garde le résultat complet (texte d'origine à accès restreint,
entités, checklist, modèle et empreinte) et la file de relecture ; ElasticSearch n'indexe que le texte
pseudonymisé (analyseur français) pour la recherche plein texte et la **similarité documentaire**
(`more_like_this`). Implémentations en mémoire pour les tests ; `docker-compose.yml` pour les vrais services.

## Démarrer

```bash
git clone … && cd pseudonymisation-ner
python -m venv .venv && source .venv/bin/activate
make install            # paquet + flair + camembert + search + outils de dev
make data               # corpus synthétique (train / dev / test à valeurs disjointes)
make train-spacy        # entraîne, évalue sur le test, enregistre (≈ 2 min)
make serve              # API sur http://127.0.0.1:8000  (docs : /docs)
```

```bash
curl -s -X POST localhost:8000/ner -H 'content-type: application/json' -d '{"idDocument": 1,
 "text": "Mme Claire DUFOUR, née le 3 mai 1970 à La Rochelle, demeurant 4 rue des Écoles, Valence, représentée par Me Paul Martin."}'
```

```json
{"entities": [{"text": "Claire", "start": 4, "end": 10, "label": "personnePhysiquePrenom", "source": "NER model"},
              {"text": "DUFOUR", "start": 11, "end": 17, "label": "personnePhysiqueNom", "source": "NER model"}, "…"],
 "check_needed": false, "checklist": [],
 "pseudonymised_text": "Mme [A] [A], née le [Date naissance 1] à La Rochelle, demeurant [Adresse 1], Valence, représentée par Me Paul Martin.",
 "replaced": 4, "model": "spacy", "seconds": 0.011}
```

Les autres modèles, MongoDB, ElasticSearch, le changement de modèle et les pannes :
[`docs/exploitation.md`](docs/exploitation.md). Pourquoi ces architectures, ce qu'on mesure et les limites :
[`docs/etat_de_l_art.md`](docs/etat_de_l_art.md).

## Qualité

- 47 tests (plus un test de non-régression par modèle enregistré) : corpus (positions exactes, pools disjoints), conversion BIO ↔ caractères, évaluation, chaque règle de
  post-traitement, pseudonymisation (cohérence, aucune fuite), API (schéma public, erreurs, catégories), MongoDB
  (`mongomock`), ElasticSearch (client simulé), registre (empreinte, seuil), non-régression des modèles enregistrés,
  Judilibre (export paginé simulé, exploration, ré-injection : cohérence, positions, civilité, règles
  professionnels, partage par décision).
- `ruff`, `mypy`, CI GitHub Actions sur Python 3.10 → 3.12 (lint, types, tests, non-régression spaCy, promotion, construction
  et démarrage de l'image Docker avec appel de `/health`, `docker compose config`).
- Conteneur de service en lecture seule, sans capacité système ; ports sur `127.0.0.1`.

## Données

Le corpus d'entraînement est **entièrement synthétique** (`synthetic.py`) : gabarits d'arrêts de cour d'appel, listes
de prénoms, noms, villes et voies courants, dates et numéros générés. Aucune personne réelle. Les pools de valeurs du
jeu de test sont disjoints de ceux de l'entraînement pour mesurer la généralisation. Les décisions réelles
(Judilibre) ne contiennent que des marques de pseudonymisation ; les valeurs ré-injectées viennent des mêmes listes.

## Limites

Le test réel est ré-injecté (majorant, voir plus haut) et 100 % cours d'appel ; Flair n'est pas réentraîné sur le
réel (seuls spaCy et CamemBERT le sont) ; l'adaptation au domaine est faite sur 780 décisions, pas sur un corpus
juridique à l'échelle de la Cour, et sans byte-pair embeddings ; CamemBERT affiné 2 époques, sans recherche
d'hyperparamètres ; les valeurs ré-injectées viennent de listes (noms rares ou étrangers sous-représentés) ;
pas de HTTPS ni d'authentification sur l'API (prévus derrière un reverse proxy) ; relecture non outillée
(la file `check_needed` est dans MongoDB, l'interface est à brancher — la Cour a `label`).
