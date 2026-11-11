# Pseudonymisation des décisions de justice : état de l'art et choix

## Le problème

L'open data des décisions de justice (loi pour une République numérique, 2016 ; décret du 29 juin 2020) impose
d'occulter les éléments permettant d'identifier les personnes physiques parties ou tiers, et laisse une
appréciation au juge pour les professionnels. Techniquement, c'est une tâche de **reconnaissance d'entités
nommées** (NER) suivie d'un **remplacement cohérent**, avec deux particularités :

- le **rappel** prime : une entité manquée est une donnée personnelle publiée, une fausse alerte n'est qu'un mot
  masqué en trop ; l'asymétrie des coûts doit se retrouver dans les seuils et la relecture ;
- les textes sont **longs, structurés et répétitifs** (en-tête, parties, motifs, dispositif) : une mention manquée
  est souvent présente ailleurs sous une forme reconnue, d'où l'intérêt de règles de propagation.

## Ce que fait la Cour de cassation (dépôts publics)

- **Moteur** (`Cour-de-cassation/moteurNER`) : Flair, BiLSTM-CRF sur des embeddings contextuels de caractères
  (fr-forward / fr-backward fine-tunés) combinés à des Byte-Pair Embeddings ; modèle de langage ré-entraîné sur
  ~2 M de décisions ; tokenizer spaCy français modifié. Corpus GOLD annoté par les agents : 183 k phrases
  d'entraînement. Scores publiés : F1 micro 0,94 ; nom de personne 0,97, prénom 0,94, personne morale 0,89,
  adresse 0,87, date de naissance 0,96.
- **Post-traitements déterministes** : propagation des noms vus ≥ 2 fois, regex pour les numéros normalisés,
  « rapport d'anonymisation » qui signale les doutes (noms très courts, prénoms proches) aux relecteurs.
- **API** (`Cour-de-cassation/pseudonymisation-api`) : `POST /ner` sur `{idDocument, text}` →
  `{entities[{text,start,end,label,source}], check_needed, checklist}` ; un second point calcule la *loss*
  d'un document après relecture (boucle d'amélioration).
- **Annotation** (`Cour-de-cassation/label`) : interface de relecture dont les corrections doivent remonter
  vers l'entraînement.

Ce dépôt reprend les catégories, le format de sortie et la logique de post-traitement, pour que les résultats
soient comparables et que le code soit directement lisible par l'équipe.

## Les trois familles d'architectures

| Architecture | Forces | Limites | Ici |
|---|---|---|---|
| **spaCy** (CNN + parser par transitions, features de forme/préfixe/suffixe hashées) | très rapide sur CPU, robuste aux mots inconnus grâce aux features de forme, simple à déployer | contexte local (fenêtre), plafonne sur les cas ambigus | référence « légère », ~10 ms / décision |
| **Flair** (BiLSTM-CRF sur embeddings de caractères contextuels) | excellent sur les noms jamais vus (modèle de caractères), le CRF impose des séquences BIO cohérentes ; c'est l'architecture en production à la Cour | embeddings à télécharger / ré-entraîner, inférence plus lente | même architecture ; variante hors ligne one-hot mot + forme pour valider la chaîne sans téléchargement |
| **CamemBERT** (Transformer pré-entraîné, classification de tokens) | contexte large, état de l'art sur le NER français (WikiNER, QUAERO) | 110 M de paramètres, GPU conseillé, alignement sous-mots → mots, longueur de séquence limitée | fine-tuning complet, première sous-pièce étiquetée |

Références : Akbik et al., *Contextual String Embeddings for Sequence Labeling* (COLING 2018) ; Martin et al.,
*CamemBERT: a Tasty French Language Model* (ACL 2020) ; Lample et al., *Neural Architectures for NER* (NAACL 2016).

## Évaluation : ce qu'on mesure et pourquoi

- **Précision / rappel / F1 par catégorie**, bornes et catégorie exactes (critère strict CoNLL).
- **Rappel sur les catégories masquées** : la métrique de sécurité.
- **Fuites** : mentions de référence à masquer non couvertes par une entité masquée, quelle que soit la
  catégorie prédite — c'est ce qui compte pour la publication, indépendamment des bornes.
- **Généralisation** : le jeu de test n'utilise que des noms, villes, voies et sociétés **absents** de
  l'entraînement, avec du bruit de surface (majuscules, accents perdus, civilité absente). Mesurer sur des
  valeurs vues reviendrait à mesurer de la mémorisation.
- **Non-régression** : chaque modèle enregistré porte un seuil (F1 − 2 points) ; un test recharge le modèle,
  vérifie son empreinte et refuse toute version qui passe dessous.

## Ce que dit la littérature sur les données synthétiques

- **Yeniterzi et al. (JAMIA 2010)**, *Effects of personal identifier resynthesis on clinical text de-identification*, et
  **Stubbs & Uzuner (J. Biomed. Inform. 2015)**, *The 2014 i2b2/UTHealth corpus* : en dé-identification médicale, les
  identifiants réels sont remplacés par des substituts réalistes (« resynthesis », « surrogates ») — c'est la
  ré-injection de ce dépôt, et le corpus de référence i2b2 est distribué ainsi. Yeniterzi mesure les quatre
  combinaisons : réel→réel F 0,96, resynthétisé→resynthétisé 0,98, resynthétisé→réel 0,73. Un score obtenu en
  entraînant et testant sur du ré-injecté est donc un majorant de la performance sur texte brut.
- **Dai & Adel (COLING 2020)**, *An Analysis of Simple Data Augmentation for Named Entity Recognition* : remplacer
  les mentions d'entités par d'autres mentions du même type (« mention replacement », ce que fait la ré-injection)
  améliore nettement le NER quand les données annotées sont rares, et le gain s'amenuise à mesure que le réel
  annoté augmente. Le synthétique est un complément, pas un substitut.
- **Barrière & Fouret (2019)**, *May I Check Again? A simple but efficient way to generate and use contextual
  dictionaries for NER — application to French legal texts* : sur des décisions de la Cour de cassation, des
  dictionnaires construits à partir du contexte (une mention reconnue une fois est reconnue partout) améliorent le
  rappel — c'est la propagation des noms du moteur de la Cour, reprise dans `postprocess.py`.
- **Cour de cassation, `moteurNER`** : le moteur en production n'est entraîné que sur du réel annoté par les
  agents (183 145 phrases), avec un modèle de langage ré-entraîné sur 2 M de décisions. Le synthétique n'y a pas
  sa place parce que le texte brut est disponible ; ici il comble son absence.

Ce que ces travaux ne tranchent pas, et que l'ablation `train-mixed` / `train-real-only` mesure : à budget réel
fixé (360 décisions ré-injectées), le synthétique ajouté à l'entraînement aide-t-il encore, ou nuit-il par son
régularité ? La décision de le garder ou non se prend sur ce chiffre. Résultat (spaCy, même test réel de 100
décisions) : synthétique seul 0,457 ; réel seul 0,845 ; synthétique + réel 0,877, avec 2,6 fois moins de fuites
(266 contre 695) et des dates de naissance retrouvées (0,41 contre 0,00 : elles sont rares dans les décisions
publiées). Le synthétique est gardé, en complément.

## Pré-entraînement adapté au domaine : ce que la littérature mesure

- **Gururangan et al., *Don't Stop Pretraining* (ACL 2020)** : continuer le pré-entraînement d'un modèle de langue
  sur le texte du domaine (DAPT), ou simplement sur le texte non étiqueté de la tâche (TAPT), améliore huit tâches
  dans quatre domaines, y compris quand les données étiquetées sont rares ; les deux se cumulent. `pseudo-ner adapt`
  est un TAPT : le texte du jeu d'entraînement, sans étiquette.
- **Niklaus et al. (2023)** : sur l'anonymisation des arrêts du Tribunal fédéral suisse, le pré-entraînement
  juridique apporte « plus de 5 % » de F1 (le résumé ne dit pas si c'est absolu ou relatif) — c'est la tâche la plus proche de la nôtre.
- **JuriBERT (2021)** : à l'inverse, sur la classification des mémoires de la Cour de cassation, CamemBERT continué
  sur le juridique (JuriBERT-FP) ne gagne rien (83,28 contre 83,22). Le gain n'est donc pas garanti ; il dépend de la
  tâche. D'où l'ablation `make train-mixed-camembert` : générique et adapté, même affinage, même test.
- **Cour de cassation** : modèle de langage Flair ré-entraîné sur ~2 M de décisions — le même principe, à une échelle
  que 360 décisions ne permettent pas d'imiter. Si l'adaptation aide déjà ici, elle aidera davantage à cette échelle.

## Travaux récents (2023–2025)

- **Niklaus et al., *Automatic Anonymization of Swiss Federal Supreme Court Rulings* (NLLP @ EMNLP 2023)** : sur
  les arrêts du Tribunal fédéral suisse, des modèles BERT pré-entraînés sur du texte juridique du domaine gagnent
  « plus de 5 % » de F1 ; la combinaison règles (expressions régulières) + modèle reste la meilleure. Ce sont les
  deux choix de ce dépôt : un modèle de langage du domaine (CamemBERT, ou les embeddings Flair ré-entraînés de la
  Cour) et un post-traitement par règles qui complète le modèle.
- **Nyffenegger, Stürmer, Niklaus, *Anonymity at Risk?* (Findings of NAACL 2024)** : un grand modèle de langage
  seul ré-identifie très rarement les personnes d'une décision pseudonymisée ; avec un accès à quelques articles de
  presse, GPT-4 en retrouve 5 sur 7. Le risque vient du croisement avec des sources externes, pas des seules
  mentions oubliées. D'où l'intérêt de pseudonymiser aussi les localités, dates et identifiants, pas seulement les
  noms.
- **Lin et al. (EMNLP 2020) et Ghaddar et al. (TACL 2021), *name regularity bias*** : les modèles de NER, y compris
  BERT, s'appuient trop sur la forme des noms et pas assez sur le contexte ; permuter les mentions dans le jeu de
  test fait chuter les scores. C'est le piège rencontré ici (dev 0,99 / rappel noms 0,13 en test avec un pool de
  25 noms), corrigé par des pools larges et disjoints (`pools.py`) et des pseudonymes jamais vus en test.
- **Kamath & Vajjala, *Does synthetic data help low-resource NER?* (2025)** : des données générées par LLM
  (GPT-4, Llama 3.1) pour 11 langues peu dotées ; les modèles entraînés sur synthétique seul restent en dessous
  du réel annoté, et 100 exemples réels battent souvent des milliers d'exemples synthétiques. Même conclusion que
  Dai & Adel : le réel annoté est ce qui compte, le synthétique complète.
- **Oh et al., *Subject-level Inference for Realistic Text Anonymization Evaluation* (2025)** : les métriques par
  mention (rappel des entités, nos « fuites ») ne suffisent pas ; même avec plus de 90 % des mentions masquées,
  une personne peut rester identifiable par inférence sur le reste du texte. Notre compteur de fuites est donc une
  borne basse du risque, pas une mesure de ré-identification.
- **Rapport sur l'évolution de l'open data des décisions de justice (ministère de la Justice, juillet 2025)** :
  environ 1,3 million de décisions sur Judilibre (≈ 550 000 Cour de cassation, ≈ 500 000 cours d'appel,
  ≈ 225 000 tribunaux judiciaires), près de 2 millions par an à terme ; pseudonymisation automatique puis relecture
  ciblée selon l'importance et la sensibilité de la décision. Le groupe de travail propose d'occulter aussi les
  noms des magistrats, greffiers et avocats. Si cette proposition est retenue, les catégories `professionnelAvocat`
  et `professionnelMagistratGreffier`, aujourd'hui détectées mais laissées en clair, deviendraient à remplacer :
  la chaîne de ce dépôt le permet sans réentraînement (règle de `pseudonymise.py`).

## Ce que la Cour de cassation a publié sur son propre moteur

Il n'existe pas d'article scientifique récent sur le moteur lui-même (code et modèles fermés « pour des raisons
de confidentialité »). Les sources sont institutionnelles et journalistiques.

- **Cour de cassation, *Préparer la Cour de cassation de demain — Cour de cassation et intelligence artificielle*
  (rapport du groupe de travail, avril 2025)** : le moteur associe un modèle de NER et un modèle de langage
  entraîné sur « près de deux millions d'arrêts » ; un nouveau modèle sur GPU est en service depuis 2024 ; un
  second algorithme, dit de *fiabilité*, signale les décisions dont le résultat automatique est douteux pour
  relecture humaine ; l'équipe d'annotation compte 20 personnes et le modèle est réentraîné régulièrement.
  Judilibre dépasse le million de décisions, avec environ un million de plus par an attendu à partir de
  décembre 2025. Le rapport propose une grille d'évaluation des cas d'usage en cinq critères (éthique, juridique,
  fonctionnel, technique, économique), privilégie des modèles pré-entraînés affinés sur du matériel modeste,
  l'hébergement interne et le contrôle humain à chaque étape. Le `check_needed` + `checklist` de ce dépôt est
  une version simple de l'algorithme de fiabilité ; le registre et le test de non-régression répondent au
  « réentraînement régulier ».
- **Cour de cassation, *La numérisation et l'anonymisation des arrêts* (formation AHJUCAF, 23 avril 2025)** :
  liste des catégories traitées — noms des parties et des tiers, noms des professionnels, adresses, références
  cadastrales, immatriculations, comptes bancaires, identifiants INSEE, téléphone / courriel, dates de naissance,
  de décès et de mariage, personnes morales ; relecture ciblée des décisions les plus sensibles, plus un tirage
  aléatoire des autres ; calendrier : tribunaux judiciaires depuis décembre 2023, prud'hommes en 2025, pénal à
  partir de 2026. Par rapport à cette liste, ce dépôt traite par règles (`numeroIdentifiant`) le numéro de sécurité sociale,
  le téléphone, l'immatriculation et le SIREN ; il manque les références cadastrales, les comptes bancaires, le
  courriel, et les dates de décès et de mariage : c'est la première extension à faire.
- **Next, *Comment la Cour de cassation a créé son outil de pseudonymisation utilisant le machine learning*
  (M. Clavey, 10 mai 2023)**, d'après l'enquête de terrain de Camille Girard-Chanudet (thèse EHESS, 2023) :
  équipe de trois développeurs, deux data scientists et un designer entre 2019 et 2021 (programme EIG d'Etalab) ;
  une quinzaine d'annotatrices ; l'outil précédent, à base de règles (2018–2019), avait un taux d'erreur
  « s'approchant de 5 % » ; modèle de langage Byte-Pair + Flair, corrections déterministes après prédiction.
- **C. Girard-Chanudet, entretien Actu-Juridique (24 février 2025)** : le modèle « ne peut pas gérer les cas
  nouveaux ou imprévus » ; il faut toujours des personnes pour le corriger. C'est l'argument pour mesurer sur du
  réel varié (Judilibre réparti sur plusieurs périodes) et garder la relecture dans la chaîne.

## Les autres périmètres IA de la Cour, et ce qui a été publié dessus

Le poste prévoit un rôle de renfort sur d'autres outils. Trois travaux publiés décrivent ces périmètres.

- **Pré-orientation des pourvois** — Douka, Abdine, Vazirgiannis, El Hamdani, Restrepo Amariles, *JuriBERT: A
  Masked-Language Model Adaptation for French Legal Text* (NLLP @ EMNLP 2021, École polytechnique / HEC) : modèles
  BERT pré-entraînés sur 6,3 Go de texte juridique (Légifrance + décisions et mémoires de la Cour de cassation), de
  6 M à 110 M de paramètres. Deux tâches fournies par la Cour : affecter un mémoire ampliatif à une chambre / section
  (8 classes, exactitude 84,0 pour JuriBERT Small contre 83,2 pour CamemBERT base) et à une matière (148 classes,
  ~72). Un petit modèle du domaine fait aussi bien qu'un grand modèle générique, pour une fraction du coût — la même
  logique que le modèle de langage Flair ré-entraîné sur les décisions pour la pseudonymisation. L'algorithme de
  pré-orientation en production (> 90 % selon la Cour, vérifié par des juristes) relève de cette famille.
- **Titrage et similarité entre décisions** — Charmet, Cherichi, Allain, Czerwinska, Fouret, Sagot, Bawden, *Complex
  Labelling and Similarity Prediction in Legal Texts: Automatic Analysis of France's Court of Cassation Rulings*
  (LREC 2022, Inria ALMAnaCH + Cour de cassation + Lab IA) : génération des titrages (séquences de mots-clés) à partir
  du sommaire, en traduction automatique (Fairseq, BPE 8 k), puis similarité entre paires de décisions à partir du
  texte, du sommaire et des titrages, pour repérer des divergences de jurisprudence. Modèles publiés sur Hugging Face
  (`CCASS-auto-titrages-base`). Trois des auteurs sont de la Cour — dont le rapporteur du rapport IA d'avril 2025.
- **Citations implicites dans les décisions** — Floro, Dhorasoo, Pellez, Holzenberger, *Where Experts Disagree,
  Models Fail: Detecting Implicit Legal Citations in French Court Decisions* (2026, Télécom Paris) : 182 000 décisions
  de première instance tirées de **Judilibre** (déc. 2023 – juil. 2025), 1 015 paires annotées par trois juristes,
  accord brut 66,6 % (κ = 0,33). Meilleur F1 0,70 ; deux tiers des faux positifs tombent sur les cas où les juristes
  eux-mêmes ne sont pas d'accord. Leçon pour tout périmètre : mesurer l'accord inter-annotateurs avant de juger le
  modèle, et trancher les cas ambigus dans le guide d'annotation.

Ce que ces trois travaux ont en commun avec la pseudonymisation : du texte juridique français long, des étiquettes
définies avec des juristes, un modèle de domaine modeste qui bat le générique, et une évaluation dont la difficulté
vient des données plus que des modèles. C'est le même métier.

## Limites et suite

- Le corpus d'entraînement est **synthétique** : il valide la chaîne et compare les architectures à budget égal, pas
  la performance réelle. Sur de vraies décisions, les difficultés viennent des formes non prévues (noms composés,
  civilités absentes, OCR, paragraphes de 2 000 mots).
- Pour mesurer cet écart sans accès aux décisions brutes : les décisions **publiées** (API Judilibre) portent les
  marques du moteur de la Cour (`[E] [A]-[M]`, `[Adresse 1]`, `[Localité 2]`, `[Date naissance 1] 1975`). En
  ré-injectant des valeurs synthétiques cohérentes à la place des marques, on obtient des décisions réelles annotées
  aux positions exactes, pour les catégories masquées (`judilibre.py`). Ce n'est pas un corpus de référence humain
  — les professionnels en clair n'y sont pas annotés, et les valeurs restent celles de nos listes — mais la
  mise en page, les formulations et les enchaînements sont ceux de la production.
- Le passage à l'échelle de la Cour (modèle de langage ré-entraîné sur le corpus, byte-pair embeddings,
  183 k phrases annotées) est la différence majeure avec cette maquette.
- Pistes : apprentissage actif à partir des corrections (`check_needed` → relecture → ré-entraînement),
  calibration de la confiance par entité, similarité documentaire pour retrouver des décisions proches
  (déjà possible via `more_like_this` sur le texte pseudonymisé).
