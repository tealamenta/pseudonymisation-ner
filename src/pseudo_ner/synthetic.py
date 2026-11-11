"""Corpus synthétique de décisions de justice, annoté par construction.

Aucune vraie décision n'est utilisée : les textes sont assemblés à partir de
gabarits qui imitent la structure d'un arrêt (en-tête, parties, composition,
exposé du litige, motifs, dispositif), et chaque entité insérée est connue
avec sa position exacte. Les noms, villes et sociétés sont tirés de listes
publiques courantes ; les dates, adresses et numéros sont générés.

C'est un corpus de *développement* : il sert à comparer des architectures, à
tester la chaîne de bout en bout et à vérifier la non-régression. Les scores
obtenus dessus ne sont pas comparables à ceux d'un corpus réel annoté.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from pseudo_ner.categories import Entity

PRENOMS_F = [
    "Marie",
    "Sophie",
    "Isabelle",
    "Nathalie",
    "Camille",
    "Catherine",
    "Céline",
    "Valérie",
    "Fatima",
    "Nadia",
    "Sandrine",
    "Aurélie",
    "Élodie",
    "Stéphanie",
    "Émilie",
    "Laetitia",
    "Hélène",
    "Anaïs",
    "Inès",
    "Léa",
    "Chloé",
    "Leïla",
    "Amina",
    "Louise",
    "Martine",
]
PRENOMS_M = [
    "Jean",
    "Pierre",
    "Nicolas",
    "Philippe",
    "Thomas",
    "Julien",
    "Laurent",
    "Antoine",
    "Mohamed",
    "Karim",
    "Olivier",
    "Sébastien",
    "Mathieu",
    "Christophe",
    "Thibault",
    "Thibaut",
    "Guillaume",
    "Vincent",
    "Yann",
    "Romain",
    "Hugo",
    "Lucas",
    "Samir",
    "Youssef",
    "Paul",
    "Jacques",
]
PRENOMS = PRENOMS_F + PRENOMS_M
NOMS = [
    "Martin",
    "Bernard",
    "Dubois",
    "Thomas",
    "Robert",
    "Richard",
    "Petit",
    "Durand",
    "Leroy",
    "Moreau",
    "Simon",
    "Laurent",
    "Lefebvre",
    "Michel",
    "Garcia",
    "David",
    "Bertrand",
    "Roux",
    "Vincent",
    "Fournier",
    "Morel",
    "Girard",
    "André",
    "Lefèvre",
    "Mercier",
    "Dupont",
    "Lambert",
    "Bonnet",
    "François",
    "Martinez",
    "Benali",
    "Haddad",
    "Nguyen",
    "Da Silva",
    "Rossi",
    "Kowalski",
    "Diallo",
    "Traoré",
    "Cohen",
    "Lopez",
    "Fontaine",
    "Chevalier",
    "Robin",
    "Masson",
    "Sanchez",
    "Gérard",
    "Nicolas",
    "Henry",
    "Roussel",
    "Le Gall",
]
VILLES = [
    "Paris",
    "Lyon",
    "Marseille",
    "Toulouse",
    "Nice",
    "Nantes",
    "Strasbourg",
    "Montpellier",
    "Bordeaux",
    "Lille",
    "Rennes",
    "Reims",
    "Le Havre",
    "Saint-Étienne",
    "Toulon",
    "Grenoble",
    "Dijon",
    "Angers",
    "Nîmes",
    "Villeurbanne",
    "Clermont-Ferrand",
    "Le Mans",
    "Aix-en-Provence",
    "Brest",
    "Tours",
    "Amiens",
    "Limoges",
    "Annecy",
    "Perpignan",
    "Metz",
    "Besançon",
    "Orléans",
    "Rouen",
    "Caen",
    "Nancy",
    "Pau",
    "Poitiers",
    "Versailles",
    "Créteil",
    "Bobigny",
]
VOIES = [
    "rue de la République",
    "avenue Victor Hugo",
    "boulevard Gambetta",
    "rue des Lilas",
    "place de la Mairie",
    "chemin des Vignes",
    "impasse du Moulin",
    "rue Jean Jaurès",
    "allée des Tilleuls",
    "route de la Gare",
    "rue Pasteur",
    "avenue de la Liberté",
    "rue du Général Leclerc",
    "quai des Chartrons",
    "rue Nationale",
]
SOCIETES_PREFIX = ["SA", "SARL", "SAS", "SCI", "EURL", "SASU"]
SOCIETES_NOM = [
    "Bâtiment Durable",
    "Transports Réunis",
    "Logistique du Nord",
    "Immobilière du Parc",
    "Électro Services",
    "Agro Distribution",
    "Menuiserie Moderne",
    "Nettoyage Pro",
    "Résidences du Lac",
    "Textile Atlantique",
    "Garage Central",
    "Boulangerie des Halles",
    "Conseil & Stratégie",
    "Médical Plus",
    "Énergie Verte",
]
TRIBUNAUX = ["tribunal judiciaire", "conseil de prud'hommes", "tribunal de commerce", "tribunal de proximité"]
CHAMBRES = ["chambre sociale", "première chambre civile", "chambre commerciale", "deuxième chambre civile"]
MOIS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]


HELD_OUT_NOMS = [
    "Perrin",
    "Morin",
    "Mathieu",
    "Clément",
    "Gauthier",
    "Dumont",
    "Lopes",
    "Fontana",
    "Marchand",
    "Duval",
    "Denis",
    "Dumas",
    "Joly",
    "Gautier",
    "Roger",
    "Roche",
    "Roy",
    "Noël",
    "Meyer",
    "Lucas",
    "Meunier",
    "Jean",
    "Perez",
    "Marchal",
    "Barbier",
    "Brun",
    "Dufour",
    "Moulin",
    "Leclerc",
    "Guerin",
]
HELD_OUT_PRENOMS_F = [
    "Agnès",
    "Brigitte",
    "Claire",
    "Delphine",
    "Florence",
    "Justine",
    "Manon",
    "Océane",
    "Salma",
    "Yasmine",
]
HELD_OUT_PRENOMS_M = [
    "Adrien",
    "Bastien",
    "Cédric",
    "Damien",
    "Étienne",
    "Florian",
    "Kevin",
    "Mehdi",
    "Rayan",
    "Xavier",
]
HELD_OUT_VILLES = [
    "Avignon",
    "Bayonne",
    "Chambéry",
    "Colmar",
    "La Rochelle",
    "Lorient",
    "Mulhouse",
    "Quimper",
    "Troyes",
    "Valence",
]
HELD_OUT_VOIES = [
    "rue des Écoles",
    "avenue Foch",
    "boulevard Carnot",
    "rue Saint-Michel",
    "chemin de la Fontaine",
    "rue Voltaire",
]
HELD_OUT_SOCIETES = [
    "Chauffage Rapide",
    "Pépinières du Sud",
    "Studio Graphique",
    "Carrelages Modernes",
    "Fret Express",
    "Optique Vision",
]


@dataclass
class Pools:
    """Listes de valeurs : disjointes entre entraînement et test pour mesurer la généralisation."""

    prenoms_f: list[str]
    prenoms_m: list[str]
    noms: list[str]
    villes: list[str]
    voies: list[str]
    societes: list[str]


POOLS = {
    "train": Pools(PRENOMS_F, PRENOMS_M, NOMS, VILLES, VOIES, SOCIETES_NOM),
    "test": Pools(HELD_OUT_PRENOMS_F, HELD_OUT_PRENOMS_M, HELD_OUT_NOMS, HELD_OUT_VILLES, HELD_OUT_VOIES, HELD_OUT_SOCIETES),
}


@dataclass
class Person:
    """Une personne physique du dossier."""

    prenom: str
    nom: str
    civilite: str
    naissance: str
    ville: str
    adresse: str

    @property
    def full(self) -> str:
        return f"{self.prenom} {self.nom}"


@dataclass
class Document:
    """Un texte et ses entités, prêt pour l'entraînement ou l'évaluation."""

    id: str
    text: str
    entities: list[Entity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "entities": [e.to_dict() for e in self.entities]}

    @classmethod
    def from_dict(cls, data: dict) -> Document:
        return cls(
            id=data["id"],
            text=data["text"],
            entities=[Entity(e["start"], e["end"], e["label"], e["text"], e.get("source", "gold")) for e in data["entities"]],
        )


class Builder:
    """Assemble un texte en enregistrant la position de chaque entité insérée."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.entities: list[Entity] = []
        self.length = 0

    def add(self, text: str, label: str | None = None) -> None:
        if label:
            self.entities.append(Entity(self.length, self.length + len(text), label, text, "gold"))
        self.parts.append(text)
        self.length += len(text)

    def person(
        self,
        p: Person,
        prenom_label: str = "personnePhysiquePrenom",
        nom_label: str = "personnePhysiqueNom",
        civilite: bool = True,
    ) -> None:
        if civilite:
            self.add(f"{p.civilite} ")
        self.add(p.prenom, prenom_label)
        self.add(" ")
        self.add(p.nom, nom_label)

    def build(self) -> str:
        return "".join(self.parts)


def _date(rng: random.Random, year_min: int, year_max: int) -> str:
    return f"{rng.randint(1, 28)} {rng.choice(MOIS)} {rng.randint(year_min, year_max)}"


def _noisy(rng: random.Random, value: str, noise: float) -> str:
    """Bruit de surface : majuscules (usage des greffes), accents perdus (OCR)."""
    if rng.random() < noise:
        value = value.upper()
    if rng.random() < noise:
        value = value.translate(str.maketrans("éèêëàâîïôûùçÉÈÀÇ", "eeeeaaiiouucEEAC"))
    return value


def _person(rng: random.Random, pools: Pools, noise: float) -> Person:
    civ = rng.choice(["M.", "Mme"])
    return Person(
        prenom=rng.choice(pools.prenoms_f if civ == "Mme" else pools.prenoms_m),
        nom=_noisy(rng, rng.choice(pools.noms), noise),
        civilite=civ,
        naissance=_date(rng, 1945, 2000),
        ville=rng.choice(pools.villes),
        adresse=f"{rng.randint(1, 180)} {rng.choice(pools.voies)}",
    )


def _societe(rng: random.Random, pools: Pools) -> str:
    return f"{rng.choice(SOCIETES_PREFIX)} {rng.choice(pools.societes)}"


def _identifiant(rng: random.Random) -> str:
    kind = rng.choice(["tel", "secu", "plaque", "siren"])
    if kind == "tel":
        return "0" + rng.choice("1679") + " " + " ".join(f"{rng.randint(0, 99):02d}" for _ in range(4))
    if kind == "secu":
        parts = [
            rng.randint(1, 2),
            rng.randint(50, 99),
            rng.randint(1, 12),
            rng.randint(1, 95),
            rng.randint(0, 999),
            rng.randint(0, 999),
            rng.randint(0, 99),
        ]
        return f"{parts[0]} {parts[1]:02d} {parts[2]:02d} {parts[3]:02d} {parts[4]:03d} {parts[5]:03d} {parts[6]:02d}"
    if kind == "plaque":
        letters = "ABCDEFGHJKLMNPRSTVWXYZ"
        return f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randint(100, 999)}-{rng.choice(letters)}{rng.choice(letters)}"
    return f"{rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(100, 999)}"


def generate_document(rng: random.Random, doc_id: str, pools: Pools | None = None, noise: float = 0.1) -> Document:
    """Un arrêt synthétique : parties, avocats, composition, faits, motifs, dispositif."""
    pools = pools or POOLS["train"]
    b = Builder()
    appelant, intime = _person(rng, pools, noise), _person(rng, pools, noise)
    while intime.nom == appelant.nom:
        intime = _person(rng, pools, noise)
    avocat1, avocat2 = _person(rng, pools, noise), _person(rng, pools, noise)
    president, greffier = _person(rng, pools, noise), _person(rng, pools, noise)
    societe = _societe(rng, pools) if rng.random() < 0.6 else None
    ville_cour = rng.choice(pools.villes)
    date_arret = _date(rng, 2015, 2024)

    b.add(f"COUR D'APPEL DE {ville_cour.upper()}\n{rng.choice(CHAMBRES).upper()}\n\nARRÊT DU {date_arret}\n\n")
    b.add(f"Numéro d'inscription au répertoire général : {rng.randint(15, 24)}/{rng.randint(1000, 9999):05d}\n\n")
    b.add("APPELANT :\n\n")
    b.person(appelant)
    b.add(", né")
    b.add("e" if appelant.civilite == "Mme" else "")
    b.add(" le ")
    b.add(appelant.naissance, "dateDeNaissance")
    b.add(" à ")
    b.add(rng.choice(pools.villes), "localite")
    b.add(", demeurant ")
    b.add(appelant.adresse, "adresse")
    b.add(", ")
    b.add(appelant.ville, "localite")
    b.add("\nreprésenté")
    b.add("e" if appelant.civilite == "Mme" else "")
    b.add(" par Me ")
    b.person(avocat1, "professionnelAvocat", "professionnelAvocat", civilite=False)
    b.add(f", avocat au barreau de {rng.choice(pools.villes)}\n\n")

    b.add("INTIMÉ")
    b.add("E" if intime.civilite == "Mme" else "")
    b.add(" :\n\n")
    if societe:
        b.add(societe, "personneMorale")
        b.add(", prise en la personne de son représentant légal, dont le siège social est ")
        b.add(f"{rng.randint(1, 120)} {rng.choice(pools.voies)}", "adresse")
        b.add(", ")
        b.add(rng.choice(pools.villes), "localite")
    else:
        b.person(intime)
        b.add(", né")
        b.add("e" if intime.civilite == "Mme" else "")
        b.add(" le ")
        b.add(intime.naissance, "dateDeNaissance")
        b.add(", demeurant ")
        b.add(intime.adresse, "adresse")
        b.add(", ")
        b.add(intime.ville, "localite")
    b.add("\nreprésenté")
    b.add("e" if (not societe and intime.civilite == "Mme") else "")
    b.add(" par Me ")
    b.person(avocat2, "professionnelAvocat", "professionnelAvocat", civilite=False)
    b.add(f", avocat au barreau de {rng.choice(pools.villes)}\n\n")

    b.add(
        "COMPOSITION DE LA COUR :\n\nEn application des dispositions des articles 805 et 907 du code de procédure civile, "
        "l'affaire a été débattue devant "
    )
    b.person(president, "professionnelMagistratGreffier", "professionnelMagistratGreffier")
    b.add(", président")
    b.add("e" if president.civilite == "Mme" else "")
    b.add(" de chambre, chargé")
    b.add("e" if president.civilite == "Mme" else "")
    b.add(" du rapport.\n\nGreffier, lors des débats : ")
    b.person(greffier, "professionnelMagistratGreffier", "professionnelMagistratGreffier")
    b.add("\n\n")

    # Exposé du litige : mentions répétées, parfois nom seul, parfois sans civilité
    b.add("EXPOSÉ DU LITIGE\n\n")
    partie_adverse = societe if societe else None
    b.add("Par contrat du ")
    b.add(_date(rng, 2010, 2020))
    b.add(", ")
    if partie_adverse:
        b.add(partie_adverse, "personneMorale")
        b.add(" a conclu avec ")
        b.person(appelant)
    else:
        b.person(intime)
        b.add(" a conclu avec ")
        b.person(appelant)
    b.add(f" un contrat de {rng.choice(['travail', 'bail', 'prestation de services', 'vente', 'prêt'])}.\n\n")
    b.add("Par lettre du ")
    b.add(_date(rng, 2018, 2022))
    b.add(", ")
    if rng.random() > noise:
        b.add(appelant.civilite + " ")
    b.add(appelant.nom, "personnePhysiqueNom")
    b.add(f" a saisi le {rng.choice(TRIBUNAUX)} de ")
    b.add(rng.choice(pools.villes), "localite")
    b.add(" aux fins de voir ")
    if partie_adverse:
        b.add("la société ")
        b.add(partie_adverse, "personneMorale")
    else:
        b.add(intime.civilite + " ")
        b.add(intime.nom, "personnePhysiqueNom")
    b.add(
        f" condamné{'e' if partie_adverse or intime.civilite == 'Mme' else ''}"
        f" au paiement de la somme de {rng.randint(1, 80) * 500} euros.\n\n"
    )
    if rng.random() < 0.7:
        b.add("Il est précisé que le numéro de ")
        b.add(rng.choice(["téléphone", "sécurité sociale", "SIREN", "immatriculation du véhicule"]))
        b.add(" mentionné au dossier est ")
        b.add(_identifiant(rng), "numeroIdentifiant")
        b.add(".\n\n")
    b.add(f"Par jugement du {_date(rng, 2020, 2023)}, le tribunal a ")
    b.add(rng.choice(["fait droit à la demande", "débouté le demandeur", "condamné le défendeur", "ordonné une expertise"]))
    b.add(".\n\n")
    b.add(appelant.civilite + " ")
    b.add(appelant.nom, "personnePhysiqueNom")
    b.add(f" a interjeté appel de cette décision le {_date(rng, 2021, 2023)}.\n\n")

    b.add("MOTIFS DE LA DÉCISION\n\n")
    b.add("Sur la demande principale, il résulte des pièces produites que ")
    b.person(appelant, civilite=False)
    b.add(rng.choice([" ne démontre pas", " établit", " justifie"]))
    b.add(
        " "
        + rng.choice(["le manquement allégué", "la réalité du préjudice", "le lien de causalité", "le caractère abusif de la rupture"])
        + ". "
    )
    if rng.random() < 0.5:
        temoin = _person(rng, pools, noise)
        b.add("L'attestation de ")
        b.person(temoin)
        b.add(", demeurant ")
        b.add(temoin.adresse, "adresse")
        b.add(" à ")
        b.add(temoin.ville, "localite")
        b.add(", ne suffit pas à l'établir. ")
    b.add("En conséquence, le jugement sera ")
    b.add(rng.choice(["confirmé", "infirmé", "confirmé en toutes ses dispositions", "infirmé partiellement"]))
    b.add(".\n\n")

    b.add("PAR CES MOTIFS\n\nLa cour, statuant publiquement, par arrêt contradictoire,\n\n")
    b.add(rng.choice(["CONFIRME", "INFIRME"]))
    b.add(" le jugement déféré ;\n\nCONDAMNE ")
    if rng.random() < 0.5:
        b.person(appelant)
    elif partie_adverse:
        b.add("la société ")
        b.add(partie_adverse, "personneMorale")
    else:
        b.person(intime)
    b.add(" aux dépens d'appel ;\n\nLe présent arrêt a été signé par ")
    b.person(president, "professionnelMagistratGreffier", "professionnelMagistratGreffier")
    b.add(", président")
    b.add("e" if president.civilite == "Mme" else "")
    b.add(", et par ")
    b.person(greffier, "professionnelMagistratGreffier", "professionnelMagistratGreffier")
    b.add(", greffier.\n")

    text = b.build()
    for e in b.entities:  # garde-fou : les positions sont exactes
        assert text[e.start : e.end] == e.text, (e, text[e.start : e.end])
    return Document(doc_id, text, sorted(b.entities))


def generate_corpus(n: int, seed: int = 0, pool: str = "train", noise: float = 0.1) -> list[Document]:
    """``n`` documents reproductibles, tirés dans le pool de valeurs demandé."""
    rng = random.Random(seed)
    return [generate_document(rng, f"synth-{pool}-{seed}-{i:04d}", POOLS[pool], noise) for i in range(n)]


def make_splits(n_train: int, n_dev: int, n_test: int, seed: int = 0, noise: float = 0.1) -> dict[str, list[Document]]:
    """Train et dev partagent le pool d'entraînement ; le test n'utilise que des valeurs jamais vues.

    Le test mesure donc la généralisation à des noms, villes et sociétés inconnus —
    la situation réelle d'un moteur de pseudonymisation — et non la mémorisation.
    """
    train = generate_corpus(n_train, seed=seed, pool="train", noise=noise)
    dev = generate_corpus(n_dev, seed=seed + 1, pool="train", noise=noise)
    test = generate_corpus(n_test, seed=seed + 2, pool="test", noise=noise)
    return {"train": train, "dev": dev, "test": test}


def save(docs: list[Document], path: Path) -> Path:
    """JSON Lines, un document par ligne."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    return path


def load(path: Path) -> list[Document]:
    """Relit un fichier JSON Lines."""
    with path.open(encoding="utf-8") as handle:
        return [Document.from_dict(json.loads(line)) for line in handle if line.strip()]
