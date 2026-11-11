"""Remplacement des entités dans le texte, de façon cohérente.

Une même valeur reçoit toujours le même pseudonyme dans une décision
(« M. [B] » partout pour le même nom), ce qui préserve la lisibilité du
raisonnement juridique. Par défaut on pseudonymise les parties, leurs dates
de naissance, adresses et numéros ; les professionnels et les localités sont
conservés, comme dans la pratique de la Cour. Le choix est paramétrable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pseudo_ner.categories import PSEUDONYMISED_BY_DEFAULT, Entity

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _letters(i: int) -> str:
    """0 → A, 25 → Z, 26 → AA…"""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = ALPHABET[r] + out
    return out


@dataclass
class Pseudonymised:
    """Texte de sortie et table de correspondance (à conserver hors publication)."""

    text: str
    mapping: dict[tuple[str, str], str] = field(default_factory=dict)
    replaced: int = 0


def pseudonymise(text: str, entities: list[Entity], labels: frozenset[str] = PSEUDONYMISED_BY_DEFAULT) -> Pseudonymised:
    """Remplace chaque entité des catégories demandées par un pseudonyme stable."""
    mapping: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}
    pieces: list[str] = []
    cursor = 0
    replaced = 0
    for e in sorted(entities):
        if e.label not in labels or e.start < cursor:
            continue
        key = (e.label, e.text.casefold())
        if key not in mapping:
            # une seule suite de lettres pour tous les noms de personnes : « [E] [A]-[M] » comme à la Cour,
            # jamais « [A] [A] » (prénom et nom indistincts) ; les autres catégories ont chacune leur numérotation
            series = "lettres" if e.label in LETTER_LABELS else e.label
            n = counters.get(series, 0)
            counters[series] = n + 1
            mapping[key] = _placeholder(e.label, n)
        pieces.append(text[cursor : e.start])
        pieces.append(mapping[key])
        cursor = e.end
        replaced += 1
    pieces.append(text[cursor:])
    return Pseudonymised("".join(pieces), mapping, replaced)


LETTER_LABELS = frozenset({"personnePhysiqueNom", "personnePhysiquePrenom", "professionnelAvocat", "professionnelMagistratGreffier"})


def _placeholder(label: str, n: int) -> str:
    if label in LETTER_LABELS:
        return f"[{_letters(n)}]"
    if label == "dateDeNaissance":
        return "[Date naissance " + str(n + 1) + "]"
    if label == "adresse":
        return "[Adresse " + str(n + 1) + "]"
    if label == "localite":
        return "[Localité " + str(n + 1) + "]"
    if label == "personneMorale":
        return f"[Société {_letters(n)}]"
    if label == "numeroIdentifiant":
        return "[Numéro " + str(n + 1) + "]"
    return f"[{label} {n + 1}]"
