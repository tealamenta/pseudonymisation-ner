"""Post-traitements déterministes et rapport de vérification.

Reprend l'esprit des règles publiées par la Cour de cassation (``moteurNER``) :

1. **Propagation** : un nom ou prénom de personne physique détecté au moins
   deux fois est annoté à toutes ses occurrences (un modèle rate parfois une
   mention isolée, par exemple « Mme Roussel » sans prénom).
2. **Numéros normalisés** par expressions régulières : sécurité sociale,
   téléphone, immatriculation, SIREN.
3. **Rapport de vérification** (``check_needed`` + ``checklist``) : les cas
   douteux sont signalés aux relecteurs plutôt que tranchés en silence —
   nom très court, prénoms proches (Thibaut / Thibault), mention étiquetée
   tantôt partie tantôt professionnel.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pseudo_ner.categories import Entity

PERSON_LABELS = {"personnePhysiqueNom", "personnePhysiquePrenom"}
PRO_LABELS = {"professionnelAvocat", "professionnelMagistratGreffier"}

PATTERNS: dict[str, re.Pattern[str]] = {
    "securite_sociale": re.compile(r"\b[12] ?\d{2} ?(0[1-9]|1[0-2]) ?\d{2} ?\d{3} ?\d{3} ?\d{2}\b"),
    "telephone": re.compile(r"\b0[1-9](?: \d{2}){4}\b"),
    "immatriculation": re.compile(r"\b[A-HJ-NP-TV-Z]{2}-\d{3}-[A-HJ-NP-TV-Z]{2}\b"),
    "siren": re.compile(r"\b\d{3} \d{3} \d{3}\b"),
}


@dataclass
class Checked:
    """Résultat du post-traitement : entités complétées et doutes à relire."""

    entities: list[Entity]
    checklist: list[str] = field(default_factory=list)

    @property
    def check_needed(self) -> bool:
        return bool(self.checklist)

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in sorted(self.entities)],
            "check_needed": self.check_needed,
            "checklist": self.checklist,
        }


def _add_if_free(entities: list[Entity], candidate: Entity) -> bool:
    if any(candidate.overlaps(e) for e in entities):
        return False
    entities.append(candidate)
    return True


def propagate_names(text: str, entities: list[Entity], min_count: int = 2) -> list[Entity]:
    """Règle 1 : toute occurrence d'un nom/prénom vu ``min_count`` fois reçoit la même étiquette."""
    result = list(entities)
    counts = Counter((e.text, e.label) for e in entities if e.label in PERSON_LABELS | PRO_LABELS)
    for (value, label), n in counts.items():
        if n < min_count or len(value) < 3:
            continue
        for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", text):
            _add_if_free(result, Entity(match.start(), match.end(), label, value, "postprocess"))
    return result


def regex_identifiers(text: str, entities: list[Entity]) -> list[Entity]:
    """Règle 2 : numéros normalisés."""
    result = list(entities)
    for pattern in PATTERNS.values():
        for match in pattern.finditer(text):
            _add_if_free(result, Entity(match.start(), match.end(), "numeroIdentifiant", match.group(0), "postprocess"))
    return result


def doubts(entities: list[Entity]) -> list[str]:
    """Règle 3 : ce qu'un relecteur doit regarder."""
    notes: list[str] = []
    seen: set[str] = set()
    by_text: dict[str, set[str]] = defaultdict(set)
    for e in entities:
        by_text[e.text.casefold()].add(e.label)
        if e.label in PERSON_LABELS and len(e.text) <= 2 and e.text not in seen:
            notes.append(f"nom ou prénom très court : « {e.text} »")
            seen.add(e.text)
    for value, labels in by_text.items():
        if labels & PERSON_LABELS and labels & PRO_LABELS:
            notes.append(f"« {value} » est étiqueté à la fois partie et professionnel")
    firstnames = sorted({e.text for e in entities if e.label == "personnePhysiquePrenom"})
    for i, a in enumerate(firstnames):
        for b in firstnames[i + 1 :]:
            if a != b and _close(a, b):
                notes.append(f"prénoms proches, possible variante orthographique : « {a} » / « {b} »")
    return notes


def _close(a: str, b: str) -> bool:
    """Distance d'édition ≤ 1 (Thibaut / Thibault)."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


def postprocess(text: str, entities: list[Entity]) -> Checked:
    """Applique les trois règles."""
    result = regex_identifiers(text, propagate_names(text, entities))
    return Checked(sorted(result), doubts(result))
