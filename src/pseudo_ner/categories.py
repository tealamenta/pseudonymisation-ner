"""Catégories d'entités à pseudonymiser.

Les libellés reprennent ceux du moteur public de la Cour de cassation
(``Cour-de-cassation/moteurNER`` et ``pseudonymisation-api``) pour que les
sorties soient comparables : ``personnePhysiqueNom``, ``personnePhysiquePrenom``,
``dateDeNaissance``, ``personneMorale``, ``adresse``, ``localite``,
``professionnelAvocat``, ``professionnelMagistratGreffier``. S'y ajoute
``numeroIdentifiant`` (numéros normalisés détectés par expressions régulières,
comme dans leur post-traitement).
"""

from __future__ import annotations

from dataclasses import dataclass

CATEGORIES: dict[str, str] = {
    "personnePhysiquePrenom": "prénom d'une partie ou d'un tiers (personne physique)",
    "personnePhysiqueNom": "nom de famille d'une partie ou d'un tiers",
    "dateDeNaissance": "date de naissance d'une personne physique",
    "personneMorale": "société, association ou organisme partie à l'affaire",
    "adresse": "adresse postale (numéro, voie)",
    "localite": "commune ou lieu-dit permettant de localiser une personne",
    "professionnelAvocat": "nom ou prénom d'un avocat",
    "professionnelMagistratGreffier": "nom ou prénom d'un magistrat ou d'un greffier",
    "numeroIdentifiant": "numéro normalisé : sécurité sociale, téléphone, immatriculation, SIREN",
}

# Catégories apprises par les modèles ; numeroIdentifiant est traité par règles.
MODEL_LABELS: tuple[str, ...] = tuple(c for c in CATEGORIES if c != "numeroIdentifiant")

# Catégories qui doivent être remplacées dans le texte publié (toutes sauf la localité,
# que la Cour conserve en général, et les professionnels, non pseudonymisés par défaut).
PSEUDONYMISED_BY_DEFAULT: frozenset[str] = frozenset(
    {"personnePhysiquePrenom", "personnePhysiqueNom", "dateDeNaissance", "adresse", "numeroIdentifiant"}
)


@dataclass(frozen=True, order=True)
class Entity:
    """Une mention à pseudonymiser, repérée par ses positions en caractères."""

    start: int
    end: int
    label: str
    text: str
    source: str = "NER model"  # ou "postprocess", comme dans l'API publique

    def to_dict(self) -> dict:
        """Forme JSON identique à celle de ``pseudonymisation-api``."""
        return {"text": self.text, "start": self.start, "end": self.end, "label": self.label, "source": self.source}

    def overlaps(self, other: Entity) -> bool:
        """Vrai si les deux mentions se chevauchent."""
        return self.start < other.end and other.start < self.end
