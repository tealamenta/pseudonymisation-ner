"""Trois architectures de NER derrière une même interface.

- ``spacy`` : CNN + transition-based parser de spaCy, rapide, sans GPU — la référence « légère ».
- ``flair`` : BiLSTM-CRF sur embeddings contextuels de caractères (fr-forward / fr-backward),
  l'architecture du moteur public de la Cour de cassation.
- ``camembert`` : Transformer pré-entraîné sur le français, fine-tuné en classification de tokens
  (Hugging Face / PyTorch).

Chaque modèle sait s'entraîner sur des documents annotés, prédire des entités
en caractères, se sauvegarder et se recharger. L'évaluation et le
post-traitement sont communs et indépendants de l'architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pseudo_ner.categories import Entity
from pseudo_ner.synthetic import Document


class NerModel(Protocol):
    """Ce que la chaîne attend d'un modèle."""

    name: str

    def train(self, train: list[Document], dev: list[Document], out_dir: Path, **kwargs: object) -> Path: ...

    def predict(self, text: str) -> list[Entity]: ...

    @classmethod
    def load(cls, path: Path) -> NerModel: ...


def get(name: str) -> Any:
    """Classe de modèle par nom (import paresseux : chaque backend est optionnel)."""
    if name == "spacy":
        from pseudo_ner.models.spacy_model import SpacyNer

        return SpacyNer
    if name == "flair":
        from pseudo_ner.models.flair_model import FlairNer

        return FlairNer
    if name == "camembert":
        from pseudo_ner.models.hf_model import CamembertNer

        return CamembertNer
    raise ValueError(f"modèle inconnu : {name} (spacy, flair, camembert)")
