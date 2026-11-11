"""Tokenisation et conversion entre positions en caractères et étiquettes BIO.

Le tokenizer est celui de spaCy pour le français (``fr`` vierge, sans modèle),
comme dans le moteur de la Cour. Les entités sont décrites en caractères
(``start``, ``end``) ; les modèles de séquence travaillent en tokens BIO ; on
convertit dans les deux sens, par phrase.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.language import Language

from pseudo_ner.categories import Entity


@lru_cache(maxsize=1)
def tokenizer() -> Language:
    """Pipeline spaCy français réduit au tokenizer et à la segmentation en phrases."""
    nlp = spacy.blank("fr")
    nlp.add_pipe("sentencizer")
    return nlp


@dataclass
class Token:
    """Un token avec ses positions dans le texte d'origine."""

    text: str
    start: int
    end: int


@dataclass
class Sentence:
    """Une phrase tokenisée, étiquetée en BIO."""

    tokens: list[Token]
    tags: list[str]

    @property
    def words(self) -> list[str]:
        return [t.text for t in self.tokens]


def sentences(text: str, entities: list[Entity] | None = None) -> list[Sentence]:
    """Découpe le texte en phrases de tokens ; étiquette BIO d'après les entités (en caractères)."""
    doc = tokenizer()(text)
    spans = sorted(entities or [])
    result: list[Sentence] = []
    for sent in doc.sents:
        tokens = [Token(t.text, t.idx, t.idx + len(t.text)) for t in sent if not t.is_space]
        if not tokens:
            continue
        tags = ["O"] * len(tokens)
        for ent in spans:
            inside = [i for i, t in enumerate(tokens) if t.start < ent.end and ent.start < t.end]
            for k, i in enumerate(inside):
                tags[i] = ("B-" if k == 0 else "I-") + ent.label
        result.append(Sentence(tokens, tags))
    return result


def spans_from_tags(tokens: list[Token], tags: list[str], text: str, source: str = "NER model") -> list[Entity]:
    """Reconstitue les entités (en caractères) à partir d'une séquence BIO."""
    entities: list[Entity] = []
    current: tuple[int, int, str] | None = None  # start, end, label
    for token, tag in zip(tokens, tags, strict=True):
        if tag.startswith("B-") or (tag.startswith("I-") and (current is None or current[2] != tag[2:])):
            if current:
                entities.append(Entity(current[0], current[1], current[2], text[current[0] : current[1]], source))
            current = (token.start, token.end, tag[2:])
        elif tag.startswith("I-") and current:
            current = (current[0], token.end, current[2])
        else:
            if current:
                entities.append(Entity(current[0], current[1], current[2], text[current[0] : current[1]], source))
            current = None
    if current:
        entities.append(Entity(current[0], current[1], current[2], text[current[0] : current[1]], source))
    return entities
