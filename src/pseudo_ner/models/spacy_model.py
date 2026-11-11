"""NER spaCy entraîné depuis zéro sur le tokenizer français."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import spacy
from spacy.training import Example
from spacy.util import minibatch

from pseudo_ner.categories import MODEL_LABELS, Entity
from pseudo_ner.synthetic import Document


def _example(nlp: spacy.language.Language, doc: Document) -> Example:
    entities = [(e.start, e.end, e.label) for e in doc.entities if e.label in MODEL_LABELS]
    return Example.from_dict(nlp.make_doc(doc.text), {"entities": entities})


class SpacyNer:
    """Pipeline ``ner`` de spaCy (fr vierge + NER)."""

    name = "spacy"

    def __init__(self, nlp: spacy.language.Language | None = None) -> None:
        self.nlp = nlp

    def train(self, train: list[Document], dev: list[Document], out_dir: Path, epochs: int = 20, seed: int = 0, **_: object) -> Path:
        random.seed(seed)
        spacy.util.fix_random_seed(seed)
        nlp = spacy.blank("fr")
        nlp.add_pipe("ner")
        ner: Any = nlp.get_pipe("ner")
        for label in MODEL_LABELS:
            ner.add_label(label)
        examples = [_example(nlp, d) for d in train]
        optimizer = nlp.initialize(lambda: examples)
        best, best_path = -1.0, out_dir
        for epoch in range(epochs):
            random.shuffle(examples)
            losses: dict[str, float] = {}
            for batch in minibatch(examples, size=8):
                nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
            self.nlp = nlp
            score = self._dev_f1(dev)
            print(f"[spacy] epoch {epoch + 1}/{epochs} loss={losses.get('ner', 0):.1f} dev_f1={score:.4f}")
            if score > best:
                best = score
                nlp.to_disk(out_dir)
        self.nlp = spacy.load(out_dir)
        return best_path

    def _dev_f1(self, dev: list[Document]) -> float:
        from pseudo_ner.evaluate import evaluate

        gold = [[e for e in d.entities if e.label in MODEL_LABELS] for d in dev]
        pred = [self.predict(d.text) for d in dev]
        return evaluate(gold, pred).micro.f1

    def predict(self, text: str) -> list[Entity]:
        assert self.nlp is not None, "modèle non chargé"
        doc = self.nlp(text)
        return [Entity(e.start_char, e.end_char, e.label_, e.text) for e in doc.ents]

    @classmethod
    def load(cls, path: Path) -> SpacyNer:
        return cls(spacy.load(path))
