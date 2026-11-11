"""Évaluation : précision, rappel, F1 par catégorie et micro-moyenne, sur les entités en caractères.

Une entité prédite compte comme correcte si ses bornes **et** sa catégorie sont
exactement celles d'une entité de référence (critère strict, celui des
campagnes CoNLL et du moteur de la Cour). Pour la pseudonymisation, le rappel
est la métrique critique : une entité manquée est une donnée personnelle
publiée ; une fausse alerte n'est qu'un mot masqué en trop.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pseudo_ner.categories import Entity


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class Report:
    """Scores par catégorie et globaux, avec les erreurs pour l'analyse."""

    per_label: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    micro: Counts = field(default_factory=Counts)
    missed: list[dict] = field(default_factory=list)  # faux négatifs (les plus graves)
    spurious: list[dict] = field(default_factory=list)  # faux positifs
    per_doc: list[Counts] = field(default_factory=list)  # pour l'intervalle de confiance

    def bootstrap_f1(self, n: int = 1000, seed: int = 0) -> tuple[float, float]:
        """Intervalle à 95 % du F1 micro par bootstrap *sur les documents* (tirage avec remise).

        Les entités d'une même décision ne sont pas indépendantes (même nom répété 40 fois) :
        rééchantillonner les entités donnerait un intervalle faussement étroit.
        """
        if not self.per_doc:
            return (self.micro.f1, self.micro.f1)
        rng = random.Random(seed)
        docs = self.per_doc
        values = []
        for _ in range(n):
            tp = fp = fn = 0
            for _ in docs:
                c = docs[rng.randrange(len(docs))]
                tp, fp, fn = tp + c.tp, fp + c.fp, fn + c.fn
            values.append(Counts(tp, fp, fn).f1)
        values.sort()
        return (values[int(0.025 * n)], values[min(n - 1, int(0.975 * n))])

    def to_dict(self) -> dict:
        lo, hi = self.bootstrap_f1()
        return {
            "micro": self.micro.to_dict(),
            "micro_f1_ci95": [round(lo, 4), round(hi, 4)],
            "documents": len(self.per_doc),
            "per_label": {k: v.to_dict() for k, v in sorted(self.per_label.items())},
            "missed": self.missed[:200],
            "spurious": self.spurious[:200],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def table(self) -> str:
        """Tableau Markdown."""
        rows = ["| Catégorie | P | R | F1 | support |", "|---|---|---|---|---|"]
        for label, c in sorted(self.per_label.items()):
            rows.append(f"| {label} | {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} | {c.tp + c.fn} |")
        m = self.micro
        rows.append(f"| **micro** | **{m.precision:.3f}** | **{m.recall:.3f}** | **{m.f1:.3f}** | {m.tp + m.fn} |")
        lo, hi = self.bootstrap_f1()
        rows.append(f"\nF1 micro : {m.f1:.3f}, intervalle à 95 % [{lo:.3f} – {hi:.3f}] (bootstrap sur {len(self.per_doc)} documents)")
        return "\n".join(rows)


def evaluate(gold: list[list[Entity]], predicted: list[list[Entity]], ids: list[str] | None = None) -> Report:
    """Compare, document par document, les entités prédites aux entités de référence."""
    report = Report()
    ids = ids or [str(i) for i in range(len(gold))]
    for doc_id, g_list, p_list in zip(ids, gold, predicted, strict=True):
        g = {(e.start, e.end, e.label) for e in g_list}
        p = {(e.start, e.end, e.label) for e in p_list}
        report.per_doc.append(Counts(len(g & p), len(p - g), len(g - p)))
        for key in g & p:
            report.per_label[key[2]].tp += 1
            report.micro.tp += 1
        for key in g - p:
            report.per_label[key[2]].fn += 1
            report.micro.fn += 1
            report.missed.append(
                {
                    "doc": doc_id,
                    "label": key[2],
                    "text": next(e.text for e in g_list if (e.start, e.end, e.label) == key),
                }
            )
        for key in p - g:
            report.per_label[key[2]].fp += 1
            report.micro.fp += 1
            report.spurious.append(
                {
                    "doc": doc_id,
                    "label": key[2],
                    "text": next(e.text for e in p_list if (e.start, e.end, e.label) == key),
                }
            )
    return report


def leaked(gold: list[list[Entity]], predicted: list[list[Entity]], labels: frozenset[str]) -> list[dict]:
    """Entités de référence à masquer qui ne sont pas couvertes par une entité masquée.

    C'est le test qui compte pour la publication : indépendamment des scores
    (bornes et catégories exactes), chaque mention à masquer doit être recouverte
    par une entité d'une catégorie masquée — peu importe laquelle, et peu importe
    que les bornes débordent. Une mention non couverte est une donnée personnelle
    publiée.
    """
    found = []
    for i, (g_list, p_list) in enumerate(zip(gold, predicted, strict=True)):
        masked = [p for p in p_list if p.label in labels]
        for e in g_list:
            if e.label in labels and not any(p.start <= e.start and e.end <= p.end for p in masked):
                found.append({"doc": i, "label": e.label, "text": e.text})
    return found
