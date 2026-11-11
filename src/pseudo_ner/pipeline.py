"""La chaîne complète : modèle → post-traitement → pseudonymisation, dans le format de l'API publique."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pseudo_ner.categories import PSEUDONYMISED_BY_DEFAULT
from pseudo_ner.models import get
from pseudo_ner.postprocess import Checked, postprocess
from pseudo_ner.pseudonymise import Pseudonymised, pseudonymise


@dataclass
class Result:
    """Sortie d'une décision traitée."""

    checked: Checked
    pseudonymised: Pseudonymised

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.checked.to_dict(),
            "pseudonymised_text": self.pseudonymised.text,
            "replaced": self.pseudonymised.replaced,
        }


class Pipeline:
    """Charge un modèle enregistré et traite des textes."""

    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def load(cls, name: str, model_dir: Path) -> Pipeline:
        return cls(get(name).load(model_dir))

    def run(self, text: str, labels: frozenset[str] = PSEUDONYMISED_BY_DEFAULT) -> Result:
        checked = postprocess(text, self.model.predict(text))
        return Result(checked, pseudonymise(text, checked.entities, labels))
