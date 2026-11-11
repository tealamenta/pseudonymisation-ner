"""Non-régression : chaque modèle enregistré doit rester au-dessus de son seuil sur le jeu de test.

Ce test est le garde-fou de mise en production : il recharge le modèle, vérifie
son empreinte, le fait prédire sur le jeu de test figé et compare au seuil
inscrit dans ``models/registry.json`` (F1 enregistré moins 2 points). Il est
sauté si le modèle ou les données ne sont pas présents (CI sans artefacts).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pseudo_ner import registry, synthetic
from pseudo_ner.categories import PSEUDONYMISED_BY_DEFAULT
from pseudo_ner.evaluate import evaluate, leaked
from pseudo_ner.models import get
from pseudo_ner.postprocess import postprocess

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "synthetic" / "test.jsonl"


def _registered() -> list[str]:
    reg = registry.load(ROOT / "models" / "registry.json")
    return [name for name, e in reg.items() if (ROOT / e["path"]).exists()]


@pytest.mark.skipif(not TEST.exists(), reason="jeu de test absent")
@pytest.mark.parametrize("name", _registered() or ["aucun"])
def test_registered_model_meets_threshold(name):
    if name == "aucun":
        pytest.skip("aucun modèle enregistré")
    reg = registry.load(ROOT / "models" / "registry.json")
    entry = reg[name]
    registry.verify(name, ROOT / "models" / "registry.json") if Path(entry["path"]).is_absolute() else None
    model = get(name.split("-")[0]).load(ROOT / entry["path"])
    docs = synthetic.load(TEST)[:30]
    gold = [d.entities for d in docs]
    pred = [postprocess(d.text, model.predict(d.text)).entities for d in docs]
    report = evaluate(gold, pred)
    assert report.micro.f1 >= entry["min_f1"], f"{name} : F1 {report.micro.f1:.4f} < seuil {entry['min_f1']}"
    assert leaked(gold, pred, PSEUDONYMISED_BY_DEFAULT) == []
