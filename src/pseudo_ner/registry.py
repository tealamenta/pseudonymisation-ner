"""Registre des modèles : empreinte, scores d'évaluation, seuil de non-régression.

``models/registry.json`` retient pour chaque modèle entraîné : le chemin, l'empreinte
SHA-256 de ses fichiers, les scores sur le jeu de test et la date. Le test de
non-régression (``tests/test_regression.py``) recharge le modèle enregistré et
vérifie que son F1 ne descend pas sous le seuil inscrit — c'est le garde-fou
avant toute mise en production.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

REGISTRY = Path("models") / "registry.json"
CURRENT = Path("models") / "current.json"  # le modèle promu, celui que l'on sert


def fingerprint(path: Path) -> str:
    """SHA-256 de tous les fichiers d'un répertoire (ou d'un fichier), par ordre de nom."""
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for file in files:
        digest.update(str(file.relative_to(path if path.is_dir() else path.parent)).encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def version_of(name: str, entry: dict[str, Any]) -> str:
    """Identifiant lisible d'un modèle : nom, date d'enregistrement, début d'empreinte."""
    return f"{name}-{entry['recorded_on']}-{entry['sha256'][:8]}"


def load(path: Path = REGISTRY) -> dict[str, Any]:
    """Lit un registre ; les entrées écrites avant l'ajout du champ ``version`` le reçoivent à la lecture."""
    if not path.exists():
        return {}
    reg: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for name, entry in reg.items():
        if isinstance(entry, dict) and "sha256" in entry and "recorded_on" in entry:
            entry.setdefault("version", version_of(name, entry))
    return reg


def record(
    name: str, model_dir: Path, scores: dict[str, Any], extra: dict[str, Any] | None = None, path: Path = REGISTRY
) -> dict[str, Any]:
    """Inscrit (ou remplace) l'entrée d'un modèle."""
    reg = load(path)
    sha = fingerprint(model_dir)
    today = dt.date.today().isoformat()
    entry = {
        "path": str(model_dir),
        "sha256": sha,
        "version": f"{name}-{today}-{sha[:8]}",  # identifiant lisible : nom, date, début d'empreinte
        "recorded_on": today,
        "micro_f1": scores["micro"]["f1"],
        "micro_recall": scores["micro"]["recall"],
        "per_label_f1": {k: v["f1"] for k, v in scores["per_label"].items()},
        "min_f1": round(scores["micro"]["f1"] - 0.02, 4),  # seuil de non-régression : 2 points de marge
        **(extra or {}),
    }
    reg[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def verify(name: str, path: Path = REGISTRY) -> str:
    """Vérifie que les fichiers du modèle sont ceux enregistrés ; renvoie l'empreinte."""
    entry = load(path)[name]
    actual = fingerprint(Path(entry["path"]))
    if actual != entry["sha256"]:
        raise RuntimeError(f"{name} : empreinte différente du registre (attendue {entry['sha256'][:12]}…, obtenue {actual[:12]}…)")
    return actual


def promote(name: str, path: Path = REGISTRY, current: Path = CURRENT) -> dict[str, Any]:
    """Désigne le modèle à servir, après vérification de l'empreinte et du seuil.

    ``models/current.json`` garde aussi l'entrée précédente, pour ``rollback``.
    """
    entry = load(path)[name]
    verify(name, path)
    if entry["micro_f1"] < entry["min_f1"]:
        raise RuntimeError(f"{name} : F1 {entry['micro_f1']} sous le seuil {entry['min_f1']}")
    previous = load(current) or None
    if previous:
        previous.pop("previous", None)
    chosen = {"model": name, **entry, "promoted_on": dt.date.today().isoformat(), "previous": previous}
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")
    return chosen


def rollback(path: Path = REGISTRY, current: Path = CURRENT) -> dict[str, Any]:
    """Remet le modèle promu précédent (après vérification de son empreinte) ; échoue s'il n'y en a pas."""
    chosen = load(current)
    previous = chosen.get("previous")
    if not previous:
        raise RuntimeError("aucun modèle précédent dans models/current.json")
    verify(previous["model"], path)
    restored = {**previous, "promoted_on": dt.date.today().isoformat(), "previous": None}
    current.write_text(json.dumps(restored, indent=2, ensure_ascii=False), encoding="utf-8")
    return restored
