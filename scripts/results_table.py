"""Tableau comparatif des modèles à partir des rapports ``reports/*.json`` (pour le README)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = [
        "| Modèle | F1 micro (modèle seul) | F1 micro (avec post-traitement) | IC 95 % | Rappel | Fuites | s / document |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in sorted((ROOT / "reports").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if "raw" not in r:  # rapport d'exploration, pas d'évaluation
            continue
        raw, post = r["raw"]["micro"], r["postprocessed"]["micro"]
        ci = r["postprocessed"].get("micro_f1_ci95")
        cells = [
            path.stem,
            f"{raw['f1']:.3f}",
            f"{post['f1']:.3f}",
            f"[{ci[0]:.3f} – {ci[1]:.3f}]" if ci else "—",
            f"{post['recall']:.3f}",
            str(r["leaks"]["count"]),
            f"{r['seconds_per_doc']:.3f}",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
