"""API HTTP, au format de ``Cour-de-cassation/pseudonymisation-api``.

``POST /ner`` reçoit une décision ``{idDocument, text, source?, meta?, categories?}``
et renvoie ``{entities, check_needed, checklist}`` — plus, ici, le texte
pseudonymisé. ``/health`` pour l'exploitation, ``/metrics`` au format Prometheus.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from pseudo_ner import __version__
from pseudo_ner.categories import CATEGORIES, PSEUDONYMISED_BY_DEFAULT
from pseudo_ner.pipeline import Pipeline

MAX_CHARS = 200_000


class Decision(BaseModel):
    """Entrée : même schéma que l'API publique de la Cour."""

    idDocument: int  # noqa: N815 - nom imposé par le schéma public
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    source: str | None = None
    meta: str | None = None
    categories: list[str] | None = None  # catégories à pseudonymiser ; défaut : parties, dates, adresses, numéros


class EntityOut(BaseModel):
    text: str
    start: int
    end: int
    label: str
    source: str


class NerResponse(BaseModel):
    idDocument: int  # noqa: N815
    entities: list[EntityOut]
    check_needed: bool
    checklist: list[str]
    pseudonymised_text: str
    replaced: int
    model: str
    seconds: float


class _Counters:
    requests = 0
    errors = 0
    seconds = 0.0
    entities = 0
    check_needed = 0  # part des décisions envoyées en relecture : premier signal de dérive
    characters = 0


def create_app(model_name: str, model_dir: Path, version: str | None = None) -> FastAPI:
    """Construit l'application avec le modèle chargé une fois.

    ``version`` est l'identifiant du modèle promu (``models/current.json``) : exposé par ``/health``,
    c'est ce qui permet de vérifier, après déploiement, que c'est bien lui qui tourne.
    """
    pipeline = Pipeline.load(model_name, model_dir)
    app = FastAPI(title="pseudonymisation-ner", version=__version__)
    counters = _Counters()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": model_name,
            "model_version": version,
            "version": __version__,
            "categories": list(CATEGORIES),
        }

    @app.post("/ner", response_model=NerResponse)
    def ner(decision: Decision) -> NerResponse:
        started = time.perf_counter()
        counters.requests += 1
        labels = frozenset(decision.categories) if decision.categories else PSEUDONYMISED_BY_DEFAULT
        unknown = labels - set(CATEGORIES)
        if unknown:
            counters.errors += 1
            raise HTTPException(status_code=422, detail=f"catégories inconnues : {sorted(unknown)}")
        result = pipeline.run(decision.text, labels)
        seconds = time.perf_counter() - started
        counters.seconds += seconds
        counters.entities += len(result.checked.entities)
        counters.check_needed += int(result.checked.check_needed)
        counters.characters += len(decision.text)
        return NerResponse(
            idDocument=decision.idDocument,
            entities=[EntityOut(**e.to_dict()) for e in sorted(result.checked.entities)],
            check_needed=result.checked.check_needed,
            checklist=result.checked.checklist,
            pseudonymised_text=result.pseudonymised.text,
            replaced=result.pseudonymised.replaced,
            model=model_name,
            seconds=round(seconds, 4),
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return (
            "# TYPE pseudo_ner_requests_total counter\n"
            f"pseudo_ner_requests_total {counters.requests}\n"
            "# TYPE pseudo_ner_errors_total counter\n"
            f"pseudo_ner_errors_total {counters.errors}\n"
            "# TYPE pseudo_ner_seconds_total counter\n"
            f"pseudo_ner_seconds_total {counters.seconds:.4f}\n"
            "# TYPE pseudo_ner_entities_total counter\n"
            f"pseudo_ner_entities_total {counters.entities}\n"
            "# TYPE pseudo_ner_check_needed_total counter\n"
            f"pseudo_ner_check_needed_total {counters.check_needed}\n"
            "# TYPE pseudo_ner_characters_total counter\n"
            f"pseudo_ner_characters_total {counters.characters}\n"
        )

    return app
