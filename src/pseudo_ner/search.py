"""Indexation et recherche des décisions pseudonymisées (ElasticSearch), stockage des résultats (MongoDB).

Deux dépôts de données, deux rôles :

- **MongoDB** conserve le résultat complet du traitement — texte d'origine (accès restreint),
  entités, checklist, texte pseudonymisé, modèle et empreinte — c'est la source de vérité
  et l'historique des relectures.
- **ElasticSearch** indexe le texte **pseudonymisé** et les métadonnées pour la recherche
  plein texte et la similarité documentaire (``more_like_this``), sans jamais indexer
  les données personnelles.

Les deux sont derrière une interface minimale, avec une implémentation en mémoire pour les
tests ; les vrais clients sont choisis par configuration (``docker-compose.yml``).
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Protocol

INDEX = "decisions"
COLLECTION = "decisions"
MAPPING = {
    "settings": {"analysis": {"analyzer": {"fr": {"type": "french"}}}},
    "mappings": {
        "properties": {
            "idDocument": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "fr"},
            "chambre": {"type": "keyword"},
            "date": {"type": "date", "ignore_malformed": True},
            "model": {"type": "keyword"},
            "check_needed": {"type": "boolean"},
            "indexed_at": {"type": "date"},
        }
    },
}


class Store(Protocol):
    """Source de vérité des traitements."""

    def save(self, record: dict[str, Any]) -> str: ...

    def get(self, id_document: str) -> dict[str, Any] | None: ...

    def pending_review(self) -> list[dict[str, Any]]: ...


class Index(Protocol):
    """Recherche sur le texte pseudonymisé."""

    def index(self, id_document: str, text: str, meta: dict[str, Any]) -> None: ...

    def search(self, query: str, size: int = 10) -> list[dict[str, Any]]: ...

    def similar(self, id_document: str, size: int = 5) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------- MongoDB


class MongoStore:
    """Collection ``decisions`` : un document par décision traitée, clé ``idDocument``."""

    def __init__(self, client: Any, database: str = "pseudo_ner") -> None:
        self.col = client[database][COLLECTION]
        self.col.create_index("idDocument", unique=True)
        self.col.create_index("check_needed")

    def save(self, record: dict[str, Any]) -> str:
        record = {**record, "saved_at": dt.datetime.now(dt.UTC).isoformat()}
        self.col.replace_one({"idDocument": record["idDocument"]}, record, upsert=True)
        return str(record["idDocument"])

    def get(self, id_document: str) -> dict[str, Any] | None:
        doc = self.col.find_one({"idDocument": id_document}, {"_id": 0})
        return dict(doc) if doc else None

    def pending_review(self) -> list[dict[str, Any]]:
        return [dict(d) for d in self.col.find({"check_needed": True}, {"_id": 0, "text": 0})]


# ---------------------------------------------------------------- ElasticSearch


class ElasticIndex:
    """Index ``decisions`` : texte pseudonymisé + métadonnées ; recherche et ``more_like_this``."""

    def __init__(self, client: Any, index: str = INDEX) -> None:
        self.es, self.name = client, index
        if not self.es.indices.exists(index=index):
            self.es.indices.create(index=index, **MAPPING)

    def index(self, id_document: str, text: str, meta: dict[str, Any]) -> None:
        body = {"idDocument": id_document, "text": text, "indexed_at": dt.datetime.now(dt.UTC).isoformat(), **meta}
        self.es.index(index=self.name, id=id_document, document=body, refresh="wait_for")

    def search(self, query: str, size: int = 10) -> list[dict[str, Any]]:
        res = self.es.search(index=self.name, size=size, query={"match": {"text": {"query": query, "operator": "and"}}})
        return [_hit(h) for h in res["hits"]["hits"]]

    def similar(self, id_document: str, size: int = 5) -> list[dict[str, Any]]:
        res = self.es.search(
            index=self.name,
            size=size,
            query={
                "more_like_this": {
                    "fields": ["text"],
                    "like": [{"_index": self.name, "_id": id_document}],
                    "min_term_freq": 1,
                    "min_doc_freq": 1,
                }
            },
        )
        return [_hit(h) for h in res["hits"]["hits"]]


def _hit(hit: dict[str, Any]) -> dict[str, Any]:
    src = hit["_source"]
    return {
        "idDocument": src["idDocument"],
        "score": round(hit["_score"], 3),
        "chambre": src.get("chambre"),
        "extract": src["text"][:200],
    }


# ---------------------------------------------------------------- En mémoire (tests, développement)


class MemoryStore:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def save(self, record: dict[str, Any]) -> str:
        self.docs[str(record["idDocument"])] = dict(record)
        return str(record["idDocument"])

    def get(self, id_document: str) -> dict[str, Any] | None:
        return self.docs.get(str(id_document))

    def pending_review(self) -> list[dict[str, Any]]:
        return [d for d in self.docs.values() if d.get("check_needed")]


class MemoryIndex:
    """Recherche par mots et similarité par recouvrement de vocabulaire (Jaccard) — suffisant pour tester la logique."""

    def __init__(self) -> None:
        self.docs: dict[str, tuple[str, dict[str, Any]]] = {}

    @staticmethod
    def _words(text: str) -> set[str]:
        return {w.strip(".,;:()'\"«»").casefold() for w in text.split() if len(w) > 3}

    def index(self, id_document: str, text: str, meta: dict[str, Any]) -> None:
        self.docs[str(id_document)] = (text, meta)

    def search(self, query: str, size: int = 10) -> list[dict[str, Any]]:
        q = self._words(query)
        hits = []
        for id_document, (text, meta) in self.docs.items():
            words = self._words(text)
            if q and q <= words:
                hits.append(
                    {
                        "idDocument": id_document,
                        "score": float(len(q)),
                        "chambre": meta.get("chambre"),
                        "extract": text[:200],
                    }
                )
        return sorted(hits, key=lambda h: -float(str(h["score"])))[:size]

    def similar(self, id_document: str, size: int = 5) -> list[dict[str, Any]]:
        ref = self._words(self.docs[str(id_document)][0])
        hits = []
        for other, (text, meta) in self.docs.items():
            if other == str(id_document):
                continue
            words = self._words(text)
            score = len(ref & words) / len(ref | words) if ref | words else 0.0
            hits.append({"idDocument": other, "score": round(score, 3), "chambre": meta.get("chambre"), "extract": text[:200]})
        return sorted(hits, key=lambda h: -float(str(h["score"])))[:size]


def connect(mongo_url: str | None, es_url: str | None) -> tuple[Store, Index]:
    """Clients réels si les URL sont données, mémoire sinon."""
    store: Store = MemoryStore()
    index: Index = MemoryIndex()
    if mongo_url:
        from pymongo import MongoClient

        store = MongoStore(MongoClient(mongo_url))
    if es_url:
        from elasticsearch import Elasticsearch

        index = ElasticIndex(Elasticsearch(es_url))
    return store, index
