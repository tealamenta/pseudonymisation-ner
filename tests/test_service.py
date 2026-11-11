"""API, stockage, recherche et registre, avec un modèle factice (sans entraînement)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pseudo_ner import registry, synthetic
from pseudo_ner.api import create_app
from pseudo_ner.categories import MODEL_LABELS, Entity
from pseudo_ner.pipeline import Pipeline
from pseudo_ner.search import MemoryIndex, MemoryStore, MongoStore, connect


class OracleModel:
    """Un « modèle » qui connaît les réponses : sert à tester la chaîne autour du modèle."""

    name = "oracle"

    def __init__(self, docs: list[synthetic.Document]) -> None:
        self.answers = {d.text: [e for e in d.entities if e.label in MODEL_LABELS] for d in docs}

    def predict(self, text: str) -> list[Entity]:
        return [Entity(e.start, e.end, e.label, e.text) for e in self.answers.get(text, [])]


@pytest.fixture(scope="module")
def docs() -> list[synthetic.Document]:
    return synthetic.generate_corpus(6, seed=21)


@pytest.fixture(scope="module")
def client(docs, monkeypatch_module):
    monkeypatch_module.setattr(Pipeline, "load", classmethod(lambda cls, name, path: cls(OracleModel(docs))))
    return TestClient(create_app("oracle", Path("models/none")))


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_health_and_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "pseudo_ner_requests_total 0" in client.get("/metrics").text


def test_ner_endpoint_matches_public_schema(client, docs):
    d = docs[0]
    r = client.post("/ner", json={"idDocument": 42, "text": d.text})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"idDocument", "entities", "check_needed", "checklist", "pseudonymised_text", "replaced"}
    assert body["idDocument"] == 42
    assert {e["label"] for e in body["entities"]} >= {"personnePhysiqueNom", "adresse"}
    assert all(set(e) == {"text", "start", "end", "label", "source"} for e in body["entities"])
    for e in body["entities"]:
        assert d.text[e["start"] : e["end"]] == e["text"]
    # aucune valeur à masquer dans le texte renvoyé
    for e in d.entities:
        if e.label in ("personnePhysiqueNom", "adresse", "dateDeNaissance"):
            assert e.text not in body["pseudonymised_text"]
    assert "pseudo_ner_requests_total 1" in client.get("/metrics").text


def test_ner_rejects_bad_input(client):
    assert client.post("/ner", json={"idDocument": 1, "text": ""}).status_code == 422
    r = client.post("/ner", json={"idDocument": 1, "text": "x", "categories": ["inconnue"]})
    assert r.status_code == 422 and "inconnue" in r.text


def test_ner_custom_categories(client, docs):
    d = docs[1]
    r = client.post("/ner", json={"idDocument": 2, "text": d.text, "categories": ["localite"]})
    body = r.json()
    assert body["replaced"] == sum(e.label == "localite" for e in d.entities)


# ---------------------------------------------------------------- stockage et recherche


def test_memory_store_and_index(docs):
    store, index = connect(None, None)
    assert isinstance(store, MemoryStore) and isinstance(index, MemoryIndex)
    for d in docs:
        store.save({"idDocument": d.id, "check_needed": d.id.endswith("1"), "text": d.text})
        index.index(d.id, d.text, {"chambre": "sociale"})
    assert store.get(docs[0].id)["text"] == docs[0].text
    assert {r["idDocument"] for r in store.pending_review()} == {d.id for d in docs if d.id.endswith("1")}
    hits = index.search("COUR D'APPEL")
    assert len(hits) == len(docs)
    sim = index.similar(docs[0].id, size=2)
    assert len(sim) == 2 and docs[0].id not in {h["idDocument"] for h in sim}


def test_mongo_store_with_mongomock():
    mongomock = pytest.importorskip("mongomock")
    store = MongoStore(mongomock.MongoClient())
    store.save({"idDocument": "a", "check_needed": True, "text": "secret"})
    store.save({"idDocument": "a", "check_needed": False, "text": "secret2"})  # remplace
    assert store.get("a")["text"] == "secret2"
    assert store.pending_review() == []
    store.save({"idDocument": "b", "check_needed": True, "text": "t"})
    pending = store.pending_review()
    assert [p["idDocument"] for p in pending] == ["b"] and "text" not in pending[0]


def test_elastic_index_calls(monkeypatch):
    """Le client ES est simulé : on vérifie la création de l'index, l'indexation et les deux requêtes."""
    from pseudo_ner.search import ElasticIndex

    calls: list[tuple] = []

    class FakeIndices:
        def exists(self, index):
            return False

        def create(self, index, **body):
            calls.append(("create", index, sorted(body)))

    class FakeEs:
        indices = FakeIndices()

        def index(self, index, id, document, refresh):
            calls.append(("index", id, document["text"]))

        def search(self, index, size, query):
            calls.append(("search", list(query)[0]))
            return {"hits": {"hits": [{"_score": 1.5, "_source": {"idDocument": "x", "text": "texte", "chambre": None}}]}}

    es = ElasticIndex(FakeEs())
    es.index("x", "texte pseudonymisé", {"chambre": "sociale"})
    assert es.search("texte")[0]["idDocument"] == "x"
    assert es.similar("x")[0]["score"] == 1.5
    assert [c[0] for c in calls] == ["create", "index", "search", "search"]
    assert calls[0][2] == ["mappings", "settings"] and calls[3][1] == "more_like_this"


# ---------------------------------------------------------------- registre


def test_registry_record_and_verify(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "w.bin").write_bytes(b"poids")
    scores = {"micro": {"f1": 0.95, "recall": 0.96}, "per_label": {"adresse": {"f1": 0.9}}}
    reg = tmp_path / "registry.json"
    entry = registry.record("test", model_dir, scores, path=reg)
    assert entry["min_f1"] == 0.93 and len(entry["sha256"]) == 64
    assert registry.verify("test", path=reg) == entry["sha256"]
    (model_dir / "w.bin").write_bytes(b"autres poids")
    with pytest.raises(RuntimeError, match="empreinte"):
        registry.verify("test", path=reg)
    assert json.loads(reg.read_text())["test"]["per_label_f1"] == {"adresse": 0.9}


def test_registry_promote_current_and_rollback(tmp_path):
    scores = {"micro": {"f1": 0.95, "recall": 0.96}, "per_label": {}}
    reg, cur = tmp_path / "registry.json", tmp_path / "current.json"
    for name, weights in (("spacy", b"v1"), ("flair", b"v2")):
        d = tmp_path / name
        d.mkdir()
        (d / "w.bin").write_bytes(weights)
        registry.record(name, d, scores, path=reg)
    first = registry.promote("spacy", path=reg, current=cur)
    assert first["model"] == "spacy" and first["previous"] is None
    assert first["version"].startswith("spacy-") and first["version"].endswith(first["sha256"][:8])
    second = registry.promote("flair", path=reg, current=cur)
    assert second["previous"]["model"] == "spacy" and registry.load(cur)["model"] == "flair"
    restored = registry.rollback(path=reg, current=cur)
    assert restored["model"] == "spacy" and restored["previous"] is None
    with pytest.raises(RuntimeError, match="aucun modèle précédent"):
        registry.rollback(path=reg, current=cur)
    # un modèle dont les fichiers ont changé ne peut pas être promu
    (tmp_path / "flair" / "w.bin").write_bytes(b"autre")
    with pytest.raises(RuntimeError, match="empreinte"):
        registry.promote("flair", path=reg, current=cur)


def test_health_reports_model_version(docs, monkeypatch_module):
    monkeypatch_module.setattr(Pipeline, "load", classmethod(lambda cls, name, path: cls(OracleModel(docs))))
    client = TestClient(create_app("oracle", Path("models/none"), version="oracle-2026-01-01-deadbeef"))
    body = client.get("/health").json()
    assert body["model_version"] == "oracle-2026-01-01-deadbeef" and body["status"] == "ok"
    client.post("/ner", json={"idDocument": 1, "text": docs[0].text})
    metrics = client.get("/metrics").text
    assert "pseudo_ner_check_needed_total" in metrics and "pseudo_ner_characters_total" in metrics


def test_registry_cli_list_and_current(tmp_path, monkeypatch, capsys):
    from pseudo_ner.cli import main

    monkeypatch.chdir(tmp_path)
    d = tmp_path / "models" / "spacy"
    d.mkdir(parents=True)
    (d / "w.bin").write_bytes(b"poids")
    scores = {"micro": {"f1": 0.9, "recall": 0.9}, "per_label": {}}
    registry.record("spacy", Path("models/spacy"), scores, path=Path("models/registry.json"))
    assert main(["registry", "current"]) == 0 and "aucun modèle promu" in capsys.readouterr().out
    assert main(["registry", "promote", "spacy"]) == 0 and "modèle servi : spacy-" in capsys.readouterr().out
    assert main(["registry", "list"]) == 0 and "← servi" in capsys.readouterr().out
    assert main(["registry", "verify", "spacy"]) == 0


def test_k8s_manifests_are_valid_and_probe_health():
    import yaml

    for name in ("deployment.yaml", "service.yaml"):
        doc = yaml.safe_load(Path("deploy/k8s", name).read_text(encoding="utf-8"))
        assert doc["kind"] in {"Deployment", "Service"}
    dep = yaml.safe_load(Path("deploy/k8s/deployment.yaml").read_text(encoding="utf-8"))
    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True


def test_registry_accepts_entries_recorded_before_version_field(tmp_path):
    reg = tmp_path / "registry.json"
    d = tmp_path / "spacy"
    d.mkdir()
    (d / "w.bin").write_bytes(b"poids")
    old = {"path": str(d), "sha256": registry.fingerprint(d), "recorded_on": "2026-11-09", "micro_f1": 0.9, "min_f1": 0.88}
    reg.write_text(json.dumps({"spacy": old}), encoding="utf-8")
    assert registry.load(reg)["spacy"]["version"] == f"spacy-2026-11-09-{old['sha256'][:8]}"
    chosen = registry.promote("spacy", path=reg, current=tmp_path / "current.json")
    assert chosen["version"].startswith("spacy-2026-11-09-")
