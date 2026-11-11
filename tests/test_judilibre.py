"""Décisions Judilibre : export paginé (simulé), exploration, ré-injection des marques."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pseudo_ner import judilibre
from pseudo_ner.cli import main

# Texte inventé, dans le format des décisions de cours d'appel publiées (marques du moteur de la Cour).
SAMPLE = """COUR D'APPEL DE LYON
Chambre sociale A
ARRÊT DU 12 MARS 2024

APPELANTE
Madame [E] [A]-[M]
[Adresse 1]
[Localité 1]
née le [Date naissance 1] 1975 à [Localité 2]
Représentée par Me Thibault ROUX, avocat au barreau de LYON

INTIMÉ
Monsieur [U] [B]
[Adresse 2]
[Localité 1]
Représenté par Me Jeanne LAMBERT de la SCP LAMBERT, avocate au barreau de LYON

COMPOSITION DE LA COUR
Madame Anne MERCIER, présidente, et Monsieur Paul GIRARD, conseiller.

EXPOSÉ DU LITIGE
Mme [A] a été engagée par la société Alpha le 3 janvier 2010. M. [B] conteste. Parcelle [Cadastre 1].
Mme [A]-[M] demande la confirmation du jugement.
"""


def _records(n: int) -> list[dict]:
    meta = {"jurisdiction": "ca", "chamber": "soc", "decision_date": "2024-03-12", "type": "arret", "solution": "confirmation"}
    return [{"id": f"d{i}", "number": f"23/{i:05d}", "text": SAMPLE, **meta} for i in range(n)]


def test_export_follows_next_batch_and_stops_at_n():
    pages = {
        "batch=0": {"results": _records(3), "next_batch": "jurisdiction=ca&batch_size=100&batch=1"},  # forme réelle : paramètres seuls
        "batch=1": {"results": _records(3), "next_batch": None},
    }
    calls: list[str] = []

    def fake(url: str, headers: dict) -> dict:
        calls.append(url)
        assert headers == {"Authorization": "Bearer t"}
        assert url.startswith(judilibre.SANDBOX_URL + "/export?")
        return pages[re.search(r"batch=\d", url).group()]  # type: ignore[union-attr]

    got = list(judilibre.export({"Authorization": "Bearer t"}, 4, fetch=fake))
    assert len(got) == 4
    assert "jurisdiction=ca" in calls[0] and "batch=0" in calls[0]
    assert calls[1] == judilibre.SANDBOX_URL + "/export?jurisdiction=ca&batch_size=100&batch=1"
    assert judilibre._absolute("https://x/y?batch=2", judilibre.SANDBOX_URL) == "https://x/y?batch=2"
    assert judilibre._absolute(None, judilibre.SANDBOX_URL) is None
    assert set(got[0]) == {"id", "jurisdiction", "chamber", "number", "decision_date", "type", "solution", "text"}


def test_explore_counts_marks_and_contexts():
    stats = judilibre.explore(_records(2))
    assert stats.documents == 2
    assert stats.marks["personne (lettre)"] == 2 * 9
    assert stats.marks["Adresse"] == 4 and stats.marks["Localité"] == 6 and stats.marks["Date naissance"] == 2
    assert stats.with_birth_date == 2 and stats.with_address == 2
    assert stats.unknown_marks == {"[Cadastre n]": 2} or stats.marks.get("Cadastre") == 2
    assert "née le" in stats.contexts["Date naissance"]
    assert "| Adresse | 4 |" in stats.table()


def test_reinject_is_consistent_and_offsets_are_exact():
    docs = judilibre.reinject(_records(1), seed=3)
    doc = docs[0]
    assert "[" not in doc.text.replace("[Cadastre 1]", "")  # seule la marque non gérée reste
    for e in doc.entities:
        assert doc.text[e.start : e.end] == e.text
    by_label = {}
    for e in doc.entities:
        by_label.setdefault(e.label, []).append(e.text)
    # [E] prénom, [A]-[M] nom composé ; [A] seul plus loin → même nom, même catégorie
    assert len(by_label["personnePhysiquePrenom"]) == 2  # [E] et [U]
    noms = by_label["personnePhysiqueNom"]
    composed = [n for n in noms if "-" in n]
    assert len(composed) == 2 and composed[0] == composed[1]
    first_a = composed[0].split("-")[0]
    assert noms.count(first_a) == 1  # « Mme [A] a été engagée »
    # [Localité 1] deux fois → même ville ; [Localité 2] différente
    loc = by_label["localite"]
    assert len(loc) == 3 and loc[0] == loc[2] and loc[0] != loc[1]
    # date : jour + mois injectés, année conservée dans l'entité
    (date,) = by_label["dateDeNaissance"]
    assert date.endswith(" 1975") and len(date.split()) == 3
    assert len(by_label["adresse"]) == 2 and by_label["adresse"][0] != by_label["adresse"][1]
    # civilité respectée : Madame [E] → prénom féminin, Monsieur [U] → masculin
    from pseudo_ner.pools import pool

    madame, monsieur = by_label["personnePhysiquePrenom"]
    assert madame in pool("test").prenoms_f and monsieur in pool("test").prenoms_m
    # professionnels en clair : annotés par règles (source « rule »), un token par entité
    assert "Thibault ROUX" in doc.text
    assert by_label["professionnelAvocat"] == ["Thibault", "ROUX", "Jeanne", "LAMBERT"]
    assert by_label["professionnelMagistratGreffier"] == ["Anne", "MERCIER", "Paul", "GIRARD"]
    assert all(e.source == "rule" for e in doc.entities if e.label.startswith("professionnel"))
    assert "professionnelAvocat" not in {e.label for e in judilibre.reinject(_records(1), professionals=False)[0].entities}


def test_reinject_pools_are_disjoint_and_test_is_unseen():
    from pseudo_ner import synthetic as syn
    from pseudo_ner.pools import pool

    train, dev, test = pool("train"), pool("dev"), pool("test")
    for field in ("noms", "prenoms_f", "prenoms_m", "villes", "voies"):
        a, b, c = ({v.casefold() for v in getattr(p, field)} for p in (train, dev, test))
        assert not (a & b) and not (a & c) and not (b & c), field
        assert not (c & {v.casefold() for v in getattr(syn, field.upper())}), field  # test : rien du synthétique d'entraînement
    assert len(train.noms) > 200 and len(test.noms) > 80  # assez de variété pour ne pas apprendre une liste
    doc = judilibre.reinject(_records(1), seed=1)[0]
    test_noms = {n.casefold() for n in test.noms}
    for e in doc.entities:
        if e.label == "personnePhysiqueNom":
            assert all(part.casefold() in test_noms for part in e.text.split("-"))


def test_cli_stats_and_reinject(tmp_path: Path, capsys):
    raw = tmp_path / "raw.jsonl"
    judilibre.save_raw(_records(2), raw)
    assert main(["judilibre", "stats", "--raw", str(raw), "--report", str(tmp_path / "s.json")]) == 0
    assert "décisions : 2" in capsys.readouterr().out
    out = tmp_path / "re.jsonl"
    assert main(["judilibre", "reinject", "--raw", str(raw), "--out", str(out)]) == 0
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[0]["entities"]
    assert "personnePhysiqueNom" in capsys.readouterr().out


def test_auth_headers_prefer_oauth(monkeypatch):
    monkeypatch.setenv("PISTE_CLIENT_ID", "id")
    monkeypatch.setenv("PISTE_CLIENT_SECRET", "secret")
    seen: dict = {}

    def fake_token(client_id: str, client_secret: str, token_url: str) -> str:
        seen.update(client_id=client_id, token_url=token_url)
        return "jeton"

    monkeypatch.setattr(judilibre, "token", fake_token)
    assert judilibre.auth_headers_from_env() == {"Authorization": "Bearer jeton"}
    assert seen == {"client_id": "id", "token_url": judilibre.SANDBOX_TOKEN_URL}
    assert judilibre.auth_headers_from_env(production=True)["Authorization"] == "Bearer jeton"
    assert seen["token_url"] == judilibre.PRODUCTION_TOKEN_URL
    monkeypatch.delenv("PISTE_CLIENT_ID")
    monkeypatch.setenv("PISTE_API_KEY", "k")
    assert judilibre.auth_headers_from_env() == {"KeyId": "k"}


def test_export_spread_covers_periods_and_dedupes():
    calls: list[str] = []

    def fake(url: str, headers: dict) -> dict:
        calls.append(url)
        start = re.search(r"date_start=([\d-]+)", url).group(1)  # type: ignore[union-attr]
        recs = _records(3)
        for r in recs:
            r["id"] = f"{start}-{r['id']}"
        recs[0]["id"] = "shared"  # même décision dans chaque tranche → gardée une fois
        return {"results": recs, "next_batch": None}

    got = list(judilibre.export_spread({"KeyId": "k"}, 6, periods=3, date_start="2024-01-01", date_end="2024-12-31", fetch=fake))
    starts = [re.search(r"date_start=([\d-]+)", u).group(1) for u in calls]  # type: ignore[union-attr]
    assert starts == ["2024-01-01", "2024-05-01", "2024-08-31"]
    assert "date_end=2024-12-31" in calls[-1]
    ids = [r["id"] for r in got]  # 2 par tranche demandées ; « shared » revient à chaque tranche et n'est gardée qu'une fois
    assert ids == ["shared", "2024-01-01-d1", "2024-05-01-d1", "2024-08-31-d1"]


def test_annotate_professionals_rules():
    from pseudo_ner.synthetic import Document

    text = (
        "Président : M. Jean DUPONT\nGreffier : Mme Lise PETIT\n"
        "En présence de Marc LEROY, conseiller, et de Me Paul MARTIN de la SELARL MARTIN, avocat.\n"
        "PRÉSIDENTE DE CHAMBRE : Anne ROUX\nMaître Claire DURAND"
    )
    gold = [judilibre.Entity(text.index("Claire"), text.index("Claire") + 6, "personnePhysiquePrenom", "Claire", "gold")]
    doc = judilibre.annotate_professionals(Document("x", text, gold))
    got = [(e.text, e.label) for e in doc.entities if e.source == "rule"]
    assert ("Jean", "professionnelMagistratGreffier") in got and ("PETIT", "professionnelMagistratGreffier") in got
    assert ("LEROY", "professionnelMagistratGreffier") in got and ("ROUX", "professionnelMagistratGreffier") in got
    assert ("Paul", "professionnelAvocat") in got and ("MARTIN", "professionnelAvocat") in got
    assert ("M", "professionnelMagistratGreffier") not in got and ("Greffier", "professionnelMagistratGreffier") not in got
    assert ("Claire", "professionnelAvocat") not in got  # déjà annotée (partie) : jamais recouverte
    assert ("DURAND", "professionnelAvocat") in got
    for e in doc.entities:
        assert text[e.start : e.end] == e.text


def test_split_is_by_decision_and_reproducible():
    records = _records(10)
    train, test = judilibre.split(records, test_share=0.2, seed=1)
    again = judilibre.split(records, test_share=0.2, seed=1)
    assert len(test) == 2 and len(train) == 8
    assert {r["id"] for r in train}.isdisjoint({r["id"] for r in test})
    assert [r["id"] for r in test] == [r["id"] for r in again[1]]


def test_cli_split_builds_mixed_sets(tmp_path: Path, capsys):
    from pseudo_ner import synthetic

    raw = tmp_path / "raw.jsonl"
    judilibre.save_raw(_records(10), raw)
    synth = tmp_path / "synthetic"
    for name, docs in synthetic.make_splits(4, 2, 2, seed=0).items():
        synthetic.save(docs, synth / f"{name}.jsonl")
    out = tmp_path / "mixed"
    assert main(["judilibre", "split", "--raw", str(raw), "--synthetic", str(synth), "--out", str(out), "--test-share", "0.2"]) == 0
    train, dev, test = (synthetic.load(out / f"{n}.jsonl") for n in ("train", "dev", "test"))
    assert len(test) == 2 and len(train) == 4 + 7 and len(dev) == 2 + 1
    assert all(e.label != "professionnelAvocat" or e.source == "gold" for d in test for e in d.entities)  # test : pas de règles
    assert "décisions réelles jamais vues" in capsys.readouterr().out


def test_cli_split_without_synthetic(tmp_path: Path):
    from pseudo_ner import synthetic

    raw = tmp_path / "raw.jsonl"
    judilibre.save_raw(_records(10), raw)
    out = tmp_path / "real"
    assert main(["judilibre", "split", "--raw", str(raw), "--out", str(out), "--no-synthetic", "--test-share", "0.2"]) == 0
    train, dev, test = (synthetic.load(out / f"{n}.jsonl") for n in ("train", "dev", "test"))
    assert (len(train), len(dev), len(test)) == (7, 1, 2)
