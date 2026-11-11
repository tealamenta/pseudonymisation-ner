"""Corpus synthétique, tokenisation BIO, évaluation, post-traitement, pseudonymisation."""

from __future__ import annotations

import random

import pytest

from pseudo_ner import synthetic
from pseudo_ner.categories import CATEGORIES, MODEL_LABELS, PSEUDONYMISED_BY_DEFAULT, Entity
from pseudo_ner.evaluate import evaluate, leaked
from pseudo_ner.postprocess import _close, doubts, postprocess, propagate_names, regex_identifiers
from pseudo_ner.pseudonymise import _letters, pseudonymise
from pseudo_ner.tokenize import sentences, spans_from_tags

# ---------------------------------------------------------------- corpus


def test_generator_offsets_are_exact_and_reproducible():
    docs = synthetic.generate_corpus(5, seed=7)
    again = synthetic.generate_corpus(5, seed=7)
    assert [d.text for d in docs] == [d.text for d in again]
    for d in docs:
        for e in d.entities:
            assert d.text[e.start : e.end] == e.text
            assert e.label in CATEGORIES
        # pas de chevauchement
        for a, b in zip(d.entities, d.entities[1:], strict=False):
            assert a.end <= b.start


def test_splits_use_disjoint_values():
    parts = synthetic.make_splits(40, 10, 10, seed=1)
    for label in ("personnePhysiqueNom", "localite", "personneMorale"):
        train = {e.text.casefold() for d in parts["train"] for e in d.entities if e.label == label}
        test = {e.text.casefold() for d in parts["test"] for e in d.entities if e.label == label}
        assert not (train & test), label


def test_every_category_is_present_in_a_small_corpus():
    docs = synthetic.generate_corpus(30, seed=2)
    seen = {e.label for d in docs for e in d.entities}
    assert seen == set(CATEGORIES)


def test_jsonl_roundtrip(tmp_path):
    docs = synthetic.generate_corpus(3, seed=0)
    path = synthetic.save(docs, tmp_path / "x.jsonl")
    loaded = synthetic.load(path)
    assert [d.to_dict() for d in loaded] == [d.to_dict() for d in docs]


# ---------------------------------------------------------------- tokenisation


def test_bio_roundtrip_recovers_entities():
    doc = synthetic.generate_corpus(1, seed=11)[0]
    gold = [e for e in doc.entities if e.label in MODEL_LABELS]
    recovered: list[Entity] = []
    for s in sentences(doc.text, gold):
        recovered.extend(spans_from_tags(s.tokens, s.tags, doc.text))
    assert {(e.start, e.end, e.label) for e in recovered} == {(e.start, e.end, e.label) for e in gold}


def test_spans_from_tags_handles_orphan_inside_tag():
    text = "Jean Dupont habite Lyon"
    toks = sentences(text)[0].tokens
    ents = spans_from_tags(toks, ["I-personnePhysiquePrenom", "B-personnePhysiqueNom", "O", "B-localite"], text)
    assert [(e.text, e.label) for e in ents] == [
        ("Jean", "personnePhysiquePrenom"),
        ("Dupont", "personnePhysiqueNom"),
        ("Lyon", "localite"),
    ]


# ---------------------------------------------------------------- évaluation


def test_evaluate_strict_spans():
    gold = [[Entity(0, 4, "personnePhysiquePrenom", "Jean"), Entity(5, 11, "personnePhysiqueNom", "Dupont")]]
    pred = [
        [
            Entity(0, 4, "personnePhysiquePrenom", "Jean"),
            Entity(5, 11, "localite", "Dupont"),
            Entity(20, 24, "adresse", "x"),
        ]
    ]
    r = evaluate(gold, pred)
    assert r.micro.tp == 1 and r.micro.fn == 1 and r.micro.fp == 2
    assert r.per_label["personnePhysiquePrenom"].f1 == 1.0
    assert r.per_label["personnePhysiqueNom"].recall == 0.0
    assert len(r.missed) == 1 and r.missed[0]["text"] == "Dupont"
    assert "| **micro** |" in r.table()
    assert r.to_dict()["micro"]["precision"] == round(1 / 3, 4)


def test_bootstrap_ci_is_on_documents_and_reproducible():
    e = Entity(0, 4, "personnePhysiquePrenom", "Jean")
    gold = [[e]] * 20
    pred = [[e]] * 15 + [[]] * 5  # 15 documents parfaits, 5 documents vides
    r = evaluate(gold, pred)
    lo, hi = r.bootstrap_f1(n=500, seed=1)
    assert lo < r.micro.f1 < hi  # l'intervalle contient l'estimation ponctuelle
    assert r.bootstrap_f1(n=500, seed=1) == (lo, hi)  # reproductible
    assert hi - lo > 0.1  # 20 documents : intervalle large, pas trompeur
    perfect = evaluate(gold, [[e]] * 20)
    assert perfect.bootstrap_f1(n=100) == (1.0, 1.0)
    assert "micro_f1_ci95" in r.to_dict() and "intervalle à 95 %" in r.table()


def test_leaked_detects_uncovered_mentions():
    gold = [[Entity(0, 4, "personnePhysiqueNom", "Roux"), Entity(7, 11, "localite", "Lyon")]]
    assert leaked(gold, [[]], PSEUDONYMISED_BY_DEFAULT) == [{"doc": 0, "label": "personnePhysiqueNom", "text": "Roux"}]
    # couverte par une entité masquée, même d'une autre catégorie ou plus large : pas de fuite
    assert leaked(gold, [[Entity(0, 6, "personnePhysiquePrenom", "Roux à")]], PSEUDONYMISED_BY_DEFAULT) == []
    # couverte par une catégorie non masquée : fuite
    assert len(leaked(gold, [[Entity(0, 4, "professionnelAvocat", "Roux")]], PSEUDONYMISED_BY_DEFAULT)) == 1


# ---------------------------------------------------------------- post-traitement


def test_propagation_annotates_missed_mentions():
    text = "M. Jean Dupont demande. Mme Dupont conteste. Dupont est débouté."
    ents = [
        Entity(3, 7, "personnePhysiquePrenom", "Jean"),
        Entity(8, 14, "personnePhysiqueNom", "Dupont"),
        Entity(28, 34, "personnePhysiqueNom", "Dupont"),
    ]
    out = propagate_names(text, ents)
    assert sum(e.text == "Dupont" for e in out) == 3
    assert any(e.source == "postprocess" for e in out)
    # une mention vue une seule fois n'est pas propagée
    assert sum(e.text == "Jean" for e in out) == 1


def test_regex_identifiers():
    text = "Tél. 06 12 34 56 78, NIR 1 85 03 75 123 456 78, véhicule AB-123-CD, SIREN 552 100 554."
    out = regex_identifiers(text, [])
    assert sorted(e.text for e in out) == ["06 12 34 56 78", "1 85 03 75 123 456 78", "552 100 554", "AB-123-CD"]
    assert all(e.label == "numeroIdentifiant" for e in out)


def test_doubts_and_close_names():
    ents = [
        Entity(0, 7, "personnePhysiquePrenom", "Thibaut"),
        Entity(10, 18, "personnePhysiquePrenom", "Thibault"),
        Entity(20, 22, "personnePhysiqueNom", "Li"),
        Entity(30, 36, "personnePhysiqueNom", "Martin"),
        Entity(40, 46, "professionnelAvocat", "Martin"),
    ]
    notes = doubts(ents)
    assert any("Thibaut" in n and "Thibault" in n for n in notes)
    assert any("très court" in n for n in notes)
    assert any("partie et professionnel" in n for n in notes)
    assert _close("Thibaut", "Thibault") and not _close("Jean", "Marie")


def test_postprocess_end_to_end_on_synthetic():
    doc = synthetic.generate_corpus(1, seed=5)[0]
    model_only = [e for e in doc.entities if e.label in MODEL_LABELS]
    checked = postprocess(doc.text, model_only)
    # les numéros, absents de la sortie modèle, sont retrouvés par règles
    gold_numbers = {e.text for e in doc.entities if e.label == "numeroIdentifiant"}
    found = {e.text for e in checked.entities if e.label == "numeroIdentifiant"}
    assert gold_numbers <= found
    assert checked.to_dict()["check_needed"] == bool(checked.checklist)


# ---------------------------------------------------------------- pseudonymisation


def test_pseudonymise_is_consistent_and_complete():
    doc = synthetic.generate_corpus(1, seed=9)[0]
    out = pseudonymise(doc.text, doc.entities)
    assert out.replaced == sum(e.label in PSEUDONYMISED_BY_DEFAULT for e in doc.entities)
    # même valeur → même pseudonyme, et aucune valeur à masquer ne survit
    assert leaked([doc.entities], [doc.entities], PSEUDONYMISED_BY_DEFAULT) == []
    # les adresses, dates et numéros, uniques dans le texte, ont disparu
    for e in doc.entities:
        if e.label in ("adresse", "dateDeNaissance", "numeroIdentifiant"):
            assert e.text not in out.text
    names = {e.text.casefold() for e in doc.entities if e.label == "personnePhysiqueNom"}
    assert len({out.mapping[("personnePhysiqueNom", n)] for n in names}) == len(names)
    # les localités et professionnels sont conservés par défaut
    assert all(e.text in out.text for e in doc.entities if e.label == "localite")


def test_pseudonymise_custom_labels_and_letters():
    text = "Me Martin et Me Durand"
    ents = [Entity(3, 9, "professionnelAvocat", "Martin"), Entity(16, 22, "professionnelAvocat", "Durand")]
    assert pseudonymise(text, ents).text == text  # non masqué par défaut
    assert pseudonymise(text, ents, frozenset({"professionnelAvocat"})).text == "Me [A] et Me [B]"
    assert [_letters(i) for i in (0, 25, 26, 27)] == ["A", "Z", "AA", "AB"]


def test_pseudonymise_letters_are_distinct_across_first_and_last_names():
    # format de la Cour : « Madame [E] [A]-[M] », une lettre par valeur ; jamais « [A] [A] »
    text = "M. Jean Dupont et Mme Marie Dupont"
    ents = [
        Entity(3, 7, "personnePhysiquePrenom", "Jean"),
        Entity(8, 14, "personnePhysiqueNom", "Dupont"),
        Entity(22, 27, "personnePhysiquePrenom", "Marie"),
        Entity(28, 34, "personnePhysiqueNom", "Dupont"),
    ]
    assert pseudonymise(text, ents).text == "M. [A] [B] et Mme [C] [B]"


def test_entity_overlap():
    a, b, c = Entity(0, 5, "x", "aaaaa"), Entity(3, 8, "x", "bbbbb"), Entity(5, 9, "x", "cccc")
    assert a.overlaps(b) and not a.overlaps(c)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_documents_never_leak_after_full_chain(seed):
    rng = random.Random(seed)
    doc = synthetic.generate_document(rng, "t")
    checked = postprocess(doc.text, [e for e in doc.entities if e.label in MODEL_LABELS])
    assert leaked([doc.entities], [checked.entities], PSEUDONYMISED_BY_DEFAULT) == []


def test_paragraphs_for_domain_adaptation():
    from pseudo_ner.models.hf_model import paragraphs

    long = "Phrase une. " * 200  # 2 400 caractères, coupée sur les points
    parts = paragraphs(["COUR D'APPEL\n\ncourt\n" + long + "\nUne ligne assez longue pour être gardée telle quelle."])
    assert all(40 <= len(p) <= 1200 for p in parts)
    assert "court" not in parts and parts[-1].startswith("Une ligne")
    assert "".join(parts[:-1]).replace(" ", "") == long.replace(" ", "")  # rien de perdu dans la coupe


def test_train_and_adapt_cli_arguments():
    from pseudo_ner.cli import build_parser

    a = build_parser().parse_args(["train", "camembert", "--out", "m", "--base", "models/camembert-judilibre"])
    assert str(a.base) == "models/camembert-judilibre"
    b = build_parser().parse_args(["adapt", "camembert", "--epochs", "2"])
    assert b.epochs == 2 and str(b.out) == "models/camembert-judilibre" and str(b.data) == "data/mixed"


def test_camembert_windows_never_truncate_long_headers():
    from pseudo_ner.models import hf_model

    class StubTokenizer:  # chaque mot de k lettres → k sous-mots
        def tokenize(self, word):
            return ["x"] * len(word)

    ner = hf_model.CamembertNer(tokenizer=StubTokenizer())
    words = ["abcde"] * 100  # 500 sous-mots, bien plus que MAX_LEN
    windows = ner._windows(words)
    assert windows[0] == (0, 38) and windows[-1][1] == 100  # 38 × 5 = 190 ≤ 192 − 2
    assert all(b - a > 0 for a, b in windows) and [a for a, _ in windows[1:]] == [b for _, b in windows[:-1]]
    assert ner._windows(["un", "mot"]) == [(0, 2)]
