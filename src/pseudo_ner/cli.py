"""Ligne de commande ``pseudo-ner``.

pseudo-ner generate --n 600 --out data/synthetic
pseudo-ner train spacy --data data/synthetic --out models/spacy
pseudo-ner evaluate spacy --data data/synthetic --model models/spacy --report reports/spacy.json
pseudo-ner pseudonymise spacy --model models/spacy --text "..."   (ou --file decision.txt)
pseudo-ner serve spacy --model models/spacy
pseudo-ner judilibre fetch --n 500 ; pseudo-ner judilibre stats ; pseudo-ner judilibre reinject
pseudo-ner evaluate flair --model-dir models/flair --file data/judilibre/reinjected.jsonl --labels reinjected \\
    --report reports/flair-judilibre.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pseudo_ner import registry, synthetic
from pseudo_ner.categories import MODEL_LABELS, PSEUDONYMISED_BY_DEFAULT
from pseudo_ner.evaluate import evaluate, leaked
from pseudo_ner.models import get
from pseudo_ner.pipeline import Pipeline
from pseudo_ner.postprocess import postprocess


def cmd_generate(a: argparse.Namespace) -> int:
    n_test = max(1, a.n * 15 // 100)
    parts = synthetic.make_splits(a.n - 2 * n_test, n_test, n_test, seed=a.seed, noise=a.noise)
    for name, part in parts.items():
        synthetic.save(part, a.out / f"{name}.jsonl")
        print(f"{name}: {len(part)} documents, {sum(len(d.entities) for d in part)} entités")
    return 0


def _load_split(data: Path) -> dict[str, list[synthetic.Document]]:
    return {name: synthetic.load(data / f"{name}.jsonl") for name in ("train", "dev", "test")}


def cmd_train(a: argparse.Namespace) -> int:
    parts = _load_split(a.data)
    model = get(a.model)()
    started = time.perf_counter()
    extra = {"base": str(a.base)} if a.base else {}
    model.train(parts["train"], parts["dev"], a.out, epochs=a.epochs, embeddings=a.embeddings, **extra)
    print(f"entraîné en {time.perf_counter() - started:.0f}s → {a.out}")
    return 0


def cmd_adapt(a: argparse.Namespace) -> int:
    """Pré-entraînement adapté au domaine de CamemBERT sur le texte (sans étiquette) du jeu d'entraînement."""
    from pseudo_ner.models.hf_model import adapt, paragraphs

    texts = [d.text for d in synthetic.load(a.data / "train.jsonl")]
    print(f"{len(texts)} décisions, {len(paragraphs(texts))} paragraphes")
    started = time.perf_counter()
    adapt(texts, a.out, base=a.base, epochs=a.epochs, max_len=a.max_len)
    print(f"adapté en {time.perf_counter() - started:.0f}s → {a.out} (puis : train camembert --base {a.out})")
    return 0


def _keep(entities: list, labels: frozenset[str] | None) -> list:
    return [e for e in entities if labels is None or e.label in labels]


def cmd_evaluate(a: argparse.Namespace) -> int:
    """Scores sur un jeu annoté : le test synthétique, ou un fichier (ex. décisions Judilibre ré-injectées).

    ``--labels reinjected`` restreint la comparaison aux catégories annotées par la ré-injection (parties,
    adresses, localités, dates de naissance) : les avocats et magistrats y sont en clair et non annotés.
    """
    from pseudo_ner.judilibre import REINJECTED_LABELS

    if a.file:
        test, split = synthetic.load(a.file), str(a.file)
    else:
        test, split = _load_split(a.data)[a.split], a.split
    if a.limit:
        test = test[: a.limit]
    labels = REINJECTED_LABELS if a.labels == "reinjected" else None
    model = get(a.model).load(a.model_dir)
    gold_model = [_keep([e for e in d.entities if e.label in MODEL_LABELS], labels) for d in test]
    started = time.perf_counter()
    raw = [model.predict(d.text) for d in test]
    seconds = time.perf_counter() - started
    report_raw = evaluate(gold_model, [_keep(p, labels) for p in raw], [d.id for d in test])
    # avec post-traitement : toutes les catégories, y compris les numéros par règles
    gold_all = [_keep(d.entities, labels) for d in test]
    post = [_keep(postprocess(d.text, p).entities, labels) for d, p in zip(test, raw, strict=True)]
    report_post = evaluate(gold_all, post, [d.id for d in test])
    # fuite : entités à masquer encore présentes après remplacement
    leaks = leaked(gold_all, post, PSEUDONYMISED_BY_DEFAULT)
    out = {
        "model": a.model,
        "split": split,
        "labels": a.labels,
        "documents": len(test),
        "seconds_per_doc": round(seconds / len(test), 3),
        "raw": report_raw.to_dict(),
        "postprocessed": report_post.to_dict(),
        "leaks": {"count": len(leaks), "examples": leaks[:50]},
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"## {a.model} — modèle seul\n{report_raw.table()}\n\n## {a.model} — avec post-traitement\n{report_post.table()}")
    print(f"\nfuites après pseudonymisation : {len(leaks)} · {seconds / len(test):.3f} s/document")
    if a.register:
        entry = registry.record(a.model, a.model_dir, report_post.to_dict(), {"raw_micro_f1": report_raw.micro.f1, "leaks": len(leaks)})
        print(f"enregistré : {entry['sha256'][:12]}… seuil F1 ≥ {entry['min_f1']}")
    return 0


def cmd_pseudonymise(a: argparse.Namespace) -> int:
    text = a.file.read_text(encoding="utf-8") if a.file else a.text
    if not text:
        print("--text ou --file requis", file=sys.stderr)
        return 1
    result = Pipeline.load(a.model, a.model_dir).run(text)
    if a.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.pseudonymised.text)
        if result.checked.check_needed:
            print("\nÀ vérifier :", *result.checked.checklist, sep="\n- ", file=sys.stderr)
    return 0


def cmd_index(a: argparse.Namespace) -> int:
    """Traite un fichier de décisions, stocke le résultat (MongoDB) et indexe le texte pseudonymisé (ElasticSearch)."""
    from pseudo_ner import registry as reg
    from pseudo_ner.search import connect

    store, index = connect(a.mongo, a.es)
    pipeline = Pipeline.load(a.model, a.model_dir)
    docs = synthetic.load(a.file)[: a.limit] if a.limit else synthetic.load(a.file)
    entry = reg.load().get(a.model, {})
    for d in docs:
        result = pipeline.run(d.text)
        record = {
            "idDocument": d.id,
            "model": a.model,
            "model_sha256": entry.get("sha256"),
            "text": d.text,
            **result.to_dict(),
        }
        store.save(record)
        index.index(
            d.id,
            result.pseudonymised.text,
            {"model": a.model, "check_needed": result.checked.check_needed, "chambre": _chambre(d.text)},
        )
    print(f"{len(docs)} décisions traitées, {len(store.pending_review())} à relire")
    if a.query:
        for hit in index.search(a.query):
            print(f"  {hit['idDocument']}  score={hit['score']}  {hit['extract'][:80]!r}")
    return 0


def _chambre(text: str) -> str | None:
    for line in text.splitlines()[:3]:
        if "CHAMBRE" in line.upper():
            return line.strip().lower()
    return None


def cmd_judilibre_fetch(a: argparse.Namespace) -> int:
    from pseudo_ner import judilibre

    base = judilibre.PRODUCTION_URL if a.production else judilibre.SANDBOX_URL
    headers = judilibre.auth_headers_from_env(a.production)
    records = judilibre.export_spread(headers, a.n, a.jurisdiction, a.date_start, a.date_end, a.periods, base_url=base)
    count = judilibre.save_raw(records, a.out)
    print(f"{count} décisions → {a.out}")
    return 0


def cmd_judilibre_stats(a: argparse.Namespace) -> int:
    from pseudo_ner import judilibre

    stats = judilibre.explore(judilibre.load_raw(a.raw))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(stats.table())
    return 0


def cmd_judilibre_reinject(a: argparse.Namespace) -> int:
    from pseudo_ner import judilibre

    docs = judilibre.reinject(judilibre.load_raw(a.raw), seed=a.seed, pool=a.pool, professionals=not a.no_professionals)
    synthetic.save(docs, a.out)
    counts: dict[str, int] = {}
    for d in docs:
        for e in d.entities:
            counts[e.label] = counts.get(e.label, 0) + 1
    print(f"{len(docs)} décisions ré-injectées → {a.out}")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")
    return 0


def cmd_judilibre_split(a: argparse.Namespace) -> int:
    """Jeu mixte : synthétique + décisions réelles ré-injectées (train / dev), décisions réelles jamais vues (test)."""
    from pseudo_ner import judilibre

    train_records, test_records = judilibre.split(judilibre.load_raw(a.raw), a.test_share, a.seed)
    n_dev = max(1, len(train_records) // 10)
    real_train = judilibre.reinject(train_records[n_dev:], seed=a.seed, pool="train")
    real_dev = judilibre.reinject(train_records[:n_dev], seed=a.seed + 1, pool="dev")
    real_test = judilibre.reinject(test_records, seed=a.seed + 2, pool="test", professionals=False)
    synth_train = [] if a.no_synthetic else synthetic.load(a.synthetic / "train.jsonl")
    synth_dev = [] if a.no_synthetic else synthetic.load(a.synthetic / "dev.jsonl")
    parts = {"train": synth_train + real_train, "dev": synth_dev + real_dev, "test": real_test}
    for name, docs in parts.items():
        synthetic.save(docs, a.out / f"{name}.jsonl")
    print(
        f"train : {len(parts['train'])} documents ({len(real_train)} réels) · dev : {len(parts['dev'])} ({len(real_dev)} réels)"
        f" · test : {len(real_test)} décisions réelles jamais vues → {a.out}"
    )
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    """Sert un modèle : celui demandé, ou à défaut le modèle promu (``models/current.json``)."""
    import uvicorn

    from pseudo_ner.api import create_app

    model, model_dir, version = a.model, a.model_dir, None
    if model is None or model_dir is None:
        current = registry.load(registry.CURRENT)
        if not current:
            print("ni modèle sur la ligne de commande, ni modèle promu (pseudo-ner registry promote <nom>)", file=sys.stderr)
            return 2
        model, model_dir, version = current["model"], Path(current["path"]), current["version"]
        registry.verify(model)
    else:  # modèle explicite : on retrouve sa version s'il est au registre
        entry = registry.load().get(model)
        if entry and Path(entry["path"]) == model_dir:
            version = entry["version"]
    uvicorn.run(create_app(model, model_dir, version=version), host=a.host, port=a.port)
    return 0


def cmd_registry(a: argparse.Namespace) -> int:
    """Registre des modèles : liste, vérification, promotion, retour arrière."""
    if a.action == "list":
        current = registry.load(registry.CURRENT)
        for name, e in registry.load().items():
            mark = " ← servi" if current.get("model") == name and current.get("sha256") == e["sha256"] else ""
            print(f"{name:12} {e['version']:32} F1 {e['micro_f1']:.4f} (seuil {e['min_f1']:.4f}){mark}")
        return 0
    if a.action == "verify":
        print(f"{a.name} : empreinte conforme ({registry.verify(a.name)[:12]}…)")
        return 0
    if a.action == "promote":
        chosen = registry.promote(a.name)
        print(f"modèle servi : {chosen['version']} (F1 {chosen['micro_f1']:.4f})")
        return 0
    if a.action == "rollback":
        restored = registry.rollback()
        print(f"retour arrière : {restored['version']}")
        return 0
    if a.action == "current":
        current = registry.load(registry.CURRENT)
        shown = {k: v for k, v in current.items() if k != "previous"}
        print(json.dumps(shown, indent=2, ensure_ascii=False) if current else "aucun modèle promu")
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pseudo-ner", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="corpus synthétique annoté")
    g.add_argument("--n", type=int, default=600)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="part de mentions bruitées (majuscules, accents perdus, civilité absente)",
    )
    g.add_argument("--out", type=Path, default=Path("data/synthetic"))
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train")
    t.add_argument("model", choices=["spacy", "flair", "camembert"])
    t.add_argument("--data", type=Path, default=Path("data/synthetic"))
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--embeddings", default="flair-fr", help="flair : flair-fr (téléchargés) ou onehot (hors ligne)")
    t.add_argument("--base", type=Path, help="camembert : modèle de départ (ex. models/camembert-judilibre, voir adapt)")
    t.set_defaults(func=cmd_train)

    ad = sub.add_parser("adapt", help="pré-entraînement adapté au domaine (MLM) de CamemBERT sur le texte du train")
    ad.add_argument("model", choices=["camembert"])
    ad.add_argument("--data", type=Path, default=Path("data/mixed"))
    ad.add_argument("--out", type=Path, default=Path("models/camembert-judilibre"))
    ad.add_argument("--base", default="camembert-base")
    ad.add_argument("--epochs", type=int, default=1)
    ad.add_argument("--max-len", type=int, default=256)
    ad.set_defaults(func=cmd_adapt)

    e = sub.add_parser("evaluate")
    e.add_argument("model", choices=["spacy", "flair", "camembert"])
    e.add_argument("--data", type=Path, default=Path("data/synthetic"))
    e.add_argument("--model-dir", type=Path, required=True)
    e.add_argument("--split", default="test", choices=["dev", "test"])
    e.add_argument("--report", type=Path, required=True)
    e.add_argument("--file", type=Path, help="fichier JSONL annoté à la place du jeu synthétique (ex. data/judilibre/reinjected.jsonl)")
    e.add_argument("--labels", default="all", choices=["all", "reinjected"], help="catégories comparées")
    e.add_argument("--limit", type=int, default=0, help="n'évaluer que les N premiers documents")
    e.add_argument("--register", action="store_true", help="inscrit le modèle et son seuil dans models/registry.json")
    e.set_defaults(func=cmd_evaluate)

    j = sub.add_parser("judilibre", help="décisions réelles de l'API Judilibre : export, exploration, ré-injection")
    js = j.add_subparsers(dest="judilibre_command", required=True)
    jf = js.add_parser("fetch", help="exporter des décisions (PISTE_CLIENT_ID / PISTE_CLIENT_SECRET)")
    jf.add_argument("--n", type=int, default=500)
    jf.add_argument("--jurisdiction", default="ca", help="ca (cours d'appel, riches en données personnelles) ou cc")
    jf.add_argument("--date-start", default="2022-07-01")
    jf.add_argument("--date-end", default="2025-12-31")
    jf.add_argument("--periods", type=int, default=8, help="tranches de dates égales, n/periods décisions par tranche (diversité)")
    jf.add_argument("--production", action="store_true", help="API de production plutôt que bac à sable")
    jf.add_argument("--out", type=Path, default=Path("data/judilibre/raw.jsonl"))
    jf.set_defaults(func=cmd_judilibre_fetch)
    jx = js.add_parser("stats", help="explorer le corpus brut : tailles, marques, contextes")
    jx.add_argument("--raw", type=Path, default=Path("data/judilibre/raw.jsonl"))
    jx.add_argument("--report", type=Path, default=Path("reports/judilibre_stats.json"))
    jx.set_defaults(func=cmd_judilibre_stats)
    jr = js.add_parser("reinject", help="remplacer les marques par des valeurs synthétiques cohérentes → jeu annoté")
    jr.add_argument("--raw", type=Path, default=Path("data/judilibre/raw.jsonl"))
    jr.add_argument("--out", type=Path, default=Path("data/judilibre/reinjected.jsonl"))
    jr.add_argument("--seed", type=int, default=0)
    jr.add_argument(
        "--pool", default="test", choices=["train", "dev", "test"], help="pools disjoints (pools.py) ; test = jamais vu à l'entraînement"
    )
    jr.add_argument("--no-professionals", action="store_true", help="sans l'annotation par règles des avocats et magistrats")
    jr.set_defaults(func=cmd_judilibre_reinject)
    jp = js.add_parser("split", help="jeu mixte synthétique + réel (train/dev) et test réel, par décision")
    jp.add_argument("--raw", type=Path, default=Path("data/judilibre/raw.jsonl"))
    jp.add_argument("--synthetic", type=Path, default=Path("data/synthetic"))
    jp.add_argument("--out", type=Path, default=Path("data/mixed"))
    jp.add_argument("--test-share", type=float, default=0.2)
    jp.add_argument("--seed", type=int, default=0)
    jp.add_argument("--no-synthetic", action="store_true", help="réel ré-injecté seul (ablation)")
    jp.set_defaults(func=cmd_judilibre_split)

    p = sub.add_parser("pseudonymise")
    p.add_argument("model", choices=["spacy", "flair", "camembert"])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--text")
    p.add_argument("--file", type=Path)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_pseudonymise)

    i = sub.add_parser("index", help="traiter, stocker (MongoDB) et indexer (ElasticSearch) un fichier de décisions")
    i.add_argument("model", choices=["spacy", "flair", "camembert"])
    i.add_argument("--model-dir", type=Path, required=True)
    i.add_argument("--file", type=Path, default=Path("data/synthetic/test.jsonl"))
    i.add_argument("--limit", type=int, default=0)
    i.add_argument("--mongo", help="ex. mongodb://localhost:27017 ; absent = mémoire")
    i.add_argument("--es", help="ex. http://localhost:9200 ; absent = mémoire")
    i.add_argument("--query", help="recherche à lancer après indexation")
    i.set_defaults(func=cmd_index)

    s = sub.add_parser("serve", help="sert un modèle ; sans argument, le modèle promu (models/current.json)")
    s.add_argument("model", nargs="?", choices=["spacy", "flair", "camembert"])
    s.add_argument("--model-dir", type=Path)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    r = sub.add_parser("registry", help="liste, vérifie, promeut (current.json) ou rétablit un modèle")
    r.add_argument("action", choices=["list", "verify", "promote", "rollback", "current"])
    r.add_argument("name", nargs="?", help="nom du modèle dans models/registry.json (verify, promote)")
    r.set_defaults(func=cmd_registry)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
