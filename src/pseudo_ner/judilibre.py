"""Corpus réel : décisions de cours d'appel de l'API Judilibre (Cour de cassation, portail PISTE).

Les décisions publiées sont **déjà pseudonymisées** par le moteur de la Cour :

    Madame [E] [A]-[M]
    [Adresse 1]
    [Localité 1]
    née le [Date naissance 1] 1975 à [Localité 2]
    Représentée par Me Thibault DUPONT, avocat au barreau de PARIS

Les parties sont des lettres entre crochets, les adresses, localités et dates de
naissance des marques numérotées ; avocats, magistrats et sociétés restent en
clair. On ne peut donc pas entraîner un NER directement dessus. Deux usages ici :

1. **exploration** d'un large corpus : longueurs, fréquence de chaque marque,
   contextes dans lesquels elles apparaissent ;
2. **ré-injection** : chaque marque est remplacée par une valeur synthétique,
   cohérente dans la décision (même lettre → même nom), ce qui donne un texte
   réaliste — vraies phrases, vraie mise en page, vrais professionnels en clair —
   annoté exactement aux positions des valeurs injectées, pour les catégories
   que la Cour masque. C'est un jeu d'évaluation bien plus proche de la
   production que le corpus synthétique, et une source d'enrichissement de
   l'entraînement.

Accès : identifiants OAuth de l'application PISTE (gratuits, après inscription sur
https://piste.gouv.fr et souscription à Judilibre) dans ``PISTE_CLIENT_ID`` et
``PISTE_CLIENT_SECRET``. Les données brutes restent hors du
dépôt (``data/judilibre/``) ; leur réutilisation suit les CGU de la Cour.
"""

from __future__ import annotations

import json
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pseudo_ner.categories import Entity
from pseudo_ner.pools import pool as value_pool
from pseudo_ner.synthetic import MOIS, Document, Pools

SANDBOX_URL = "https://sandbox-api.piste.gouv.fr/cassation/judilibre/v1.0"
PRODUCTION_URL = "https://api.piste.gouv.fr/cassation/judilibre/v1.0"

# Marques produites par le moteur de la Cour. Les lettres désignent une personne physique ;
# les autres marques sont numérotées et cohérentes dans la décision.
LETTERS = re.compile(r"\[(?P<letters>[A-Z]{1,3})\]")
MARK_KINDS = "Adresse|Localité|Date naissance|Cadastre|Immatriculation|Numéro identifiant|Courriel|Téléphone|Compte bancaire|Site web"
MARK = re.compile(rf"\[(?P<kind>{MARK_KINDS})\s*(?P<num>\d+)?\]")
NAME_GROUP = re.compile(r"\[[A-Z]{1,3}\](?:[ \-]\[[A-Z]{1,3}\])*")
YEAR_AFTER = re.compile(r" ?(?P<year>(?:19|20)\d\d)")
ANY_MARK = re.compile(r"\[[^\]\n]{1,40}\]")

LABEL_OF_MARK = {"Adresse": "adresse", "Localité": "localite", "Date naissance": "dateDeNaissance"}

# Catégories annotées par la ré-injection : celles dont la marque est présente dans le texte publié.
REINJECTED_LABELS: frozenset[str] = frozenset({"personnePhysiquePrenom", "personnePhysiqueNom", "adresse", "localite", "dateDeNaissance"})

Headers = dict[str, str]
Fetch = Callable[[str, Headers], dict]

SANDBOX_TOKEN_URL = "https://sandbox-oauth.piste.gouv.fr/api/oauth/token"
PRODUCTION_TOKEN_URL = "https://oauth.piste.gouv.fr/api/oauth/token"


# ---------------------------------------------------------------- API


def _get(url: str, headers: Headers, retries: int = 5) -> dict:
    """GET JSON ; attend et réessaie sur 429 (quota)."""
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={**headers, "accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data: dict = json.loads(response.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as err:
            if err.code == 429 and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    raise RuntimeError("quota API dépassé")


def token(client_id: str, client_secret: str, token_url: str = SANDBOX_TOKEN_URL) -> str:
    """Jeton OAuth 2 (client credentials) du portail PISTE, valable une heure."""
    body = urllib.parse.urlencode(
        {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret, "scope": "openid"}
    ).encode()
    request = urllib.request.Request(token_url, data=body, headers={"content-type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    access: str = data["access_token"]
    return access


def auth_headers_from_env(production: bool = False) -> Headers:
    """``PISTE_CLIENT_ID`` + ``PISTE_CLIENT_SECRET`` (OAuth, recommandé) ou ``PISTE_API_KEY`` (en-tête ``KeyId``, historique)."""
    client_id, secret = os.environ.get("PISTE_CLIENT_ID", ""), os.environ.get("PISTE_CLIENT_SECRET", "")
    if client_id and secret:
        return {"Authorization": "Bearer " + token(client_id, secret, PRODUCTION_TOKEN_URL if production else SANDBOX_TOKEN_URL)}
    key = os.environ.get("PISTE_API_KEY", "")
    if key:
        return {"KeyId": key}
    raise SystemExit("PISTE_CLIENT_ID et PISTE_CLIENT_SECRET absents : identifiants OAuth de l'application PISTE (API Judilibre)")


def export(
    headers: Headers,
    n: int,
    jurisdiction: str = "ca",
    date_start: str = "2023-01-01",
    date_end: str = "2024-12-31",
    base_url: str = SANDBOX_URL,
    batch_size: int = 100,
    fetch: Fetch = _get,
) -> Iterator[dict]:
    """Décisions complètes par lots (``/export``), en suivant ``next_batch`` jusqu'à ``n`` décisions."""
    params = {
        "jurisdiction": jurisdiction,
        "date_start": date_start,
        "date_end": date_end,
        "date_type": "creation",
        "batch_size": min(batch_size, 1000),
        "batch": 0,
        "order": "desc",
    }
    url: str | None = f"{base_url}/export?{urllib.parse.urlencode(params)}"
    seen = 0
    while url and seen < n:
        page = fetch(url, headers)
        for result in page.get("results", []):
            if not result.get("text"):
                continue
            yield {k: result.get(k) for k in ("id", "jurisdiction", "chamber", "number", "decision_date", "type", "solution", "text")}
            seen += 1
            if seen >= n:
                return
        url = _absolute(page.get("next_batch"), base_url)


def export_spread(
    headers: Headers,
    n: int,
    jurisdiction: str = "ca",
    date_start: str = "2022-07-01",
    date_end: str = "2025-12-31",
    periods: int = 8,
    base_url: str = SANDBOX_URL,
    fetch: Fetch = _get,
) -> Iterator[dict]:
    """``n`` décisions réparties sur ``periods`` tranches de dates égales, pour ne pas n'avoir que le dernier lot
    chargé (qui peut être dominé par un seul contentieux). Doublons écartés par identifiant."""
    import datetime as dt

    start, end = dt.date.fromisoformat(date_start), dt.date.fromisoformat(date_end)
    step = (end - start) / periods
    per_period = -(-n // periods)
    seen: set[str] = set()
    for i in range(periods):
        a = start + step * i
        b = start + step * (i + 1) - dt.timedelta(days=1) if i < periods - 1 else end
        for record in export(headers, per_period, jurisdiction, a.isoformat(), b.isoformat(), base_url, min(per_period, 100), fetch):
            if record["id"] in seen or len(seen) >= n:
                continue
            seen.add(record["id"])
            yield record
        if len(seen) >= n:
            return


def _absolute(next_batch: str | None, base_url: str) -> str | None:
    """``next_batch`` est tantôt une URL complète, tantôt la seule chaîne de paramètres (API réelle)."""
    if not next_batch:
        return None
    if next_batch.startswith("http"):
        return next_batch
    return f"{base_url}/export?{next_batch.lstrip('?')}"


def save_raw(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_raw(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------- exploration


@dataclass
class Stats:
    documents: int
    chars_median: int
    chars_max: int
    marks: dict[str, int]
    letters_per_document: float
    with_birth_date: int
    with_address: int
    unknown_marks: Counter[str]
    contexts: dict[str, dict[str, int]]
    chambers: dict[str, int]
    years: dict[str, int]

    def table(self) -> str:
        lines = [
            f"- décisions : {self.documents} · longueur médiane {self.chars_median} caractères (max {self.chars_max})",
            f"- lettres de personnes par décision : {self.letters_per_document:.1f} (mentions)",
            f"- décisions avec date de naissance : {self.with_birth_date} · avec adresse : {self.with_address}",
            "",
            "| Marque | mentions |",
            "|---|---|",
        ]
        lines += [f"| {k} | {v} |" for k, v in sorted(self.marks.items(), key=lambda kv: -kv[1])]
        if self.unknown_marks:
            lines += ["", "Autres marques (non ré-injectées) : " + ", ".join(f"`{k}` × {v}" for k, v in self.unknown_marks.most_common(10))]
        lines += ["", "Chambres : " + ", ".join(f"{k} × {v}" for k, v in sorted(self.chambers.items(), key=lambda kv: -kv[1])[:8])]
        lines += ["Années de décision : " + ", ".join(f"{k} × {v}" for k, v in sorted(self.years.items()))]
        lines += ["", "Contextes les plus fréquents (trois mots avant la marque) :", ""]
        for kind, ctx in self.contexts.items():
            top = ", ".join(f"« {c} » × {n}" for c, n in Counter(ctx).most_common(6))
            lines.append(f"- {kind} : {top}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "chars_median": self.chars_median,
            "chars_max": self.chars_max,
            "marks": self.marks,
            "letters_per_document": self.letters_per_document,
            "with_birth_date": self.with_birth_date,
            "with_address": self.with_address,
            "unknown_marks": dict(self.unknown_marks),
            "contexts": {k: dict(Counter(v).most_common(20)) for k, v in self.contexts.items()},
            "chambers": self.chambers,
            "years": self.years,
        }


def explore(records: list[dict]) -> Stats:
    """Ce qu'il y a dans le corpus : tailles, marques, contextes. Sert à cibler l'enrichissement des données."""
    marks: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    contexts: dict[str, Counter[str]] = {"personne": Counter(), "Adresse": Counter(), "Localité": Counter(), "Date naissance": Counter()}
    letters = 0
    with_birth = with_address = 0
    lengths = []
    chambers: Counter[str] = Counter()
    years: Counter[str] = Counter()
    for record in records:
        text = record["text"]
        lengths.append(len(text))
        chambers[str(record.get("chamber") or "?")] += 1
        years[str(record.get("decision_date") or "?")[:4]] += 1
        found_letters = LETTERS.findall(text)
        letters += len(found_letters)
        marks["personne (lettre)"] += len(found_letters)
        for m in LETTERS.finditer(text):
            contexts["personne"][_before(text, m.start())] += 1
        kinds = Counter(m.group("kind") for m in MARK.finditer(text))
        marks.update(kinds)
        with_birth += kinds["Date naissance"] > 0
        with_address += kinds["Adresse"] > 0
        for m in MARK.finditer(text):
            if m.group("kind") in contexts:
                contexts[m.group("kind")][_before(text, m.start())] += 1
        for m in ANY_MARK.finditer(text):
            if not LETTERS.fullmatch(m.group()) and not MARK.fullmatch(m.group()):
                unknown[re.sub(r"\d+", "n", m.group())] += 1
    return Stats(
        documents=len(records),
        chars_median=int(statistics.median(lengths)) if lengths else 0,
        chars_max=max(lengths, default=0),
        marks=dict(marks),
        letters_per_document=letters / len(records) if records else 0.0,
        with_birth_date=with_birth,
        with_address=with_address,
        unknown_marks=unknown,
        contexts={k: dict(v) for k, v in contexts.items()},
        chambers=dict(chambers),
        years=dict(years),
    )


def _before(text: str, pos: int, words: int = 3) -> str:
    """Les mots qui précèdent une marque sur la même ligne, marques voisines retirées."""
    line = text[max(0, pos - 80) : pos].rsplit("\n", 1)[-1]
    return " ".join(ANY_MARK.sub(" ", line).split()[-words:]).casefold()


# ---------------------------------------------------------------- ré-injection


class Reinjector:
    """Remplace les marques d'une décision par des valeurs synthétiques cohérentes, en notant chaque position."""

    def __init__(self, rng: random.Random, pools: Pools, upper_names: float = 0.5) -> None:
        self.rng = rng
        self.pools = pools
        self.upper_names = upper_names
        self.values: dict[str, tuple[str, str]] = {}  # marque → (valeur, catégorie)

    def _value(self, key: str, label: str, gender: str | None = None) -> str:
        if key not in self.values:
            if label == "personnePhysiquePrenom":
                by_gender = {"f": self.pools.prenoms_f, "m": self.pools.prenoms_m}
                value = self.rng.choice(by_gender.get(gender or "", self.pools.prenoms_f + self.pools.prenoms_m))
            elif label == "personnePhysiqueNom":
                value = self.rng.choice(self.pools.noms)
                if self.rng.random() < self.upper_names:
                    value = value.upper()
            elif label == "adresse":
                value = f"{self.rng.randint(1, 180)} {self.rng.choice(self.pools.voies)}"
            elif label == "localite":
                value = self.rng.choice(self.pools.villes)
            else:  # dateDeNaissance : jour et mois ; l'année suit en clair dans le texte publié
                value = f"{self.rng.randint(1, 28)} {self.rng.choice(MOIS)}"
            self.values[key] = (value, label)
        return self.values[key][0]

    def _label_of_letter(self, key: str, first_in_group: bool, group_size: int) -> str:
        if key in self.values:
            return self.values[key][1]
        return "personnePhysiquePrenom" if first_in_group and group_size > 1 else "personnePhysiqueNom"

    def document(self, record: dict) -> Document:
        text = record["text"]
        out: list[str] = []
        entities: list[Entity] = []
        pos = 0
        length = 0

        def emit(chunk: str, label: str | None = None) -> None:
            nonlocal length
            if label:
                entities.append(Entity(length, length + len(chunk), label, chunk, "gold"))
            out.append(chunk)
            length += len(chunk)

        for m in sorted(list(NAME_GROUP.finditer(text)) + list(MARK.finditer(text)), key=lambda m: m.start()):
            if m.start() < pos:
                continue
            emit(text[pos : m.start()])
            if m.re is NAME_GROUP:
                self._name_group(m.group(), emit, _gender(text[max(0, m.start() - 12) : m.start()]))
                pos = m.end()
            else:
                kind, label = m.group("kind"), LABEL_OF_MARK.get(m.group("kind"))
                key = f"{kind} {m.group('num') or ''}".strip()
                if label is None:
                    emit(m.group())  # marque non gérée : laissée telle quelle
                    pos = m.end()
                    continue
                value = self._value(key, label)
                year = YEAR_AFTER.match(text, m.end()) if label == "dateDeNaissance" else None
                if year:
                    emit(f"{value} {year.group('year')}", label)
                    pos = year.end()
                else:
                    emit(value, label)
                    pos = m.end()
        emit(text[pos:])
        return Document(id=str(record.get("id", "")), text="".join(out), entities=entities)

    def _name_group(self, group: str, emit: Callable[..., None], gender: str | None) -> None:
        """« [E] [A]-[M] » → prénom, puis nom (les lettres liées par un tiret forment un seul nom composé)."""
        parts = re.split(r"([ \-])", group)
        keys = [p for p in parts if p not in (" ", "-")]
        label_first = self._label_of_letter(keys[0][1:-1], True, len(keys))
        current: list[str] = []
        current_label: str | None = None

        def flush() -> None:
            if current and current_label:
                emit("".join(current), current_label)
                current.clear()

        for i, part in enumerate(parts):
            if part == " ":
                flush()
                emit(" ")
            elif part == "-":
                current.append("-")
            else:
                key = part[1:-1]
                label = label_first if i == 0 else self._label_of_letter(key, False, len(keys))
                if current_label and label != current_label:
                    flush()
                current_label = label
                current.append(self._value(key, label, gender))
        flush()


def _gender(before: str) -> str | None:
    """Civilité qui précède un nom : « Madame [E] [A] » → prénom féminin."""
    words = before.casefold().split()
    if not words:
        return None
    if words[-1] in {"madame", "mme", "mme.", "mlle", "mademoiselle"}:
        return "f"
    if words[-1] in {"monsieur", "m.", "mr", "mr."}:
        return "m"
    return None


def reinject(records: list[dict], seed: int = 0, pool: str = "test", professionals: bool = True) -> list[Document]:
    """Un ``Document`` annoté par décision ; valeurs tirées du pool demandé (``pools.py`` : ``train``, ``dev`` et
    ``test`` sont disjoints, ``test`` n'a rien vu de l'entraînement synthétique non plus).

    ``professionals`` ajoute une annotation *faible*, par règles, des avocats et magistrats en clair
    (voir ``annotate_professionals``) : utile pour l'entraînement, exclue de l'évaluation ``--labels reinjected``.
    """
    rng = random.Random(seed)
    docs = []
    for record in records:
        doc = Reinjector(rng, value_pool(pool)).document(record)
        if professionals:
            doc = annotate_professionals(doc)
        for e in doc.entities:
            assert doc.text[e.start : e.end] == e.text, (record.get("id"), e)
        docs.append(doc)
    return docs


# ---------------------------------------------------------------- professionnels (annotation faible)

CIVILITES = {"m", "m.", "mme", "mme.", "mlle", "mr", "monsieur", "madame", "mademoiselle", "me", "maître"}
CAP = r"[A-ZÀ-ÖØ-Þ][\w'’\-]*\.?"
AVOCAT = re.compile(rf"\b(?:Me|Maître)\s+((?:{CAP}\s+)*{CAP})")
TITRES = r"(?i:vice-présidente?|présidente?|conseill(?:er|ère)|greffi(?:er|ère)|magistrate?|juges?)(?i:\s+de\s+chambre)?"
MAGISTRAT_AVANT = re.compile(rf"((?:{CAP}[ \t]+){{0,4}}{CAP})(?=,?[ \t]*{TITRES}\b)")
MAGISTRAT_APRES = re.compile(rf"{TITRES}[ \t]*:[ \t]*(?:M\.|Mme|Madame|Monsieur)?[ \t]*((?:{CAP}[ \t]*){{1,4}})")
NOT_A_NAME = {
    "cour",
    "chambre",
    "appel",
    "tribunal",
    "arrêt",
    "composition",
    "de",
    "la",
    "le",
    "du",
    "des",
    "et",
    "en",
    "par",
    "président",
    "présidente",
    "conseiller",
    "conseillère",
    "greffier",
    "greffière",
    "magistrat",
    "juge",
    "vice-président",
}


def annotate_professionals(doc: Document) -> Document:
    """Avocats (« Me Prénom NOM ») et magistrats / greffiers (« Prénom NOM, présidente », « Greffier : Mme X »)
    en clair dans les décisions publiées : étiquetés par règles, un token par entité comme dans le corpus
    synthétique, sans recouvrir une entité déjà présente. Annotation faible : sert à l'entraînement, pas au score."""
    found: list[Entity] = []
    taken = list(doc.entities)

    def add(start: int, end: int, label: str) -> None:
        for token in re.finditer(r"\S+", doc.text[start:end]):
            a, b = start + token.start(), start + token.end()
            word = token.group().rstrip(".,")
            b = a + len(word)
            if not word or word.casefold() in CIVILITES or word.casefold() in NOT_A_NAME or not word[0].isupper():
                continue
            ent = Entity(a, b, label, word, "rule")
            if any(ent.overlaps(other) for other in taken):
                continue
            taken.append(ent)
            found.append(ent)

    for m in AVOCAT.finditer(doc.text):
        add(m.start(1), m.end(1), "professionnelAvocat")
    for m in MAGISTRAT_AVANT.finditer(doc.text):
        add(m.start(1), m.end(1), "professionnelMagistratGreffier")
    for m in MAGISTRAT_APRES.finditer(doc.text):
        add(m.start(1), m.end(1), "professionnelMagistratGreffier")
    return Document(doc.id, doc.text, sorted(doc.entities + found))


# ---------------------------------------------------------------- partage entraînement / test


def split(records: list[dict], test_share: float = 0.2, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Partage *par décision* (jamais une même décision des deux côtés), reproductible."""
    rng = random.Random(seed)
    shuffled = sorted(records, key=lambda r: str(r.get("id")))
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_share))
    return shuffled[n_test:], shuffled[:n_test]
