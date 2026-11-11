"""NER Flair : BiLSTM-CRF sur embeddings contextuels de caractères (fr-forward / fr-backward).

C'est l'architecture du moteur public de la Cour de cassation (``moteurNER``),
qui y ajoute des Byte-Pair Embeddings et un modèle de langage ré-entraîné sur
2 millions de décisions. Ici : embeddings Flair français pré-entraînés tels
quels, pour comparer l'architecture à budget égal.
"""

from __future__ import annotations

from pathlib import Path

from pseudo_ner.categories import MODEL_LABELS, Entity
from pseudo_ner.synthetic import Document
from pseudo_ner.tokenize import Token, sentences, spans_from_tags


def shape(word: str) -> str:
    """Forme du mot (Xxx, XXX, dd, xx, …) : permet de généraliser à des noms jamais vus."""
    out = []
    for ch in word[:6]:
        out.append("X" if ch.isupper() else "x" if ch.isalpha() else "d" if ch.isdigit() else ch)
    compact = "".join(out)
    while "XX" in compact or "xx" in compact or "dd" in compact:
        compact = compact.replace("XX", "X").replace("xx", "x").replace("dd", "d")
    return compact


def _sentence(words: list[str]):
    from flair.data import Sentence as FlairSentence

    fs = FlairSentence(words)
    for tok in fs.tokens:
        tok.add_label("shape", shape(tok.text))
    return fs


def _flair_sentences(docs: list[Document]):
    out = []
    for d in docs:
        for s in sentences(d.text, [e for e in d.entities if e.label in MODEL_LABELS]):
            fs = _sentence(s.words)
            start = None
            for i, tag in enumerate(s.tags + ["O"]):
                if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and start is None):
                    if start is not None:
                        fs[start:i].add_label("ner", s.tags[start][2:])
                    start = i if tag != "O" else None
            out.append(fs)
    return out


class FlairNer:
    """``SequenceTagger`` Flair, étiquetage BIO."""

    name = "flair"

    def __init__(self, tagger=None) -> None:
        self.tagger = tagger

    def train(
        self,
        train: list[Document],
        dev: list[Document],
        out_dir: Path,
        epochs: int = 10,
        hidden_size: int = 128,
        learning_rate: float = 0.1,
        mini_batch_size: int = 16,
        embeddings: str = "flair-fr",
        **_: object,
    ) -> Path:
        """``embeddings`` : ``flair-fr`` (fr-forward + fr-backward, téléchargés) ou ``onehot`` (mot + forme, appris de zéro, hors ligne)."""
        import flair
        import torch
        from flair.data import Corpus
        from flair.datasets import FlairDatapointDataset
        from flair.embeddings import FlairEmbeddings, OneHotEmbeddings, StackedEmbeddings
        from flair.models import SequenceTagger
        from flair.trainers import ModelTrainer

        flair.set_seed(0)
        flair.device = torch.device("cpu")
        corpus = Corpus(
            train=FlairDatapointDataset(_flair_sentences(train)),
            dev=FlairDatapointDataset(_flair_sentences(dev)),
            test=FlairDatapointDataset(_flair_sentences(dev[:1])),
        )
        label_dict = corpus.make_label_dictionary(label_type="ner", add_unk=False)
        if embeddings == "flair-fr":
            stack = StackedEmbeddings([FlairEmbeddings("fr-forward"), FlairEmbeddings("fr-backward")])
        elif embeddings == "onehot":
            stack = StackedEmbeddings(
                [
                    OneHotEmbeddings.from_corpus(corpus, min_freq=2, embedding_length=64),
                    OneHotEmbeddings.from_corpus(corpus, field="shape", min_freq=1, embedding_length=16),
                ]
            )
        else:
            raise ValueError(f"embeddings inconnus : {embeddings}")
        tagger = SequenceTagger(
            hidden_size=hidden_size,
            embeddings=stack,
            tag_dictionary=label_dict,
            tag_type="ner",
            use_crf=True,
            tag_format="BIO",
        )
        trainer = ModelTrainer(tagger, corpus)
        trainer.train(
            str(out_dir),
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=epochs,
            patience=2,
            anneal_factor=0.5,
            embeddings_storage_mode="none",
            write_weights=False,
            save_final_model=False,
            create_file_logs=False,
            create_loss_file=False,
        )
        self.tagger = SequenceTagger.load(out_dir / "best-model.pt")
        _restore_stack_order(self.tagger)
        return out_dir

    def predict(self, text: str) -> list[Entity]:
        assert self.tagger is not None, "modèle non chargé"
        entities: list[Entity] = []
        for s in sentences(text):
            fs = _sentence(s.words)
            self.tagger.predict(fs, label_name="ner")
            tags = ["O"] * len(s.tokens)
            for span in fs.get_spans("ner"):
                first, last = span[0].idx - 1, span[-1].idx - 1
                tags[first] = "B-" + span.tag
                for i in range(first + 1, last + 1):
                    tags[i] = "I-" + span.tag
            tokens = [Token(t.text, t.start, t.end) for t in s.tokens]
            entities.extend(spans_from_tags(tokens, tags, text))
        return entities

    @classmethod
    def load(cls, path: Path) -> FlairNer:
        from flair.models import SequenceTagger

        tagger = SequenceTagger.load(path / "best-model.pt")
        _restore_stack_order(tagger)
        return cls(tagger)


def _restore_stack_order(tagger) -> None:
    """Contourne un défaut de Flair : ``StackedEmbeddings`` nomme ses sous-embeddings ``0-…``, ``1-…``
    à l'entraînement, mais pas au rechargement ; or la concaténation suit l'ordre alphabétique des noms.
    Sans ce correctif, l'entrée du LSTM est permutée et le modèle rechargé prédit n'importe quoi."""
    stack = getattr(tagger.embeddings, "embeddings", None)
    if not stack:
        return
    for i, emb in enumerate(stack):
        if not emb.name.startswith(f"{i}-"):
            emb.name = f"{i}-{emb.name}"
