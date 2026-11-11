"""NER par Transformer : CamemBERT fine-tuné en classification de tokens (Hugging Face / PyTorch).

Alignement sous-mots → mots : seule la première pièce de chaque mot porte
l'étiquette (les autres sont ignorées dans la perte), prédiction reportée sur
le mot entier, puis conversion BIO → caractères par le tokenizer commun.
"""

from __future__ import annotations

import json
from pathlib import Path

from pseudo_ner.categories import MODEL_LABELS, Entity
from pseudo_ner.synthetic import Document
from pseudo_ner.tokenize import Token, sentences, spans_from_tags

BASE = "camembert-base"
MAX_LEN = 192


def _tags() -> list[str]:
    return ["O"] + [f"{p}-{lab}" for lab in MODEL_LABELS for p in ("B", "I")]


def paragraphs(texts: list[str], min_chars: int = 40, max_chars: int = 1200) -> list[str]:
    """Découpe des décisions en paragraphes pour le pré-entraînement : lignes non vides, trop courtes
    exclues, trop longues coupées sur les points."""
    out: list[str] = []
    for text in texts:
        for line in text.splitlines():
            line = line.strip()
            if len(line) < min_chars:
                continue
            while len(line) > max_chars:
                cut = line.rfind(". ", 0, max_chars)
                cut = cut + 1 if cut > min_chars else max_chars
                out.append(line[:cut].strip())
                line = line[cut:].strip()
            if len(line) >= min_chars:
                out.append(line)
    return out


def adapt(
    texts: list[str],
    out_dir: Path,
    base: str = BASE,
    epochs: int = 1,
    max_len: int = 256,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
) -> Path:
    """Pré-entraînement adapté au domaine (« don't stop pretraining ») : on continue l'objectif de
    masquage de CamemBERT sur le texte des décisions, sans étiquette, avant l'affinage NER.

    C'est, à petite échelle, ce que fait la Cour avec son modèle de langage ré-entraîné sur
    deux millions de décisions. Le modèle produit se passe ensuite à ``train camembert --base``.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForMaskedLM.from_pretrained(base)
    parts = paragraphs(texts)
    ds = Dataset.from_dict({"text": parts}).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len), batched=True, remove_columns=["text"]
    )
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        seed=0,
        use_cpu=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15),
    )
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "adapted_from.json").write_text(
        json.dumps({"base": base, "paragraphs": len(parts), "epochs": epochs, "max_len": max_len}), encoding="utf-8"
    )
    return out_dir


class CamembertNer:
    """``AutoModelForTokenClassification`` sur CamemBERT."""

    name = "camembert"

    def __init__(self, model=None, tokenizer=None, tags: list[str] | None = None) -> None:
        self.model, self.tokenizer, self.tags = model, tokenizer, tags or _tags()

    def _encode(self, docs: list[Document]):
        from datasets import Dataset

        tag2id = {t: i for i, t in enumerate(self.tags)}
        words, labels = [], []
        for d in docs:
            for s in sentences(d.text, [e for e in d.entities if e.label in MODEL_LABELS]):
                for lo, hi in self._windows(s.words):  # mêmes fenêtres qu'à la prédiction
                    words.append(s.words[lo:hi])
                    labels.append([tag2id[t] for t in s.tags[lo:hi]])
        ds = Dataset.from_dict({"words": words, "labels": labels})

        def tokenize(batch):
            enc = self.tokenizer(batch["words"], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
            aligned = []
            for i, labs in enumerate(batch["labels"]):
                word_ids = enc.word_ids(batch_index=i)
                prev, row = None, []
                for w in word_ids:
                    if w is None or w == prev:
                        row.append(-100)
                    else:
                        row.append(labs[w])
                    prev = w
                aligned.append(row)
            enc["labels"] = aligned
            return enc

        return ds.map(tokenize, batched=True, remove_columns=["words", "labels"])

    def train(
        self,
        train: list[Document],
        dev: list[Document],
        out_dir: Path,
        epochs: int = 2,
        base: str = BASE,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        **_: object,
    ) -> Path:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )

        torch.manual_seed(0)
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForTokenClassification.from_pretrained(
            base,
            num_labels=len(self.tags),
            id2label=dict(enumerate(self.tags)),
            label2id={t: i for i, t in enumerate(self.tags)},
        )
        args = TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=20,
            report_to=[],
            seed=0,
            use_cpu=True,
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self._encode(train),
            eval_dataset=self._encode(dev),
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
        )
        trainer.train()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        (out_dir / "tags.json").write_text(json.dumps(self.tags), encoding="utf-8")
        self.model.eval()
        return out_dir

    def _windows(self, words: list[str]) -> list[tuple[int, int]]:
        """Bornes (début, fin) de fenêtres de mots dont les sous-mots tiennent dans ``MAX_LEN``.

        Les en-têtes de décision (parties, adresses, sans ponctuation) forment une seule « phrase » de
        plusieurs centaines de mots : tronquer à ``MAX_LEN`` ferait disparaître tout ce qui suit. On coupe
        donc par fenêtres de mots entiers, sans chevauchement, jamais au milieu d'un mot.
        """
        assert self.tokenizer is not None
        budget = MAX_LEN - 2  # <s> et </s>
        lengths = [max(1, len(self.tokenizer.tokenize(w))) for w in words]
        windows, start, used = [], 0, 0
        for i, n in enumerate(lengths):
            if used + n > budget and i > start:
                windows.append((start, i))
                start, used = i, 0
            used += n
        windows.append((start, len(words)))
        return windows

    def predict(self, text: str) -> list[Entity]:
        import torch

        assert self.model is not None and self.tokenizer is not None, "modèle non chargé"
        entities: list[Entity] = []
        for s in sentences(text):
            tags = ["O"] * len(s.words)
            for lo, hi in self._windows(s.words):
                enc = self.tokenizer(s.words[lo:hi], is_split_into_words=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                with torch.no_grad():
                    logits = self.model(**enc).logits[0]
                pred_ids = logits.argmax(-1).tolist()
                prev = None
                for pos, w in enumerate(enc.word_ids(0)):
                    if w is not None and w != prev:
                        tags[lo + w] = self.tags[pred_ids[pos]]
                    prev = w
            tokens = [Token(t.text, t.start, t.end) for t in s.tokens]
            entities.extend(spans_from_tags(tokens, tags, text))
        return entities

    @classmethod
    def load(cls, path: Path) -> CamembertNer:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        tags = json.loads((path / "tags.json").read_text(encoding="utf-8"))
        model = AutoModelForTokenClassification.from_pretrained(path)
        model.eval()
        return cls(model, AutoTokenizer.from_pretrained(path), tags)
