"""Independent all-span biaffine NER.

Scientific reference: Yu, Bohnet and Poesio (ACL 2020), *Named Entity
Recognition as Dependency Parsing*, https://aclanthology.org/2020.acl-main.577/.
Separate start/end projections and a class-specific biaffine score implement
the paper's span-boundary classification principle. Every contiguous token
span competes with a none class; flat decoding rejects overlapping spans.

This controlled variant is NOT a reproduction of that paper's complete model:
it uses frozen task-trained biomedbert-small states (384 dimensions), without
BERT-large, fastText, character CNN or the paper's BiLSTM stack. It does not use
ACT-NER candidates, CRF evidence, lexical priors, routing or anchor overrides.
Training includes original Train plus the permitted inner-training documents;
ACT-NER's refinement head uses only the latter. This support asymmetry and the
upstream backbone's historical Validation selection preclude an unconditional
independent-generalization interpretation of the conditional cross-fit result.

Nothing trains or writes on import. The notebook must first call
freeze_protocol(root), inspect its output, and then explicitly invoke oneunit.
All historical artifacts remain read-only. Synthetic selftest() is software
verification, not evidence of experimental model quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable
import hashlib
import json
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from core import ENTITY_TYPES, finalize_prediction_frame, gold_for_metrics
from prospective_model import strict_metrics
from refinement_analysis import counts_by_document, f1_counts, paired_evidence
from refinement_experiment import role_partition


RUN = "extended_20261016"
OLD_RUN = "refinement_20260907"
MODEL = "Independent biaffine NER (frozen backbone)"
SEEDS = (17, 29, 43)
SETTINGS = {
    "hidden_size": 384, "projection_size": 128, "dropout": 0.2,
    "max_epochs": 8, "patience": 3, "batch_size": 512,
    "learning_rate": 0.001, "weight_decay": 0.0001,
    "gradient_clip": 1.0, "negative_per_positive": 10,
    "minimum_negatives_per_sentence": 32, "score_batch_size": 8192,
    "threshold_grid": [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "class_weights": "none=1; positive inverse-sqrt frequency, mean=1, clipped [0.5,3]",
    "selection": "innerSelection strict typed F1; ties boundary F1, recall, macro F1, lower threshold, earlier epoch",
    "candidate_policy": "All contiguous WordPiece start/end pairs; no maximum width; no gold injection",
    "negative_sampling": "All representable positives and uniform none-class negatives, redrawn each epoch",
    "decode": "Argmax over none+four types; entity probability threshold; descending-score non-overlap",
    "calibration": "None fitted; confidence columns contain raw probabilities explicitly marked uncalibrated",
}


def _root(root: Path) -> Path:
    result = Path(root).resolve()
    assert result.name == "01_Drug_Entity_Recognition", "Incorrect project root"
    return result


def _hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def _report(root: Path) -> Path:
    return root / "Reports" / RUN / "span_comparator"


def _json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _inputs(root: Path) -> dict[str, str]:
    paths = [root / "Models/Comparators" / f"context_{split}.pt"
             for split in ("train", "validation")]
    paths += [root / "dataset/Processed" / f"{split}_{kind}.parquet"
              for split in ("train", "validation")
              for kind in ("sentences", "entities", "tokens")]
    paths += [root / "Models/Comparators/biomedical_transformer.pt",
              root / "Reports" / OLD_RUN / "membership.csv",
              root / "src/refinement_freeze.json", Path(__file__).resolve()]
    paths += [root / "src" / name for name in (
        "core.py", "refinement_experiment.py", "prospective_data.py",
        "prospective_model.py", "refinement_analysis.py")]
    return {str(p): _hash(p) for p in paths}


def _roles(root: Path, fold: int) -> tuple[pd.DataFrame, dict[str, set]]:
    documents = pd.read_csv(root / "Reports" / OLD_RUN / "membership.csv")
    assert len(documents) == 142 and documents.document_id.is_unique
    return documents, role_partition(documents, fold)


def freeze_protocol(root: Path) -> dict:
    """Write/check an immutable, preregistered computational protocol; no fit.

    Existing files must match exactly. Source or configuration changes after
    freezing require a new run namespace, not replacement of this protocol.
    """
    root = _root(root)
    audit = json.loads((root / "Reports" / RUN / "pretraining_audit.json").read_text())
    assert audit["audit_completed"], "Complete the six-notebook audit before training"
    roles = {str(f): {k: sorted(v) for k, v in _roles(root, f)[1].items()} for f in range(5)}
    protocol = {
        "run": RUN, "model": MODEL, "seeds": list(SEEDS), "outer_folds": 5,
        "settings": SETTINGS, "classes": ["none", *ENTITY_TYPES], "roles": roles,
        "input_sha256": _inputs(root),
        "fit_support": "Original Train + innerTrain; no innerSelection/calibration/outer labels in optimization",
        "upstream_scope": "Conditional cross-fit; frozen backbone previously selected using original Validation",
        "head_support_difference": "Biaffine additionally fits original Train; ACT refinement fits innerTrain only",
        "text_dependence": "Existing Train-Validation shared-text exposure retained and disclosed, not relabelled independent",
        "external_test_used": False, "historical_test_used": False,
        "statistics": "142 documents / 140 text-dependency clusters; average three seed F1 differences; 5000 paired cluster resamples",
        "holm_family": ["Full ACT-NER minus independent biaffine", "Parameter-matched span MLP minus independent biaffine"],
        "source": "https://aclanthology.org/2020.acl-main.577/",
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
    }
    path = _report(root) / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Frozen protocol changed: use a new run identifier"
    else:
        _json_new(path, protocol)
        _json_new(path.with_name("registration.json"), {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": _hash(path), "training_started": False})
    return protocol


def _verify_protocol(root: Path) -> dict:
    path = _report(root) / "protocol.json"
    assert path.exists(), "Inspect and freeze the protocol before invoking oneunit"
    protocol = json.loads(path.read_text())
    assert protocol["settings"] == SETTINGS and protocol["input_sha256"] == _inputs(root)
    assert protocol["seeds"] == list(SEEDS)
    return protocol


class BiaffineSpanNER(nn.Module):
    """Class-specific augmented bilinear matrix includes both linear terms."""

    def __init__(self, hidden_size: int = 384, projection_size: int = 128,
                 dropout: float = 0.2, classes: int = 5):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(hidden_size, projection_size), nn.GELU(), nn.Dropout(dropout))
        self.end = nn.Sequential(nn.Linear(hidden_size, projection_size), nn.GELU(), nn.Dropout(dropout))
        self.weight = nn.Parameter(torch.empty(classes, projection_size + 1, projection_size + 1))
        nn.init.xavier_uniform_(self.weight)

    def project(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.start(hidden), self.end(hidden)
        return F.pad(start, (0, 1), value=1), F.pad(end, (0, 1), value=1)

    def pair_scores(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,cij,bj->bc", start, self.weight, end)

    def forward(self, start_hidden: torch.Tensor, end_hidden: torch.Tensor) -> torch.Tensor:
        start = F.pad(self.start(start_hidden), (0, 1), value=1)
        end = F.pad(self.end(end_hidden), (0, 1), value=1)
        return self.pair_scores(start, end)


def enumerate_spans(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Input length alone determines candidates; gold is never an argument."""
    assert length >= 0
    return tuple(x.astype(np.int64) for x in np.triu_indices(length))


@dataclass
class SentenceSpans:
    document: str
    sentence: str
    offset: int
    char_start: np.ndarray
    char_end: np.ndarray
    start: np.ndarray
    end: np.ndarray
    targets: np.ndarray


@dataclass
class Corpus:
    sentences: pd.DataFrame
    entities: pd.DataFrame
    hidden: torch.Tensor
    parts: list[SentenceSpans]
    train_documents: set
    validation_documents: set


@lru_cache(maxsize=1)
def _corpus(root_string: str) -> Corpus:
    """Load only Train/Validation; cache format/order is checked, not inferred."""
    root = _root(Path(root_string))
    frames, gold_frames, pieces, parts, document_sets = [], [], [], [], {}
    offset = 0
    for split in ("train", "validation"):
        folder = root / "dataset/Processed"
        sentences = pd.read_parquet(folder / f"{split}_sentences.parquet")
        entities = pd.read_parquet(folder / f"{split}_entities.parquet")
        tokens = pd.read_parquet(folder / f"{split}_tokens.parquet")
        hidden = torch.load(root / f"Models/Comparators/context_{split}.pt",
                            map_location="cpu", weights_only=False)
        gold = gold_for_metrics(entities)
        assert set(hidden) == set(tokens.sentence_id)
        assert set(gold.sentence_id) <= set(hidden), "Gold in uncached sentences requires investigation"
        assert not tokens.duplicated(["sentence_id", "char_start", "char_end"]).any()
        assert not gold.duplicated(["sentence_id", "char_start", "char_end"]).any()
        truth = {(r.sentence_id, r.char_start, r.char_end): ENTITY_TYPES.index(r.entity_type) + 1
                 for r in gold.itertuples()}
        for sid, frame in tokens.groupby("sentence_id", sort=True):
            frame = frame.sort_values("token_index")
            assert np.array_equal(frame.token_index, np.arange(len(frame)))
            h = hidden[sid].detach().cpu()
            assert h.shape == (len(frame), SETTINGS["hidden_size"]) and h.dtype == torch.float32
            assert torch.isfinite(h).all()
            left, right = enumerate_spans(len(frame))
            starts, ends = frame.char_start.to_numpy(), frame.char_end.to_numpy()
            targets = np.fromiter((truth.get((sid, int(starts[i]), int(ends[j])), 0)
                                  for i, j in zip(left, right)), dtype=np.int64, count=len(left))
            parts.append(SentenceSpans(str(frame.document_id.iloc[0]), str(sid), offset,
                                       starts, ends, left, right, targets))
            pieces.append(h)
            offset += len(frame)
        frames.append(sentences)
        gold_frames.append(entities)
        document_sets[split] = set(sentences.document_id)
    assert document_sets["train"].isdisjoint(document_sets["validation"])
    all_sentences = pd.concat(frames, ignore_index=True)
    assert all_sentences.sentence_id.is_unique
    return Corpus(all_sentences, pd.concat(gold_frames, ignore_index=True), torch.cat(pieces),
                  parts, document_sets["train"], document_sets["validation"])


def coverage(corpus: Corpus, documents: set, role: str) -> dict:
    selected = [p for p in corpus.parts if p.document in documents]
    gold = gold_for_metrics(corpus.entities.loc[corpus.entities.document_id.isin(documents)])
    representable = sum(int(np.count_nonzero(p.targets)) for p in selected)
    assert representable <= len(gold)
    return {"Role": role, "Documents": len(documents), "Sentences": int(corpus.sentences.document_id.isin(documents).sum()),
            "Candidate spans": sum(len(p.start) for p in selected), "Gold entities": len(gold),
            "Representable gold": representable, "Unrepresentable gold retained": len(gold) - representable,
            "Candidate typed coverage": representable / len(gold) if len(gold) else None}


def _training_pairs(corpus: Corpus, documents: set, rng: np.random.Generator) -> tuple:
    starts, ends, labels = [], [], []
    for part in corpus.parts:
        if part.document not in documents:
            continue
        positive, negative = np.flatnonzero(part.targets), np.flatnonzero(part.targets == 0)
        count = min(len(negative), max(SETTINGS["minimum_negatives_per_sentence"],
                                     SETTINGS["negative_per_positive"] * len(positive)))
        indices = np.concatenate((positive, rng.choice(negative, count, replace=False)))
        starts.append(part.start[indices] + part.offset)
        ends.append(part.end[indices] + part.offset)
        labels.append(part.targets[indices])
    values = tuple(np.concatenate(x) for x in (starts, ends, labels))
    order = rng.permutation(len(values[0]))
    return tuple(x[order] for x in values)


def _class_weights(corpus: Corpus, documents: set) -> torch.Tensor:
    counts = sum((np.bincount(p.targets[p.targets > 0], minlength=5)
                  for p in corpus.parts if p.document in documents), start=np.zeros(5, dtype=int))
    assert (counts[1:] > 0).all(), "Every entity class must occur in fitting data"
    positive = 1 / np.sqrt(counts[1:])
    positive = np.clip(positive / positive.mean(), 0.5, 3)
    return torch.tensor(np.r_[1.0, positive], dtype=torch.float32)


@torch.inference_mode()
def score_all(model: BiaffineSpanNER, hidden: torch.Tensor, corpus: Corpus,
              documents: set) -> pd.DataFrame:
    """Exhaustive scores are generated without consulting targets or gold."""
    model.eval()
    meta, starts, ends, token_ids = [], [], [], []
    local_offset = 0
    for part in corpus.parts:
        if part.document not in documents:
            continue
        n = len(part.start)
        meta.append(pd.DataFrame({"document_id": np.repeat(part.document, n),
            "sentence_id": np.repeat(part.sentence, n), "token_start": part.start,
            "token_end": part.end + 1, "char_start": part.char_start[part.start],
            "char_end": part.char_end[part.end]}))
        starts.append(part.start + local_offset)
        ends.append(part.end + local_offset)
        token_ids.append(np.arange(len(part.char_start)) + part.offset)
        local_offset += len(part.char_start)
    assert meta, "No token-containing sentences in requested documents"
    starts, ends = np.concatenate(starts), np.concatenate(ends)
    token_ids = torch.as_tensor(np.concatenate(token_ids), device=hidden.device)
    start_h, end_h = model.project(hidden[token_ids])
    # Compute the expensive bilinear transform once per token, not per span.
    weighted_start = torch.einsum("ni,cij->ncj", start_h, model.weight)
    chunks = []
    for begin in range(0, len(starts), SETTINGS["score_batch_size"]):
        end = begin + SETTINGS["score_batch_size"]
        a = torch.as_tensor(starts[begin:end], device=hidden.device)
        b = torch.as_tensor(ends[begin:end], device=hidden.device)
        logits = (weighted_start[a] * end_h[b, None, :]).sum(-1)
        chunks.append(logits.softmax(-1).cpu().numpy())
    probabilities = np.concatenate(chunks)
    scores = pd.concat(meta, ignore_index=True)
    for index, name in enumerate(("none", *ENTITY_TYPES)):
        scores[f"p_{name}"] = probabilities[:, index]
    scores["span_probability"] = 1 - probabilities[:, 0]
    scores["entity_probability"] = probabilities[:, 1:].max(1)
    scores["type_probability"] = scores.entity_probability / scores.span_probability.clip(lower=1e-12)
    scores["predicted_type"] = np.asarray(ENTITY_TYPES)[probabilities[:, 1:].argmax(1)]
    scores["non_none_argmax"] = probabilities.argmax(1) != 0
    assert not scores.duplicated(["sentence_id", "char_start", "char_end"]).any()
    return scores


def decode(scores: pd.DataFrame, sentences: pd.DataFrame, threshold: float,
           seed: int, split: str = "Validation conditional cross-fit") -> pd.DataFrame:
    selected = scores.loc[scores.non_none_argmax & scores.entity_probability.ge(threshold)]
    emitted = []
    for _, frame in selected.groupby("sentence_id", sort=False):
        occupied = []
        for row in frame.sort_values(["entity_probability", "char_start", "char_end"],
                                     ascending=[False, True, True], kind="stable").itertuples():
            if any(row.char_start < right and left < row.char_end for left, right in occupied):
                continue
            occupied.append((row.char_start, row.char_end))
            emitted.append({"document_id": row.document_id, "sentence_id": row.sentence_id,
                "char_start": int(row.char_start), "char_end": int(row.char_end),
                "token_start": int(row.token_start), "token_end": int(row.token_end),
                "predicted_type": row.predicted_type, "boundary_probability": row.span_probability,
                "type_probability": row.type_probability})
    result = finalize_prediction_frame(emitted, sentences, MODEL, split, seed)
    result["calibrated_confidence"] = result.boundary_probability * result.type_probability
    result["confidence_category"] = "uncalibrated"
    result["review_required"] = False
    return result


def _selection(scores: pd.DataFrame, sentences: pd.DataFrame, entities: pd.DataFrame, seed: int) -> tuple:
    table = pd.DataFrame([{"threshold": t, **strict_metrics(entities, decode(scores, sentences, t, seed))}
                          for t in SETTINGS["threshold_grid"]])
    row = table.sort_values(["typed_f1", "boundary_f1", "typed_recall", "macro_f1", "threshold"],
                            ascending=[False, False, False, False, True], kind="stable").iloc[0]
    key = tuple(float(row[k]) for k in ("typed_f1", "boundary_f1", "typed_recall", "macro_f1")) + (-float(row.threshold),)
    return row.to_dict(), table, key


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _unit_paths(root: Path, fold: int, seed: int) -> tuple[Path, Path, Path]:
    relative = Path("biaffine") / f"fold_{fold}" / f"seed_{seed}"
    return (root / "Models/Extended" / RUN / relative,
            root / "Predictions/Extended" / RUN / relative,
            _report(root) / relative)


def _verified_completion(root: Path, fold: int, seed: int) -> dict | None:
    path = _unit_paths(root, fold, seed)[2] / "complete.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    assert result["protocol_sha256"] == _hash(_report(root) / "protocol.json")
    assert result["fold"] == fold and result["seed"] == seed
    for relative, expected in result["artifacts_sha256"].items():
        assert _hash(root / relative) == expected, f"Changed completed artifact: {relative}"
    return result


def _train_epoch(model, hidden, corpus, fit_ids, optimizer, class_weights, seed,
                 epoch, device, emit) -> tuple[float, int]:
    starts, ends, labels = _training_pairs(corpus, fit_ids, np.random.default_rng(seed * 1009 + epoch))
    model.train()
    total, seen = 0.0, 0
    batches = (len(labels) + SETTINGS["batch_size"] - 1) // SETTINGS["batch_size"]
    for batch, begin in enumerate(range(0, len(labels), SETTINGS["batch_size"]), 1):
        finish = begin + SETTINGS["batch_size"]
        a, b, y = (torch.as_tensor(x[begin:finish], device=device) for x in (starts, ends, labels))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(hidden[a], hidden[b]), y, weight=class_weights)
        assert torch.isfinite(loss), "Nonfinite training loss"
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_clip"])
        optimizer.step()
        total += float(loss.detach().cpu()) * len(y)
        seen += len(y)
        if batch == 1 or batch % 25 == 0 or batch == batches:
            emit({"stage": "training", "epoch": epoch, "batch": batch, "batches": batches,
                  "mean_loss": total / seen})
    return total / seen, seen


def oneunit(root: Path, fold: int, seed: int,
            progress: Callable[[dict], None] | None = None) -> dict:
    """Train one explicitly requested fit; completed, hash-verified units skip.

    Partial attempts are retained and not overwritten. A failed attempt needs
    an explicit new run namespace, so no damaged unit is silently resumed.
    """
    root = _root(root)
    assert fold in range(5) and seed in SEEDS
    protocol = _verify_protocol(root)
    existing = _verified_completion(root, fold, seed)
    if existing:
        if progress:
            progress({"stage": "existing unit", "model": MODEL, "fold": fold, "seed": seed})
        return existing
    model_dir, prediction_dir, report_dir = _unit_paths(root, fold, seed)
    assert not any(p.exists() for p in (model_dir, prediction_dir, report_dir)), "Partial attempt retained; inspect before creating a new run"
    for path in (model_dir, prediction_dir, report_dir):
        path.mkdir(parents=True)

    def emit(row):
        event = {"model": MODEL, "fold": fold, "seed": seed, **row}
        with (report_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if progress:
            progress(event)

    emit({"stage": "input verification complete"})
    try:
        result = _fit_unit(root, fold, seed, protocol, model_dir, prediction_dir, report_dir, emit)
    except Exception as error:
        emit({"stage": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise
    emit({"stage": "complete", "typed_f1": result["metrics"]["typed_f1"]})
    return result


def _fit_unit(root, fold, seed, protocol, model_dir, prediction_dir, report_dir, emit):
    corpus = _corpus(str(root))
    documents, roles = _roles(root, fold)
    assert {k: sorted(v) for k, v in roles.items()} == protocol["roles"][str(fold)]
    assert corpus.validation_documents == set(documents.document_id)
    fit_ids = corpus.train_documents | roles["train"]
    assert fit_ids.isdisjoint(roles["selection"] | roles["calibration"] | roles["outer"])
    _json_new(report_dir / "roles.json", {"fit_original_train": sorted(corpus.train_documents),
        **{k: sorted(v) for k, v in roles.items()}})
    pd.DataFrame([coverage(corpus, ids, role) for role, ids in
                  {"fitting": fit_ids, "selection": roles["selection"], "outer": roles["outer"]}.items()]).to_csv(report_dir / "coverage.csv", index=False)
    selection_s = corpus.sentences.loc[corpus.sentences.document_id.isin(roles["selection"])]
    selection_e = corpus.entities.loc[corpus.entities.document_id.isin(roles["selection"])]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device()
    model = BiaffineSpanNER(SETTINGS["hidden_size"], SETTINGS["projection_size"], SETTINGS["dropout"]).to(device)
    hidden = corpus.hidden.to(device)
    weights = _class_weights(corpus, fit_ids).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"],
                                 weight_decay=SETTINGS["weight_decay"])
    best_key, best, stale, history, threshold_tables = None, None, 0, [], []
    started = time.perf_counter()
    for epoch in range(1, SETTINGS["max_epochs"] + 1):
        mean_loss, n_examples = _train_epoch(model, hidden, corpus, fit_ids, optimizer, weights,
                                            seed, epoch, device, emit)
        emit({"stage": "innerSelection exhaustive inference", "epoch": epoch})
        selection_scores = score_all(model, hidden, corpus, roles["selection"])
        selected, thresholds, key = _selection(selection_scores, selection_s, selection_e, seed)
        threshold_tables.append(thresholds.assign(epoch=epoch))
        history.append({"epoch": epoch, "loss": mean_loss, "sampled_spans": n_examples,
                        **{f"selection_{k}": v for k, v in selected.items()}})
        pd.DataFrame(history).to_csv(report_dir / "history.csv", index=False)
        emit({"stage": "epoch complete", "epoch": epoch, "selection_typed_f1": selected["typed_f1"],
              "threshold": selected["threshold"], "loss": mean_loss})
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            best = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch, "threshold": selected["threshold"], "selection": selected}
        else:
            stale += 1
        if stale >= SETTINGS["patience"]:
            break
    _synchronize(device)
    training_seconds = time.perf_counter() - started
    assert best is not None
    checkpoint = {**best, "model": MODEL, "seed": seed, "fold": fold,
        "settings": SETTINGS, "roles": {k: sorted(v) for k, v in roles.items()},
        "fit_original_train": sorted(corpus.train_documents), "class_weights": weights.cpu(),
        "protocol_sha256": _hash(_report(root) / "protocol.json")}
    torch.save(checkpoint, model_dir / "model.pt")
    pd.concat(threshold_tables, ignore_index=True).to_csv(report_dir / "selection_thresholds.csv", index=False)
    model.load_state_dict(best["state_dict"])
    emit({"stage": "frozen outer-fold exhaustive inference", "selected_epoch": best["epoch"]})
    _synchronize(device)
    started = time.perf_counter()
    scores = score_all(model, hidden, corpus, roles["outer"])
    outer_s = corpus.sentences.loc[corpus.sentences.document_id.isin(roles["outer"])]
    outer_e = corpus.entities.loc[corpus.entities.document_id.isin(roles["outer"])]
    prediction = decode(scores, outer_s, best["threshold"], seed)
    _synchronize(device)
    inference_seconds = time.perf_counter() - started
    assert set(prediction.document_id) <= roles["outer"] and prediction.entity_id.is_unique
    metrics = strict_metrics(outer_e, prediction)
    scores.to_parquet(prediction_dir / "scores.parquet", index=False)
    prediction.to_parquet(prediction_dir / "predictions.parquet", index=False)
    upstream = torch.load(root / "Models/Comparators/biomedical_transformer.pt", map_location="cpu", weights_only=False)
    frozen_parameters = sum(v.numel() for k, v in upstream["state_dict"].items() if k.startswith("bert."))
    trainable_parameters = sum(p.numel() for p in model.parameters())
    record = {"run": RUN, "model": MODEL, "fold": fold, "seed": seed, "device": str(device),
        "selected_epoch": best["epoch"], "threshold": best["threshold"], "metrics": metrics,
        "backbone": upstream["backbone"], "frozen_encoder_parameters": frozen_parameters,
        "trainable_parameters": trainable_parameters, "total_inference_parameters": frozen_parameters + trainable_parameters,
        "training_seconds_including_selection": training_seconds, "head_inference_seconds_including_decode": inference_seconds,
        "timing_scope": "Cached-encoder span head only; excludes frozen backbone encoding",
        "head_checkpoint_bytes": (model_dir / "model.pt").stat().st_size,
        "protocol_sha256": _hash(_report(root) / "protocol.json")}
    artifacts = [p for directory in (model_dir, prediction_dir, report_dir)
                 for p in directory.iterdir() if p.is_file() and p.name != "progress.jsonl"]
    record["artifacts_sha256"] = {str(p.relative_to(root)): _hash(p) for p in sorted(artifacts)}
    _json_new(report_dir / "complete.json", record)
    del model, hidden
    return record


def aggregate(root: Path) -> dict[str, pd.DataFrame]:
    """Merge only 15 verified completed units; reuse ACT/MLP predictions read-only."""
    root = _root(root)
    _verify_protocol(root)
    complete = [_verified_completion(root, fold, seed) for fold in range(5) for seed in SEEDS]
    assert all(complete), "All five outer folds and three seeds must finish before aggregation"
    corpus = _corpus(str(root))
    documents, _ = _roles(root, 0)
    sentences = corpus.sentences.loc[corpus.sentences.document_id.isin(corpus.validation_documents)]
    entities = corpus.entities.loc[corpus.entities.document_id.isin(corpus.validation_documents)]
    predictions = {MODEL: {}}
    for seed in SEEDS:
        parts = [pd.read_parquet(_unit_paths(root, fold, seed)[1] / "predictions.parquet") for fold in range(5)]
        predictions[MODEL][seed] = pd.concat(parts, ignore_index=True)
        assert predictions[MODEL][seed].entity_id.is_unique
    historical = root / "Predictions/Refinement" / OLD_RUN / "Validation_CrossFit"
    for name in ("Full ACT-NER", "Parameter-matched span MLP"):
        predictions[name] = {seed: pd.read_parquet(historical / name.lower().replace(" ", "_") / f"seed_{seed}.parquet")
                             for seed in SEEDS}
    rows = [{"Model": name, "Seed": seed, "Split": "Validation conditional cross-fit",
             **strict_metrics(entities, prediction)}
            for name, repeats in predictions.items() for seed, prediction in repeats.items()]
    metrics = pd.DataFrame(rows)
    assert metrics.N.nunique() == 1
    comparisons = [("Full ACT-NER", MODEL), ("Parameter-matched span MLP", MODEL)]
    statistics = paired_evidence(predictions, sentences, entities, documents, comparisons, replicates=5000)
    quality = _quality_intervals(predictions, sentences, entities, documents, metrics)
    units = pd.DataFrame([{k: v for k, v in row.items() if k not in ("metrics", "artifacts_sha256")}
                          | row["metrics"] for row in complete])
    classwise = metrics.melt(id_vars=["Model", "Seed", "Split", "N"],
        value_vars=[f"{label}_f1" for label in ENTITY_TYPES], var_name="Type", value_name="F1")
    result = {"metrics": metrics, "summary": quality, "classwise": classwise,
              "statistics": statistics, "units": units}
    destination = _report(root) / "aggregate"
    if destination.exists():
        for name, frame in result.items():
            pd.testing.assert_frame_equal(pd.read_csv(destination / f"{name}.csv"), frame,
                                          check_dtype=False, rtol=1e-10, atol=1e-12)
    else:
        destination.mkdir()
        for name, frame in result.items():
            frame.to_csv(destination / f"{name}.csv", index=False)
        prediction_destination = root / "Predictions/Extended" / RUN / "biaffine/Validation_CrossFit"
        prediction_destination.mkdir(parents=True, exist_ok=False)
        for seed, prediction in predictions[MODEL].items():
            prediction.to_parquet(prediction_destination / f"seed_{seed}.parquet", index=False)
        _json_new(destination / "provenance.json", {"protocol_sha256": _hash(_report(root) / "protocol.json"),
            "unit_complete_sha256": {str(_unit_paths(root, f, s)[2] / "complete.json"):
                _hash(_unit_paths(root, f, s)[2] / "complete.json") for f in range(5) for s in SEEDS},
            "historical_predictions_sha256": {str(p): _hash(p) for name in ("full_act-ner", "parameter-matched_span_mlp")
                for p in [historical / name / f"seed_{s}.parquet" for s in SEEDS]}})
    return result


def _quality_intervals(predictions, sentences, entities, documents, metrics):
    """Cluster bootstrap for mean seed micro-F1; seeds are not independent units."""
    groups = documents.dependency_group.nunique()
    rng = np.random.default_rng(9017)
    weights = rng.multinomial(groups, np.full(groups, 1 / groups), size=5000)
    rows = []
    for name, repeats in predictions.items():
        counts = np.stack([counts_by_document(sentences, entities, repeats[s], documents) for s in SEEDS])
        bootstrap = f1_counts(np.einsum("bg,sgc->bsc", weights, counts)).mean(1)
        lo, hi = np.quantile(bootstrap, [.025, .975])
        frame = metrics.loc[metrics.Model.eq(name)]
        rows.append({"Model": name, "N documents": len(documents), "Clusters": groups,
            "N gold": int(frame.N.iloc[0]), "Seeds": len(SEEDS),
            "Typed F1 mean": float(frame.typed_f1.mean()), "Typed F1 SD": float(frame.typed_f1.std(ddof=1)),
            "Typed F1 CI low": float(lo), "Typed F1 CI high": float(hi),
            "Boundary F1 mean": float(frame.boundary_f1.mean()),
            "Boundary F1 SD": float(frame.boundary_f1.std(ddof=1)),
            "Macro-F1 mean": float(frame.macro_f1.mean()), "Macro-F1 SD": float(frame.macro_f1.std(ddof=1))})
    return pd.DataFrame(rows)


def selftest() -> pd.DataFrame:
    """Small synthetic mathematics tests; does not fit or access project data."""
    torch.manual_seed(7)
    model = BiaffineSpanNER(3, 2, 0.0, 5)
    a, b = torch.randn(4, 3), torch.randn(4, 3)
    left = F.pad(model.start(a), (0, 1), value=1)
    right = F.pad(model.end(b), (0, 1), value=1)
    manual = torch.stack([torch.stack([left[i] @ w @ right[i] for w in model.weight]) for i in range(4)])
    assert torch.allclose(model(a, b), manual, atol=1e-6)
    for n in (0, 1, 2, 7):
        x, y = enumerate_spans(n)
        assert len(x) == n * (n + 1) // 2 and (x <= y).all()
        assert len(set(zip(x, y))) == len(x)
    sentences = pd.DataFrame([{"document_id": "d", "sentence_id": "s", "text": "abc def"}])
    scores = pd.DataFrame([
        ["d", "s", 0, 3, 0, 1, "drug", .9, .9, .81, True],
        ["d", "s", 0, 7, 0, 2, "brand", .8, .8, .64, True],
        ["d", "s", 4, 7, 1, 2, "drug", .9, .9, .81, False]],
        columns=["document_id", "sentence_id", "char_start", "char_end", "token_start", "token_end",
                 "predicted_type", "span_probability", "type_probability", "entity_probability", "non_none_argmax"])
    prediction = decode(scores, sentences, .5, 17)
    assert len(prediction) == 1 and prediction.iloc[0].entity_text == "abc"
    assert np.isclose(prediction.calibrated_confidence.iloc[0], .81)
    assert decode(scores, sentences, .99, 17).empty
    loss = F.cross_entropy(model(a, b), torch.tensor([0, 1, 2, 4]))
    loss.backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    model.eval()
    hidden = torch.randn(5, 3)
    x, y = enumerate_spans(2)
    part = SentenceSpans("d", "s", 3, np.array([0, 4]), np.array([3, 7]),
                         x, y, np.array([1, 0, 4]))
    corpus = Corpus(sentences, pd.DataFrame(), hidden, [part], {"d"}, set())
    scores = score_all(model, hidden, corpus, {"d"})
    direct = model(hidden[x + 3], hidden[y + 3]).softmax(-1).detach().numpy()
    assert np.allclose(scores[[f"p_{t}" for t in ("none", *ENTITY_TYPES)]], direct, atol=1e-6)
    part.targets[:] = 0
    pd.testing.assert_frame_equal(scores, score_all(model, hidden, corpus, {"d"}))
    return pd.DataFrame({"Synthetic software test": ["Biaffine equals direct matrix products",
        "All contiguous spans exactly once", "None rejection and flat overlap decoding",
        "Threshold rejection", "Five-class differentiable loss", "Cached all-span scoring equals pairwise forward",
        "Inference independent of gold labels"], "Status": ["PASS"] * 7})
