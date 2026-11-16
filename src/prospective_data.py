"""Prospective development-only membership and fold-specific expert training.

No historical model is loaded here. No test XML, labels or predictions are read.
The existing shared processed development data are immutable inputs.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from core import (TYPED_TAGS, build_training_lexicon, gold_for_metrics,
                  load_processed, predictions_from_tag_sequences, stable_sha256)
from models import (TransformerSequenceDataset, build_transformer_model,
                    feature_sequences, train_linear_crf,
                    train_transformer_comparator, transformer_collate)
from act_ner import CandidateBundle, build_candidate_bundle
from analysis_tools import check_frozen


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_once_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == value, "Immutable protocol mismatch"
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)


@dataclass
class Development:
    sentences: pd.DataFrame
    entities: pd.DataFrame
    tokens: pd.DataFrame
    documents: pd.DataFrame

    def subset(self, document_ids) -> "Development":
        identifiers = set(document_ids)
        return Development(*(frame.loc[frame.document_id.isin(identifiers)].copy()
                             for frame in (self.sentences, self.entities,
                                           self.tokens, self.documents)))


def normalized_sentence(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def dependency_components(sentences: pd.DataFrame) -> dict[str, str]:
    """Keep documents sharing substantial exact text together, transitively."""
    parent = {value: value for value in sorted(sentences.document_id.unique())}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    owner = {}
    for row in sentences.itertuples(index=False):
        key = normalized_sentence(row.text)
        if len(key) < 40 or len(key.split()) < 6:
            continue
        if key in owner:
            left, right = find(row.document_id), find(owner[key])
            parent[max(left, right)] = min(left, right)
        else:
            owner[key] = row.document_id
    return {value: find(value) for value in parent}


def load_development(root: Path) -> Development:
    train = load_processed(root, "train")
    validation = load_processed(root, "validation")
    sentences, entities, tokens = (
        pd.concat([train[index], validation[index]], ignore_index=True)
        for index in range(3)
    )
    # Existing split identities remain on disk; these copies form development.
    for frame in (sentences, entities, tokens):
        frame["split"] = "train"
    documents = sentences[["document_id", "source"]].drop_duplicates().sort_values("document_id")
    assert len(documents) == 714 and sentences.sentence_id.is_unique
    assert entities.entity_id.is_unique
    gold = gold_for_metrics(entities)
    assert not gold.duplicated(["sentence_id", "char_start", "char_end", "entity_type"]).any()
    components = dependency_components(sentences)
    documents["dependency_group"] = documents.document_id.map(components)
    counts = gold.groupby(["document_id", "entity_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(documents.document_id, fill_value=0)
    density = gold.groupby("document_id").size().reindex(documents.document_id, fill_value=0)
    documents["stratum"] = [
        f"{source}|{int(brand > 0)}|{int(rare > 0)}|{int(n > 20)}"
        for source, brand, rare, n in zip(documents.source,
            counts.get("brand", pd.Series(0, index=counts.index)),
            counts.get("drug_n", pd.Series(0, index=counts.index)), density)
    ]
    frequency = documents.groupby("stratum").dependency_group.nunique()
    documents.loc[documents.stratum.map(frequency) < 5, "stratum"] = documents.source
    return Development(sentences, entities, tokens, documents.reset_index(drop=True))


def group_folds(documents: pd.DataFrame, folds: int, seed: int) -> pd.Series:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = pd.Series(index=documents.document_id, dtype=int)
    strata = documents.stratum.copy()
    if strata.value_counts().min() < folds:
        strata = documents.source
    for fold, (_, holdout) in enumerate(splitter.split(documents, strata, documents.dependency_group)):
        result.loc[documents.iloc[holdout].document_id] = fold
    joined = documents.assign(fold=documents.document_id.map(result))
    assert joined.groupby("dependency_group").fold.nunique().eq(1).all()
    return result.astype(int)


def split_group_holdout(documents: pd.DataFrame, fraction: float, seed: int):
    membership = group_folds(documents, max(2, round(1 / fraction)), seed)
    return set(membership.index[membership != 0]), set(membership.index[membership == 0])


def prepare_protocol(root: Path) -> tuple[dict, Development, pd.DataFrame, Path]:
    check_frozen(root, include_posthoc=True)
    protocol = json.loads((root / "src/prospective_protocol.json").read_text())
    data = load_development(root)
    path = root / "Reports/Prospective" / protocol["protocol_id"]
    path.mkdir(parents=True, exist_ok=True)
    memberships = []
    for repeat, seed in enumerate(protocol["partition_seeds"]):
        folds = group_folds(data.documents, protocol["outer_folds"], seed)
        frame = data.documents.copy()
        frame["repeat"] = repeat
        frame["outer_fold"] = frame.document_id.map(folds)
        frame["partition_seed"] = seed
        memberships.append(frame)
    membership = pd.concat(memberships, ignore_index=True)
    assert membership.groupby("repeat").document_id.nunique().eq(714).all()
    shared = root / "dataset/Splits/Prospective" / protocol["protocol_id"]
    shared.mkdir(parents=True, exist_ok=True)
    member_path = shared / "outer_documents.csv"
    if member_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(member_path), membership.reset_index(drop=True))
    else:
        membership.to_csv(member_path, index=False)
    config = AutoConfig.from_pretrained(protocol["transformer"]["backbone"], local_files_only=True)
    seal = {"protocol": protocol, "protocol_sha256": canonical_digest(protocol),
            "membership_sha256": stable_sha256(member_path),
            "pretrained_revision": getattr(config, "_commit_hash", None),
            "historical_contract_sha256": stable_sha256(root / "src/frozen_contract.json")}
    assert seal["pretrained_revision"], "Pin the pretrained model before training"
    write_once_json(path / "protocol_seal.json", seal)
    return protocol, data, membership, path


def unit_roles(data: Development, membership: pd.DataFrame, protocol: dict,
               repeat: int, outer_fold: int) -> dict[str, set[str]]:
    table = membership.loc[membership.repeat == repeat]
    outer_ids = set(table.loc[table.outer_fold == outer_fold, "document_id"])
    available = data.documents.loc[~data.documents.document_id.isin(outer_ids)]
    role_seed = protocol["partition_seeds"][repeat] + outer_fold * 100
    grow, calibration = split_group_holdout(available, protocol["calibration_fraction"], role_seed)
    grow_docs = data.documents.loc[data.documents.document_id.isin(grow)]
    head_train, selection = split_group_holdout(grow_docs, protocol["head_selection_fraction"], role_seed + 1)
    roles = {"grow": grow, "calibration": calibration, "outer": outer_ids,
             "head_train": head_train, "selection": selection}
    assert not (grow & calibration or grow & outer_ids or calibration & outer_ids)
    assert head_train | selection == grow and not head_train & selection
    return roles


def fit_experts(root: Path, data: Development, fit_ids: set[str], stop_ids: set[str],
                protocol: dict, seed: int, output: Path, callback=None) -> dict:
    """Fit a new pair, selecting checkpoints using only their internal stop set."""
    assert not fit_ids & stop_ids
    output.mkdir(parents=True, exist_ok=True)
    specification = {"fit_ids": sorted(fit_ids), "early_stop_ids": sorted(stop_ids),
                     "seed": seed, "protocol_sha256": canonical_digest(protocol),
                     "implementation": {name: stable_sha256(root / "src" / name)
                         for name in ("models.py", "prospective_data.py")}}
    write_once_json(output / "membership.json", specification)
    complete = output / "complete.json"
    if complete.exists():
        record = json.loads(complete.read_text())
        assert all(stable_sha256(output / name) == digest for name, digest in record["files"].items())
        return record
    training, stopping = data.subset(fit_ids), data.subset(stop_ids)
    tokenizer = AutoTokenizer.from_pretrained(protocol["transformer"]["backbone"], local_files_only=True)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        crf, _, history = train_linear_crf(training.tokens, stopping.tokens,
            stopping.sentences, stopping.entities, candidates=protocol["crf"]["candidates"])
    joblib.dump(crf, output / "crf.joblib")
    history.to_csv(output / "crf_selection.csv", index=False)
    crf_seconds = time.perf_counter() - started
    if callback:
        callback({"phase": "Transformer training", "epoch": 0, "crf_seconds": crf_seconds})
    settings = protocol["transformer"]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model, history, _, elapsed, _ = train_transformer_comparator(
            settings["backbone"], tokenizer, training.tokens, stopping.tokens,
            stopping.sentences, stopping.entities, seed,
            max_epochs=settings["max_epochs"], patience=settings["patience"],
            learning_rate=settings["learning_rate"], batch_size=settings["batch_size"],
            epoch_callback=callback)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"state_dict": state, "membership": specification}, output / "transformer.pt")
    history.to_csv(output / "transformer_history.csv", index=False)
    record = {"crf_seconds": crf_seconds, "transformer_seconds": elapsed,
              "train_documents": len(fit_ids), "stop_documents": len(stop_ids),
              "files": {name: stable_sha256(output / name) for name in
                        ("crf.joblib", "transformer.pt", "crf_selection.csv", "transformer_history.csv")}}
    write_once_json(complete, record)
    del model, state, crf
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return record


def expert_evidence(data: Development, checkpoint: Path, protocol: dict):
    tokenizer = AutoTokenizer.from_pretrained(protocol["transformer"]["backbone"], local_files_only=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = build_transformer_model(protocol["transformer"]["backbone"])
    record = torch.load(checkpoint / "transformer.pt", weights_only=False, map_location="cpu")
    consumed = set(record["membership"]["fit_ids"]) | set(record["membership"]["early_stop_ids"])
    assert not consumed & set(data.documents.document_id), "Expert evidence must be held out"
    model.load_state_dict(record["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    dataset = TransformerSequenceDataset(data.tokens, tokenizer.cls_token_id, tokenizer.sep_token_id)
    loader = DataLoader(dataset, batch_size=24, shuffle=False,
                        collate_fn=lambda rows: transformer_collate(rows, tokenizer.pad_token_id))
    transformer_rows, crf_rows = {}, {}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            result = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device), output_hidden_states=True)
            for index, sentence in enumerate(batch["sentence_ids"]):
                length = int(batch["attention_mask"][index].sum()) - 2
                probabilities = result.logits[index, 1:length + 1].softmax(-1).cpu()
                transformer_rows[sentence] = {"probabilities": probabilities,
                    "hidden": result.hidden_states[-1][index, 1:length + 1].cpu(),
                    "tags": [TYPED_TAGS[i] for i in probabilities.argmax(-1).tolist()]}
    crf = joblib.load(checkpoint / "crf.joblib")
    sequences, _, identifiers = feature_sequences(data.tokens)
    for sentence, features in zip(identifiers, sequences):
        tags = crf.predict_single(features)
        marginals = crf.predict_marginals_single(features)
        probabilities = torch.tensor([[row.get(tag, 0.0) for tag in TYPED_TAGS] for row in marginals])
        crf.tagger_.set(features)
        crf_rows[sentence] = {"tags": list(tags), "probabilities": probabilities,
            "sequence_probability": float(crf.tagger_.probability(tags)),
            "transition_confidence": float(np.mean([row.get(tag, 0.0) for row, tag in zip(marginals, tags)]))}
    del model, record, crf
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return transformer_rows, crf_rows, time.perf_counter() - started


def candidate_evidence(root: Path, observed: Development, training: Development,
                       checkpoint: Path, protocol: dict, include_gold_support: bool):
    evidence = expert_evidence(observed, checkpoint, protocol)
    lexicon = build_training_lexicon(gold_for_metrics(training.entities).assign(split="train"))
    # In evaluation, gold labels cannot create candidates, even incidentally.
    labels = observed.entities if include_gold_support else observed.entities.iloc[:0]
    bundle, _, elapsed = build_candidate_bundle(root, observed.sentences, labels,
        observed.tokens, protocol["candidate_marginal_threshold"],
        expert_evidence=evidence, training_lexicon=lexicon)
    if not include_gold_support:
        assert bundle.frame.generated_candidate.all()
    expert_predictions = {}
    for name, rows in zip(("Biomedical Transformer", "Linear-chain CRF"), evidence[:2]):
        tags = {key: value["tags"] for key, value in rows.items()}
        probabilities = {key: value["probabilities"].max(-1).values.tolist() for key, value in rows.items()}
        expert_predictions[name] = predictions_from_tag_sequences(observed.tokens,
            observed.sentences, tags, probabilities, name, "prospective")
    return bundle, expert_predictions, elapsed


def concatenate_bundles(bundles: list[CandidateBundle]) -> CandidateBundle:
    frame = pd.concat([bundle.frame for bundle in bundles], ignore_index=True)
    assert not frame.duplicated(["sentence_id", "token_start", "token_end"]).any()
    keys = ("context", "transformer", "crf", "orthographic", "lexical", "transformer_type", "crf_type")
    return CandidateBundle(frame, *(torch.cat([getattr(bundle, key) for bundle in bundles]) for key in keys))
