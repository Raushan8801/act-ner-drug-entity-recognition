from __future__ import annotations

import copy
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn_crfsuite import CRF as SklearnCRF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForTokenClassification
from transformers.utils import logging as transformers_logging
from scipy.optimize import minimize_scalar

from core import (
    BOUNDARY_TAGS,
    ENTITY_TYPES,
    TYPED_TAGS,
    EfficiencyRecord,
    evaluate_predictions,
    finalize_prediction_frame,
    load_settings,
    normalize_surface,
    predictions_from_tag_sequences,
)

transformers_logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def compute_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _shape(value: str) -> str:
    output = []
    for character in value:
        if character.isupper():
            code = "X"
        elif character.islower():
            code = "x"
        elif character.isdigit():
            code = "d"
        else:
            code = character
        if not output or output[-1] != code:
            output.append(code)
    return "".join(output)[:16]


def _token_features(rows: list[dict], index: int) -> dict:
    row = rows[index]
    surface = row["surface_text"] or row["token_text"]
    lower = surface.lower()
    features = {
        "bias": 1.0,
        "token.lower": lower,
        "token.shape": _shape(surface),
        "token.isupper": surface.isupper(),
        "token.istitle": surface.istitle(),
        "token.isdigit": surface.isdigit(),
        "token.hasdigit": any(character.isdigit() for character in surface),
        "token.hashyphen": "-" in surface,
        "token.length": min(len(surface), 30),
        "prefix1": lower[:1],
        "prefix2": lower[:2],
        "prefix3": lower[:3],
        "suffix1": lower[-1:],
        "suffix2": lower[-2:],
        "suffix3": lower[-3:],
        "relative_position": index / max(len(rows) - 1, 1),
    }
    for offset, label in ((-1, "prev"), (1, "next")):
        neighbor = index + offset
        if 0 <= neighbor < len(rows):
            neighbor_surface = rows[neighbor]["surface_text"] or rows[neighbor]["token_text"]
            features[f"{label}.lower"] = neighbor_surface.lower()
            features[f"{label}.shape"] = _shape(neighbor_surface)
        else:
            features[f"{label}.boundary"] = True
    return features


def feature_sequences(token_frame: pd.DataFrame) -> tuple[list[list[dict]], list[list[str]], list[str]]:
    x_sequences: list[list[dict]] = []
    y_sequences: list[list[str]] = []
    sentence_ids: list[str] = []
    for sentence_id, group in token_frame.sort_values(["sentence_id", "token_index"]).groupby(
        "sentence_id", sort=False
    ):
        rows = group.to_dict("records")
        x_sequences.append([_token_features(rows, index) for index in range(len(rows))])
        y_sequences.append(group["typed_tag"].tolist())
        sentence_ids.append(sentence_id)
    return x_sequences, y_sequences, sentence_ids


class LongestMatchLexicon:
    def __init__(self, lexicon: pd.DataFrame):
        records = []
        for row in lexicon.itertuples(index=False):
            surface = str(row.surface_form)
            records.append(
                {
                    "surface": surface,
                    "lower": surface.casefold(),
                    "normalized": row.normalized_surface,
                    "entity_type": row.majority_type,
                    "frequency": int(row.frequency),
                    "type_probability": float(getattr(row, f"p_{row.majority_type}")),
                    "ambiguity": float(row.ambiguity_score),
                }
            )
        by_initial: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            if record["lower"]:
                by_initial[record["lower"][0]].append(record)
        for initial in by_initial:
            by_initial[initial].sort(key=lambda item: (-len(item["lower"]), item["lower"]))
        self.by_initial = dict(by_initial)
        self.max_frequency = max((record["frequency"] for record in records), default=1)

    def matches(self, text: str) -> list[dict]:
        lower_text = text.casefold()
        candidates = []
        for start, character in enumerate(lower_text):
            if start > 0 and lower_text[start - 1].isalnum():
                continue
            for record in self.by_initial.get(character, []):
                end = start + len(record["lower"])
                if not lower_text.startswith(record["lower"], start):
                    continue
                if end < len(lower_text) and lower_text[end].isalnum():
                    continue
                candidates.append((start, end, record))
        candidates.sort(key=lambda value: (value[0], -(value[1] - value[0]), value[1]))
        selected = []
        occupied_until = -1
        for start, end, record in candidates:
            if start < occupied_until:
                continue
            selected.append(
                {
                    "char_start": start,
                    "char_end": end,
                    "entity_type": record["entity_type"],
                    "frequency": record["frequency"],
                    "type_probability": record["type_probability"],
                    "ambiguity": record["ambiguity"],
                    "exact_case": text[start:end] == record["surface"],
                }
            )
            occupied_until = end
        return selected


def lexicon_predictions(
    matcher: LongestMatchLexicon,
    sentences: pd.DataFrame,
    tokens: pd.DataFrame,
    split: str,
    model: str = "Longest-match lexicon",
) -> pd.DataFrame:
    spans = []
    token_groups = {key: group.sort_values("token_index") for key, group in tokens.groupby("sentence_id")}
    for sentence in sentences.itertuples(index=False):
        sentence_tokens = token_groups.get(sentence.sentence_id)
        if sentence_tokens is None or sentence_tokens.empty:
            continue
        for match in matcher.matches(sentence.text):
            covered = sentence_tokens[
                (sentence_tokens.char_end > match["char_start"])
                & (sentence_tokens.char_start < match["char_end"])
            ]
            if covered.empty:
                continue
            frequency_confidence = math.log1p(match["frequency"]) / math.log1p(matcher.max_frequency)
            spans.append(
                {
                    "document_id": sentence.document_id,
                    "sentence_id": sentence.sentence_id,
                    "char_start": match["char_start"],
                    "char_end": match["char_end"],
                    "token_start": int(covered.token_index.min()),
                    "token_end": int(covered.token_index.max()) + 1,
                    "predicted_type": match["entity_type"],
                    "boundary_probability": float(0.50 + 0.49 * frequency_confidence),
                    "type_probability": match["type_probability"],
                }
            )
    return finalize_prediction_frame(spans, sentences, model, split)


def o_only_predictions(sentences: pd.DataFrame, split: str) -> pd.DataFrame:
    return finalize_prediction_frame([], sentences, "O-only", split)


def train_logistic_regression(
    train_tokens: pd.DataFrame,
    validation_tokens: pd.DataFrame,
    validation_sentences: pd.DataFrame,
    validation_entities: pd.DataFrame,
    c_values: Sequence[float] = (0.25, 1.0, 4.0),
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    train_x, train_y, _ = feature_sequences(train_tokens)
    validation_x, _, validation_ids = feature_sequences(validation_tokens)
    vectorizer = DictVectorizer(sparse=True)
    train_flat = [feature for sequence in train_x for feature in sequence]
    validation_flat = [feature for sequence in validation_x for feature in sequence]
    x_train = vectorizer.fit_transform(train_flat)
    x_validation = vectorizer.transform(validation_flat)
    y_train = [label for sequence in train_y for label in sequence]
    tuning_rows = []
    fitted = []
    for c_value in c_values:
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=300,
            solver="lbfgs",
            random_state=17,
        )
        model.fit(x_train, y_train)
        predicted_flat = model.predict(x_validation)
        probability_flat = model.predict_proba(x_validation).max(axis=1)
        tag_sequences = {}
        probability_sequences = {}
        cursor = 0
        for sentence_id, sequence in zip(validation_ids, validation_x):
            length = len(sequence)
            tag_sequences[sentence_id] = predicted_flat[cursor : cursor + length].tolist()
            probability_sequences[sentence_id] = probability_flat[cursor : cursor + length].tolist()
            cursor += length
        predictions = predictions_from_tag_sequences(
            validation_tokens,
            validation_sentences,
            tag_sequences,
            probability_sequences,
            "Logistic Regression",
            "validation",
        )
        metrics, _ = evaluate_predictions(
            validation_entities, predictions, "Logistic Regression", "validation"
        )
        tuning_rows.append(
            {
                "Model": "Logistic Regression",
                "Split": "Validation",
                "N": len(validation_entities),
                "C": c_value,
                "Strict Boundary F1": metrics.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metrics.iloc[0]["Strict Typed Entity F1"],
            }
        )
        fitted.append((c_value, model, predictions))
    tuning = pd.DataFrame(tuning_rows)
    selected_c = float(
        tuning.sort_values(
            ["Strict Typed Entity F1", "Strict Boundary F1", "C"], ascending=[False, False, True]
        ).iloc[0]["C"]
    )
    selected_model, selected_predictions = next(
        (model, predictions) for c_value, model, predictions in fitted if c_value == selected_c
    )
    artifact = {"vectorizer": vectorizer, "model": selected_model, "selected_c": selected_c}
    return artifact, selected_predictions, tuning


def predict_logistic_regression(
    artifact: dict,
    tokens: pd.DataFrame,
    sentences: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    x_sequences, _, sentence_ids = feature_sequences(tokens)
    x_flat = artifact["vectorizer"].transform([feature for sequence in x_sequences for feature in sequence])
    predicted_flat = artifact["model"].predict(x_flat)
    probability_flat = artifact["model"].predict_proba(x_flat).max(axis=1)
    tags, probabilities, cursor = {}, {}, 0
    for sentence_id, sequence in zip(sentence_ids, x_sequences):
        length = len(sequence)
        tags[sentence_id] = predicted_flat[cursor : cursor + length].tolist()
        probabilities[sentence_id] = probability_flat[cursor : cursor + length].tolist()
        cursor += length
    return predictions_from_tag_sequences(
        tokens, sentences, tags, probabilities, "Logistic Regression", split
    )


def train_linear_crf(
    train_tokens: pd.DataFrame,
    validation_tokens: pd.DataFrame,
    validation_sentences: pd.DataFrame,
    validation_entities: pd.DataFrame,
    candidates: Sequence[tuple[float, float]] = ((0.05, 0.05), (0.10, 0.10), (0.10, 0.01)),
) -> tuple[SklearnCRF, pd.DataFrame, pd.DataFrame]:
    train_x, train_y, _ = feature_sequences(train_tokens)
    validation_x, _, validation_ids = feature_sequences(validation_tokens)
    tuning_rows = []
    fitted = []
    for c1, c2 in candidates:
        model = SklearnCRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=120,
            all_possible_transitions=True,
        )
        model.fit(train_x, train_y)
        predicted = model.predict(validation_x)
        marginals = model.predict_marginals(validation_x)
        probabilities = [
            [float(token_distribution.get(tag, 0.0)) for token_distribution, tag in zip(sequence, tags)]
            for sequence, tags in zip(marginals, predicted)
        ]
        tag_map = dict(zip(validation_ids, predicted))
        probability_map = dict(zip(validation_ids, probabilities))
        predictions = predictions_from_tag_sequences(
            validation_tokens,
            validation_sentences,
            tag_map,
            probability_map,
            "Linear-chain CRF",
            "validation",
        )
        metrics, _ = evaluate_predictions(validation_entities, predictions, "Linear-chain CRF", "validation")
        tuning_rows.append(
            {
                "Model": "Linear-chain CRF",
                "Split": "Validation",
                "N": len(validation_entities),
                "c1": c1,
                "c2": c2,
                "Strict Boundary F1": metrics.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metrics.iloc[0]["Strict Typed Entity F1"],
            }
        )
        fitted.append((c1, c2, model, predictions))
    tuning = pd.DataFrame(tuning_rows)
    selected = tuning.sort_values(
        ["Strict Typed Entity F1", "Strict Boundary F1", "c1", "c2"],
        ascending=[False, False, True, True],
    ).iloc[0]
    model, predictions = next(
        (model, predictions)
        for c1, c2, model, predictions in fitted
        if c1 == selected.c1 and c2 == selected.c2
    )
    return model, predictions, tuning


def predict_linear_crf(
    model: SklearnCRF,
    tokens: pd.DataFrame,
    sentences: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    x_sequences, _, sentence_ids = feature_sequences(tokens)
    predicted = model.predict(x_sequences)
    marginals = model.predict_marginals(x_sequences)
    probabilities = [
        [float(token_distribution.get(tag, 0.0)) for token_distribution, tag in zip(sequence, tags)]
        for sequence, tags in zip(marginals, predicted)
    ]
    return predictions_from_tag_sequences(
        tokens,
        sentences,
        dict(zip(sentence_ids, predicted)),
        dict(zip(sentence_ids, probabilities)),
        "Linear-chain CRF",
        split,
    )


def baseline_oof_predictions(
    train_tokens: pd.DataFrame,
    train_sentences: pd.DataFrame,
    lexicon_builder,
    logistic_c: float,
    crf_c1: float,
    crf_c2: float,
    n_folds: int = 3,
) -> dict[str, pd.DataFrame]:
    documents = train_sentences[["document_id"]].drop_duplicates().sort_values("document_id")
    splitter = GroupKFold(n_splits=n_folds)
    outputs = defaultdict(list)
    document_array = documents.document_id.to_numpy()
    for fold, (fit_indices, hold_indices) in enumerate(
        splitter.split(document_array, groups=document_array), start=1
    ):
        fit_documents = set(document_array[fit_indices])
        hold_documents = set(document_array[hold_indices])
        fit_tokens = train_tokens[train_tokens.document_id.isin(fit_documents)]
        hold_tokens = train_tokens[train_tokens.document_id.isin(hold_documents)]
        fit_sentences = train_sentences[train_sentences.document_id.isin(fit_documents)]
        hold_sentences = train_sentences[train_sentences.document_id.isin(hold_documents)]
        outputs["O-only"].append(o_only_predictions(hold_sentences, "oof").assign(fold=fold))

        matcher = lexicon_builder(fit_documents)
        outputs["Longest-match lexicon"].append(
            lexicon_predictions(matcher, hold_sentences, hold_tokens, "oof").assign(fold=fold)
        )

        fit_x, fit_y, _ = feature_sequences(fit_tokens)
        hold_x, _, hold_ids = feature_sequences(hold_tokens)
        vectorizer = DictVectorizer(sparse=True)
        x_fit = vectorizer.fit_transform([feature for sequence in fit_x for feature in sequence])
        x_hold = vectorizer.transform([feature for sequence in hold_x for feature in sequence])
        y_fit = [label for sequence in fit_y for label in sequence]
        logistic = LogisticRegression(
            C=logistic_c,
            class_weight="balanced",
            max_iter=300,
            solver="lbfgs",
            random_state=17 + fold,
        ).fit(x_fit, y_fit)
        flat_tags = logistic.predict(x_hold)
        flat_probabilities = logistic.predict_proba(x_hold).max(axis=1)
        tag_map, probability_map, cursor = {}, {}, 0
        for sentence_id, sequence in zip(hold_ids, hold_x):
            length = len(sequence)
            tag_map[sentence_id] = flat_tags[cursor : cursor + length].tolist()
            probability_map[sentence_id] = flat_probabilities[cursor : cursor + length].tolist()
            cursor += length
        outputs["Logistic Regression"].append(
            predictions_from_tag_sequences(
                hold_tokens,
                hold_sentences,
                tag_map,
                probability_map,
                "Logistic Regression",
                "oof",
            ).assign(fold=fold)
        )

        crf = SklearnCRF(
            algorithm="lbfgs",
            c1=crf_c1,
            c2=crf_c2,
            max_iterations=120,
            all_possible_transitions=True,
        ).fit(fit_x, fit_y)
        crf_tags = crf.predict(hold_x)
        crf_marginals = crf.predict_marginals(hold_x)
        crf_probabilities = [
            [float(distribution.get(tag, 0.0)) for distribution, tag in zip(sequence, tags)]
            for sequence, tags in zip(crf_marginals, crf_tags)
        ]
        outputs["Linear-chain CRF"].append(
            predictions_from_tag_sequences(
                hold_tokens,
                hold_sentences,
                dict(zip(hold_ids, crf_tags)),
                dict(zip(hold_ids, crf_probabilities)),
                "Linear-chain CRF",
                "oof",
            ).assign(fold=fold)
        )
    return {
        model: pd.concat(parts, ignore_index=True).sort_values(["document_id", "sentence_id", "char_start"])
        for model, parts in outputs.items()
    }


class LinearChainCRF(nn.Module):
    def __init__(self, n_tags: int, allowed_transitions: torch.Tensor | None = None):
        super().__init__()
        self.n_tags = n_tags
        self.transitions = nn.Parameter(torch.empty(n_tags, n_tags))
        self.start_transitions = nn.Parameter(torch.empty(n_tags))
        self.end_transitions = nn.Parameter(torch.empty(n_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        if allowed_transitions is None:
            allowed_transitions = torch.ones(n_tags, n_tags, dtype=torch.bool)
        self.register_buffer("allowed_transitions", allowed_transitions.bool())

    def constrained_transitions(self) -> torch.Tensor:
        return self.transitions.masked_fill(~self.allowed_transitions, -1e4)

    def log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transitions = self.constrained_transitions()
        numerator = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for time_index in range(1, emissions.size(1)):
            active = mask[:, time_index]
            transition_score = transitions[tags[:, time_index - 1], tags[:, time_index]]
            emission_score = emissions[:, time_index].gather(1, tags[:, time_index : time_index + 1]).squeeze(1)
            numerator = numerator + (transition_score + emission_score) * active
        lengths = mask.long().sum(1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        numerator = numerator + self.end_transitions[last_tags]

        score = self.start_transitions + emissions[:, 0]
        for time_index in range(1, emissions.size(1)):
            next_score = torch.logsumexp(score.unsqueeze(2) + transitions.unsqueeze(0), dim=1) + emissions[:, time_index]
            score = torch.where(mask[:, time_index].unsqueeze(1), next_score, score)
        denominator = torch.logsumexp(score + self.end_transitions, dim=1)
        return numerator - denominator

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        transitions = self.constrained_transitions()
        score = self.start_transitions + emissions[:, 0]
        history = []
        for time_index in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + transitions.unsqueeze(0)
            best_score, best_path = next_score.max(dim=1)
            best_score = best_score + emissions[:, time_index]
            score = torch.where(mask[:, time_index].unsqueeze(1), best_score, score)
            history.append(best_path)
        score = score + self.end_transitions
        best_last = score.argmax(dim=1)
        paths = []
        lengths = mask.long().sum(1).tolist()
        for batch_index, length in enumerate(lengths):
            tag = int(best_last[batch_index])
            path = [tag]
            for backpointer in reversed(history[: max(length - 1, 0)]):
                tag = int(backpointer[batch_index, tag])
                path.append(tag)
            paths.append(list(reversed(path)))
        return paths


def typed_transition_mask() -> torch.Tensor:
    mask = torch.zeros(len(TYPED_TAGS), len(TYPED_TAGS), dtype=torch.bool)
    for previous_index, previous in enumerate(TYPED_TAGS):
        for next_index, next_tag in enumerate(TYPED_TAGS):
            if previous == "O" or previous.startswith(("E-", "S-")):
                allowed = next_tag == "O" or next_tag.startswith(("B-", "S-"))
            elif previous.startswith(("B-", "I-")):
                previous_type = previous.split("-", 1)[1]
                allowed = next_tag in (f"I-{previous_type}", f"E-{previous_type}")
            else:
                allowed = False
            mask[previous_index, next_index] = allowed
    return mask


def boundary_transition_mask() -> torch.Tensor:
    mask = torch.zeros(len(BOUNDARY_TAGS), len(BOUNDARY_TAGS), dtype=torch.bool)
    for previous_index, previous in enumerate(BOUNDARY_TAGS):
        for next_index, next_tag in enumerate(BOUNDARY_TAGS):
            if previous in ("O", "E", "S"):
                allowed = next_tag in ("O", "B", "S")
            else:
                allowed = next_tag in ("I", "E")
            mask[previous_index, next_index] = allowed
    return mask


class TokenSequenceDataset(Dataset):
    def __init__(self, token_frame: pd.DataFrame, tag_space: Sequence[str] = TYPED_TAGS):
        self.tag_to_id = {tag: index for index, tag in enumerate(tag_space)}
        self.records = []
        for sentence_id, group in token_frame.sort_values(["sentence_id", "token_index"]).groupby(
            "sentence_id", sort=False
        ):
            self.records.append(
                {
                    "sentence_id": sentence_id,
                    "input_ids": torch.tensor(group.input_id.to_numpy(), dtype=torch.long),
                    "labels": torch.tensor(
                        [self.tag_to_id[tag] for tag in group.typed_tag], dtype=torch.long
                    ),
                    "length": len(group),
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_sequences(batch: list[dict]) -> dict:
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=0)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    mask = torch.arange(input_ids.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "sentence_ids": [item["sentence_id"] for item in batch],
        "input_ids": input_ids,
        "labels": labels,
        "mask": mask,
    }


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_tags: int,
        embedding_dim: int = 96,
        hidden_size: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.emission = nn.Linear(hidden_size * 2, n_tags)
        self.crf = LinearChainCRF(n_tags, typed_transition_mask())

    def emissions(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        encoded, _ = self.encoder(embedded)
        return self.emission(self.layer_norm(self.dropout(encoded)))

    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return -self.crf.log_likelihood(self.emissions(input_ids), labels, mask).mean()

    def decode(self, input_ids: torch.Tensor, mask: torch.Tensor):
        emissions = self.emissions(input_ids)
        paths = self.crf.decode(emissions, mask)
        probabilities = emissions.softmax(-1)
        path_probabilities = []
        for batch_index, path in enumerate(paths):
            path_probabilities.append(
                [float(probabilities[batch_index, index, tag].detach().cpu()) for index, tag in enumerate(path)]
            )
        return paths, path_probabilities


def predict_bilstm(
    model: BiLSTMCRF,
    token_frame: pd.DataFrame,
    sentence_frame: pd.DataFrame,
    split: str,
    batch_size: int = 48,
) -> tuple[pd.DataFrame, float]:
    device = next(model.parameters()).device
    dataset = TokenSequenceDataset(token_frame)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    tag_sequences: dict[str, list[str]] = {}
    probability_sequences: dict[str, list[float]] = {}
    id_to_tag = dict(enumerate(TYPED_TAGS))
    model.eval()
    start = perf_counter()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            paths, probabilities = model.decode(input_ids, mask)
            for sentence_id, path, probability in zip(batch["sentence_ids"], paths, probabilities):
                tag_sequences[sentence_id] = [id_to_tag[tag] for tag in path]
                probability_sequences[sentence_id] = probability
    elapsed = perf_counter() - start
    predictions = predictions_from_tag_sequences(
        token_frame,
        sentence_frame,
        tag_sequences,
        probability_sequences,
        "BiLSTM-CRF",
        split,
    )
    return predictions, elapsed


def train_bilstm_comparator(
    train_tokens: pd.DataFrame,
    validation_tokens: pd.DataFrame,
    validation_sentences: pd.DataFrame,
    validation_entities: pd.DataFrame,
    vocab_size: int,
    seed: int,
    max_epochs: int = 8,
    patience: int = 2,
    learning_rate: float = 2e-3,
    batch_size: int = 48,
) -> tuple[BiLSTMCRF, pd.DataFrame, pd.DataFrame, float]:
    set_seed(seed)
    device = compute_device()
    model = BiLSTMCRF(vocab_size=vocab_size, n_tags=len(TYPED_TAGS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_dataset = TokenSequenceDataset(train_tokens)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=collate_sequences,
    )
    history = []
    selected_state = copy.deepcopy(model.state_dict())
    selected_score = -1.0
    epochs_without_improvement = 0
    training_start = perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(
                batch["input_ids"].to(device),
                batch["labels"].to(device),
                batch["mask"].to(device),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        predictions, _ = predict_bilstm(
            model, validation_tokens, validation_sentences, "validation", batch_size=batch_size
        )
        metrics, _ = evaluate_predictions(validation_entities, predictions, "BiLSTM-CRF", "validation")
        score = float(metrics.iloc[0]["Strict Typed Entity F1"])
        history.append(
            {
                "Model": "BiLSTM-CRF",
                "Epoch": epoch,
                "Train Loss": float(np.mean(epoch_losses)),
                "Validation Strict Boundary F1": metrics.iloc[0]["Strict Boundary F1"],
                "Validation Strict Typed Entity F1": score,
            }
        )
        if score > selected_score + 1e-6:
            selected_score = score
            selected_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    training_elapsed = perf_counter() - training_start
    model.load_state_dict(selected_state)
    predictions, inference_elapsed = predict_bilstm(
        model, validation_tokens, validation_sentences, "validation", batch_size=batch_size
    )
    return model, predictions, pd.DataFrame(history), training_elapsed + inference_elapsed


class TransformerSequenceDataset(Dataset):
    def __init__(self, token_frame: pd.DataFrame, cls_id: int, sep_id: int):
        tag_to_id = {tag: index for index, tag in enumerate(TYPED_TAGS)}
        self.records = []
        for sentence_id, group in token_frame.sort_values(["sentence_id", "token_index"]).groupby(
            "sentence_id", sort=False
        ):
            input_ids = [cls_id] + group.input_id.astype(int).tolist() + [sep_id]
            labels = [-100] + [tag_to_id[tag] for tag in group.typed_tag] + [-100]
            self.records.append(
                {
                    "sentence_id": sentence_id,
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def transformer_collate(batch: list[dict], pad_token_id: int = 0) -> dict:
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id
    )
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(pad_token_id)
    return {
        "sentence_ids": [item["sentence_id"] for item in batch],
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_transformer_model(backbone: str) -> AutoModelForTokenClassification:
    return AutoModelForTokenClassification.from_pretrained(
        backbone,
        num_labels=len(TYPED_TAGS),
        id2label={index: tag for index, tag in enumerate(TYPED_TAGS)},
        label2id={tag: index for index, tag in enumerate(TYPED_TAGS)},
        ignore_mismatched_sizes=True,
    )


def predict_transformer(
    model,
    token_frame: pd.DataFrame,
    sentence_frame: pd.DataFrame,
    tokenizer,
    split: str,
    batch_size: int = 24,
) -> tuple[pd.DataFrame, float, dict[str, list[float]]]:
    device = next(model.parameters()).device
    dataset = TransformerSequenceDataset(token_frame, tokenizer.cls_token_id, tokenizer.sep_token_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: transformer_collate(batch, tokenizer.pad_token_id),
    )
    tag_sequences: dict[str, list[str]] = {}
    probability_sequences: dict[str, list[float]] = {}
    context_confidence: dict[str, list[float]] = {}
    model.eval()
    start = perf_counter()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probabilities = outputs.logits.softmax(-1)
            tags = probabilities.argmax(-1)
            for row_index, sentence_id in enumerate(batch["sentence_ids"]):
                length = int(batch["attention_mask"][row_index].sum()) - 2
                row_probabilities = probabilities[row_index, 1 : length + 1]
                row_tags = tags[row_index, 1 : length + 1]
                tag_sequences[sentence_id] = [TYPED_TAGS[int(value)] for value in row_tags.detach().cpu()]
                maximum = row_probabilities.max(-1).values.detach().cpu().tolist()
                probability_sequences[sentence_id] = maximum
                context_confidence[sentence_id] = maximum
    elapsed = perf_counter() - start
    predictions = predictions_from_tag_sequences(
        token_frame,
        sentence_frame,
        tag_sequences,
        probability_sequences,
        "Biomedical Transformer",
        split,
    )
    return predictions, elapsed, context_confidence


def train_transformer_comparator(
    backbone: str,
    tokenizer,
    train_tokens: pd.DataFrame,
    validation_tokens: pd.DataFrame,
    validation_sentences: pd.DataFrame,
    validation_entities: pd.DataFrame,
    seed: int,
    max_epochs: int = 5,
    patience: int = 2,
    learning_rate: float = 5e-5,
    batch_size: int = 24,
    unfreeze_depth: int | None = None,
    epoch_callback=None,
) -> tuple[nn.Module, pd.DataFrame, pd.DataFrame, float, dict[str, list[float]]]:
    set_seed(seed)
    device = compute_device()
    model = build_transformer_model(backbone).to(device)
    if unfreeze_depth is not None:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        if unfreeze_depth > 0:
            encoder = getattr(model, "bert", getattr(model, "base_model", None))
            layers = encoder.encoder.layer
            for layer in layers[-unfreeze_depth:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )
    train_dataset = TransformerSequenceDataset(
        train_tokens, tokenizer.cls_token_id, tokenizer.sep_token_id
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: transformer_collate(batch, tokenizer.pad_token_id),
    )
    selected_state = copy.deepcopy(model.state_dict())
    selected_score = -1.0
    epochs_without_improvement = 0
    history = []
    training_start = perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(outputs.loss.detach().cpu()))
        predictions, _, _ = predict_transformer(
            model,
            validation_tokens,
            validation_sentences,
            tokenizer,
            "validation",
            batch_size=batch_size,
        )
        metrics, _ = evaluate_predictions(
            validation_entities, predictions, "Biomedical Transformer", "validation"
        )
        score = float(metrics.iloc[0]["Strict Typed Entity F1"])
        history.append(
            {
                "Model": "Biomedical Transformer",
                "Epoch": epoch,
                "Train Loss": float(np.mean(losses)),
                "Validation Strict Boundary F1": metrics.iloc[0]["Strict Boundary F1"],
                "Validation Strict Typed Entity F1": score,
            }
        )
        if epoch_callback is not None:
            epoch_callback(history[-1])
        if score > selected_score + 1e-6:
            selected_score = score
            selected_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    training_elapsed = perf_counter() - training_start
    model.load_state_dict(selected_state)
    predictions, inference_elapsed, context_confidence = predict_transformer(
        model,
        validation_tokens,
        validation_sentences,
        tokenizer,
        "validation",
        batch_size=batch_size,
    )
    return (
        model,
        predictions,
        pd.DataFrame(history),
        training_elapsed + inference_elapsed,
        context_confidence,
    )


def extract_context_embeddings(
    transformer_model,
    token_frame: pd.DataFrame,
    tokenizer,
    batch_size: int = 24,
) -> dict[str, torch.Tensor]:
    device = next(transformer_model.parameters()).device
    dataset = TransformerSequenceDataset(token_frame, tokenizer.cls_token_id, tokenizer.sep_token_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: transformer_collate(batch, tokenizer.pad_token_id),
    )
    transformer_model.eval()
    outputs_by_sentence = {}
    with torch.inference_mode():
        for batch in loader:
            outputs = transformer_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            for row_index, sentence_id in enumerate(batch["sentence_ids"]):
                length = int(batch["attention_mask"][row_index].sum()) - 2
                outputs_by_sentence[sentence_id] = hidden[row_index, 1 : length + 1].detach().cpu()
    return outputs_by_sentence


def _stable_bucket(value: str, modulus: int = 128) -> int:
    import hashlib

    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % modulus


def _orthographic_features(surface: str, position: int, length: int) -> tuple[list[int], list[float]]:
    value = surface or "_"
    clean = value.replace("##", "")
    if clean.isupper() and any(character.isalpha() for character in clean):
        capitalization = 2
    elif clean.istitle():
        capitalization = 1
    elif clean.islower():
        capitalization = 0
    else:
        capitalization = 3
    if clean.isdigit():
        digit = 2
    elif any(character.isdigit() for character in clean):
        digit = 1
    else:
        digit = 0
    greek = int(bool(re.search(r"[α-ωΑ-Ω]", clean)))
    punctuation = int(all(not character.isalnum() for character in clean))
    morphology_suffixes = (
        "cillin",
        "mycin",
        "mab",
        "vir",
        "azole",
        "pril",
        "sartan",
        "olol",
        "caine",
        "idine",
        "amine",
        "oxide",
    )
    morphology = next(
        (index + 1 for index, suffix in enumerate(morphology_suffixes) if clean.lower().endswith(suffix)),
        0,
    )
    categorical = [
        capitalization,
        digit,
        int("-" in clean),
        greek,
        punctuation,
        morphology,
        _stable_bucket(clean.lower()[:3]),
        _stable_bucket(clean.lower()[-3:]),
    ]
    numeric = [min(len(clean), 30) / 30.0, position / max(length - 1, 1)]
    return categorical, numeric


def add_act_features(
    token_frame: pd.DataFrame,
    sentence_frame: pd.DataFrame,
    lexicon: pd.DataFrame,
) -> pd.DataFrame:
    matcher = LongestMatchLexicon(lexicon)
    sentence_text = sentence_frame.set_index("sentence_id")["text"].to_dict()
    rows = []
    for sentence_id, group in token_frame.sort_values(["sentence_id", "token_index"]).groupby(
        "sentence_id", sort=False
    ):
        text = sentence_text[sentence_id]
        matches = matcher.matches(text)
        group_records = group.to_dict("records")
        for index, token in enumerate(group_records):
            overlapping = [
                match
                for match in matches
                if match["char_end"] > token["char_start"] and match["char_start"] < token["char_end"]
            ]
            lexical = [0.0] * 12
            if overlapping:
                match = max(overlapping, key=lambda item: (item["char_end"] - item["char_start"], item["frequency"]))
                covered_indices = [
                    candidate_index
                    for candidate_index, candidate in enumerate(group_records)
                    if candidate["char_end"] > match["char_start"]
                    and candidate["char_start"] < match["char_end"]
                ]
                if len(covered_indices) == 1:
                    position_code = 3
                elif index == covered_indices[0]:
                    position_code = 0
                elif index == covered_indices[-1]:
                    position_code = 2
                else:
                    position_code = 1
                lexical[0] = float(match["exact_case"])
                lexical[1] = 1.0
                lexical[2 + position_code] = 1.0
                lexical[6] = math.log1p(match["frequency"]) / math.log1p(matcher.max_frequency)
                lexical[7] = match["ambiguity"]
                majority_index = ENTITY_TYPES.index(match["entity_type"])
                residual = max(0.0, 1.0 - match["type_probability"])
                for type_index in range(4):
                    lexical[8 + type_index] = (
                        match["type_probability"]
                        if type_index == majority_index
                        else residual / max(len(ENTITY_TYPES) - 1, 1)
                    )
            categorical, numeric = _orthographic_features(
                token["surface_text"], index, len(group_records)
            )
            row = token.copy()
            row["lexical_features"] = lexical
            row["orthographic_categories"] = categorical
            row["orthographic_numeric"] = numeric
            rows.append(row)
    return pd.DataFrame(rows)


def _boundary_spans_from_tags(tags: Sequence[str]) -> list[tuple[int, int]]:
    spans = []
    current = None
    for index, tag in enumerate(tags):
        prefix = tag.split("-", 1)[0]
        if prefix == "S":
            if current is not None:
                spans.append((current, index))
                current = None
            spans.append((index, index + 1))
        elif prefix == "B":
            if current is not None:
                spans.append((current, index))
            current = index
        elif prefix == "I":
            if current is None:
                current = index
        elif prefix == "E":
            if current is None:
                current = index
            spans.append((current, index + 1))
            current = None
        else:
            if current is not None:
                spans.append((current, index))
                current = None
    if current is not None:
        spans.append((current, len(tags)))
    return spans


class ACTSequenceDataset(Dataset):
    def __init__(self, feature_frame: pd.DataFrame, context_cache: dict[str, torch.Tensor]):
        boundary_to_id = {tag: index for index, tag in enumerate(BOUNDARY_TAGS)}
        type_to_id = {entity_type: index for index, entity_type in enumerate(ENTITY_TYPES)}
        self.records = []
        for sentence_id, group in feature_frame.sort_values(["sentence_id", "token_index"]).groupby(
            "sentence_id", sort=False
        ):
            tags = group.typed_tag.tolist()
            spans = []
            for start, end in _boundary_spans_from_tags(tags):
                typed = next(
                    (tag.split("-", 1)[1] for tag in tags[start:end] if "-" in tag),
                    "drug",
                )
                spans.append((start, end, type_to_id[typed]))
            context = context_cache[sentence_id].float()
            assert len(context) == len(group)
            self.records.append(
                {
                    "sentence_id": sentence_id,
                    "context": context,
                    "lexical": torch.tensor(np.stack(group.lexical_features), dtype=torch.float32),
                    "ortho_cat": torch.tensor(np.stack(group.orthographic_categories), dtype=torch.long),
                    "ortho_num": torch.tensor(np.stack(group.orthographic_numeric), dtype=torch.float32),
                    "boundary_labels": torch.tensor(
                        [boundary_to_id[tag.split("-", 1)[0]] for tag in tags], dtype=torch.long
                    ),
                    "gold_spans": spans,
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_act(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(item["context"]) for item in batch], dtype=torch.long)
    span_batch_indices = []
    span_starts = []
    span_ends = []
    span_types = []
    for batch_index, item in enumerate(batch):
        for start, end, type_id in item["gold_spans"]:
            span_batch_indices.append(batch_index)
            span_starts.append(start)
            span_ends.append(end)
            span_types.append(type_id)
    return {
        "sentence_ids": [item["sentence_id"] for item in batch],
        "context": pad_sequence([item["context"] for item in batch], batch_first=True),
        "lexical": pad_sequence([item["lexical"] for item in batch], batch_first=True),
        "ortho_cat": pad_sequence([item["ortho_cat"] for item in batch], batch_first=True),
        "ortho_num": pad_sequence([item["ortho_num"] for item in batch], batch_first=True),
        "boundary_labels": pad_sequence(
            [item["boundary_labels"] for item in batch], batch_first=True, padding_value=0
        ),
        "mask": torch.arange(int(lengths.max())).unsqueeze(0) < lengths.unsqueeze(1),
        "gold_spans": [item["gold_spans"] for item in batch],
        "span_batch_indices": torch.tensor(span_batch_indices, dtype=torch.long),
        "span_starts": torch.tensor(span_starts, dtype=torch.long),
        "span_ends": torch.tensor(span_ends, dtype=torch.long),
        "span_types": torch.tensor(span_types, dtype=torch.long),
    }


@dataclass
class ACTConfig:
    context_dim: int = 384
    fusion_dim: int = 128
    gate_hidden_size: int = 128
    local_hidden_size: int = 64
    dropout: float = 0.25
    use_lexical: bool = True
    use_orthographic: bool = True
    use_adaptive_gate: bool = True
    use_local_refinement: bool = True
    use_crf: bool = True
    use_span_attention: bool = True
    use_type_head: bool = True
    use_confidence_loss: bool = True
    parameter_matched_control: bool = False


class ACTNER(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        dimension = config.fusion_dim
        self.context_projection = nn.Linear(config.context_dim, dimension)
        self.lexical_projection = nn.Sequential(
            nn.Linear(12, dimension), nn.GELU(), nn.Dropout(config.dropout), nn.LayerNorm(dimension)
        )
        self.capitalization_embedding = nn.Embedding(4, 4)
        self.digit_embedding = nn.Embedding(3, 3)
        self.hyphen_embedding = nn.Embedding(2, 2)
        self.greek_embedding = nn.Embedding(2, 2)
        self.punctuation_embedding = nn.Embedding(2, 2)
        self.morphology_embedding = nn.Embedding(13, 6)
        self.prefix_embedding = nn.Embedding(128, 8)
        self.suffix_embedding = nn.Embedding(128, 8)
        ortho_input = 4 + 3 + 2 + 2 + 2 + 6 + 8 + 8 + 2
        self.ortho_projection = nn.Sequential(
            nn.Linear(ortho_input, dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(dimension),
        )
        self.gate = nn.Sequential(
            nn.Linear(dimension * 3, config.gate_hidden_size),
            nn.GELU(),
            nn.Linear(config.gate_hidden_size, dimension),
            nn.Sigmoid(),
        )
        self.lexico_ortho_projection = nn.Linear(dimension * 2, dimension)
        self.concatenation_projection = nn.Sequential(
            nn.Linear(dimension * 3, dimension), nn.GELU(), nn.LayerNorm(dimension)
        )
        self.parameter_control = nn.Sequential(
            nn.Linear(dimension, dimension * 2), nn.GELU(), nn.Linear(dimension * 2, dimension)
        )
        self.local_encoder = nn.LSTM(
            dimension,
            config.local_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        local_dimension = config.local_hidden_size * 2 if config.use_local_refinement else dimension
        self.local_projection = nn.Linear(local_dimension, dimension)
        self.local_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(config.dropout)
        self.boundary_emission = nn.Linear(dimension, len(BOUNDARY_TAGS))
        self.context_emission = nn.Linear(dimension, len(BOUNDARY_TAGS))
        self.crf = LinearChainCRF(len(BOUNDARY_TAGS), boundary_transition_mask())
        self.attention_vector = nn.Parameter(torch.zeros(dimension))
        nn.init.normal_(self.attention_vector, std=0.02)
        self.type_head = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dimension, 4)
        )

    def orthographic_representation(
        self, categories: torch.Tensor, numeric: torch.Tensor
    ) -> torch.Tensor:
        embeddings = [
            self.capitalization_embedding(categories[..., 0]),
            self.digit_embedding(categories[..., 1]),
            self.hyphen_embedding(categories[..., 2]),
            self.greek_embedding(categories[..., 3]),
            self.punctuation_embedding(categories[..., 4]),
            self.morphology_embedding(categories[..., 5]),
            self.prefix_embedding(categories[..., 6]),
            self.suffix_embedding(categories[..., 7]),
            numeric,
        ]
        return self.ortho_projection(torch.cat(embeddings, dim=-1))

    def encode(
        self,
        context: torch.Tensor,
        lexical: torch.Tensor,
        ortho_cat: torch.Tensor,
        ortho_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_representation = self.context_projection(context)
        lexical_representation = self.lexical_projection(lexical)
        orthographic_representation = self.orthographic_representation(ortho_cat, ortho_num)
        if not self.config.use_lexical:
            lexical_representation = torch.zeros_like(lexical_representation)
        if not self.config.use_orthographic:
            orthographic_representation = torch.zeros_like(orthographic_representation)
        combined = torch.cat(
            [context_representation, lexical_representation, orthographic_representation], dim=-1
        )
        if self.config.use_adaptive_gate:
            gate = self.gate(combined)
            lexical_ortho = self.lexico_ortho_projection(
                torch.cat([lexical_representation, orthographic_representation], dim=-1)
            )
            fused = gate * context_representation + (1.0 - gate) * lexical_ortho
        else:
            gate = torch.full_like(context_representation, 0.5)
            fused = self.concatenation_projection(combined)
        if self.config.parameter_matched_control:
            fused = fused + self.parameter_control(fused)
        if self.config.use_local_refinement:
            local, _ = self.local_encoder(self.dropout(fused))
            refined = self.local_projection(local)
        else:
            refined = self.local_projection(fused)
        refined = self.local_norm(self.dropout(refined))
        return refined, gate, lexical_representation, context_representation

    def span_representation(self, sequence: torch.Tensor, start: int, end: int) -> torch.Tensor:
        span = sequence[start:end]
        if not self.config.use_span_attention:
            return span[0]
        weights = torch.softmax(span @ self.attention_vector, dim=0)
        return torch.sum(span * weights.unsqueeze(-1), dim=0)

    def forward(self, batch: dict) -> dict:
        refined, gate, lexical_representation, context_representation = self.encode(
            batch["context"], batch["lexical"], batch["ortho_cat"], batch["ortho_num"]
        )
        return {
            "refined": refined,
            "gate": gate,
            "lexical_representation": lexical_representation,
            "context_representation": context_representation,
            "emissions": self.boundary_emission(refined),
            "context_emissions": self.context_emission(context_representation),
        }

    def composite_loss(
        self,
        batch: dict,
        outputs: dict,
        lambda_type: float,
        lambda_confidence: float,
        lambda_regularization: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch["boundary_labels"]
        mask = batch["mask"]
        emissions = outputs["emissions"]
        if self.config.use_crf:
            boundary_loss = -self.crf.log_likelihood(emissions, labels, mask).mean()
        else:
            boundary_loss = F.cross_entropy(emissions[mask], labels[mask])
        if self.config.use_type_head and batch["span_types"].numel() > 0:
            span_sequences = outputs["refined"][batch["span_batch_indices"]]
            if self.config.use_span_attention:
                positions = torch.arange(span_sequences.size(1), device=emissions.device).unsqueeze(0)
                span_mask = (positions >= batch["span_starts"].unsqueeze(1)) & (
                    positions < batch["span_ends"].unsqueeze(1)
                )
                attention_scores = span_sequences @ self.attention_vector
                attention_scores = attention_scores.masked_fill(~span_mask, -1e4)
                attention_weights = torch.softmax(attention_scores, dim=1)
                span_representations = torch.sum(
                    span_sequences * attention_weights.unsqueeze(-1), dim=1
                )
            else:
                span_representations = span_sequences[
                    torch.arange(len(span_sequences), device=emissions.device), batch["span_starts"]
                ]
            stacked_type_logits = self.type_head(span_representations)
            stacked_type_targets = batch["span_types"]
            type_loss = F.cross_entropy(stacked_type_logits, stacked_type_targets)
            type_probabilities = stacked_type_logits.softmax(-1)
            type_one_hot = F.one_hot(stacked_type_targets, num_classes=4).float()
            type_brier = torch.mean((type_probabilities - type_one_hot) ** 2)
        else:
            type_loss = torch.zeros((), device=emissions.device)
            type_brier = torch.zeros((), device=emissions.device)
        boundary_probabilities = emissions.softmax(-1)
        boundary_one_hot = F.one_hot(labels, num_classes=len(BOUNDARY_TAGS)).float()
        boundary_brier = torch.mean((boundary_probabilities[mask] - boundary_one_hot[mask]) ** 2)
        confidence_loss = boundary_brier + type_brier if self.config.use_confidence_loss else torch.zeros_like(boundary_loss)
        gate_penalty = (outputs["gate"][mask].mean() - 0.5) ** 2
        transition_penalty = torch.mean(self.crf.transitions**2)
        regularization = gate_penalty + transition_penalty
        total = (
            boundary_loss
            + lambda_type * type_loss
            + lambda_confidence * confidence_loss
            + lambda_regularization * regularization
        )
        components = {
            "boundary": float(boundary_loss.detach().cpu()),
            "type": float(type_loss.detach().cpu()),
            "confidence": float(confidence_loss.detach().cpu()),
            "regularization": float(regularization.detach().cpu()),
        }
        return total, components

    def decode_boundaries(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        if self.config.use_crf:
            return self.crf.decode(emissions, mask)
        lengths = mask.long().sum(1).tolist()
        tags = emissions.argmax(-1)
        return [tags[index, :length].tolist() for index, length in enumerate(lengths)]


def _act_batch_to_device(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if torch.is_tensor(value) else value
    return output


def predict_act(
    model: ACTNER,
    feature_frame: pd.DataFrame,
    sentence_frame: pd.DataFrame,
    context_cache: dict[str, torch.Tensor],
    split: str,
    seed: int,
    batch_size: int = 40,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    device = next(model.parameters()).device
    dataset = ACTSequenceDataset(feature_frame, context_cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_act)
    token_groups = {
        key: group.sort_values("token_index").reset_index(drop=True)
        for key, group in feature_frame.groupby("sentence_id")
    }
    span_rows = []
    gate_rows = []
    model.eval()
    start_time = perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _act_batch_to_device(raw_batch, device)
            outputs = model(batch)
            paths = model.decode_boundaries(outputs["emissions"], batch["mask"])
            boundary_probabilities = outputs["emissions"].softmax(-1)
            context_probabilities = outputs["context_emissions"].softmax(-1)
            for batch_index, (sentence_id, path) in enumerate(zip(batch["sentence_ids"], paths)):
                tokens = token_groups[sentence_id]
                tags = [BOUNDARY_TAGS[tag] for tag in path]
                for token_index, gate_value in enumerate(outputs["gate"][batch_index, : len(path)].mean(-1)):
                    gate_rows.append(
                        {
                            "sentence_id": sentence_id,
                            "token_index": token_index,
                            "gate_context_weight": float(gate_value.detach().cpu()),
                            "gold_boundary_tag": tokens.iloc[token_index].boundary_tag,
                            "lexicon_match": bool(tokens.iloc[token_index].lexical_features[1] > 0),
                        }
                    )
                for span_index, (span_start, span_end) in enumerate(_boundary_spans_from_tags(tags)):
                    representation = model.span_representation(
                        outputs["refined"][batch_index], span_start, span_end
                    )
                    if model.config.use_type_head:
                        type_distribution = model.type_head(representation).softmax(-1)
                        type_id = int(type_distribution.argmax())
                        type_probability = float(type_distribution[type_id].detach().cpu())
                    else:
                        lexical_type = batch["lexical"][batch_index, span_start:span_end, 8:12].mean(0)
                        type_id = int(lexical_type.argmax()) if float(lexical_type.sum()) > 0 else 0
                        type_probability = float(max(float(lexical_type[type_id].detach().cpu()), 0.25))
                    boundary_values = [
                        boundary_probabilities[batch_index, token_index, path[token_index]]
                        for token_index in range(span_start, span_end)
                    ]
                    boundary_probability = float(torch.stack(boundary_values).mean().detach().cpu())
                    context_probability = float(
                        context_probabilities[batch_index, span_start:span_end].max(-1).values.mean().detach().cpu()
                    )
                    lexical_slice = batch["lexical"][batch_index, span_start:span_end]
                    lexical_present = float(lexical_slice[:, 1].max().detach().cpu())
                    lexical_frequency = float(lexical_slice[:, 6].max().detach().cpu())
                    lexical_probability = 0.25 + 0.75 * lexical_present * lexical_frequency
                    raw_probability = float(
                        np.prod(
                            [
                                max(boundary_probability, 1e-6),
                                max(type_probability, 1e-6),
                                max(lexical_probability, 1e-6),
                                max(context_probability, 1e-6),
                            ]
                        )
                        ** 0.25
                    )
                    span_rows.append(
                        {
                            "document_id": tokens.iloc[0].document_id,
                            "sentence_id": sentence_id,
                            "char_start": int(tokens.iloc[span_start].char_start),
                            "char_end": int(tokens.iloc[span_end - 1].char_end),
                            "token_start": int(tokens.iloc[span_start].token_index),
                            "token_end": int(tokens.iloc[span_end - 1].token_index) + 1,
                            "predicted_type": ENTITY_TYPES[type_id],
                            "boundary_probability": boundary_probability,
                            "type_probability": type_probability,
                            "lexical_probability": lexical_probability,
                            "context_probability": context_probability,
                            "raw_probability": raw_probability,
                        }
                    )
    elapsed = perf_counter() - start_time
    required_spans = [
        {key: value for key, value in row.items() if key not in ("lexical_probability", "context_probability", "raw_probability")}
        for row in span_rows
    ]
    predictions = finalize_prediction_frame(
        required_spans, sentence_frame, "ACT-NER", split, seed
    )
    if span_rows:
        extras = pd.DataFrame(span_rows)[
            [
                "document_id",
                "sentence_id",
                "char_start",
                "char_end",
                "lexical_probability",
                "context_probability",
                "raw_probability",
            ]
        ]
        predictions = predictions.merge(
            extras,
            on=["document_id", "sentence_id", "char_start", "char_end"],
            how="left",
            validate="one_to_one",
        )
        predictions["calibrated_confidence"] = predictions["raw_probability"]
    return predictions, elapsed, pd.DataFrame(gate_rows)


def train_act_model(
    config: ACTConfig,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    train_context: dict[str, torch.Tensor],
    validation_context: dict[str, torch.Tensor],
    validation_sentences: pd.DataFrame,
    validation_entities: pd.DataFrame,
    seed: int,
    lambda_type: float,
    lambda_confidence: float,
    lambda_regularization: float,
    learning_rate: float = 1e-3,
    max_epochs: int = 9,
    patience: int = 2,
    batch_size: int = 40,
    device_name: str = "cpu",
) -> tuple[ACTNER, pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    set_seed(seed)
    device = torch.device(device_name) if device_name != "auto" else compute_device()
    model = ACTNER(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    dataset = ACTSequenceDataset(train_features, train_context)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_act,
    )
    selected_state = copy.deepcopy(model.state_dict())
    selected_score = -1.0
    epochs_without_improvement = 0
    history = []
    training_start = perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        components = defaultdict(list)
        for raw_batch in loader:
            batch = _act_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, loss_components = model.composite_loss(
                batch,
                outputs,
                lambda_type,
                lambda_confidence,
                lambda_regularization,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in loss_components.items():
                components[key].append(value)
        predictions, _, _ = predict_act(
            model,
            validation_features,
            validation_sentences,
            validation_context,
            "validation",
            seed,
            batch_size=batch_size,
        )
        metrics, _ = evaluate_predictions(validation_entities, predictions, "ACT-NER", "validation")
        score = float(metrics.iloc[0]["Strict Typed Entity F1"])
        history.append(
            {
                "Model": "ACT-NER",
                "Seed": seed,
                "Epoch": epoch,
                "Train Composite Loss": float(np.mean(losses)),
                "CRF Loss": float(np.mean(components["boundary"])),
                "Type Loss": float(np.mean(components["type"])),
                "Confidence Loss": float(np.mean(components["confidence"])),
                "Regularization Loss": float(np.mean(components["regularization"])),
                "Validation Strict Boundary F1": metrics.iloc[0]["Strict Boundary F1"],
                "Validation Strict Typed Entity F1": score,
            }
        )
        if score > selected_score + 1e-6:
            selected_score = score
            selected_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    training_elapsed = perf_counter() - training_start
    model.load_state_dict(selected_state)
    predictions, inference_elapsed, gates = predict_act(
        model,
        validation_features,
        validation_sentences,
        validation_context,
        "validation",
        seed,
        batch_size=batch_size,
    )
    return model, predictions, pd.DataFrame(history), training_elapsed + inference_elapsed, gates


def tune_act_loss_weights(
    config: ACTConfig,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    train_context: dict[str, torch.Tensor],
    validation_context: dict[str, torch.Tensor],
    validation_sentences: pd.DataFrame,
    validation_entities: pd.DataFrame,
    seed: int,
    candidates: Sequence[tuple[float, float, float]] = (
        (1.0, 0.00, 0.001),
        (1.0, 0.05, 0.001),
        (1.0, 0.10, 0.001),
        (0.5, 0.05, 0.001),
    ),
) -> tuple[dict, pd.DataFrame]:
    rows = []
    trained = []
    for lambda_type, lambda_confidence, lambda_regularization in candidates:
        model, predictions, history, elapsed, gates = train_act_model(
            config,
            train_features,
            validation_features,
            train_context,
            validation_context,
            validation_sentences,
            validation_entities,
            seed,
            lambda_type,
            lambda_confidence,
            lambda_regularization,
            max_epochs=6,
            patience=1,
        )
        metrics, _ = evaluate_predictions(validation_entities, predictions, "ACT-NER", "validation")
        rows.append(
            {
                "Model": "ACT-NER",
                "Split": "Validation",
                "N": len(validation_entities),
                "lambda_type": lambda_type,
                "lambda_confidence": lambda_confidence,
                "lambda_regularization": lambda_regularization,
                "Strict Boundary F1": metrics.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metrics.iloc[0]["Strict Typed Entity F1"],
                "ECE Proxy": float(
                    np.mean((predictions.raw_probability - 1.0) ** 2) if len(predictions) else 1.0
                ),
            }
        )
        trained.append((lambda_type, lambda_confidence, lambda_regularization, model, predictions))
    tuning = pd.DataFrame(rows)
    selected = tuning.sort_values(
        ["Strict Typed Entity F1", "Strict Boundary F1", "ECE Proxy", "lambda_confidence"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return {
        "lambda_type": float(selected.lambda_type),
        "lambda_confidence": float(selected.lambda_confidence),
        "lambda_regularization": float(selected.lambda_regularization),
    }, tuning


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


class TemperatureCalibrator:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray):
        logits = _logit(probabilities)

        def objective(log_temperature):
            temperature = math.exp(float(log_temperature))
            calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
            return float(np.mean((calibrated - outcomes) ** 2))

        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        self.temperature = math.exp(float(result.x))
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-_logit(probabilities) / self.temperature))


class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray):
        self.model.fit(_logit(probabilities).reshape(-1, 1), outcomes)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray):
        self.model.fit(probabilities, outcomes)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict(probabilities)


class ConfidenceFusionCalibrator:
    feature_columns = (
        "boundary_probability",
        "type_probability",
        "lexical_probability",
        "context_probability",
    )

    def __init__(self):
        self.model = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs")

    def fit(self, frame: pd.DataFrame, outcomes: np.ndarray):
        self.model.fit(frame[list(self.feature_columns)].to_numpy(float), outcomes)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[list(self.feature_columns)].to_numpy(float))[:, 1]


def _ece(probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    if not len(probabilities):
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    return float(
        sum(
            (bins == index).mean()
            * abs(probabilities[bins == index].mean() - outcomes[bins == index].mean())
            for index in range(n_bins)
            if (bins == index).any()
        )
    )


def fit_validation_calibration(
    predictions: pd.DataFrame,
    gold_entities: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[object, str, pd.DataFrame, pd.DataFrame]:
    gold_entities = gold_entities.query("not discontinuous and offset_valid").copy()
    gold_set = set(
        zip(
            gold_entities.sentence_id,
            gold_entities.char_start,
            gold_entities.char_end,
            gold_entities.entity_type,
        )
    )
    frame = predictions.copy()
    frame["outcome"] = [
        int(value in gold_set)
        for value in zip(frame.sentence_id, frame.char_start, frame.char_end, frame.predicted_type)
    ]
    outcomes = frame.outcome.to_numpy(int)
    groups = frame.document_id.to_numpy()
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    methods = ["Raw", "Temperature Scaling", "Platt", "Confidence Fusion"]
    if len(frame) >= 200 and outcomes.sum() >= 30 and (1 - outcomes).sum() >= 30:
        methods.append("Isotonic")
    cross_fitted = {method: np.zeros(len(frame), dtype=float) for method in methods}
    for fit_indices, hold_indices in splitter.split(frame, outcomes, groups):
        fit_frame = frame.iloc[fit_indices]
        hold_frame = frame.iloc[hold_indices]
        fit_outcomes = outcomes[fit_indices]
        raw_fit = fit_frame.raw_probability.to_numpy(float)
        raw_hold = hold_frame.raw_probability.to_numpy(float)
        cross_fitted["Raw"][hold_indices] = raw_hold
        temperature = TemperatureCalibrator().fit(raw_fit, fit_outcomes)
        cross_fitted["Temperature Scaling"][hold_indices] = temperature.predict(raw_hold)
        platt = PlattCalibrator().fit(raw_fit, fit_outcomes)
        cross_fitted["Platt"][hold_indices] = platt.predict(raw_hold)
        fusion = ConfidenceFusionCalibrator().fit(fit_frame, fit_outcomes)
        cross_fitted["Confidence Fusion"][hold_indices] = fusion.predict(hold_frame)
        if "Isotonic" in methods:
            isotonic = IsotonicCalibrator().fit(raw_fit, fit_outcomes)
            cross_fitted["Isotonic"][hold_indices] = isotonic.predict(raw_hold)
    rows = []
    for method in methods:
        probability = np.clip(cross_fitted[method], 0, 1)
        rows.append(
            {
                "Model": "ACT-NER",
                "Split": "Validation cross-fit",
                "N": len(frame),
                "Calibration Method": method,
                "ECE": _ece(probability, outcomes),
                "Brier Score": float(np.mean((probability - outcomes) ** 2)),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["ECE Rank"] = comparison.ECE.rank(method="min")
    comparison["Brier Rank"] = comparison["Brier Score"].rank(method="min")
    comparison["Mean Rank"] = comparison[["ECE Rank", "Brier Rank"]].mean(axis=1)
    selected_method = comparison.sort_values(
        ["Mean Rank", "Brier Score", "ECE", "Calibration Method"]
    ).iloc[0]["Calibration Method"]
    raw = frame.raw_probability.to_numpy(float)
    if selected_method == "Raw":
        calibrator = None
    elif selected_method == "Temperature Scaling":
        calibrator = TemperatureCalibrator().fit(raw, outcomes)
    elif selected_method == "Platt":
        calibrator = PlattCalibrator().fit(raw, outcomes)
    elif selected_method == "Isotonic":
        calibrator = IsotonicCalibrator().fit(raw, outcomes)
    else:
        calibrator = ConfidenceFusionCalibrator().fit(frame, outcomes)
    frame["cross_fitted_confidence"] = cross_fitted[selected_method]
    return calibrator, selected_method, comparison, frame


def apply_calibration(predictions: pd.DataFrame, calibrator: object, method: str) -> pd.DataFrame:
    frame = predictions.copy()
    if method == "Raw":
        confidence = frame.raw_probability.to_numpy(float)
    elif method == "Confidence Fusion":
        confidence = calibrator.predict(frame)
    else:
        confidence = calibrator.predict(frame.raw_probability.to_numpy(float))
    frame["calibrated_confidence"] = np.clip(confidence, 0, 1)
    return frame


def select_confidence_policy(calibration_frame: pd.DataFrame) -> dict:
    frame = calibration_frame.sort_values("cross_fitted_confidence", ascending=False)
    total = len(frame)
    candidate_high = []
    for threshold in np.unique(frame.cross_fitted_confidence):
        selected = frame[frame.cross_fitted_confidence >= threshold]
        if len(selected) >= max(20, int(0.10 * total)) and selected.outcome.mean() >= 0.90:
            candidate_high.append(float(threshold))
    high = min(candidate_high) if candidate_high else float(frame.cross_fitted_confidence.quantile(0.80))
    candidate_low = []
    for threshold in np.unique(frame.cross_fitted_confidence):
        selected = frame[frame.cross_fitted_confidence < threshold]
        if len(selected) >= max(20, int(0.10 * total)) and selected.outcome.mean() <= 0.50:
            candidate_low.append(float(threshold))
    low = max(candidate_low) if candidate_low else float(frame.cross_fitted_confidence.quantile(0.25))
    if low >= high:
        low = float(frame.cross_fitted_confidence.quantile(0.25))
        high = float(frame.cross_fitted_confidence.quantile(0.75))
    return {
        "low_threshold": low,
        "high_threshold": high,
        "selection_split": "Validation cross-fit",
        "high_target_precision": 0.90,
        "low_target_max_accuracy": 0.50,
    }


def apply_confidence_policy(predictions: pd.DataFrame, policy: dict) -> pd.DataFrame:
    frame = predictions.copy()
    confidence = frame.calibrated_confidence
    frame["confidence_category"] = np.select(
        [confidence >= policy["high_threshold"], confidence < policy["low_threshold"]],
        ["high", "low"],
        default="medium",
    )
    frame["review_required"] = frame.confidence_category.eq("low")
    return frame
