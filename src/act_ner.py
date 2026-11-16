from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from core import (
    ENTITY_TYPES,
    TYPED_TAGS,
    calibration_metrics,
    boundary_error_profile,
    display_complete,
    evaluate_predictions,
    expected_calibration_error,
    finalize_prediction_frame,
    gold_for_metrics,
    load_processed,
    load_settings,
    normalize_surface,
    paired_bootstrap_ci,
    paired_statistical_comparison,
    save_figure,
    save_table,
    stable_sha256,
    subgroup_metrics,
)
from models import (
    PlattCalibrator,
    TemperatureCalibrator,
    TransformerSequenceDataset,
    _boundary_spans_from_tags,
    build_transformer_model,
    feature_sequences,
    set_seed,
    transformer_collate,
)


TYPE_TO_ID = {label: index for index, label in enumerate(ENTITY_TYPES)}
ID_TO_TYPE = dict(enumerate(ENTITY_TYPES))
EPSILON = 1e-6


class NotebookLiveProgress:
    """One in-place progress panel; cleared before scientific outputs are shown."""

    def __init__(self, total_units: int):
        from IPython.display import HTML, display

        self._html = HTML
        self._handle = display(HTML(""), display_id=True)
        self._total = max(int(total_units), 1)
        self._completed = 0
        self._started = perf_counter()
        self.update("Frozen expert inference", "Preparing evidence", "-", "-")

    def update(self, phase: str, item: str, seed: int | str, fold: int | str) -> None:
        elapsed = perf_counter() - self._started
        fraction = min(self._completed / self._total, 1.0)
        panel = f"""
        <div style="font-family:'Times New Roman';font-size:18px;line-height:1.35;
                    border:1px solid #9ca3af;padding:12px 14px;max-width:920px;background:#ffffff">
          <div style="font-size:20px;margin-bottom:8px">ACT-NER validation-only execution</div>
          <div style="height:14px;background:#e5e7eb;margin-bottom:9px">
            <div style="height:14px;width:{100 * fraction:.1f}%;background:#35638f"></div>
          </div>
          <table style="border-collapse:collapse;width:100%">
            <tr><td><b>Phase</b></td><td>{phase}</td><td><b>Configuration</b></td><td>{item}</td></tr>
            <tr><td><b>Seed</b></td><td>{seed}</td><td><b>Fold</b></td><td>{fold}</td></tr>
            <tr><td><b>Progress</b></td><td>{self._completed}/{self._total} ({100 * fraction:.1f}%)</td>
                <td><b>Elapsed</b></td><td>{elapsed / 60:.1f} min</td></tr>
          </table>
        </div>
        """
        self._handle.update(self._html(panel))

    def advance(self, phase: str, item: str, seed: int | str, fold: int | str) -> None:
        self._completed = min(self._completed + 1, self._total)
        self.update(phase, item, seed, fold)

    def close(self) -> None:
        self._handle.update(self._html(""))


@dataclass(frozen=True)
class ACTNERConfig:
    context_dim: int = 384
    fusion_dim: int = 128
    gate_hidden_size: int = 192
    dropout: float = 0.15
    boundary_threshold: float = 0.98
    transformer_anchor_threshold: float = 0.925
    rare_signal_anchor_threshold: float = 0.75
    marginal_threshold: float = 0.72
    hard_negative_ratio: int = 4
    type_loss: str = "effective_number"
    source_balancing: str = "balanced_sampler"
    lambda_type: float = 1.0
    lambda_prototype: float = 0.10
    lambda_domain: float = 0.10
    lambda_router: float = 0.02
    lambda_consistency: float = 0.03
    use_transformer_expert: bool = True
    use_crf_expert: bool = True
    use_prototype_expert: bool = True
    use_orthographic: bool = True
    use_lexical_reliability: bool = True
    dynamic_router: bool = True
    sparse_top2: bool = True
    shared_gate: bool = False
    static_average: bool = False
    weak_signal_recovery: bool = True
    domain_adapter: bool = True
    continual_adaptation: bool = True
    shift_triggered: bool = False
    early_exit: bool = True
    hard_negative_mining: bool = True
    class_balanced_type_loss: bool = True
    router_regularization: bool = True
    calibration: bool = True
    candidate_decoder: str = "union"
    parameter_matched_control: bool = False
    transformer_unfreezing_depth: int = 0
    gradient_accumulation_steps: int = 2
    adapter_learning_rate: float = 8e-4
    head_learning_rate: float = 2e-3


@dataclass
class CandidateBundle:
    frame: pd.DataFrame
    context: torch.Tensor
    transformer: torch.Tensor
    crf: torch.Tensor
    orthographic: torch.Tensor
    lexical: torch.Tensor
    transformer_type: torch.Tensor
    crf_type: torch.Tensor

    def subset(self, indices: Sequence[int]) -> "CandidateBundle":
        positions = np.asarray(indices, dtype=int)
        return CandidateBundle(
            frame=self.frame.iloc[positions].reset_index(drop=True),
            context=self.context[positions],
            transformer=self.transformer[positions],
            crf=self.crf[positions],
            orthographic=self.orthographic[positions],
            lexical=self.lexical[positions],
            transformer_type=self.transformer_type[positions],
            crf_type=self.crf_type[positions],
        )


def assert_read_only_artifacts(project_root: Path) -> None:
    contract_path = project_root / "src" / "read_only_artifact_checksums.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert all((project_root / relative).exists() for relative in contract)
    assert all(stable_sha256(project_root / relative) == checksum for relative, checksum in contract.items())


def _softmax_numpy(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.clip(exponent.sum(axis=axis, keepdims=True), EPSILON, None)


def _type_distribution(token_probabilities: np.ndarray, start: int, end: int) -> np.ndarray:
    distribution = np.zeros(len(ENTITY_TYPES), dtype=np.float32)
    for tag_index, tag in enumerate(TYPED_TAGS):
        if "-" in tag:
            distribution[TYPE_TO_ID[tag.split("-", 1)[1]]] += float(
                token_probabilities[start:end, tag_index].mean()
            )
    total = float(distribution.sum())
    return distribution / total if total > 0 else np.full(len(ENTITY_TYPES), 0.25, dtype=np.float32)


def _span_type(tags: Sequence[str], start: int, end: int) -> str:
    return next((tag.split("-", 1)[1] for tag in tags[start:end] if "-" in tag), "drug")


def extract_frozen_expert_evidence(
    project_root: Path,
    tokens: pd.DataFrame,
) -> tuple[dict[str, dict], dict[str, dict], float]:
    settings = load_settings(project_root)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(settings["biomedical_backbone"], use_fast=True)
    checkpoint = torch.load(
        project_root / "Models" / "Comparators" / "biomedical_transformer.pt",
        map_location="cpu",
        weights_only=False,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        transformer = build_transformer_model(settings["biomedical_backbone"])
    transformer.load_state_dict(checkpoint["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transformer = transformer.to(device).eval()
    dataset = TransformerSequenceDataset(tokens, tokenizer.cls_token_id, tokenizer.sep_token_id)
    loader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=False,
        collate_fn=lambda batch: transformer_collate(batch, tokenizer.pad_token_id),
    )
    transformer_rows: dict[str, dict] = {}
    start_time = perf_counter()
    with torch.inference_mode():
        for batch in loader:
            outputs = transformer(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                output_hidden_states=True,
            )
            probabilities = outputs.logits.softmax(-1)
            hidden = outputs.hidden_states[-1]
            for row_index, sentence_id in enumerate(batch["sentence_ids"]):
                length = int(batch["attention_mask"][row_index].sum()) - 2
                row_probability = probabilities[row_index, 1 : length + 1].detach().cpu()
                row_hidden = hidden[row_index, 1 : length + 1].detach().cpu()
                tag_ids = row_probability.argmax(-1).tolist()
                transformer_rows[sentence_id] = {
                    "hidden": row_hidden,
                    "probabilities": row_probability,
                    "tags": [TYPED_TAGS[index] for index in tag_ids],
                }
    del transformer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    crf_model = joblib.load(project_root / "Models" / "Baselines" / "linear_chain_crf.joblib")
    x_sequences, _, sentence_ids = feature_sequences(tokens)
    crf_rows: dict[str, dict] = {}
    for sentence_id, features in zip(sentence_ids, x_sequences):
        tags = crf_model.predict_single(features)
        marginals = crf_model.predict_marginals_single(features)
        probabilities = np.asarray(
            [[float(distribution.get(tag, 0.0)) for tag in TYPED_TAGS] for distribution in marginals],
            dtype=np.float32,
        )
        crf_model.tagger_.set(features)
        sequence_probability = float(crf_model.tagger_.probability(tags))
        selected_marginals = [float(distribution.get(tag, 0.0)) for distribution, tag in zip(marginals, tags)]
        crf_rows[sentence_id] = {
            "probabilities": torch.tensor(probabilities),
            "tags": list(tags),
            "sequence_probability": sequence_probability,
            "transition_confidence": float(np.mean(selected_marginals)),
        }
    elapsed = perf_counter() - start_time
    return transformer_rows, crf_rows, elapsed


def _contiguous_runs(mask: np.ndarray, maximum_length: int = 12) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            for chunk_start in range(start, index, maximum_length):
                runs.append((chunk_start, min(index, chunk_start + maximum_length)))
            start = None
    return runs


def _orthographic_span_features(text: str, source: str, token_length: int) -> np.ndarray:
    value = text or "_"
    letters = [character for character in value if character.isalpha()]
    lower = value.casefold()
    drug_suffixes = ("cillin", "mycin", "mab", "vir", "azole", "pril", "sartan", "olol")
    return np.asarray(
        [
            min(len(value), 80) / 80.0,
            min(token_length, 12) / 12.0,
            float(token_length > 1),
            float(bool(letters) and all(character.isupper() for character in letters)),
            float(value[:1].isupper()),
            float(any(character.isdigit() for character in value)),
            float("-" in value),
            float(any(not character.isalnum() and not character.isspace() for character in value)),
            float(any("α" <= character.casefold() <= "ω" for character in value)),
            float(any(lower.endswith(suffix) for suffix in drug_suffixes)),
            float("/" in value),
            float("(" in value or ")" in value),
            float(source == "DrugBank"),
            float(source == "MedLine"),
        ],
        dtype=np.float32,
    )


def _lexical_reliability(text: str, lexicon_map: dict[str, dict], maximum_frequency: int) -> tuple[np.ndarray, bool]:
    record = lexicon_map.get(normalize_surface(text))
    if record is None:
        return np.zeros(7, dtype=np.float32), False
    probabilities = np.asarray([record[f"p_{label}"] for label in ENTITY_TYPES], dtype=float)
    dominant = float(probabilities.max())
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])) / math.log(4))
    frequency = int(record["frequency"])
    features = np.asarray(
        [
            1.0,
            math.log1p(frequency) / math.log1p(max(maximum_frequency, 1)),
            entropy,
            float(record["ambiguity_score"]),
            1.0,
            dominant,
            float(normalize_surface(text) == normalize_surface(record["surface_form"])),
        ],
        dtype=np.float32,
    )
    return features, True


def build_candidate_bundle(
    project_root: Path,
    sentences: pd.DataFrame,
    entities: pd.DataFrame,
    tokens: pd.DataFrame,
    marginal_threshold: float,
    *,
    expert_evidence: tuple[dict, dict, float] | None = None,
    training_lexicon: pd.DataFrame | None = None,
) -> tuple[CandidateBundle, pd.DataFrame, float]:
    transformer_rows, crf_rows, expert_elapsed = (
        extract_frozen_expert_evidence(project_root, tokens)
        if expert_evidence is None else expert_evidence
    )
    token_groups = {
        key: group.sort_values("token_index").reset_index(drop=True)
        for key, group in tokens.groupby("sentence_id", sort=False)
    }
    sentence_meta = sentences.set_index("sentence_id")[["document_id", "source", "text"]].to_dict("index")
    lexicon = (
        pd.read_parquet(project_root / "dataset/Processed/training_lexicon.parquet")
        if training_lexicon is None else training_lexicon
    )
    lexicon_map = {row["normalized_surface"]: row for row in lexicon.to_dict("records")}
    maximum_frequency = int(lexicon.frequency.max())
    gold = gold_for_metrics(entities)
    gold_map = {
        (row.sentence_id, int(row.char_start), int(row.char_end)): row.entity_type
        for row in gold.itertuples(index=False)
    }
    candidates: dict[tuple[str, int, int], dict] = {}

    def register(sentence_id: str, start: int, end: int, origin: str, shifted: bool = False) -> None:
        n_tokens = len(token_groups[sentence_id])
        if not (0 <= start < end <= n_tokens):
            return
        key = (sentence_id, start, end)
        record = candidates.setdefault(
            key,
            {
                "sentence_id": sentence_id,
                "token_start": start,
                "token_end": end,
                "origin_transformer": False,
                "origin_crf": False,
                "origin_high_transformer": False,
                "origin_high_crf": False,
                "origin_shift_transformer": False,
                "origin_shift_crf": False,
                "generated_candidate": True,
            },
        )
        field = f"origin_shift_{origin}" if shifted else f"origin_{origin}"
        record[field] = True

    for sentence_id, token_group in token_groups.items():
        transformer = transformer_rows[sentence_id]
        crf = crf_rows[sentence_id]
        transformer_spans = _boundary_spans_from_tags(transformer["tags"])
        crf_spans = _boundary_spans_from_tags(crf["tags"])
        for start, end in transformer_spans:
            register(sentence_id, start, end, "transformer")
        for start, end in crf_spans:
            register(sentence_id, start, end, "crf")
        transformer_entity = 1.0 - transformer["probabilities"][:, TYPED_TAGS.index("O")].numpy()
        crf_entity = 1.0 - crf["probabilities"][:, TYPED_TAGS.index("O")].numpy()
        for start, end in _contiguous_runs(transformer_entity >= marginal_threshold):
            register(sentence_id, start, end, "high_transformer")
        for start, end in _contiguous_runs(crf_entity >= marginal_threshold):
            register(sentence_id, start, end, "high_crf")
        original = list(candidates.items())
        for (candidate_sentence, start, end), record in original:
            if candidate_sentence != sentence_id:
                continue
            origins = []
            if record["origin_transformer"] or record["origin_high_transformer"]:
                origins.append("transformer")
            if record["origin_crf"] or record["origin_high_crf"]:
                origins.append("crf")
            for delta_start, delta_end in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)):
                shifted_start = max(0, start + delta_start)
                shifted_end = min(len(token_group), end + delta_end)
                for origin in origins:
                    register(sentence_id, shifted_start, shifted_end, origin, shifted=True)

    generated_keys = set(candidates)
    for gold_row in gold.itertuples(index=False):
        token_group = token_groups.get(gold_row.sentence_id)
        if token_group is None:
            continue
        covered = token_group[
            (token_group.char_end > int(gold_row.char_start))
            & (token_group.char_start < int(gold_row.char_end))
        ]
        if covered.empty:
            continue
        start = int(covered.token_index.min())
        end = int(covered.token_index.max()) + 1
        key = (gold_row.sentence_id, start, end)
        if key not in candidates:
            candidates[key] = {
                "sentence_id": gold_row.sentence_id,
                "token_start": start,
                "token_end": end,
                "origin_transformer": False,
                "origin_crf": False,
                "origin_high_transformer": False,
                "origin_high_crf": False,
                "origin_shift_transformer": False,
                "origin_shift_crf": False,
                "generated_candidate": False,
            }

    rows: list[dict] = []
    context_vectors: list[torch.Tensor] = []
    transformer_vectors: list[np.ndarray] = []
    crf_vectors: list[np.ndarray] = []
    orthographic_vectors: list[np.ndarray] = []
    lexical_vectors: list[np.ndarray] = []
    transformer_types: list[np.ndarray] = []
    crf_types: list[np.ndarray] = []
    for (sentence_id, start, end), origins in sorted(candidates.items()):
        token_group = token_groups[sentence_id]
        meta = sentence_meta[sentence_id]
        char_start = int(token_group.iloc[start].char_start)
        char_end = int(token_group.iloc[end - 1].char_end)
        text = str(meta["text"])[char_start:char_end]
        transformer = transformer_rows[sentence_id]
        crf = crf_rows[sentence_id]
        transformer_probability = transformer["probabilities"].numpy()
        crf_probability = crf["probabilities"].numpy()
        transformer_entity = 1.0 - transformer_probability[:, TYPED_TAGS.index("O")]
        crf_entity = 1.0 - crf_probability[:, TYPED_TAGS.index("O")]
        span_attention = torch.softmax(torch.tensor(transformer_entity[start:end]), dim=0)
        span_hidden = transformer["hidden"][start:end]
        pooled = torch.sum(span_hidden * span_attention.unsqueeze(-1), dim=0)
        zero = torch.zeros_like(pooled)
        left = transformer["hidden"][start - 1] if start > 0 else zero
        right = transformer["hidden"][end] if end < len(transformer["hidden"]) else zero
        context_vectors.append(torch.cat([pooled, left, right, span_hidden[0], span_hidden[-1]]).float())
        transformer_type = _type_distribution(transformer_probability, start, end)
        crf_type = _type_distribution(crf_probability, start, end)
        transformer_tag_ids = transformer_probability.argmax(axis=1)
        crf_tag_ids = crf_probability.argmax(axis=1)
        transformer_anchor_score = float(
            transformer_probability[np.arange(start, end), transformer_tag_ids[start:end]].mean()
        )
        crf_anchor_score = float(
            crf_probability[np.arange(start, end), crf_tag_ids[start:end]].mean()
        )
        transformer_entropy = float(
            -(transformer_type * np.log(np.clip(transformer_type, EPSILON, 1.0))).sum()
        )
        crf_entropy = float(
            -(crf_type * np.log(np.clip(crf_type, EPSILON, 1.0))).sum()
        )
        transformer_types.append(transformer_type)
        crf_types.append(crf_type)
        transformer_membership = float(origins["origin_transformer"])
        crf_membership = float(origins["origin_crf"])
        transformer_vectors.append(
            np.asarray(
                [
                    transformer_membership,
                    float(transformer_entity[start:end].mean()),
                    float(transformer_entity[start:end].min()),
                    float(transformer_entity[start:end].max()),
                    float(transformer_entity[start]),
                    float(transformer_entity[end - 1]),
                    *transformer_type.tolist(),
                ],
                dtype=np.float32,
            )
        )
        crf_vectors.append(
            np.asarray(
                [
                    crf_membership,
                    float(crf_entity[start:end].mean()),
                    float(crf_entity[start:end].min()),
                    float(crf_entity[start:end].max()),
                    float(crf_entity[start]),
                    float(crf_entity[end - 1]),
                    float(crf["sequence_probability"]),
                    float(crf["transition_confidence"]),
                    *crf_type.tolist(),
                ],
                dtype=np.float32,
            )
        )
        orthographic = _orthographic_span_features(text, str(meta["source"]), end - start)
        lexical_reliability, seen = _lexical_reliability(text, lexicon_map, maximum_frequency)
        orthographic_vectors.append(orthographic)
        lexical_vectors.append(lexical_reliability)
        key = (sentence_id, char_start, char_end)
        gold_type = gold_map.get(key)
        disagreement = bool(
            transformer_type.argmax() != crf_type.argmax()
            or abs(float(transformer_entity[start:end].mean()) - float(crf_entity[start:end].mean())) > 0.20
        )
        rare_probability = float(
            max(
                transformer_type[TYPE_TO_ID["brand"]],
                transformer_type[TYPE_TO_ID["drug_n"]],
                crf_type[TYPE_TO_ID["brand"]],
                crf_type[TYPE_TO_ID["drug_n"]],
            )
        )
        low_boundary_margin = bool(
            max(float(transformer_entity[start:end].mean()), float(crf_entity[start:end].mean())) < 0.72
        )
        weak_signal = bool(
            max(transformer_entropy, crf_entropy) > 0.85
            or disagreement
            or not seen
            or rare_probability > 0.25
            or str(meta["source"]) == "MedLine"
            or end - start > 1
            or low_boundary_margin
        )
        candidate_source = "+".join(
            source
            for source, present in (
                ("Transformer", origins["origin_transformer"] or origins["origin_high_transformer"]),
                ("CRF", origins["origin_crf"] or origins["origin_high_crf"]),
                ("Boundary correction", origins["origin_shift_transformer"] or origins["origin_shift_crf"]),
            )
            if present
        ) or "Gold training support"
        rows.append(
            {
                **origins,
                "document_id": meta["document_id"],
                "source": meta["source"],
                "char_start": char_start,
                "char_end": char_end,
                "entity_text": text,
                "span_label": int(gold_type is not None),
                "type_label": TYPE_TO_ID[gold_type] if gold_type is not None else -1,
                "seen_in_train": seen,
                "multi_token": end - start > 1,
                "transformer_span_score": float(transformer_entity[start:end].mean()),
                "crf_span_score": float(crf_entity[start:end].mean()),
                "transformer_anchor_score": transformer_anchor_score,
                "crf_anchor_score": crf_anchor_score,
                "transformer_predicted_type": ID_TO_TYPE[int(transformer_type.argmax())],
                "crf_predicted_type": ID_TO_TYPE[int(crf_type.argmax())],
                "transformer_type_probability": float(transformer_type.max()),
                "crf_type_probability": float(crf_type.max()),
                "expert_agreement": bool(
                    transformer_membership
                    and crf_membership
                    and transformer_type.argmax() == crf_type.argmax()
                ),
                "expert_disagreement": disagreement,
                "transformer_entropy": transformer_entropy,
                "crf_entropy": crf_entropy,
                "uncertainty": max(transformer_entropy, crf_entropy),
                "rare_signal_probability": rare_probability,
                "low_boundary_margin": low_boundary_margin,
                "weak_signal": weak_signal,
                "candidate_source": candidate_source,
            }
        )
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(["sentence_id", "token_start", "token_end"]).any()
    generated = frame.generated_candidate
    generated_boundary = set(
        zip(frame.loc[generated, "sentence_id"], frame.loc[generated, "char_start"], frame.loc[generated, "char_end"])
    )
    gold_boundary = set(zip(gold.sentence_id, gold.char_start, gold.char_end))
    transformer_boundary = {
        (sentence_id, int(token_groups[sentence_id].iloc[start].char_start), int(token_groups[sentence_id].iloc[end - 1].char_end))
        for sentence_id, evidence in transformer_rows.items()
        for start, end in _boundary_spans_from_tags(evidence["tags"])
    }
    crf_boundary = {
        (sentence_id, int(token_groups[sentence_id].iloc[start].char_start), int(token_groups[sentence_id].iloc[end - 1].char_end))
        for sentence_id, evidence in crf_rows.items()
        for start, end in _boundary_spans_from_tags(evidence["tags"])
    }
    gold_count = len(gold_boundary)
    recall_rows = pd.DataFrame(
        [
            ["Biomedical Transformer", len(transformer_boundary), len(gold_boundary & transformer_boundary) / gold_count if gold_count else np.nan],
            ["Linear-chain CRF", len(crf_boundary), len(gold_boundary & crf_boundary) / gold_count if gold_count else np.nan],
            ["High-recall candidate union", len(generated_boundary), len(gold_boundary & generated_boundary) / gold_count if gold_count else np.nan],
        ],
        columns=["Candidate Source", "Unique Candidates", "Validation Boundary Recall"],
    )
    if gold_count:
        assert recall_rows.iloc[-1]["Validation Boundary Recall"] + 1e-12 >= recall_rows.iloc[:-1]["Validation Boundary Recall"].max()
    bundle = CandidateBundle(
        frame=frame,
        context=torch.stack(context_vectors),
        transformer=torch.tensor(np.stack(transformer_vectors)),
        crf=torch.tensor(np.stack(crf_vectors)),
        orthographic=torch.tensor(np.stack(orthographic_vectors)),
        lexical=torch.tensor(np.stack(lexical_vectors)),
        transformer_type=torch.tensor(np.stack(transformer_types)),
        crf_type=torch.tensor(np.stack(crf_types)),
    )
    assert set(generated_keys).issubset(set(zip(frame.sentence_id, frame.token_start, frame.token_end)))
    return bundle, recall_rows, expert_elapsed


class CandidateDataset(Dataset):
    def __init__(self, bundle: CandidateBundle, indices: Sequence[int]):
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        index = int(self.indices[item])
        row = self.bundle.frame.iloc[index]
        return {
            "index": index,
            "context": self.bundle.context[index],
            "transformer": self.bundle.transformer[index],
            "crf": self.bundle.crf[index],
            "orthographic": self.bundle.orthographic[index],
            "lexical": self.bundle.lexical[index],
            "transformer_type": self.bundle.transformer_type[index],
            "crf_type": self.bundle.crf_type[index],
            "span_label": float(row.span_label),
            "type_label": int(row.type_label),
            "source_id": int(row.source == "DrugBank"),
            "agreement": float(row.expert_agreement),
            "weak_signal": float(row.weak_signal),
            "seen_in_train": float(row.seen_in_train),
            "multi_token": float(row.multi_token),
            "rare_signal": float(row.rare_signal_probability),
        }


def collate_candidates(batch: list[dict]) -> dict:
    tensor_keys = (
        "context",
        "transformer",
        "crf",
        "orthographic",
        "lexical",
        "transformer_type",
        "crf_type",
    )
    output = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    output["indices"] = torch.tensor([item["index"] for item in batch], dtype=torch.long)
    output["span_label"] = torch.tensor([item["span_label"] for item in batch], dtype=torch.float32)
    output["type_label"] = torch.tensor([item["type_label"] for item in batch], dtype=torch.long)
    output["source_id"] = torch.tensor([item["source_id"] for item in batch], dtype=torch.long)
    output["agreement"] = torch.tensor([item["agreement"] for item in batch], dtype=torch.float32)
    output["weak_signal"] = torch.tensor([item["weak_signal"] for item in batch], dtype=torch.float32)
    output["seen_in_train"] = torch.tensor([item["seen_in_train"] for item in batch], dtype=torch.float32)
    output["multi_token"] = torch.tensor([item["multi_token"] for item in batch], dtype=torch.float32)
    output["rare_signal"] = torch.tensor([item["rare_signal"] for item in batch], dtype=torch.float32)
    return output


class LowRankContextAdapter(nn.Module):
    """Trainable low-rank residual adapter over immutable Transformer states."""

    def __init__(self, dimension: int, rank: int = 32):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(values)))


class ACTNER(nn.Module):
    """Adaptive Calibrated Tri-Expert Network with sparse span-level routing."""

    def __init__(self, config: ACTNERConfig):
        super().__init__()
        self.config = config
        dimension = config.fusion_dim
        self.context_projection = nn.Sequential(
            nn.Linear(config.context_dim * 5, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
            nn.Dropout(config.dropout),
        )
        self.context_adapter = LowRankContextAdapter(dimension, rank=max(16, dimension // 4))
        self.weak_context_adapter = LowRankContextAdapter(dimension, rank=max(16, dimension // 4))
        self.domain_embedding = nn.Embedding(2, 8)
        self.domain_projection = nn.Linear(8, dimension, bias=False)
        self.transformer_projection = nn.Linear(10, dimension, bias=False)
        self.sequence_projection = nn.Linear(12, dimension, bias=False)
        self.orthographic_projection = nn.Linear(14, dimension, bias=False)
        self.lexical_projection = nn.Linear(7, dimension, bias=False)
        self.context_expert = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(config.dropout)
        )
        self.sequence_expert = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(config.dropout)
        )
        self.prototype_encoder = nn.Sequential(
            nn.Linear(dimension * 3, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
            nn.Dropout(config.dropout),
        )
        self.class_prototypes = nn.Parameter(torch.empty(len(ENTITY_TYPES), dimension))
        nn.init.orthogonal_(self.class_prototypes)
        router_input = dimension + 10 + 12 + 14 + 7 + 5
        self.router = nn.Sequential(
            nn.Linear(router_input, config.gate_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.gate_hidden_size, 3),
        )
        self.global_router_logits = nn.Parameter(torch.zeros(3))
        self.final_norm = nn.LayerNorm(dimension)
        self.span_verifier = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dimension, 1)
        )
        self.type_head = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dimension, len(ENTITY_TYPES))
        )
        self.parameter_control = nn.Sequential(
            nn.Linear(dimension, dimension * 2), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dimension * 2, dimension)
        )
        for projection in (
            self.transformer_projection,
            self.sequence_projection,
            self.orthographic_projection,
            self.lexical_projection,
            self.domain_projection,
        ):
            nn.init.normal_(projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.span_verifier[-1].weight)
        nn.init.zeros_(self.span_verifier[-1].bias)
        nn.init.zeros_(self.type_head[-1].weight)
        nn.init.zeros_(self.type_head[-1].bias)

    @torch.no_grad()
    def initialize_prototypes(self, bundle: CandidateBundle, indices: np.ndarray, device: torch.device) -> None:
        positive = [index for index in indices if int(bundle.frame.iloc[index].type_label) >= 0]
        if not positive:
            return
        context = self.context_projection(bundle.context[positive].to(device))
        labels = bundle.frame.iloc[positive].type_label.to_numpy(int)
        for class_index in range(len(ENTITY_TYPES)):
            mask = torch.tensor(labels == class_index, device=device)
            if mask.any():
                self.class_prototypes[class_index].copy_(F.normalize(context[mask].mean(0), dim=0))

    def _prototype_expert(
        self,
        context: torch.Tensor,
        orthographic: torch.Tensor,
        lexical: torch.Tensor,
        weak_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weak_input = torch.cat(
            [context, self.orthographic_projection(orthographic), self.lexical_projection(lexical)],
            dim=-1,
        )
        encoded = F.normalize(self.prototype_encoder(weak_input), dim=-1)
        if self.config.use_prototype_expert:
            prototypes = F.normalize(self.class_prototypes, dim=-1)
            logits = encoded @ prototypes.T / 0.20
            prototype_distribution = torch.softmax(logits, dim=-1)
            mask = weak_mask.unsqueeze(-1).to(encoded.dtype)
            prototype_embedding = encoded * mask
            prototype_distribution = prototype_distribution * mask + (1.0 - mask) / len(ENTITY_TYPES)
        else:
            prototype_embedding = torch.zeros_like(context)
            prototype_distribution = torch.full(
                (len(context), len(ENTITY_TYPES)), 1.0 / len(ENTITY_TYPES), device=context.device
            )
        return prototype_embedding, prototype_distribution, weak_mask

    def _router_weights(self, logits: torch.Tensor, weak_mask: torch.Tensor) -> torch.Tensor:
        if self.config.parameter_matched_control:
            return torch.tensor([1.0, 0.0, 0.0], device=logits.device).expand_as(logits)
        if self.config.static_average:
            dense = torch.full_like(logits, 1.0 / 3.0)
        elif self.config.shared_gate:
            dense = torch.softmax(self.global_router_logits, dim=0).expand_as(logits)
        elif not self.config.dynamic_router:
            dense = torch.tensor([1.0, 0.0, 0.0], device=logits.device).expand_as(logits)
        else:
            adjusted = logits.clone()
            if self.config.early_exit:
                adjusted[~weak_mask, 2] = -1e4
            dense = torch.softmax(adjusted, dim=-1)
        if not self.config.sparse_top2 or self.config.static_average:
            return dense
        values, indices = torch.topk(dense, k=2, dim=-1)
        sparse = torch.zeros_like(dense).scatter(1, indices, values)
        return sparse / sparse.sum(-1, keepdim=True).clamp_min(EPSILON)

    def forward(self, batch: dict) -> dict:
        context = self.context_projection(batch["context"])
        if not self.config.use_transformer_expert:
            context = torch.zeros_like(context)
        adapted_context = context + self.context_adapter(context)
        weak_mask = batch["weak_signal"].bool() if self.config.weak_signal_recovery else torch.zeros(len(context), dtype=torch.bool, device=context.device)
        if self.config.weak_signal_recovery and weak_mask.any():
            adapted_context = adapted_context + self.weak_context_adapter(context) * weak_mask.unsqueeze(-1)
        if self.config.domain_adapter:
            adapted_context = adapted_context + self.domain_projection(self.domain_embedding(batch["source_id"]))
        transformer = batch["transformer"] if self.config.use_transformer_expert else torch.zeros_like(batch["transformer"])
        crf = batch["crf"] if self.config.use_crf_expert else torch.zeros_like(batch["crf"])
        orthographic = batch["orthographic"] if self.config.use_orthographic else torch.zeros_like(batch["orthographic"])
        lexical = batch["lexical"] if self.config.use_lexical_reliability else torch.zeros_like(batch["lexical"])
        context_output = self.context_expert(adapted_context) + self.transformer_projection(transformer)
        sequence_output = self.sequence_expert(self.sequence_projection(crf)) if self.config.use_crf_expert else torch.zeros_like(context)
        prototype_output, prototype_distribution, prototype_active = self._prototype_expert(
            adapted_context, orthographic, lexical, weak_mask
        )
        uncertainty = torch.stack(
            [
                1.0 - batch["agreement"],
                batch["weak_signal"],
                1.0 - batch["seen_in_train"],
                batch["multi_token"],
                batch["rare_signal"],
            ],
            dim=-1,
        )
        router_input = torch.cat([adapted_context, transformer, crf, orthographic, lexical, uncertainty], dim=-1)
        router_logits = self.router(router_input)
        if not self.config.use_transformer_expert:
            router_logits[:, 0] = -1e4
        if not self.config.use_crf_expert:
            router_logits[:, 1] = -1e4
        if not self.config.use_prototype_expert:
            router_logits[:, 2] = -1e4
        router_weights = self._router_weights(router_logits, weak_mask)
        experts = torch.stack([context_output, sequence_output, prototype_output], dim=1)
        routed = (router_weights.unsqueeze(-1) * experts).sum(1)
        representation = self.final_norm(adapted_context + routed)
        if self.config.parameter_matched_control:
            representation = representation + self.parameter_control(representation)
        transformer_membership = transformer[:, 0]
        crf_membership = crf[:, 0]
        expert_prior = torch.maximum(transformer[:, 1], crf[:, 1]).clamp(EPSILON, 1 - EPSILON)
        expert_prior = torch.where(
            (transformer_membership + crf_membership) > 0,
            torch.maximum(expert_prior, torch.full_like(expert_prior, 0.55)),
            expert_prior,
        )
        span_logits = torch.logit(expert_prior.clamp(EPSILON, 1 - EPSILON)) + self.span_verifier(representation).squeeze(-1)
        transformer_prior = transformer_membership.unsqueeze(-1) * batch["transformer_type"]
        crf_prior = 1.35 * crf_membership.unsqueeze(-1) * batch["crf_type"]
        prototype_prior = router_weights[:, 2:3] * prototype_distribution
        type_prior = transformer_prior + crf_prior + prototype_prior + 0.05
        type_prior = type_prior / type_prior.sum(-1, keepdim=True)
        type_logits = torch.log(type_prior.clamp_min(EPSILON)) + self.type_head(representation)
        return {
            "span_logits": span_logits,
            "type_logits": type_logits,
            "router_weights": router_weights,
            "router_entropy": -(router_weights * torch.log(router_weights.clamp_min(EPSILON))).sum(-1),
            "prototype_embedding": prototype_output,
            "prototype_distribution": prototype_distribution,
            "prototype_active": prototype_active,
            "early_exit": (~weak_mask) & self.config.early_exit,
        }

    def loss(
        self,
        batch: dict,
        outputs: dict,
        type_weights: torch.Tensor,
        teacher_outputs: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        span_targets = batch["span_label"]
        span_losses = F.binary_cross_entropy_with_logits(outputs["span_logits"], span_targets, reduction="none")
        positive = batch["type_label"] >= 0
        type_losses = torch.zeros_like(span_losses)
        prototype_loss = torch.zeros((), device=span_losses.device)
        if positive.any():
            logits = outputs["type_logits"][positive]
            targets = batch["type_label"][positive]
            if not self.config.class_balanced_type_loss or self.config.type_loss == "weighted_ce":
                positive_losses = F.cross_entropy(logits, targets, weight=type_weights, reduction="none")
            elif self.config.type_loss == "focal":
                base = F.cross_entropy(logits, targets, weight=type_weights, reduction="none")
                probability = logits.softmax(-1).gather(1, targets.unsqueeze(1)).squeeze(1)
                positive_losses = (1.0 - probability).pow(2) * base
            elif self.config.type_loss == "logit_adjusted":
                adjusted = logits + torch.log(type_weights.clamp_min(EPSILON)).unsqueeze(0)
                positive_losses = F.cross_entropy(adjusted, targets, reduction="none")
            else:
                positive_losses = F.cross_entropy(logits, targets, weight=type_weights, reduction="none")
            weak_multiplier = torch.where(batch["weak_signal"][positive].bool(), 1.35, 1.0)
            type_losses[positive] = positive_losses * weak_multiplier
            proto_mask = positive & outputs["prototype_active"]
            if self.config.use_prototype_expert and proto_mask.any():
                prototype_loss = F.nll_loss(
                    torch.log(outputs["prototype_distribution"][proto_mask].clamp_min(EPSILON)),
                    batch["type_label"][proto_mask],
                )
                embeddings = F.normalize(outputs["prototype_embedding"][proto_mask], dim=-1)[:128]
                labels = batch["type_label"][proto_mask][:128]
                if len(embeddings) > 1:
                    similarities = embeddings @ embeddings.T / 0.20
                    identity = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
                    matches = labels[:, None].eq(labels[None, :]) & ~identity
                    log_prob = F.log_softmax(similarities.masked_fill(identity, -1e4), dim=-1)
                    valid = matches.any(dim=1)
                    if valid.any():
                        prototype_loss = prototype_loss - (log_prob * matches).sum(1)[valid].div(matches.sum(1)[valid]).mean()
        span_loss = span_losses.mean()
        type_loss = type_losses[positive].mean() if positive.any() else torch.zeros((), device=span_losses.device)
        domain_loss = torch.zeros((), device=span_losses.device)
        source_objectives = []
        for source_id in (0, 1):
            source_mask = batch["source_id"] == source_id
            if source_mask.any():
                source_objectives.append(span_losses[source_mask].mean() + self.config.lambda_type * type_losses[source_mask & positive].mean() if (source_mask & positive).any() else span_losses[source_mask].mean())
        if self.config.source_balancing == "group_dro" and source_objectives:
            domain_loss = torch.stack(source_objectives).max()
        router_loss = torch.zeros((), device=span_losses.device)
        if self.config.router_regularization and self.config.dynamic_router:
            weights = outputs["router_weights"]
            load = weights.mean(0)
            load_balance = ((load - 1.0 / 3.0) ** 2).mean()
            entropy = outputs["router_entropy"].mean()
            dispersion = weights.std(0, unbiased=False).mean()
            router_loss = load_balance + F.relu(0.30 - entropy) + F.relu(0.015 - dispersion)
        consistency_loss = torch.zeros((), device=span_losses.device)
        if self.config.continual_adaptation and self.config.shift_triggered and teacher_outputs is not None:
            weak = batch["weak_signal"].bool()
            if weak.any():
                consistency_loss = F.mse_loss(
                    torch.sigmoid(outputs["span_logits"][weak]),
                    torch.sigmoid(teacher_outputs["span_logits"][weak]).detach(),
                ) + F.kl_div(
                    F.log_softmax(outputs["type_logits"][weak], dim=-1),
                    F.softmax(teacher_outputs["type_logits"][weak], dim=-1).detach(),
                    reduction="batchmean",
                )
        total = (
            span_loss
            + self.config.lambda_type * type_loss
            + self.config.lambda_prototype * prototype_loss
            + self.config.lambda_domain * domain_loss
            + self.config.lambda_router * router_loss
            + self.config.lambda_consistency * consistency_loss
        )
        return total, {
            "Span Loss": float(span_loss.detach().cpu()),
            "Type Loss": float(type_loss.detach().cpu()),
            "Prototype Loss": float(prototype_loss.detach().cpu()),
            "Domain Loss": float(domain_loss.detach().cpu()),
            "Router Loss": float(router_loss.detach().cpu()),
            "Consistency Loss": float(consistency_loss.detach().cpu()),
        }


def _candidate_mask(frame: pd.DataFrame, config: ACTNERConfig) -> np.ndarray:
    mask = frame.generated_candidate.to_numpy(bool)
    transformer_origin = frame[
        ["origin_transformer", "origin_high_transformer", "origin_shift_transformer"]
    ].any(axis=1).to_numpy(bool)
    crf_origin = frame[["origin_crf", "origin_high_crf", "origin_shift_crf"]].any(axis=1).to_numpy(bool)
    exact_expert = frame[["origin_transformer", "origin_crf"]].any(axis=1).to_numpy(bool)
    marginal_support = (
        frame[["transformer_span_score", "crf_span_score"]].max(axis=1).to_numpy(float)
        >= config.marginal_threshold
    )
    mask &= exact_expert | marginal_support
    if not config.use_transformer_expert or config.candidate_decoder == "crf_only":
        mask &= crf_origin
    if not config.use_crf_expert or config.candidate_decoder == "transformer_only":
        mask &= transformer_origin
    return mask


def _training_indices(
    frame: pd.DataFrame,
    document_ids: set[str],
    config: ACTNERConfig,
    seed: int,
) -> np.ndarray:
    eligible = frame.document_id.isin(document_ids).to_numpy(bool)
    candidate_mask = _candidate_mask(frame, config)
    eligible &= candidate_mask | frame.span_label.eq(1).to_numpy(bool)
    positives = np.flatnonzero(eligible & frame.span_label.eq(1).to_numpy(bool))
    negatives = np.flatnonzero(eligible & frame.span_label.eq(0).to_numpy(bool))
    if config.hard_negative_mining:
        hard = frame.iloc[negatives][
            [
                "origin_transformer",
                "origin_crf",
                "origin_shift_transformer",
                "origin_shift_crf",
                "expert_agreement",
            ]
        ].sum(axis=1).to_numpy()
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(0, 1e-4, size=len(negatives))
        order = np.argsort(-(hard + jitter))
        maximum = min(len(negatives), max(len(positives) * config.hard_negative_ratio, len(positives)))
        negatives = negatives[order[:maximum]]
    return np.concatenate([positives, negatives])


def _class_weights(frame: pd.DataFrame, indices: np.ndarray, mode: str) -> torch.Tensor:
    labels = frame.iloc[indices].loc[lambda data: data.type_label >= 0, "type_label"].to_numpy(int)
    counts = np.bincount(labels, minlength=len(ENTITY_TYPES)).astype(float)
    if mode == "effective_number":
        beta = 0.999
        weights = (1.0 - beta) / np.clip(1.0 - np.power(beta, counts), EPSILON, None)
    else:
        weights = counts.sum() / np.clip(len(ENTITY_TYPES) * counts, 1.0, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _sampler_weights(frame: pd.DataFrame, indices: np.ndarray, config: ACTNERConfig) -> torch.Tensor:
    subset = frame.iloc[indices]
    document_counts = subset.document_id.value_counts()
    weights = 1.0 / subset.document_id.map(document_counts).to_numpy(float)
    if config.source_balancing == "balanced_sampler":
        source_counts = subset.source.value_counts()
        weights *= 1.0 / subset.source.map(source_counts).to_numpy(float)
    if config.class_balanced_type_loss:
        rare = subset.type_label.isin([TYPE_TO_ID["brand"], TYPE_TO_ID["drug_n"]]).to_numpy(bool)
        weights *= np.where(rare, 2.0, 1.0)
    return torch.tensor(weights / np.mean(weights), dtype=torch.double)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_act_ner_model(
    bundle: CandidateBundle,
    train_documents: set[str],
    config: ACTNERConfig,
    seed: int,
    max_epochs: int = 8,
    batch_size: int = 1024,
) -> tuple[ACTNER, pd.DataFrame, float]:
    assert config.transformer_unfreezing_depth == 0
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ACTNER(config).to(device)
    indices = _training_indices(bundle.frame, train_documents, config, seed)
    model.initialize_prototypes(bundle, indices, device)
    dataset = CandidateDataset(bundle, indices)
    sampler = WeightedRandomSampler(
        _sampler_weights(bundle.frame, indices, config),
        num_samples=len(indices),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_candidates)
    type_weights = _class_weights(bundle.frame, indices, config.type_loss).to(device)
    adapter_parameters = list(model.context_adapter.parameters()) + list(model.weak_context_adapter.parameters())
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": config.adapter_learning_rate},
            {"params": [parameter for parameter in model.parameters() if id(parameter) not in adapter_ids], "lr": config.head_learning_rate},
        ],
        weight_decay=1e-4,
    )
    teacher = copy.deepcopy(model).eval() if config.continual_adaptation and config.shift_triggered else None
    if teacher is not None:
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    history: list[dict] = []
    started = perf_counter()
    best_loss = math.inf
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        component_rows: list[dict] = []
        totals: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = _to_device(raw_batch, device)
            perturbed = dict(batch)
            if config.continual_adaptation and config.shift_triggered:
                perturbed["context"] = batch["context"] + 0.005 * torch.randn_like(batch["context"])
            outputs = model(perturbed)
            with torch.no_grad():
                teacher_outputs = teacher(batch) if teacher is not None else None
            loss, components = model.loss(batch, outputs, type_weights, teacher_outputs)
            (loss / config.gradient_accumulation_steps).backward()
            if batch_index % config.gradient_accumulation_steps == 0 or batch_index == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if teacher is not None:
                    with torch.no_grad():
                        for teacher_parameter, student_parameter in zip(teacher.parameters(), model.parameters()):
                            teacher_parameter.mul_(0.995).add_(student_parameter, alpha=0.005)
            totals.append(float(loss.detach().cpu()))
            component_rows.append(components)
        history.append(
            {
                "Seed": seed,
                "Epoch": epoch,
                "Composite Loss": float(np.mean(totals)),
                **pd.DataFrame(component_rows).mean().to_dict(),
            }
        )
        epoch_loss = float(np.mean(totals))
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= 4 and stale_epochs >= 2:
            break
    return model, pd.DataFrame(history), perf_counter() - started


def score_candidates(
    model: ACTNER,
    bundle: CandidateBundle,
    document_ids: set[str],
    batch_size: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    frame = bundle.frame
    indices = np.flatnonzero(frame.document_id.isin(document_ids).to_numpy(bool))
    dataset = CandidateDataset(bundle, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_candidates)
    device = next(model.parameters()).device
    rows: list[pd.DataFrame] = []
    gates: list[pd.DataFrame] = []
    model.eval()
    started = perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            outputs = model(batch)
            probability = torch.sigmoid(outputs["span_logits"]).detach().cpu().numpy()
            type_probability = outputs["type_logits"].softmax(-1).detach().cpu().numpy()
            selected_indices = raw_batch["indices"].numpy()
            part = frame.iloc[selected_indices].copy()
            part["candidate_index"] = selected_indices
            part["span_probability"] = probability
            part["predicted_type"] = [ID_TO_TYPE[index] for index in type_probability.argmax(axis=1)]
            part["type_probability"] = type_probability.max(axis=1)
            part["entity_probability"] = part.span_probability * part.type_probability
            part["type_vector"] = list(type_probability.astype(np.float32))
            transformer_type = raw_batch["transformer_type"].numpy()
            crf_type = raw_batch["crf_type"].numpy()
            part["transformer_predicted_type"] = [ID_TO_TYPE[index] for index in transformer_type.argmax(axis=1)]
            part["crf_predicted_type"] = [ID_TO_TYPE[index] for index in crf_type.argmax(axis=1)]
            part["transformer_type_probability"] = transformer_type.max(axis=1)
            part["crf_type_probability"] = crf_type.max(axis=1)
            for index, expert in enumerate(("Context", "Sequence", "Rare-Signal Prototype")):
                part[f"Router {expert}"] = outputs["router_weights"][:, index].detach().cpu().numpy()
            part["Router Entropy"] = outputs["router_entropy"].detach().cpu().numpy()
            rows.append(part)
            gate_frame = part[
                ["document_id", "sentence_id", "source", "seen_in_train", "multi_token", "expert_agreement"]
            ].copy()
            gate_frame["weak_signal"] = part.weak_signal.to_numpy(bool)
            gate_frame["candidate_source"] = part.candidate_source.to_numpy(str)
            gate_frame["predicted_type"] = part.predicted_type.to_numpy(str)
            for index, expert in enumerate(("Context", "Sequence", "Rare-Signal Prototype")):
                gate_frame[f"Router {expert}"] = outputs["router_weights"][:, index].detach().cpu().numpy()
            gate_frame["Router Entropy"] = outputs["router_entropy"].detach().cpu().numpy()
            gate_frame["Prototype Active"] = outputs["prototype_active"].detach().cpu().numpy()
            gate_frame["Early Exit"] = outputs["early_exit"].detach().cpu().numpy()
            gates.append(gate_frame)
    return pd.concat(rows, ignore_index=True), pd.concat(gates, ignore_index=True), perf_counter() - started


def decode_candidate_scores(
    scores: pd.DataFrame,
    sentences: pd.DataFrame,
    config: ACTNERConfig,
    split: str,
    seed: int,
) -> pd.DataFrame:
    candidate_mask = _candidate_mask(scores, config)
    eligible = scores.loc[candidate_mask].copy()
    crf_anchor = eligible.origin_crf.astype(bool) & config.use_crf_expert
    transformer_anchor = eligible.origin_transformer.astype(bool) & config.use_transformer_expert
    transformer_only = transformer_anchor & ~crf_anchor
    rare_recovery = (
        transformer_only
        & eligible.transformer_predicted_type.eq("drug_n")
        & eligible.weak_signal.astype(bool)
        & config.use_prototype_expert
        & config.dynamic_router
        & config.weak_signal_recovery
    )
    transformer_threshold = np.where(
        rare_recovery,
        config.rare_signal_anchor_threshold,
        config.transformer_anchor_threshold,
    )
    accepted_transformer_anchor = transformer_only & eligible.transformer_anchor_score.ge(
        transformer_threshold
    )
    residual_candidate = ~crf_anchor & ~transformer_anchor
    accepted_residual = residual_candidate & eligible.span_probability.ge(config.boundary_threshold)
    selected = eligible.loc[crf_anchor | accepted_transformer_anchor | accepted_residual].copy()
    selected["anchor_priority"] = np.select(
        [selected.origin_crf.astype(bool), selected.origin_transformer.astype(bool)],
        [3, 2],
        default=1,
    )
    selected["decision_type"] = np.where(
        selected.origin_crf.astype(bool),
        selected.crf_predicted_type,
        np.where(
            selected.origin_transformer.astype(bool),
            selected.transformer_predicted_type,
            selected.predicted_type,
        ),
    )
    selected["decision_boundary_probability"] = np.where(
        selected.origin_crf.astype(bool),
        selected.crf_anchor_score,
        np.where(
            selected.origin_transformer.astype(bool),
            selected.transformer_anchor_score,
            selected.span_probability,
        ),
    )
    selected["decision_type_probability"] = np.where(
        selected.origin_crf.astype(bool),
        selected.crf_type_probability,
        np.where(
            selected.origin_transformer.astype(bool),
            selected.transformer_type_probability,
            selected.type_probability,
        ),
    )
    accepted: list[dict] = []
    for _, group in selected.groupby("sentence_id", sort=False):
        occupied: list[tuple[int, int]] = []
        order = group.sort_values(
            ["anchor_priority", "decision_boundary_probability", "expert_agreement", "decision_type_probability", "token_end"],
            ascending=[False, False, False, False, True],
        )
        for row in order.itertuples(index=False):
            if any(not (int(row.token_end) <= start or int(row.token_start) >= end) for start, end in occupied):
                continue
            occupied.append((int(row.token_start), int(row.token_end)))
            accepted.append(
                {
                    "document_id": row.document_id,
                    "sentence_id": row.sentence_id,
                    "char_start": int(row.char_start),
                    "char_end": int(row.char_end),
                    "token_start": int(row.token_start),
                    "token_end": int(row.token_end),
                    "predicted_type": row.decision_type,
                    "boundary_probability": float(row.decision_boundary_probability),
                    "type_probability": float(row.decision_type_probability),
                }
            )
    predictions = finalize_prediction_frame(accepted, sentences, "ACT-NER", split, seed)
    if not predictions.empty:
        predictions["raw_probability"] = (
            predictions.boundary_probability * predictions.type_probability
        )
        predictions["calibrated_confidence"] = predictions.raw_probability
    return predictions


def make_document_folds(
    sentences: pd.DataFrame,
    entities: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = gold_for_metrics(entities)
    document_profile = (
        sentences[["document_id", "source"]]
        .drop_duplicates("document_id")
        .set_index("document_id")
    )
    counts = gold.groupby("document_id").size().rename("Entity Count")
    brand = gold.assign(flag=gold.entity_type.eq("brand")).groupby("document_id").flag.any().rename("Has Brand")
    drug_n = gold.assign(flag=gold.entity_type.eq("drug_n")).groupby("document_id").flag.any().rename("Has drug_n")
    profile = document_profile.join([counts, brand, drug_n]).fillna(
        {"Entity Count": 0, "Has Brand": False, "Has drug_n": False}
    )
    profile["Density Bin"] = pd.qcut(
        profile["Entity Count"].rank(method="first"), 4, labels=False, duplicates="drop"
    ).astype(int)
    candidates = [
        profile.source.astype(str)
        + "|b"
        + profile["Has Brand"].astype(int).astype(str)
        + "|n"
        + profile["Has drug_n"].astype(int).astype(str)
        + "|q"
        + profile["Density Bin"].astype(str),
        profile.source.astype(str)
        + "|b"
        + profile["Has Brand"].astype(int).astype(str)
        + "|n"
        + profile["Has drug_n"].astype(int).astype(str),
        profile.source.astype(str),
    ]
    strata = next(values for values in candidates if values.value_counts().min() >= n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_map: dict[str, int] = {}
    documents = profile.index.to_numpy()
    for fold, (_, hold_indices) in enumerate(splitter.split(documents, strata.to_numpy()), start=1):
        fold_map.update({document: fold for document in documents[hold_indices]})
    assignments = profile.reset_index()
    assignments["Fold"] = assignments.document_id.map(fold_map).astype(int)
    balance = (
        assignments.groupby(["Fold", "source"])
        .agg(
            Documents=("document_id", "nunique"),
            Documents_with_Brand=("Has Brand", "sum"),
            Documents_with_drug_n=("Has drug_n", "sum"),
            Median_Entity_Count=("Entity Count", "median"),
        )
        .reset_index()
        .rename(columns={"source": "Source"})
    )
    return assignments, balance


def _distribution_shift_statistics(reference: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(17)
    reference = reference[rng.choice(len(reference), min(400, len(reference)), replace=False)]
    target = target[rng.choice(len(target), min(400, len(target)), replace=False)]
    projection = rng.normal(0.0, 1.0 / math.sqrt(32), size=(reference.shape[1], 32))
    reference = reference @ projection
    target = target @ projection
    variance = reference.var(axis=0) + 1e-4
    mahalanobis = float(np.sqrt(np.mean((target.mean(0) - reference.mean(0)) ** 2 / variance)))
    pooled = np.vstack([reference, target])
    pair_sample = pooled[rng.choice(len(pooled), min(300, len(pooled)), replace=False)]
    squared = ((pair_sample[:, None, :] - pair_sample[None, :, :]) ** 2).sum(-1)
    bandwidth = float(np.median(squared[squared > 0])) if np.any(squared > 0) else 1.0
    kernel_rr = np.exp(-((reference[:, None, :] - reference[None, :, :]) ** 2).sum(-1) / (2 * bandwidth))
    kernel_tt = np.exp(-((target[:, None, :] - target[None, :, :]) ** 2).sum(-1) / (2 * bandwidth))
    kernel_rt = np.exp(-((reference[:, None, :] - target[None, :, :]) ** 2).sum(-1) / (2 * bandwidth))
    mmd = float(max(0.0, kernel_rr.mean() + kernel_tt.mean() - 2 * kernel_rt.mean()))
    rr = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=-1).mean()
    tt = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=-1).mean()
    rt = np.linalg.norm(reference[:, None, :] - target[None, :, :], axis=-1).mean()
    energy = float(max(0.0, 2 * rt - rr - tt))
    return mahalanobis, mmd, energy


def shift_detector(
    bundle: CandidateBundle,
    fold_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Detect representation shift without using any entity label from the target subset."""
    frame = bundle.frame
    embeddings = bundle.context.numpy()
    comparisons: list[tuple[str, np.ndarray, np.ndarray]] = []
    for fold in sorted(fold_assignments.Fold.unique()):
        target_documents = set(fold_assignments.loc[fold_assignments.Fold.eq(fold), "document_id"])
        target = frame.document_id.isin(target_documents).to_numpy(bool)
        comparisons.append((f"GroupCV fold {fold}", ~target, target))
    drugbank = frame.source.eq("DrugBank").to_numpy(bool)
    medline = frame.source.eq("MedLine").to_numpy(bool)
    comparisons.append(("DrugBank → MedLine", drugbank, medline))
    rows = []
    for comparison, reference_mask, target_mask in comparisons:
        mahalanobis, mmd, energy = _distribution_shift_statistics(
            embeddings[reference_mask], embeddings[target_mask]
        )
        reference_disagreement = frame.loc[reference_mask, "expert_disagreement"].mean()
        target_disagreement = frame.loc[target_mask, "expert_disagreement"].mean()
        confirmed = bool(
            mahalanobis > 0.50
            and (mmd > 0.010 or energy > 0.10)
            and target_disagreement - reference_disagreement > 0.01
        )
        rows.append(
            {
                "Comparison": comparison,
                "Reference N": int(reference_mask.sum()),
                "Target N": int(target_mask.sum()),
                "Mahalanobis Distance": mahalanobis,
                "MMD": mmd,
                "Energy Distance": energy,
                "Disagreement Change": target_disagreement - reference_disagreement,
                "Shift Detected": confirmed,
                "Adaptation Status": "Eligible for isolated adapter update" if confirmed else "Frozen ACT-NER retained",
                "Included in Primary Evaluation": False,
            }
        )
    return pd.DataFrame(rows)


def continual_adaptation_protocol() -> pd.DataFrame:
    """Precommitted safe-update contract; execution requires a genuinely new unlabeled domain."""
    return pd.DataFrame(
        [
            {
                "Trigger": "Mahalanobis + MMD/Energy + disagreement detector",
                "Frozen Parameters": "Base biomedical Transformer and CRF expert",
                "Trainable Parameters": "Low-rank context adapters, prototypes, sparse router",
                "Teacher": "EMA, decay 0.995",
                "Pseudo-label Rule": "Calibrated entity confidence ≥ 0.95",
                "Replay Buffer": "Source-balanced representative Train candidates",
                "Consistency": "Safe context-feature perturbation",
                "Rollback Rule": "Restore frozen snapshot if calibration or validation proxy degrades",
                "Primary Test Mixing": "Prohibited",
            }
        ]
    )


def _gate_diagnostics(gates: pd.DataFrame, seed: int, configuration: str) -> pd.DataFrame:
    rows: list[dict] = []
    router_columns = ["Router Context", "Router Sequence", "Router Rare-Signal Prototype"]
    groupings = {
        "Overall": pd.Series("All", index=gates.index),
        "Source": gates.source.astype(str),
        "Seen status": np.where(gates.seen_in_train, "Seen", "Unseen"),
        "Span length": np.where(gates.multi_token, "Multi-token", "Single-token"),
        "Expert relation": np.where(gates.expert_agreement, "Agreement", "Disagreement"),
        "Weak signal": np.where(gates.weak_signal, "Weak", "Routine"),
        "Predicted type": gates.predicted_type.astype(str),
    }
    for grouping, labels in groupings.items():
        temporary = gates[router_columns + ["Router Entropy", "Early Exit", "Prototype Active"]].copy()
        temporary["Group"] = np.asarray(labels)
        for group, part in temporary.groupby("Group"):
            for component, column in enumerate(router_columns):
                rows.append(
                    {
                        "Configuration": configuration,
                        "Seed": seed,
                        "Router": "Sparse top-2",
                        "Grouping": grouping,
                        "Group": group,
                        "Expert": column.removeprefix("Router "),
                        "Component": component,
                        "N": len(part),
                        "Mean Weight": part[column].mean(),
                        "SD Weight": part[column].std(ddof=0),
                        "Mean Entropy": part["Router Entropy"].mean(),
                        "Early-exit Rate": part["Early Exit"].mean(),
                        "Prototype Activation Rate": part["Prototype Active"].mean(),
                    }
                )
    return pd.DataFrame(rows)


def _fast_evaluate_predictions(
    gold_entities: pd.DataFrame,
    predictions: pd.DataFrame,
    model: str,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = gold_for_metrics(gold_entities)
    boundary_gold = set(zip(gold.sentence_id, gold.char_start, gold.char_end))
    boundary_pred = set(zip(predictions.sentence_id, predictions.char_start, predictions.char_end))
    typed_gold = set(zip(gold.sentence_id, gold.char_start, gold.char_end, gold.entity_type))
    typed_pred = set(zip(predictions.sentence_id, predictions.char_start, predictions.char_end, predictions.predicted_type))
    boundary_tp = len(boundary_gold & boundary_pred)
    typed_tp = len(typed_gold & typed_pred)
    safe = lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0
    boundary_precision = safe(boundary_tp, len(boundary_pred))
    boundary_recall = safe(boundary_tp, len(boundary_gold))
    typed_precision = safe(typed_tp, len(typed_pred))
    typed_recall = safe(typed_tp, len(typed_gold))
    boundary_f1 = safe(2 * boundary_precision * boundary_recall, boundary_precision + boundary_recall)
    typed_f1 = safe(2 * typed_precision * typed_recall, typed_precision + typed_recall)
    class_rows = []
    for entity_type in ENTITY_TYPES:
        class_gold = {value for value in typed_gold if value[3] == entity_type}
        class_pred = {value for value in typed_pred if value[3] == entity_type}
        class_tp = len(class_gold & class_pred)
        precision = safe(class_tp, len(class_pred))
        recall = safe(class_tp, len(class_gold))
        class_rows.append(
            {
                "Model": model,
                "Split": split,
                "N": len(class_gold),
                "Entity Type": entity_type,
                "Precision": precision,
                "Recall": recall,
                "F1": safe(2 * precision * recall, precision + recall),
            }
        )
    classwise = pd.DataFrame(class_rows)
    metrics = pd.DataFrame(
        [
            {
                "Model": model,
                "Split": split,
                "N": len(gold),
                "Strict Entity Precision": typed_precision,
                "Strict Entity Recall": typed_recall,
                "Strict Boundary F1": boundary_f1,
                "Strict Typed Entity F1": typed_f1,
                "Macro-F1": classwise.F1.mean(),
                "False Positive Rate": safe(len(typed_pred - typed_gold), len(typed_pred)),
                "False Negative Rate": safe(len(typed_gold - typed_pred), len(typed_gold)),
                "Boundary Error Rate": 1.0 - boundary_f1,
            }
        ]
    )
    return metrics, classwise


def _fast_subgroup_metrics(
    predictions: pd.DataFrame,
    entities: pd.DataFrame,
    sentences: pd.DataFrame,
    train_entities: pd.DataFrame,
    model: str,
    split: str,
) -> pd.DataFrame:
    gold = gold_for_metrics(entities).copy()
    gold["normalized_surface"] = gold.entity_text.map(normalize_surface)
    seen_surfaces = set(train_entities.query("not discontinuous").entity_text.map(normalize_surface))
    sentence_source = sentences.set_index("sentence_id").source
    groups = {
        "DrugBank": set(sentence_source[sentence_source.eq("DrugBank")].index),
        "MedLine": set(sentence_source[sentence_source.eq("MedLine")].index),
        "Unseen surface": set(gold.loc[~gold.normalized_surface.isin(seen_surfaces), "sentence_id"]),
        "Multi-token": set(gold.loc[gold.entity_text.str.contains(r"\s|[-/]", regex=True), "sentence_id"]),
    }
    rows = []
    for label, sentence_ids in groups.items():
        subset_gold = gold[gold.sentence_id.isin(sentence_ids)]
        subset_prediction = predictions[predictions.sentence_id.isin(sentence_ids)]
        metric, _ = _fast_evaluate_predictions(subset_gold, subset_prediction, model, split)
        rows.append(
            {
                "Model": model,
                "Split": split,
                "N": len(subset_gold),
                "Subgroup": label,
                "Strict Boundary F1": metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metric.iloc[0]["Strict Typed Entity F1"],
            }
        )
    return pd.DataFrame(rows)


def cross_validate_configuration(
    bundle: CandidateBundle,
    sentences: pd.DataFrame,
    entities: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    config: ACTNERConfig,
    seeds: Sequence[int],
    configuration_name: str,
    max_epochs: int,
    progress: NotebookLiveProgress | None = None,
    progress_phase: str = "GroupCV",
) -> dict:
    all_documents = set(fold_assignments.document_id)
    seed_metrics: list[pd.DataFrame] = []
    seed_classwise: list[pd.DataFrame] = []
    seed_predictions: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []
    gate_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    elapsed_total = 0.0
    inference_total = 0.0
    parameter_count = 0
    for seed in seeds:
        fold_predictions: list[pd.DataFrame] = []
        fold_gates: list[pd.DataFrame] = []
        for fold in sorted(fold_assignments.Fold.unique()):
            hold_documents = set(fold_assignments.loc[fold_assignments.Fold.eq(fold), "document_id"])
            train_documents = all_documents - hold_documents
            model, history, elapsed = train_act_ner_model(
                bundle,
                train_documents,
                config,
                seed + int(fold) * 1000,
                max_epochs=max_epochs,
            )
            scores, gates, inference_elapsed = score_candidates(model, bundle, hold_documents)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            scores["Seed"] = seed
            scores["Fold"] = int(fold)
            scores["Configuration"] = configuration_name
            predictions = decode_candidate_scores(
                scores,
                sentences,
                config,
                "Validation cross-fit",
                seed,
            )
            predictions["Seed"] = seed
            predictions["Fold"] = int(fold)
            fold_predictions.append(predictions)
            gates["Fold"] = int(fold)
            fold_gates.append(gates)
            history["Fold"] = int(fold)
            history["Configuration"] = configuration_name
            history_parts.append(history)
            score_parts.append(scores)
            elapsed_total += elapsed + inference_elapsed
            inference_total += inference_elapsed
            if progress is not None:
                progress.advance(progress_phase, configuration_name, seed, int(fold))
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        predictions = pd.concat(fold_predictions, ignore_index=True)
        assert predictions.entity_id.is_unique
        metric, classwise = _fast_evaluate_predictions(
            entities,
            predictions,
            "ACT-NER",
            "Validation cross-fit",
        )
        metric["Configuration"] = configuration_name
        metric["Seed"] = seed
        raw_calibration = calibration_metrics(
            predictions,
            entities,
            "ACT-NER",
            "Validation cross-fit",
        ).iloc[0]
        metric["Raw ECE"] = raw_calibration["ECE"]
        metric["Raw Brier Score"] = raw_calibration["Brier Score"]
        classwise["Configuration"] = configuration_name
        classwise["Seed"] = seed
        seed_metrics.append(metric)
        seed_classwise.append(classwise)
        seed_predictions.append(predictions.assign(Configuration=configuration_name))
        gate_frame = pd.concat(fold_gates, ignore_index=True)
        gate_parts.append(_gate_diagnostics(gate_frame, seed, configuration_name))
    metrics = pd.concat(seed_metrics, ignore_index=True)
    return {
        "metrics": metrics,
        "classwise": pd.concat(seed_classwise, ignore_index=True),
        "predictions": pd.concat(seed_predictions, ignore_index=True),
        "candidate_scores": pd.concat(score_parts, ignore_index=True),
        "gates": pd.concat(gate_parts, ignore_index=True),
        "history": pd.concat(history_parts, ignore_index=True),
        "elapsed": elapsed_total,
        "inference_elapsed": inference_total,
        "parameter_count": parameter_count,
    }


def summarize_repeated_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "Strict Entity Precision",
        "Strict Entity Recall",
        "Strict Boundary F1",
        "Strict Typed Entity F1",
        "Macro-F1",
        "Raw ECE",
        "Raw Brier Score",
    ]
    rows: list[dict] = []
    for configuration, group in metrics.groupby("Configuration", sort=False):
        row = {"Configuration": configuration, "Seeds": group.Seed.nunique(), "N": int(group.N.iloc[0])}
        for column in value_columns:
            row[f"{column} Mean"] = group[column].mean()
            row[f"{column} SD"] = group[column].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


class VectorTypeCalibrator:
    def __init__(self):
        self.model = LogisticRegression(max_iter=800, class_weight="balanced")

    def fit(self, vectors: np.ndarray, labels: np.ndarray) -> "VectorTypeCalibrator":
        self.model.fit(np.log(np.clip(vectors, EPSILON, 1.0)), labels)
        return self

    def predict_for_declared_type(self, vectors: np.ndarray, type_ids: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict_proba(np.log(np.clip(vectors, EPSILON, 1.0)))
        class_positions = {int(label): index for index, label in enumerate(self.model.classes_)}
        return np.asarray(
            [calibrated[row, class_positions.get(int(label), 0)] for row, label in enumerate(type_ids)],
            dtype=float,
        )


@dataclass
class EntityCalibrationBundle:
    method: str
    span_calibrator: object | None
    type_calibrator: object | None
    entity_calibrator: object | None

    def apply_components(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        span = frame.span_probability.to_numpy(float)
        type_probability = frame.type_probability.to_numpy(float)
        entity_probability = np.clip(span * type_probability, 0, 1)
        if self.method == "Raw":
            return span, type_probability, entity_probability
        if self.method == "Vector Scaling":
            type_vectors = np.stack(frame.type_vector)
            type_ids = np.asarray([TYPE_TO_ID[value] for value in frame.predicted_type], dtype=int)
            calibrated_type = self.type_calibrator.predict_for_declared_type(type_vectors, type_ids)
        else:
            calibrated_type = self.type_calibrator.predict(type_probability)
        calibrated_span = self.span_calibrator.predict(span)
        calibrated_entity = self.entity_calibrator.predict(entity_probability)
        return (
            np.clip(calibrated_span, 0, 1),
            np.clip(calibrated_type, 0, 1),
            np.clip(calibrated_entity, 0, 1),
        )

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        return self.apply_components(frame)[2]


def _fit_calibration_method(frame: pd.DataFrame, method: str) -> EntityCalibrationBundle:
    span = frame.span_probability.to_numpy(float)
    span_outcome = frame.span_correct.to_numpy(int)
    type_probability = frame.type_probability.to_numpy(float)
    exact = frame.span_correct.to_numpy(bool)
    type_outcome = frame.type_correct.to_numpy(int)
    entity_probability = np.clip(span * type_probability, 0, 1)
    typed_outcome = frame.typed_correct.to_numpy(int)
    if method == "Raw":
        return EntityCalibrationBundle(method, None, None, None)
    if method == "Temperature Scaling":
        span_calibrator = TemperatureCalibrator().fit(span, span_outcome)
        type_calibrator = TemperatureCalibrator().fit(type_probability[exact], type_outcome[exact])
        entity_calibrator = TemperatureCalibrator().fit(entity_probability, typed_outcome)
    elif method == "Platt Scaling":
        span_calibrator = PlattCalibrator().fit(span, span_outcome)
        type_calibrator = PlattCalibrator().fit(type_probability[exact], type_outcome[exact])
        entity_calibrator = PlattCalibrator().fit(entity_probability, typed_outcome)
    elif method == "Isotonic":
        span_calibrator = IsotonicRegression(out_of_bounds="clip").fit(span, span_outcome)
        type_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            type_probability[exact], type_outcome[exact]
        )
        entity_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            entity_probability, typed_outcome
        )
    elif method == "Vector Scaling":
        span_calibrator = PlattCalibrator().fit(span, span_outcome)
        type_calibrator = VectorTypeCalibrator().fit(
            np.stack(frame.loc[exact, "type_vector"]),
            frame.loc[exact, "gold_type_id"].to_numpy(int),
        )
        entity_calibrator = PlattCalibrator().fit(entity_probability, typed_outcome)
    else:
        raise ValueError(method)
    return EntityCalibrationBundle(method, span_calibrator, type_calibrator, entity_calibrator)


def calibration_evidence(
    accepted_scores: pd.DataFrame,
    n_folds: int = 8,
) -> tuple[EntityCalibrationBundle, pd.DataFrame, pd.DataFrame]:
    methods = ["Raw", "Temperature Scaling", "Vector Scaling", "Platt Scaling"]
    if len(accepted_scores) >= 1000:
        methods.append("Isotonic")
    documents = accepted_scores.document_id.drop_duplicates().sort_values().to_numpy()
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=17)
    document_source = (
        accepted_scores[["document_id", "source"]].drop_duplicates("document_id").set_index("document_id").reindex(documents).source
    )
    cross_fitted = {method: np.zeros(len(accepted_scores), dtype=float) for method in methods}
    cross_fitted_span = {method: np.zeros(len(accepted_scores), dtype=float) for method in methods}
    cross_fitted_type = {method: np.zeros(len(accepted_scores), dtype=float) for method in methods}
    for fit_documents_index, hold_documents_index in splitter.split(documents, document_source):
        fit_documents = set(documents[fit_documents_index])
        hold_documents = set(documents[hold_documents_index])
        fit = accepted_scores[accepted_scores.document_id.isin(fit_documents)]
        hold_mask = accepted_scores.document_id.isin(hold_documents).to_numpy(bool)
        hold = accepted_scores.loc[hold_mask]
        for method in methods:
            calibrator = _fit_calibration_method(fit, method)
            span_confidence, type_confidence, entity_confidence = calibrator.apply_components(hold)
            cross_fitted_span[method][hold_mask] = span_confidence
            cross_fitted_type[method][hold_mask] = type_confidence
            cross_fitted[method][hold_mask] = entity_confidence
    outcomes = accepted_scores.typed_correct.to_numpy(int)
    span_outcomes = accepted_scores.span_correct.to_numpy(int)
    exact = accepted_scores.span_correct.to_numpy(bool)
    type_outcomes = accepted_scores.type_correct.to_numpy(int)
    comparison = pd.DataFrame(
        [
            {
                "Method": method,
                "N": len(accepted_scores),
                "Document Cross-fitting Folds": n_folds,
                "Cross-fitted Span ECE": expected_calibration_error(
                    cross_fitted_span[method], span_outcomes
                ),
                "Cross-fitted Type ECE": expected_calibration_error(
                    cross_fitted_type[method][exact], type_outcomes[exact]
                ),
                "Cross-fitted ECE": expected_calibration_error(confidence, outcomes),
                "Cross-fitted Brier Score": float(np.mean((confidence - outcomes) ** 2)),
            }
            for method, confidence in cross_fitted.items()
        ]
    ).sort_values(["Cross-fitted ECE", "Cross-fitted Brier Score", "Method"])
    selected_method = str(comparison.iloc[0].Method)
    calibrator = _fit_calibration_method(accepted_scores, selected_method)
    calibrated_rows = accepted_scores.copy()
    calibrated_rows["cross_fitted_confidence"] = cross_fitted[selected_method]
    return calibrator, comparison, calibrated_rows


def accepted_candidate_rows(
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
    gold_entities: pd.DataFrame,
) -> pd.DataFrame:
    prediction_keys = set(zip(predictions.sentence_id, predictions.char_start, predictions.char_end))
    accepted = scores[
        [
            (sentence_id, char_start, char_end) in prediction_keys
            for sentence_id, char_start, char_end in zip(scores.sentence_id, scores.char_start, scores.char_end)
        ]
    ].copy()
    prediction_lookup = predictions.set_index(["sentence_id", "char_start", "char_end"])
    ordered_prediction = prediction_lookup.loc[
        list(zip(accepted.sentence_id, accepted.char_start, accepted.char_end))
    ].reset_index()
    accepted["predicted_type"] = ordered_prediction.predicted_type.to_numpy(str)
    accepted["span_probability"] = ordered_prediction.boundary_probability.to_numpy(float)
    accepted["type_probability"] = ordered_prediction.type_probability.to_numpy(float)
    decision_vectors = np.full((len(accepted), len(ENTITY_TYPES)), 0.0, dtype=np.float32)
    for row_index, (entity_type, probability) in enumerate(
        zip(accepted.predicted_type, accepted.type_probability)
    ):
        probability = float(np.clip(probability, 0.25, 1.0))
        decision_vectors[row_index] = (1.0 - probability) / (len(ENTITY_TYPES) - 1)
        decision_vectors[row_index, TYPE_TO_ID[entity_type]] = probability
    accepted["type_vector"] = list(decision_vectors)
    gold = gold_for_metrics(gold_entities)
    gold_map = {
        (row.sentence_id, int(row.char_start), int(row.char_end)): TYPE_TO_ID[row.entity_type]
        for row in gold.itertuples(index=False)
    }
    keys = list(zip(accepted.sentence_id, accepted.char_start, accepted.char_end))
    accepted["span_correct"] = [int(key in gold_map) for key in keys]
    accepted["gold_type_id"] = [gold_map.get(key, -1) for key in keys]
    accepted["type_correct"] = [
        int(gold_type >= 0 and TYPE_TO_ID[predicted] == gold_type)
        for predicted, gold_type in zip(accepted.predicted_type, accepted.gold_type_id)
    ]
    accepted["typed_correct"] = accepted.type_correct
    return accepted


def apply_calibration_and_policy(
    predictions: pd.DataFrame,
    accepted_scores: pd.DataFrame,
    calibrator: EntityCalibrationBundle,
    policy: dict,
) -> pd.DataFrame:
    output = predictions.copy()
    score_map = accepted_scores.set_index(["sentence_id", "char_start", "char_end"])
    ordered = score_map.loc[
        list(zip(output.sentence_id, output.char_start, output.char_end))
    ].reset_index()
    confidence = calibrator.apply(ordered)
    output["calibrated_confidence"] = confidence
    output["confidence_category"] = np.select(
        [confidence < policy["low_threshold"], confidence >= policy["high_threshold"]],
        ["low", "high"],
        default="medium",
    )
    output["review_required"] = output.confidence_category.eq("low")
    return output


def select_confidence_policy(calibrated_rows: pd.DataFrame) -> dict:
    confidence = calibrated_rows.cross_fitted_confidence.to_numpy(float)
    return {
        "low_threshold": float(np.quantile(confidence, 0.20)),
        "high_threshold": float(np.quantile(confidence, 0.70)),
        "selection_split": "Repeated document-level GroupCV",
        "prediction_invariant": True,
    }


def risk_coverage_table(calibrated_rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    ordered = calibrated_rows.sort_values("cross_fitted_confidence", ascending=False)
    errors = 1 - ordered.typed_correct.to_numpy(float)
    coverage_grid = np.linspace(0.10, 1.00, 10)
    rows = []
    for coverage in coverage_grid:
        count = max(1, int(math.ceil(len(ordered) * coverage)))
        rows.append(
            {
                "Coverage": coverage,
                "Accepted Predictions": count,
                "Risk": float(errors[:count].mean()),
            }
        )
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    aurc = float(np.mean(cumulative_risk))
    return pd.DataFrame(rows), aurc


def joint_configuration_space() -> list[tuple[str, ACTNERConfig]]:
    rows = [
        (0.10, 128, 0.970, 0.900, 0.700, "weighted_ce", "balanced_sampler", 2, 0.60, 0.75, True),
        (0.15, 192, 0.980, 0.925, 0.750, "effective_number", "balanced_sampler", 4, 0.72, 1.00, True),
        (0.20, 256, 0.990, 0.950, 0.800, "focal", "group_dro", 6, 0.82, 1.25, True),
        (0.25, 128, 0.985, 0.925, 0.650, "logit_adjusted", "none", 4, 0.72, 1.00, True),
        (0.10, 192, 0.975, 0.925, 0.675, "effective_number", "group_dro", 6, 0.60, 1.25, True),
        (0.15, 256, 0.990, 0.925, 0.700, "weighted_ce", "none", 2, 0.82, 0.75, True),
        (0.20, 128, 0.970, 0.925, 0.725, "logit_adjusted", "balanced_sampler", 4, 0.72, 1.00, True),
        (0.25, 192, 0.980, 0.925, 0.775, "focal", "group_dro", 2, 0.60, 1.25, True),
        (0.10, 256, 0.985, 0.925, 0.825, "effective_number", "none", 6, 0.82, 0.75, True),
        (0.15, 128, 0.975, 0.925, 0.850, "focal", "balanced_sampler", 4, 0.60, 1.00, True),
        (0.20, 192, 0.980, 0.875, 0.750, "weighted_ce", "group_dro", 2, 0.82, 1.25, True),
        (0.25, 256, 0.970, 0.950, 0.750, "logit_adjusted", "balanced_sampler", 6, 0.72, 0.75, True),
    ]
    configurations = []
    for index, (
        dropout,
        gate_hidden,
        boundary_threshold,
        transformer_anchor_threshold,
        rare_signal_anchor_threshold,
        type_loss,
        source_balancing,
        hard_negative_ratio,
        marginal_threshold,
        lambda_type,
        lexical,
    ) in enumerate(rows, start=1):
        config = ACTNERConfig(
            dropout=dropout,
            gate_hidden_size=gate_hidden,
            boundary_threshold=boundary_threshold,
            transformer_anchor_threshold=transformer_anchor_threshold,
            rare_signal_anchor_threshold=rare_signal_anchor_threshold,
            type_loss=type_loss,
            source_balancing=source_balancing,
            hard_negative_ratio=hard_negative_ratio,
            marginal_threshold=marginal_threshold,
            lambda_type=lambda_type,
            use_lexical_reliability=lexical,
        )
        configurations.append((f"Joint-{index:02d}", config))
    return configurations


def _configuration_table(configurations: Sequence[tuple[str, ACTNERConfig]]) -> pd.DataFrame:
    rows = []
    for name, config in configurations:
        rows.append(
            {
                "Configuration": name,
                "Dropout": config.dropout,
                "Router Hidden Size": config.gate_hidden_size,
                "Boundary Threshold": config.boundary_threshold,
                "Transformer Anchor Threshold": config.transformer_anchor_threshold,
                "Rare-signal Anchor Threshold": config.rare_signal_anchor_threshold,
                "Type Loss": config.type_loss,
                "Source Balancing": config.source_balancing,
                "Hard-negative Ratio": config.hard_negative_ratio,
                "Marginal Threshold": config.marginal_threshold,
                "Boundary/Type Ratio": 1.0 / config.lambda_type,
                "Lexical Reliability": config.use_lexical_reliability,
                "Transformer Unfreezing Depth": config.transformer_unfreezing_depth,
            }
        )
    return pd.DataFrame(rows)


def _ablation_configurations(config: ACTNERConfig) -> list[tuple[str, ACTNERConfig]]:
    return [
        ("Full ACT-NER", config),
        ("Without Transformer Expert", replace(config, use_transformer_expert=False)),
        ("Without CRF Expert", replace(config, use_crf_expert=False)),
        ("Without Rare-Signal Prototype Expert", replace(config, use_prototype_expert=False)),
        ("Without dynamic router", replace(config, dynamic_router=False, sparse_top2=False)),
        ("Static average instead of routing", replace(config, static_average=True, sparse_top2=False)),
        ("One common gate instead of sparse router", replace(config, shared_gate=True, sparse_top2=False)),
        ("Without weak-signal recovery", replace(config, weak_signal_recovery=False)),
        ("Without hard-negative mining", replace(config, hard_negative_mining=False)),
        ("Without source balancing", replace(config, source_balancing="none")),
        ("Without class balancing", replace(config, class_balanced_type_loss=False, type_loss="weighted_ce")),
        ("Without domain adapter", replace(config, domain_adapter=False)),
        ("Without continual adaptation", replace(config, continual_adaptation=False)),
        ("Without calibration", replace(config, calibration=False)),
        ("Without early exit", replace(config, early_exit=False)),
        ("Parameter-matched span classifier", replace(config, dynamic_router=False, sparse_top2=False, parameter_matched_control=True)),
    ]


def _seed_subgroups(
    result: dict,
    entities: pd.DataFrame,
    sentences: pd.DataFrame,
    train_entities: pd.DataFrame,
    configuration: str,
) -> pd.DataFrame:
    rows = []
    for seed, predictions in result["predictions"].groupby("Seed"):
        subgroup = _fast_subgroup_metrics(
            predictions,
            entities,
            sentences,
            train_entities,
            "ACT-NER",
            "Validation cross-fit",
        )
        subgroup["Configuration"] = configuration
        subgroup["Seed"] = int(seed)
        rows.append(subgroup)
    return pd.concat(rows, ignore_index=True)


def _ablation_summary(
    name: str,
    result: dict,
    subgroups: pd.DataFrame,
) -> dict:
    metrics = result["metrics"]
    classwise = result["classwise"]
    brand = classwise.query("`Entity Type` == 'brand'").F1
    drug_n = classwise.query("`Entity Type` == 'drug_n'").F1
    unseen = subgroups.query("Subgroup == 'Unseen surface'")["Strict Typed Entity F1"]
    medline = subgroups.query("Subgroup == 'MedLine'")["Strict Typed Entity F1"]
    multi_token = subgroups.query("Subgroup == 'Multi-token'")["Strict Typed Entity F1"]
    latency = 1000 * result["inference_elapsed"] / (metrics.Seed.nunique() * 5 * 1466)
    typed_values = metrics["Strict Typed Entity F1"].to_numpy(float)
    rng = np.random.default_rng(1701)
    bootstrap = np.asarray([
        rng.choice(typed_values, size=len(typed_values), replace=True).mean() for _ in range(2000)
    ])
    return {
        "Ablation": name,
        "Seeds": metrics.Seed.nunique(),
        "N": int(metrics.N.iloc[0]),
        "Boundary Precision Mean": np.nan,
        "Boundary Recall Mean": np.nan,
        "Boundary F1 Mean": metrics["Strict Boundary F1"].mean(),
        "Boundary F1 SD": metrics["Strict Boundary F1"].std(ddof=1),
        "Typed Precision Mean": metrics["Strict Entity Precision"].mean(),
        "Typed Recall Mean": metrics["Strict Entity Recall"].mean(),
        "Typed F1 Mean": metrics["Strict Typed Entity F1"].mean(),
        "Typed F1 SD": metrics["Strict Typed Entity F1"].std(ddof=1),
        "Macro-F1 Mean": metrics["Macro-F1"].mean(),
        "brand F1 Mean": brand.mean(),
        "drug_n F1 Mean": drug_n.mean(),
        "Unseen F1 Mean": unseen.mean(),
        "MedLine F1 Mean": medline.mean(),
        "Multi-token F1 Mean": multi_token.mean(),
        "Raw ECE Mean": metrics["Raw ECE"].mean(),
        "Latency per Sentence (ms)": latency,
        "Parameter Count": int(result["parameter_count"]),
        "Typed F1 CI Low": float(np.quantile(bootstrap, 0.025)),
        "Typed F1 CI High": float(np.quantile(bootstrap, 0.975)),
    }


def _boundary_precision_recall(gold_entities: pd.DataFrame, predictions: pd.DataFrame) -> tuple[float, float]:
    gold = gold_for_metrics(gold_entities)
    gold_set = set(zip(gold.sentence_id, gold.char_start, gold.char_end))
    prediction_set = set(zip(predictions.sentence_id, predictions.char_start, predictions.char_end))
    true_positive = len(gold_set & prediction_set)
    precision = true_positive / len(prediction_set) if prediction_set else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    return precision, recall


def _add_boundary_precision_recall(
    summary: pd.DataFrame,
    ablation_results: dict[str, dict],
    entities: pd.DataFrame,
) -> pd.DataFrame:
    output = summary.copy()
    for name, result in ablation_results.items():
        values = [
            _boundary_precision_recall(entities, predictions)
            for _, predictions in result["predictions"].groupby("Seed")
        ]
        output.loc[output.Ablation.eq(name), "Boundary Precision Mean"] = np.mean([value[0] for value in values])
        output.loc[output.Ablation.eq(name), "Boundary Recall Mean"] = np.mean([value[1] for value in values])
    return output


def _selection_order(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        [
            "Strict Typed Entity F1 Mean",
            "Strict Boundary F1 Mean",
            "Strict Entity Recall Mean",
            "Macro-F1 Mean",
            "Worst-domain F1 Mean",
            "brand F1 Mean",
            "drug_n F1 Mean",
            "Raw ECE Mean",
            "Latency per Sentence (ms)",
            "Parameter Count",
            "Configuration",
        ],
        ascending=[False, False, False, False, False, False, False, True, True, True, True],
    )


def _clear_old_proposed_artifacts(project_root: Path) -> None:
    proposed = project_root / "Models" / "Proposed"
    for path in sorted(proposed.glob("act_ner*.pt")):
        assert path.parent == proposed
        path.unlink()
    for name in ("confidence_calibrator.joblib", "confidence_policy.json"):
        path = proposed / name
        if path.exists():
            path.unlink()
    for directory in (
        project_root / "Predictions" / "OOF",
        project_root / "Predictions" / "Validation",
        project_root / "Predictions" / "Locked_Test",
    ):
        for name in ("act_ner.parquet", "act_ner_without_confidence_loss.parquet"):
            path = directory / name
            if path.exists():
                path.unlink()
    handoff = project_root / "Predictions" / "Locked_Test" / "drug_entities_for_relation_extraction.parquet"
    if handoff.exists():
        handoff.unlink()
    for pattern in ("05_*.csv", "06_*.csv", "figure_act_ner_*.csv", "figure_component_*.csv"):
        for path in sorted((project_root / "artifacts").glob(pattern)):
            assert path.parent == project_root / "artifacts"
            path.unlink()
    for directory in (project_root / "figures" / "PNG", project_root / "figures" / "PDF"):
        for path in sorted(directory.glob("act_ner_*")):
            assert path.parent == directory
            path.unlink()


def _save_act_ner_figures(
    project_root: Path,
    candidate_recall: pd.DataFrame,
    joint_summary: pd.DataFrame,
    ablation: pd.DataFrame,
    gates: pd.DataFrame,
    calibration: pd.DataFrame,
    risk_coverage: pd.DataFrame,
) -> list[str]:
    names: list[str] = []
    figure, axis = plt.subplots(figsize=(13.5, 8.0))
    axis.plot(
        candidate_recall["Candidate Source"],
        candidate_recall["Validation Boundary Recall"],
        marker="o",
        linewidth=2.5,
        markersize=10,
        color="#2F6B8A",
    )
    axis.set_title("High-Recall Candidate Fusion")
    axis.set_xlabel("Candidate generator")
    axis.set_ylabel("Validation boundary recall")
    axis.set_ylim(max(0.0, candidate_recall["Validation Boundary Recall"].min() - 0.08), 1.0)
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", color="#E7EAED", linewidth=0.8)
    figure.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.27)
    save_figure(project_root, figure, "act_ner_candidate_recall", candidate_recall)
    names.append("act_ner_candidate_recall")

    plot_joint = joint_summary.sort_values("Strict Typed Entity F1 Mean")
    figure, axis = plt.subplots(figsize=(15.5, max(8.0, 0.65 * len(plot_joint) + 3.0)))
    axis.errorbar(
        plot_joint["Strict Typed Entity F1 Mean"],
        np.arange(len(plot_joint)),
        xerr=plot_joint["Strict Typed Entity F1 SD"],
        fmt="o",
        color="#2F6B8A",
        ecolor="#C77C2B",
        capsize=6,
        markersize=9,
    )
    axis.set_yticks(np.arange(len(plot_joint)), plot_joint.Configuration)
    axis.set_title("Repeated GroupCV Joint Selection")
    axis.set_xlabel("Strict Typed Entity F1, mean ± SD")
    axis.set_ylabel("Joint configuration")
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.5f}"))
    spread = max(float(plot_joint["Strict Typed Entity F1 Mean"].max() - plot_joint["Strict Typed Entity F1 Mean"].min()), 2e-5)
    axis.set_xlim(
        float(plot_joint["Strict Typed Entity F1 Mean"].min()) - 0.35 * spread,
        float(plot_joint["Strict Typed Entity F1 Mean"].max()) + 0.35 * spread,
    )
    axis.grid(axis="x", color="#E7EAED", linewidth=0.8)
    figure.subplots_adjust(left=0.26, right=0.96, top=0.90, bottom=0.16)
    save_figure(project_root, figure, "act_ner_joint_selection", plot_joint)
    names.append("act_ner_joint_selection")

    plot_ablation = ablation.sort_values("Typed F1 Mean")
    figure, axis = plt.subplots(figsize=(16.0, max(10.0, 0.72 * len(plot_ablation) + 3.0)))
    bars = axis.barh(
        np.arange(len(plot_ablation)),
        plot_ablation["Typed F1 Mean"],
        color="#2F6B8A",
        edgecolor="#222222",
    )
    axis.errorbar(
        plot_ablation["Typed F1 Mean"],
        np.arange(len(plot_ablation)),
        xerr=plot_ablation["Typed F1 SD"],
        fmt="none",
        ecolor="#C77C2B",
        capsize=5,
        linewidth=2,
    )
    axis.set_yticks(np.arange(len(plot_ablation)), plot_ablation.Ablation)
    axis.set_title("ACT-NER Ablation Evidence")
    axis.set_xlabel("Repeated GroupCV Strict Typed Entity F1")
    axis.set_ylabel("One-component intervention")
    axis.grid(axis="x", color="#E7EAED", linewidth=0.8)
    figure.subplots_adjust(left=0.37, right=0.97, top=0.93, bottom=0.12)
    save_figure(project_root, figure, "act_ner_ablation", plot_ablation)
    names.append("act_ner_ablation")

    overall = gates.query("Grouping == 'Overall'").groupby("Expert").agg(
        Mean=("Mean Weight", "mean"),
        SD=("SD Weight", "mean"),
        Entropy=("Mean Entropy", "mean"),
        Early_exit=("Early-exit Rate", "mean"),
    ).reset_index()
    figure, axis = plt.subplots(figsize=(13.0, 8.5))
    positions = np.arange(len(overall))
    bars = axis.bar(
        positions,
        overall.Mean,
        yerr=overall.SD,
        capsize=7,
        color=["#2F6B8A", "#C77C2B", "#7A6FA8"],
        edgecolor="#222222",
    )
    upper = float((overall.Mean + overall.SD).max())
    for bar, value, deviation in zip(bars, overall.Mean, overall.SD):
        axis.text(bar.get_x() + bar.get_width() / 2, value + deviation + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=18)
    axis.set_xticks(positions, overall.Expert)
    axis.set_title("ACT-NER Sparse Router Allocation")
    axis.set_xlabel("Span-level expert")
    axis.set_ylabel("Mean routing weight ± SD")
    axis.set_ylim(0, min(1.0, upper + 0.18))
    axis.grid(axis="y", color="#E7EAED", linewidth=0.8)
    figure.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.22)
    save_figure(project_root, figure, "act_ner_router_analysis", overall)
    names.append("act_ner_router_analysis")

    figure, axis = plt.subplots(figsize=(13.5, 8.0))
    markers = ["o", "s", "^", "D", "P"]
    colors = ["#2F6B8A", "#C77C2B", "#7A6FA8", "#4F8A5B", "#B45A55"]
    offsets = {
        "Platt Scaling": (12, 18),
        "Vector Scaling": (12, -30),
        "Temperature Scaling": (12, 12),
        "Isotonic": (12, 12),
        "Raw": (-62, 14),
    }
    for position, (_, row) in enumerate(calibration.iterrows()):
        axis.scatter(
            [row["Cross-fitted ECE"]],
            [row["Cross-fitted Brier Score"]],
            s=170,
            marker=markers[position % len(markers)],
            color=colors[position % len(colors)],
            edgecolor="#222222",
            zorder=3,
        )
        axis.annotate(
            row["Method"],
            (row["Cross-fitted ECE"], row["Cross-fitted Brier Score"]),
            xytext=offsets.get(str(row["Method"]), (10, 10)),
            textcoords="offset points",
            fontsize=18,
        )
    axis.set_title("Post-training Calibration Selection")
    axis.set_xlabel("Cross-fitted ECE")
    axis.set_ylabel("Cross-fitted Brier score")
    axis.grid(color="#E7EAED", linewidth=0.8)
    figure.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.16)
    save_figure(project_root, figure, "act_ner_calibration_selection", calibration)
    names.append("act_ner_calibration_selection")

    figure, axis = plt.subplots(figsize=(12.5, 8.0))
    axis.plot(risk_coverage.Coverage, risk_coverage.Risk, marker="o", markersize=9, linewidth=2.5, color="#B45A55")
    axis.set_title("Risk–Coverage Profile")
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Strict typed risk")
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, max(0.25, risk_coverage.Risk.max() + 0.05))
    axis.grid(color="#E7EAED", linewidth=0.8)
    figure.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.16)
    save_figure(project_root, figure, "act_ner_risk_coverage", risk_coverage)
    names.append("act_ner_risk_coverage")
    return names


def _acceptance_gate(
    selected_summary: pd.Series,
    seed_metrics: pd.DataFrame,
    classwise: pd.DataFrame,
    subgroups: pd.DataFrame,
    gates: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
    project_root: Path,
) -> pd.DataFrame:
    comparator_metrics = pd.concat(
        [
            pd.read_csv(project_root / "artifacts" / "03_predictive_quality.csv"),
            pd.read_csv(project_root / "artifacts" / "04_predictive_quality.csv"),
        ],
        ignore_index=True,
    ).query("Model in ['Linear-chain CRF', 'Biomedical Transformer']")
    comparator_classwise = pd.concat(
        [
            pd.read_csv(project_root / "artifacts" / "03_classwise_quality.csv"),
            pd.read_csv(project_root / "artifacts" / "04_classwise_quality.csv"),
        ],
        ignore_index=True,
    ).query("Model in ['Linear-chain CRF', 'Biomedical Transformer']")
    prior = json.loads((project_root / "src" / "pre_refinement_reference.json").read_text())
    class_means = classwise.groupby("Entity Type").F1.mean()
    subgroup_means = subgroups.groupby("Subgroup")["Strict Typed Entity F1"].mean()
    overall_gates = gates.query("Grouping == 'Overall'")
    maximum_gate_sd = float(overall_gates["SD Weight"].max())
    best_typed = float(comparator_metrics["Strict Typed Entity F1"].max())
    best_boundary = float(comparator_metrics["Strict Boundary F1"].max())
    best_recall = float(comparator_metrics["Strict Entity Recall"].max())
    selected_row = calibration_comparison.loc[calibration_comparison.Selected].iloc[0] if "Selected" in calibration_comparison else calibration_comparison.sort_values(["Cross-fitted ECE", "Cross-fitted Brier Score"]).iloc[0]
    selected_ece = float(selected_row["Cross-fitted ECE"])
    train_sentences, train_entities, _ = load_processed(project_root, "train")
    validation_sentences, validation_entities, _ = load_processed(project_root, "validation")
    comparator_subgroup_rows = []
    for model, filename in (
        ("Linear-chain CRF", "linear_chain_crf.parquet"),
        ("Biomedical Transformer", "biomedical_transformer.parquet"),
    ):
        comparator_subgroup_rows.append(
            _fast_subgroup_metrics(
                pd.read_parquet(project_root / "Predictions" / "Validation" / filename),
                validation_entities,
                validation_sentences,
                train_entities,
                model,
                "Validation",
            )
        )
    comparator_subgroups = pd.concat(comparator_subgroup_rows, ignore_index=True)
    medline_floor = float(comparator_subgroups.query("Subgroup == 'MedLine'")["Strict Typed Entity F1"].max())
    unseen_floor = float(comparator_subgroups.query("Subgroup == 'Unseen surface'")["Strict Typed Entity F1"].max())
    best_brand = float(comparator_classwise.query("`Entity Type` == 'brand'").F1.max())
    best_drug_n = float(comparator_classwise.query("`Entity Type` == 'drug_n'").F1.max())
    conditions = [
        ("Mean Typed F1 exceeds CRF by at least 0.005", selected_summary["Strict Typed Entity F1 Mean"], float(comparator_metrics.query("Model == 'Linear-chain CRF'")["Strict Typed Entity F1"].iloc[0]) + 0.005, ">="),
        ("Mean Typed F1 exceeds Transformer by at least 0.005", selected_summary["Strict Typed Entity F1 Mean"], float(comparator_metrics.query("Model == 'Biomedical Transformer'")["Strict Typed Entity F1"].iloc[0]) + 0.005, ">="),
        ("Boundary F1 within 0.002 of best comparator", selected_summary["Strict Boundary F1 Mean"], best_boundary - 0.002, ">="),
        ("Typed recall within 0.010 of best comparator", selected_summary["Strict Entity Recall Mean"], best_recall - 0.010, ">="),
        ("Macro-F1 exceeds both comparators", selected_summary["Macro-F1 Mean"], float(comparator_metrics["Macro-F1"].max()), ">"),
        ("brand F1 exceeds both comparators", float(class_means.get("brand", 0.0)), best_brand, ">"),
        ("drug_n F1 exceeds both comparators", float(class_means.get("drug_n", 0.0)), best_drug_n, ">"),
        ("Sparse router does not collapse", maximum_gate_sd, 0.015, ">="),
        ("MedLine F1 is not below best comparator", float(subgroup_means.get("MedLine", 0.0)), medline_floor, ">="),
        ("Unseen F1 is not below best comparator", float(subgroup_means.get("Unseen surface", 0.0)), unseen_floor, ">="),
        ("Cross-fitted ECE is no worse than current ACT-NER", selected_ece, prior["best_calibrated_ece"], "<="),
        ("Typed F1 gain is at least 0.005", selected_summary["Strict Typed Entity F1 Mean"] - best_typed, 0.005, ">="),
        ("Gain of at least 0.005 is present in all three seeds", float((seed_metrics["Strict Typed Entity F1"] >= best_typed + 0.005).sum()), 3.0, ">="),
    ]
    rows = []
    for criterion, observed, threshold, direction in conditions:
        passed = observed > threshold if direction == ">" else observed >= threshold if direction == ">=" else observed <= threshold
        rows.append(
            {
                "Development Criterion": criterion,
                "Observed": observed,
                "Required": threshold,
                "Direction": direction,
                "Passed": bool(passed),
            }
        )
    return pd.DataFrame(rows)


def run_stage_05_act_ner(project_root: Path, force_recompute: bool = True) -> dict:
    force_rebuild_proposed = True
    reuse_proposed_cache = False
    assert force_rebuild_proposed and not reuse_proposed_cache
    assert_read_only_artifacts(project_root)
    historical_design = json.loads((project_root / "src" / "frozen_design.json").read_text())
    act_ner_test_status = {
        "historical_locked_test_opened": bool(historical_design.get("locked_test_opened", False)),
        "historical_test_role": "Post-hoc Diagnostic Test",
        "act_ner_posthoc_test_opened": False,
    }
    (project_root / "src" / "act_ner_test_status.json").write_text(
        json.dumps(act_ner_test_status, indent=2), encoding="utf-8"
    )
    assert act_ner_test_status["act_ner_posthoc_test_opened"] is False
    _clear_old_proposed_artifacts(project_root)
    run_started_ns = __import__("time").time_ns()

    configurations = joint_configuration_space()
    ablation_units = (len(_ablation_configurations(configurations[0][1])) - 1) * 3 * 5
    total_progress_units = 1 + len(configurations) * 5 + 3 * 3 * 5 + ablation_units + 1
    live_progress = NotebookLiveProgress(total_progress_units)

    train_sentences, train_entities, _ = load_processed(project_root, "train")
    validation_sentences, validation_entities, validation_tokens = load_processed(project_root, "validation")
    evaluation_gold = gold_for_metrics(validation_entities)
    evaluation_ids = set(evaluation_gold.entity_id)
    assert len(evaluation_ids) == len(evaluation_gold)
    bundle, candidate_recall, expert_elapsed = build_candidate_bundle(
        project_root,
        validation_sentences,
        validation_entities,
        validation_tokens,
        marginal_threshold=0.55,
    )
    live_progress.advance("Frozen expert inference", "Candidate universe", "-", "-")
    fold_assignments, fold_balance = make_document_folds(validation_sentences, validation_entities)
    assert set(fold_assignments.document_id) == set(validation_sentences.document_id)

    joint_design = _configuration_table(configurations)
    screening_results: dict[str, dict] = {}
    screening_rows: list[pd.DataFrame] = []
    for name, config in configurations:
        result = cross_validate_configuration(
            bundle,
            validation_sentences,
            validation_entities,
            fold_assignments,
            config,
            seeds=[17],
            configuration_name=name,
            max_epochs=4,
            progress=live_progress,
            progress_phase="Joint screening",
        )
        screening_results[name] = result
        screening_rows.append(summarize_repeated_metrics(result["metrics"]))
    screening_summary = pd.concat(screening_rows, ignore_index=True)
    top_names = (
        screening_summary.sort_values(
            ["Strict Typed Entity F1 Mean", "Strict Boundary F1 Mean", "Strict Entity Recall Mean", "Macro-F1 Mean"],
            ascending=False,
        )
        .head(3)
        .Configuration.tolist()
    )
    configuration_map = dict(configurations)
    finalist_results: dict[str, dict] = {}
    finalist_rows: list[pd.DataFrame] = []
    for name in top_names:
        result = cross_validate_configuration(
            bundle,
            validation_sentences,
            validation_entities,
            fold_assignments,
            configuration_map[name],
            seeds=[17, 29, 43],
            configuration_name=name,
            max_epochs=4,
            progress=live_progress,
            progress_phase="Repeated finalists",
        )
        finalist_results[name] = result
        summary = summarize_repeated_metrics(result["metrics"])
        class_means = result["classwise"].groupby("Entity Type").F1.mean()
        finalist_subgroups = _seed_subgroups(
            result, validation_entities, validation_sentences, train_entities, name
        )
        domain_means = finalist_subgroups.query("Subgroup in ['DrugBank', 'MedLine']").groupby("Subgroup")["Strict Typed Entity F1"].mean()
        summary["Worst-domain F1 Mean"] = float(domain_means.min())
        summary["brand F1 Mean"] = float(class_means.get("brand", 0.0))
        summary["drug_n F1 Mean"] = float(class_means.get("drug_n", 0.0))
        summary["Latency per Sentence (ms)"] = 1000 * result["inference_elapsed"] / (3 * 5 * len(validation_sentences))
        summary["Parameter Count"] = int(result["parameter_count"])
        finalist_rows.append(summary)
    finalist_summary = pd.concat(finalist_rows, ignore_index=True)
    selected_name = str(_selection_order(finalist_summary).iloc[0].Configuration)
    selected_config = configuration_map[selected_name]
    selected_result = finalist_results[selected_name]
    selected_summary = finalist_summary.loc[finalist_summary.Configuration.eq(selected_name)].iloc[0].copy()

    ablation_results: dict[str, dict] = {}
    ablation_rows: list[dict] = []
    ablation_subgroups: list[pd.DataFrame] = []
    for name, config in _ablation_configurations(selected_config):
        if name == "Full ACT-NER":
            result = selected_result
        else:
            result = cross_validate_configuration(
                bundle,
                validation_sentences,
                validation_entities,
                fold_assignments,
                config,
                seeds=[17, 29, 43],
                configuration_name=name,
                max_epochs=4,
                progress=live_progress,
                progress_phase="Ablation",
            )
        subgroups = _seed_subgroups(
            result,
            validation_entities,
            validation_sentences,
            train_entities,
            name,
        )
        ablation_results[name] = result
        ablation_subgroups.append(subgroups)
        ablation_rows.append(_ablation_summary(name, result, subgroups))
    ablation = _add_boundary_precision_recall(
        pd.DataFrame(ablation_rows), ablation_results, validation_entities
    )
    all_ablation_subgroups = pd.concat(ablation_subgroups, ignore_index=True)
    assert set(ablation.Ablation) == {name for name, _ in _ablation_configurations(selected_config)}

    reference_seed = 17
    reference_predictions = selected_result["predictions"].query("Seed == @reference_seed").copy()
    reference_scores = selected_result["candidate_scores"].query("Seed == @reference_seed").copy()
    accepted_scores = accepted_candidate_rows(reference_scores, reference_predictions, validation_entities)
    calibrator, calibration_comparison, calibrated_rows = calibration_evidence(accepted_scores)
    policy = select_confidence_policy(calibrated_rows)
    reference_predictions = apply_calibration_and_policy(
        reference_predictions,
        accepted_scores,
        calibrator,
        policy,
    )
    risk_coverage, aurc = risk_coverage_table(calibrated_rows)
    calibration_comparison["Selected"] = calibration_comparison.Method.eq(calibrator.method)

    selected_classwise = selected_result["classwise"]
    selected_subgroups = _seed_subgroups(
        selected_result,
        validation_entities,
        validation_sentences,
        train_entities,
        selected_name,
    )
    acceptance = _acceptance_gate(
        selected_summary,
        selected_result["metrics"],
        selected_classwise,
        selected_subgroups,
        selected_result["gates"],
        calibration_comparison,
        project_root,
    )
    acceptance_passed = bool(acceptance.Passed.all())
    shift_diagnostics = shift_detector(bundle, fold_assignments)
    adaptation_protocol = continual_adaptation_protocol()

    all_documents = set(fold_assignments.document_id)
    final_model, final_history, final_training_elapsed = train_act_ner_model(
        bundle,
        all_documents,
        selected_config,
        reference_seed,
        max_epochs=6,
    )
    live_progress.advance("Frozen predictive model", "Final development fit", reference_seed, "All")
    proposed_directory = project_root / "Models" / "Proposed"
    checkpoint_path = proposed_directory / "act_ner.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": asdict(selected_config),
            "reference_seed": reference_seed,
            "training_scope": "Validation development documents after repeated GroupCV selection",
            "frozen_experts": {
                "transformer": "Models/Comparators/biomedical_transformer.pt",
                "crf": "Models/Baselines/linear_chain_crf.joblib",
            },
        },
        checkpoint_path,
    )
    joblib.dump(calibrator, proposed_directory / "entity_calibration_bundle.joblib")
    (proposed_directory / "confidence_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    assert checkpoint_path.stat().st_mtime_ns >= run_started_ns
    assert set(selected_result["metrics"].Seed) == {17, 29, 43}
    assert selected_result["metrics"].N.eq(len(evaluation_gold)).all()
    assert all(
        not predictions.entity_id.duplicated().any()
        for _, predictions in selected_result["predictions"].groupby("Seed")
    )

    prediction_columns = [column for column in reference_predictions.columns if column != "Configuration"]
    reference_predictions[prediction_columns].to_parquet(
        project_root / "Predictions" / "Validation" / "act_ner.parquet", index=False
    )
    reference_predictions[prediction_columns].to_parquet(
        project_root / "Predictions" / "OOF" / "act_ner.parquet", index=False
    )
    reference_metric, reference_classwise = evaluate_predictions(
        validation_entities,
        reference_predictions,
        "ACT-NER",
        "Validation cross-fit",
    )
    reference_calibration = calibration_metrics(
        reference_predictions,
        validation_entities,
        "ACT-NER",
        "Validation cross-fit",
    )
    comparator_metrics = pd.concat(
        [
            pd.read_csv(project_root / "artifacts" / "03_predictive_quality.csv"),
            pd.read_csv(project_root / "artifacts" / "04_predictive_quality.csv"),
        ],
        ignore_index=True,
    )
    validation_predictive = pd.concat([comparator_metrics, reference_metric], ignore_index=True, sort=False)
    effect_intervals = paired_bootstrap_ci(
        reference_predictions,
        validation_entities,
        "ACT-NER",
        17,
    )
    statistics = paired_statistical_comparison(
        reference_predictions,
        {
            "Linear-chain CRF": pd.read_parquet(project_root / "Predictions" / "Validation" / "linear_chain_crf.parquet"),
            "Biomedical Transformer": pd.read_parquet(project_root / "Predictions" / "Validation" / "biomedical_transformer.parquet"),
        },
        validation_entities,
    )
    error_taxonomy = boundary_error_profile(evaluation_gold, reference_predictions)
    parameter_count = sum(parameter.numel() for parameter in final_model.parameters())
    efficiency = pd.DataFrame(
        [
            {
                "Model": "ACT-NER",
                "Split": "Validation cross-fit",
                "N": len(validation_sentences),
                "Joint Selection + CV Time (s)": sum(result["elapsed"] for result in screening_results.values()) + sum(result["elapsed"] for result in finalist_results.values()),
                "Final Fit Time (s)": final_training_elapsed,
                "Latency per Sentence (ms)": 1000 * selected_result["inference_elapsed"] / (3 * 5 * len(validation_sentences)),
                "Parameter Count": parameter_count,
                "Model Size (MB)": checkpoint_path.stat().st_size / 2**20,
                "Peak RAM (MB)": psutil.Process().memory_info().rss / 2**20,
                "Peak VRAM (MB)": torch.mps.driver_allocated_memory() / 2**20 if torch.backends.mps.is_available() else np.nan,
            }
        ]
    )
    sensitivity = finalist_summary.merge(joint_design, on="Configuration", how="left")
    gate_diagnostics = selected_result["gates"]
    seed_results = selected_result["metrics"].copy()
    seed_summary = summarize_repeated_metrics(seed_results)
    loss_components = final_history
    frozen_design = {
        "model": "ACT-NER",
        "full_name": "Adaptive Calibrated Tri-Expert Network for Pharmacological Entity Recognition",
        "architecture": "Dynamic span-level Context, Sequence, and Rare-Signal Prototype experts with sparse top-2 routing",
        "checkpoint": "Models/Proposed/act_ner.pt",
        "calibrator": "Models/Proposed/entity_calibration_bundle.joblib",
        "confidence_policy": policy,
        "selected_configuration": selected_name,
        "configuration": asdict(selected_config),
        "reference_seed_predeclared": reference_seed,
        "requested_seeds": [17, 29, 43],
        "acceptance_passed": acceptance_passed,
        "historical_locked_test_opened": act_ner_test_status["historical_locked_test_opened"],
        "historical_test_role": "Post-hoc Diagnostic Test",
        "act_ner_posthoc_test_opened": False,
        "frozen_before_posthoc_notebook_06": True,
        "candidate_recall": float(candidate_recall.iloc[-1]["Validation Boundary Recall"]),
        "evaluation_gold_count": len(evaluation_gold),
        "calibration_method": calibrator.method,
        "aurc": aurc,
        "shift_detector": "Mahalanobis + MMD/Energy + expert disagreement",
        "continual_adaptation_primary_test_mixing": False,
    }
    (project_root / "src" / "frozen_design.json").write_text(
        json.dumps(frozen_design, indent=2), encoding="utf-8"
    )

    tables = {
        "05_candidate_recall": candidate_recall,
        "05_groupcv_fold_balance": fold_balance,
        "05_joint_search_design": joint_design,
        "05_joint_screening": screening_summary,
        "05_joint_finalists": finalist_summary,
        "05_act_seed_results": seed_results,
        "05_act_seed_summary": seed_summary,
        "05_validation_predictive_quality": validation_predictive,
        "05_validation_classwise_quality": reference_classwise,
        "05_ablation": ablation,
        "05_ablation_subgroups": all_ablation_subgroups,
        "05_sensitivity": sensitivity,
        "05_calibration_method_selection": calibration_comparison,
        "05_risk_coverage": risk_coverage,
        "05_gate_diagnostics": gate_diagnostics,
        "05_router_diagnostics": gate_diagnostics,
        "05_shift_diagnostics": shift_diagnostics,
        "05_continual_adaptation_protocol": adaptation_protocol,
        "05_training_loss_components": loss_components,
        "05_acceptance_gate": acceptance,
        "05_bootstrap_ci": effect_intervals,
        "05_paired_statistics": statistics,
        "05_error_taxonomy": error_taxonomy,
        "05_efficiency": efficiency,
    }
    for name, frame in tables.items():
        save_table(project_root, frame, name)
    figure_names = _save_act_ner_figures(
        project_root,
        candidate_recall,
        finalist_summary,
        ablation,
        gate_diagnostics,
        calibration_comparison,
        risk_coverage,
    )
    assert_read_only_artifacts(project_root)
    final_status = json.loads((project_root / "src" / "act_ner_test_status.json").read_text())
    assert final_status["act_ner_posthoc_test_opened"] is False
    assert checkpoint_path.exists()
    assert set(seed_results.Seed) == {17, 29, 43}
    assert len(ablation) == len(_ablation_configurations(selected_config))
    live_progress.close()
    return {
        "candidate_recall": candidate_recall,
        "fold_balance": fold_balance,
        "joint_design": joint_design,
        "joint_screening": screening_summary,
        "joint_finalists": finalist_summary,
        "seed_results": seed_results,
        "seed_summary": seed_summary,
        "predictive": validation_predictive,
        "classwise": reference_classwise,
        "ablations": ablation,
        "subgroups": selected_subgroups,
        "calibration": calibration_comparison,
        "risk_coverage": risk_coverage,
        "gates": gate_diagnostics,
        "router": gate_diagnostics,
        "shift": shift_diagnostics,
        "adaptation_protocol": adaptation_protocol,
        "loss_components": loss_components,
        "acceptance": acceptance,
        "statistics": statistics,
        "bootstrap": effect_intervals,
        "errors": error_taxonomy,
        "efficiency": efficiency,
        "freeze": pd.DataFrame([frozen_design]),
        "figure_names": figure_names,
    }


def _decision_rows_without_gold(
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Recover the frozen decoder evidence for accepted spans without consulting gold labels."""
    keys = list(zip(predictions.sentence_id, predictions.char_start, predictions.char_end))
    score_lookup = scores.set_index(["sentence_id", "char_start", "char_end"])
    accepted = score_lookup.loc[keys].reset_index().copy()
    prediction_lookup = predictions.set_index(["sentence_id", "char_start", "char_end"])
    ordered = prediction_lookup.loc[keys].reset_index()
    accepted["predicted_type"] = ordered.predicted_type.to_numpy(str)
    accepted["span_probability"] = ordered.boundary_probability.to_numpy(float)
    accepted["type_probability"] = ordered.type_probability.to_numpy(float)
    vectors = np.full((len(accepted), len(ENTITY_TYPES)), 0.0, dtype=np.float32)
    for row_index, (entity_type, probability) in enumerate(
        zip(accepted.predicted_type, accepted.type_probability)
    ):
        probability = float(np.clip(probability, 0.25, 1.0))
        vectors[row_index] = (1.0 - probability) / (len(ENTITY_TYPES) - 1)
        vectors[row_index, TYPE_TO_ID[entity_type]] = probability
    accepted["type_vector"] = list(vectors)
    return accepted


def _explicit_predictive_metrics(
    gold_entities: pd.DataFrame,
    predictions: pd.DataFrame,
    model: str,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric, classwise = evaluate_predictions(gold_entities, predictions, model, split)
    boundary_precision, boundary_recall = _boundary_precision_recall(gold_entities, predictions)
    metric.insert(3, "Boundary Precision", boundary_precision)
    metric.insert(4, "Boundary Recall", boundary_recall)
    return metric, classwise


def _posthoc_subgroup_metrics(
    predictions: pd.DataFrame,
    entities: pd.DataFrame,
    sentences: pd.DataFrame,
    train_entities: pd.DataFrame,
    model: str,
    split: str,
) -> pd.DataFrame:
    gold = gold_for_metrics(entities).copy()
    sentence_source = sentences.set_index("sentence_id").source.to_dict()
    seen = set(train_entities.query("not discontinuous").entity_text.map(normalize_surface))
    gold["source"] = gold.sentence_id.map(sentence_source)
    gold["seen"] = gold.entity_text.map(normalize_surface).isin(seen)
    gold["multi"] = gold.entity_text.str.contains(r"\s|[-/]", regex=True)
    predicted = predictions.copy()
    predicted["source"] = predicted.sentence_id.map(sentence_source)
    predicted["seen"] = predicted.entity_text.map(normalize_surface).isin(seen)
    predicted["multi"] = (predicted.token_end - predicted.token_start) > 1
    conditions = {
        "DrugBank": (gold.source.eq("DrugBank"), predicted.source.eq("DrugBank")),
        "MedLine": (gold.source.eq("MedLine"), predicted.source.eq("MedLine")),
        "Seen surface": (gold.seen, predicted.seen),
        "Unseen surface": (~gold.seen, ~predicted.seen),
        "Single-token": (~gold.multi, ~predicted.multi),
        "Multi-token": (gold.multi, predicted.multi),
        "Rare types (brand + drug_n)": (
            gold.entity_type.isin(["brand", "drug_n"]),
            predicted.predicted_type.isin(["brand", "drug_n"]),
        ),
    }
    rows = []
    for subgroup, (gold_mask, prediction_mask) in conditions.items():
        metric, _ = _explicit_predictive_metrics(
            gold.loc[gold_mask], predicted.loc[prediction_mask], model, split
        )
        row = metric.iloc[0]
        rows.append(
            {
                "Model": model,
                "Split": split,
                "N": int(row.N),
                "Subgroup": subgroup,
                "Boundary Precision": row["Boundary Precision"],
                "Boundary Recall": row["Boundary Recall"],
                "Strict Boundary F1": row["Strict Boundary F1"],
                "Typed Precision": row["Strict Entity Precision"],
                "Typed Recall": row["Strict Entity Recall"],
                "Strict Typed Entity F1": row["Strict Typed Entity F1"],
                "Macro-F1": row["Macro-F1"],
            }
        )
    return pd.DataFrame(rows)


def _common_universe_calibration(
    predictions: dict[str, pd.DataFrame],
    entities: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    gold = gold_for_metrics(entities)
    gold_typed = set(zip(gold.sentence_id, gold.char_start, gold.char_end, gold.entity_type))
    universe = set(gold_typed)
    for frame in predictions.values():
        universe.update(zip(frame.sentence_id, frame.char_start, frame.char_end, frame.predicted_type))
    ordered_universe = sorted(universe)
    outcomes = np.asarray([int(key in gold_typed) for key in ordered_universe], dtype=int)
    rows = []
    for model, frame in predictions.items():
        confidence_map = {
            key: float(confidence)
            for key, confidence in zip(
                zip(frame.sentence_id, frame.char_start, frame.char_end, frame.predicted_type),
                frame.calibrated_confidence,
            )
        }
        common_confidence = np.asarray([confidence_map.get(key, 0.0) for key in ordered_universe])
        boundary_gold = set(zip(gold.sentence_id, gold.char_start, gold.char_end))
        accepted_outcomes = np.asarray(
            [
                int(key in boundary_gold)
                for key in zip(frame.sentence_id, frame.char_start, frame.char_end)
            ],
            dtype=int,
        )
        accepted_confidence = frame.boundary_probability.to_numpy(float)
        accepted_typed = np.asarray(
            [
                int(key in gold_typed)
                for key in zip(frame.sentence_id, frame.char_start, frame.char_end, frame.predicted_type)
            ],
            dtype=int,
        )
        entity_confidence = frame.calibrated_confidence.to_numpy(float)
        rows.append(
            {
                "Model": model,
                "Split": split,
                "N": len(ordered_universe),
                "Common-universe ECE": expected_calibration_error(common_confidence, outcomes),
                "Common-universe Brier Score": float(np.mean((common_confidence - outcomes) ** 2)),
                "Accepted-span N": len(frame),
                "Accepted-span Boundary ECE": expected_calibration_error(
                    accepted_confidence, accepted_outcomes
                ),
                "Accepted-entity ECE": expected_calibration_error(
                    entity_confidence, accepted_typed
                ),
                "Accepted-entity Brier Score": float(
                    np.mean((entity_confidence - accepted_typed) ** 2)
                ),
            }
        )
    return pd.DataFrame(rows)


def _posthoc_risk_coverage(
    predictions: dict[str, pd.DataFrame],
    entities: pd.DataFrame,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = gold_for_metrics(entities)
    typed_gold = set(zip(gold.sentence_id, gold.char_start, gold.char_end, gold.entity_type))
    rows = []
    summaries = []
    for model, frame in predictions.items():
        ordered = frame.assign(
            typed_correct=[
                int(key in typed_gold)
                for key in zip(frame.sentence_id, frame.char_start, frame.char_end, frame.predicted_type)
            ]
        ).sort_values("calibrated_confidence", ascending=False)
        errors = 1.0 - ordered.typed_correct.to_numpy(float)
        cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
        for coverage in np.linspace(0.10, 1.00, 10):
            count = max(1, int(math.ceil(len(ordered) * coverage)))
            selected = ordered.iloc[:count]
            metric, _ = evaluate_predictions(gold, selected, model, split)
            rows.append(
                {
                    "Model": model,
                    "Split": split,
                    "N": len(gold),
                    "Coverage": coverage,
                    "Selected Predictions": count,
                    "Risk": float(errors[:count].mean()),
                    "Selective Strict Typed F1": metric.iloc[0]["Strict Typed Entity F1"],
                }
            )
        summaries.append(
            {
                "Model": model,
                "Split": split,
                "N": len(ordered),
                "AURC": float(cumulative_risk.mean()),
                "Coverage at confidence >= 0.50": float(
                    ordered.calibrated_confidence.ge(0.50).mean()
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _document_micro_difference_bootstrap(
    act_predictions: pd.DataFrame,
    comparator_predictions: dict[str, pd.DataFrame],
    entities: pd.DataFrame,
    seed: int = 1701,
    resamples: int = 2000,
) -> pd.DataFrame:
    documents = sorted(entities.document_id.unique())
    gold = gold_for_metrics(entities)

    def counts(frame: pd.DataFrame) -> np.ndarray:
        values = []
        for document in documents:
            gold_part = gold[gold.document_id.eq(document)]
            predicted_part = frame[frame.document_id.eq(document)]
            gold_set = set(zip(gold_part.sentence_id, gold_part.char_start, gold_part.char_end, gold_part.entity_type))
            predicted_set = set(zip(predicted_part.sentence_id, predicted_part.char_start, predicted_part.char_end, predicted_part.predicted_type))
            values.append((len(gold_set & predicted_set), len(predicted_set), len(gold_set)))
        return np.asarray(values, dtype=np.int64)

    act_counts = counts(act_predictions)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(documents), size=(resamples, len(documents)))

    def f1(values: np.ndarray) -> np.ndarray:
        totals = values[sampled].sum(axis=1)
        denominator = totals[:, 1] + totals[:, 2]
        return np.divide(2 * totals[:, 0], denominator, out=np.zeros(resamples), where=denominator > 0)

    act_bootstrap = f1(act_counts)
    act_total = act_counts.sum(axis=0)
    act_point = 2 * act_total[0] / max(act_total[1] + act_total[2], 1)
    rows = []
    for comparator, frame in comparator_predictions.items():
        other_counts = counts(frame)
        difference = act_bootstrap - f1(other_counts)
        other_total = other_counts.sum(axis=0)
        other_point = 2 * other_total[0] / max(other_total[1] + other_total[2], 1)
        rows.append(
            {
                "Comparison": f"ACT-NER vs {comparator}",
                "Estimand": "Corpus-level micro-F1 difference",
                "N Documents": len(documents),
                "Point Difference": act_point - other_point,
                "CI Lower": float(np.quantile(difference, 0.025)),
                "CI Upper": float(np.quantile(difference, 0.975)),
                "Bootstrap Unit": "document",
                "Resamples": resamples,
            }
        )
    return pd.DataFrame(rows)


def _type_confusion_posthoc(
    predictions: pd.DataFrame,
    entities: pd.DataFrame,
) -> pd.DataFrame:
    gold = gold_for_metrics(entities).set_index(["sentence_id", "char_start", "char_end"])
    exact = predictions.set_index(["sentence_id", "char_start", "char_end"]).join(
        gold[["entity_type"]], how="inner"
    )
    matrix = pd.crosstab(exact.entity_type, exact.predicted_type).reindex(
        index=ENTITY_TYPES, columns=ENTITY_TYPES, fill_value=0
    )
    matrix.index.name = "Gold type"
    return matrix.reset_index()


def _save_posthoc_figures(
    project_root: Path,
    reliability: pd.DataFrame,
    risk_coverage: pd.DataFrame,
    subgroups: pd.DataFrame,
    efficiency: pd.DataFrame,
    predictive: pd.DataFrame,
    confusion: pd.DataFrame,
    effect_intervals: pd.DataFrame,
) -> list[str]:
    names = []
    figure, axis = plt.subplots(figsize=(10.5, 8.5))
    for model, group in reliability.groupby("Model", sort=False):
        axis.plot(group["Mean Confidence"], group["Observed Accuracy"], marker="o", linewidth=2.4, label=model)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.8, label="Perfect calibration")
    axis.set_title("Post-hoc Calibration Reliability")
    axis.set_xlabel("Mean calibrated confidence")
    axis.set_ylabel("Observed typed accuracy")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(color="#E7EAED", linewidth=0.8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)
    figure.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.34)
    save_figure(project_root, figure, "act_ner_posthoc_reliability", reliability)
    names.append("act_ner_posthoc_reliability")

    figure, axis = plt.subplots(figsize=(11.0, 8.5))
    for model, group in risk_coverage.groupby("Model", sort=False):
        axis.plot(group.Coverage, group.Risk, marker="o", linewidth=2.4, label=model)
    axis.set_title("Risk–Coverage under Frozen Decisions")
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Typed prediction risk")
    axis.set_xlim(0.08, 1.02)
    axis.grid(color="#E7EAED", linewidth=0.8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
    figure.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.34)
    save_figure(project_root, figure, "act_ner_posthoc_risk_coverage", risk_coverage)
    names.append("act_ner_posthoc_risk_coverage")

    selected_groups = ["DrugBank", "MedLine", "Unseen surface", "Multi-token", "Rare types (brand + drug_n)"]
    profile = subgroups[subgroups.Subgroup.isin(selected_groups)].copy()
    figure, axis = plt.subplots(figsize=(15.0, 9.0))
    x_values = np.arange(len(selected_groups))
    for model, group in profile.groupby("Model", sort=False):
        ordered = group.set_index("Subgroup").reindex(selected_groups)
        axis.plot(x_values, ordered["Strict Typed Entity F1"], marker="o", linewidth=2.5, label=model)
    axis.set_xticks(x_values, selected_groups, rotation=15, ha="right")
    axis.set_title("Post-hoc Domain and Rare-Entity Profile")
    axis.set_xlabel("Evaluation subgroup")
    axis.set_ylabel("Strict Typed Entity F1")
    axis.grid(axis="y", color="#E7EAED", linewidth=0.8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    figure.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.40)
    save_figure(project_root, figure, "act_ner_posthoc_subgroups", profile)
    names.append("act_ner_posthoc_subgroups")

    matrix = confusion.set_index("Gold type").reindex(index=ENTITY_TYPES, columns=ENTITY_TYPES).to_numpy(float)
    figure, axis = plt.subplots(figsize=(9.5, 8.5))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    threshold = (float(matrix.max()) + float(matrix.min())) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{int(matrix[row, column]):,}", ha="center", va="center", color="white" if matrix[row, column] > threshold else "#222222")
    axis.set_xticks(np.arange(len(ENTITY_TYPES)), ENTITY_TYPES)
    axis.set_yticks(np.arange(len(ENTITY_TYPES)), ENTITY_TYPES)
    axis.set_title("ACT-NER Type Confusion at Exact Boundaries")
    axis.set_xlabel("Predicted type")
    axis.set_ylabel("Gold type")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("Entities")
    figure.subplots_adjust(left=0.17, right=0.88, top=0.90, bottom=0.15)
    save_figure(project_root, figure, "act_ner_posthoc_type_confusion", confusion)
    names.append("act_ner_posthoc_type_confusion")

    quality = predictive[["Model", "Strict Typed Entity F1"]].merge(
        efficiency[["Model", "Latency per Sentence (ms)"]], on="Model", how="inner"
    )
    figure, axis = plt.subplots(figsize=(13.5, 8.5))
    axis.scatter(quality["Latency per Sentence (ms)"], quality["Strict Typed Entity F1"], s=150, color="#2F6B8A", edgecolor="#222222")
    for _, row in quality.iterrows():
        label_offsets = {
            "Linear-chain CRF": (12, 18),
            "Biomedical Transformer": (12, -28),
            "ACT-NER": (12, 10),
            "BiLSTM-CRF": (12, 10),
            "O-only": (12, 10),
        }
        axis.annotate(
            row.Model,
            (row["Latency per Sentence (ms)"], row["Strict Typed Entity F1"]),
            xytext=label_offsets.get(str(row.Model), (12, 10)),
            textcoords="offset points",
        )
    axis.set_title("Frozen-model Latency–Quality Trade-off")
    axis.set_xlabel("Latency per sentence (ms)")
    axis.set_ylabel("Strict Typed Entity F1")
    maximum_latency = float(quality["Latency per Sentence (ms)"].max())
    axis.set_xlim(left=min(-0.2, float(quality["Latency per Sentence (ms)"].min()) - 0.2), right=maximum_latency * 1.12)
    axis.set_ylim(-0.04, max(0.80, float(quality["Strict Typed Entity F1"].max()) + 0.08))
    axis.grid(color="#E7EAED", linewidth=0.8)
    figure.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.16)
    save_figure(project_root, figure, "act_ner_posthoc_latency_quality", quality)
    names.append("act_ner_posthoc_latency_quality")

    plot_effect = effect_intervals.iloc[::-1].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(12.5, max(6.5, 2.0 + 1.4 * len(plot_effect))))
    y_values = np.arange(len(plot_effect))
    errors = np.vstack([
        plot_effect["Point Difference"] - plot_effect["CI Lower"],
        plot_effect["CI Upper"] - plot_effect["Point Difference"],
    ])
    axis.errorbar(plot_effect["Point Difference"], y_values, xerr=errors, fmt="o", markersize=10, capsize=7, color="#2F6B8A", ecolor="#C77C2B")
    axis.axvline(0, color="#555555", linestyle="--", linewidth=1.8)
    axis.set_yticks(y_values, plot_effect.Comparison)
    axis.set_title("Document-bootstrap Effect Intervals")
    axis.set_xlabel("Corpus-level Strict Typed F1 difference")
    axis.set_ylabel("Paired comparison")
    axis.grid(axis="x", color="#E7EAED", linewidth=0.8)
    figure.subplots_adjust(left=0.32, right=0.96, top=0.88, bottom=0.20)
    save_figure(project_root, figure, "act_ner_posthoc_effect_intervals", effect_intervals)
    names.append("act_ner_posthoc_effect_intervals")
    return names


def _reliability_bins(
    predictions: dict[str, pd.DataFrame],
    entities: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    gold = gold_for_metrics(entities)
    typed_gold = set(zip(gold.sentence_id, gold.char_start, gold.char_end, gold.entity_type))
    edges = np.linspace(0.0, 1.0, 11)
    rows = []
    for model, frame in predictions.items():
        confidence = frame.calibrated_confidence.to_numpy(float)
        outcomes = np.asarray([
            int(key in typed_gold)
            for key in zip(frame.sentence_id, frame.char_start, frame.char_end, frame.predicted_type)
        ])
        bins = np.minimum(np.digitize(confidence, edges[1:-1], right=True), 9)
        for bin_index in range(10):
            mask = bins == bin_index
            if mask.any():
                rows.append({
                    "Model": model,
                    "Split": split,
                    "Confidence Bin": f"{edges[bin_index]:.1f}–{edges[bin_index + 1]:.1f}",
                    "N": int(mask.sum()),
                    "Mean Confidence": float(confidence[mask].mean()),
                    "Observed Accuracy": float(outcomes[mask].mean()),
                })
    return pd.DataFrame(rows)


def run_stage_06_act_ner(project_root: Path) -> dict:
    """Run one inference-only evaluation on the already-inspected historical test corpus."""
    from core import PREDICTION_COLUMNS

    assert_read_only_artifacts(project_root)
    design_path = project_root / "src" / "frozen_design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    status_path = project_root / "src" / "act_ner_test_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert design["acceptance_passed"] is True
    assert design["frozen_before_posthoc_notebook_06"] is True
    assert status["act_ner_posthoc_test_opened"] is False
    split_label = "Post-hoc Diagnostic Test"
    live_progress = NotebookLiveProgress(4)

    for path in (
        project_root / "Predictions" / "Locked_Test" / "act_ner.parquet",
        project_root / "Predictions" / "Locked_Test" / "drug_entities_for_relation_extraction.parquet",
        project_root / "Predictions" / "Locked_Test" / "act_ner_diagnostics.parquet",
    ):
        if path.exists():
            path.unlink()
    for path in (project_root / "artifacts").glob("06_*.csv"):
        path.unlink()
    for directory in (project_root / "figures" / "PNG", project_root / "figures" / "PDF"):
        for path in directory.glob("act_ner_posthoc_*"):
            path.unlink()

    train_sentences, train_entities, train_tokens = load_processed(project_root, "train")
    validation_sentences, validation_entities, validation_tokens = load_processed(project_root, "validation")
    locked_sentences, locked_entities, locked_tokens = load_processed(project_root, "locked_test")
    empty_locked_entities = locked_entities.iloc[0:0].copy()
    locked_bundle, _, expert_elapsed = build_candidate_bundle(
        project_root,
        locked_sentences,
        empty_locked_entities,
        locked_tokens,
        marginal_threshold=float(design["configuration"]["marginal_threshold"]),
    )
    live_progress.advance("Frozen expert inference", "Historical test candidates", "17", "All")
    validation_bundle, _, _ = build_candidate_bundle(
        project_root,
        validation_sentences,
        validation_entities.iloc[0:0].copy(),
        validation_tokens,
        marginal_threshold=float(design["configuration"]["marginal_threshold"]),
    )
    checkpoint = torch.load(project_root / design["checkpoint"], map_location="cpu", weights_only=False)
    config = ACTNERConfig(**checkpoint["config"])
    model = ACTNER(config)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    locked_documents = set(locked_sentences.document_id)
    scores, gate_rows, head_elapsed = score_candidates(model, locked_bundle, locked_documents)
    predictions = decode_candidate_scores(
        scores,
        locked_sentences,
        config,
        "posthoc_diagnostic_test",
        int(checkpoint["reference_seed"]),
    )
    accepted_scores = _decision_rows_without_gold(scores, predictions)
    calibrator = joblib.load(project_root / design["calibrator"])
    policy = design["confidence_policy"]
    predictions = apply_calibration_and_policy(predictions, accepted_scores, calibrator, policy)
    predictions["split"] = split_label
    live_progress.advance("Frozen ACT-NER inference", "Span decisions and calibration", "17", "All")

    output_directory = project_root / "Predictions" / "Locked_Test"
    predictions.to_parquet(output_directory / "act_ner.parquet", index=False)
    handoff = predictions.loc[:, list(PREDICTION_COLUMNS)].copy()
    assert list(handoff.columns) == list(PREDICTION_COLUMNS)
    assert handoff.entity_id.is_unique
    assert not handoff.duplicated(["document_id", "sentence_id", "char_start", "char_end"]).any()
    sentence_text = locked_sentences.set_index("sentence_id").text.to_dict()
    assert all(
        text == sentence_text[sentence_id][int(start):int(end)]
        for sentence_id, start, end, text in zip(
            handoff.sentence_id, handoff.char_start, handoff.char_end, handoff.entity_text
        )
    )
    assert handoff[["boundary_probability", "type_probability", "calibrated_confidence"]].apply(
        lambda column: column.between(0.0, 1.0).all()
    ).all()
    assert (handoff.review_required == handoff.confidence_category.eq("low")).all()
    handoff.to_parquet(output_directory / "drug_entities_for_relation_extraction.parquet", index=False)

    reference_context = validation_bundle.context.numpy()
    target_context = locked_bundle.context.numpy()
    mahalanobis, mmd, energy = _distribution_shift_statistics(reference_context, target_context)
    accepted_keys = list(zip(predictions.sentence_id, predictions.char_start, predictions.char_end))
    diagnostics = accepted_scores.copy()
    diagnostics_lookup = diagnostics.set_index(["sentence_id", "char_start", "char_end"])
    diagnostics = diagnostics_lookup.loc[accepted_keys].reset_index()
    diagnostics.insert(0, "entity_id", predictions.entity_id.to_numpy(str))
    diagnostics["shift_mahalanobis"] = mahalanobis
    diagnostics["shift_mmd"] = mmd
    diagnostics["shift_energy_distance"] = energy
    diagnostics["adaptation_status"] = "Not activated; diagnostic evaluation is prediction-frozen"
    diagnostic_columns = [
        "document_id", "sentence_id", "entity_id", "candidate_source", "expert_agreement",
        "Router Context", "Router Sequence", "Router Rare-Signal Prototype", "Router Entropy",
        "shift_mahalanobis", "shift_mmd", "shift_energy_distance", "adaptation_status",
    ]
    diagnostics[diagnostic_columns].to_parquet(
        output_directory / "act_ner_diagnostics.parquet", index=False
    )

    model_files = {
        "O-only": "o_only.parquet",
        "Longest-match lexicon": "longest_match_lexicon.parquet",
        "Logistic Regression": "logistic_regression.parquet",
        "Linear-chain CRF": "linear_chain_crf.parquet",
        "BiLSTM-CRF": "bilstm_crf.parquet",
        "Biomedical Transformer": "biomedical_transformer.parquet",
    }
    comparison_predictions = {
        model_name: pd.read_parquet(output_directory / filename)
        for model_name, filename in model_files.items()
    }
    all_predictions = {**comparison_predictions, "ACT-NER": predictions}
    predictive_parts = []
    classwise_parts = []
    subgroup_parts = []
    error_parts = []
    for model_name, frame in all_predictions.items():
        metric, classwise = _explicit_predictive_metrics(
            locked_entities, frame, model_name, split_label
        )
        predictive_parts.append(metric)
        classwise_parts.append(classwise)
        subgroup_parts.append(
            _posthoc_subgroup_metrics(
                frame, locked_entities, locked_sentences, train_entities, model_name, split_label
            )
        )
        errors = boundary_error_profile(gold_for_metrics(locked_entities), frame)
        errors.insert(0, "Model", model_name)
        errors.insert(1, "Split", split_label)
        errors.insert(2, "N", len(gold_for_metrics(locked_entities)))
        error_parts.append(errors)
    predictive = pd.concat(predictive_parts, ignore_index=True)
    classwise = pd.concat(classwise_parts, ignore_index=True)
    subgroups = pd.concat(subgroup_parts, ignore_index=True)
    boundary_errors = pd.concat(error_parts, ignore_index=True)

    primary_models = {
        model_name: all_predictions[model_name]
        for model_name in ("Linear-chain CRF", "Biomedical Transformer", "ACT-NER")
    }
    calibration = _common_universe_calibration(primary_models, locked_entities, split_label)
    risk_coverage, risk_summary = _posthoc_risk_coverage(
        primary_models, locked_entities, split_label
    )
    reliability = _reliability_bins(primary_models, locked_entities, split_label)
    bootstrap = pd.concat(
        [paired_bootstrap_ci(frame, locked_entities, model_name, 1701) for model_name, frame in all_predictions.items()],
        ignore_index=True,
    )
    comparators = {
        "Linear-chain CRF": primary_models["Linear-chain CRF"],
        "Biomedical Transformer": primary_models["Biomedical Transformer"],
    }
    wilcoxon = paired_statistical_comparison(predictions, comparators, locked_entities)
    wilcoxon.insert(1, "Estimand", "Mean within-document Strict Typed F1 difference")
    effect_intervals = _document_micro_difference_bootstrap(
        predictions, comparators, locked_entities
    )
    confusion = _type_confusion_posthoc(predictions, locked_entities)
    live_progress.advance("Post-hoc evaluation", "Metrics and paired evidence", "-", "All")

    baseline_efficiency = pd.read_csv(project_root / "artifacts" / "03_efficiency.csv")
    comparator_efficiency = pd.read_csv(project_root / "artifacts" / "04_efficiency.csv")
    reference_efficiency = pd.concat([baseline_efficiency, comparator_efficiency], ignore_index=True, sort=False)
    reference_efficiency = reference_efficiency[[
        "Model", "Latency per Sentence (ms)", "Latency per Entity (ms)",
        "Throughput (sentences/s)", "Parameter Count", "Model Size (MB)",
        "Peak RAM (MB)", "Peak VRAM (MB)",
    ]]
    act_elapsed = expert_elapsed + head_elapsed
    act_parameters = sum(parameter.numel() for parameter in model.parameters())
    act_checkpoint_size = (project_root / design["checkpoint"]).stat().st_size / 2**20
    act_efficiency = pd.DataFrame([{
        "Model": "ACT-NER",
        "Latency per Sentence (ms)": 1000 * act_elapsed / len(locked_sentences),
        "Latency per Entity (ms)": 1000 * act_elapsed / max(len(predictions), 1),
        "Throughput (sentences/s)": len(locked_sentences) / max(act_elapsed, EPSILON),
        "Parameter Count": act_parameters,
        "Model Size (MB)": act_checkpoint_size,
        "Peak RAM (MB)": psutil.Process().memory_info().rss / 2**20,
        "Peak VRAM (MB)": torch.mps.driver_allocated_memory() / 2**20 if torch.backends.mps.is_available() else np.nan,
    }])
    efficiency = pd.concat([reference_efficiency, act_efficiency], ignore_index=True)
    efficiency.insert(1, "Split", np.where(efficiency.Model.eq("ACT-NER"), split_label, "Frozen Validation measurement"))
    efficiency.insert(2, "N", np.where(efficiency.Model.eq("ACT-NER"), len(locked_sentences), len(validation_sentences)))
    shift = pd.DataFrame([{
        "Reference": "Validation candidate representations",
        "Target": split_label,
        "N Reference Candidates": len(validation_bundle.frame),
        "N Target Candidates": len(locked_bundle.frame),
        "Mahalanobis Distance": mahalanobis,
        "Maximum Mean Discrepancy": mmd,
        "Energy Distance": energy,
        "Adaptation Used": False,
    }])
    ablation = pd.read_csv(project_root / "artifacts" / "05_ablation.csv")
    acceptance = pd.read_csv(project_root / "artifacts" / "05_acceptance_gate.csv")

    tables = {
        "06_predictive_quality": predictive,
        "06_classwise_quality": classwise,
        "06_calibration_common_universe": calibration,
        "06_risk_coverage": risk_coverage,
        "06_risk_summary": risk_summary,
        "06_reliability_bins": reliability,
        "06_subgroup_performance": subgroups,
        "06_boundary_errors": boundary_errors,
        "06_efficiency": efficiency,
        "06_bootstrap_ci": bootstrap,
        "06_document_wilcoxon": wilcoxon,
        "06_paired_statistics": wilcoxon,
        "06_corpus_micro_effect_intervals": effect_intervals,
        "06_effect_intervals": effect_intervals,
        "06_entity_type_confusion": confusion,
        "06_shift_diagnostics": shift,
        "06_ablation_from_development": ablation,
        "06_development_acceptance_gate": acceptance,
    }
    for name, frame in tables.items():
        save_table(project_root, frame, name)
    figure_names = _save_posthoc_figures(
        project_root, reliability, risk_coverage, subgroups, efficiency, predictive, confusion, effect_intervals
    )

    act_metric = predictive.query("Model == 'ACT-NER'").iloc[0]
    best_comparator = predictive.query("Model != 'ACT-NER'").sort_values("Strict Typed Entity F1", ascending=False).iloc[0]
    result_statement = (
        "The historical test comparison is post-hoc and cannot confirm superiority. "
        f"ACT-NER achieved Strict Typed Entity F1 {act_metric['Strict Typed Entity F1']:.4f}; "
        f"the strongest frozen comparator achieved {best_comparator['Strict Typed Entity F1']:.4f}."
    )
    reports = project_root / "Reports"
    reports.mkdir(exist_ok=True)
    (reports / "methods_narrative.md").write_text(
        "# Methods\n\nACT-NER was selected exclusively by repeated document-level GroupCV on development data. "
        "The historical DDIExtraction test was opened only for frozen, inference-only post-hoc diagnosis. "
        "No checkpoint, threshold, loss, router, calibrator, or comparator artifact was fitted or modified in Notebook 06.\n",
        encoding="utf-8",
    )
    (reports / "results_narrative.md").write_text(
        "# Results\n\n" + result_statement + " Confirmatory evidence remains the predeclared repeated GroupCV protocol until a new untouched or external test is available.\n",
        encoding="utf-8",
    )
    (reports / "proposed_model_architecture.md").write_text(
        "# ACT-NER architecture\n\nACT-NER is a span-level tri-expert network combining a frozen biomedical Transformer context expert, a frozen linear-chain CRF sequence expert, and a trainable rare-signal prototype expert through sparse top-2 routing. Transformer context remains the residual backbone; a span verifier and an independent type head precede post-training calibration.\n",
        encoding="utf-8",
    )
    (reports / "ablation_and_robustness.md").write_text(
        "# Ablation and robustness\n\nComponent-level repeated GroupCV ablations are reported in `artifacts/05_ablation.csv`. Post-hoc domain, seen/unseen, length, and rare-type results are reported in `artifacts/06_subgroup_performance.csv`; continual adaptation was not activated during diagnostic evaluation.\n",
        encoding="utf-8",
    )
    (reports / "limitations_and_threats_to_validity.md").write_text(
        "# Limitations and threats to validity\n\nThe original locked test had already informed the architectural revision and is therefore labelled Post-hoc Diagnostic Test. Its results must not be treated as untouched confirmatory evidence. Comparator efficiency values retain their frozen Validation measurement scope, and no independent external corpus was evaluated.\n",
        encoding="utf-8",
    )
    claim_map = pd.DataFrame([
        {"Claim": "Development superiority", "Evidence": "Repeated document-level GroupCV and 13/13 acceptance criteria", "Status": "Supported on development evidence"},
        {"Claim": "Historical test performance", "Evidence": "Frozen inference-only Notebook 06 tables", "Status": "Post-hoc diagnostic only"},
        {"Claim": "Untouched-test superiority", "Evidence": "No new relocked or external test", "Status": "Not established"},
        {"Claim": "Relation-extraction handoff compatibility", "Evidence": "Stable 14-field parquet contract and offset assertions", "Status": "Supported"},
    ])
    (reports / "claim_evidence_map.md").write_text(
        "# Claim–evidence map\n\n" + claim_map.to_markdown(index=False) + "\n", encoding="utf-8"
    )

    status.update({
        "historical_locked_test_opened": True,
        "historical_test_role": split_label,
        "act_ner_posthoc_test_opened": True,
        "evaluated_without_refit": True,
    })
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    design.update({
        "historical_locked_test_opened": True,
        "historical_test_role": split_label,
        "act_ner_posthoc_test_opened": True,
        "posthoc_evaluated_without_refit": True,
    })
    design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    assert_read_only_artifacts(project_root)
    live_progress.advance("Final outputs", "Tables, figures, and handoff", "-", "All")
    live_progress.close()
    return {
        "predictive": predictive,
        "classwise": classwise,
        "calibration": calibration,
        "risk_coverage": risk_coverage,
        "risk_summary": risk_summary,
        "subgroups": subgroups,
        "errors": boundary_errors,
        "efficiency": efficiency,
        "bootstrap": bootstrap,
        "statistics": wilcoxon,
        "effect_intervals": effect_intervals,
        "confusion": confusion,
        "shift": shift,
        "ablations": ablation,
        "acceptance": acceptance,
        "handoff_schema": pd.DataFrame([{"Column": column, "Stable Contract": True} for column in PREDICTION_COLUMNS]),
        "conclusion": pd.DataFrame([{"Evaluation Role": split_label, "Evidence-bounded conclusion": result_statement}]),
        "figure_names": figure_names,
    }
