from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from matplotlib import font_manager
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from transformers import AutoTokenizer

ENTITY_TYPES = ("drug", "brand", "group", "drug_n")
BOUNDARY_TAGS = ("O", "B", "I", "E", "S")
TYPED_TAGS = ("O",) + tuple(
    f"{prefix}-{entity_type}"
    for entity_type in ENTITY_TYPES
    for prefix in ("B", "I", "E", "S")
)
PREDICTION_COLUMNS = (
    "document_id",
    "sentence_id",
    "entity_id",
    "char_start",
    "char_end",
    "token_start",
    "token_end",
    "entity_text",
    "predicted_type",
    "boundary_probability",
    "type_probability",
    "calibrated_confidence",
    "confidence_category",
    "review_required",
)


def project_root_from_cwd() -> Path:
    current = Path.cwd()
    if current.name == "notebooks":
        current = current.parent
    required = current / "PROJECT_REQUIREMENTS.md"
    if not required.exists():
        raise RuntimeError("Run from the repository root or its notebooks directory.")
    return current


def shared_root(project_root: Path) -> Path:
    # This is the sole notebook data contract: repository root + dataset.
    root = (project_root / "dataset").resolve()
    if not (root / "Raw" / "ACQUISITION.json").exists():
        raise RuntimeError("Immutable shared dataset acquisition is missing.")
    return root


def load_settings(project_root: Path) -> dict:
    return json.loads((project_root / "src" / "experiment.json").read_text())


def configure_notebook() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 80)
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 20,
            "figure.dpi": 350,
            "savefig.dpi": 350,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.grid": False,
        }
    )
    resolved = font_manager.findfont("Times New Roman", fallback_to_default=False)
    assert "Times New Roman" in resolved


def display_complete(frame: pd.DataFrame, precision: int = 3) -> None:
    formatted = frame.copy()
    float_columns = formatted.select_dtypes(include=["float", "float32", "float64"]).columns
    formatter = {column: f"{{:.{precision}f}}" for column in float_columns}
    with pd.option_context(
        "display.max_rows",
        max(len(formatted), 1),
        "display.max_columns",
        max(len(formatted.columns), 1),
        "display.max_colwidth",
        80,
    ):
        display(formatted.style.format(formatter, na_rep="NA"))


def save_table(project_root: Path, frame: pd.DataFrame, name: str) -> Path:
    path = project_root / "artifacts" / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def adaptive_figsize(
    n_categories: int = 1,
    n_panels: int = 1,
    max_label_chars: int = 12,
    legend_rows: int = 0,
    heatmap_shape: tuple[int, int] | None = None,
) -> tuple[float, float]:
    if heatmap_shape is not None:
        rows, columns = heatmap_shape
        width = max(9.5, 2.0 * columns + 3.0)
        height = max(7.0, 1.35 * rows + 2.8)
        return width, height
    width = max(10.0, 1.15 * n_categories + 0.10 * max_label_chars + 3.5)
    width = max(width, 7.2 * min(n_panels, 2))
    height = 7.0 + 0.45 * max(0, legend_rows) + 2.0 * max(0, math.ceil(n_panels / 2) - 1)
    return min(width, 24.0), min(height, 20.0)


def save_figure(
    project_root: Path,
    figure: mpl.figure.Figure,
    name: str,
    numeric_data: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    figure.canvas.draw()
    for text_artist in figure.findobj(match=mpl.text.Text):
        if not text_artist.get_text().strip():
            continue
        size = float(text_artist.get_fontsize())
        assert 17.99 <= size <= 20.01, (name, text_artist.get_text(), size)
        text_artist.set_fontfamily("Times New Roman")
    png_path = project_root / "figures" / "PNG" / f"{name}.png"
    pdf_path = project_root / "figures" / "PDF" / f"{name}.pdf"
    figure.savefig(png_path, dpi=350, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    assert mpl.image.imread(png_path).size > 0
    if numeric_data is not None:
        numeric_data.to_csv(project_root / "artifacts" / f"figure_{name}_data.csv", index=False)
    return png_path, pdf_path


def display_saved_figure(project_root: Path, name: str, width: int = 1100) -> None:
    display(Image(filename=str(project_root / "figures" / "PNG" / f"{name}.png"), width=width))


def _source_from_path(path: Path) -> str:
    if path.parent.name == "DrugBank":
        return "DrugBank"
    if path.parent.name == "MedLine":
        return "MedLine"
    raise ValueError(f"Unknown source for {path.name}")


def _parse_offsets(raw_offset: str) -> list[tuple[int, int]]:
    fragments = []
    for fragment in raw_offset.split(";"):
        start, inclusive_end = (int(value) for value in fragment.split("-"))
        fragments.append((start, inclusive_end + 1))
    return fragments


def parse_xml_files(paths: Iterable[Path], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    document_rows: list[dict] = []
    sentence_rows: list[dict] = []
    entity_rows: list[dict] = []
    for path in sorted(paths):
        document = ET.parse(path).getroot()
        document_id = document.attrib["id"]
        source = _source_from_path(path)
        sentences = document.findall("sentence")
        document_rows.append(
            {
                "document_id": document_id,
                "source": source,
                "split": split,
                "n_sentences": len(sentences),
            }
        )
        for sentence_index, sentence in enumerate(sentences):
            sentence_id = sentence.attrib["id"]
            text = sentence.attrib.get("text", "")
            entity_nodes = sentence.findall("entity")
            sentence_rows.append(
                {
                    "document_id": document_id,
                    "sentence_id": sentence_id,
                    "sentence_index": sentence_index,
                    "source": source,
                    "split": split,
                    "text": text,
                    "n_characters": len(text),
                    "n_entities_raw": len(entity_nodes),
                }
            )
            for entity in entity_nodes:
                fragments = _parse_offsets(entity.attrib["charOffset"])
                discontinuous = len(fragments) > 1
                char_start = min(start for start, _ in fragments)
                char_end = max(end for _, end in fragments)
                fragment_texts = [text[start:end] for start, end in fragments]
                expected_text = entity.attrib.get("text", "")
                exact_text = expected_text == (" ".join(fragment_texts) if discontinuous else fragment_texts[0])
                valid = all(0 <= start < end <= len(text) for start, end in fragments)
                entity_rows.append(
                    {
                        "document_id": document_id,
                        "sentence_id": sentence_id,
                        "entity_id": entity.attrib["id"],
                        "source": source,
                        "split": split,
                        "char_start": char_start,
                        "char_end": char_end,
                        "char_offset_xml": entity.attrib["charOffset"],
                        "entity_text": expected_text,
                        "entity_type": entity.attrib["type"],
                        "discontinuous": discontinuous,
                        "n_fragments": len(fragments),
                        "offset_valid": valid,
                        "text_exact": exact_text,
                    }
                )
    documents = pd.DataFrame(document_rows)
    sentences = pd.DataFrame(sentence_rows)
    entities = pd.DataFrame(entity_rows)
    assert documents["document_id"].is_unique
    assert sentences["sentence_id"].is_unique
    assert entities["entity_id"].is_unique
    assert set(entities["entity_type"]) <= set(ENTITY_TYPES)
    return documents, sentences, entities


def _manifest_paths(project_root: Path, split: str) -> list[Path]:
    shared = shared_root(project_root)
    manifest = pd.read_csv(shared / "Splits" / f"{split}_documents.csv")
    return [shared / value for value in manifest["relative_xml"]]


def load_development_corpus(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for split in ("train", "validation"):
        frames.append(parse_xml_files(_manifest_paths(project_root, split), split))
    documents = pd.concat([item[0] for item in frames], ignore_index=True)
    sentences = pd.concat([item[1] for item in frames], ignore_index=True)
    entities = pd.concat([item[2] for item in frames], ignore_index=True)
    assert not set(documents.query("split == 'train'")["document_id"]) & set(
        documents.query("split == 'validation'")["document_id"]
    )
    return documents, sentences, entities


def load_split_corpus(project_root: Path, split: str, allow_locked: bool = False):
    if split == "locked_test" and not allow_locked:
        raise PermissionError("Locked Test can be read only by Notebook 06.")
    return parse_xml_files(_manifest_paths(project_root, split), split)


def build_split_manifests(project_root: Path) -> pd.DataFrame:
    shared = shared_root(project_root)
    settings = load_settings(project_root)
    corpus_root = shared / "Raw" / "DDICorpus-2013" / "DDICorpus"
    development_paths = sorted((corpus_root / "Train").glob("*/*.xml"))
    locked_paths = sorted((corpus_root / "Test" / "Test for DrugNER task").glob("*/*.xml"))
    documents, _, entities = parse_xml_files(development_paths, "development")
    entity_counts = (
        entities.query("not discontinuous")
        .groupby(["document_id", "entity_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ENTITY_TYPES, fill_value=0)
    )
    profile = documents.set_index("document_id").join(entity_counts, how="left").fillna(0)
    rng = np.random.default_rng(settings["split_seed"])
    validation_ids: set[str] = set()
    fraction = float(settings["validation_fraction_of_original_train"])
    for source, source_frame in profile.groupby("source"):
        ids = source_frame.index.to_numpy()
        n_validation = int(round(len(ids) * fraction))
        target = source_frame[list(ENTITY_TYPES)].sum().to_numpy(float) * fraction
        scales = np.maximum(target, 1.0)
        best_ids = None
        best_score = float("inf")
        for _ in range(4000):
            candidate = rng.choice(ids, size=n_validation, replace=False)
            observed = source_frame.loc[candidate, list(ENTITY_TYPES)].sum().to_numpy(float)
            score = float(np.mean(((observed - target) / scales) ** 2))
            if score < best_score:
                best_score = score
                best_ids = candidate
        assert best_ids is not None
        validation_ids.update(best_ids.tolist())
    relative = lambda path: path.relative_to(shared).as_posix()
    path_by_id = {ET.parse(path).getroot().attrib["id"]: path for path in development_paths}
    rows = []
    for document_id in sorted(profile.index):
        split = "validation" if document_id in validation_ids else "train"
        rows.append(
            {
                "document_id": document_id,
                "source": profile.loc[document_id, "source"],
                "original_partition": "Train",
                "relative_xml": relative(path_by_id[document_id]),
                "split": split,
            }
        )
    development_manifest = pd.DataFrame(rows)
    for split in ("train", "validation"):
        development_manifest.query("split == @split").drop(columns="split").to_csv(
            shared / "Splits" / f"{split}_documents.csv", index=False
        )
    locked_rows = []
    for path in locked_paths:
        document_id = ET.parse(path).getroot().attrib["id"]
        locked_rows.append(
            {
                "document_id": document_id,
                "source": _source_from_path(path),
                "original_partition": "Test for DrugNER task",
                "relative_xml": relative(path),
            }
        )
    pd.DataFrame(locked_rows).sort_values("document_id").to_csv(
        shared / "Splits" / "locked_test_documents.csv", index=False
    )
    assert set(development_manifest.query("split == 'train'")["document_id"]).isdisjoint(validation_ids)
    assert len(locked_rows) == 112
    return development_manifest


def normalize_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower().strip()
    normalized = re.sub(r"[‐‑‒–—−]", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def add_eda_attributes(sentences: pd.DataFrame, entities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tokenizer_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    sentence_frame = sentences.copy()
    sentence_frame["n_tokens"] = sentence_frame["text"].map(lambda value: len(tokenizer_pattern.findall(value)))
    entity_frame = entities.copy()
    entity_frame["n_characters"] = entity_frame["char_end"] - entity_frame["char_start"]
    entity_frame["n_tokens_regex"] = entity_frame["entity_text"].map(
        lambda value: len(tokenizer_pattern.findall(value))
    )
    entity_frame["normalized_surface"] = entity_frame["entity_text"].map(normalize_surface)
    entity_frame["surface_frequency"] = entity_frame.groupby("normalized_surface")["entity_id"].transform("size")
    return sentence_frame, entity_frame


def audit_development_corpus(
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
    entities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sentence_frame, entity_frame = add_eda_attributes(sentences, entities)
    contiguous = entity_frame.query("not discontinuous")
    overlap_count = 0
    for _, group in contiguous.sort_values(["sentence_id", "char_start"]).groupby("sentence_id"):
        intervals = group[["char_start", "char_end"]].to_numpy()
        for index in range(1, len(intervals)):
            overlap_count += int(intervals[index, 0] < intervals[index - 1, 1])
    conflict_count = int(
        entity_frame.groupby(["sentence_id", "char_start", "char_end"])["entity_type"].nunique().gt(1).sum()
    )
    repeated_surface_mentions = int(
        entity_frame.duplicated(["sentence_id", "normalized_surface", "entity_type"], keep=False).sum()
    )
    exact_duplicates = int(
        entity_frame.duplicated(
            ["sentence_id", "char_start", "char_end", "entity_type"], keep=False
        ).sum()
    )
    quality = pd.DataFrame(
        [
            ("Documents", len(documents), "document"),
            ("Sentences", len(sentence_frame), "sentence"),
            ("Entities (raw)", len(entity_frame), "entity"),
            ("Unique surface forms", entity_frame["normalized_surface"].nunique(), "normalized surface"),
            ("Invalid offsets", int((~entity_frame["offset_valid"]).sum()), "entity"),
            ("Text/offset mismatches", int((~entity_frame["text_exact"]).sum()), "entity"),
            ("Empty spans", int((entity_frame["char_end"] <= entity_frame["char_start"]).sum()), "entity"),
            ("Discontinuous entities", int(entity_frame["discontinuous"].sum()), "entity"),
            ("Overlapping contiguous pairs", overlap_count, "pair"),
            ("Conflicting exact-span types", conflict_count, "span"),
            ("Exact duplicate annotations", exact_duplicates, "entity"),
            ("Repeated surface mentions within a sentence", repeated_surface_mentions, "entity"),
        ],
        columns=["Check", "Count", "Unit"],
    )
    source_profile = (
        documents.groupby(["split", "source"]).size().rename("Documents").reset_index()
        .merge(
            sentence_frame.groupby(["split", "source"]).size().rename("Sentences").reset_index(),
            on=["split", "source"],
        )
        .merge(
            entity_frame.groupby(["split", "source"]).size().rename("Entities").reset_index(),
            on=["split", "source"],
        )
    )
    return quality, source_profile


def build_training_lexicon(entities: pd.DataFrame) -> pd.DataFrame:
    train_entities = entities.query("split == 'train' and not discontinuous").copy()
    train_entities["normalized_surface"] = train_entities["entity_text"].map(normalize_surface)
    counts = (
        train_entities.groupby(["normalized_surface", "entity_type"])
        .size()
        .rename("type_count")
        .reset_index()
    )
    totals = counts.groupby("normalized_surface")["type_count"].sum().rename("frequency")
    type_counts = counts.pivot(index="normalized_surface", columns="entity_type", values="type_count").fillna(0)
    type_counts = type_counts.reindex(columns=ENTITY_TYPES, fill_value=0)
    probabilities = type_counts.div(type_counts.sum(axis=1), axis=0)
    entropy = -(probabilities.where(probabilities > 0, 1.0) * np.log(probabilities.where(probabilities > 0, 1.0))).sum(axis=1)
    entropy = entropy / math.log(len(ENTITY_TYPES))
    majority = type_counts.idxmax(axis=1)
    representative = (
        train_entities.sort_values(["normalized_surface", "entity_text"])
        .groupby("normalized_surface")["entity_text"]
        .first()
    )
    lexicon = pd.DataFrame(
        {
            "normalized_surface": totals.index,
            "surface_form": representative.reindex(totals.index).values,
            "frequency": totals.values,
            "majority_type": majority.reindex(totals.index).values,
            "ambiguity_score": entropy.reindex(totals.index).values,
        }
    )
    for entity_type in ENTITY_TYPES:
        lexicon[f"p_{entity_type}"] = probabilities[entity_type].reindex(totals.index).values
    return lexicon.sort_values(["frequency", "normalized_surface"], ascending=[False, True]).reset_index(drop=True)


def _bioes_labels(n_tokens: int, entity_type: str) -> list[str]:
    if n_tokens == 1:
        return [f"S-{entity_type}"]
    if n_tokens == 2:
        return [f"B-{entity_type}", f"E-{entity_type}"]
    return [f"B-{entity_type}"] + [f"I-{entity_type}"] * (n_tokens - 2) + [f"E-{entity_type}"]


def tokenize_split(
    sentences: pd.DataFrame,
    entities: pd.DataFrame,
    tokenizer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_rows: list[dict] = []
    alignment_rows: list[dict] = []
    entities_by_sentence = {key: group for key, group in entities.groupby("sentence_id")}
    for sentence in sentences.itertuples(index=False):
        encoded = tokenizer(sentence.text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        typed_labels = ["O"] * len(tokens)
        sentence_entities = entities_by_sentence.get(sentence.sentence_id, pd.DataFrame())
        for entity in sentence_entities.itertuples(index=False):
            if entity.discontinuous:
                alignment_rows.append(
                    {
                        "entity_id": entity.entity_id,
                        "sentence_id": entity.sentence_id,
                        "split": entity.split,
                        "alignment_exact": False,
                        "alignment_status": "discontinuous_excluded",
                    }
                )
                continue
            covered = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > entity.char_start and start < entity.char_end
            ]
            assert covered
            exact = offsets[covered[0]][0] == entity.char_start and offsets[covered[-1]][1] == entity.char_end
            labels = _bioes_labels(len(covered), entity.entity_type)
            assert all(typed_labels[index] == "O" for index in covered)
            for index, label in zip(covered, labels):
                typed_labels[index] = label
            alignment_rows.append(
                {
                    "entity_id": entity.entity_id,
                    "sentence_id": entity.sentence_id,
                    "split": entity.split,
                    "alignment_exact": exact,
                    "alignment_status": "exact" if exact else "outward_projection",
                }
            )
        for token_index, (input_id, token, offset, label) in enumerate(
            zip(encoded["input_ids"], tokens, offsets, typed_labels)
        ):
            start, end = offset
            token_rows.append(
                {
                    "document_id": sentence.document_id,
                    "sentence_id": sentence.sentence_id,
                    "source": sentence.source,
                    "split": sentence.split,
                    "token_index": token_index,
                    "input_id": int(input_id),
                    "token_text": token,
                    "surface_text": sentence.text[start:end],
                    "char_start": start,
                    "char_end": end,
                    "typed_tag": label,
                    "boundary_tag": label.split("-", 1)[0],
                    "entity_type_target": label.split("-", 1)[1] if "-" in label else "O",
                }
            )
    token_frame = pd.DataFrame(token_rows)
    alignment_frame = pd.DataFrame(alignment_rows)
    return token_frame, alignment_frame


def prepare_modeling_data(project_root: Path) -> dict[str, pd.DataFrame]:
    settings = load_settings(project_root)
    tokenizer = AutoTokenizer.from_pretrained(settings["biomedical_backbone"], use_fast=True)
    outputs: dict[str, pd.DataFrame] = {}
    all_entities = []
    feature_dictionary = {
        "entity_types": list(ENTITY_TYPES),
        "boundary_tags": list(BOUNDARY_TAGS),
        "typed_tags": list(TYPED_TAGS),
        "tokenizer": settings["biomedical_backbone"],
        "max_sequence_length": settings["max_sequence_length"],
        "orthographic_features": [
            "capitalization",
            "digits",
            "hyphenation",
            "greek_or_latin",
            "punctuation",
            "token_length",
            "prefix_bucket",
            "suffix_bucket",
            "drug_morphology",
            "relative_position",
        ],
    }
    shared = shared_root(project_root)
    for split in ("train", "validation"):
        documents, sentences, entities = load_split_corpus(project_root, split)
        tokens, alignment = tokenize_split(sentences, entities, tokenizer)
        documents.to_parquet(shared / "Processed" / f"{split}_documents.parquet", index=False)
        sentences.to_parquet(shared / "Processed" / f"{split}_sentences.parquet", index=False)
        entities.to_parquet(shared / "Processed" / f"{split}_entities.parquet", index=False)
        tokens.to_parquet(shared / "Processed" / f"{split}_tokens.parquet", index=False)
        alignment.to_parquet(shared / "Processed" / f"{split}_alignment.parquet", index=False)
        outputs[f"{split}_documents"] = documents
        outputs[f"{split}_sentences"] = sentences
        outputs[f"{split}_entities"] = entities
        outputs[f"{split}_tokens"] = tokens
        outputs[f"{split}_alignment"] = alignment
        all_entities.append(entities)
    lexicon = build_training_lexicon(pd.concat(all_entities, ignore_index=True))
    lexicon.to_parquet(shared / "Processed" / "training_lexicon.parquet", index=False)
    (shared / "Processed" / "feature_dictionary.json").write_text(
        json.dumps(feature_dictionary, indent=2), encoding="utf-8"
    )
    outputs["training_lexicon"] = lexicon
    return outputs


def load_processed(project_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shared = shared_root(project_root)
    sentences = pd.read_parquet(shared / "Processed" / f"{split}_sentences.parquet")
    entities = pd.read_parquet(shared / "Processed" / f"{split}_entities.parquet")
    tokens = pd.read_parquet(shared / "Processed" / f"{split}_tokens.parquet")
    return sentences, entities, tokens


def tags_to_spans(
    sentence_tokens: pd.DataFrame,
    tags: Sequence[str],
    probabilities: Sequence[float] | None = None,
) -> list[dict]:
    assert len(sentence_tokens) == len(tags)
    tokens = sentence_tokens.sort_values("token_index").reset_index(drop=True)
    probabilities = list(probabilities) if probabilities is not None else [1.0] * len(tags)
    spans: list[dict] = []
    current: list[int] = []
    current_type: str | None = None

    def close_span() -> None:
        nonlocal current, current_type
        if not current:
            return
        start_index, end_index = current[0], current[-1]
        row = tokens.iloc[start_index]
        spans.append(
            {
                "document_id": row.document_id,
                "sentence_id": row.sentence_id,
                "char_start": int(tokens.iloc[start_index].char_start),
                "char_end": int(tokens.iloc[end_index].char_end),
                "token_start": int(tokens.iloc[start_index].token_index),
                "token_end": int(tokens.iloc[end_index].token_index) + 1,
                "predicted_type": current_type or "drug",
                "boundary_probability": float(np.mean([probabilities[index] for index in current])),
                "type_probability": float(np.mean([probabilities[index] for index in current])),
            }
        )
        current = []
        current_type = None

    for index, raw_tag in enumerate(tags):
        if raw_tag == "O":
            close_span()
            continue
        if "-" in raw_tag:
            prefix, entity_type = raw_tag.split("-", 1)
        else:
            prefix, entity_type = raw_tag, "drug"
        if prefix == "S":
            close_span()
            current, current_type = [index], entity_type
            close_span()
        elif prefix == "B":
            close_span()
            current, current_type = [index], entity_type
        elif prefix == "I" and current and entity_type == current_type:
            current.append(index)
        elif prefix == "E" and current and entity_type == current_type:
            current.append(index)
            close_span()
        else:
            close_span()
            current, current_type = [index], entity_type
            if prefix == "E":
                close_span()
    close_span()
    return spans


def finalize_prediction_frame(
    spans: list[dict],
    sentences: pd.DataFrame,
    model: str,
    split: str,
    seed: int | None = None,
) -> pd.DataFrame:
    if not spans:
        columns = list(PREDICTION_COLUMNS) + ["model", "split", "seed"]
        return pd.DataFrame(columns=columns)
    sentence_text = sentences.set_index("sentence_id")["text"].to_dict()
    frame = pd.DataFrame(spans).sort_values(["document_id", "sentence_id", "char_start", "char_end"])
    frame["entity_text"] = frame.apply(
        lambda row: sentence_text[row.sentence_id][int(row.char_start) : int(row.char_end)], axis=1
    )
    local_index = frame.groupby(["document_id", "sentence_id"]).cumcount()
    frame["entity_id"] = [
        f"{sentence_id}.pred-e{value}"
        for sentence_id, value in zip(frame["sentence_id"].astype(str), local_index)
    ]
    frame["calibrated_confidence"] = np.sqrt(
        frame["boundary_probability"].clip(0, 1) * frame["type_probability"].clip(0, 1)
    )
    frame["confidence_category"] = "unassigned"
    frame["review_required"] = False
    frame["model"] = model
    frame["split"] = split
    frame["seed"] = seed
    return frame[list(PREDICTION_COLUMNS) + ["model", "split", "seed"]]


def predictions_from_tag_sequences(
    token_frame: pd.DataFrame,
    sentence_frame: pd.DataFrame,
    tag_sequences: dict[str, Sequence[str]],
    probability_sequences: dict[str, Sequence[float]] | None,
    model: str,
    split: str,
    seed: int | None = None,
) -> pd.DataFrame:
    spans: list[dict] = []
    for sentence_id, group in token_frame.groupby("sentence_id", sort=False):
        probabilities = probability_sequences.get(sentence_id) if probability_sequences else None
        spans.extend(tags_to_spans(group, tag_sequences[sentence_id], probabilities))
    return finalize_prediction_frame(spans, sentence_frame, model, split, seed)


def gold_for_metrics(entities: pd.DataFrame) -> pd.DataFrame:
    return entities.query("not discontinuous and offset_valid").copy()


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_predictions(
    gold_entities: pd.DataFrame,
    predictions: pd.DataFrame,
    model: str,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = gold_for_metrics(gold_entities)
    boundary_gold = set(zip(gold.sentence_id, gold.char_start, gold.char_end))
    boundary_pred = set(zip(predictions.sentence_id, predictions.char_start, predictions.char_end))
    typed_gold = set(zip(gold.sentence_id, gold.char_start, gold.char_end, gold.entity_type))
    typed_pred = set(
        zip(predictions.sentence_id, predictions.char_start, predictions.char_end, predictions.predicted_type)
    )
    boundary_tp = len(boundary_gold & boundary_pred)
    typed_tp = len(typed_gold & typed_pred)
    boundary_p = _safe_ratio(boundary_tp, len(boundary_pred))
    boundary_r = _safe_ratio(boundary_tp, len(boundary_gold))
    typed_p = _safe_ratio(typed_tp, len(typed_pred))
    typed_r = _safe_ratio(typed_tp, len(typed_gold))
    boundary_f1 = _safe_ratio(2 * boundary_p * boundary_r, boundary_p + boundary_r)
    typed_f1 = _safe_ratio(2 * typed_p * typed_r, typed_p + typed_r)
    boundary_errors = boundary_error_profile(gold, predictions)
    class_rows = []
    for entity_type in ENTITY_TYPES:
        class_gold = {value for value in typed_gold if value[3] == entity_type}
        class_pred = {value for value in typed_pred if value[3] == entity_type}
        class_tp = len(class_gold & class_pred)
        precision = _safe_ratio(class_tp, len(class_pred))
        recall = _safe_ratio(class_tp, len(class_gold))
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        class_rows.append(
            {
                "Model": model,
                "Split": split,
                "N": len(class_gold),
                "Entity Type": entity_type,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )
    class_frame = pd.DataFrame(class_rows)
    macro_f1 = class_frame["F1"].mean()
    metric_frame = pd.DataFrame(
        [
            {
                "Model": model,
                "Split": split,
                "N": len(gold),
                "Strict Entity Precision": typed_p,
                "Strict Entity Recall": typed_r,
                "Strict Boundary F1": boundary_f1,
                "Strict Typed Entity F1": typed_f1,
                "Macro-F1": macro_f1,
                "False Positive Rate": _safe_ratio(len(typed_pred - typed_gold), len(typed_pred)),
                "False Negative Rate": _safe_ratio(len(typed_gold - typed_pred), len(typed_gold)),
                "Boundary Error Rate": 1.0 - boundary_f1,
            }
        ]
    )
    return metric_frame, class_frame


def boundary_error_profile(gold: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    gold_by_sentence = {key: group for key, group in gold.groupby("sentence_id")}
    counts = Counter()
    matched_gold: set[tuple[str, int, int]] = set()
    for prediction in predictions.itertuples(index=False):
        candidates = gold_by_sentence.get(prediction.sentence_id)
        if candidates is None or candidates.empty:
            counts["Spurious"] += 1
            continue
        overlaps = candidates.assign(
            overlap=candidates.apply(
                lambda row: max(
                    0,
                    min(int(prediction.char_end), int(row.char_end))
                    - max(int(prediction.char_start), int(row.char_start)),
                ),
                axis=1,
            )
        )
        target = overlaps.sort_values(["overlap", "char_start"], ascending=[False, True]).iloc[0]
        if target.overlap <= 0:
            counts["Spurious"] += 1
            continue
        key = (prediction.sentence_id, int(target.char_start), int(target.char_end))
        matched_gold.add(key)
        if prediction.char_start == target.char_start and prediction.char_end == target.char_end:
            counts["Exact"] += 1
        elif prediction.char_start <= target.char_start and prediction.char_end >= target.char_end:
            counts["Overlong"] += 1
        elif prediction.char_start >= target.char_start and prediction.char_end <= target.char_end:
            counts["Underlong"] += 1
        elif prediction.char_start < target.char_start:
            counts["Left shift"] += 1
        else:
            counts["Right shift"] += 1
    gold_keys = set(zip(gold.sentence_id, gold.char_start, gold.char_end))
    counts["Missed"] = len(gold_keys - matched_gold)
    order = ["Exact", "Overlong", "Underlong", "Left shift", "Right shift", "Spurious", "Missed"]
    return pd.DataFrame({"Error": order, "Count": [counts[value] for value in order]})


def calibration_rows(predictions: pd.DataFrame, gold_entities: pd.DataFrame) -> pd.DataFrame:
    gold = gold_for_metrics(gold_entities)
    typed_gold = set(zip(gold.sentence_id, gold.char_start, gold.char_end, gold.entity_type))
    rows = predictions.copy()
    rows["correct"] = [
        int(value in typed_gold)
        for value in zip(rows.sentence_id, rows.char_start, rows.char_end, rows.predicted_type)
    ]
    return rows


def expected_calibration_error(
    confidence: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> float:
    confidence_array = np.asarray(confidence, dtype=float)
    outcome_array = np.asarray(outcomes, dtype=float)
    if len(confidence_array) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(confidence_array, edges[1:-1], right=True), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if mask.any():
            ece += mask.mean() * abs(confidence_array[mask].mean() - outcome_array[mask].mean())
    return float(ece)


def calibration_metrics(
    predictions: pd.DataFrame,
    gold_entities: pd.DataFrame,
    model: str,
    split: str,
) -> pd.DataFrame:
    rows = calibration_rows(predictions, gold_entities)
    if rows.empty:
        return pd.DataFrame(
            [{"Model": model, "Split": split, "N": 0, "ECE": np.nan, "Brier Score": np.nan}]
        )
    confidence = rows["calibrated_confidence"].astype(float).clip(0, 1)
    outcomes = rows["correct"].astype(int)
    return pd.DataFrame(
        [
            {
                "Model": model,
                "Split": split,
                "N": len(rows),
                "ECE": expected_calibration_error(confidence, outcomes),
                "Brier Score": float(np.mean((confidence - outcomes) ** 2)),
            }
        ]
    )


def subgroup_metrics(
    predictions: pd.DataFrame,
    entities: pd.DataFrame,
    sentences: pd.DataFrame,
    train_entities: pd.DataFrame,
    model: str,
    split: str,
) -> pd.DataFrame:
    sentence_meta = sentences.set_index("sentence_id")[["source"]]
    gold = gold_for_metrics(entities).copy()
    gold["normalized_surface"] = gold["entity_text"].map(normalize_surface)
    seen_surfaces = set(train_entities.query("not discontinuous")["entity_text"].map(normalize_surface))
    groups = {
        "DrugBank": set(sentence_meta.query("source == 'DrugBank'").index),
        "MedLine": set(sentence_meta.query("source == 'MedLine'").index),
        "Unseen surface": set(gold.loc[~gold.normalized_surface.isin(seen_surfaces), "sentence_id"]),
        "Multi-token": set(
            gold.loc[gold.entity_text.str.contains(r"\s|[-/]", regex=True), "sentence_id"]
        ),
    }
    rows = []
    for label, sentence_ids in groups.items():
        group_gold = gold[gold.sentence_id.isin(sentence_ids)]
        group_pred = predictions[predictions.sentence_id.isin(sentence_ids)]
        metric, _ = evaluate_predictions(group_gold, group_pred, model, split)
        rows.append(
            {
                "Model": model,
                "Split": split,
                "N": len(group_gold),
                "Subgroup": label,
                "Strict Boundary F1": metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metric.iloc[0]["Strict Typed Entity F1"],
            }
        )
    return pd.DataFrame(rows)


def document_level_scores(
    predictions: pd.DataFrame,
    entities: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    rows = []
    document_by_sentence = entities[["sentence_id", "document_id"]].drop_duplicates().set_index("sentence_id")
    predicted = predictions.copy()
    if not predicted.empty:
        predicted["document_id"] = predicted["sentence_id"].map(document_by_sentence["document_id"])
    for document_id, gold_group in gold_for_metrics(entities).groupby("document_id"):
        prediction_group = predicted[predicted.document_id == document_id] if not predicted.empty else predicted
        metric, _ = evaluate_predictions(gold_group, prediction_group, model, "document")
        rows.append(
            {
                "document_id": document_id,
                "Model": model,
                "Strict Boundary F1": metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metric.iloc[0]["Strict Typed Entity F1"],
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_ci(
    predictions: pd.DataFrame,
    entities: pd.DataFrame,
    model: str,
    seed: int,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    documents = sorted(entities.document_id.unique())
    gold = gold_for_metrics(entities)
    rng = np.random.default_rng(seed)
    document_counts = []
    for document_id in documents:
        gold_part = gold[gold.document_id == document_id]
        pred_part = predictions[predictions.document_id == document_id]
        gold_set = set(
            zip(
                gold_part.sentence_id,
                gold_part.char_start,
                gold_part.char_end,
                gold_part.entity_type,
            )
        )
        pred_set = set(
            zip(
                pred_part.sentence_id,
                pred_part.char_start,
                pred_part.char_end,
                pred_part.predicted_type,
            )
        )
        document_counts.append((len(gold_set & pred_set), len(pred_set), len(gold_set)))
    counts = np.asarray(document_counts, dtype=np.int64)
    sampled_indices = rng.integers(
        0, len(documents), size=(n_bootstrap, len(documents)), endpoint=False
    )
    totals = counts[sampled_indices].sum(axis=1)
    denominators = totals[:, 1] + totals[:, 2]
    scores = np.divide(
        2.0 * totals[:, 0],
        denominators,
        out=np.zeros(n_bootstrap, dtype=float),
        where=denominators > 0,
    )
    point_denominator = int(counts[:, 1].sum() + counts[:, 2].sum())
    point_estimate = (
        float(2 * counts[:, 0].sum() / point_denominator) if point_denominator else 0.0
    )
    return pd.DataFrame(
        [
            {
                "Model": model,
                "Metric": "Strict Typed Entity F1",
                "Point Estimate": point_estimate,
                "CI Lower": float(np.quantile(scores, 0.025)),
                "CI Upper": float(np.quantile(scores, 0.975)),
                "Bootstrap Unit": "document",
                "Resamples": n_bootstrap,
            }
        ]
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    n_values = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (n_values - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_statistical_comparison(
    act_predictions: pd.DataFrame,
    comparator_predictions: dict[str, pd.DataFrame],
    entities: pd.DataFrame,
) -> pd.DataFrame:
    act_scores = document_level_scores(act_predictions, entities, "ACT-NER").set_index("document_id")
    rows = []
    for comparator, predictions in comparator_predictions.items():
        other_scores = document_level_scores(predictions, entities, comparator).set_index("document_id")
        paired = act_scores[["Strict Typed Entity F1"]].join(
            other_scores[["Strict Typed Entity F1"]], lsuffix="_act", rsuffix="_other", how="inner"
        )
        differences = paired.iloc[:, 0] - paired.iloc[:, 1]
        nonzero = differences[np.abs(differences) > 1e-12]
        if len(nonzero) >= 10:
            statistic, p_value = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        else:
            statistic, p_value = np.nan, 1.0
        denominator = differences.std(ddof=1)
        effect = differences.mean() / denominator if denominator > 0 else 0.0
        positive = int((nonzero > 0).sum())
        negative = int((nonzero < 0).sum())
        rank_biserial = _safe_ratio(positive - negative, positive + negative)
        rows.append(
            {
                "Comparison": f"ACT-NER vs {comparator}",
                "N Documents": len(paired),
                "Mean Paired Difference": differences.mean(),
                "Wilcoxon Statistic": statistic,
                "Raw p-value": p_value,
                "Cohen dz": effect,
                "Rank-biserial Effect": rank_biserial,
            }
        )
    frame = pd.DataFrame(rows)
    frame["Holm-adjusted p-value"] = holm_adjust(frame["Raw p-value"])
    frame["Holm Significant"] = frame["Holm-adjusted p-value"] < 0.05
    return frame


def stable_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class EfficiencyRecord:
    model: str
    split: str
    n_sentences: int
    n_entities: int
    elapsed_seconds: float
    parameter_count: int
    model_size_mb: float
    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Model": self.model,
                    "Split": self.split,
                    "N": self.n_sentences,
                    "Latency per Sentence (ms)": 1000 * self.elapsed_seconds / max(self.n_sentences, 1),
                    "Latency per Entity (ms)": 1000 * self.elapsed_seconds / max(self.n_entities, 1),
                    "Throughput (sentences/s)": self.n_sentences / max(self.elapsed_seconds, 1e-12),
                    "Parameter Count": self.parameter_count,
                    "Model Size (MB)": self.model_size_mb,
                    "Peak RAM (MB)": self.peak_ram_mb,
                    "Peak VRAM (MB)": self.peak_vram_mb,
                }
            ]
        )
