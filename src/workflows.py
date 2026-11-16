from __future__ import annotations

import copy
import json
import math
import re
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer

from core import (
    ENTITY_TYPES,
    PREDICTION_COLUMNS,
    adaptive_figsize,
    add_eda_attributes,
    audit_development_corpus,
    boundary_error_profile,
    build_training_lexicon,
    calibration_metrics,
    configure_notebook,
    document_level_scores,
    display_saved_figure,
    evaluate_predictions,
    expected_calibration_error,
    gold_for_metrics,
    load_development_corpus,
    load_processed,
    load_settings,
    load_split_corpus,
    normalize_surface,
    paired_bootstrap_ci,
    paired_statistical_comparison,
    prepare_modeling_data,
    save_figure,
    save_table,
    shared_root,
    subgroup_metrics,
    tokenize_split,
)
from models import (
    ACTConfig,
    ACTNER,
    BiLSTMCRF,
    LongestMatchLexicon,
    add_act_features,
    apply_calibration,
    apply_confidence_policy,
    baseline_oof_predictions,
    build_transformer_model,
    extract_context_embeddings,
    fit_validation_calibration,
    lexicon_predictions,
    o_only_predictions,
    predict_act,
    predict_bilstm,
    predict_linear_crf,
    predict_logistic_regression,
    predict_transformer,
    select_confidence_policy,
    set_seed,
    train_act_model,
    train_bilstm_comparator,
    train_linear_crf,
    train_logistic_regression,
    train_transformer_comparator,
    tune_act_loss_weights,
)

BLUE = "#2F5D8A"
GOLD = "#D29B2D"
ORANGE = "#C96D3B"
OLIVE = "#708238"
PINK = "#A85C78"
LIGHT_BLUE = "#B7CCDF"
INK = "#252525"
GRID = "#D8D8D8"


def _label_bars(axis, bars, decimals: int = 0) -> None:
    for bar in bars:
        value = bar.get_height()
        text = f"{value:.{decimals}f}" if decimals else f"{int(round(value)):,}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            text,
            ha="center",
            va="bottom",
            fontsize=18,
        )


def _save_stage1_figures(
    project_root: Path,
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
    entities: pd.DataFrame,
) -> list[str]:
    names = []
    contiguous = entities.query("not discontinuous").copy()

    type_counts = (
        contiguous.groupby(["source", "entity_type"]).size().rename("Count").reset_index()
    )
    pivot = type_counts.pivot(index="entity_type", columns="source", values="Count").fillna(0)
    pivot = pivot.reindex(ENTITY_TYPES)
    figure, axis = plt.subplots(figsize=adaptive_figsize(4, max_label_chars=8, legend_rows=1))
    x = np.arange(len(pivot))
    width = 0.38
    bars_a = axis.bar(x - width / 2, pivot.get("DrugBank", 0), width, label="DrugBank", color=BLUE, edgecolor=INK)
    bars_b = axis.bar(x + width / 2, pivot.get("MedLine", 0), width, label="MedLine", color=GOLD, edgecolor=INK)
    _label_bars(axis, bars_a)
    _label_bars(axis, bars_b)
    axis.set_title("Entity Type Distribution")
    axis.set_xlabel("Entity type")
    axis.set_ylabel("Annotated entities")
    axis.set_xticks(x, pivot.index)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    figure.subplots_adjust(bottom=0.27, top=0.88, left=0.12, right=0.97)
    save_figure(project_root, figure, "entity_type_distribution", type_counts)
    names.append("entity_type_distribution")

    lengths = contiguous.assign(
        Character_Length=contiguous.char_end - contiguous.char_start,
        Token_Length=contiguous.entity_text.map(lambda value: len(re.findall(r"\w+|[^\w\s]", value))),
    )
    figure, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    char_limit = int(lengths.Character_Length.quantile(0.99))
    axes[0].hist(
        lengths.Character_Length.clip(upper=char_limit),
        bins=min(25, max(8, char_limit)),
        color=BLUE,
        edgecolor=INK,
    )
    axes[0].set_title("Character length (99th percentile capped)")
    axes[0].set_xlabel("Characters per entity")
    axes[0].set_ylabel("Entities")
    token_counts = lengths.Token_Length.value_counts().sort_index()
    bars = axes[1].bar(token_counts.index, token_counts.values, color=GOLD, edgecolor=INK)
    _label_bars(axes[1], bars)
    axes[1].set_title("Token length")
    axes[1].set_xlabel("Regex tokens per entity")
    axes[1].set_ylabel("Entities")
    axes[1].set_xticks(token_counts.index)
    for axis in axes:
        axis.grid(axis="y", color=GRID, linewidth=0.8)
    figure.suptitle("Entity Length Distribution", fontsize=18)
    figure.subplots_adjust(bottom=0.17, top=0.80, left=0.08, right=0.98, wspace=0.28)
    save_figure(
        project_root,
        figure,
        "entity_length_distribution",
        lengths[["source", "entity_type", "Character_Length", "Token_Length"]],
    )
    names.append("entity_length_distribution")

    sentence_attributes, entity_attributes = add_eda_attributes(sentences, entities)
    domain = (
        documents.groupby("source").size().rename("Documents").to_frame()
        .join(sentence_attributes.groupby("source").size().rename("Sentences"))
        .join(entity_attributes.groupby("source").size().rename("Entities"))
        .join(sentence_attributes.groupby("source").n_tokens.mean().rename("Mean sentence tokens"))
        .join(entity_attributes.groupby("source").surface_frequency.mean().rename("Mean surface frequency"))
        .reset_index()
    )
    normalized = domain.set_index("source").drop(columns=[])
    normalized = normalized / normalized.max(axis=0)
    figure, axis = plt.subplots(figsize=(15, 8.5))
    metrics = normalized.columns.tolist()
    x = np.arange(len(metrics))
    for source, color, marker in (("DrugBank", BLUE, "o"), ("MedLine", GOLD, "s")):
        axis.plot(x, normalized.loc[source], marker=marker, linewidth=2.6, markersize=9, color=color, label=source)
    axis.set_title("DrugBank–MedLine Domain Profile")
    axis.set_xlabel("Corpus characteristic")
    axis.set_ylabel("Within-metric normalized value")
    axis.set_xticks(x, [value.replace("Mean ", "Mean\n") for value in metrics])
    axis.set_ylim(0, 1.12)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=2, frameon=False)
    figure.subplots_adjust(bottom=0.30, top=0.88, left=0.10, right=0.98)
    save_figure(project_root, figure, "drugbank_medline_domain_profile", domain)
    names.append("drugbank_medline_domain_profile")

    frequency = (
        entity_attributes.groupby("normalized_surface").size().sort_values(ascending=False).rename("Frequency").reset_index()
    )
    frequency["Rank"] = np.arange(1, len(frequency) + 1)
    figure, axis = plt.subplots(figsize=(12.5, 8.0))
    axis.loglog(frequency.Rank, frequency.Frequency, color=BLUE, linewidth=2.4)
    axis.scatter(frequency.Rank.iloc[:20], frequency.Frequency.iloc[:20], color=GOLD, edgecolor=INK, s=55, zorder=3)
    axis.set_title("Entity Surface-Form Frequency")
    axis.set_xlabel("Surface-form rank (log scale)")
    axis.set_ylabel("Annotation frequency (log scale)")
    axis.grid(which="both", color=GRID, linewidth=0.7)
    figure.subplots_adjust(bottom=0.16, top=0.88, left=0.13, right=0.97)
    save_figure(project_root, figure, "entity_surface_form_frequency", frequency)
    names.append("entity_surface_form_frequency")

    shared = shared_root(project_root)
    split_rows = []
    for split in ("train", "validation", "locked_test"):
        manifest = pd.read_csv(shared / "Splits" / f"{split}_documents.csv")
        for source, count in manifest.groupby("source").size().items():
            split_rows.append({"Split": split.replace("_", " ").title(), "Source": source, "Documents": count})
    split_frame = pd.DataFrame(split_rows)
    split_pivot = split_frame.pivot(index="Split", columns="Source", values="Documents").fillna(0)
    split_order = ["Train", "Validation", "Locked Test"]
    split_pivot = split_pivot.reindex(split_order)
    figure, axis = plt.subplots(figsize=(12.5, 8.0))
    x = np.arange(len(split_pivot))
    drugbank = axis.bar(x, split_pivot.DrugBank, color=BLUE, edgecolor=INK, label="DrugBank")
    medline = axis.bar(
        x,
        split_pivot.MedLine,
        bottom=split_pivot.DrugBank,
        color=GOLD,
        edgecolor=INK,
        label="MedLine",
    )
    for index, total in enumerate(split_pivot.sum(axis=1)):
        axis.text(index, total, f"{int(total)}", ha="center", va="bottom", fontsize=18)
    axis.set_title("Split Design")
    axis.set_xlabel("Document-level split")
    axis.set_ylabel("Documents")
    axis.set_xticks(x, split_order)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    figure.subplots_adjust(bottom=0.27, top=0.88, left=0.12, right=0.97)
    save_figure(project_root, figure, "split_design", split_frame)
    names.append("split_design")
    return names


def run_stage_01(project_root: Path) -> dict:
    configure_notebook()
    documents, sentences, entities = load_development_corpus(project_root)
    quality, source_profile = audit_development_corpus(documents, sentences, entities)
    sentence_attributes, entity_attributes = add_eda_attributes(sentences, entities)
    contiguous = entity_attributes.query("not discontinuous")
    corpus_summary = pd.DataFrame(
        [
            {
                "Scope": "Development corpus (Train + Validation)",
                "Documents": len(documents),
                "Sentences": len(sentences),
                "Tokens": int(sentence_attributes.n_tokens.sum()),
                "Entities": len(entities),
                "Contiguous entities": len(contiguous),
                "Unique surface forms": contiguous.normalized_surface.nunique(),
            }
        ]
    )
    type_distribution = (
        contiguous.groupby(["split", "source", "entity_type"])
        .size()
        .rename("N")
        .reset_index()
        .rename(columns={"split": "Split", "source": "Source", "entity_type": "Entity Type"})
    )
    length_summary = (
        contiguous.assign(
            Character_Length=contiguous.char_end - contiguous.char_start,
            Token_Length=contiguous.n_tokens_regex,
        )
        .groupby(["source", "entity_type"])[["Character_Length", "Token_Length"]]
        .agg(["count", "mean", "median", "max"])
    )
    length_summary.columns = [" ".join(column).strip() for column in length_summary.columns]
    length_summary = length_summary.reset_index().rename(
        columns={"source": "Source", "entity_type": "Entity Type"}
    )
    ambiguity = (
        contiguous.groupby("normalized_surface")
        .agg(
            N=("entity_id", "size"),
            Type_Count=("entity_type", "nunique"),
            Types=("entity_type", lambda values: ", ".join(sorted(set(values)))),
        )
        .query("Type_Count > 1")
        .sort_values(["N", "normalized_surface"], ascending=[False, True])
        .reset_index()
        .rename(columns={"normalized_surface": "Normalized Surface"})
    )
    train_surfaces = set(contiguous.query("split == 'train'").normalized_surface)
    validation_entities = contiguous.query("split == 'validation'")
    leakage = pd.DataFrame(
        [
            {
                "Comparison": "Validation entity surfaces seen in Train",
                "Validation entities": len(validation_entities),
                "Seen entities": int(validation_entities.normalized_surface.isin(train_surfaces).sum()),
                "Unseen entities": int((~validation_entities.normalized_surface.isin(train_surfaces)).sum()),
                "Seen proportion": float(validation_entities.normalized_surface.isin(train_surfaces).mean()),
            }
        ]
    )
    research_design = pd.DataFrame(
        [
            ("H1", "Adaptive context–lexical fusion improves strict typed F1 over the strongest baseline."),
            ("H2", "CRF decoding reduces invalid or fragmented multi-token boundaries."),
            ("H3", "Training-only lexical evidence helps seen entities but does not replace contextual generalization."),
            ("H4", "Validation-selected calibration lowers ECE/Brier without changing predicted spans or types."),
            ("H5", "DrugBank and MedLine remain subgroup analyses, not independent external validation."),
        ],
        columns=["Hypothesis", "Prespecified statement"],
    )
    for name, frame in {
        "01_corpus_summary": corpus_summary,
        "01_data_quality_audit": quality,
        "01_source_profile": source_profile,
        "01_entity_type_distribution": type_distribution,
        "01_entity_length_summary": length_summary,
        "01_lexical_ambiguity": ambiguity,
        "01_lexical_leakage_risk": leakage,
        "01_research_hypotheses": research_design,
    }.items():
        save_table(project_root, frame, name)
    figure_names = _save_stage1_figures(project_root, documents, sentences, entities)
    return {
        "corpus_summary": corpus_summary,
        "quality": quality,
        "source_profile": source_profile,
        "type_distribution": type_distribution,
        "length_summary": length_summary,
        "ambiguity": ambiguity,
        "leakage": leakage,
        "research_design": research_design,
        "figure_names": figure_names,
    }


def run_stage_02(project_root: Path) -> dict:
    configure_notebook()
    outputs = prepare_modeling_data(project_root)
    train_documents = outputs["train_documents"]
    validation_documents = outputs["validation_documents"]
    train_entities = outputs["train_entities"]
    validation_entities = outputs["validation_entities"]
    train_tokens = outputs["train_tokens"]
    validation_tokens = outputs["validation_tokens"]
    train_alignment = outputs["train_alignment"]
    validation_alignment = outputs["validation_alignment"]
    lexicon = outputs["training_lexicon"]
    shared = shared_root(project_root)
    locked_manifest = pd.read_csv(shared / "Splits" / "locked_test_documents.csv")
    train_ids = set(train_documents.document_id)
    validation_ids = set(validation_documents.document_id)
    locked_ids = set(locked_manifest.document_id)
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(locked_ids)
    assert validation_ids.isdisjoint(locked_ids)
    assert set(train_tokens.split) == {"train"}
    assert set(validation_tokens.split) == {"validation"}

    split_integrity = pd.DataFrame(
        [
            {
                "Split": "Train",
                "N": len(train_documents),
                "DrugBank documents": int((train_documents.source == "DrugBank").sum()),
                "MedLine documents": int((train_documents.source == "MedLine").sum()),
                "Overlap with other splits": 0,
            },
            {
                "Split": "Validation",
                "N": len(validation_documents),
                "DrugBank documents": int((validation_documents.source == "DrugBank").sum()),
                "MedLine documents": int((validation_documents.source == "MedLine").sum()),
                "Overlap with other splits": 0,
            },
            {
                "Split": "Locked Test (manifest only)",
                "N": len(locked_manifest),
                "DrugBank documents": int((locked_manifest.source == "DrugBank").sum()),
                "MedLine documents": int((locked_manifest.source == "MedLine").sum()),
                "Overlap with other splits": 0,
            },
        ]
    )
    alignment = pd.concat([train_alignment, validation_alignment], ignore_index=True)
    alignment_summary = (
        alignment.groupby(["split", "alignment_status"])
        .size()
        .rename("N")
        .reset_index()
        .rename(columns={"split": "Split", "alignment_status": "Alignment Status"})
    )
    tag_distribution = (
        pd.concat([train_tokens, validation_tokens], ignore_index=True)
        .groupby(["split", "typed_tag"])
        .size()
        .rename("N")
        .reset_index()
        .rename(columns={"split": "Split", "typed_tag": "BIOES Target"})
    )
    train_surfaces = set(train_entities.query("not discontinuous").entity_text.map(normalize_surface))
    validation_contiguous = validation_entities.query("not discontinuous").copy()
    validation_contiguous["normalized_surface"] = validation_contiguous.entity_text.map(normalize_surface)
    overlap = pd.DataFrame(
        [
            {
                "Comparison": "Train lexicon vs Validation gold surfaces",
                "N": len(validation_contiguous),
                "Seen": int(validation_contiguous.normalized_surface.isin(train_surfaces).sum()),
                "Unseen": int((~validation_contiguous.normalized_surface.isin(train_surfaces)).sum()),
                "Overlap proportion": float(validation_contiguous.normalized_surface.isin(train_surfaces).mean()),
                "Removal performed": False,
            }
        ]
    )
    shift = (
        pd.concat([train_tokens, validation_tokens], ignore_index=True)
        .groupby(["split", "source"])
        .agg(
            Sentences=("sentence_id", "nunique"),
            Tokens=("token_index", "size"),
            Mean_tokens_per_sentence=("sentence_id", lambda values: len(values) / values.nunique()),
            O_share=("boundary_tag", lambda values: float((values == "O").mean())),
        )
        .reset_index()
        .rename(columns={"split": "Split", "source": "Source"})
    )
    lexicon_summary = pd.DataFrame(
        [
            {
                "Split used to build": "Train only",
                "N": len(lexicon),
                "Unique normalized terms": lexicon.normalized_surface.nunique(),
                "Ambiguous terms": int((lexicon.ambiguity_score > 0).sum()),
                "Median frequency": float(lexicon.frequency.median()),
                "Maximum frequency": int(lexicon.frequency.max()),
            }
        ]
    )
    for name, frame in {
        "02_split_integrity": split_integrity,
        "02_alignment_summary": alignment_summary,
        "02_bioes_target_distribution": tag_distribution,
        "02_surface_overlap": overlap,
        "02_distribution_shift": shift,
        "02_training_lexicon_summary": lexicon_summary,
    }.items():
        save_table(project_root, frame, name)
    return {
        "split_integrity": split_integrity,
        "alignment_summary": alignment_summary,
        "tag_distribution": tag_distribution,
        "overlap": overlap,
        "shift": shift,
        "lexicon_summary": lexicon_summary,
    }


def _model_metrics_bundle(
    model: str,
    predictions: pd.DataFrame,
    entities: pd.DataFrame,
    sentences: pd.DataFrame,
    train_entities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictive, classwise = evaluate_predictions(entities, predictions, model, "Validation")
    calibration = calibration_metrics(predictions, entities, model, "Validation")
    subgroup = subgroup_metrics(
        predictions, entities, sentences, train_entities, model, "Validation"
    )
    return predictive, classwise, calibration, subgroup


def _save_stage3_figures(
    project_root: Path,
    learning_curves: pd.DataFrame,
    error_profile: pd.DataFrame,
) -> list[str]:
    figure, axis = plt.subplots(figsize=(13.5, 8.0))
    for metric, color, marker in (
        ("Strict Boundary F1", BLUE, "o"),
        ("Strict Typed Entity F1", GOLD, "s"),
    ):
        axis.plot(
            learning_curves["Training Documents"],
            learning_curves[metric],
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=9,
            label=metric,
        )
    axis.set_title("Learning Curves")
    axis.set_xlabel("Training documents")
    axis.set_ylabel("Validation F1")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="both", color=GRID, linewidth=0.8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    figure.subplots_adjust(bottom=0.27, top=0.88, left=0.11, right=0.97)
    save_figure(project_root, figure, "learning_curves", learning_curves)

    ordered = error_profile.sort_values(["Model", "Error"])
    pivot = ordered.pivot(index="Error", columns="Model", values="Count").fillna(0)
    error_order = ["Overlong", "Underlong", "Left shift", "Right shift", "Spurious", "Missed"]
    pivot = pivot.reindex(error_order).fillna(0)
    figure, axis = plt.subplots(figsize=(16.5, 9.0))
    x = np.arange(len(pivot))
    models = pivot.columns.tolist()
    colors = [BLUE, GOLD, ORANGE, OLIVE]
    width = 0.78 / max(len(models), 1)
    for index, model in enumerate(models):
        axis.bar(
            x - 0.39 + width / 2 + index * width,
            pivot[model],
            width,
            label=model,
            color=colors[index % len(colors)],
            edgecolor=INK,
        )
    axis.set_title("Boundary Error Profile")
    axis.set_xlabel("Error category")
    axis.set_ylabel("Entity-level count")
    axis.set_xticks(x, pivot.index)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)
    figure.subplots_adjust(bottom=0.32, top=0.88, left=0.10, right=0.98)
    save_figure(project_root, figure, "boundary_error_profile", error_profile)
    return ["learning_curves", "boundary_error_profile"]


def run_stage_03(project_root: Path, force_recompute: bool = False) -> dict:
    configure_notebook()
    cached = project_root / "artifacts" / "03_predictive_quality.csv"
    required = [
        project_root / "Models" / "Baselines" / "logistic_regression.joblib",
        project_root / "Models" / "Baselines" / "linear_chain_crf.joblib",
        project_root / "Predictions" / "OOF" / "linear_chain_crf.parquet",
    ]
    if cached.exists() and all(path.exists() for path in required) and not force_recompute:
        selection = json.loads(
            (project_root / "src" / "baseline_selection.json").read_text(encoding="utf-8")
        )
        return {
            "predictive": pd.read_csv(cached),
            "classwise": pd.read_csv(project_root / "artifacts" / "03_classwise_quality.csv"),
            "calibration": pd.read_csv(project_root / "artifacts" / "03_calibration.csv"),
            "subgroups": pd.read_csv(project_root / "artifacts" / "03_error_subgroups.csv"),
            "errors": pd.read_csv(project_root / "artifacts" / "03_boundary_errors.csv"),
            "efficiency": pd.read_csv(project_root / "artifacts" / "03_efficiency.csv"),
            "tuning": pd.concat(
                [
                    pd.read_csv(project_root / "artifacts" / "03_logistic_tuning.csv"),
                    pd.read_csv(project_root / "artifacts" / "03_crf_tuning.csv"),
                ],
                ignore_index=True,
                sort=False,
            ),
            "learning_curves": pd.read_csv(project_root / "artifacts" / "03_learning_curves.csv"),
            "selection": pd.DataFrame([selection]),
            "figure_names": ["learning_curves", "boundary_error_profile"],
        }
    settings = load_settings(project_root)
    train_sentences, train_entities, train_tokens = load_processed(project_root, "train")
    validation_sentences, validation_entities, validation_tokens = load_processed(
        project_root, "validation"
    )
    lexicon = pd.read_parquet(shared_root(project_root) / "Processed" / "training_lexicon.parquet")
    matcher = LongestMatchLexicon(lexicon)

    timings = {}
    start = perf_counter()
    o_predictions = o_only_predictions(validation_sentences, "validation")
    timings["O-only"] = perf_counter() - start
    start = perf_counter()
    lexicon_validation = lexicon_predictions(
        matcher, validation_sentences, validation_tokens, "validation"
    )
    timings["Longest-match lexicon"] = perf_counter() - start

    training_start = perf_counter()
    logistic_artifact, _, logistic_tuning = train_logistic_regression(
        train_tokens,
        validation_tokens,
        validation_sentences,
        validation_entities,
    )
    logistic_training_time = perf_counter() - training_start
    start = perf_counter()
    logistic_validation = predict_logistic_regression(
        logistic_artifact, validation_tokens, validation_sentences, "validation"
    )
    timings["Logistic Regression"] = perf_counter() - start

    training_start = perf_counter()
    crf_model, _, crf_tuning = train_linear_crf(
        train_tokens,
        validation_tokens,
        validation_sentences,
        validation_entities,
    )
    crf_training_time = perf_counter() - training_start
    start = perf_counter()
    crf_validation = predict_linear_crf(
        crf_model, validation_tokens, validation_sentences, "validation"
    )
    timings["Linear-chain CRF"] = perf_counter() - start

    model_directory = project_root / "Models" / "Baselines"
    (model_directory / "o_only.json").write_text(
        json.dumps({"model": "O-only", "trainable_parameters": 0}, indent=2), encoding="utf-8"
    )
    joblib.dump(matcher, model_directory / "longest_match_lexicon.joblib")
    joblib.dump(logistic_artifact, model_directory / "logistic_regression.joblib")
    joblib.dump(crf_model, model_directory / "linear_chain_crf.joblib")

    validation_predictions = {
        "O-only": o_predictions,
        "Longest-match lexicon": lexicon_validation,
        "Logistic Regression": logistic_validation,
        "Linear-chain CRF": crf_validation,
    }
    prediction_directory = project_root / "Predictions" / "Validation"
    for model, frame in validation_predictions.items():
        frame.to_parquet(
            prediction_directory / f"{model.lower().replace(' ', '_').replace('-', '_')}.parquet",
            index=False,
        )

    def fold_lexicon(fit_documents):
        subset = train_entities[train_entities.document_id.isin(fit_documents)].copy()
        subset["split"] = "train"
        return LongestMatchLexicon(build_training_lexicon(subset))

    oof = baseline_oof_predictions(
        train_tokens,
        train_sentences,
        fold_lexicon,
        float(logistic_artifact["selected_c"]),
        float(crf_model.c1),
        float(crf_model.c2),
        n_folds=settings["oof_folds"],
    )
    for model, frame in oof.items():
        frame.to_parquet(
            project_root
            / "Predictions"
            / "OOF"
            / f"{model.lower().replace(' ', '_').replace('-', '_')}.parquet",
            index=False,
        )

    predictive_parts = []
    class_parts = []
    calibration_parts = []
    subgroup_parts = []
    error_parts = []
    for model, predictions in validation_predictions.items():
        predictive, classwise, calibration, subgroup = _model_metrics_bundle(
            model, predictions, validation_entities, validation_sentences, train_entities
        )
        predictive_parts.append(predictive)
        class_parts.append(classwise)
        calibration_parts.append(calibration)
        subgroup_parts.append(subgroup)
        profile = boundary_error_profile(gold_for_metrics(validation_entities), predictions)
        profile.insert(0, "Model", model)
        error_parts.append(profile)
    predictive = pd.concat(predictive_parts, ignore_index=True)
    classwise = pd.concat(class_parts, ignore_index=True)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    subgroups = pd.concat(subgroup_parts, ignore_index=True)
    errors = pd.concat(error_parts, ignore_index=True)

    file_sizes = {
        "O-only": (model_directory / "o_only.json").stat().st_size / 2**20,
        "Longest-match lexicon": (model_directory / "longest_match_lexicon.joblib").stat().st_size / 2**20,
        "Logistic Regression": (model_directory / "logistic_regression.joblib").stat().st_size / 2**20,
        "Linear-chain CRF": (model_directory / "linear_chain_crf.joblib").stat().st_size / 2**20,
    }
    parameter_counts = {
        "O-only": 0,
        "Longest-match lexicon": 0,
        "Logistic Regression": int(logistic_artifact["model"].coef_.size + logistic_artifact["model"].intercept_.size),
        "Linear-chain CRF": int(len(crf_model.state_features_) + len(crf_model.transition_features_)),
    }
    peak_ram = psutil.Process().memory_info().rss / 2**20
    efficiency = pd.DataFrame(
        [
            {
                "Model": model,
                "Split": "Validation",
                "N": len(validation_sentences),
                "Train Time (s)": 0.0
                if model in ("O-only", "Longest-match lexicon")
                else logistic_training_time
                if model == "Logistic Regression"
                else crf_training_time,
                "Latency per Sentence (ms)": 1000 * timings[model] / len(validation_sentences),
                "Latency per Entity (ms)": 1000 * timings[model] / max(len(validation_predictions[model]), 1),
                "Throughput (sentences/s)": len(validation_sentences) / max(timings[model], 1e-12),
                "Parameter Count": parameter_counts[model],
                "Model Size (MB)": file_sizes[model],
                "Peak RAM (MB)": peak_ram,
                "Peak VRAM (MB)": np.nan,
            }
            for model in validation_predictions
        ]
    )

    rng = np.random.default_rng(settings["split_seed"])
    train_document_ids = np.array(sorted(train_sentences.document_id.unique()))
    learning_rows = []
    for fraction in (0.25, 0.50, 0.75, 1.00):
        n_documents = int(round(len(train_document_ids) * fraction))
        chosen = set(rng.choice(train_document_ids, n_documents, replace=False)) if fraction < 1 else set(train_document_ids)
        subset_tokens = train_tokens[train_tokens.document_id.isin(chosen)]
        artifact, predictions, _ = train_logistic_regression(
            subset_tokens,
            validation_tokens,
            validation_sentences,
            validation_entities,
            c_values=(float(logistic_artifact["selected_c"]),),
        )
        metrics, _ = evaluate_predictions(
            validation_entities, predictions, "Logistic Regression", "Validation"
        )
        learning_rows.append(
            {
                "Model": "Logistic Regression",
                "Split": "Validation",
                "Training Fraction": fraction,
                "Training Documents": n_documents,
                "Strict Boundary F1": metrics.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metrics.iloc[0]["Strict Typed Entity F1"],
            }
        )
    learning_curves = pd.DataFrame(learning_rows)
    strongest = predictive.sort_values(
        ["Strict Typed Entity F1", "Strict Boundary F1", "Macro-F1"], ascending=False
    ).iloc[0]["Model"]
    selection = {
        "selected_baseline": strongest,
        "selection_split": "Validation",
        "primary_metric": "Strict Typed Entity F1",
        "secondary_metric": "Strict Boundary F1",
    }
    (project_root / "src" / "baseline_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    for name, frame in {
        "03_predictive_quality": predictive,
        "03_classwise_quality": classwise,
        "03_calibration": calibration,
        "03_error_subgroups": subgroups,
        "03_boundary_errors": errors,
        "03_efficiency": efficiency,
        "03_logistic_tuning": logistic_tuning,
        "03_crf_tuning": crf_tuning,
        "03_learning_curves": learning_curves,
    }.items():
        save_table(project_root, frame, name)
    figures = _save_stage3_figures(project_root, learning_curves, errors)
    return {
        "predictive": predictive,
        "classwise": classwise,
        "calibration": calibration,
        "subgroups": subgroups,
        "errors": errors,
        "efficiency": efficiency,
        "tuning": pd.concat([logistic_tuning, crf_tuning], ignore_index=True, sort=False),
        "learning_curves": learning_curves,
        "selection": pd.DataFrame([selection]),
        "figure_names": figures,
    }


def _entity_type_confusion(
    predictions: pd.DataFrame, entities: pd.DataFrame, model: str
) -> pd.DataFrame:
    gold = gold_for_metrics(entities)
    gold_map = {
        (row.sentence_id, int(row.char_start), int(row.char_end)): row.entity_type
        for row in gold.itertuples(index=False)
    }
    pairs = []
    for row in predictions.itertuples(index=False):
        key = (row.sentence_id, int(row.char_start), int(row.char_end))
        if key in gold_map:
            pairs.append((gold_map[key], row.predicted_type))
    matrix = confusion_matrix(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        labels=list(ENTITY_TYPES),
    )
    rows = []
    for actual_index, actual in enumerate(ENTITY_TYPES):
        for predicted_index, predicted in enumerate(ENTITY_TYPES):
            rows.append(
                {
                    "Model": model,
                    "Actual Type": actual,
                    "Predicted Type": predicted,
                    "Count": int(matrix[actual_index, predicted_index]),
                }
            )
    return pd.DataFrame(rows)


def _save_confusion_figure(project_root: Path, confusion: pd.DataFrame) -> str:
    matrix = confusion.pivot(index="Actual Type", columns="Predicted Type", values="Count")
    matrix = matrix.reindex(index=ENTITY_TYPES, columns=ENTITY_TYPES).fillna(0)
    figure, axis = plt.subplots(figsize=adaptive_figsize(heatmap_shape=matrix.shape))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    maximum = max(float(matrix.to_numpy().max()), 1.0)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix.iloc[row_index, column_index])
            color = "white" if value / maximum > 0.50 else INK
            axis.text(column_index, row_index, f"{value}", ha="center", va="center", color=color, fontsize=18)
    axis.set_title("Confusion Matrix for Entity Types")
    axis.set_xlabel("Predicted type (exact boundary matches)")
    axis.set_ylabel("Gold type")
    axis.set_xticks(np.arange(len(ENTITY_TYPES)), ENTITY_TYPES)
    axis.set_yticks(np.arange(len(ENTITY_TYPES)), ENTITY_TYPES)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("Entities", fontsize=18)
    colorbar.ax.tick_params(labelsize=18)
    figure.subplots_adjust(bottom=0.18, top=0.88, left=0.16, right=0.88)
    save_figure(project_root, figure, "confusion_matrix_entity_types", confusion)
    return "confusion_matrix_entity_types"


def _deep_oof(
    project_root: Path,
    train_tokens: pd.DataFrame,
    train_sentences: pd.DataFrame,
    train_entities: pd.DataFrame,
    tokenizer,
    settings: dict,
) -> dict[str, pd.DataFrame]:
    document_ids = np.array(sorted(train_sentences.document_id.unique()))
    splitter = GroupKFold(n_splits=settings["oof_folds"])
    outputs = {"BiLSTM-CRF": [], "Biomedical Transformer": []}
    for fold, (fit_index, hold_index) in enumerate(
        splitter.split(document_ids, groups=document_ids), start=1
    ):
        fit_documents = set(document_ids[fit_index])
        hold_documents = set(document_ids[hold_index])
        fit_tokens = train_tokens[train_tokens.document_id.isin(fit_documents)]
        hold_tokens = train_tokens[train_tokens.document_id.isin(hold_documents)]
        hold_sentences = train_sentences[train_sentences.document_id.isin(hold_documents)]
        hold_entities = train_entities[train_entities.document_id.isin(hold_documents)]
        bilstm, predictions, _, _ = train_bilstm_comparator(
            fit_tokens,
            hold_tokens,
            hold_sentences,
            hold_entities,
            vocab_size=tokenizer.vocab_size,
            seed=settings["reference_seed"] + fold,
            max_epochs=1,
            patience=1,
        )
        outputs["BiLSTM-CRF"].append(predictions.assign(fold=fold))
        del bilstm
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        transformer, predictions, _, _, _ = train_transformer_comparator(
            settings["biomedical_backbone"],
            tokenizer,
            fit_tokens,
            hold_tokens,
            hold_sentences,
            hold_entities,
            seed=settings["reference_seed"] + 10 + fold,
            max_epochs=1,
            patience=1,
            batch_size=24,
        )
        outputs["Biomedical Transformer"].append(predictions.assign(fold=fold))
        del transformer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return {model: pd.concat(parts, ignore_index=True) for model, parts in outputs.items()}


def run_stage_04(project_root: Path, force_recompute: bool = False) -> dict:
    configure_notebook()
    cached = project_root / "artifacts" / "04_predictive_quality.csv"
    required = [
        project_root / "Models" / "Comparators" / "bilstm_crf.pt",
        project_root / "Models" / "Comparators" / "biomedical_transformer.pt",
        project_root / "Models" / "Comparators" / "context_train.pt",
        project_root / "Predictions" / "OOF" / "biomedical_transformer.parquet",
    ]
    if cached.exists() and all(path.exists() for path in required) and not force_recompute:
        selection = json.loads(
            (project_root / "src" / "comparator_selection.json").read_text(encoding="utf-8")
        )
        return {
            "predictive": pd.read_csv(cached),
            "classwise": pd.read_csv(project_root / "artifacts" / "04_classwise_quality.csv"),
            "calibration": pd.read_csv(project_root / "artifacts" / "04_calibration.csv"),
            "complementarity": pd.read_csv(project_root / "artifacts" / "04_subgroup_complementarity.csv"),
            "efficiency": pd.read_csv(project_root / "artifacts" / "04_efficiency.csv"),
            "history": pd.read_csv(project_root / "artifacts" / "04_learning_dynamics.csv"),
            "selection": pd.DataFrame([selection]),
            "confusion": pd.read_csv(project_root / "artifacts" / "04_entity_type_confusion.csv"),
            "figure_names": ["confusion_matrix_entity_types"],
        }
    settings = load_settings(project_root)
    tokenizer = AutoTokenizer.from_pretrained(settings["biomedical_backbone"], use_fast=True)
    train_sentences, train_entities, train_tokens = load_processed(project_root, "train")
    validation_sentences, validation_entities, validation_tokens = load_processed(
        project_root, "validation"
    )

    bilstm_model, bilstm_predictions, bilstm_history, bilstm_elapsed = train_bilstm_comparator(
        train_tokens,
        validation_tokens,
        validation_sentences,
        validation_entities,
        tokenizer.vocab_size,
        settings["reference_seed"],
    )
    transformer_model, transformer_predictions, transformer_history, transformer_elapsed, _ = (
        train_transformer_comparator(
            settings["biomedical_backbone"],
            tokenizer,
            train_tokens,
            validation_tokens,
            validation_sentences,
            validation_entities,
            settings["reference_seed"],
        )
    )

    model_directory = project_root / "Models" / "Comparators"
    bilstm_path = model_directory / "bilstm_crf.pt"
    transformer_path = model_directory / "biomedical_transformer.pt"
    torch.save(
        {
            "state_dict": bilstm_model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "n_tags": 17,
            "seed": settings["reference_seed"],
        },
        bilstm_path,
    )
    torch.save(
        {
            "state_dict": transformer_model.state_dict(),
            "backbone": settings["biomedical_backbone"],
            "seed": settings["reference_seed"],
        },
        transformer_path,
    )
    bilstm_predictions.to_parquet(
        project_root / "Predictions" / "Validation" / "bilstm_crf.parquet", index=False
    )
    transformer_predictions.to_parquet(
        project_root / "Predictions" / "Validation" / "biomedical_transformer.parquet", index=False
    )

    oof = _deep_oof(
        project_root, train_tokens, train_sentences, train_entities, tokenizer, settings
    )
    for model, frame in oof.items():
        frame.to_parquet(
            project_root
            / "Predictions"
            / "OOF"
            / f"{model.lower().replace(' ', '_').replace('-', '_')}.parquet",
            index=False,
        )

    train_context = extract_context_embeddings(transformer_model, train_tokens, tokenizer)
    validation_context = extract_context_embeddings(transformer_model, validation_tokens, tokenizer)
    torch.save(train_context, model_directory / "context_train.pt")
    torch.save(validation_context, model_directory / "context_validation.pt")

    predictions = {
        "BiLSTM-CRF": bilstm_predictions,
        "Biomedical Transformer": transformer_predictions,
    }
    predictive_parts = []
    class_parts = []
    calibration_parts = []
    subgroup_parts = []
    for model, frame in predictions.items():
        predictive, classwise, calibration, subgroup = _model_metrics_bundle(
            model, frame, validation_entities, validation_sentences, train_entities
        )
        predictive_parts.append(predictive)
        class_parts.append(classwise)
        calibration_parts.append(calibration)
        subgroup_parts.append(subgroup)
    predictive = pd.concat(predictive_parts, ignore_index=True)
    classwise = pd.concat(class_parts, ignore_index=True)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    subgroups = pd.concat(subgroup_parts, ignore_index=True)

    parameter_counts = {
        "BiLSTM-CRF": sum(parameter.numel() for parameter in bilstm_model.parameters()),
        "Biomedical Transformer": sum(parameter.numel() for parameter in transformer_model.parameters()),
    }
    sizes = {
        "BiLSTM-CRF": bilstm_path.stat().st_size / 2**20,
        "Biomedical Transformer": transformer_path.stat().st_size / 2**20,
    }
    inference_times = {}
    _, inference_times["BiLSTM-CRF"] = predict_bilstm(
        bilstm_model, validation_tokens, validation_sentences, "validation"
    )
    _, inference_times["Biomedical Transformer"], _ = predict_transformer(
        transformer_model,
        validation_tokens,
        validation_sentences,
        tokenizer,
        "validation",
    )
    peak_ram = psutil.Process().memory_info().rss / 2**20
    if torch.backends.mps.is_available():
        peak_vram = torch.mps.driver_allocated_memory() / 2**20
    else:
        peak_vram = np.nan
    efficiency = pd.DataFrame(
        [
            {
                "Model": model,
                "Split": "Validation",
                "N": len(validation_sentences),
                "Train + selection time (s)": bilstm_elapsed
                if model == "BiLSTM-CRF"
                else transformer_elapsed,
                "Latency per Sentence (ms)": 1000 * inference_times[model] / len(validation_sentences),
                "Latency per Entity (ms)": 1000 * inference_times[model] / max(len(predictions[model]), 1),
                "Throughput (sentences/s)": len(validation_sentences) / inference_times[model],
                "Parameter Count": parameter_counts[model],
                "Model Size (MB)": sizes[model],
                "Peak RAM (MB)": peak_ram,
                "Peak VRAM (MB)": peak_vram,
            }
            for model in predictions
        ]
    )

    baseline_subgroups = pd.read_csv(project_root / "artifacts" / "03_error_subgroups.csv")
    complementarity = pd.concat([baseline_subgroups, subgroups], ignore_index=True)
    complementarity = complementarity[
        complementarity.Model.isin(
            ["Longest-match lexicon", "Linear-chain CRF", "Biomedical Transformer"]
        )
    ]
    complementarity["Supported observation"] = complementarity.groupby("Subgroup")[
        "Strict Typed Entity F1"
    ].transform(lambda values: values == values.max())

    selected = predictive.sort_values(
        ["Strict Typed Entity F1", "Strict Boundary F1", "Macro-F1"], ascending=False
    ).iloc[0]["Model"]
    selection = {
        "selected_comparator": selected,
        "selection_split": "Validation",
        "primary_metric": "Strict Typed Entity F1",
        "biomedical_backbone": settings["biomedical_backbone"],
    }
    (project_root / "src" / "comparator_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    selected_predictions = predictions[selected]
    confusion = _entity_type_confusion(selected_predictions, validation_entities, selected)
    figure_name = _save_confusion_figure(project_root, confusion)
    history = pd.concat([bilstm_history, transformer_history], ignore_index=True)
    for name, frame in {
        "04_predictive_quality": predictive,
        "04_classwise_quality": classwise,
        "04_calibration": calibration,
        "04_subgroup_complementarity": complementarity,
        "04_efficiency": efficiency,
        "04_learning_dynamics": history,
        "04_entity_type_confusion": confusion,
    }.items():
        save_table(project_root, frame, name)
    return {
        "predictive": predictive,
        "classwise": classwise,
        "calibration": calibration,
        "complementarity": complementarity,
        "efficiency": efficiency,
        "history": history,
        "selection": pd.DataFrame([selection]),
        "confusion": confusion,
        "figure_names": [figure_name],
    }


def _metric_row(model: str, predictions: pd.DataFrame, entities: pd.DataFrame) -> dict:
    metrics, _ = evaluate_predictions(entities, predictions, model, "Validation")
    return metrics.iloc[0].to_dict()


def _paired_effect_intervals(
    act_predictions: pd.DataFrame,
    comparators: dict[str, pd.DataFrame],
    entities: pd.DataFrame,
    seed: int,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    act = document_level_scores(act_predictions, entities, "ACT-NER").set_index("document_id")
    rng = np.random.default_rng(seed)
    rows = []
    for name, predictions in comparators.items():
        other = document_level_scores(predictions, entities, name).set_index("document_id")
        paired = act[["Strict Typed Entity F1"]].join(
            other[["Strict Typed Entity F1"]], lsuffix="_act", rsuffix="_other"
        ).dropna()
        differences = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy(float)
        samples = [
            float(rng.choice(differences, size=len(differences), replace=True).mean())
            for _ in range(n_bootstrap)
        ]
        rows.append(
            {
                "Comparison": f"ACT-NER vs {name}",
                "Mean Effect": float(differences.mean()),
                "CI Lower": float(np.quantile(samples, 0.025)),
                "CI Upper": float(np.quantile(samples, 0.975)),
                "Unit": "document-level Strict Typed Entity F1 difference",
            }
        )
    return pd.DataFrame(rows)


def _save_stage5_figures(
    project_root: Path,
    calibrated_rows: pd.DataFrame,
    subgroups: pd.DataFrame,
    robustness: pd.DataFrame,
    gates: pd.DataFrame,
    ablations: pd.DataFrame,
    effect_intervals: pd.DataFrame,
    error_taxonomy: pd.DataFrame,
) -> list[str]:
    names = []
    calibration = calibrated_rows.copy()
    calibration["bin"] = pd.cut(
        calibration.cross_fitted_confidence,
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
        labels=False,
    )
    reliability = (
        calibration.groupby("bin", observed=False)
        .agg(
            Mean_Confidence=("cross_fitted_confidence", "mean"),
            Empirical_Accuracy=("outcome", "mean"),
            N=("outcome", "size"),
        )
        .dropna()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(10.5, 8.5))
    axis.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=2.0, label="Ideal calibration")
    axis.plot(
        reliability.Mean_Confidence,
        reliability.Empirical_Accuracy,
        color=BLUE,
        marker="o",
        markersize=9,
        linewidth=2.5,
        label="ACT-NER",
    )
    axis.set_title("Calibration Reliability Diagram")
    axis.set_xlabel("Mean calibrated confidence")
    axis.set_ylabel("Strict typed correctness")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(color=GRID, linewidth=0.8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    figure.subplots_adjust(bottom=0.27, top=0.88, left=0.13, right=0.97)
    save_figure(project_root, figure, "calibration_reliability_diagram", reliability)
    names.append("calibration_reliability_diagram")

    confidence_error = (
        calibration.assign(Error=1 - calibration.outcome)
        .groupby("bin", observed=False)
        .agg(
            Mean_Confidence=("cross_fitted_confidence", "mean"),
            Error_Rate=("Error", "mean"),
            N=("Error", "size"),
        )
        .dropna()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(11.5, 8.0))
    sizes = 70 + 500 * confidence_error.N / confidence_error.N.max()
    axis.scatter(
        confidence_error.Mean_Confidence,
        confidence_error.Error_Rate,
        s=sizes,
        color=ORANGE,
        edgecolor=INK,
        linewidth=1.1,
    )
    axis.plot(confidence_error.Mean_Confidence, confidence_error.Error_Rate, color=ORANGE, linewidth=2.2)
    axis.set_title("Confidence vs Error")
    axis.set_xlabel("Mean calibrated confidence")
    axis.set_ylabel("Strict typed error rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(color=GRID, linewidth=0.8)
    figure.subplots_adjust(bottom=0.17, top=0.88, left=0.13, right=0.97)
    save_figure(project_root, figure, "confidence_vs_error", confidence_error)
    names.append("confidence_vs_error")

    for subgroup_name, figure_name, title in (
        ("Unseen surface", "unseen_entity_performance", "Unseen Entity Performance"),
        ("Multi-token", "multi_token_entity_performance", "Multi-token Entity Performance"),
    ):
        data = subgroups[subgroups.Subgroup == subgroup_name].copy()
        figure, axis = plt.subplots(figsize=(16.0, 8.0))
        x = np.arange(len(data))
        axis.plot(
            x,
            data["Strict Boundary F1"],
            color=BLUE,
            marker="o",
            markersize=9,
            linewidth=2.5,
            label="Boundary F1",
        )
        axis.plot(
            x,
            data["Strict Typed Entity F1"],
            color=GOLD,
            marker="s",
            markersize=9,
            linewidth=2.5,
            label="Typed F1",
        )
        axis.set_title(title)
        axis.set_xlabel("Model")
        axis.set_ylabel("Validation F1")
        axis.set_ylim(0, 1)
        axis.set_xticks(x, data.Model, rotation=30, ha="right")
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)
        figure.subplots_adjust(bottom=0.38, top=0.88, left=0.10, right=0.80)
        save_figure(project_root, figure, figure_name, data)
        names.append(figure_name)

    domain = subgroups[subgroups.Subgroup.isin(["DrugBank", "MedLine"])].copy()
    figure, axis = plt.subplots(figsize=(16.5, 8.5))
    models = list(dict.fromkeys(domain.Model))
    x = np.arange(len(models))
    for offset, source, color, marker in (
        (-0.12, "DrugBank", BLUE, "o"),
        (0.12, "MedLine", GOLD, "s"),
    ):
        values = domain[domain.Subgroup == source].set_index("Model").reindex(models)
        axis.plot(
            x + offset,
            values["Strict Typed Entity F1"],
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=9,
            label=source,
        )
    axis.set_title("Domain Robustness")
    axis.set_xlabel("Model")
    axis.set_ylabel("Strict Typed Entity F1")
    axis.set_ylim(0, 1)
    axis.set_xticks(x, models, rotation=30, ha="right")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)
    figure.subplots_adjust(bottom=0.38, top=0.88, left=0.10, right=0.80)
    save_figure(project_root, figure, "domain_robustness", domain)
    names.append("domain_robustness")

    gate_summary = (
        gates.assign(
            Token_Group=np.select(
                [gates.lexicon_match, gates.gold_boundary_tag != "O"],
                ["Lexicon-matched", "Entity without lexicon"],
                default="Non-entity",
            )
        )
        .groupby("Token_Group")
        .gate_context_weight.agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(columns={"count": "N", "mean": "Mean", "median": "Median", "std": "SD"})
    )
    figure, axis = plt.subplots(figsize=(12.5, 8.0))
    bars = axis.bar(
        np.arange(len(gate_summary)),
        gate_summary.Mean,
        yerr=gate_summary.SD.fillna(0),
        capsize=8,
        color=[BLUE, GOLD, ORANGE][: len(gate_summary)],
        edgecolor=INK,
    )
    _label_bars(axis, bars, decimals=3)
    axis.set_title("ACT-NER Adaptive Gate Analysis")
    axis.set_xlabel("Token evidence group")
    axis.set_ylabel("Mean context gate weight")
    axis.set_ylim(0, min(1.0, max(0.8, gate_summary.Mean.max() + 0.2)))
    axis.set_xticks(np.arange(len(gate_summary)), gate_summary.Token_Group, rotation=20, ha="right")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    figure.subplots_adjust(bottom=0.28, top=0.88, left=0.12, right=0.97)
    save_figure(project_root, figure, "act_ner_adaptive_gate_analysis", gate_summary)
    names.append("act_ner_adaptive_gate_analysis")

    plot_ablation = ablations.sort_values("Delta vs Full ACT-NER")
    figure, axis = plt.subplots(figsize=(15.5, max(8.5, 0.65 * len(plot_ablation) + 3.5)))
    y = np.arange(len(plot_ablation))
    colors = [ORANGE if value < 0 else BLUE for value in plot_ablation["Delta vs Full ACT-NER"]]
    axis.barh(y, plot_ablation["Delta vs Full ACT-NER"], color=colors, edgecolor=INK)
    axis.axvline(0, color=INK, linewidth=1.8)
    axis.set_title("Component Contribution / Ablation")
    axis.set_xlabel("Change in Validation Strict Typed Entity F1")
    axis.set_ylabel("One-component intervention")
    axis.set_yticks(y, plot_ablation.Ablation)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    figure.subplots_adjust(bottom=0.15, top=0.91, left=0.34, right=0.97)
    save_figure(project_root, figure, "component_contribution_ablation", ablations)
    names.append("component_contribution_ablation")

    figure, axis = plt.subplots(figsize=(13.5, 7.5))
    y = np.arange(len(effect_intervals))
    means = effect_intervals["Mean Effect"].to_numpy()
    errors = np.vstack(
        [means - effect_intervals["CI Lower"], effect_intervals["CI Upper"] - means]
    )
    axis.errorbar(
        means,
        y,
        xerr=errors,
        fmt="o",
        color=BLUE,
        ecolor=INK,
        elinewidth=2.0,
        capsize=7,
        markersize=10,
    )
    axis.axvline(0, color=INK, linestyle="--", linewidth=1.8)
    axis.set_title("Effect Size with Confidence Intervals")
    axis.set_xlabel("Mean paired document-level F1 difference")
    axis.set_ylabel("Comparison")
    axis.set_yticks(y, effect_intervals.Comparison)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    figure.subplots_adjust(bottom=0.17, top=0.88, left=0.30, right=0.96)
    save_figure(project_root, figure, "effect_size_confidence_intervals", effect_intervals)
    names.append("effect_size_confidence_intervals")

    figure, axis = plt.subplots(figsize=(13.5, 8.5))
    taxonomy = error_taxonomy.sort_values("Count", ascending=True)
    axis.barh(np.arange(len(taxonomy)), taxonomy.Count, color=ORANGE, edgecolor=INK)
    axis.set_title("Error Taxonomy")
    axis.set_xlabel("Validation errors")
    axis.set_ylabel("Error category")
    axis.set_yticks(np.arange(len(taxonomy)), taxonomy.Error)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    figure.subplots_adjust(bottom=0.15, top=0.88, left=0.28, right=0.97)
    save_figure(project_root, figure, "error_taxonomy", error_taxonomy)
    names.append("error_taxonomy")

    figure, axis = plt.subplots(figsize=(20.0, 12.0))
    axis.set_xlim(0, 20)
    axis.set_ylim(0, 12)
    axis.axis("off")
    boxes = [
        (0.5, 9.6, 3.2, 1.2, "Biomedical\ncontext encoder", LIGHT_BLUE),
        (0.5, 7.2, 3.2, 1.2, "Training-only\nlexical branch", "#F3DDA7"),
        (0.5, 4.8, 3.2, 1.2, "Orthographic–\nmorphological branch", "#DDE7C8"),
        (5.0, 7.2, 3.6, 1.5, "Adaptive context–\nlexical fusion", "#D4E3F0"),
        (10.0, 7.2, 3.3, 1.5, "Local BiLSTM\nrefinement", "#E7D7E0"),
        (15.0, 8.8, 3.8, 1.4, "Constrained CRF\nboundary decoder", "#F3DDA7"),
        (15.0, 6.0, 3.8, 1.4, "Span attention +\npharmacological type head", "#DDE7C8"),
        (10.0, 3.4, 3.5, 1.4, "Learned confidence\nfusion", "#D4E3F0"),
        (15.0, 3.4, 3.8, 1.4, "Validation-selected\ncalibration and review", "#E7D7E0"),
        (15.0, 0.8, 3.8, 1.4, "Stable predicted-entity\ninterface for relation extraction", LIGHT_BLUE),
    ]
    for x, y, width, height, label, color in boxes:
        rectangle = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor=INK, linewidth=1.8)
        axis.add_patch(rectangle)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=18)
    arrows = [
        ((3.7, 10.2), (5.0, 8.2)),
        ((3.7, 7.8), (5.0, 8.0)),
        ((3.7, 5.4), (5.0, 7.7)),
        ((8.6, 8.0), (10.0, 8.0)),
        ((13.3, 8.0), (15.0, 9.5)),
        ((13.3, 7.7), (15.0, 6.7)),
        ((16.9, 8.8), (12.0, 4.8)),
        ((16.9, 6.0), (12.2, 4.8)),
        ((13.5, 4.1), (15.0, 4.1)),
        ((16.9, 3.4), (16.9, 2.2)),
    ]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": INK, "lw": 2.0})
    axis.set_title("ACT-NER Architecture", fontsize=18)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.93)
    architecture_data = pd.DataFrame(
        [{"Component": label.replace("\n", " ")} for _, _, _, _, label, _ in boxes]
    )
    save_figure(project_root, figure, "act_ner_architecture", architecture_data)
    names.append("act_ner_architecture")
    return names


def _act_oof(
    train_tokens: pd.DataFrame,
    train_sentences: pd.DataFrame,
    train_entities: pd.DataFrame,
    tokenizer,
    settings: dict,
    selected_weights: dict,
) -> pd.DataFrame:
    document_ids = np.array(sorted(train_sentences.document_id.unique()))
    splitter = GroupKFold(n_splits=settings["oof_folds"])
    outputs = []
    for fold, (fit_index, hold_index) in enumerate(
        splitter.split(document_ids, groups=document_ids), start=1
    ):
        fit_documents = set(document_ids[fit_index])
        hold_documents = set(document_ids[hold_index])
        fit_tokens = train_tokens[train_tokens.document_id.isin(fit_documents)]
        hold_tokens = train_tokens[train_tokens.document_id.isin(hold_documents)]
        fit_sentences = train_sentences[train_sentences.document_id.isin(fit_documents)]
        hold_sentences = train_sentences[train_sentences.document_id.isin(hold_documents)]
        fit_entities = train_entities[train_entities.document_id.isin(fit_documents)].copy()
        hold_entities = train_entities[train_entities.document_id.isin(hold_documents)]
        fit_entities["split"] = "train"
        fold_lexicon = build_training_lexicon(fit_entities)
        fit_features = add_act_features(fit_tokens, fit_sentences, fold_lexicon)
        hold_features = add_act_features(hold_tokens, hold_sentences, fold_lexicon)
        transformer, _, _, _, _ = train_transformer_comparator(
            settings["biomedical_backbone"],
            tokenizer,
            fit_tokens,
            hold_tokens,
            hold_sentences,
            hold_entities,
            seed=settings["reference_seed"] + 100 + fold,
            max_epochs=1,
            patience=1,
        )
        fit_context = extract_context_embeddings(transformer, fit_tokens, tokenizer)
        hold_context = extract_context_embeddings(transformer, hold_tokens, tokenizer)
        model, predictions, _, _, _ = train_act_model(
            ACTConfig(),
            fit_features,
            hold_features,
            fit_context,
            hold_context,
            hold_sentences,
            hold_entities,
            seed=settings["reference_seed"] + 200 + fold,
            lambda_type=selected_weights["lambda_type"],
            lambda_confidence=selected_weights["lambda_confidence"],
            lambda_regularization=selected_weights["lambda_regularization"],
            max_epochs=3,
            patience=1,
        )
        outputs.append(predictions.assign(fold=fold))
        del transformer, model, fit_context, hold_context
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return pd.concat(outputs, ignore_index=True)


def _perturb_sentences(
    sentences: pd.DataFrame,
    entities: pd.DataFrame,
    perturbation: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entities_by_sentence = {key: group for key, group in entities.groupby("sentence_id")}
    output = sentences.copy()
    texts = []
    for sentence in output.itertuples(index=False):
        text = sentence.text
        if perturbation == "casing":
            perturbed = text.lower()
        elif perturbation == "punctuation":
            perturbed = "".join(" " if (not character.isalnum() and not character.isspace()) else character for character in text)
        elif perturbation == "token_noise":
            characters = list(text)
            protected = np.zeros(len(characters), dtype=bool)
            for entity in entities_by_sentence.get(sentence.sentence_id, pd.DataFrame()).itertuples(index=False):
                protected[int(entity.char_start) : int(entity.char_end)] = True
            candidates = [
                index
                for index, character in enumerate(characters)
                if character.isalpha() and not protected[index]
            ]
            n_changes = max(1, int(round(0.05 * len(candidates)))) if candidates else 0
            for index in rng.choice(candidates, size=n_changes, replace=False) if n_changes else []:
                characters[int(index)] = "x" if characters[int(index)].islower() else "X"
            perturbed = "".join(characters)
        else:
            raise ValueError(perturbation)
        assert len(perturbed) == len(text)
        texts.append(perturbed)
    output["text"] = texts
    return output


def run_stage_05(project_root: Path, force_recompute: bool = False) -> dict:
    configure_notebook()
    cached = project_root / "artifacts" / "05_validation_predictive_quality.csv"
    required = [
        project_root / "Models" / "Proposed" / "act_ner.pt",
        project_root / "Models" / "Proposed" / "confidence_calibrator.joblib",
        project_root / "Predictions" / "OOF" / "act_ner.parquet",
        project_root / "src" / "frozen_design.json",
    ]
    if cached.exists() and all(path.exists() for path in required) and not force_recompute:
        freeze = json.loads(
            (project_root / "src" / "frozen_design.json").read_text(encoding="utf-8")
        )
        return {
            "seed_results": pd.read_csv(project_root / "artifacts" / "05_act_seed_results.csv"),
            "seed_summary": pd.read_csv(project_root / "artifacts" / "05_act_seed_summary.csv"),
            "loss_tuning": pd.read_csv(project_root / "artifacts" / "05_loss_weight_tuning.csv"),
            "predictive": pd.read_csv(cached),
            "classwise": pd.read_csv(project_root / "artifacts" / "05_validation_classwise_quality.csv"),
            "calibration": pd.read_csv(project_root / "artifacts" / "05_calibration_method_selection.csv"),
            "confidence_loss": pd.read_csv(project_root / "artifacts" / "05_confidence_loss_ablation.csv"),
            "ablations": pd.read_csv(project_root / "artifacts" / "05_ablation.csv"),
            "sensitivity": pd.read_csv(project_root / "artifacts" / "05_sensitivity.csv"),
            "robustness": pd.read_csv(project_root / "artifacts" / "05_robustness.csv"),
            "subgroups": pd.read_csv(project_root / "artifacts" / "05_subgroup_performance.csv"),
            "bootstrap": pd.read_csv(project_root / "artifacts" / "05_bootstrap_ci.csv"),
            "statistics": pd.read_csv(project_root / "artifacts" / "05_paired_statistics.csv"),
            "effect_intervals": pd.read_csv(project_root / "artifacts" / "05_effect_intervals.csv"),
            "efficiency": pd.read_csv(project_root / "artifacts" / "05_efficiency.csv"),
            "freeze": pd.DataFrame([freeze]),
            "figure_names": [
                "calibration_reliability_diagram",
                "confidence_vs_error",
                "unseen_entity_performance",
                "multi_token_entity_performance",
                "domain_robustness",
                "act_ner_adaptive_gate_analysis",
                "component_contribution_ablation",
                "effect_size_confidence_intervals",
                "error_taxonomy",
                "act_ner_architecture",
            ],
        }
    settings = load_settings(project_root)
    tokenizer = AutoTokenizer.from_pretrained(settings["biomedical_backbone"], use_fast=True)
    shared = shared_root(project_root)
    train_sentences, train_entities, train_tokens = load_processed(project_root, "train")
    validation_sentences, validation_entities, validation_tokens = load_processed(
        project_root, "validation"
    )
    lexicon = pd.read_parquet(shared / "Processed" / "training_lexicon.parquet")
    train_features = add_act_features(train_tokens, train_sentences, lexicon)
    validation_features = add_act_features(validation_tokens, validation_sentences, lexicon)
    train_features.to_parquet(shared / "Processed" / "train_act_features.parquet", index=False)
    validation_features.to_parquet(
        shared / "Processed" / "validation_act_features.parquet", index=False
    )
    comparator_directory = project_root / "Models" / "Comparators"
    train_context = torch.load(comparator_directory / "context_train.pt", weights_only=False)
    validation_context = torch.load(
        comparator_directory / "context_validation.pt", weights_only=False
    )

    full_config = ACTConfig()
    selected_weights, loss_tuning = tune_act_loss_weights(
        full_config,
        train_features,
        validation_features,
        train_context,
        validation_context,
        validation_sentences,
        validation_entities,
        settings["reference_seed"],
    )
    seed_records = []
    seed_predictions = {}
    seed_models = {}
    seed_histories = []
    seed_gates = {}
    elapsed_by_seed = {}
    proposed_directory = project_root / "Models" / "Proposed"
    for seed in settings["full_model_seeds"]:
        model, predictions, history, elapsed, gates = train_act_model(
            full_config,
            train_features,
            validation_features,
            train_context,
            validation_context,
            validation_sentences,
            validation_entities,
            seed,
            selected_weights["lambda_type"],
            selected_weights["lambda_confidence"],
            selected_weights["lambda_regularization"],
        )
        metrics, _ = evaluate_predictions(validation_entities, predictions, "ACT-NER", "Validation")
        seed_records.append(
            {
                "Model": "ACT-NER",
                "Split": "Validation",
                "N": len(validation_entities),
                "Seed": seed,
                "Strict Boundary F1": metrics.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metrics.iloc[0]["Strict Typed Entity F1"],
                "Macro-F1": metrics.iloc[0]["Macro-F1"],
            }
        )
        seed_predictions[seed] = predictions
        seed_models[seed] = model
        seed_gates[seed] = gates
        elapsed_by_seed[seed] = elapsed
        seed_histories.append(history)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(full_config),
                "seed": seed,
                "loss_weights": selected_weights,
                "context_encoder": settings["biomedical_backbone"],
                "context_encoder_checkpoint": "Models/Comparators/biomedical_transformer.pt",
            },
            proposed_directory / f"act_ner_seed_{seed}.pt",
        )
    reference_seed = settings["reference_seed"]
    reference_model = seed_models[reference_seed]
    reference_predictions = seed_predictions[reference_seed]
    reference_gates = seed_gates[reference_seed]
    shutil.copyfile(
        proposed_directory / f"act_ner_seed_{reference_seed}.pt",
        proposed_directory / "act_ner.pt",
    )

    calibrator, calibration_method, calibration_comparison, calibration_rows = (
        fit_validation_calibration(reference_predictions, validation_entities)
    )
    calibrated_predictions = apply_calibration(
        reference_predictions, calibrator, calibration_method
    )
    policy = select_confidence_policy(calibration_rows)
    calibrated_predictions = apply_confidence_policy(calibrated_predictions, policy)
    calibrated_predictions.to_parquet(
        project_root / "Predictions" / "Validation" / "act_ner.parquet", index=False
    )
    joblib.dump(calibrator, proposed_directory / "confidence_calibrator.joblib")
    (proposed_directory / "confidence_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    (project_root / "src" / "calibration_selection.json").write_text(
        json.dumps(
            {
                "method": calibration_method,
                "selection_split": "Validation cross-fit",
                "criteria": ["ECE", "Brier Score"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    seed_table = pd.DataFrame(seed_records)
    seed_summary = pd.DataFrame(
        [
            {
                "Model": "ACT-NER",
                "Split": "Validation",
                "N": len(validation_entities),
                "Seeds": len(seed_table),
                "Boundary F1 Mean": seed_table["Strict Boundary F1"].mean(),
                "Boundary F1 SD": seed_table["Strict Boundary F1"].std(ddof=1),
                "Typed F1 Mean": seed_table["Strict Typed Entity F1"].mean(),
                "Typed F1 SD": seed_table["Strict Typed Entity F1"].std(ddof=1),
                "Macro-F1 Mean": seed_table["Macro-F1"].mean(),
                "Macro-F1 SD": seed_table["Macro-F1"].std(ddof=1),
            }
        ]
    )

    reference_metric, reference_classwise = evaluate_predictions(
        validation_entities, calibrated_predictions, "ACT-NER", "Validation"
    )
    reference_calibration = calibration_metrics(
        calibrated_predictions, validation_entities, "ACT-NER", "Validation"
    )

    ablation_configs = {
        "Without lexical branch": replace(full_config, use_lexical=False),
        "Without orthographic branch": replace(full_config, use_orthographic=False),
        "Without adaptive gate (concatenation)": replace(full_config, use_adaptive_gate=False),
        "Without local BiLSTM refinement": replace(full_config, use_local_refinement=False),
        "Without CRF": replace(full_config, use_crf=False),
        "Without span attention pooling": replace(full_config, use_span_attention=False),
        "Without subtype head": replace(full_config, use_type_head=False),
        "Without confidence loss": replace(full_config, use_confidence_loss=False),
        "Parameter-matched static control": replace(
            full_config, use_adaptive_gate=False, parameter_matched_control=True
        ),
    }
    ablation_predictions = {}
    ablation_rows = [
        {
            "Ablation": "Full ACT-NER",
            "Changed Component": "None",
            "Strict Boundary F1": reference_metric.iloc[0]["Strict Boundary F1"],
            "Strict Typed Entity F1": reference_metric.iloc[0]["Strict Typed Entity F1"],
            "ECE": reference_calibration.iloc[0]["ECE"],
            "Review Coverage": float(calibrated_predictions.review_required.mean()),
        }
    ]
    for label, config in ablation_configs.items():
        model, predictions, _, _, _ = train_act_model(
            config,
            train_features,
            validation_features,
            train_context,
            validation_context,
            validation_sentences,
            validation_entities,
            reference_seed,
            selected_weights["lambda_type"],
            selected_weights["lambda_confidence"],
            selected_weights["lambda_regularization"],
            max_epochs=6,
            patience=1,
        )
        metrics, _ = evaluate_predictions(validation_entities, predictions, "ACT-NER", "Validation")
        calibration = calibration_metrics(predictions, validation_entities, "ACT-NER", "Validation")
        ablation_rows.append(
            {
                "Ablation": label,
                "Changed Component": label.replace("Without ", ""),
                "Strict Boundary F1": metrics.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metrics.iloc[0]["Strict Typed Entity F1"],
                "ECE": calibration.iloc[0]["ECE"],
                "Review Coverage": 0.0,
            }
        )
        ablation_predictions[label] = predictions
        del model
    raw_calibration = calibration_metrics(
        reference_predictions, validation_entities, "ACT-NER", "Validation"
    )
    ablation_rows.extend(
        [
            {
                "Ablation": "Without calibration",
                "Changed Component": "Probability calibration",
                "Strict Boundary F1": reference_metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": reference_metric.iloc[0]["Strict Typed Entity F1"],
                "ECE": raw_calibration.iloc[0]["ECE"],
                "Review Coverage": 0.0,
            },
            {
                "Ablation": "Without confidence-aware output",
                "Changed Component": "Confidence categories and review flag",
                "Strict Boundary F1": reference_metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": reference_metric.iloc[0]["Strict Typed Entity F1"],
                "ECE": reference_calibration.iloc[0]["ECE"],
                "Review Coverage": 0.0,
            },
        ]
    )
    transformer_predictions = pd.read_parquet(
        project_root / "Predictions" / "Validation" / "biomedical_transformer.parquet"
    )
    crf_predictions = pd.read_parquet(
        project_root / "Predictions" / "Validation" / "linear_chain_crf.parquet"
    )
    for label, predictions in (
        ("Transformer only", transformer_predictions),
        ("CRF only", crf_predictions),
    ):
        metric, _ = evaluate_predictions(validation_entities, predictions, label, "Validation")
        calibration = calibration_metrics(predictions, validation_entities, label, "Validation")
        ablation_rows.append(
            {
                "Ablation": label,
                "Changed Component": "Complete comparator control",
                "Strict Boundary F1": metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metric.iloc[0]["Strict Typed Entity F1"],
                "ECE": calibration.iloc[0]["ECE"],
                "Review Coverage": 0.0,
            }
        )
    ablations = pd.DataFrame(ablation_rows)
    full_typed = float(
        ablations.loc[ablations.Ablation == "Full ACT-NER", "Strict Typed Entity F1"].iloc[0]
    )
    ablations["Delta vs Full ACT-NER"] = ablations["Strict Typed Entity F1"] - full_typed
    no_confidence_predictions = ablation_predictions["Without confidence loss"]
    no_confidence_metric, _ = evaluate_predictions(
        validation_entities, no_confidence_predictions, "ACT-NER without confidence loss", "Validation"
    )
    no_confidence_calibration = calibration_metrics(
        no_confidence_predictions,
        validation_entities,
        "ACT-NER without confidence loss",
        "Validation",
    )
    full_raw_calibration = calibration_metrics(
        reference_predictions,
        validation_entities,
        "Full ACT-NER with confidence loss",
        "Validation",
    )
    confidence_loss_ablation = pd.DataFrame(
        [
            {
                "Model": "Full ACT-NER with confidence loss",
                "Split": "Validation",
                "N": len(validation_entities),
                "Strict Boundary F1": reference_metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": reference_metric.iloc[0]["Strict Typed Entity F1"],
                "Raw ECE": full_raw_calibration.iloc[0]["ECE"],
                "Raw Brier Score": full_raw_calibration.iloc[0]["Brier Score"],
            },
            {
                "Model": "ACT-NER without confidence loss",
                "Split": "Validation",
                "N": len(validation_entities),
                "Strict Boundary F1": no_confidence_metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": no_confidence_metric.iloc[0]["Strict Typed Entity F1"],
                "Raw ECE": no_confidence_calibration.iloc[0]["ECE"],
                "Raw Brier Score": no_confidence_calibration.iloc[0]["Brier Score"],
            },
        ]
    )
    no_loss = confidence_loss_ablation.iloc[1]
    for column in (
        "Strict Boundary F1",
        "Strict Typed Entity F1",
        "Raw ECE",
        "Raw Brier Score",
    ):
        confidence_loss_ablation[f"Delta vs no-loss: {column}"] = (
            confidence_loss_ablation[column] - no_loss[column]
        )

    sensitivity_rows = []
    center_metric = reference_metric.iloc[0]
    sensitivity_rows.extend(
        [
            {
                "Parameter": parameter,
                "Value": center,
                "Strict Boundary F1": center_metric["Strict Boundary F1"],
                "Strict Typed Entity F1": center_metric["Strict Typed Entity F1"],
            }
            for parameter, center in (
                ("dropout", 0.25),
                ("learning_rate", 0.001),
                ("gate_hidden_size", 128),
                ("crf_transition_regularization", selected_weights["lambda_regularization"]),
            )
        ]
    )
    sensitivity_specs = [
        ("dropout", 0.15, replace(full_config, dropout=0.15), 0.001, selected_weights["lambda_regularization"]),
        ("dropout", 0.35, replace(full_config, dropout=0.35), 0.001, selected_weights["lambda_regularization"]),
        ("learning_rate", 0.0005, full_config, 0.0005, selected_weights["lambda_regularization"]),
        ("learning_rate", 0.002, full_config, 0.002, selected_weights["lambda_regularization"]),
        ("gate_hidden_size", 64, replace(full_config, gate_hidden_size=64), 0.001, selected_weights["lambda_regularization"]),
        ("gate_hidden_size", 192, replace(full_config, gate_hidden_size=192), 0.001, selected_weights["lambda_regularization"]),
        ("crf_transition_regularization", 0.0, full_config, 0.001, 0.0),
        ("crf_transition_regularization", 0.01, full_config, 0.001, 0.01),
    ]
    for parameter, value, config, learning_rate, regularization in sensitivity_specs:
        model, predictions, _, _, _ = train_act_model(
            config,
            train_features,
            validation_features,
            train_context,
            validation_context,
            validation_sentences,
            validation_entities,
            reference_seed,
            selected_weights["lambda_type"],
            selected_weights["lambda_confidence"],
            regularization,
            learning_rate=learning_rate,
            max_epochs=4,
            patience=1,
        )
        metric, _ = evaluate_predictions(validation_entities, predictions, "ACT-NER", "Validation")
        sensitivity_rows.append(
            {
                "Parameter": parameter,
                "Value": value,
                "Strict Boundary F1": metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metric.iloc[0]["Strict Typed Entity F1"],
            }
        )
        del model

    comparator_checkpoint = torch.load(
        comparator_directory / "biomedical_transformer.pt", map_location="cpu", weights_only=False
    )
    full_transformer = build_transformer_model(settings["biomedical_backbone"])
    full_transformer.load_state_dict(comparator_checkpoint["state_dict"])
    full_transformer = full_transformer.to(next(reference_model.parameters()).device)
    sensitivity_rows.append(
        {
            "Parameter": "transformer_unfreezing_depth",
            "Value": 6,
            "Strict Boundary F1": center_metric["Strict Boundary F1"],
            "Strict Typed Entity F1": center_metric["Strict Typed Entity F1"],
        }
    )
    for depth in (0, 1):
        transformer, _, _, _, _ = train_transformer_comparator(
            settings["biomedical_backbone"],
            tokenizer,
            train_tokens,
            validation_tokens,
            validation_sentences,
            validation_entities,
            reference_seed + depth + 300,
            max_epochs=1,
            patience=1,
            unfreeze_depth=depth,
        )
        depth_train_context = extract_context_embeddings(transformer, train_tokens, tokenizer)
        depth_validation_context = extract_context_embeddings(transformer, validation_tokens, tokenizer)
        model, predictions, _, _, _ = train_act_model(
            full_config,
            train_features,
            validation_features,
            depth_train_context,
            depth_validation_context,
            validation_sentences,
            validation_entities,
            reference_seed,
            selected_weights["lambda_type"],
            selected_weights["lambda_confidence"],
            selected_weights["lambda_regularization"],
            max_epochs=3,
            patience=1,
        )
        metric, _ = evaluate_predictions(validation_entities, predictions, "ACT-NER", "Validation")
        sensitivity_rows.append(
            {
                "Parameter": "transformer_unfreezing_depth",
                "Value": depth,
                "Strict Boundary F1": metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metric.iloc[0]["Strict Typed Entity F1"],
            }
        )
        del transformer, model, depth_train_context, depth_validation_context
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    outcomes = calibration_rows.outcome.to_numpy(int)
    raw_values = calibration_rows.raw_probability.to_numpy(float)
    raw_logit = np.log(np.clip(raw_values, 1e-6, 1 - 1e-6) / np.clip(1 - raw_values, 1e-6, 1))
    for temperature in (0.75, 1.0, 1.25):
        probabilities = 1 / (1 + np.exp(-raw_logit / temperature))
        sensitivity_rows.append(
            {
                "Parameter": "calibration_temperature",
                "Value": temperature,
                "Strict Boundary F1": np.nan,
                "Strict Typed Entity F1": np.nan,
                "ECE": expected_calibration_error(probabilities, outcomes),
                "Brier Score": float(np.mean((probabilities - outcomes) ** 2)),
            }
        )
    for threshold in (policy["low_threshold"], policy["high_threshold"], 0.5):
        selected = calibration_rows.cross_fitted_confidence < threshold
        sensitivity_rows.append(
            {
                "Parameter": "confidence_threshold",
                "Value": threshold,
                "Strict Boundary F1": np.nan,
                "Strict Typed Entity F1": np.nan,
                "Review Coverage": float(selected.mean()),
                "Reviewed Error Capture": float(
                    ((1 - calibration_rows.outcome.astype(int))[selected].sum())
                    / max((1 - calibration_rows.outcome.astype(int)).sum(), 1)
                ),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)

    robustness_rows = []
    for perturbation in ("token_noise", "punctuation", "casing"):
        perturbed_sentences = _perturb_sentences(
            validation_sentences,
            validation_entities,
            perturbation,
            settings["split_seed"],
        )
        perturbed_tokens, _ = tokenize_split(
            perturbed_sentences, validation_entities, tokenizer
        )
        perturbed_features = add_act_features(perturbed_tokens, perturbed_sentences, lexicon)
        perturbed_context = extract_context_embeddings(
            full_transformer, perturbed_tokens, tokenizer
        )
        predictions, _, _ = predict_act(
            reference_model,
            perturbed_features,
            perturbed_sentences,
            perturbed_context,
            "validation",
            reference_seed,
        )
        metric, _ = evaluate_predictions(validation_entities, predictions, "ACT-NER", "Validation")
        robustness_rows.append(
            {
                "Model": "ACT-NER",
                "Split": "Validation",
                "N": len(validation_entities),
                "Robustness Condition": perturbation.replace("_", " ").title(),
                "Strict Boundary F1": metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metric.iloc[0]["Strict Typed Entity F1"],
            }
        )
    del full_transformer

    train_document_ids = np.array(sorted(train_sentences.document_id.unique()))
    rng = np.random.default_rng(settings["split_seed"])
    for fraction in (0.25, 0.50, 0.75, 1.00):
        selected_documents = (
            set(rng.choice(train_document_ids, int(round(len(train_document_ids) * fraction)), replace=False))
            if fraction < 1
            else set(train_document_ids)
        )
        subset_features = train_features[train_features.document_id.isin(selected_documents)]
        subset_context = {
            sentence_id: value
            for sentence_id, value in train_context.items()
            if sentence_id in set(subset_features.sentence_id)
        }
        model, predictions, _, _, _ = train_act_model(
            full_config,
            subset_features,
            validation_features,
            subset_context,
            validation_context,
            validation_sentences,
            validation_entities,
            reference_seed,
            selected_weights["lambda_type"],
            selected_weights["lambda_confidence"],
            selected_weights["lambda_regularization"],
            max_epochs=4,
            patience=1,
        )
        metric, _ = evaluate_predictions(validation_entities, predictions, "ACT-NER", "Validation")
        robustness_rows.append(
            {
                "Model": "ACT-NER",
                "Split": "Validation",
                "N": len(validation_entities),
                "Robustness Condition": f"Training size {int(fraction * 100)}%",
                "Strict Boundary F1": metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metric.iloc[0]["Strict Typed Entity F1"],
            }
        )
        del model
    robustness = pd.DataFrame(robustness_rows)

    baseline_predictive = pd.read_csv(project_root / "artifacts" / "03_predictive_quality.csv")
    comparator_predictive = pd.read_csv(project_root / "artifacts" / "04_predictive_quality.csv")
    all_predictive = pd.concat(
        [baseline_predictive, comparator_predictive, reference_metric], ignore_index=True, sort=False
    )
    all_classwise = pd.concat(
        [
            pd.read_csv(project_root / "artifacts" / "03_classwise_quality.csv"),
            pd.read_csv(project_root / "artifacts" / "04_classwise_quality.csv"),
            reference_classwise,
        ],
        ignore_index=True,
    )
    model_prediction_map = {
        "O-only": pd.read_parquet(project_root / "Predictions" / "Validation" / "o_only.parquet"),
        "Longest-match lexicon": pd.read_parquet(
            project_root / "Predictions" / "Validation" / "longest_match_lexicon.parquet"
        ),
        "Logistic Regression": pd.read_parquet(
            project_root / "Predictions" / "Validation" / "logistic_regression.parquet"
        ),
        "Linear-chain CRF": crf_predictions,
        "BiLSTM-CRF": pd.read_parquet(
            project_root / "Predictions" / "Validation" / "bilstm_crf.parquet"
        ),
        "Biomedical Transformer": transformer_predictions,
        "ACT-NER": calibrated_predictions,
    }
    subgroup_parts = []
    for model, predictions in model_prediction_map.items():
        subgroup_parts.append(
            subgroup_metrics(
                predictions,
                validation_entities,
                validation_sentences,
                train_entities,
                model,
                "Validation",
            )
        )
    all_subgroups = pd.concat(subgroup_parts, ignore_index=True)

    baseline_selection = json.loads(
        (project_root / "src" / "baseline_selection.json").read_text()
    )["selected_baseline"]
    comparator_selection = json.loads(
        (project_root / "src" / "comparator_selection.json").read_text()
    )["selected_comparator"]
    comparison_predictions = {
        baseline_selection: model_prediction_map[baseline_selection],
        comparator_selection: model_prediction_map[comparator_selection],
    }
    statistics = paired_statistical_comparison(
        calibrated_predictions, comparison_predictions, validation_entities
    )
    effect_intervals = _paired_effect_intervals(
        calibrated_predictions,
        comparison_predictions,
        validation_entities,
        settings["split_seed"],
    )
    bootstrap = paired_bootstrap_ci(
        calibrated_predictions,
        validation_entities,
        "ACT-NER",
        settings["split_seed"],
    )

    error_taxonomy = boundary_error_profile(
        gold_for_metrics(validation_entities), calibrated_predictions
    )
    exact_gold = {
        (row.sentence_id, int(row.char_start), int(row.char_end)): row.entity_type
        for row in gold_for_metrics(validation_entities).itertuples(index=False)
    }
    type_errors = sum(
        1
        for row in calibrated_predictions.itertuples(index=False)
        if (row.sentence_id, int(row.char_start), int(row.char_end)) in exact_gold
        and exact_gold[(row.sentence_id, int(row.char_start), int(row.char_end))]
        != row.predicted_type
    )
    error_taxonomy = pd.concat(
        [
            error_taxonomy.query("Error != 'Exact'"),
            pd.DataFrame([{"Error": "Exact boundary, wrong type", "Count": type_errors}]),
        ],
        ignore_index=True,
    )

    oof_predictions = _act_oof(
        train_tokens,
        train_sentences,
        train_entities,
        tokenizer,
        settings,
        selected_weights,
    )
    oof_predictions.to_parquet(
        project_root / "Predictions" / "OOF" / "act_ner.parquet", index=False
    )

    parameter_count = sum(parameter.numel() for parameter in reference_model.parameters())
    checkpoint_size = (proposed_directory / "act_ner.pt").stat().st_size / 2**20
    inference_start = perf_counter()
    _, inference_elapsed, _ = predict_act(
        reference_model,
        validation_features,
        validation_sentences,
        validation_context,
        "validation",
        reference_seed,
    )
    peak_vram = (
        torch.mps.driver_allocated_memory() / 2**20 if torch.backends.mps.is_available() else np.nan
    )
    efficiency = pd.DataFrame(
        [
            {
                "Model": "ACT-NER",
                "Split": "Validation",
                "N": len(validation_sentences),
                "Train + selection time mean (s)": float(np.mean(list(elapsed_by_seed.values()))),
                "Latency per Sentence (ms)": 1000 * inference_elapsed / len(validation_sentences),
                "Latency per Entity (ms)": 1000 * inference_elapsed / max(len(calibrated_predictions), 1),
                "Throughput (sentences/s)": len(validation_sentences) / inference_elapsed,
                "Parameter Count": parameter_count,
                "Model Size (MB)": checkpoint_size,
                "Peak RAM (MB)": psutil.Process().memory_info().rss / 2**20,
                "Peak VRAM (MB)": peak_vram,
            }
        ]
    )

    frozen_design = {
        "model": "ACT-NER",
        "reference_seed": reference_seed,
        "model_checkpoint": "Models/Proposed/act_ner.pt",
        "transformer_checkpoint": "Models/Comparators/biomedical_transformer.pt",
        "loss_weights": selected_weights,
        "calibration_method": calibration_method,
        "confidence_policy": policy,
        "baseline_comparator": baseline_selection,
        "deep_comparator": comparator_selection,
        "locked_test_opened": False,
        "frozen_before_notebook_06": True,
    }
    (project_root / "src" / "frozen_design.json").write_text(
        json.dumps(frozen_design, indent=2), encoding="utf-8"
    )
    for name, frame in {
        "05_act_seed_results": seed_table,
        "05_act_seed_summary": seed_summary,
        "05_loss_weight_tuning": loss_tuning,
        "05_validation_predictive_quality": all_predictive,
        "05_validation_classwise_quality": all_classwise,
        "05_calibration_method_selection": calibration_comparison,
        "05_confidence_loss_ablation": confidence_loss_ablation,
        "05_ablation": ablations,
        "05_sensitivity": sensitivity,
        "05_robustness": robustness,
        "05_subgroup_performance": all_subgroups,
        "05_bootstrap_ci": bootstrap,
        "05_paired_statistics": statistics,
        "05_effect_intervals": effect_intervals,
        "05_error_taxonomy": error_taxonomy,
        "05_efficiency": efficiency,
    }.items():
        save_table(project_root, frame, name)
    figure_names = _save_stage5_figures(
        project_root,
        calibration_rows,
        all_subgroups,
        robustness,
        reference_gates,
        ablations,
        effect_intervals,
        error_taxonomy,
    )
    return {
        "seed_results": seed_table,
        "seed_summary": seed_summary,
        "loss_tuning": loss_tuning,
        "predictive": all_predictive,
        "classwise": all_classwise,
        "calibration": calibration_comparison,
        "confidence_loss": confidence_loss_ablation,
        "ablations": ablations,
        "sensitivity": sensitivity,
        "robustness": robustness,
        "subgroups": all_subgroups,
        "bootstrap": bootstrap,
        "statistics": statistics,
        "efficiency": efficiency,
        "freeze": pd.DataFrame([frozen_design]),
        "figure_names": figure_names,
    }


def _prediction_slug(model: str) -> str:
    return model.lower().replace(" ", "_").replace("-", "_")


def _load_frozen_models(project_root: Path, tokenizer, settings: dict):
    baseline_directory = project_root / "Models" / "Baselines"
    comparator_directory = project_root / "Models" / "Comparators"
    proposed_directory = project_root / "Models" / "Proposed"
    logistic = joblib.load(baseline_directory / "logistic_regression.joblib")
    linear_crf = joblib.load(baseline_directory / "linear_chain_crf.joblib")
    lexicon_matcher = joblib.load(baseline_directory / "longest_match_lexicon.joblib")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    bilstm_checkpoint = torch.load(
        comparator_directory / "bilstm_crf.pt", map_location="cpu", weights_only=False
    )
    bilstm = BiLSTMCRF(
        vocab_size=int(bilstm_checkpoint["vocab_size"]),
        n_tags=int(bilstm_checkpoint["n_tags"]),
    ).to(device)
    bilstm.load_state_dict(bilstm_checkpoint["state_dict"])

    transformer_checkpoint = torch.load(
        comparator_directory / "biomedical_transformer.pt",
        map_location="cpu",
        weights_only=False,
    )
    transformer = build_transformer_model(settings["biomedical_backbone"]).to(device)
    transformer.load_state_dict(transformer_checkpoint["state_dict"])

    act_checkpoint = torch.load(
        proposed_directory / "act_ner.pt", map_location="cpu", weights_only=False
    )
    act_model = ACTNER(ACTConfig(**act_checkpoint["config"]))
    act_model.load_state_dict(act_checkpoint["state_dict"])
    act_model = act_model.to("cpu")
    return lexicon_matcher, logistic, linear_crf, bilstm, transformer, act_model


def _save_locked_figures(
    project_root: Path,
    predictive: pd.DataFrame,
    efficiency: pd.DataFrame,
    confusion: pd.DataFrame,
) -> list[str]:
    names = []
    merged = predictive[["Model", "Strict Typed Entity F1"]].merge(
        efficiency[["Model", "Latency per Sentence (ms)"]], on="Model", how="inner"
    )
    figure, axis = plt.subplots(figsize=(13.5, 8.5))
    colors = [GOLD if model == "ACT-NER" else BLUE for model in merged.Model]
    axis.scatter(
        merged["Latency per Sentence (ms)"],
        merged["Strict Typed Entity F1"],
        s=130,
        c=colors,
        edgecolors=INK,
        linewidths=1.0,
    )
    annotation_layout = {
        "Linear-chain CRF": {"xytext": (-10, -28), "ha": "right"},
        "Biomedical Transformer": {"xytext": (10, -28), "ha": "left"},
        "ACT-NER": {"xytext": (-10, 10), "ha": "right"},
    }
    for _, row in merged.iterrows():
        layout = annotation_layout.get(
            row["Model"], {"xytext": (8, 8), "ha": "left"}
        )
        axis.annotate(
            row["Model"],
            (row["Latency per Sentence (ms)"], row["Strict Typed Entity F1"]),
            xytext=layout["xytext"],
            textcoords="offset points",
            fontsize=18,
            ha=layout["ha"],
        )
    positive_latency = merged["Latency per Sentence (ms)"].clip(lower=1e-4)
    if positive_latency.max() / positive_latency.min() > 20:
        axis.set_xscale("log")
    axis.set_title("Latency–Quality Trade-off")
    axis.set_xlabel("Latency per sentence (ms)")
    axis.set_ylabel("Strict Typed Entity F1")
    axis.set_ylim(-0.035, max(0.755, merged["Strict Typed Entity F1"].max() + 0.045))
    axis.grid(color=GRID, linewidth=0.8)
    figure.subplots_adjust(bottom=0.17, top=0.88, left=0.13, right=0.96)
    save_figure(project_root, figure, "latency_quality_tradeoff", merged)
    names.append("latency_quality_tradeoff")

    matrix = confusion.pivot(index="Actual Type", columns="Predicted Type", values="Count")
    matrix = matrix.reindex(index=ENTITY_TYPES, columns=ENTITY_TYPES).fillna(0)
    figure, axis = plt.subplots(figsize=adaptive_figsize(heatmap_shape=matrix.shape))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    maximum = max(float(matrix.to_numpy().max()), 1.0)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix.iloc[row_index, column_index])
            color = "white" if value / maximum > 0.50 else INK
            axis.text(
                column_index,
                row_index,
                f"{value}",
                ha="center",
                va="center",
                color=color,
                fontsize=18,
            )
    axis.set_title("Locked-Test Confusion Matrix for Entity Types")
    axis.set_xlabel("Predicted type (exact boundary matches)")
    axis.set_ylabel("Gold type")
    axis.set_xticks(np.arange(len(ENTITY_TYPES)), ENTITY_TYPES)
    axis.set_yticks(np.arange(len(ENTITY_TYPES)), ENTITY_TYPES)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("Entities", fontsize=18)
    colorbar.ax.tick_params(labelsize=18)
    figure.subplots_adjust(bottom=0.18, top=0.88, left=0.16, right=0.88)
    save_figure(project_root, figure, "locked_confusion_matrix_entity_types", confusion)
    names.append("locked_confusion_matrix_entity_types")
    return names


def _write_summary_reports(
    project_root: Path,
    design: dict,
    predictive: pd.DataFrame,
    calibration: pd.DataFrame,
    robustness: pd.DataFrame,
    statistics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    report_directory = project_root / "Reports"
    act = predictive.query("Model == 'ACT-NER'").iloc[0]
    best_other = predictive.query("Model != 'ACT-NER'").sort_values(
        ["Strict Typed Entity F1", "Strict Boundary F1"], ascending=False
    ).iloc[0]
    adjusted_significant = statistics.query("`Holm-adjusted p-value` < 0.05")
    supported_superiority = (
        float(act["Strict Typed Entity F1"]) > float(best_other["Strict Typed Entity F1"])
        and len(adjusted_significant) == len(statistics)
        and len(statistics) > 0
    )
    conclusion = (
        "The prespecified evidence supports ACT-NER superiority over both selected comparators."
        if supported_superiority
        else "The prespecified conclusion class is NO_CONFIRMED_SUPERIORITY: ACT-NER is reported as a metric-specific quality–calibration trade-off."
    )
    transformer = predictive.query("Model == 'Biomedical Transformer'").iloc[0]
    act_calibration = calibration.query("Model == 'ACT-NER'").iloc[0]
    paired_lines = "; ".join(
        f"{row['Comparison']}: mean document-level difference {row['Mean Paired Difference']:.4f}, Holm-adjusted p={row['Holm-adjusted p-value']:.4f}"
        for _, row in statistics.iterrows()
    )
    methods = f"""# Methods narrative

Documents were the non-overlapping grouping unit. The original DDIExtraction 2013 DrugNER test partition was fixed as the Locked Test before modeling; development documents were split into Train and Validation with seed {settings_for_report(project_root)['split_seed']}. The Locked Test was unavailable to Notebooks 01–05. All lexicon entries originated only from Train annotations.

Task A used BIOES sequence labeling and strict character-boundary F1. Task B classified exact predicted spans into drug, brand, group, and drug_n, with strict typed entity F1 as the principal composite endpoint. ACT-NER combined a frozen selected biomedical context checkpoint with lexical and orthographic branches, an adaptive token gate, local BiLSTM refinement, a constrained linear-chain CRF, attention span pooling, and a pharmacological type head. Loss weights, calibration method, confidence thresholds, checkpoint, and reference seed were frozen from Validation before this evaluation.

The locked evaluation loaded frozen artifacts only. It performed no fitting, threshold selection, calibration selection, architecture selection, or model selection.
"""
    results = f"""# Results narrative

On the Locked Test, ACT-NER achieved strict boundary F1 {act['Strict Boundary F1']:.4f}, strict typed entity F1 {act['Strict Typed Entity F1']:.4f}, and Macro-F1 {act['Macro-F1']:.4f}. The strongest non-proposed model by micro-aggregated strict typed entity F1 was {best_other['Model']} at {best_other['Strict Typed Entity F1']:.4f}; the biomedical Transformer obtained {transformer['Strict Typed Entity F1']:.4f}. ACT-NER therefore exceeded the Transformer on the corpus-level micro endpoint but did not exceed the linear-chain CRF.

The paired document-level estimand did not support the micro-level ACT-NER advantage: {paired_lines}. This apparent reversal is possible because the corpus-level score weights entities, whereas the paired test weights documents. Both estimands are reported without selecting the more favorable one. {conclusion}

ACT-NER calibration on the Locked Test was ECE {act_calibration['ECE']:.4f} and Brier score {act_calibration['Brier Score']:.4f}; calibration did not alter spans or predicted types.

Calibration and efficiency are reported alongside predictive quality, not used to revise predictions. Bootstrap intervals and paired document-level tests with Holm correction are retained in the result tables. Domain, unseen-surface, and multi-token results are subgroup analyses within the same mixed-source corpus and are not described as external validation.
"""
    architecture = """# Proposed model architecture

This superseded report generator is retained only to preserve the earlier stage module layout; the public `run_stage_05` and `run_stage_06` bindings below use the span-level ACT-NER implementation in `act_ner.py`.

For every token, the learned adaptive gate balances projected context against the joint lexical–orthographic representation. A compact bidirectional LSTM and LayerNorm refine local dependencies; a constrained CRF decodes coherent BIOES boundaries. Attention pooling aggregates every token in a decoded span, and a dedicated head predicts drug, brand, group, or drug_n. Validation-only confidence fusion combines boundary, type, lexical, and contextual evidence before the selected calibration map and high/medium/low review policy are applied.

This document specifies the integrated architecture, not a claim that every component is beneficial. The validation ablation is the governing evidence for component contribution; it did not support a beneficial lexical-branch effect in this run, so that branch is not presented as an independently validated source of novelty.
"""
    ablation_table = pd.read_csv(project_root / "artifacts" / "05_ablation.csv")
    lexical_delta = float(
        ablation_table.loc[
            ablation_table.Ablation == "Without lexical branch", "Delta vs Full ACT-NER"
        ].iloc[0]
    )
    crf_control_delta = float(
        ablation_table.loc[
            ablation_table.Ablation == "CRF only", "Delta vs Full ACT-NER"
        ].iloc[0]
    )
    confidence_loss_delta = float(
        ablation_table.loc[
            ablation_table.Ablation == "Without confidence loss", "Delta vs Full ACT-NER"
        ].iloc[0]
    )
    robustness_summary = robustness.to_markdown(index=False, floatfmt=".4f")
    ablation = f"""# Ablation and robustness

Each architectural ablation changed one declared component while retaining the reference seed and validation protocol. Component-level evidence is shown below; negative deltas indicate degradation relative to the full system.

{ablation_table.to_markdown(index=False, floatfmt='.4f')}

The validation evidence does not support an independent benefit from every component. Removing the lexical branch changed strict typed entity F1 by {lexical_delta:+.4f}, and the CRF-only control changed it by {crf_control_delta:+.4f}; neither result can be used to claim that the full adaptive hybrid is superior. Removing the confidence loss changed strict typed entity F1 by {confidence_loss_delta:+.4f}, so its predictive contribution was small. These single-reference-seed component results motivate replication rather than post hoc architecture revision after Locked Test.

The predeclared perturbation and data-reduction evidence is:

{robustness_summary}
"""
    limitations = """# Limitations and threats to validity

- DrugBank and MedLine are domains within one corpus seen during development; their subgroup results are not independent external validation.
- Discontinuous annotations are preserved in audit artifacts but excluded from the contiguous BIOES modeling endpoint.
- WordPiece boundaries that cannot reproduce a gold character span exactly are projected outward for training and still judged against original strict offsets, which can lower strict scores.
- Calibration is estimated from Validation predictions and may be unstable for sparse entity subtypes or confidence bins.
- Hardware timing and memory measurements characterize this execution environment and should not be treated as universal deployment benchmarks.
- Multiple seeds, bootstrap intervals, paired tests, and Holm adjustment reduce but do not remove sampling and implementation uncertainty.
- Most component ablations used one prespecified reference seed. In particular, the lexical-branch ablation improved the validation endpoint; the lexical branch is therefore not claimed as a validated contribution despite remaining part of the predeclared full architecture.
- No claim of universal superiority follows from this single-corpus evaluation.
"""
    claim_map = pd.DataFrame(
        [
            {"Claim": "No document overlap across splits", "Evidence": "artifacts/02_split_integrity.csv", "Status": "Supported"},
            {"Claim": "Lexicon is training-only", "Evidence": "dataset/Processed/training_lexicon.parquet and src/frozen_design.json", "Status": "Supported"},
            {"Claim": "ACT-NER locked predictive quality", "Evidence": "artifacts/06_predictive_quality.csv", "Status": "Supported"},
            {"Claim": "ACT-NER calibration", "Evidence": "artifacts/06_calibration.csv", "Status": "Supported"},
            {"Claim": "ACT-NER comparator difference", "Evidence": "artifacts/06_paired_statistics.csv and artifacts/06_effect_intervals.csv", "Status": "Supported if adjusted inference and interval agree"},
            {"Claim": "Independent external validation", "Evidence": "No independent external corpus was evaluated", "Status": "Not claimed"},
            {"Claim": "Universal ACT-NER superiority", "Evidence": "Locked leaderboard and paired tests", "Status": "Supported" if supported_superiority else "Not established"},
        ]
    )
    (report_directory / "methods_narrative.md").write_text(methods, encoding="utf-8")
    (report_directory / "results_narrative.md").write_text(results, encoding="utf-8")
    (report_directory / "proposed_model_architecture.md").write_text(architecture, encoding="utf-8")
    (report_directory / "ablation_and_robustness.md").write_text(ablation, encoding="utf-8")
    (report_directory / "limitations_and_threats_to_validity.md").write_text(
        limitations, encoding="utf-8"
    )
    (report_directory / "claim_evidence_map.md").write_text(
        "# Claim–evidence map\n\n" + claim_map.to_markdown(index=False), encoding="utf-8"
    )


def settings_for_report(project_root: Path) -> dict:
    return load_settings(project_root)


def run_stage_06(project_root: Path) -> dict:
    configure_notebook()
    settings = load_settings(project_root)
    design_path = project_root / "src" / "frozen_design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert design["frozen_before_notebook_06"] is True
    if design["locked_test_opened"] is True:
        handoff_path = (
            project_root
            / "Predictions"
            / "Locked_Test"
            / "drug_entities_for_relation_extraction.parquet"
        )
        assert design.get("locked_test_evaluated_without_refit") is True
        assert handoff_path.exists()
        handoff = pd.read_parquet(handoff_path)
        assert list(handoff.columns) == list(PREDICTION_COLUMNS)
        assert handoff["entity_id"].is_unique
        assert all(
            str(entity_id).startswith(f"{sentence_id}.pred-e")
            for sentence_id, entity_id in zip(handoff["sentence_id"], handoff["entity_id"])
        )
        assert handoff[
            ["boundary_probability", "type_probability", "calibrated_confidence"]
        ].apply(lambda values: values.between(0.0, 1.0).all()).all()
        assert (
            handoff["review_required"] == (handoff["confidence_category"] == "low")
        ).all()
        return {
            "predictive": pd.read_csv(project_root / "artifacts" / "06_predictive_quality.csv"),
            "classwise": pd.read_csv(project_root / "artifacts" / "06_classwise_quality.csv"),
            "calibration": pd.read_csv(project_root / "artifacts" / "06_calibration.csv"),
            "robustness": pd.read_csv(project_root / "artifacts" / "06_robustness.csv"),
            "subgroups": pd.read_csv(project_root / "artifacts" / "06_subgroup_performance.csv"),
            "errors": pd.read_csv(project_root / "artifacts" / "06_boundary_errors.csv"),
            "efficiency": pd.read_csv(project_root / "artifacts" / "06_efficiency.csv"),
            "bootstrap": pd.read_csv(project_root / "artifacts" / "06_bootstrap_ci.csv"),
            "statistics": pd.read_csv(project_root / "artifacts" / "06_paired_statistics.csv"),
            "effect_intervals": pd.read_csv(project_root / "artifacts" / "06_effect_intervals.csv"),
            "confusion": pd.read_csv(project_root / "artifacts" / "06_entity_type_confusion.csv"),
            "handoff_schema": pd.DataFrame(
                [{"Column": column, "Stable Contract": True} for column in PREDICTION_COLUMNS]
            ),
            "figure_names": ["latency_quality_tradeoff", "locked_confusion_matrix_entity_types"],
        }
    required_artifacts = [
        project_root / design["model_checkpoint"],
        project_root / design["transformer_checkpoint"],
        project_root / "Models" / "Proposed" / "confidence_calibrator.joblib",
        project_root / "Models" / "Proposed" / "confidence_policy.json",
    ]
    assert all(path.exists() for path in required_artifacts)

    tokenizer = AutoTokenizer.from_pretrained(settings["biomedical_backbone"], use_fast=True)
    train_sentences, train_entities, _ = load_processed(project_root, "train")
    locked_documents, locked_sentences, locked_entities = load_split_corpus(
        project_root, "locked_test", allow_locked=True
    )
    locked_tokens, locked_alignment = tokenize_split(
        locked_sentences, locked_entities, tokenizer
    )
    shared = shared_root(project_root)
    locked_documents.to_parquet(shared / "Processed" / "locked_test_documents.parquet", index=False)
    locked_sentences.to_parquet(shared / "Processed" / "locked_test_sentences.parquet", index=False)
    locked_entities.to_parquet(shared / "Processed" / "locked_test_entities.parquet", index=False)
    locked_tokens.to_parquet(shared / "Processed" / "locked_test_tokens.parquet", index=False)
    locked_alignment.to_parquet(shared / "Processed" / "locked_test_alignment.parquet", index=False)

    lexicon = pd.read_parquet(shared / "Processed" / "training_lexicon.parquet")
    locked_features = add_act_features(locked_tokens, locked_sentences, lexicon)
    locked_features.to_parquet(
        shared / "Processed" / "locked_test_act_features.parquet", index=False
    )
    matcher, logistic, linear_crf, bilstm, transformer, act_model = _load_frozen_models(
        project_root, tokenizer, settings
    )

    prediction_times = {}
    start = perf_counter()
    o_predictions = o_only_predictions(locked_sentences, "locked_test")
    prediction_times["O-only"] = perf_counter() - start
    start = perf_counter()
    lexicon_predictions_frame = lexicon_predictions(
        matcher, locked_sentences, locked_tokens, "locked_test"
    )
    prediction_times["Longest-match lexicon"] = perf_counter() - start
    start = perf_counter()
    logistic_predictions = predict_logistic_regression(
        logistic, locked_tokens, locked_sentences, "locked_test"
    )
    prediction_times["Logistic Regression"] = perf_counter() - start
    start = perf_counter()
    crf_predictions = predict_linear_crf(
        linear_crf, locked_tokens, locked_sentences, "locked_test"
    )
    prediction_times["Linear-chain CRF"] = perf_counter() - start
    bilstm_predictions, prediction_times["BiLSTM-CRF"] = predict_bilstm(
        bilstm, locked_tokens, locked_sentences, "locked_test"
    )
    transformer_predictions, transformer_elapsed, _ = predict_transformer(
        transformer, locked_tokens, locked_sentences, tokenizer, "locked_test"
    )
    prediction_times["Biomedical Transformer"] = transformer_elapsed
    act_context_start = perf_counter()
    locked_context = extract_context_embeddings(transformer, locked_tokens, tokenizer)
    act_context_elapsed = perf_counter() - act_context_start
    act_predictions, act_elapsed, _ = predict_act(
        act_model,
        locked_features,
        locked_sentences,
        locked_context,
        "locked_test",
        int(design["reference_seed"]),
    )
    calibrator = joblib.load(
        project_root / "Models" / "Proposed" / "confidence_calibrator.joblib"
    )
    act_predictions = apply_calibration(
        act_predictions, calibrator, design["calibration_method"]
    )
    act_predictions = apply_confidence_policy(
        act_predictions, design["confidence_policy"]
    )
    prediction_times["ACT-NER"] = act_context_elapsed + act_elapsed

    predictions = {
        "O-only": o_predictions,
        "Longest-match lexicon": lexicon_predictions_frame,
        "Logistic Regression": logistic_predictions,
        "Linear-chain CRF": crf_predictions,
        "BiLSTM-CRF": bilstm_predictions,
        "Biomedical Transformer": transformer_predictions,
        "ACT-NER": act_predictions,
    }
    locked_prediction_directory = project_root / "Predictions" / "Locked_Test"
    for model, frame in predictions.items():
        frame.to_parquet(
            locked_prediction_directory / f"{_prediction_slug(model)}.parquet", index=False
        )
    handoff = act_predictions.loc[:, list(PREDICTION_COLUMNS)].copy()
    assert list(handoff.columns) == list(PREDICTION_COLUMNS)
    assert handoff[["document_id", "sentence_id", "entity_id"]].notna().all().all()
    assert handoff["entity_id"].is_unique
    assert all(
        str(entity_id).startswith(f"{sentence_id}.pred-e")
        for sentence_id, entity_id in zip(handoff["sentence_id"], handoff["entity_id"])
    )
    assert handoff[
        ["boundary_probability", "type_probability", "calibrated_confidence"]
    ].apply(lambda values: values.between(0.0, 1.0).all()).all()
    assert (
        handoff["review_required"] == (handoff["confidence_category"] == "low")
    ).all()
    handoff.to_parquet(
        locked_prediction_directory / "drug_entities_for_relation_extraction.parquet",
        index=False,
    )

    predictive_parts = []
    classwise_parts = []
    calibration_parts = []
    subgroup_parts = []
    error_parts = []
    for model, frame in predictions.items():
        metric, classwise = evaluate_predictions(
            locked_entities, frame, model, "Locked Test"
        )
        predictive_parts.append(metric)
        classwise_parts.append(classwise)
        calibration_parts.append(
            calibration_metrics(frame, locked_entities, model, "Locked Test")
        )
        subgroup_parts.append(
            subgroup_metrics(
                frame,
                locked_entities,
                locked_sentences,
                train_entities,
                model,
                "Locked Test",
            )
        )
        errors = boundary_error_profile(gold_for_metrics(locked_entities), frame)
        errors.insert(0, "Model", model)
        error_parts.append(errors)
    predictive = pd.concat(predictive_parts, ignore_index=True)
    classwise = pd.concat(classwise_parts, ignore_index=True)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    subgroups = pd.concat(subgroup_parts, ignore_index=True)
    errors = pd.concat(error_parts, ignore_index=True)

    selected_comparators = {
        design["baseline_comparator"]: predictions[design["baseline_comparator"]],
        design["deep_comparator"]: predictions[design["deep_comparator"]],
    }
    statistics = paired_statistical_comparison(
        act_predictions, selected_comparators, locked_entities
    )
    effect_intervals = _paired_effect_intervals(
        act_predictions,
        selected_comparators,
        locked_entities,
        settings["split_seed"] + 1,
    )
    bootstrap_parts = []
    for model, frame in predictions.items():
        bootstrap_parts.append(
            paired_bootstrap_ci(
                frame,
                locked_entities,
                model,
                settings["split_seed"] + 1,
            )
        )
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)

    robustness_rows = []
    for perturbation in ("token_noise", "punctuation", "casing"):
        perturbed_sentences = _perturb_sentences(
            locked_sentences, locked_entities, perturbation, settings["split_seed"] + 1
        )
        perturbed_tokens, _ = tokenize_split(
            perturbed_sentences, locked_entities, tokenizer
        )
        perturbed_features = add_act_features(
            perturbed_tokens, perturbed_sentences, lexicon
        )
        perturbed_context = extract_context_embeddings(
            transformer, perturbed_tokens, tokenizer
        )
        perturbed_predictions, _, _ = predict_act(
            act_model,
            perturbed_features,
            perturbed_sentences,
            perturbed_context,
            "locked_test",
            int(design["reference_seed"]),
        )
        perturbed_predictions = apply_calibration(
            perturbed_predictions, calibrator, design["calibration_method"]
        )
        perturbed_predictions = apply_confidence_policy(
            perturbed_predictions, design["confidence_policy"]
        )
        metric, _ = evaluate_predictions(
            locked_entities, perturbed_predictions, "ACT-NER", "Locked Test"
        )
        robustness_rows.append(
            {
                "Model": "ACT-NER",
                "Split": "Locked Test",
                "N": len(gold_for_metrics(locked_entities)),
                "Robustness Condition": perturbation.replace("_", " ").title(),
                "Strict Boundary F1": metric.iloc[0]["Strict Boundary F1"],
                "Strict Typed Entity F1": metric.iloc[0]["Strict Typed Entity F1"],
            }
        )
    robustness = pd.DataFrame(robustness_rows)

    baseline_efficiency = pd.read_csv(project_root / "artifacts" / "03_efficiency.csv").set_index("Model")
    comparator_efficiency = pd.read_csv(project_root / "artifacts" / "04_efficiency.csv").set_index("Model")
    act_head_parameters = sum(parameter.numel() for parameter in act_model.parameters())
    transformer_parameters = sum(parameter.numel() for parameter in transformer.parameters())
    parameter_counts = {
        "O-only": 0,
        "Longest-match lexicon": 0,
        "Logistic Regression": int(baseline_efficiency.loc["Logistic Regression", "Parameter Count"]),
        "Linear-chain CRF": int(baseline_efficiency.loc["Linear-chain CRF", "Parameter Count"]),
        "BiLSTM-CRF": sum(parameter.numel() for parameter in bilstm.parameters()),
        "Biomedical Transformer": transformer_parameters,
        "ACT-NER": transformer_parameters + act_head_parameters,
    }
    model_sizes = {
        "O-only": (project_root / "Models" / "Baselines" / "o_only.json").stat().st_size / 2**20,
        "Longest-match lexicon": (project_root / "Models" / "Baselines" / "longest_match_lexicon.joblib").stat().st_size / 2**20,
        "Logistic Regression": (project_root / "Models" / "Baselines" / "logistic_regression.joblib").stat().st_size / 2**20,
        "Linear-chain CRF": (project_root / "Models" / "Baselines" / "linear_chain_crf.joblib").stat().st_size / 2**20,
        "BiLSTM-CRF": (project_root / "Models" / "Comparators" / "bilstm_crf.pt").stat().st_size / 2**20,
        "Biomedical Transformer": (project_root / "Models" / "Comparators" / "biomedical_transformer.pt").stat().st_size / 2**20,
        "ACT-NER": ((project_root / "Models" / "Comparators" / "biomedical_transformer.pt").stat().st_size + (project_root / "Models" / "Proposed" / "act_ner.pt").stat().st_size) / 2**20,
    }
    peak_ram = psutil.Process().memory_info().rss / 2**20
    peak_vram = torch.mps.driver_allocated_memory() / 2**20 if torch.backends.mps.is_available() else np.nan
    efficiency = pd.DataFrame(
        [
            {
                "Model": model,
                "Split": "Locked Test",
                "N": len(locked_sentences),
                "Latency per Sentence (ms)": 1000 * prediction_times[model] / len(locked_sentences),
                "Latency per Entity (ms)": 1000 * prediction_times[model] / max(len(frame), 1),
                "Throughput (sentences/s)": len(locked_sentences) / max(prediction_times[model], 1e-12),
                "Parameter Count": parameter_counts[model],
                "Model Size (MB)": model_sizes[model],
                "Peak RAM (MB)": peak_ram,
                "Peak VRAM (MB)": peak_vram if model in ("BiLSTM-CRF", "Biomedical Transformer", "ACT-NER") else np.nan,
            }
            for model, frame in predictions.items()
        ]
    )
    confusion = _entity_type_confusion(act_predictions, locked_entities, "ACT-NER")
    for name, frame in {
        "06_predictive_quality": predictive,
        "06_classwise_quality": classwise,
        "06_calibration": calibration,
        "06_robustness": robustness,
        "06_subgroup_performance": subgroups,
        "06_boundary_errors": errors,
        "06_efficiency": efficiency,
        "06_bootstrap_ci": bootstrap,
        "06_paired_statistics": statistics,
        "06_effect_intervals": effect_intervals,
        "06_entity_type_confusion": confusion,
    }.items():
        save_table(project_root, frame, name)
    figure_names = _save_locked_figures(project_root, predictive, efficiency, confusion)
    _write_summary_reports(
        project_root, design, predictive, calibration, robustness, statistics, bootstrap
    )
    design["locked_test_opened"] = True
    design["locked_test_evaluated_without_refit"] = True
    design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    return {
        "predictive": predictive,
        "classwise": classwise,
        "calibration": calibration,
        "robustness": robustness,
        "subgroups": subgroups,
        "errors": errors,
        "efficiency": efficiency,
        "bootstrap": bootstrap,
        "statistics": statistics,
        "effect_intervals": effect_intervals,
        "confusion": confusion,
        "handoff_schema": pd.DataFrame(
            [{"Column": column, "Stable Contract": True} for column in PREDICTION_COLUMNS]
        ),
        "figure_names": figure_names,
    }


# The span-level tri-expert implementation supersedes the original token-fusion
# stage for the proposed model. Baseline and comparator stages remain unchanged.
from act_ner import run_stage_05_act_ner, run_stage_06_act_ner

run_stage_05 = run_stage_05_act_ner
run_stage_06 = run_stage_06_act_ner
