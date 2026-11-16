"""Additional same-corpus holdout: explicit provenance and frozen-only access."""
import contextlib
import io
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from core import (parse_xml_files, tokenize_split, load_processed, load_settings,
                  predictions_from_tag_sequences, gold_for_metrics)
from act_ner import extract_frozen_expert_evidence, build_candidate_bundle
from prospective_data import normalized_sentence, dependency_components
from prospective_model import score, strict_metrics
from refinement_model import decode
from prospective_calibration import apply_confidence
from refinement_experiment import RUN, digest, load_model, prediction_diagnostics, verify_protected


def seal_holdout_plan(root):
    """Inspect only root IDs/paths; do not read sentence or entity annotations."""
    shared = root / "dataset"
    base = shared / "Raw/DDICorpus-2013/DDICorpus/Test/Test for DDI Extraction task"
    rows = []
    for path in sorted(base.glob("*/*.xml")):
        with path.open("rb") as stream:
            _, element = next(ET.iterparse(stream, events=("start",)))
            identifier = element.attrib["id"]
        rows.append({"document_id": identifier, "source": path.parent.name,
                     "relative_xml": str(path.relative_to(shared)), "sha256": digest(path)})
    manifest = pd.DataFrame(rows)
    assert len(manifest) == 191 and manifest.document_id.is_unique
    for split in ("train", "validation", "locked_test"):
        known = pd.read_csv(shared / "Splits" / f"{split}_documents.csv")
        assert not set(known.document_id) & set(manifest.document_id)
    directory = root / "Reports" / RUN
    target = directory / "additional_holdout_plan.json"
    if target.exists():
        previous = json.loads(target.read_text())
        assert previous["files"] == rows
        return previous
    plan = {"created_at_ns": time.time_ns(), "files": rows,
        "role": "Additional reserved same-corpus holdout",
        "provenance": "Recorded prior access was root-ID inventory only; no reserve training or predictions found in inspected code and artifacts. Absence of unrecorded exposure cannot be certified.",
        "source": "https://aclanthology.org/S13-2056/",
        "exclusion": "Exclude whole reserve documents sharing an exact normalized sentence (at least 40 characters and 6 whitespace tokens) with any original Train, Validation, or Historical Test document. Report all-document results separately as sensitivity.",
        "eligibility": "Contiguous offset-valid gold entities, unchanged four-type schema",
        "primary_model_seed": 17, "selection_on_this_holdout": False,
        "comparisons": ["ACT-NER vs Linear-chain CRF", "ACT-NER vs Biomedical Transformer"],
        "primary_statistic": "Typed corpus micro-F1, paired dependency-cluster bootstrap and randomization, Holm adjustment",
        "label_access": "Only Notebook 06, after final model and calibrator hashes exist"}
    target.write_text(json.dumps(plan, indent=2))
    (directory / "additional_holdout_plan.sha256").write_text(digest(target))
    return plan


def additional_population(root):
    seal = json.loads((root / "src/refinement_freeze.json").read_text())
    assert seal["all_seeds_completed"] and seal["all_ablations_completed"]
    assert all(digest(root/"src"/name) == value for name, value in seal["inference_source_sha256"].items())
    path = root / "Reports" / RUN / "additional_holdout_plan.json"
    assert digest(path) == path.with_suffix(".sha256").read_text()
    plan = json.loads(path.read_text())
    checkpoint = torch.load(root / seal["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["trained_at_ns"] > plan["created_at_ns"]
    for kind in ("checkpoint", "calibrator"):
        assert digest(root / seal[kind]) == seal[f"{kind}_sha256"]
    shared = root / "dataset"
    paths = [shared / row["relative_xml"] for row in plan["files"]]
    assert all(digest(p) == row["sha256"] for p, row in zip(paths, plan["files"]))
    # Record first label access before parsing; later evaluations cannot erase it.
    access_directory = root / "Reports" / RUN / "additional_label_access"
    access_directory.mkdir(exist_ok=True)
    access = {"accessed_at_ns": time.time_ns(), "checkpoint_sha256": seal["checkpoint_sha256"],
              "calibrator_sha256": seal["calibrator_sha256"], "plan_sha256": digest(path),
              "purpose": "Frozen Notebook-06 evaluation; no model selection"}
    with (access_directory / f"{access['accessed_at_ns']}.json").open("x") as stream:
        json.dump(access, stream, indent=2)
    documents, sentences, entities = parse_xml_files(paths, "additional_holdout")
    known = set()
    for split in ("train", "validation", "locked_test"):
        previous = pd.read_parquet(shared / "Processed" / f"{split}_sentences.parquet")
        known.update(normalized_sentence(s) for s in previous.text
                     if len(normalized_sentence(s)) >= 40 and len(normalized_sentence(s).split()) >= 6)
    overlap = sentences.text.map(normalized_sentence).isin(known)
    excluded = set(sentences.loc[overlap, "document_id"])
    documents["eligible_independent_text"] = ~documents.document_id.isin(excluded)
    manifest_path = shared / "Splits" / f"{RUN}_additional_holdout_documents.csv"
    documents.to_csv(manifest_path, index=False)
    directory = shared / "Interim" / RUN / "additional_holdout"
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in (("documents", documents), ("sentences", sentences), ("entities", entities)):
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(load_settings(root)["biomedical_backbone"],
                                                 use_fast=True, local_files_only=True)
    # Inference features receive no gold BIOES labels.
    tokens, _ = tokenize_split(sentences, entities.iloc[:0], tokenizer)
    assert tokens.groupby("sentence_id").size().max() + 2 <= 512
    tokens.to_parquet(directory / "tokens.parquet", index=False)
    return documents, sentences, entities, tokens


def additional_inference(root, sentences, entities, tokens):
    seal = json.loads((root / "src/refinement_freeze.json").read_text())
    for kind in ("checkpoint", "calibrator"):
        assert digest(root/seal[kind]) == seal[f"{kind}_sha256"]
    assert all(digest(root/"src"/name) == value for name, value in seal["inference_source_sha256"].items())
    transformer, crf, seconds = extract_frozen_expert_evidence(root, tokens)
    evidence = (transformer, crf, seconds)
    bundle, _, _ = build_candidate_bundle(root, sentences, entities.iloc[:0], tokens, 0.72,
                                          expert_evidence=evidence)
    model = load_model(root / seal["checkpoint"])
    scores, _ = score(model, bundle)
    predictions = {"ACT-NER": {17: apply_confidence(decode(scores, sentences, model.config,
        seal["threshold"], 17, "Additional same-corpus holdout"), joblib.load(root / seal["calibrator"]))}}
    for name, source in (("Biomedical Transformer", transformer), ("Linear-chain CRF", crf)):
        tags = {sid: row["tags"] for sid, row in source.items()}
        # The confidence of the decoded label, including Viterbi labels for CRF.
        from core import TYPED_TAGS
        probabilities = {sid: [float(row["probabilities"][i, TYPED_TAGS.index(tag)])
                              for i, tag in enumerate(row["tags"])] for sid, row in source.items()}
        predictions[name] = {0: predictions_from_tag_sequences(tokens, sentences, tags,
            probabilities, name, "Additional same-corpus holdout")}
    directory = root / "Predictions" / "Additional_Holdout" / RUN
    directory.mkdir(parents=True, exist_ok=True)
    for name, repeats in predictions.items():
        next(iter(repeats.values())).to_parquet(directory / f"{name.replace(' ', '_')}.parquet", index=False)
    scores.to_parquet(directory / "candidate_diagnostics.parquet", index=False)
    verify_protected(root)
    return predictions, scores
