"""Validation-only ACT-NER experiments with immutable historical comparators."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats

from core import ENTITY_TYPES, PREDICTION_COLUMNS, gold_for_metrics, load_processed, normalize_surface
from act_ner import ACTNERConfig, build_candidate_bundle
from prospective_data import dependency_components, group_folds, split_group_holdout
from prospective_experiments import annotate_candidates, save_bundle
from prospective_model import score, strict_metrics
from prospective_calibration import apply_confidence, fit_confidence_bundle, candidate_targets, ece
from refinement_model import ACTNER, fit_model, component_ablations, decode

RUN = "refinement_20260907"
SEEDS = (17, 29, 43)
SETTINGS = {"max_epochs": 8, "patience": 3, "batch_size": 512,
            "learning_rate": 0.001, "gradient_clip": 1.0,
            "threshold_grid": [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
CALIBRATION = {"methods": ["identity", "temperature", "platt"],
               "selection_folds": 3, "min_positive_and_negative": 10}
MODELS = {"O-only": "o_only", "Longest-match lexicon": "longest_match_lexicon",
          "Logistic Regression": "logistic_regression", "Linear-chain CRF": "linear_chain_crf",
          "BiLSTM-CRF": "bilstm_crf", "Biomedical Transformer": "biomedical_transformer"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def initialize_run(root, force_rebuild_proposed=True, reuse_proposed_cache=False):
    assert force_rebuild_proposed and not reuse_proposed_cache
    directory = root / "Reports" / RUN
    directory.mkdir(exist_ok=True)
    protected = []
    for prefix in ("Models/Baselines", "Models/Comparators", "Predictions/OOF", "Predictions/Validation"):
        protected.extend(p for p in (root / prefix).rglob("*") if p.is_file())
    protected.extend(p for p in (root / "notebooks").glob("0[1-4]*.ipynb"))
    contract = {str(p.relative_to(root)): digest(p) for p in protected}
    path = directory / "protected.json"
    if path.exists():
        assert json.loads(path.read_text()) == contract
    else:
        path.write_text(json.dumps(contract, indent=2))
    archive = directory / "before"
    archive.mkdir(exist_ok=True)
    for path in [*(root / "notebooks").glob("0[56]*.ipynb"), *(root / "Models/Proposed").glob("*.*")]:
        destination = archive / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
    protocol = {"run": RUN, "seeds": list(SEEDS), "outer_folds": 5, "settings": SETTINGS,
        "selection": "Inner document holdout only; best mean inner typed F1 across three seeds",
        "final_seed": 17, "test_role": "Post-hoc Diagnostic Test",
        "upstream_scope": "Historical frozen experts; conditional validation evidence, not independent upstream model selection",
        "architecture": "Trainable residual logits; no anchor override; sparse tri-expert span verifier and type classifier",
        "calibration": CALIBRATION, "gold": "Contiguous, offset_valid; no oracle candidates",
        "ablation_estimand": "Learned evidence contribution conditional on the same frozen candidate union; not total removal of upstream candidate generators",
        "model_source_sha256": digest(root / "src/refinement_model.py"),
        "base_source_sha256": digest(root / "src/prospective_model.py")}
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    (root / "Models/Proposed" / RUN).mkdir(exist_ok=True)
    model_directory = root / "Models/Proposed" / RUN
    if any(model_directory.iterdir()):
        previous = directory / "previous_runs" / str(time.time_ns())
        previous.mkdir(parents=True)
        shutil.move(str(model_directory), str(previous / "models"))
        model_directory.mkdir()
        for item in [*directory.glob("fold_*.csv"), *directory.glob("selection_fold_*.csv"),
                     root / "src/refinement_freeze.json"]:
            if item.exists():
                shutil.move(str(item), str(previous / item.name))
    return directory


def verify_protected(root):
    contract = json.loads((root / "Reports" / RUN / "protected.json").read_text())
    assert all(digest(root / path) == value for path, value in contract.items())


def development(root):
    sentences, entities, tokens = load_processed(root, "validation")
    gold = gold_for_metrics(entities)
    assert sentences.sentence_id.is_unique and gold.entity_id.is_unique
    documents = sentences[["document_id", "source"]].drop_duplicates().sort_values("document_id")
    documents["dependency_group"] = documents.document_id.map(dependency_components(sentences))
    presence = gold.groupby(["document_id", "entity_type"]).size().unstack(fill_value=0)
    documents["stratum"] = [f"{r.source}|{int(presence.loc[r.document_id].get('brand', 0)>0)}|"
        f"{int(presence.loc[r.document_id].get('drug_n', 0)>0)}" if r.document_id in presence.index else r.source
        for r in documents.itertuples()]
    documents["outer_fold"] = documents.document_id.map(group_folds(documents, 5, 1701))
    membership = root / "Reports" / RUN / "membership.csv"
    if membership.exists():
        pd.testing.assert_frame_equal(pd.read_csv(membership), documents.reset_index(drop=True))
    else:
        documents.to_csv(membership, index=False)
    return sentences, entities, tokens, documents


def role_partition(documents, fold):
    outer = set(documents.loc[documents.outer_fold == fold, "document_id"]) if fold >= 0 else set()
    available = documents.loc[~documents.document_id.isin(outer)]
    grow, calibration = split_group_holdout(available, 0.2, 2301 + fold * 10)
    train, selection = split_group_holdout(available.loc[available.document_id.isin(grow)], 0.2, 2302 + fold * 10)
    roles = {"train": train, "selection": selection, "calibration": calibration, "outer": outer}
    assert set.union(*roles.values()) == set(documents.document_id)
    assert sum(map(len, roles.values())) == len(documents)
    group_sets = {k: set(documents.loc[documents.document_id.isin(v), "dependency_group"]) for k,v in roles.items()}
    assert sum(map(len, group_sets.values())) == len(set.union(*group_sets.values()))
    return roles


def subset_bundle(bundle, identifiers):
    return bundle.subset(np.flatnonzero(bundle.frame.document_id.isin(identifiers)))


def generate_evidence(root, sentences, entities, tokens):
    # Candidate generation receives no gold labels. Targets are joined afterward.
    bundle, _, elapsed = build_candidate_bundle(root, sentences, entities.iloc[:0], tokens, 0.72)
    assert bundle.frame.generated_candidate.all()
    bundle = annotate_candidates(bundle, entities)
    cache = root / "dataset/Interim" / RUN / "validation"
    save_bundle(bundle, cache)
    gold = gold_for_metrics(entities)
    truth = set(zip(gold.sentence_id, gold.char_start, gold.char_end))
    rows = []
    for name, mask in [("Transformer", bundle.frame.origin_transformer),
                       ("CRF", bundle.frame.origin_crf), ("Candidate union", bundle.frame.generated_candidate)]:
        part = bundle.frame.loc[mask]
        keys = set(zip(part.sentence_id, part.char_start, part.char_end))
        rows.append({"Source": name, "N": len(truth), "Candidates": len(keys), "Boundary recall": len(keys & truth)/len(truth)})
    result = pd.DataFrame(rows)
    assert result.iloc[-1]["Boundary recall"] >= result.iloc[:-1]["Boundary recall"].max()
    return bundle, result, elapsed


def configurations():
    base = ACTNERConfig(continual_adaptation=False, lambda_consistency=0.0,
        shift_triggered=False, boundary_threshold=0.5, use_lexical_reliability=True)
    return {"Joint-A": replace(base, dropout=0.10, gate_hidden_size=128, type_loss="effective_number"),
            "Joint-B": replace(base, dropout=0.15, gate_hidden_size=192, type_loss="focal"),
            "Joint-C": replace(base, dropout=0.25, gate_hidden_size=256, type_loss="logit_adjusted")}


def fit_unit(root, bundle, sentences, entities, roles, fold, name, config, seed, callback=None):
    safe_name = name.lower().replace(" ", "_")
    target = root / "Models/Proposed" / RUN / f"fold_{fold}" / safe_name / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    assert not (target / "model.pt").exists(), "This experiment must be a fresh fit, not a historical replay"
    select_sentences = sentences.loc[sentences.document_id.isin(roles["selection"])]
    select_entities = entities.loc[entities.document_id.isin(roles["selection"])]
    train_bundle = subset_bundle(bundle, roles["train"])
    select_bundle = subset_bundle(bundle, roles["selection"])
    model, history, threshold, epoch, elapsed = fit_model(train_bundle, select_bundle, roles["train"],
        select_sentences, select_entities, config, seed, SETTINGS, callback)
    checkpoint = {"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
        "config": asdict(config), "threshold": threshold, "seed": seed, "epoch": epoch,
        "run": RUN, "roles": {k:sorted(v) for k,v in roles.items()}, "trained_at_ns": time.time_ns()}
    torch.save(checkpoint, target / "model.pt")
    history.to_csv(target / "history.csv", index=False)
    info = {"configuration": name, "seed": seed, "fold": fold, "threshold": threshold,
            "selection_typed_f1": float(history.selection_typed_f1.max()), "epoch": epoch,
            "training_seconds": elapsed, "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "checkpoint": str((target / "model.pt").relative_to(root))}
    (target / "fit.json").write_text(json.dumps(info, indent=2))
    return model, info, target


def prediction_diagnostics(prediction, entities):
    truth = gold_for_metrics(entities)
    keys = set(zip(truth.sentence_id, truth.char_start, truth.char_end, truth.entity_type))
    correct = np.array([key in keys for key in prediction[["sentence_id","char_start","char_end","predicted_type"]].itertuples(index=False,name=None)])
    confidence = prediction.calibrated_confidence.to_numpy(float)
    raw = (prediction.boundary_probability * prediction.type_probability).to_numpy(float)
    return {"ece": ece(confidence, correct), "raw_ece": ece(raw, correct),
            "brier": float(np.mean((confidence-correct)**2)) if len(correct) else np.nan}


def evaluate_unit(model, bundle, sentences, entities, documents, roles, seed, target):
    calibration_bundle = subset_bundle(bundle, roles["calibration"])
    calibration_scores, _ = score(model, calibration_bundle)
    calibration_docs = documents.loc[documents.document_id.isin(roles["calibration"])]
    mapping, calibration_table = fit_confidence_bundle(calibration_scores, calibration_docs, CALIBRATION)
    joblib.dump(mapping, target / "calibrator.joblib")
    calibration_table.to_csv(target / "calibration.csv", index=False)
    evaluation = subset_bundle(bundle, roles["outer"])
    scores, elapsed = score(model, evaluation)
    observed_sentences = sentences.loc[sentences.document_id.isin(roles["outer"])]
    observed_entities = entities.loc[entities.document_id.isin(roles["outer"])]
    threshold = json.loads((target / "fit.json").read_text())["threshold"]
    prediction = decode(scores, observed_sentences, model.config, threshold, seed, "Validation cross-fit")
    prediction = apply_confidence(prediction, mapping)
    prediction.to_parquet(target / "predictions.parquet", index=False)
    scores.to_parquet(target / "scores.parquet", index=False)
    metric = {**strict_metrics(observed_entities, prediction), **prediction_diagnostics(prediction, observed_entities),
              "head_inference_seconds": elapsed, "sentences": len(observed_sentences)}
    return prediction, scores, metric


def nested_fold(root, bundle, sentences, entities, documents, fold, callback=None):
    original_frame = bundle.frame.copy(deep=True)
    roles = role_partition(documents, fold)
    candidates, fitted, selection = configurations(), {}, []
    for name, config in candidates.items():
        for seed in SEEDS:
            report = (lambda row, n=name, s=seed: callback(fold,n,s,row)) if callback else None
            model, info, target = fit_unit(root,bundle,sentences,entities,roles,fold,name,config,seed,report)
            fitted[(name,seed)] = (info,target)
            selection.append(info)
            del model
    selection_table = pd.DataFrame(selection)
    means = selection_table.groupby("configuration").selection_typed_f1.mean()
    selected = str(means.sort_values(ascending=False, kind="stable").index[0])
    selection_table.to_csv(root / "Reports" / RUN / f"selection_fold_{fold}.csv", index=False)
    variants = {"Full ACT-NER": candidates[selected], **component_ablations(candidates[selected])}
    results = []
    for name, config in variants.items():
        for seed in SEEDS:
            if name == "Full ACT-NER":
                info, target = fitted[(selected,seed)]
                model = load_model(target / "model.pt")
            else:
                report = (lambda row, n=name, s=seed: callback(fold,n,s,row)) if callback else None
                model, info, target = fit_unit(root,bundle,sentences,entities,roles,fold,name,config,seed,report)
            _, _, metrics = evaluate_unit(model,bundle,sentences,entities,documents,roles,seed,target)
            results.append({**info, **metrics, "configuration":name, "selected_joint":selected})
            del model
    table = pd.DataFrame(results)
    table.to_csv(root / "Reports" / RUN / f"fold_{fold}.csv", index=False)
    verify_protected(root)
    pd.testing.assert_frame_equal(bundle.frame, original_frame)
    return table


def load_model(path):
    pack = torch.load(path, map_location="cpu", weights_only=False)
    model = ACTNER(ACTNERConfig(**pack["config"]))
    model.load_state_dict(pack["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(device).eval()
