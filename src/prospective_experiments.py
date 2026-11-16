"""Resumable, provenance-checked outer-fold experiments without test access."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from core import ENTITY_TYPES, gold_for_metrics, normalize_surface, stable_sha256
from act_ner import ACTNERConfig, CandidateBundle, TYPE_TO_ID
from prospective_data import (Development, canonical_digest, candidate_evidence,
    concatenate_bundles, fit_experts, group_folds, split_group_holdout, unit_roles, write_once_json)
from prospective_model import (ACTNER, ablations, configuration_candidates, decode,
    fit_head, score, strict_metrics)
from prospective_calibration import apply_confidence, candidate_targets, ece, fit_confidence_bundle


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def annotate_candidates(bundle, entities):
    """Attach targets AFTER candidate generation; never create an oracle span."""
    gold = gold_for_metrics(entities)
    mapping = {(row.sentence_id, row.char_start, row.char_end): TYPE_TO_ID[row.entity_type]
               for row in gold.itertuples(index=False)}
    frame = bundle.frame.copy()
    labels = [mapping.get((row.sentence_id, row.char_start, row.char_end), -1)
              for row in frame.itertuples(index=False)]
    frame["type_label"] = labels
    frame["span_label"] = np.asarray(labels) >= 0
    bundle.frame = frame
    return bundle


def save_bundle(bundle, directory):
    directory.mkdir(parents=True, exist_ok=True)
    bundle.frame.to_parquet(directory / "candidates.parquet", index=False)
    keys = ("context", "transformer", "crf", "orthographic", "lexical", "transformer_type", "crf_type")
    torch.save({key: getattr(bundle, key) for key in keys}, directory / "features.pt")
    write_once_json(directory / "complete.json", {name: stable_sha256(directory / name)
        for name in ("candidates.parquet", "features.pt")})


def load_bundle(directory):
    contract = json.loads((directory / "complete.json").read_text())
    assert all(stable_sha256(directory / name) == value for name, value in contract.items())
    features = torch.load(directory / "features.pt", map_location="cpu", weights_only=True)
    return CandidateBundle(pd.read_parquet(directory / "candidates.parquet"), **features)


def train_frontend_unit(root, data, membership, protocol, repeat, outer_fold, seed, callback=None):
    """Three genuinely OOF expert sets, plus the final fold-training pair."""
    roles = unit_roles(data, membership, protocol, repeat, outer_fold)
    unit_name = f"repeat_{repeat}/fold_{outer_fold}/seed_{seed}"
    checkpoint_root = root / "Models/Comparators/Prospective" / protocol["protocol_id"] / unit_name
    evidence_root = root / "dataset/Interim/Prospective" / protocol["protocol_id"] / unit_name
    head_training = data.subset(roles["head_train"])
    inner = group_folds(head_training.documents, protocol["inner_crossfit_folds"],
                        protocol["partition_seeds"][repeat] + outer_fold * 100 + 2)
    write_once_json(checkpoint_root / "roles.json", {key: sorted(value) for key, value in roles.items()})
    for fold in range(protocol["inner_crossfit_folds"]):
        observed_ids = set(inner.index[inner == fold])
        available_ids = set(inner.index[inner != fold])
        available = data.subset(available_ids)
        fit_ids, stop_ids = split_group_holdout(available.documents,
            protocol["expert_early_stop_fraction"], 901 + repeat * 100 + outer_fold * 10 + fold)
        expert_path = checkpoint_root / f"inner_{fold}"
        if callback:
            callback({"phase": "Cross-fitted experts", "inner_fold": fold + 1})
        fit_experts(root, data, fit_ids, stop_ids, protocol, seed, expert_path, callback)
        cache = evidence_root / f"inner_{fold}"
        if not (cache / "complete.json").exists():
            bundle, _, elapsed = candidate_evidence(root, data.subset(observed_ids),
                data.subset(fit_ids), expert_path, protocol, include_gold_support=True)
            save_bundle(bundle, cache)
            del bundle
    fit_ids, stop_ids = split_group_holdout(head_training.documents,
        protocol["expert_early_stop_fraction"], 1801 + repeat * 100 + outer_fold)
    if callback:
        callback({"phase": "Final fold-training experts", "inner_fold": "final"})
    fit_experts(root, data, fit_ids, stop_ids, protocol, seed, checkpoint_root / "final", callback)
    for role in ("selection", "calibration"):
        cache = evidence_root / role
        if (cache / "complete.json").exists():
            continue
        bundle, predictions, elapsed = candidate_evidence(root, data.subset(roles[role]),
            data.subset(roles["head_train"]), checkpoint_root / "final", protocol, include_gold_support=False)
        bundle = annotate_candidates(bundle, data.subset(roles[role]).entities)
        save_bundle(bundle, cache)
        for name, prediction in predictions.items():
            prediction.to_parquet(cache / f"{slug(name)}.parquet", index=False)
        write_once_json(cache / "efficiency.json", {"expert_inference_seconds": elapsed})
        del bundle
    # There is intentionally NO outer-fold inference in this function.
    return checkpoint_root, evidence_root, roles


def head_result(root, protocol, directory, name, config, seed, training_bundle,
                selection_bundle, roles, data, callback=None):
    directory.mkdir(parents=True, exist_ok=True)
    specification = {"configuration": asdict(config), "seed": seed, "name": name,
        "protocol_sha256": canonical_digest(protocol), "training_ids": sorted(roles["head_train"]),
        "selection_ids": sorted(roles["selection"]),
        "implementation": stable_sha256(root / "src/prospective_model.py")}
    write_once_json(directory / "specification.json", specification)
    if (directory / "complete.json").exists():
        record = json.loads((directory / "complete.json").read_text())
        assert all(stable_sha256(directory / path) == checksum for path, checksum in record["files"].items())
        return record
    selection = data.subset(roles["selection"])
    model, history, threshold, epoch, elapsed = fit_head(training_bundle, selection_bundle,
        roles["head_train"], selection.sentences, selection.entities, config, seed, protocol["head"], callback)
    candidates, latency = score(model, selection_bundle)
    predictions = decode(candidates, selection.sentences, config, threshold, seed, "inner_selection")
    metrics = strict_metrics(selection.entities, predictions)
    for source in ("DrugBank", "MedLine"):
        ids = set(selection.sentences.loc[selection.sentences.source == source, "sentence_id"])
        metrics[f"{source}_f1"] = strict_metrics(selection.entities.loc[selection.entities.sentence_id.isin(ids)],
            predictions.loc[predictions.sentence_id.isin(ids)])["typed_f1"]
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, "config": asdict(config), "threshold": threshold,
                "epoch": epoch, "seed": seed}, directory / "act_ner.pt")
    history.to_csv(directory / "history.csv", index=False)
    predictions.to_parquet(directory / "selection_predictions.parquet", index=False)
    candidates.to_parquet(directory / "selection_candidates.parquet", index=False)
    record = {"metrics": metrics, "threshold": threshold, "epoch": epoch, "train_seconds": elapsed,
              "span_inference_seconds": latency, "parameters_total": sum(p.numel() for p in model.parameters()),
              "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "files": {name: stable_sha256(directory / name) for name in
                  ("act_ner.pt", "history.csv", "selection_predictions.parquet", "selection_candidates.parquet")}}
    write_once_json(directory / "complete.json", record)
    del model, state, candidates
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return record


def train_heads_unit(root, data, protocol, repeat, outer_fold, seed, checkpoint_root, evidence_root, roles, callback=None):
    bundles = [load_bundle(evidence_root / f"inner_{fold}") for fold in range(protocol["inner_crossfit_folds"])]
    training_bundle = concatenate_bundles(bundles)
    del bundles
    selection_bundle = load_bundle(evidence_root / "selection")
    base = root / "Models/Proposed/Prospective" / protocol["protocol_id"] / f"repeat_{repeat}/fold_{outer_fold}/seed_{seed}"
    candidates = configuration_candidates(protocol)
    rows = []
    for index, config in enumerate(candidates):
        if callback:
            callback({"phase": "Joint inner selection", "configuration": index + 1})
        record = head_result(root, protocol, base / f"selection_{index}", "ACT-NER", config,
            seed, training_bundle, selection_bundle, roles, data, callback)
        rows.append({"configuration": index, **record["metrics"], "latency": record["span_inference_seconds"]})
    selection_table = pd.DataFrame(rows)
    near = selection_table.loc[selection_table.typed_f1 >= selection_table.typed_f1.max() - protocol["selection_tolerance_typed_f1"]].copy()
    near["worst_domain"] = near[["DrugBank_f1", "MedLine_f1"]].min(axis=1)
    selected = int(near.sort_values(["boundary_f1", "typed_recall", "macro_f1", "worst_domain",
        "brand_f1", "drug_n_f1", "latency", "configuration"], ascending=[False] * 6 + [True, True]).iloc[0].configuration)
    selection_table.to_csv(base / "joint_selection.csv", index=False)
    variants = ablations(candidates[selected])
    assert list(variants) == protocol["ablations"]
    for name, config in variants.items():
        if callback:
            callback({"phase": "Paired ablation training", "configuration": name})
        head_result(root, protocol, base / slug(name), name, config, seed,
                    training_bundle, selection_bundle, roles, data, callback)
    # All predictive choices are sealed BEFORE any outer-fold inference.
    write_once_json(base / "predictive_freeze.json", {
        "selected_configuration": selected, "selection_documents": sorted(roles["selection"]),
        "outer_documents": sorted(roles["outer"]), "protocol_sha256": canonical_digest(protocol),
        "checkpoints": {name: stable_sha256(base / slug(name) / "act_ner.pt") for name in variants}})
    del training_bundle, selection_bundle
    return base


def load_head(path):
    record = torch.load(path, weights_only=False, map_location="cpu")
    config = ACTNERConfig(**record["config"])
    model = ACTNER(config)
    model.load_state_dict(record["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(device).eval(), config, record["threshold"]


def expert_candidate_scores(bundle, expert):
    rows = bundle.frame.copy()
    evidence = bundle.transformer if expert == "Biomedical Transformer" else bundle.crf
    distribution = bundle.transformer_type if expert == "Biomedical Transformer" else bundle.crf_type
    rows["span_probability"] = evidence[:, 1].numpy()
    rows["type_vector"] = list(distribution.numpy())
    rows["predicted_type"] = [ENTITY_TYPES[i] for i in distribution.argmax(1).tolist()]
    rows["type_probability"] = distribution.max(1).values.numpy()
    rows["entity_probability"] = rows.span_probability * rows.type_probability
    return rows


def seal_calibration(base, evidence_root, roles, data, protocol, callback=None):
    assert (base / "predictive_freeze.json").exists()
    bundle = load_bundle(evidence_root / "calibration")
    mappings = {}
    for name in protocol["ablations"] + ["Linear-chain CRF", "Biomedical Transformer"]:
        if callback:
            callback({"phase": "Independent calibration", "configuration": name})
        folder = base / "calibration" / slug(name)
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists():
            contract = json.loads((folder / "complete.json").read_text())
            assert all(stable_sha256(folder / p) == h for p, h in contract.items())
            mappings[name] = joblib.load(folder / "confidence.joblib")
            continue
        enabled = True
        if name in protocol["ablations"]:
            model, config, _ = load_head(base / slug(name) / "act_ner.pt")
            candidates, _ = score(model, bundle)
            enabled = config.calibration
            del model
        else:
            candidates = expert_candidate_scores(bundle, name)
        mapping, method_table = fit_confidence_bundle(candidates,
            data.subset(roles["calibration"]).documents, protocol["calibration"], enabled)
        joblib.dump(mapping, folder / "confidence.joblib")
        method_table.to_csv(folder / "crossfit_selection.csv", index=False)
        mappings[name] = mapping
        write_once_json(folder / "complete.json", {p: stable_sha256(folder / p)
            for p in ("confidence.joblib", "crossfit_selection.csv")})
    write_once_json(base / "calibration_freeze.json", {"documents": sorted(roles["calibration"]),
        "calibrators": {name: stable_sha256(base / "calibration" / slug(name) / "confidence.joblib") for name in mappings}})
    del bundle
    return mappings


def document_counts(data, prediction, name):
    gold = gold_for_metrics(data.entities)
    truth = {doc: set(zip(frame.sentence_id, frame.char_start, frame.char_end, frame.entity_type))
             for doc, frame in gold.groupby("document_id")}
    predicted = {doc: set(zip(frame.sentence_id, frame.char_start, frame.char_end, frame.predicted_type))
                 for doc, frame in prediction.groupby("document_id")}
    rows = []
    for row in data.documents.itertuples(index=False):
        actual, emitted = truth.get(row.document_id, set()), predicted.get(row.document_id, set())
        boundary_actual, boundary_emitted = {x[:3] for x in actual}, {x[:3] for x in emitted}
        record = {"document_id": row.document_id, "dependency_group": row.dependency_group,
            "source": row.source, "Model": name, "typed_tp": len(actual & emitted),
            "predicted_n": len(emitted), "gold_n": len(actual),
            "boundary_tp": len(boundary_actual & boundary_emitted),
            "boundary_predicted_n": len(boundary_emitted), "boundary_gold_n": len(boundary_actual)}
        for subtype in ENTITY_TYPES:
            a, p = {x for x in actual if x[3] == subtype}, {x for x in emitted if x[3] == subtype}
            record.update({f"{subtype}_tp": len(a & p), f"{subtype}_predicted_n": len(p), f"{subtype}_gold_n": len(a)})
        rows.append(record)
    return pd.DataFrame(rows)


def subgroup_results(training, evaluation, prediction, name):
    seen = set(gold_for_metrics(training.entities).entity_text.map(normalize_surface))
    def enrich(frame, gold):
        result = gold_for_metrics(frame) if gold else frame.copy()
        result["source"] = result.sentence_id.map(evaluation.sentences.set_index("sentence_id").source)
        result["seen"] = result.entity_text.map(normalize_surface).isin(seen)
        # Word-level entity length avoids calling WordPiece fragments multi-word mentions.
        result["multi_token"] = result.entity_text.str.findall(r"\w+|[^\w\s]").str.len() > 1
        return result
    actual, emitted = enrich(evaluation.entities, True), enrich(prediction, False)
    groups = [("DrugBank", "source", "DrugBank"), ("MedLine", "source", "MedLine"),
              ("seen", "seen", True), ("unseen", "seen", False),
              ("single-token", "multi_token", False), ("multi-token", "multi_token", True)]
    return pd.DataFrame([{"Model": name, "Subgroup": label,
        **strict_metrics(actual.loc[actual[column] == value], emitted.loc[emitted[column] == value])}
        for label, column, value in groups])


def confidence_results(candidates, prediction, gold_entities, mapping, name):
    frame = candidate_targets(candidates.loc[candidates.generated_candidate])
    common_probability = mapping["maps"]["entity"].predict(frame.entity_probability)
    truth = set(zip(gold_entities.sentence_id, gold_entities.char_start, gold_entities.char_end, gold_entities.entity_type))
    accepted_target = np.asarray([(row.sentence_id, row.char_start, row.char_end, row.predicted_type) in truth
                                  for row in prediction.itertuples(index=False)], dtype=int)
    confidence = prediction.calibrated_confidence.to_numpy(float)
    order = np.argsort(-confidence, kind="stable")
    risk = 1 - np.cumsum(accepted_target[order]) / np.arange(1, len(order) + 1) if len(order) else np.array([])
    record = {"Model": name, "common_universe_N": len(frame), "accepted_N": len(prediction),
        "common_universe_ece": ece(common_probability, frame.typed_correct),
        "common_universe_brier": float(np.mean((common_probability - frame.typed_correct) ** 2)),
        "accepted_ece": ece(confidence, accepted_target),
        "accepted_brier": float(np.mean((confidence - accepted_target) ** 2)) if len(confidence) else np.nan,
        "discrete_aurc": float(risk.mean()) if len(risk) else np.nan,
        "candidate_acceptance_coverage": len(prediction) / max(len(frame), 1)}
    curve = []
    for fraction in np.arange(0.1, 1.01, 0.1):
        n = min(len(order), max(1, round(len(order) * fraction)))
        if n:
            tp = int(accepted_target[order[:n]].sum())
            curve.append({"Model": name, "coverage": n / len(order), "accepted_N": n,
                "risk": 1 - tp / n, "selective_typed_f1": 2 * tp / max(n + len(truth), 1)})
    return record, pd.DataFrame(curve)


def evaluate_outer_unit(root, data, protocol, repeat, outer_fold, seed, checkpoint_root,
                        evidence_root, roles, base, callback=None):
    mappings = seal_calibration(base, evidence_root, roles, data, protocol, callback)
    output = root / "Predictions/OOF/Prospective" / protocol["protocol_id"] / f"repeat_{repeat}/fold_{outer_fold}/seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        contract = json.loads((output / "complete.json").read_text())
        assert all(stable_sha256(output / name) == h for name, h in contract.items())
        return output
    assert (base / "predictive_freeze.json").exists() and (base / "calibration_freeze.json").exists()
    evaluation = data.subset(roles["outer"])
    training = data.subset(roles["head_train"])
    if callback:
        callback({"phase": "Frozen outer-fold inference", "configuration": "all preselected systems"})
    bundle, expert_predictions, expert_seconds = candidate_evidence(root, evaluation, training,
        checkpoint_root / "final", protocol, include_gold_support=False)
    bundle = annotate_candidates(bundle, evaluation.entities)
    generated = bundle.frame.loc[bundle.frame.generated_candidate]
    pool = set(zip(generated.sentence_id, generated.char_start, generated.char_end))
    gold = gold_for_metrics(evaluation.entities)
    gold_spans = set(zip(gold.sentence_id, gold.char_start, gold.char_end))
    candidate_recall = len(pool & gold_spans) / max(len(gold_spans), 1)
    for prediction in expert_predictions.values():
        spans = set(zip(prediction.sentence_id, prediction.char_start, prediction.char_end))
        assert spans <= pool
    quality, calibration, efficiency, counts, subgroups, curves = [], [], [], [], [], []
    for name in protocol["ablations"] + ["Linear-chain CRF", "Biomedical Transformer"]:
        if name in protocol["ablations"]:
            model, config, threshold = load_head(base / slug(name) / "act_ner.pt")
            candidates, elapsed = score(model, bundle)
            prediction = decode(candidates, evaluation.sentences, config, threshold, seed, "outer")
            del model
            training_record = json.loads((base / slug(name) / "complete.json").read_text())
            efficiency.append({"Model": name, "span_inference_seconds": elapsed,
                "shared_expert_inference_seconds": expert_seconds,
                "head_train_seconds": training_record["train_seconds"],
                "parameters_total": training_record["parameters_total"],
                "parameters_trainable": training_record["parameters_trainable"],
                "checkpoint_bytes": (base / slug(name) / "act_ner.pt").stat().st_size})
        else:
            candidates = expert_candidate_scores(bundle, name)
            prediction = expert_predictions[name].copy()
            # Confidence must use the same definition used for calibration fitting.
            key = ["sentence_id", "char_start", "char_end"]
            evidence = candidates.set_index(key)
            lookup = pd.MultiIndex.from_frame(prediction[key])
            prediction["boundary_probability"] = evidence.loc[lookup, "span_probability"].to_numpy()
            vectors = np.stack(evidence.loc[lookup, "type_vector"].to_numpy())
            assigned = prediction.predicted_type.map(TYPE_TO_ID).to_numpy(int)
            prediction["type_probability"] = vectors[np.arange(len(vectors)), assigned]
        prediction = apply_confidence(prediction, mappings[name])
        prediction["model"] = "ACT-NER" if name in protocol["ablations"] else name
        assert set(prediction.sentence_id) <= set(evaluation.sentences.sentence_id)
        assert prediction.entity_id.is_unique
        metric = strict_metrics(evaluation.entities, prediction)
        quality.append({"Model": name, **metric, "candidate_recall": candidate_recall})
        counts.append(document_counts(evaluation, prediction, name))
        subgroups.append(subgroup_results(training, evaluation, prediction, name))
        calibration_row, curve = confidence_results(candidates, prediction, gold, mappings[name], name)
        calibration.append(calibration_row)
        curves.append(curve)
        prediction.to_parquet(output / f"{slug(name)}.parquet", index=False)
        candidates.to_parquet(output / f"{slug(name)}_candidates.parquet", index=False)
    assert len({row["N"] for row in quality}) == 1
    for name, table in {"quality": pd.DataFrame(quality), "calibration": pd.DataFrame(calibration),
        "efficiency": pd.DataFrame(efficiency), "document_counts": pd.concat(counts, ignore_index=True),
        "subgroups": pd.concat(subgroups, ignore_index=True), "risk_coverage": pd.concat(curves, ignore_index=True)}.items():
        table.assign(repeat=repeat, outer_fold=outer_fold, seed=seed).to_csv(output / f"{name}.csv", index=False)
    files = [path for path in output.iterdir() if path.suffix in (".csv", ".parquet")]
    write_once_json(output / "complete.json", {path.name: stable_sha256(path) for path in files})
    return output
