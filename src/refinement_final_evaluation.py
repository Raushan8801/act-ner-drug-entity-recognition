"""Frozen recognition evaluation with an explicit, pre-evaluation scope amendment."""
import contextlib
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from act_ner import build_candidate_bundle
from core import ENTITY_TYPES, PREDICTION_COLUMNS, load_processed
from prospective_calibration import apply_confidence
from prospective_model import score, strict_metrics
from refinement_acceptance import development_acceptance
from refinement_analysis import validate_entity_interface
from refinement_experiment import RUN, MODELS, digest, load_model, prediction_diagnostics, verify_protected
from refinement_model import decode


def authorize_evaluation(root):
    """Record the user's evaluation authorization before any final-label access."""
    root = Path(root)
    directory = root / "Reports" / RUN
    target = directory / "final_evaluation_authorization.json"
    assert not target.exists(), "An existing authorization must not be replaced."
    assert not (directory / "additional_label_access").exists()
    criteria = development_acceptance(root)
    failed = criteria.loc[~criteria.Pass, "Criterion"].tolist()
    assert failed == ["Accepted-entity ECE change against prior ACT-NER"]
    paths = [root / "src/refinement_freeze.json"]
    paths += list((root / "notebooks").glob("0[1-5]*.ipynb"))
    for prefix in ("Models", "Predictions/OOF", "Predictions/Validation", "Predictions/Refinement",
                   "Predictions/Locked_Test"):
        paths += [p for p in (root / prefix).rglob("*") if p.is_file()]
    contract = {str(p.relative_to(root)): digest(p) for p in paths}
    protection = directory / "final_evaluation_protected.json"
    with protection.open("x") as stream:
        json.dump(contract, stream, indent=2)
    record = {"authorized_at_ns": time.time_ns(), "scope": "Frozen ACT-NER entity-recognition evaluation",
        "authorization": "User explicitly requested full Notebook 06 execution after the reported development decision.",
        "amendment": "Proceed with the observed recognition-calibration trade-off; the failed ECE criterion remains failed.",
        "waived_criterion": failed[0], "development_criteria": criteria.to_dict("records"),
        "reference_seed": 17, "model_selection_permitted": False, "training_permitted": False,
        "freeze_sha256": digest(root / "src/refinement_freeze.json"),
        "evaluation_source_sha256": digest(root / "src/refinement_final_evaluation.py"),
        "protection_sha256": digest(protection)}
    with target.open("x") as stream:
        json.dump(record, stream, indent=2)
    return record


def verify_evaluation_scope(root):
    directory = root / "Reports" / RUN
    record = json.loads((directory / "final_evaluation_authorization.json").read_text())
    assert not record["training_permitted"] and not record["model_selection_permitted"]
    assert digest(root / "src/refinement_freeze.json") == record["freeze_sha256"]
    assert digest(root / "src/refinement_final_evaluation.py") == record["evaluation_source_sha256"]
    contract_path = directory / "final_evaluation_protected.json"
    assert digest(contract_path) == record["protection_sha256"]
    contract = json.loads(contract_path.read_text())
    assert all(digest(root / path) == value for path, value in contract.items())
    verify_protected(root)
    criteria = development_acceptance(root)
    assert criteria.loc[~criteria.Pass, "Criterion"].tolist() == [record["waived_criterion"]]
    return criteria


def frozen_recognition_inference(root, progress=None):
    """Fresh inference only: the sealed model, decoder, threshold and map are unchanged."""
    seal = json.loads((root / "src/refinement_freeze.json").read_text())
    assert seal["all_seeds_completed"] and seal["all_ablations_completed"]
    for key in ("checkpoint", "calibrator"):
        assert digest(root / seal[key]) == seal[f"{key}_sha256"]
    assert all(digest(root / "src" / name) == value
               for name, value in seal["inference_source_sha256"].items())
    checkpoint = torch.load(root / seal["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["threshold"] == seal["threshold"]
    assert checkpoint["seed"] == seal["reference_seed"] == 17
    sentences, entities, tokens = load_processed(root, "locked_test")
    started = time.perf_counter()
    if progress:
        progress("Frozen expert inference and candidate construction", len(sentences))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bundle, _, expert_seconds = build_candidate_bundle(root, sentences, entities.iloc[:0], tokens, 0.72)
        model = load_model(root / seal["checkpoint"])
        scores, head_seconds = score(model, bundle)
    if progress:
        progress("Frozen span decoding and confidence mapping", len(scores))
    prediction = decode(scores, sentences, model.config, seal["threshold"], 17, "Post-hoc Diagnostic Test")
    prediction = apply_confidence(prediction, joblib.load(root / seal["calibrator"]))
    inference_seconds = time.perf_counter() - started
    entity_output = prediction[list(PREDICTION_COLUMNS)].copy()
    validate_entity_interface(entity_output, sentences, tokens)
    directory = root / "Predictions/Locked_Test"
    entity_output.to_parquet(directory / f"{RUN}_recognized_entities.parquet", index=False)
    prediction.to_parquet(directory / f"{RUN}_act_ner.parquet", index=False)
    scores.to_parquet(directory / f"{RUN}_diagnostics.parquet", index=False)
    predictions = {"ACT-NER": {17: prediction}}
    for name, slug in MODELS.items():
        predictions[name] = {0: pd.read_parquet(directory / f"{slug}.parquet")}
    metrics = pd.DataFrame([{"Model": name, "Split": "Post-hoc Diagnostic Test", "Seed": next(iter(repeats)),
        **strict_metrics(entities, next(iter(repeats.values()))),
        **prediction_diagnostics(next(iter(repeats.values())), entities)}
        for name, repeats in predictions.items()])
    efficiency = {"Model": "ACT-NER", "Sentences": len(sentences), "Entities": len(prediction),
        "End-to-end seconds": inference_seconds, "Expert seconds": expert_seconds, "Head seconds": head_seconds,
        "Milliseconds per sentence": 1000 * inference_seconds / len(sentences),
        "Milliseconds per entity": 1000 * inference_seconds / len(prediction) if len(prediction) else np.nan,
        "Sentences per second": len(sentences) / inference_seconds,
        "Trainable parameters": sum(p.numel() for p in model.parameters()),
        "Head MiB": (root / seal["checkpoint"]).stat().st_size / 2**20}
    verify_evaluation_scope(root)
    return sentences, entities, tokens, predictions, metrics, scores, efficiency


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--authorize"]
    authorize_evaluation(Path(__file__).resolve().parents[1])
