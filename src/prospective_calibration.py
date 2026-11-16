"""Post-training, group-cross-fitted confidence; predictions never change."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from prospective_data import group_folds


def ece(probability, target, bins=10):
    p, y = np.asarray(probability, float), np.asarray(target, float)
    if not len(y):
        return np.nan
    assignments = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.mean(assignments == k) * abs(y[assignments == k].mean() - p[assignments == k].mean())
                     for k in range(bins) if np.any(assignments == k)))


@dataclass
class ProbabilityMap:
    method: str = "identity"
    temperature: float = 1.0
    logistic: object = None

    def predict(self, values):
        probability = np.clip(np.asarray(values, float), 1e-6, 1 - 1e-6)
        if probability.size == 0:
            return probability
        if self.method == "temperature":
            return expit(logit(probability) / self.temperature)
        if self.method == "platt":
            return self.logistic.predict_proba(logit(probability).reshape(-1, 1))[:, 1]
        return probability


def fit_probability_map(method, probability, target):
    mapping = ProbabilityMap(method)
    p = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    y = np.asarray(target, int)
    if len(np.unique(y)) < 2:
        return ProbabilityMap("identity")
    if method == "temperature":
        def objective(log_temperature):
            calibrated = np.clip(expit(logit(p) / np.exp(log_temperature)), 1e-8, 1 - 1e-8)
            return -np.mean(y * np.log(calibrated) + (1 - y) * np.log(1 - calibrated))
        result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
        mapping.temperature = float(np.exp(result.x))
    elif method == "platt":
        mapping.logistic = LogisticRegression(C=1.0, max_iter=500, random_state=9017)
        mapping.logistic.fit(logit(p).reshape(-1, 1), y)
    return mapping


def crossfit_mapping(frame, probability_column, target_column, documents, settings):
    eligible_docs = documents.loc[documents.document_id.isin(frame.document_id)]
    folds = group_folds(eligible_docs, settings["selection_folds"], 9017)
    assignment = frame.document_id.map(folds).to_numpy(int)
    p, y = frame[probability_column].to_numpy(float), frame[target_column].to_numpy(int)
    methods = settings["methods"] if np.bincount(y, minlength=2).min() >= settings["min_positive_and_negative"] else ["identity"]
    evaluations, crossfit_predictions = [], {}
    for method in methods:
        out_of_fold = np.full(len(frame), np.nan)
        for fold in range(settings["selection_folds"]):
            held = assignment == fold
            if held.any():
                mapping = fit_probability_map(method, p[~held], y[~held])
                out_of_fold[held] = mapping.predict(p[held])
        assert np.isfinite(out_of_fold).all()
        evaluations.append({"method": method, "N": len(y), "brier": float(np.mean((out_of_fold - y) ** 2)),
                            "ece": ece(out_of_fold, y)})
        crossfit_predictions[method] = out_of_fold
    table = pd.DataFrame(evaluations).sort_values(["brier", "ece", "method"])
    selected = table.iloc[0].method
    return fit_probability_map(selected, p, y), table, crossfit_predictions[selected]


def candidate_targets(scores):
    frame = scores.copy()
    labels = frame.type_label.to_numpy(int)
    predicted = frame.predicted_type.map({"drug": 0, "brand": 1, "group": 2, "drug_n": 3}).to_numpy(int)
    frame["typed_correct"] = (labels == predicted).astype(int)
    return frame


def fit_confidence_bundle(scores, documents, settings, enabled=True):
    frame = candidate_targets(scores.loc[scores.generated_candidate].copy())
    mappings, rows, crossfitted = {}, [], {}
    for channel, probability, target, subset in [
        ("boundary", "span_probability", "span_label", frame),
        ("type", "type_probability", "typed_correct", frame.loc[frame.span_label == 1]),
        ("entity", "entity_probability", "typed_correct", frame),
    ]:
        if enabled and len(subset) and subset.document_id.nunique() >= settings["selection_folds"]:
            mapping, table, heldout = crossfit_mapping(subset, probability, target, documents, settings)
        else:
            mapping = ProbabilityMap()
            heldout = mapping.predict(subset[probability])
            table = pd.DataFrame([{"method": "identity", "N": len(subset),
                "brier": float(np.mean((heldout - subset[target].to_numpy(float)) ** 2)) if len(subset) else np.nan,
                "ece": ece(heldout, subset[target])}])
        mappings[channel] = mapping
        rows.append(table.assign(channel=channel))
        crossfitted[channel] = heldout
    # Category labels express relative confidence, not clinical safety guarantees.
    lower, upper = np.quantile(crossfitted["entity"], [0.5, 0.9])
    policy = {"low": float(lower), "high": float(upper),
              "definition": "calibration-universe median and 90th percentile; no entity filtering"}
    return {"maps": mappings, "policy": policy}, pd.concat(rows, ignore_index=True)


def apply_confidence(predictions, confidence_bundle):
    result = predictions.copy()
    identity = result[["entity_id", "char_start", "char_end", "predicted_type"]].copy()
    raw = result.boundary_probability.to_numpy(float) * result.type_probability.to_numpy(float)
    result["calibrated_confidence"] = confidence_bundle["maps"]["entity"].predict(raw)
    low, high = confidence_bundle["policy"]["low"], confidence_bundle["policy"]["high"]
    result["confidence_category"] = np.where(result.calibrated_confidence >= high, "high",
        np.where(result.calibrated_confidence >= low, "medium", "low"))
    result["review_required"] = result.confidence_category == "low"
    pd.testing.assert_frame_equal(identity, result[identity.columns])
    return result
