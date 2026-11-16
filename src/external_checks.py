"""Synthetic software invariants, not biomedical evaluation or model training.

This module never writes files, fits an estimator, or opens external Test data.
It deliberately uses a tiny invented fixture only to test computational logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import external_model as model_code


def _fixture():
    text = "aspirin treats fever"
    features = {"unit_document": {"text": text,
        "offsets": np.array([[0, 7], [8, 14], [15, 20]]),
        "tokens": ["aspirin", "treats", "fever"], "hidden": torch.randn(3, 384)}}
    gold = pd.DataFrame([
        {"document_id": "unit_document", "char_start": 0, "char_end": 7,
         "entity_text": "aspirin", "entity_type": "Chemical"},
        {"document_id": "unit_document", "char_start": 15, "char_end": 20,
         "entity_text": "fever", "entity_type": "Disease"}])
    probabilities = np.full((3, 9), .01)
    probabilities[0, 4] = .92; probabilities[1, 0] = .92; probabilities[2, 8] = .92
    probabilities /= probabilities.sum(1, keepdims=True)
    spans = [(0, 1, "Chemical"), (2, 3, "Disease")]
    evidence = {"unit_document": {"t": probabilities, "c": probabilities,
        "t_spans": spans, "c_spans": spans, "sequence_confidence": .9}}
    return features, gold, evidence


def _model_checks():
    features, gold, evidence = _fixture()
    labelled = model_code.build_external_candidates(features, evidence, gold, gold)
    unlabelled = model_code.build_external_candidates(features, evidence, gold)
    result = {"Candidate features independent of evaluation gold":
        labelled["frame"].drop(columns=["span_label", "type_label"]).equals(
            unlabelled["frame"].drop(columns=["span_label", "type_label"])) and
        all(torch.equal(labelled[key], unlabelled[key]) for key in labelled if key != "frame")}
    model = model_code.ExternalACTNER(1920)
    output = model(model_code._batch(labelled, np.arange(len(labelled["frame"]))))
    loss = output["span_logits"].sum() + output["type_logits"].sum()
    loss = loss + output["prototype_logits"].sum() + model_code._contrastive(
        output["embedding"], torch.tensor([0, 0, 1, 1, 1]))
    loss.backward()
    result["Trainable router receives gradient"] = bool(model.router[0].weight.grad.abs().sum() > 0)
    result["Trainable prototypes receive gradient"] = bool(model.prototypes.grad.abs().sum() > 0)
    result["Context gradient remains finite"] = bool(torch.isfinite(model.context[0].weight.grad).all())
    result["Sparse routing has at most two active experts"] = bool((output["weights"].gt(0).sum(-1) <= 2).all())
    result["Sparse weights sum to one"] = torch.allclose(output["weights"].sum(-1), torch.ones(5))
    easy = labelled["signals"][:, 1] < .5
    result["Easy spans mask the prototype route"] = bool((output["weights"][easy, 2] == 0).all())
    scores = model_code._score_external(model, labelled)
    result["Expert membership never overrides the learned verifier"] = model_code.decode_external(
        scores.assign(boundary_probability=.01), .5).empty
    representation = model_code.external_representation_audit(features, gold)
    result["Exact-boundary flat ceiling retains full gold denominator"] = bool(
        representation["Gold entities"].sum() == 2 and representation["Maximum flat aligned entities"].sum() == 2)
    return result, features, gold


def _statistical_checks(features, gold):
    documents = pd.DataFrame([{"document_id": "unit_document", "text": features["unit_document"]["text"]},
                             {"document_id": "empty_document", "text": "Control text without entities."}])
    predictions = gold.rename(columns={"entity_type": "predicted_type"})
    fixture = {"documents": documents, "entities": gold, "predictions": {
        "ACT-NER architecture transfer": predictions, "Contextual BIOES": predictions.iloc[:1],
        "Linear-chain CRF": predictions.iloc[1:]}}
    statistics = model_code.external_cluster_statistics(fixture)
    intervals, comparisons = statistics["metric_intervals"], statistics["comparisons"]
    extra = pd.DataFrame([{"document_id": "empty_document", "char_start": 0, "char_end": 7,
                           "predicted_type": "Chemical"}])
    false_positive_metric = model_code.external_metrics(gold, pd.concat([predictions, extra]))
    return {
        "Zero-gold documents remain independent resampling units": bool(
            (intervals.Documents == 2).all() and (intervals["Independent clusters"] == 2).all()),
        "False positives in zero-gold documents affect Micro-F1": bool(
            np.isclose(false_positive_metric["Typed F1"], .8)),
        "External Holm family contains exactly two comparisons": len(comparisons) == 2,
        "Adjusted p-values are valid and no smaller than raw p": bool(
            comparisons["Holm p (2 comparisons)"].between(0, 1).all() and
            (comparisons["Holm p (2 comparisons)"] >= comparisons["Paired p"]).all()),
        "Paired confidence intervals are ordered": bool((comparisons["CI lower"] <= comparisons["CI upper"]).all())}


def selftest():
    """Return explicit PASS/FAIL checks; does not execute biomedical training."""
    with torch.random.fork_rng():
        torch.manual_seed(17)
        checks, features, gold = _model_checks()
        checks.update(_statistical_checks(features, gold))
    table = pd.DataFrame([{"Check": name, "Status": "PASS" if bool(passed) else "FAIL",
        "Evidence type": "Synthetic software invariant; not biomedical performance"}
        for name, passed in checks.items()])
    assert table.Status.eq("PASS").all(), "An external-model software invariant failed"
    return table
