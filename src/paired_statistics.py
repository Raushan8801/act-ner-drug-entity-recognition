"""Read-only paired statistics for the frozen ACT-NER runs.

This module never trains, calls initialize_run, changes a split, or writes an
artifact. Notebook 07 may save the returned frames in its new review namespace.
The original 13-comparison Holm family is retained even for a five-row display.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import json

import numpy as np
import pandas as pd
import torch

from core import gold_for_metrics
from prospective_data import dependency_components, normalized_sentence
from prospective_model import strict_metrics
from refinement_analysis import assert_complete_refits, paired_evidence
from refinement_experiment import MODELS, role_partition


RUN = "refinement_20260907"
SEEDS = (17, 29, 43)
KEYS = ["document_id", "sentence_id", "char_start", "char_end", "predicted_type"]
FOCUS = (
    "Without Context Expert", "Without Sequence Expert", "Without Prototype Expert",
    "Static routing", "Parameter-matched span MLP",
)


def _context(project_root: Path) -> tuple:
    root = Path(project_root).resolve()
    assert root.name == "01_Drug_Entity_Recognition", "Wrong research project"
    report = root / "Reports" / RUN
    documents = pd.read_csv(report / "membership.csv")
    shared = root / "dataset"
    sentences = pd.read_parquet(shared / "Processed/validation_sentences.parquet")
    entities = pd.read_parquet(shared / "Processed/validation_entities.parquet")
    units = pd.concat([pd.read_csv(report / f"fold_{fold}.csv")
                       for fold in range(5)], ignore_index=True)
    assert documents.document_id.is_unique and sentences.sentence_id.is_unique
    assert entities.entity_id.is_unique
    assert set(documents.document_id) == set(sentences.document_id)
    assert len(documents) == 142
    assert_complete_refits(units)
    return root, report, documents, sentences, entities, units


def _same_metrics(actual: dict, expected: object) -> float:
    differences = []
    for column, value in actual.items():
        reference = getattr(expected, column)
        assert np.isclose(value, reference, rtol=1e-10, atol=1e-12), column
        differences.append(abs(float(value) - float(reference)))
    return max(differences, default=0.0)


def _prediction_checks(prediction: pd.DataFrame, sentence_map: pd.DataFrame,
                       permitted_documents: set[str]) -> None:
    assert prediction.entity_id.is_unique
    assert not prediction.duplicated(KEYS[:-1]).any()
    assert set(prediction.document_id) <= permitted_documents
    assert set(prediction.sentence_id) <= set(sentence_map.index)
    if not prediction.empty:
        expected = prediction.sentence_id.map(sentence_map.document_id)
        assert (expected.to_numpy() == prediction.document_id.to_numpy()).all()


def audit_refinement_integrity(project_root: Path,
                               progress: Callable[[dict], None] | None = None) -> dict[str, pd.DataFrame]:
    """Verify all 210 evaluated fits and all 42 merged seed prediction files.

    All 142 documents remain in the evaluation universe, including documents
    without gold entities or predictions. The result is independent of whether
    a model emits a row for a document. Progress is optional and UI-owned.
    """
    root, report, documents, sentences, entities, units = _context(project_root)
    components = dependency_components(sentences)
    assert documents.document_id.map(components).eq(documents.dependency_group).all()
    assert documents.groupby("dependency_group").outer_fold.nunique().eq(1).all()
    roles_by_fold = {fold: role_partition(documents, fold) for fold in range(5)}
    sentence_map = sentences.set_index("sentence_id")
    unit_rows, role_rows, parts = [], [], {}
    for fold, roles in roles_by_fold.items():
        selection = pd.read_csv(report / f"selection_fold_{fold}.csv")
        chosen = selection.groupby("configuration").selection_typed_f1.mean().sort_values(
            ascending=False, kind="stable").index[0]
        assert set(units.loc[units.fold.eq(fold), "selected_joint"]) == {chosen}
        for role, ids in roles.items():
            subset = documents.loc[documents.document_id.isin(ids)]
            role_rows.append({"Fold": fold, "Role": role, "Documents": len(ids),
                "Clusters": subset.dependency_group.nunique(), "Selected joint": chosen,
                "Disjoint role check": "PASS"})
    for index, row in enumerate(units.itertuples(index=False), 1):
        path = root / row.checkpoint
        record = torch.load(path, map_location="cpu", weights_only=False)
        roles = roles_by_fold[row.fold]
        assert {key: set(value) for key, value in record["roles"].items()} == roles
        assert record["seed"] == row.seed
        history = pd.read_csv(path.parent / "history.csv")
        best = history.loc[history.selection_typed_f1.idxmax()]
        assert row.epoch == record["epoch"] == best.epoch
        assert row.threshold == record["threshold"] == best.threshold
        prediction = pd.read_parquet(path.parent / "predictions.parquet")
        scores = pd.read_parquet(path.parent / "scores.parquet")
        _prediction_checks(prediction, sentence_map, roles["outer"])
        assert set(scores.document_id) <= roles["outer"]
        assert scores.generated_candidate.all()
        assert not scores.duplicated(["sentence_id", "char_start", "char_end"]).any()
        assert (path.parent / "calibrator.joblib").is_file()
        assert (path.parent / "calibration.csv").is_file()
        measured = strict_metrics(entities.loc[entities.document_id.isin(roles["outer"])], prediction)
        error = _same_metrics(measured, row)
        parts.setdefault((row.configuration, row.seed), []).append(prediction)
        unit_rows.append({"Model": row.configuration, "Seed": row.seed, "Fold": row.fold,
            "Outer documents": len(roles["outer"]), "Gold entities": measured["N"],
            "Emitted entities": len(prediction), "Max metric discrepancy": error,
            "Epoch": row.epoch, "Threshold": row.threshold, "Status": "PASS"})
        if progress:
            progress({"stage": "Saved-fold verification", "model": row.configuration,
                      "fold": row.fold, "seed": row.seed, "completed": index, "total": len(units)})
    crossfits = _verify_crossfits(root, parts, sentences, entities, documents)
    gold = gold_for_metrics(entities)
    summary = pd.DataFrame([
        {"Check": "Evaluation documents", "Observed": len(documents), "Status": "PASS"},
        {"Check": "Dependency clusters", "Observed": documents.dependency_group.nunique(), "Status": "PASS"},
        {"Check": "Eligible gold entities", "Observed": len(gold), "Status": "PASS"},
        {"Check": "Documents without eligible gold", "Observed": len(set(documents.document_id) - set(gold.document_id)), "Status": "RETAINED"},
        {"Check": "Verified fold-model-seed units", "Observed": len(unit_rows), "Status": "PASS"},
        {"Check": "Verified merged seed prediction files", "Observed": len(crossfits), "Status": "PASS"},
    ])
    return {"summary": summary, "roles": pd.DataFrame(role_rows),
            "units": pd.DataFrame(unit_rows), "crossfits": crossfits}


def _verify_crossfits(root: Path, parts: dict, sentences: pd.DataFrame,
                     entities: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    saved = pd.read_csv(root / "artifacts/refinement_05_seed_metrics.csv")
    rows = []
    directory = root / "Predictions/Refinement" / RUN / "Validation_CrossFit"
    for (model, seed), fold_predictions in parts.items():
        prediction = pd.read_parquet(directory / model.lower().replace(" ", "_") / f"seed_{seed}.parquet")
        _prediction_checks(prediction, sentences.set_index("sentence_id"), set(documents.document_id))
        joined = pd.concat(fold_predictions, ignore_index=True)
        pd.testing.assert_frame_equal(joined.sort_values("entity_id").reset_index(drop=True),
                                      prediction.sort_values("entity_id").reset_index(drop=True))
        expected = saved.loc[saved.Model.eq(model) & saved.Seed.eq(seed)]
        assert len(expected) == 1
        measured = strict_metrics(entities, prediction)
        error = _same_metrics(measured, next(expected.itertuples(index=False)))
        rows.append({"Model": model, "Seed": seed, "Evaluation documents": len(documents),
            "Gold entities": measured["N"], "Emitted entities": len(prediction),
            "Typed F1": measured["typed_f1"], "Max metric discrepancy": error, "Status": "PASS"})
    assert len(rows) == 42
    return pd.DataFrame(rows)


def load_crossfit_predictions(project_root: Path) -> dict[str, dict[int, pd.DataFrame]]:
    """Load only the frozen refinement predictions, never older OOF variants."""
    root, _, _, _, _, units = _context(project_root)
    directory = root / "Predictions/Refinement" / RUN / "Validation_CrossFit"
    return {name: {seed: pd.read_parquet(directory / name.lower().replace(" ", "_") /
                                       f"seed_{seed}.parquet") for seed in SEEDS}
            for name in units.configuration.unique()}


def reproduce_component_evidence(project_root: Path) -> dict[str, pd.DataFrame]:
    """Recompute the exact original 13-test family from entity-level predictions."""
    root, _, documents, sentences, entities, _ = _context(project_root)
    saved = pd.read_csv(root / "artifacts/refinement_05_component_statistics.csv")
    comparisons = [tuple(value.split(" − ")) for value in saved.Comparison]
    assert len(comparisons) == 13 and all(left == "Full ACT-NER" for left, _ in comparisons)
    calculated = paired_evidence(load_crossfit_predictions(root), sentences, entities,
                                 documents, comparisons, replicates=5000)
    assert calculated.Comparison.tolist() == saved.Comparison.tolist()
    differences = []
    for column in saved.select_dtypes("number").columns:
        discrepancy = float(np.max(np.abs(calculated[column].to_numpy() - saved[column].to_numpy())))
        assert np.allclose(calculated[column], saved[column], rtol=1e-10, atol=1e-12), column
        differences.append({"Quantity": column, "Maximum absolute discrepancy": discrepancy, "Status": "PASS"})
    interpreted = calculated.copy()
    supported = interpreted["Micro-F1 Holm p"].lt(.05) & interpreted["CI low"].gt(0)
    interpreted["Component benefit"] = np.where(supported, "Statistically supported", "Not statistically established")
    focus = interpreted.loc[interpreted.Comparison.isin([f"Full ACT-NER − {name}" for name in FOCUS])].copy()
    focus["Holm family size"] = len(comparisons)
    return {"all_comparisons": calculated, "focus": focus,
            "verification": pd.DataFrame(differences), "protocol": statistical_protocol(root)}


def load_existing_model_summary(project_root: Path) -> dict[str, pd.DataFrame]:
    """Recalculate recognition metrics rather than replaying aggregate scores."""
    root, _, documents, sentences, entities, _ = _context(project_root)
    predictions = load_crossfit_predictions(root)
    for name, slug in MODELS.items():
        predictions[name] = {0: pd.read_parquet(root / "Predictions/Validation" / f"{slug}.parquet")}
    rows = []
    for name, repeats in predictions.items():
        for seed, prediction in repeats.items():
            _prediction_checks(prediction, sentences.set_index("sentence_id"), set(documents.document_id))
            rows.append({"Model": name, "Seed": seed,
                "Evaluation role": "Conditional refinement cross-fit" if seed else "Historically selected frozen comparator",
                "Evaluation documents": len(documents), **strict_metrics(entities, prediction)})
    per_seed = pd.DataFrame(rows)
    summaries = []
    metric_names = ["boundary_precision", "boundary_recall", "boundary_f1", "typed_precision",
                    "typed_recall", "typed_f1", "macro_f1", "drug_f1", "brand_f1", "group_f1", "drug_n_f1"]
    for name, group in per_seed.groupby("Model", sort=False):
        record = {"Model": name, "Evaluation role": group["Evaluation role"].iloc[0],
                  "Documents": len(documents), "Gold entities": int(group.N.iloc[0]),
                  "Optimizer seeds": len(group) if group.Seed.ne(0).all() else pd.NA}
        for metric in metric_names:
            record[metric + " mean"] = group[metric].mean()
            record[metric + " SD"] = group[metric].std(ddof=1)
        summaries.append(record)
    return {"per_seed": per_seed, "summary": pd.DataFrame(summaries)}


def statistical_protocol(project_root: Path) -> pd.DataFrame:
    _, _, documents, _, entities, _ = _context(project_root)
    rows = [
        ("Population", f"{len(documents)} Validation documents; {len(gold_for_metrics(entities))} eligible entities"),
        ("Resampling unit", f"{documents.dependency_group.nunique()} substantial-exact-text document dependency clusters"),
        ("Document universe", "Explicit membership, including zero-gold and zero-prediction documents"),
        ("Strict typed match", "Same sentence ID, half-open character start/end and entity type"),
        ("Gold eligibility", "Contiguous and offset_valid; raw entity rows are not the denominator"),
        ("Corpus Micro-F1", "2 * sum(TP) / (sum(gold) + sum(predictions)); not mean fold F1"),
        ("Optimizer seed aggregation", "Compute corpus F1 separately at seeds17,29,43, then average the paired differences"),
        ("Partition repetition", "One fixed five-fold partition, seed1701; optimizer seeds are not new partitions"),
        ("Confidence interval", "5000 cluster bootstrap replicates; shared resample across models and seeds; 2.5/97.5 percentiles"),
        ("Primary p value", "5000 paired cluster label swaps, shared across seeds; two-sided (1 + exceedances) /5001"),
        ("Multiplicity", "Holm over all13 component comparisons; fixed-comparator comparisons form a separate family of2"),
        ("Secondary estimand", "Mean cluster-level F1 difference, averaging seeds within each cluster; Wilcoxon excluding numerical zeros"),
        ("Secondary effect size", "Matched-pairs rank-biserial correlation; not the corpus micro-F1 effect"),
        ("Resampling RNG", "numpy.default_rng(9017); bootstrap weights generated before permutation switches"),
        ("Conditioning", "Intervals condition on fitted predictions; they do not bootstrap the complete retraining procedure"),
        ("Upstream selection", "Frozen CRF regularization and Transformer stopping were selected on originalValidation; refinement CV cannot undo that exposure"),
        ("Control interpretation", "Fixed upstream candidate union; learned-evidence ablations, not complete generator removal"),
        ("Static routing control", "Uniform weighting among weak-mask eligible experts; no learned router or top2selection"),
        ("Span MLP control", "Same frozen candidates, evidence and residual priors; not an independent competitive span-based NER model"),
    ]
    return pd.DataFrame(rows, columns=["Protocol element", "Verified implementation"])


def upstream_overlap_and_legacy_oof(project_root: Path) -> dict[str, pd.DataFrame]:
    """Expose text-dependency risk and distinguish older OOF provenance."""
    root, _, documents, validation, _, _ = _context(project_root)
    train = pd.read_parquet(root / "dataset/Processed/train_sentences.parquet")
    components = dependency_components(pd.concat([train, validation], ignore_index=True))
    train_groups = {components[value] for value in train.document_id.unique()}
    normalized = train.text.map(normalized_sentence)
    known = set(normalized[normalized.str.len().ge(40) & normalized.str.split().str.len().ge(6)])
    direct = validation.loc[validation.text.map(normalized_sentence).isin(known)]
    overlap = documents.loc[documents.document_id.map(components).isin(train_groups), ["document_id", "source"]].copy()
    overlap["Train-connected dependency group"] = overlap.document_id.map(components)
    overlap["Direct matching sentences"] = overlap.document_id.map(direct.document_id.value_counts()).fillna(0).astype(int)
    overlap["Overlap type"] = np.where(overlap["Direct matching sentences"].gt(0), "Direct exact text", "Transitive component")
    records = []
    configured = json.loads((root / "src/experiment.json").read_text())["oof_folds"]
    for path in sorted((root / "Predictions/OOF").glob("*.parquet")):
        frame = pd.read_parquet(path)
        column = "fold" if "fold" in frame else "Fold"
        folds = sorted(frame[column].dropna().unique().tolist()) if column in frame else []
        records.append({"Artifact": str(path.relative_to(root)), "Emitted rows": len(frame),
            "Documents with predictions": frame.document_id.nunique(),
            "Observed fold labels": ", ".join(map(str, folds)) if folds else "Not inferable from empty predictions",
            "Stored split labels": ", ".join(map(str, frame.split.unique())),
            "Current experiment.json oof_folds": configured,
            "Interpretation": "Legacy OOF namespace; not the refinement42-file cross-fit collection"})
    return {"upstream_text_overlap": overlap.reset_index(drop=True),
            "legacy_oof_inventory": pd.DataFrame(records)}
