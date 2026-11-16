"""Evaluate the requested development gate without changing model selection."""
from pathlib import Path

import pandas as pd

from core import load_processed
from refinement_experiment import prediction_diagnostics, SEEDS


def development_acceptance(root, metrics=None, subgroups=None, routing=None):
    root = Path(root)
    if metrics is None:
        metrics = pd.read_csv(root / "artifacts/refinement_05_seed_metrics.csv")
    if subgroups is None:
        subgroups = pd.read_csv(root / "artifacts/refinement_05_subgroups.csv")
    if routing is None:
        routing = pd.read_csv(root / "artifacts/refinement_05_router.csv")
    act = metrics.loc[metrics.Model.eq("Full ACT-NER")]
    assert len(act) == 3 and set(act.Seed) == set(SEEDS)
    comparators = metrics.loc[metrics.Model.isin(["Linear-chain CRF", "Biomedical Transformer"])]
    assert len(comparators) == 2 and metrics.N.nunique() == 1
    mean = act.mean(numeric_only=True)
    best = comparators.max(numeric_only=True)
    rows = []

    def criterion(name, observed, threshold, direction):
        passed = observed >= threshold if direction == ">=" else observed <= threshold
        rows.append({"Criterion": name, "Observed": observed, "Required": threshold,
                     "Direction": direction, "Pass": bool(passed)})

    criterion("Typed F1 gain over stronger comparator", mean.typed_f1-best.typed_f1, .005, ">=")
    criterion("Boundary F1 loss against stronger comparator", best.boundary_f1-mean.boundary_f1, .002, "<=")
    criterion("Typed recall loss against stronger comparator", best.typed_recall-mean.typed_recall, .01, "<=")
    for label in ("macro_f1", "brand_f1", "drug_n_f1"):
        criterion(label+" gain over stronger comparator", mean[label]-best[label], 1e-12, ">=")
    for group in ("MedLine", "Unseen"):
        subgroup = subgroups.loc[subgroups.Subgroup.eq(group)]
        value = subgroup.loc[subgroup.Model.eq("Full ACT-NER"), "typed_f1"].mean()
        reference = subgroup.loc[subgroup.Model.isin(comparators.Model), "typed_f1"].max()
        criterion(group+" Typed F1 change", value-reference, 0., ">=")
    previous = pd.read_parquet(root / "Predictions/Validation/act_ner.parquet")
    _, entities, _ = load_processed(root, "validation")
    old_ece = prediction_diagnostics(previous, entities)["ece"]
    criterion("Accepted-entity ECE change against prior ACT-NER", mean.ece-old_ece, 0., "<=")
    # A fixed expert has no optimizer replicates in this comparison.
    criterion("Seeds exceeding both fixed comparators", int(act.typed_f1.gt(best.typed_f1).sum()), 3, ">=")
    overall = routing.loc[routing.Dimension.eq("Overall")]
    active = overall.assign(Variable=(overall['Mean weight'] > .02) & (overall['Weight SD'] > .01))
    noncollapsed = active.groupby(["Seed", "Fold"]).Variable.sum().ge(2).groupby("Seed").all()
    criterion("Seeds with noncollapsed routing", int(noncollapsed.sum()), 3, ">=")
    return pd.DataFrame(rows)
