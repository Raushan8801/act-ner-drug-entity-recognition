"""R1.4: pinned BiomedBERT-base, three Train-only fine-tuning trajectories.

Source: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext,
revision e1354b7a3a09615f6aba48dfad4b7a613eef7062 (MIT model card).
Gu et al., Domain-Specific Language Model Pretraining for Biomedical Natural
Language Processing, https://arxiv.org/abs/2007.15779.

Each seed fits original Train only for five fixed epochs. Five outer-fold
evaluations select an epoch from the SAME training trajectory using only
their original innerSelection documents. This is three trained trajectories,
not fifteen independent fold refits. Outer labels never influence gradients,
epoch logs or selection; all epoch choices are locked before outer scoring.
The historical Train/Validation text dependence and ACT-NER's additional
innerTrain fitting support remain explicit limitations of this comparison.

The decoder matches the original small Transformer: 17 BIOES token logits,
argmax and core.tags_to_spans repair, NOT a learned or constrained CRF. Own
tokenization is offset-audited; original eligible gold boundaries are retained.
There is no threshold tuning or calibration. All writes use new Extended
namespaces. Import performs no training, downloads or file writes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable
import hashlib
import json
import math
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from core import (ENTITY_TYPES, TYPED_TAGS, gold_for_metrics,
                  predictions_from_tag_sequences, tokenize_split)
from models import TransformerSequenceDataset, transformer_collate
from prospective_model import strict_metrics
from refinement_analysis import counts_by_document, f1_counts, paired_evidence
from refinement_experiment import role_partition


RUN = "extended_20261016"
OLD_RUN = "refinement_20260907"
MODEL = "BiomedBERT-base (full fine-tuning)"
MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
SEEDS = (17, 29, 43)
SPAN_MODEL = "Independent biaffine NER (frozen backbone)"
SETTINGS = {
    "epochs": 5, "microbatch_size": 8, "gradient_accumulation": 3,
    "inference_batch_size": 16,
    "effective_batch_size": 24, "learning_rate": 0.00002,
    "weight_decay": 0.01, "warmup_fraction": 0.1, "gradient_clip": 1.0,
    "length_bucket_pool": 256, "precision": "float32", "optimizer": "AdamW",
    "schedule": "linear warmup then linear decay, fixed before training",
    "loss": "Token cross-entropy, special/padding labels -100; token-weighted gradient accumulation",
    "fit_support": "Original Train 572 documents only; no Validation optimization",
    "epoch_selection": "InnerSelection typed F1, boundary F1, typed recall, macro F1; earliest epoch on ties",
    "decoder": "17 BIOES logits; argmax and the original core.tags_to_spans deterministic repair",
    "calibration": "No fitted calibrator; output confidence explicitly uncalibrated",
}


def _root(root) -> Path:
    root = Path(root).resolve()
    assert root.name == "01_Drug_Entity_Recognition", "Incorrect project root"
    return root


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _pretrained(root: Path) -> Path:
    return root / "Models/Extended" / RUN / "biomedbert_base_pretrained"


def _report(root: Path) -> Path:
    return root / "Reports" / RUN / "transformer_comparator"


def _inputs(root: Path) -> dict:
    paths = [root / "dataset/Processed" / f"{split}_{kind}.parquet"
             for split in ("train", "validation") for kind in ("sentences", "entities")]
    paths += [_pretrained(root) / name for name in
              ("config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt", "README.md")]
    paths += [root / "Reports" / OLD_RUN / "membership.csv",
              root / "src/refinement_freeze.json", Path(__file__).resolve()]
    paths += [root / "src" / name for name in
              ("core.py", "models.py", "prospective_model.py", "refinement_analysis.py",
               "refinement_experiment.py", "prospective_data.py")]
    hashes = {str(path): _hash(path) for path in paths}
    metadata = (_pretrained(root) / ".cache/huggingface/download/pytorch_model.bin.metadata").read_text().splitlines()
    assert metadata[0] == REVISION, "Pinned download revision mismatch"
    assert hashes[str(_pretrained(root) / "pytorch_model.bin")] == metadata[1], "Pinned weight checksum mismatch"
    return hashes


def _roles(root: Path) -> tuple[pd.DataFrame, dict]:
    documents = pd.read_csv(root / "Reports" / OLD_RUN / "membership.csv")
    assert len(documents) == 142 and documents.document_id.is_unique
    roles = {fold: role_partition(documents, fold) for fold in range(5)}
    return documents, roles


def freeze_protocol(root: Path) -> dict:
    """Freeze computational choices and source hashes, without any fit."""
    root = _root(root)
    audit = json.loads((root / "Reports" / RUN / "pretraining_audit.json").read_text())
    assert audit["audit_completed"]
    documents, roles = _roles(root)
    protocol = {"run": RUN, "model": MODEL, "model_id": MODEL_ID, "revision": REVISION,
        "settings": SETTINGS, "seeds": list(SEEDS), "training_trajectories": 3,
        "outer_evaluations_per_trajectory": 5, "tags": list(TYPED_TAGS),
        "input_sha256": _inputs(root),
        "roles": {str(f): {k: sorted(v) for k, v in role.items()} for f, role in roles.items()},
        "evaluation_universe": {"documents": len(documents), "clusters": int(documents.dependency_group.nunique())},
        "support_difference": "ACT refinement additionally fits innerTrain Validation labels; new Transformer fits original Train only",
        "independence": "Conditional document cross-fit; historical Train/Validation shared text is disclosed, not removed",
        "no_test_usage": "Historical and text-disjoint reserve tests are neither loaded nor used",
        "statistics": "Mean seed strict micro-F1; same text-cluster resampling across seeds; 5000 paired replicates",
        "holm_family": [["Full ACT-NER", SPAN_MODEL], ["Full ACT-NER", MODEL],
                        ["Parameter-matched span MLP", SPAN_MODEL]],
        "holm_scope": "Final consolidated R1.4 family, frozen before base fine-tuning; earlier biaffine two-contrast table remains an intermediate result",
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()}}
    path = _report(root) / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Frozen protocol changed; use a new run namespace"
    else:
        _new_json(path, protocol)
        _new_json(path.with_name("registration.json"), {
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": _hash(path), "training_started": False})
    return protocol


def _verify_protocol(root: Path) -> dict:
    path = _report(root) / "protocol.json"
    assert path.exists(), "Freeze and inspect the protocol before training"
    protocol = json.loads(path.read_text())
    assert protocol["settings"] == SETTINGS and protocol["input_sha256"] == _inputs(root)
    assert protocol["seeds"] == list(SEEDS) and protocol["revision"] == REVISION
    return protocol


@lru_cache(maxsize=1)
def _data(root_string: str) -> dict:
    """Own tokenizer; Validation tokens carry O placeholders, never gold labels."""
    root = _root(root_string)
    tokenizer = AutoTokenizer.from_pretrained(_pretrained(root), use_fast=True, local_files_only=True)
    assert tokenizer.is_fast
    result = {"tokenizer": tokenizer}
    rows = []
    for split in ("train", "validation"):
        folder = root / "dataset/Processed"
        sentences = pd.read_parquet(folder / f"{split}_sentences.parquet")
        entities = pd.read_parquet(folder / f"{split}_entities.parquet")
        gold = gold_for_metrics(entities)
        labels = gold if split == "train" else entities.iloc[:0]
        tokens, alignment = tokenize_split(sentences, labels, tokenizer)
        assert sentences.sentence_id.is_unique and not tokens.duplicated(["sentence_id", "token_index"]).any()
        texts = sentences.set_index("sentence_id").text.to_dict()
        assert all(texts[r.sentence_id][r.char_start:r.char_end] == r.surface_text for r in tokens.itertuples())
        groups = {sid: p.sort_values("token_index") for sid, p in tokens.groupby("sentence_id")}
        exact = sum(any(p.char_start.eq(r.char_start)) and any(p.char_end.eq(r.char_end))
                    for r in gold.itertuples() for p in [groups.get(r.sentence_id, pd.DataFrame(columns=tokens.columns))])
        lengths = tokens.groupby("sentence_id").size() + 2
        assert lengths.max() <= 512, "Never silently truncate inputs"
        assert set(gold.sentence_id) <= set(tokens.sentence_id)
        if split == "validation":
            assert tokens.typed_tag.eq("O").all()
        result[split] = {"sentences": sentences, "entities": entities,
                         "tokens": tokens, "alignment": alignment}
        rows.append({"Split": split, "Documents": sentences.document_id.nunique(),
            "Sentences": len(sentences), "Tokenized sentences": len(groups), "WordPieces": len(tokens),
            "Maximum input including specials": int(lengths.max()), "Strict gold": len(gold),
            "Exact boundary representable": int(exact), "Nonexact gold retained": int(len(gold) - exact),
            "Truncated sentences": 0})
    assert set(result["train"]["sentences"].document_id).isdisjoint(result["validation"]["sentences"].document_id)
    assert result["train"]["sentences"].document_id.nunique() == 572
    assert result["validation"]["sentences"].document_id.nunique() == 142
    result["audit"] = pd.DataFrame(rows)
    return result


def input_audit(root: Path) -> pd.DataFrame:
    """Read-only tokenizer/alignment checks; no model fitting or writes."""
    return _data(str(_root(root)))["audit"].copy()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def _sync(device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bucket_batches(lengths: list[int], seed: int, epoch: int) -> list[list[int]]:
    """Shuffle pools, length-sort locally, then shuffle batches; no label access."""
    rng = np.random.default_rng(seed * 1009 + epoch)
    indices = rng.permutation(len(lengths))
    batches = []
    pool = SETTINGS["length_bucket_pool"]
    for begin in range(0, len(indices), pool):
        ordered = sorted(indices[begin:begin + pool], key=lambda i: lengths[i])
        batches.extend(ordered[start:start + SETTINGS["microbatch_size"]]
                       for start in range(0, len(ordered), SETTINGS["microbatch_size"]))
    return [batches[i] for i in rng.permutation(len(batches))]


def _loader(dataset, tokenizer, batches):
    return DataLoader(dataset, batch_sampler=batches, num_workers=0,
                      collate_fn=lambda b: transformer_collate(b, tokenizer.pad_token_id))


@torch.inference_mode()
def _predict(model, tokens, sentences, tokenizer, seed, emit):
    """Validation inference receives tokens and text only, no annotations."""
    dataset = TransformerSequenceDataset(tokens, tokenizer.cls_token_id, tokenizer.sep_token_id)
    size = SETTINGS["inference_batch_size"]
    batches = [list(range(i, min(i + size, len(dataset)))) for i in range(0, len(dataset), size)]
    loader = _loader(dataset, tokenizer, batches)
    device = next(model.parameters()).device
    tags, confidence, probabilities = {}, {}, []
    model.eval()
    for index, batch in enumerate(loader, 1):
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        values = logits.softmax(-1).cpu().numpy()
        for j, sid in enumerate(batch["sentence_ids"]):
            length = int(batch["attention_mask"][j].sum()) - 2
            p = values[j, 1:length + 1]
            tags[sid] = [TYPED_TAGS[k] for k in p.argmax(-1)]
            confidence[sid] = p.max(-1).tolist()
            item = pd.DataFrame(p, columns=[f"p_{tag}" for tag in TYPED_TAGS])
            item.insert(0, "token_index", np.arange(length))
            item.insert(0, "sentence_id", sid)
            probabilities.append(item)
        if index == 1 or index % 25 == 0 or index == len(loader):
            emit({"stage": "Validation inference without scoring gold", "batch": index, "batches": len(loader)})
    prediction = predictions_from_tag_sequences(tokens, sentences, tags, confidence, MODEL,
                                               "Validation epoch inference; not yet selected", seed)
    prediction["confidence_category"] = "uncalibrated"
    assert prediction.entity_id.is_unique
    return prediction, pd.concat(probabilities, ignore_index=True)


def _paths(root: Path, seed: int) -> tuple[Path, Path, Path]:
    relative = Path("biomedbert_base") / f"seed_{seed}"
    return (root / "Models/Extended" / RUN / relative,
            root / "Predictions/Extended" / RUN / relative,
            _report(root) / f"seed_{seed}")


def _completion(root: Path, seed: int) -> dict | None:
    path = _paths(root, seed)[2] / "complete.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    assert record["seed"] == seed and record["protocol_sha256"] == _hash(_report(root) / "protocol.json")
    for name, expected in record["artifacts_sha256"].items():
        assert _hash(root / name) == expected, f"Completed artifact changed: {name}"
    return record


def train_seed(root: Path, seed: int, progress: Callable[[dict], None] | None = None) -> dict:
    """Fit one fixed trajectory; no Validation metric/threshold is computed here."""
    root = _root(root)
    assert seed in SEEDS
    _verify_protocol(root)
    complete = _completion(root, seed)
    if complete:
        if progress:
            progress({"stage": "completed seed", "model": MODEL, "seed": seed})
        return complete
    model_dir, prediction_dir, report_dir = _paths(root, seed)
    assert not any(p.exists() for p in (model_dir, prediction_dir, report_dir)), "Incomplete attempt retained; inspect before a new versioned attempt"
    for path in (model_dir, prediction_dir, report_dir):
        path.mkdir(parents=True)

    def emit(row):
        event = {"model": MODEL, "seed": seed, **row}
        with (report_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if progress:
            progress(event)

    try:
        return _fit_seed(root, seed, model_dir, prediction_dir, report_dir, emit)
    except Exception as error:
        emit({"stage": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise


def _fit_seed(root, seed, model_dir, prediction_dir, report_dir, emit):
    data = _data(str(root))
    data["audit"].to_csv(report_dir / "input_alignment_audit.csv", index=False)
    data["train"]["alignment"].to_parquet(report_dir / "train_alignment.parquet", index=False)
    tokenizer = data["tokenizer"]
    dataset = TransformerSequenceDataset(data["train"]["tokens"], tokenizer.cls_token_id, tokenizer.sep_token_id)
    lengths = [len(x["input_ids"]) for x in dataset.records]
    valid_counts = [int(x["labels"].ne(-100).sum()) for x in dataset.records]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    device = _device()
    emit({"stage": "loading pinned model", "device": str(device)})
    model = AutoModelForTokenClassification.from_pretrained(_pretrained(root), local_files_only=True,
        num_labels=len(TYPED_TAGS), id2label=dict(enumerate(TYPED_TAGS)),
        label2id={tag: i for i, tag in enumerate(TYPED_TAGS)}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    updates_per_epoch = math.ceil(len(bucket_batches(lengths, seed, 1)) / SETTINGS["gradient_accumulation"])
    total_updates = updates_per_epoch * SETTINGS["epochs"]
    warmup = max(1, round(total_updates * SETTINGS["warmup_fraction"]))
    schedule = lambda step: min(step / warmup, max(0, (total_updates - step) / max(1, total_updates - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rows = []
    for epoch in range(1, SETTINGS["epochs"] + 1):
        batch_plan = bucket_batches(lengths, seed, epoch)
        started = time.perf_counter()
        loss, updates = _train_epoch(model, optimizer, scheduler, dataset, tokenizer,
                                     batch_plan, valid_counts, device, epoch, emit)
        _sync(device)
        training_seconds = time.perf_counter() - started
        checkpoint = model_dir / f"epoch_{epoch:02d}.pt"
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "seed": seed, "epoch": epoch, "model_id": MODEL_ID, "revision": REVISION,
            "model_config": model.config.to_dict(), "settings": SETTINGS,
            "protocol_sha256": _hash(_report(root) / "protocol.json")}, checkpoint)
        started = time.perf_counter()
        prediction, probabilities = _predict(model, data["validation"]["tokens"],
            data["validation"]["sentences"], tokenizer, seed, lambda event: emit({"epoch": epoch, **event}))
        _sync(device)
        inference_seconds = time.perf_counter() - started
        prediction.to_parquet(prediction_dir / f"epoch_{epoch:02d}.parquet", index=False)
        probabilities.to_parquet(prediction_dir / f"epoch_{epoch:02d}_token_probabilities.parquet", index=False)
        row = {"Epoch": epoch, "Train token CE": loss, "Optimizer updates": updates,
            "Training seconds": training_seconds, "Inference seconds": inference_seconds,
            "Emitted Validation entities (unscored)": len(prediction),
            "Learning rate after epoch": scheduler.get_last_lr()[0]}
        rows.append(row)
        pd.DataFrame(rows).to_csv(report_dir / "learning.csv", index=False)
        emit({"stage": "epoch saved; outer labels not scored", "epoch": epoch, "train_loss": loss})
    record = {"run": RUN, "model": MODEL, "seed": seed, "epochs": SETTINGS["epochs"],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
        "training_documents": len(set(data["train"]["sentences"].document_id)),
        "validation_gradient_updates": 0, "training_seconds": sum(r["Training seconds"] for r in rows),
        "validation_inference_seconds": sum(r["Inference seconds"] for r in rows),
        "protocol_sha256": _hash(_report(root) / "protocol.json")}
    artifacts = [p for directory in (model_dir, prediction_dir, report_dir)
                 for p in directory.iterdir() if p.is_file() and p.name != "progress.jsonl"]
    record["artifacts_sha256"] = {str(p.relative_to(root)): _hash(p) for p in sorted(artifacts)}
    _new_json(report_dir / "complete.json", record)
    emit({"stage": "complete trajectory", "epochs": SETTINGS["epochs"]})
    del optimizer, scheduler, model
    if device.type == "mps":
        torch.mps.empty_cache()
    return record


def _train_epoch(model, optimizer, scheduler, dataset, tokenizer, plan, valid_counts, device, epoch, emit):
    model.train()
    loader = _loader(dataset, tokenizer, plan)
    accumulation = SETTINGS["gradient_accumulation"]
    numerator, denominator, updates = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(loader):
        group_start = index // accumulation * accumulation
        group_end = min(group_start + accumulation, len(plan))
        group_tokens = sum(valid_counts[item] for members in plan[group_start:group_end] for item in members)
        batch_tokens = int(batch["labels"].ne(-100).sum())
        output = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
        assert torch.isfinite(output.loss), "Nonfinite token loss"
        (output.loss * batch_tokens / group_tokens).backward()
        numerator += float(output.loss.detach().cpu()) * batch_tokens
        denominator += batch_tokens
        if index + 1 == group_end:
            torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(plan):
            emit({"stage": "original Train fine-tuning", "epoch": epoch, "batch": index + 1,
                  "batches": len(plan), "train_token_ce": numerator / denominator})
    return numerator / denominator, updates


def _locked_selection(root, data, roles, epochs) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, locked = [], []
    for seed in SEEDS:
        for fold, role in roles.items():
            gold = data["validation"]["entities"].loc[data["validation"]["entities"].document_id.isin(role["selection"])]
            options = []
            for epoch, prediction in epochs[seed].items():
                selected = prediction.loc[prediction.document_id.isin(role["selection"])]
                options.append({"Seed": seed, "Fold": fold, "Epoch": epoch, **strict_metrics(gold, selected)})
            table = pd.DataFrame(options)
            chosen = table.sort_values(["typed_f1", "boundary_f1", "typed_recall", "macro_f1", "Epoch"],
                                       ascending=[False, False, False, False, True], kind="stable").iloc[0]
            rows.extend(options)
            epoch = int(chosen.Epoch)
            checkpoint = _paths(root, seed)[0] / f"epoch_{epoch:02d}.pt"
            locked.append({"Seed": seed, "Fold": fold, "Selected epoch": epoch,
                "Selection documents": sorted(role["selection"]), "Outer documents": sorted(role["outer"]),
                "Checkpoint": str(checkpoint.relative_to(root)), "Checkpoint SHA256": _hash(checkpoint)})
    path = _report(root) / "locked_fold_selections.json"
    record = {"protocol_sha256": _hash(_report(root) / "protocol.json"), "selections": locked}
    if path.exists():
        assert json.loads(path.read_text()) == record
    else:
        _new_json(path, record)
    return pd.DataFrame(rows), pd.DataFrame(locked)


def aggregate(root: Path) -> dict[str, pd.DataFrame]:
    """Lock all innerSelection epoch choices, then compute outer-only metrics."""
    root = _root(root)
    _verify_protocol(root)
    completed = [_completion(root, seed) for seed in SEEDS]
    assert all(completed), "All three complete five-epoch trajectories are required"
    data = _data(str(root))
    documents, roles = _roles(root)
    epochs = {seed: {epoch: pd.read_parquet(_paths(root, seed)[1] / f"epoch_{epoch:02d}.parquet")
                     for epoch in range(1, SETTINGS["epochs"] + 1)} for seed in SEEDS}
    selection, locks = _locked_selection(root, data, roles, epochs)
    predictions, fold_rows = {MODEL: {}}, []
    entities, sentences = data["validation"]["entities"], data["validation"]["sentences"]
    for seed in SEEDS:
        parts = []
        for fold, role in roles.items():
            epoch = int(locks.loc[locks.Seed.eq(seed) & locks.Fold.eq(fold), "Selected epoch"].iloc[0])
            prediction = epochs[seed][epoch].loc[epochs[seed][epoch].document_id.isin(role["outer"])].copy()
            prediction["split"] = "Validation conditional cross-fit"
            parts.append(prediction)
            fold_rows.append({"Model": MODEL, "Seed": seed, "Fold": fold, "Selected epoch": epoch,
                **strict_metrics(entities.loc[entities.document_id.isin(role["outer"])], prediction)})
        predictions[MODEL][seed] = pd.concat(parts, ignore_index=True)
        assert predictions[MODEL][seed].entity_id.is_unique
    historical = root / "Predictions/Refinement" / OLD_RUN / "Validation_CrossFit"
    for name in ("Full ACT-NER", "Parameter-matched span MLP"):
        predictions[name] = {seed: pd.read_parquet(historical / name.lower().replace(" ", "_") / f"seed_{seed}.parquet")
                             for seed in SEEDS}
    predictions[SPAN_MODEL] = _biaffine_predictions(root)
    small = "Frozen biomedbert-small"
    predictions[small] = {0: pd.read_parquet(root / "Predictions/Validation/biomedical_transformer.parquet")}
    metrics = pd.DataFrame([{"Model": name, "Seed": seed, **strict_metrics(entities, prediction)}
                           for name, repeats in predictions.items() for seed, prediction in repeats.items()])
    assert metrics.N.nunique() == 1
    comparisons = [("Full ACT-NER", SPAN_MODEL), ("Full ACT-NER", MODEL),
                   ("Parameter-matched span MLP", SPAN_MODEL)]
    protocol = json.loads((_report(root) / "protocol.json").read_text())
    assert [list(pair) for pair in comparisons] == protocol["holm_family"]
    statistics = paired_evidence(predictions, sentences, entities, documents, comparisons, replicates=5000)
    statistics["Holm family"] = "Final consolidated R1.4 primary contrasts (3)"
    classwise = metrics.melt(id_vars=["Model", "Seed", "N"], value_vars=[f"{t}_f1" for t in ENTITY_TYPES],
                             var_name="Type", value_name="F1")
    result = {"metrics": metrics, "summary": _summary(predictions, metrics, sentences, entities, documents),
        "classwise": classwise, "statistics": statistics, "selection": selection, "fold_metrics": pd.DataFrame(fold_rows),
        "trajectories": pd.DataFrame([{k: v for k, v in r.items() if k != "artifacts_sha256"} for r in completed])}
    _save_aggregate(root, result, predictions[MODEL], locks)
    return result


def _biaffine_predictions(root):
    from biaffine_span import _unit_paths, _verified_completion
    predictions = {}
    for seed in SEEDS:
        parts = []
        for fold in range(5):
            assert _verified_completion(root, fold, seed), "Complete all independent biaffine units first"
            parts.append(pd.read_parquet(_unit_paths(root, fold, seed)[1] / "predictions.parquet"))
        predictions[seed] = pd.concat(parts, ignore_index=True)
        assert predictions[seed].entity_id.is_unique
    return predictions


def _summary(predictions, metrics, sentences, entities, documents):
    clusters = documents.dependency_group.nunique()
    weights = np.random.default_rng(9017).multinomial(clusters, np.full(clusters, 1 / clusters), size=5000)
    rows = []
    for name, repeats in predictions.items():
        values = np.stack([counts_by_document(sentences, entities, p, documents) for p in repeats.values()])
        interval = np.quantile(f1_counts(np.einsum("bg,sgc->bsc", weights, values)).mean(1), [.025, .975])
        frame = metrics.loc[metrics.Model.eq(name)]
        row = {"Model": name, "Documents": len(documents), "Clusters": clusters,
               "Gold": int(frame.N.iloc[0]), "Seeds": len(frame), "Typed F1 CI low": interval[0], "Typed F1 CI high": interval[1]}
        for column in ("typed_f1", "boundary_f1", "macro_f1", "typed_precision", "typed_recall"):
            row[column + " mean"] = frame[column].mean()
            row[column + " SD"] = frame[column].std(ddof=1) if len(frame) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _save_aggregate(root, result, predictions, locks):
    directory = _report(root) / "aggregate"
    if directory.exists():
        for name, frame in result.items():
            pd.testing.assert_frame_equal(pd.read_csv(directory / f"{name}.csv"), frame,
                                          check_dtype=False, rtol=1e-10, atol=1e-12)
        return
    directory.mkdir()
    for name, frame in result.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    destination = root / "Predictions/Extended" / RUN / "biomedbert_base/Validation_CrossFit"
    destination.mkdir(parents=True, exist_ok=False)
    for seed, prediction in predictions.items():
        prediction.to_parquet(destination / f"seed_{seed}.parquet", index=False)
    locks.to_parquet(directory / "fold_checkpoint_references.parquet", index=False)
    source_predictions = [root / "Predictions/Refinement" / OLD_RUN / "Validation_CrossFit" / name / f"seed_{s}.parquet"
                          for name in ("full_act-ner", "parameter-matched_span_mlp") for s in SEEDS]
    source_predictions += [root / "Predictions/Validation/biomedical_transformer.parquet"]
    source_predictions += [root / "Predictions/Extended" / RUN / "biaffine" / f"fold_{f}" / f"seed_{s}" / "predictions.parquet"
                           for f in range(5) for s in SEEDS]
    _new_json(directory / "provenance.json", {"protocol_sha256": _hash(_report(root) / "protocol.json"),
        "locked_selection_sha256": _hash(_report(root) / "locked_fold_selections.json"),
        "complete_seed_sha256": {str(s): _hash(_paths(root, s)[2] / "complete.json") for s in SEEDS},
        "crossfit_prediction_sha256": {str(s): _hash(destination / f"seed_{s}.parquet") for s in SEEDS},
        "comparison_prediction_sha256": {str(p): _hash(p) for p in source_predictions}})


def selftest() -> pd.DataFrame:
    """Synthetic software checks only; no model download or fitting."""
    lengths = [i % 47 + 2 for i in range(531)]
    plan = bucket_batches(lengths, 17, 1)
    assert sorted(i for batch in plan for i in batch) == list(range(len(lengths)))
    assert all(0 < len(batch) <= SETTINGS["microbatch_size"] for batch in plan)
    assert plan == bucket_batches(lengths, 17, 1) and plan != bucket_batches(lengths, 17, 2)
    # Weighted accumulation exactly matches a concatenated token-mean loss.
    torch.manual_seed(17)
    scores = torch.randn(7, len(TYPED_TAGS), requires_grad=True)
    labels = torch.tensor([0, 1, 2, 5, 16, 0, 7])
    loss_a = torch.nn.functional.cross_entropy(scores[:2], labels[:2]) * 2 / 7
    loss_b = torch.nn.functional.cross_entropy(scores[2:], labels[2:]) * 5 / 7
    reference = torch.nn.functional.cross_entropy(scores, labels)
    assert torch.allclose(loss_a + loss_b, reference, atol=1e-7)
    assert torch.allclose(torch.autograd.grad(loss_a + loss_b, scores, retain_graph=True)[0],
                          torch.autograd.grad(reference, scores)[0], atol=1e-7)
    return pd.DataFrame({"Synthetic software test": ["Every training sentence exactly once per epoch",
        "Microbatch size bound", "Seed/epoch reproducibility", "Token-weighted gradient accumulation"],
        "Status": ["PASS"] * 4})
