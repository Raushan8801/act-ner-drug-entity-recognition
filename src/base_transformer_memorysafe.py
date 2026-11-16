"""Resource-only retry of R1.4 BiomedBERT-base; original attempt is immutable.

Only execution mechanics change: microbatch 2, accumulation 12, fixed padding
64/128/256/320, inference batch 2 and non-reentrant gradient checkpointing.
The pretrained revision, five epochs, three seeds, effective batch 24, learning
rate, full fine-tuning, original-Train-only support, decoder and final statistical
family remain those of base_transformer.py. Pooling examples into
microbatches changes stochastic optimization order; numerical identity with the
abandoned attempt is not claimed. Historical outputs are never overwritten.

resource_pilot() is an explicitly non-quality Train-only stress experiment:
48 real forward/backward microsteps and four optimizer updates across all four
padding shapes. The fresh model is discarded, never checkpointed or selected.
Actual training reloads the pinned pretrained weights and resets its seed.
Both pilot and training abort above 9 decimal GB of MPS driver allocation.
There is no training or write on import. Main must invoke every operation in
Jupyter. A PASS pilot is required before any quality training may start.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

import base_transformer as original
from core import TYPED_TAGS, ENTITY_TYPES, gold_for_metrics, tokenize_split, predictions_from_tag_sequences
from models import TransformerSequenceDataset
from prospective_model import strict_metrics
from refinement_analysis import paired_evidence


RUN, MODEL, MODEL_ID, REVISION, SEEDS = (
    original.RUN, original.MODEL, original.MODEL_ID, original.REVISION, original.SEEDS)
ATTEMPT = "biomedbert_base_memorysafe_20261016"
BUCKETS = (64, 128, 256, 320)
SETTINGS = {**original.SETTINGS, "microbatch_size": 2, "gradient_accumulation": 12,
    "inference_batch_size": 2, "padding_buckets": list(BUCKETS),
    "gradient_checkpointing": "enabled, use_reentrant=False",
    "mps_driver_limit_bytes": 9_000_000_000,
    "cache_policy": "release unused MPS allocator cache after each optimizer update"}
SPAN_MODEL = original.SPAN_MODEL


class ResourceLimit(RuntimeError):
    """The resource-only safety boundary was crossed; no quality conclusion."""


def _report(root):
    return root / "Reports" / RUN / "transformer_memorysafe_20261016"


def _paths(root, seed):
    return (root / "Models/Extended" / RUN / ATTEMPT / f"seed_{seed}",
            root / "Predictions/Extended" / RUN / ATTEMPT / f"seed_{seed}",
            _report(root) / f"seed_{seed}")


def _inputs(root):
    hashes = original._inputs(root)
    hashes[str(Path(__file__).resolve())] = original._hash(Path(__file__).resolve())
    original_protocol = original._report(root) / "protocol.json"
    hashes[str(original_protocol)] = original._hash(original_protocol)
    return hashes


def freeze_protocol(root):
    """Register a new resource-only attempt; do not alter the original freeze."""
    root = original._root(root)
    previous = original._verify_protocol(root)
    assert SETTINGS["microbatch_size"] * SETTINGS["gradient_accumulation"] == 24
    protocol = {**previous, "attempt": ATTEMPT, "settings": SETTINGS,
        "input_sha256": _inputs(root),
        "resource_change_only": "New attempt after memory failure; no outer result used to revise training choices",
        "pilot": {"source": "Original Train only", "microsteps": 48, "optimizer_updates": 4,
                  "purpose": "Worst-observed Train lengths in each fixed padding bucket; no predictive evaluation",
                  "reuse_pilot_weights": False, "require_pass_before_training": True}}
    path = _report(root) / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Frozen retry changed; use another namespace"
    else:
        original._new_json(path, protocol)
        original._new_json(path.with_name("registration.json"), {
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": original._hash(path), "training_started": False})
    return protocol


def _verify(root):
    path = _report(root) / "protocol.json"
    assert path.exists(), "Freeze the memory-safe attempt before pilot/training"
    record = json.loads(path.read_text())
    assert record["attempt"] == ATTEMPT and record["settings"] == SETTINGS
    assert record["input_sha256"] == _inputs(root), "A frozen source/input changed"
    return record


def input_audit(root):
    """Same tokenization audit as the original comparator; no fitting."""
    return original.input_audit(root)


def _bucket(length):
    for bound in BUCKETS:
        if length <= bound:
            return bound
    raise ValueError(f"Length {length} exceeds frozen 320-token cap; no truncation is allowed")


def fixed_collate(records, tokenizer):
    """Exactly two rows and one of four lengths; dummy rows have no loss."""
    assert 0 < len(records) <= 2
    width = _bucket(max(len(r["input_ids"]) for r in records))
    ids = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((2, width), -100, dtype=torch.long)
    mask = torch.zeros((2, width), dtype=torch.long)
    for index, record in enumerate(records):
        size = len(record["input_ids"])
        ids[index, :size], labels[index, :size], mask[index, :size] = record["input_ids"], record["labels"], 1
    for index in range(len(records), 2):
        ids[index, :2] = torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id])
        mask[index, :2] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask,
            "sentence_ids": [r["sentence_id"] for r in records],
            "lengths": [len(r["input_ids"]) for r in records], "padding_bucket": width}


def bucket_batches(lengths, seed, epoch):
    rng = np.random.default_rng(seed * 1009 + epoch)
    indices = rng.permutation(len(lengths))
    batches = []
    for start in range(0, len(indices), SETTINGS["length_bucket_pool"]):
        ordered = sorted(indices[start:start + SETTINGS["length_bucket_pool"]], key=lambda i: lengths[i])
        batches.extend(ordered[i:i + 2] for i in range(0, len(ordered), 2))
    return [batches[i] for i in rng.permutation(len(batches))]


def _loader(dataset, tokenizer, plan):
    return DataLoader(dataset, batch_sampler=plan, num_workers=0,
                      collate_fn=lambda rows: fixed_collate(rows, tokenizer))


def memory_snapshot(device, phase, step=0):
    original._sync(device)
    return {"phase": phase, "microstep": step,
        "MPS driver GB": torch.mps.driver_allocated_memory() / 1e9 if device.type == "mps" else None,
        "MPS tensor GB": torch.mps.current_allocated_memory() / 1e9 if device.type == "mps" else None,
        "Process RSS GB": psutil.Process().memory_info().rss / 1e9,
        "System swap GB": psutil.swap_memory().used / 1e9}


def enforce_memory(row):
    amount = row["MPS driver GB"] if row["MPS driver GB"] is not None else row["Process RSS GB"]
    if amount * 1e9 > SETTINGS["mps_driver_limit_bytes"]:
        raise ResourceLimit(f"Resource cap exceeded at {row['phase']}: {amount:.3f} GB > 9.000 GB")


def _reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _new_model(root, seed, device):
    _reset_seed(seed)
    model = AutoModelForTokenClassification.from_pretrained(original._pretrained(root),
        local_files_only=True, num_labels=len(TYPED_TAGS),
        id2label=dict(enumerate(TYPED_TAGS)), label2id={t: i for i, t in enumerate(TYPED_TAGS)})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.is_gradient_checkpointing
    model.config.use_cache = False
    return model.to(device)


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])


def _clear(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _optimize(model, optimizer, scheduler, dataset, tokenizer, plan, device, observe, emit, epoch):
    """Bounded-memory token-weighted accumulation; all graph references expire."""
    counts = [int(r["labels"].ne(-100).sum()) for r in dataset.records]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    numerator, denominator, updates = 0.0, 0, 0
    accumulation = SETTINGS["gradient_accumulation"]
    for index, batch in enumerate(_loader(dataset, tokenizer, plan)):
        group_start = index // accumulation * accumulation
        group_end = min(group_start + accumulation, len(plan))
        normalizer = sum(counts[i] for members in plan[group_start:group_end] for i in members)
        valid = int(batch["labels"].ne(-100).sum())
        output = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                       labels=batch["labels"].to(device))
        observe("after forward", index + 1)
        assert torch.isfinite(output.loss), "Nonfinite Train loss"
        (output.loss * valid / normalizer).backward()
        observe("after backward", index + 1)
        numerator += float(output.loss.detach().cpu()) * valid
        denominator += valid
        del output
        if index + 1 == group_end:
            torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_clip"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            observe("after optimizer update", index + 1)
            if device.type == "mps":
                torch.mps.empty_cache()
        if index == 0 or (index + 1) % 12 == 0 or index + 1 == len(plan):
            emit({"stage": "Train forward/backward", "epoch": epoch, "microstep": index + 1,
                  "microsteps": len(plan), "padding_bucket": batch["padding_bucket"],
                  "optimizer_updates": updates, "train_token_ce": numerator / denominator})
    return numerator / denominator, updates


def _pilot_train_data(root):
    """Read ONLY original Train text/annotations; never loads Validation gold."""
    folder = root / "dataset/Processed"
    sentences = pd.read_parquet(folder / "train_sentences.parquet")
    entities = pd.read_parquet(folder / "train_entities.parquet")
    tokenizer = AutoTokenizer.from_pretrained(original._pretrained(root), use_fast=True, local_files_only=True)
    tokens, _ = tokenize_split(sentences, gold_for_metrics(entities), tokenizer)
    dataset = TransformerSequenceDataset(tokens, tokenizer.cls_token_id, tokenizer.sep_token_id)
    assert sentences.document_id.nunique() == 572
    lengths = [len(r["input_ids"]) for r in dataset.records]
    assert max(lengths) <= max(BUCKETS)
    stress = []
    for bound in BUCKETS:
        items = sorted([i for i, n in enumerate(lengths) if _bucket(n) == bound], key=lambda i: lengths[i], reverse=True)
        assert items, f"No Train example for the registered {bound}-token bucket"
        stress.append(items[:2] if len(items) >= 2 else [items[0], items[0]])
    plan = [stress[i % len(stress)] for i in range(48)]
    return tokenizer, dataset, plan, [{"Bucket": bound, "Observed lengths": [lengths[i] for i in pair]}
                                    for bound, pair in zip(BUCKETS, stress)]


def resource_pilot(root, progress=None):
    """48 real Train-only microsteps; reports resources, never quality scores.

    Returns PASS/FAIL and detailed resource rows. No checkpoint is written.
    Any failure is converted to a status after references have been discarded,
    avoiding a Jupyter traceback retaining a multi-GB model/optimizer graph.
    """
    root = original._root(root)
    _verify(root)
    folder = _report(root) / "pilots" / str(time.time_ns())
    folder.mkdir(parents=True)
    device = original._device()
    model = optimizer = None
    observations, events, details = [], [], []
    started, status, error, updates = time.perf_counter(), "FAIL", None, 0
    completed_steps, completed_updates = 0, 0

    def emit(event):
        event = {"stage_context": "resource-only pilot", "model": MODEL, "attempt": ATTEMPT, **event}
        events.append(event)
        if progress:
            progress(event)

    def observe(phase, step=0):
        nonlocal completed_steps, completed_updates
        row = {**memory_snapshot(device, phase, step), "Elapsed seconds": time.perf_counter() - started}
        observations.append(row)
        if phase == "after backward":
            completed_steps = step
        if phase == "after optimizer update":
            completed_updates += 1
        if phase in ("after backward", "after optimizer update") or step == 0:
            emit({"stage": phase, "microstep": step, "MPS driver GB": row["MPS driver GB"],
                  "Process RSS GB": row["Process RSS GB"], "Elapsed seconds": row["Elapsed seconds"]})
        enforce_memory(row)

    try:
        _clear(device)
        observe("baseline before model")
        tokenizer, dataset, plan, details = _pilot_train_data(root)
        model = _new_model(root, 170043, device)
        observe("model loaded")
        optimizer = _optimizer(model)
        _, updates = _optimize(model, optimizer, None, dataset, tokenizer, plan, device, observe, emit, 0)
        assert updates == 4
        status = "PASS"
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        # This pilot never serializes model/optimizer weights or retains them.
        model = optimizer = None
        _clear(device)
        observations.append({**memory_snapshot(device, "after pilot model disposal", 48 if status == "PASS" else 0),
                             "Elapsed seconds": time.perf_counter() - started})
    frame = pd.DataFrame(observations)
    frame.to_csv(folder / "memory.csv", index=False)
    pd.DataFrame(events).to_json(folder / "events.jsonl", orient="records", lines=True)
    pd.DataFrame(details).to_json(folder / "stress_buckets.json", orient="records", indent=2)
    summary = {"Status": status, "Attempt": ATTEMPT, "Device": str(device),
        "Requested microsteps": 48, "Completed microsteps": completed_steps,
        "Completed optimizer updates": completed_updates,
        "Elapsed seconds": time.perf_counter() - started,
        "Peak MPS driver GB": float(frame["MPS driver GB"].max()) if device.type == "mps" else None,
        "Peak process RSS GB": float(frame["Process RSS GB"].max()),
        "Validation gold used": False, "Quality metrics computed": False,
        "Model checkpoint written": False, "Pilot weights reused": False,
        "Error": error, "protocol_sha256": original._hash(_report(root) / "protocol.json")}
    summary["artifacts_sha256"] = {p.name: original._hash(p) for p in folder.iterdir() if p.is_file()}
    original._new_json(folder / "result.json", summary)
    emit({"stage": "pilot complete; weights discarded", "status": status,
          "peak_driver_gb": summary["Peak MPS driver GB"], "error": error})
    return {"summary": pd.DataFrame([{k: v for k, v in summary.items() if k != "artifacts_sha256"}]),
            "memory": frame, "result_path": str(folder / "result.json")}


def _passed_pilot(root):
    expected = original._hash(_report(root) / "protocol.json")
    for path in sorted((_report(root) / "pilots").glob("*/result.json"), reverse=True):
        result = json.loads(path.read_text())
        if result["Status"] == "PASS" and result["protocol_sha256"] == expected:
            assert result["Completed optimizer updates"] == 4 and result["Requested microsteps"] == 48
            for name, value in result["artifacts_sha256"].items():
                assert original._hash(path.parent / name) == value
            return path
    raise AssertionError("A verified PASS resource-only pilot is required; no quality training started")


@torch.inference_mode()
def _predict(model, tokens, sentences, tokenizer, seed, observe, emit):
    dataset = TransformerSequenceDataset(tokens, tokenizer.cls_token_id, tokenizer.sep_token_id)
    plan = [list(range(i, min(i + 2, len(dataset)))) for i in range(0, len(dataset), 2)]
    device = next(model.parameters()).device
    tags, confidence, rows = {}, {}, []
    model.eval()
    for index, batch in enumerate(_loader(dataset, tokenizer, plan), 1):
        output = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        values = output.logits.softmax(-1).cpu().numpy()
        del output
        observe("Validation inference without gold scoring", index)
        for j, sid in enumerate(batch["sentence_ids"]):
            n = batch["lengths"][j] - 2
            p = values[j, 1:n + 1]
            tags[sid] = [TYPED_TAGS[k] for k in p.argmax(-1)]
            confidence[sid] = p.max(-1).tolist()
            item = pd.DataFrame(p, columns=[f"p_{t}" for t in TYPED_TAGS])
            item.insert(0, "token_index", np.arange(n))
            item.insert(0, "sentence_id", sid)
            rows.append(item)
        if index == 1 or index % 25 == 0 or index == len(plan):
            emit({"stage": "Validation inference without gold scoring", "batch": index, "batches": len(plan)})
        if device.type == "mps" and index % 12 == 0:
            torch.mps.empty_cache()
    prediction = predictions_from_tag_sequences(tokens, sentences, tags, confidence, MODEL,
                                                "Validation epoch inference; not selected", seed)
    prediction["confidence_category"] = "uncalibrated"
    assert prediction.entity_id.is_unique
    return prediction, pd.concat(rows, ignore_index=True)


def _completion(root, seed):
    path = _paths(root, seed)[2] / "complete.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    assert result["seed"] == seed and result["protocol_sha256"] == original._hash(_report(root) / "protocol.json")
    for name, digest in result["artifacts_sha256"].items():
        assert original._hash(root / name) == digest
    return result


def train_seed(root, seed, progress=None):
    """Fresh seed-reset quality run, gated by a verified resource PASS."""
    root = original._root(root)
    assert seed in SEEDS
    _verify(root)
    pilot = _passed_pilot(root)
    done = _completion(root, seed)
    if done:
        if progress:
            progress({"stage": "completed seed", "seed": seed, "attempt": ATTEMPT})
        return done
    model_dir, pred_dir, report_dir = _paths(root, seed)
    assert not any(p.exists() for p in (model_dir, pred_dir, report_dir)), "Partial attempt retained; do not overwrite"
    for path in (model_dir, pred_dir, report_dir):
        path.mkdir(parents=True)

    def emit(row):
        row = {"model": MODEL, "seed": seed, "attempt": ATTEMPT, **row}
        with (report_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if progress:
            progress(row)

    # _fit owns/disposes model tensors and returns errors as strings, so kernel
    # tracebacks cannot accidentally retain an aborted multi-GB training graph.
    result, error = _fit(root, seed, pilot, model_dir, pred_dir, report_dir, emit)
    if error is not None:
        original._new_json(report_dir / "failed.json", {"status": "ABORTED", "error": error})
        raise RuntimeError(error)
    return result


def _fit(root, seed, pilot, model_dir, pred_dir, report_dir, emit):
    device = original._device()
    model = optimizer = scheduler = None
    peak, history, error, result = 0.0, [], None, None

    def observe(phase, step=0):
        nonlocal peak
        row = memory_snapshot(device, phase, step)
        current = row["MPS driver GB"] if device.type == "mps" else row["Process RSS GB"]
        peak = max(peak, current)
        if step == 0 or step % 12 == 0 or current > 9:
            with (report_dir / "memory.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        enforce_memory(row)

    try:
        _clear(device)
        observe("before fresh model")
        data = original._data(str(root))
        data["audit"].to_csv(report_dir / "input_alignment_audit.csv", index=False)
        tokenizer = data["tokenizer"]
        dataset = TransformerSequenceDataset(data["train"]["tokens"], tokenizer.cls_token_id, tokenizer.sep_token_id)
        lengths = [len(r["input_ids"]) for r in dataset.records]
        model = _new_model(root, seed, device)  # no pilot state enters this model
        observe("fresh pinned model, seed reset")
        optimizer = _optimizer(model)
        updates_per_epoch = math.ceil(len(bucket_batches(lengths, seed, 1)) / SETTINGS["gradient_accumulation"])
        total = updates_per_epoch * SETTINGS["epochs"]
        warmup = max(1, round(total * SETTINGS["warmup_fraction"]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: min(step / warmup, max(0, (total - step) / max(1, total - warmup))))
        for epoch in range(1, SETTINGS["epochs"] + 1):
            start = time.perf_counter()
            loss, updates = _optimize(model, optimizer, scheduler, dataset, tokenizer,
                bucket_batches(lengths, seed, epoch), device, observe, emit, epoch)
            original._sync(device)
            seconds = time.perf_counter() - start
            checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_id": MODEL_ID, "revision": REVISION, "seed": seed, "epoch": epoch,
                "attempt": ATTEMPT, "settings": SETTINGS, "model_config": model.config.to_dict(),
                "protocol_sha256": original._hash(_report(root) / "protocol.json")}
            torch.save(checkpoint, model_dir / f"epoch_{epoch:02d}.pt")
            del checkpoint
            start = time.perf_counter()
            prediction, probabilities = _predict(model, data["validation"]["tokens"],
                data["validation"]["sentences"], tokenizer, seed, observe,
                lambda event: emit({"epoch": epoch, **event}))
            inference_seconds = time.perf_counter() - start
            prediction.to_parquet(pred_dir / f"epoch_{epoch:02d}.parquet", index=False)
            probabilities.to_parquet(pred_dir / f"epoch_{epoch:02d}_token_probabilities.parquet", index=False)
            history.append({"Epoch": epoch, "Train token CE": loss, "Optimizer updates": updates,
                "Training seconds": seconds, "Inference seconds": inference_seconds,
                "Peak allocated GB": peak, "Emitted Validation entities (unscored)": len(prediction)})
            pd.DataFrame(history).to_csv(report_dir / "learning.csv", index=False)
            emit({"stage": "epoch saved; no outer gold scoring", "epoch": epoch, "train_loss": loss, "peak_gb": peak})
        result = {"run": RUN, "attempt": ATTEMPT, "model": MODEL, "seed": seed, "epochs": SETTINGS["epochs"],
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
            "training_documents": 572, "validation_gradient_updates": 0,
            "training_seconds": sum(r["Training seconds"] for r in history),
            "validation_inference_seconds": sum(r["Inference seconds"] for r in history), "peak_allocated_gb": peak,
            "pilot_result": str(pilot.relative_to(root)), "pilot_sha256": original._hash(pilot),
            "protocol_sha256": original._hash(_report(root) / "protocol.json")}
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        emit({"stage": "aborted; fresh attempt artifacts retained", "error": error, "peak_gb": peak})
    finally:
        model = optimizer = scheduler = None
        _clear(device)
    if error is None:
        artifacts = [p for directory in (model_dir, pred_dir, report_dir) for p in directory.iterdir()
                     if p.is_file() and p.name != "progress.jsonl"]
        result["artifacts_sha256"] = {str(p.relative_to(root)): original._hash(p) for p in sorted(artifacts)}
        original._new_json(report_dir / "complete.json", result)
        emit({"stage": "complete trajectory", "epochs": SETTINGS["epochs"]})
    return result, error


def _locked_selection(root, data, roles, epochs):
    choices, records = [], []
    for seed in SEEDS:
        for fold, role in roles.items():
            gold = data["validation"]["entities"].loc[data["validation"]["entities"].document_id.isin(role["selection"])]
            rows = [{"Seed": seed, "Fold": fold, "Epoch": epoch,
                     **strict_metrics(gold, p.loc[p.document_id.isin(role["selection"])])} for epoch, p in epochs[seed].items()]
            chosen = pd.DataFrame(rows).sort_values(["typed_f1", "boundary_f1", "typed_recall", "macro_f1", "Epoch"],
                                                    ascending=[False, False, False, False, True], kind="stable").iloc[0]
            epoch = int(chosen.Epoch)
            checkpoint = _paths(root, seed)[0] / f"epoch_{epoch:02d}.pt"
            choices.extend(rows)
            records.append({"Seed": seed, "Fold": fold, "Selected epoch": epoch,
                "Selection documents": sorted(role["selection"]), "Outer documents": sorted(role["outer"]),
                "Checkpoint": str(checkpoint.relative_to(root)), "Checkpoint SHA256": original._hash(checkpoint)})
    record = {"protocol_sha256": original._hash(_report(root) / "protocol.json"), "selections": records}
    path = _report(root) / "locked_fold_selections.json"
    if path.exists():
        assert json.loads(path.read_text()) == record
    else:
        original._new_json(path, record)
    return pd.DataFrame(choices), pd.DataFrame(records)


def aggregate(root):
    """Same frozen three-contrast family; only this attempt's completed fits."""
    root = original._root(root)
    protocol = _verify(root)
    completed = [_completion(root, seed) for seed in SEEDS]
    assert all(completed), "All three full trajectories must be complete"
    data = original._data(str(root))
    documents, roles = original._roles(root)
    epochs = {seed: {epoch: pd.read_parquet(_paths(root, seed)[1] / f"epoch_{epoch:02d}.parquet")
                     for epoch in range(1, SETTINGS["epochs"] + 1)} for seed in SEEDS}
    selection, locks = _locked_selection(root, data, roles, epochs)
    entities, sentences = data["validation"]["entities"], data["validation"]["sentences"]
    predictions, fold_rows = {MODEL: {}}, []
    for seed in SEEDS:
        parts = []
        for fold, role in roles.items():
            epoch = int(locks.loc[locks.Seed.eq(seed) & locks.Fold.eq(fold), "Selected epoch"].iloc[0])
            p = epochs[seed][epoch].loc[epochs[seed][epoch].document_id.isin(role["outer"])].copy()
            p["split"] = "Validation conditional cross-fit"
            parts.append(p)
            fold_rows.append({"Model": MODEL, "Seed": seed, "Fold": fold, "Selected epoch": epoch,
                **strict_metrics(entities.loc[entities.document_id.isin(role["outer"])], p)})
        predictions[MODEL][seed] = pd.concat(parts, ignore_index=True)
        assert predictions[MODEL][seed].entity_id.is_unique
    historical = root / "Predictions/Refinement" / original.OLD_RUN / "Validation_CrossFit"
    for name in ("Full ACT-NER", "Parameter-matched span MLP"):
        predictions[name] = {seed: pd.read_parquet(historical / name.lower().replace(" ", "_") / f"seed_{seed}.parquet")
                             for seed in SEEDS}
    predictions[SPAN_MODEL] = original._biaffine_predictions(root)
    predictions["Frozen biomedbert-small"] = {0: pd.read_parquet(root / "Predictions/Validation/biomedical_transformer.parquet")}
    metrics = pd.DataFrame([{"Model": name, "Seed": seed, **strict_metrics(entities, p)}
                           for name, repeats in predictions.items() for seed, p in repeats.items()])
    assert metrics.N.nunique() == 1
    comparisons = [("Full ACT-NER", SPAN_MODEL), ("Full ACT-NER", MODEL), ("Parameter-matched span MLP", SPAN_MODEL)]
    assert [list(p) for p in comparisons] == protocol["holm_family"]
    statistics = paired_evidence(predictions, sentences, entities, documents, comparisons, replicates=5000)
    statistics["Holm family"] = "Final consolidated R1.4 primary contrasts (3)"
    result = {"metrics": metrics, "summary": original._summary(predictions, metrics, sentences, entities, documents),
        "classwise": metrics.melt(id_vars=["Model", "Seed", "N"], value_vars=[f"{t}_f1" for t in ENTITY_TYPES],
                                   var_name="Type", value_name="F1"),
        "statistics": statistics, "selection": selection, "fold_metrics": pd.DataFrame(fold_rows),
        "trajectories": pd.DataFrame([{k: v for k, v in row.items() if k != "artifacts_sha256"} for row in completed])}
    _save_aggregate(root, result, predictions[MODEL], locks)
    return result


def _save_aggregate(root, result, predictions, locks):
    folder = _report(root) / "aggregate"
    if folder.exists():
        for name, frame in result.items():
            pd.testing.assert_frame_equal(pd.read_csv(folder / f"{name}.csv"), frame,
                                          check_dtype=False, rtol=1e-10, atol=1e-12)
        return
    folder.mkdir()
    for name, frame in result.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    destination = root / "Predictions/Extended" / RUN / ATTEMPT / "Validation_CrossFit"
    destination.mkdir(parents=True, exist_ok=False)
    for seed, p in predictions.items():
        p.to_parquet(destination / f"seed_{seed}.parquet", index=False)
    locks.to_parquet(folder / "fold_checkpoint_references.parquet", index=False)
    original._new_json(folder / "provenance.json", {
        "attempt": ATTEMPT, "protocol_sha256": original._hash(_report(root) / "protocol.json"),
        "locked_selection_sha256": original._hash(_report(root) / "locked_fold_selections.json"),
        "completed_seed_sha256": {str(s): original._hash(_paths(root, s)[2] / "complete.json") for s in SEEDS},
        "crossfit_prediction_sha256": {str(s): original._hash(destination / f"seed_{s}.parquet") for s in SEEDS}})


def selftest():
    """Synthetic CPU checks; never loads pretrained weights or trains models."""
    from types import SimpleNamespace
    tokenizer = SimpleNamespace(pad_token_id=0, cls_token_id=1, sep_token_id=2)
    def record(n, identifier):
        return {"sentence_id": identifier, "input_ids": torch.arange(n) + 3,
                "labels": torch.tensor([-100] + [0] * (n - 2) + [-100])}
    for size, expected in [(17, 64), (64, 64), (65, 128), (258, 320)]:
        item = fixed_collate([record(size, "s")], tokenizer)
        assert item["input_ids"].shape == (2, expected)
        assert item["labels"][1].eq(-100).all() and int(item["attention_mask"][0].sum()) == size
    lengths = [i % 257 + 2 for i in range(531)]
    plan = bucket_batches(lengths, 17, 1)
    assert sorted(i for batch in plan for i in batch) == list(range(len(lengths)))
    assert all(1 <= len(batch) <= 2 for batch in plan)
    assert plan == bucket_batches(lengths, 17, 1) and plan != bucket_batches(lengths, 17, 2)
    try:
        enforce_memory({"MPS driver GB": 9.001, "Process RSS GB": 1, "phase": "synthetic"})
    except ResourceLimit:
        pass
    else:
        raise AssertionError("Resource abort did not trigger")
    original.selftest()  # Includes exact token-weighted gradient-accumulation mathematics.
    return pd.DataFrame({"Synthetic software test": ["Fixed two-row shapes and safe dummy labels",
        "Every real sentence exactly once", "Seed/epoch determinism", "9GB memory abort",
        "Token-weighted accumulation matches full-batch loss"], "Status": ["PASS"] * 5})
