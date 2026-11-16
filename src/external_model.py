"""BC5CDR architecture transfer: new two-type ACT-NER and matched data budgets.

This is NOT zero-shot evaluation of the original four-type ACT-NER. Its frozen
DDI-trained Transformer supplies identical contextual features to all neural
systems. Fresh Chemical/Disease experts learn from official Train; ACT learns
from three-fold cross-fitted Train evidence. Official Development supplies only
selection and calibration. Test is read only by an explicit post-freeze call.
No operation changes any September checkpoint, prediction, or split.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import copy
import hashlib
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, logsumexp
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import sklearn_crfsuite


RUN = "extended_20261016"
TYPES = ("Chemical", "Disease")
TAGS = ("O",) + tuple(f"{prefix}-{label}" for label in TYPES for prefix in "BIES")
BOUNDARY_GRID = (.25, .4, .5, .6, .7, .8, .9)
SEED = 17


def _root(project_root):
    root = Path(project_root).resolve()
    assert root.name == "01_Drug_Entity_Recognition"
    return root


def _directory(root):
    directory = _root(root) / "Models/Extended" / RUN / "BC5CDR" / "seed_17"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _notify(callback, stage, **values):
    if callback:
        callback({"stage": stage, **values})


def _verify_protocol(root):
    protocol = json.loads((_directory(root) / "protocol.json").read_text())
    assert protocol["module_sha256"] == _hash(Path(__file__)), "Code changed after protocol registration"
    assert _hash(_root(root) / "Models/Comparators/biomedical_transformer.pt") == protocol["encoder_checkpoint_sha256"]
    for path, digest in protocol["source_sha256"].items():
        assert _hash(Path(path)) == digest, "External source changed after registration"
    assert _hash(_directory(root) / "development_role_exclusions.csv") == protocol["exclusions_sha256"]
    return protocol


def _sealed_package(path):
    """Load a completed package only after checking its byte-level seal."""
    path = Path(path); checksum = path.with_suffix(path.suffix + ".sha256")
    assert checksum.exists(), f"Incomplete artifact requires a new attempt: {path.name}"
    assert _hash(path) == checksum.read_text().strip()
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_package(path, package):
    path = Path(path)
    assert not path.exists() and not path.with_suffix(path.suffix + ".sha256").exists()
    torch.save(package, path)
    path.with_suffix(path.suffix + ".sha256").write_text(_hash(path))


def _load_split(root, split, annotations=False):
    from external_data import load_bc5cdr_split
    documents, entities, _ = load_bc5cdr_split(_root(root), split, include_annotations=annotations)
    documents = documents.copy()
    documents["document_id"] = documents.document_id.astype(str)
    entities = entities.copy()
    if "document_id" in entities:
        entities["document_id"] = entities.document_id.astype(str)
    return documents, entities


def _text_groups(documents):
    """Transitive document groups sharing substantial normalized exact text."""
    from prospective_data import dependency_components
    rows = []
    for row in documents.itertuples():
        pieces = [row.text, *re.split(r"(?<=[.!?])\s+", row.text)]
        rows.extend({"document_id": row.document_id, "text": text} for text in pieces)
    return dependency_components(pd.DataFrame(rows))


def prepare_external_protocol(project_root):
    """Prespecify membership and settings before fitting; never open Test labels."""
    root = _root(project_root)
    train, _ = _load_split(root, "train")
    dev, _ = _load_split(root, "validation")
    test, _ = _load_split(root, "test")
    assert not set(train.document_id) & set(dev.document_id)
    combined = pd.concat([train, dev, test], ignore_index=True)
    assert combined.document_id.is_unique
    components = _text_groups(combined)
    train_groups = set(train.document_id.map(components))
    test_groups = set(test.document_id.map(components))
    assert not train_groups & test_groups, "Train/Test text dependency requires a separate reviewed protocol"
    train = train.assign(dependency_group=train.document_id.map(components))
    dev = dev.assign(dependency_group=dev.document_id.map(components))
    # Label-free screening: keep all official files unchanged, quarantine only
    # Development components connected to Train or Test from adaptive roles.
    excluded = dev.loc[dev.dependency_group.isin(train_groups | test_groups)].copy()
    exclusions = pd.DataFrame([{"document_id": row.document_id, "dependency_group": row.dependency_group,
        "Official split": "validation", "Excluded roles": "Selection and Calibration",
        "Reason": "Shared substantial exact sentence component with " +
            ("Train" if row.dependency_group in train_groups else "Test")} for row in excluded.itertuples()],
        columns=["document_id", "dependency_group", "Official split", "Excluded roles", "Reason"])
    dev = dev.loc[~dev.document_id.isin(excluded.document_id)].copy()
    assert not set(dev.dependency_group) & (train_groups | test_groups)
    train["expert_fold"] = -1
    for fold, (_, held) in enumerate(GroupKFold(3).split(train, groups=train.dependency_group)):
        train.loc[train.index[held], "expert_fold"] = fold
    selection, calibration = next(GroupShuffleSplit(n_splits=1, test_size=.5,
        random_state=1701).split(dev, groups=dev.dependency_group))
    roles = {"train": sorted(train.document_id), "selection": sorted(dev.iloc[selection].document_id),
             "calibration": sorted(dev.iloc[calibration].document_id)}
    data_folder = root / "dataset/External" / RUN / "BC5CDR/Processed"
    # Fingerprinting a label file protects integrity without exposing labels to fitting.
    source_paths = [data_folder / f"{split}_{kind}.parquet" for split in ("train", "validation", "test")
                    for kind in ("documents", "entities")]
    source_paths.extend(root / "src" / filename for filename in
        ("experiment.json", "models.py", "prospective_data.py", "external_data.py"))
    from transformers import AutoTokenizer
    backbone = json.loads((root / "src/experiment.json").read_text())["biomedical_backbone"]
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True, local_files_only=True)
    tokenizer_sha256 = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    exclusions_path = _directory(root) / "development_role_exclusions.csv"
    exclusion_bytes = exclusions.to_csv(index=False).encode()
    if exclusions_path.exists(): assert exclusions_path.read_bytes() == exclusion_bytes
    else: exclusions_path.write_bytes(exclusion_bytes)
    protocol = {"run": RUN, "corpus": "BC5CDR", "types": list(TYPES), "seed": SEED,
        "interpretation": "Architecture transfer with newly trained two-type heads; not zero-shot ACT-NER",
        "roles": roles, "expert_folds": dict(zip(train.document_id, train.expert_fold.astype(int))),
        "test_documents": sorted(test.document_id),
        "development_exclusions": exclusions.to_dict("records"),
        "exclusions_sha256": _hash(exclusions_path),
        "role_policy": "Official 500/500/500 manifests unchanged; label-free Development components linked to Train/Test excluded from Selection and Calibration; Test kept intact",
        "dependency_groups": components, "window_tokens": 384, "window_overlap": 64,
        "experts": {"token_epochs": 15, "token_patience": 3, "token_lr": .001,
                    "crf_c1": .1, "crf_c2": .1, "crf_iterations": 100},
        "act": {"epochs": 15, "patience": 3, "lr": .001, "batch_size": 256,
                "hidden": 128, "dropout": .15, "hard_negative_ratio": 4,
                "prototype_loss": .05, "router_loss": .01, "threshold_grid": list(BOUNDARY_GRID)},
        "candidate_marginal_threshold": .72, "marginal_max_span": 12,
        "statistics": {"bootstrap_draws": 5000, "paired_cluster_swaps": 5000,
                       "random_seed": 9017, "holm_family_size": 2},
        "label_budget": "All systems fit officialTrain labels only; Development selection/calibration separate",
        "architecture_adaptation": "New Chemical/Disease BIOES-9 contextual head and CRF; 1920-to-128 context projection; conditional two-prototype contrastive expert; top-2 span router; residual verifier/type heads. No DrugBank/MedLine domain embedding, continual adapter, or inherited four-type head is transferred.",
        "decoder": "Learned verifier threshold; entity score=span_probability*max(type_probability); greedy nonoverlap",
        "calibration": "Binary temperature scaling on reserved Development calibration candidates only",
        "encoder": "Frozen original DDI Biomedical Transformer; no encoder fine-tuning",
        "tokenizer_sha256": tokenizer_sha256,
        "encoder_checkpoint_sha256": _hash(root / "Models/Comparators/biomedical_transformer.pt"),
        "source_sha256": {str(path): _hash(path) for path in source_paths},
        "module_sha256": _hash(Path(__file__)), "test_used_for_selection": False}
    path = _directory(root) / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Existing external protocol differs"
    else:
        path.write_text(json.dumps(protocol, indent=2))
    return protocol, pd.DataFrame([{"Role": role, "Documents": len(ids)} for role, ids in roles.items()] +
        [{"Role": "Development quarantined from adaptive roles", "Documents": len(exclusions)},
         {"Role": "Official external Test (not used for selection)", "Documents": len(test)}])


def prepare_external_features(project_root, split, progress=None):
    """Losslessly encode full documents with overlapping windows and mean fusion."""
    root = _root(project_root)
    protocol = _verify_protocol(root)
    assert _hash(root / "Models/Comparators/biomedical_transformer.pt") == protocol["encoder_checkpoint_sha256"]
    if split == "test":
        verify_external_freeze(root)
    documents, _ = _load_split(root, split)
    cache_dir = root / "dataset/External" / RUN / "BC5CDR/Features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{split}_frozen_ddi_context.pt"
    text_hash = hashlib.sha256(documents[["document_id", "text"]].to_json(orient="records").encode()).hexdigest()
    if cache.exists():
        package = _sealed_package(cache)
        assert package["text_sha256"] == text_hash
        assert package["encoder_sha256"] == protocol["encoder_checkpoint_sha256"]
        assert package["module_sha256"] == protocol["module_sha256"]
        assert package["tokenizer_sha256"] == protocol["tokenizer_sha256"]
        return package["documents"]
    from transformers import AutoTokenizer
    from models import build_transformer_model
    settings = json.loads((root / "src/experiment.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(settings["biomedical_backbone"], use_fast=True, local_files_only=True)
    assert hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() == protocol["tokenizer_sha256"]
    encoder = build_transformer_model(settings["biomedical_backbone"])
    checkpoint = root / "Models/Comparators/biomedical_transformer.pt"
    assert _hash(checkpoint) == protocol["encoder_checkpoint_sha256"]
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder.to(device).eval().requires_grad_(False)
    result = {}
    for number, row in enumerate(documents.itertuples(), 1):
        encoded = tokenizer(row.text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        ids, offsets = encoded["input_ids"], np.asarray(encoded["offset_mapping"], dtype=np.int64)
        assert len(ids) == len(offsets) and (offsets[:, 0] < offsets[:, 1]).all()
        assert (offsets[:, 0] >= 0).all() and (offsets[:, 1] <= len(row.text)).all()
        assert (np.diff(offsets[:, 0]) >= 0).all()
        total, count = None, np.zeros(len(ids), np.float32)
        for start in range(0, len(ids), protocol["window_tokens"] - protocol["window_overlap"]):
            stop = min(len(ids), start + protocol["window_tokens"])
            inputs = torch.tensor([[tokenizer.cls_token_id, *ids[start:stop], tokenizer.sep_token_id]], device=device)
            with torch.inference_mode():
                hidden = encoder(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                 output_hidden_states=True).hidden_states[-1][0, 1:-1].cpu().float().numpy()
            if total is None:
                total = np.zeros((len(ids), hidden.shape[1]), np.float32)
            total[start:stop] += hidden
            count[start:stop] += 1
            if stop == len(ids):
                break
        assert len(ids) and (count > 0).all()
        result[row.document_id] = {"text": row.text, "offsets": offsets,
            "tokens": [row.text[a:b] for a, b in offsets],
            "hidden": torch.from_numpy(total / count[:, None]).half()}
        _notify(progress, "Frozen biomedical features", split=split, completed=number, total=len(documents))
    assert _hash(checkpoint) == protocol["encoder_checkpoint_sha256"]
    _save_package(cache, {"text_sha256": text_hash, "encoder_sha256": protocol["encoder_checkpoint_sha256"],
                "module_sha256": protocol["module_sha256"], "tokenizer_sha256": protocol["tokenizer_sha256"], "documents": result})
    del encoder
    return result


def token_targets(features, entities):
    """Keep complete gold externally; mask ambiguous overlapping token targets."""
    targets, diagnostics = {}, []
    for document_id, feature in features.items():
        offsets = feature["offsets"]
        assigned = [set() for _ in offsets]
        gold = entities.loc[entities.document_id.eq(document_id)]
        for row in gold.itertuples():
            indices = np.flatnonzero((offsets[:, 1] > row.char_start) & (offsets[:, 0] < row.char_end))
            if not len(indices):
                continue
            prefix = ["S"] if len(indices) == 1 else ["B", *(["I"] * (len(indices) - 2)), "E"]
            for index, tag in zip(indices, prefix):
                assigned[index].add(TAGS.index(f"{tag}-{row.entity_type}"))
        labels = np.array([next(iter(values)) if len(values) == 1 else (-100 if values else 0)
                           for values in assigned], dtype=np.int64)
        targets[document_id] = labels
        diagnostics.append({"document_id": document_id, "Tokens": len(labels),
                            "Ambiguous masked tokens": int((labels == -100).sum()), "Gold entities": len(gold)})
    return targets, pd.DataFrame(diagnostics)


def external_representation_audit(features, entities):
    """Count exact-boundary and flat-decoder ceilings without dropping gold."""
    records = []
    for key, feature in features.items():
        gold = entities.loc[entities.document_id.eq(key)].drop_duplicates(
            ["char_start", "char_end", "entity_type"])
        starts, stops = set(feature["offsets"][:, 0]), set(feature["offsets"][:, 1])
        aligned = gold.loc[gold.char_start.isin(starts) & gold.char_end.isin(stops)]
        maximum, end = 0, -1
        for row in aligned.sort_values(["char_end", "char_start"]).itertuples():
            if row.char_start >= end:
                maximum += 1; end = row.char_end
        records.append({"document_id": key, "Gold entities": len(gold),
            "Exact token-boundary entities": len(aligned), "Maximum flat aligned entities": maximum,
            "Tokens": len(feature["offsets"]), "Zero-gold document": len(gold) == 0})
    return pd.DataFrame(records)


def _crf_features(tokens):
    rows = []
    for index, token in enumerate(tokens):
        row = {"bias": 1., "word": token.lower(), "suffix3": token[-3:].lower(),
            "prefix3": token[:3].lower(), "upper": token.isupper(), "title": token.istitle(),
            "digit": token.isdigit(), "hyphen": "-" in token, "length": len(token)}
        for delta in (-2, -1, 1, 2):
            position = index + delta
            if 0 <= position < len(tokens):
                row[f"w{delta}"] = tokens[position].lower()
        rows.append(row)
    return rows


def _tag_spans(tags):
    """Deterministic BIOES repair; no gold information enters decoding."""
    spans, start, active = [], None, None
    for index, tag in enumerate([*tags, "O"]):
        prefix, label = (tag.split("-", 1) if tag != "O" else ("O", None))
        continuation = start is not None and label == active and prefix in ("I", "E")
        if start is not None and not continuation:
            spans.append((start, index, active)); start, active = None, None
        if prefix == "S":
            spans.append((index, index + 1, label))
        elif prefix in ("B", "I", "E"):
            if start is None:
                start, active = index, label
            if prefix == "E":
                spans.append((start, index + 1, active)); start, active = None, None
    return spans


def external_metrics(entities, predictions):
    gold = set(entities[["document_id", "char_start", "char_end", "entity_type"]].itertuples(index=False, name=None))
    pred = set(predictions[["document_id", "char_start", "char_end", "predicted_type"]].itertuples(index=False, name=None))
    def quality(a, b):
        tp = len(a & b)
        return tp / max(1, len(b)), tp / max(1, len(a)), 2 * tp / max(1, len(a) + len(b))
    precision, recall, typed = quality(gold, pred)
    _, _, boundary = quality({x[:3] for x in gold}, {x[:3] for x in pred})
    classes = {label + " F1": quality({x for x in gold if x[3] == label},
                                      {x for x in pred if x[3] == label})[2] for label in TYPES}
    return {"Gold entities": len(gold), "Predicted entities": len(pred), "Typed precision": precision,
        "Typed recall": recall, "Typed F1": typed, "Boundary F1": boundary,
        "Macro-F1": np.mean(list(classes.values())), **classes}


def _span_frame(features, spans_by_document, model):
    rows = []
    for document_id, spans in spans_by_document.items():
        feature = features[document_id]
        for start, stop, label in spans:
            a, b = int(feature["offsets"][start, 0]), int(feature["offsets"][stop - 1, 1])
            rows.append({"document_id": document_id, "char_start": a, "char_end": b,
                "entity_text": feature["text"][a:b], "predicted_type": label, "Model": model})
    return pd.DataFrame(rows, columns=["document_id", "char_start", "char_end", "entity_text", "predicted_type", "Model"])


def fit_external_expert_unit(project_root, train_features, dev_features, fold=None, progress=None):
    """Train one new expert pair; fold=None means full officialTrain fitting."""
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
    started = time.perf_counter()
    root = _root(project_root)
    protocol = _verify_protocol(root)
    assert set(train_features) == set(protocol["roles"]["train"])
    excluded_ids = {row["document_id"] for row in protocol["development_exclusions"]}
    assert set(dev_features) == set(protocol["roles"]["selection"]) | set(protocol["roles"]["calibration"]) | excluded_ids
    name = "full" if fold is None else f"crossfit_{int(fold)}"
    directory = _directory(root) / "experts" / name
    if (directory / "complete.json").exists():
        metadata = json.loads((directory / "complete.json").read_text())
        assert all(_hash(directory / key) == digest for key, digest in metadata["files"].items())
        assert metadata["protocol_sha256"] == _hash(_directory(root) / "protocol.json")
        _notify(progress, "Verified completed external expert pair", model=name, seed=SEED, fold=fold)
        return metadata, pd.read_csv(directory / "token_history.csv")
    assert not directory.exists() or not any(directory.iterdir()), "Incomplete expert attempt: preserve it and register a new run"
    directory.mkdir(parents=True, exist_ok=True)
    fit_ids = set(protocol["roles"]["train"])
    if fold is not None:
        fit_ids = {key for key in fit_ids if protocol["expert_folds"][key] != fold}
    _, train_gold = _load_split(root, "train", True)
    _, dev_gold = _load_split(root, "validation", True)
    fit = {key: train_features[key] for key in sorted(fit_ids)}
    labels, alignment = token_targets(fit, train_gold)
    alignment.to_csv(directory / "label_alignment.csv", index=False)
    selection_ids = set(protocol["roles"]["selection"])
    selection = {key: dev_features[key] for key in sorted(selection_ids)}
    selection_gold = dev_gold.loc[dev_gold.document_id.isin(selection_ids)]
    dimension = next(iter(fit.values()))["hidden"].shape[1]
    head = nn.Linear(dimension, len(TAGS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=protocol["experts"]["token_lr"])
    x = torch.cat([feature["hidden"].float() for feature in fit.values()])
    y = torch.tensor(np.concatenate([labels[key] for key in fit]), dtype=torch.long)
    frequencies = torch.bincount(y[y >= 0], minlength=len(TAGS)).clamp_min(1).float()
    weights = frequencies.sum().div(frequencies).sqrt(); weights /= weights.mean()
    history, best, stale, state = [], -1., 0, None
    for epoch in range(1, protocol["experts"]["token_epochs"] + 1):
        head.train(); order = torch.randperm(len(y)); losses = []
        for start in range(0, len(order), 4096):
            indices = order[start:start + 4096]; optimizer.zero_grad()
            loss = F.cross_entropy(head(x[indices]), y[indices], weight=weights, ignore_index=-100)
            loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():
            spans = {key: _tag_spans([TAGS[i] for i in head(feature["hidden"].float()).argmax(-1).tolist()])
                     for key, feature in selection.items()}
        metric = external_metrics(selection_gold, _span_frame(selection, spans, "Contextual BIOES"))["Typed F1"]
        history.append({"Epoch": epoch, "Loss": np.mean(losses), "Selection Typed F1": metric})
        _notify(progress, "External contextual expert", model=name, seed=SEED, fold=fold, epoch=epoch, typed_f1=metric)
        if metric > best + 1e-6:
            best, stale, state = metric, 0, copy.deepcopy(head.state_dict())
        else:
            stale += 1
        if stale >= protocol["experts"]["token_patience"]:
            break
    head.load_state_dict(state)
    torch.save({"state_dict": state, "dimension": dimension, "tags": TAGS}, directory / "contextual_head.pt")
    pd.DataFrame(history).to_csv(directory / "token_history.csv", index=False)
    sequences, targets = [], []
    for key, feature in fit.items():
        sequence = _crf_features(feature["tokens"])
        # Ambiguous token labels split the training sequence, never become false O targets.
        segment, tags = [], []
        for item, label in zip(sequence, labels[key]):
            if label < 0:
                if segment: sequences.append(segment); targets.append(tags)
                segment, tags = [], []
            else:
                segment.append(item); tags.append(TAGS[label])
        if segment: sequences.append(segment); targets.append(tags)
    _notify(progress, "External CRF fitting", model=name, seed=SEED, fold=fold, documents=len(fit))
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=.1, c2=.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(sequences, targets)
    joblib.dump(crf, directory / "crf.joblib")
    metadata = {"fit_documents": sorted(fit_ids), "selection_documents": sorted(selection_ids),
        "fold": fold, "seed": SEED, "contextual_parameters": sum(v.numel() for v in head.parameters()),
        "crf_parameters": len(crf.state_features_) + len(crf.transition_features_),
        "training_seconds": time.perf_counter() - started,
        "protocol_sha256": _hash(_directory(root) / "protocol.json"),
        "module_sha256": protocol["module_sha256"], "configuration": protocol["experts"],
        "files": {file: _hash(directory / file) for file in ("contextual_head.pt", "crf.joblib", "token_history.csv", "label_alignment.csv")}}
    (directory / "complete.json").write_text(json.dumps(metadata, indent=2))
    return metadata, pd.DataFrame(history)


def _crf_sequence_confidence(crf, sequence, tags):
    """Length-normalized Viterbi sequence probability computed in log-space."""
    import pycrfsuite
    labels = list(crf.classes_); index = {label: i for i, label in enumerate(labels)}
    states = crf.state_features_; transitions = crf.transition_features_
    emission = np.zeros((len(sequence), len(labels)))
    for i, features in enumerate(pycrfsuite.ItemSequence(sequence).items()):
        for feature, value in features.items():
            for label, j in index.items():
                emission[i, j] += states.get((feature, label), 0.) * value
    transition = np.array([[transitions.get((a, b), 0.) for b in labels] for a in labels])
    forward = emission[0].copy()
    for row in emission[1:]:
        forward = row + logsumexp(forward[:, None] + transition, axis=0)
    path = [index[label] for label in tags]
    score = sum(emission[i, label] for i, label in enumerate(path))
    score += sum(transition[a, b] for a, b in zip(path[:-1], path[1:]))
    return float(np.exp(min(0., (score - logsumexp(forward)) / len(sequence))))


def external_expert_evidence(project_root, features, fold=None):
    protocol = _verify_protocol(project_root)
    directory = _directory(project_root) / "experts" / ("full" if fold is None else f"crossfit_{int(fold)}")
    metadata = json.loads((directory / "complete.json").read_text())
    assert all(_hash(directory / key) == value for key, value in metadata["files"].items())
    assert metadata["module_sha256"] == protocol["module_sha256"]
    assert metadata["protocol_sha256"] == _hash(_directory(project_root) / "protocol.json")
    if fold is not None:
        assert not set(features) & set(metadata["fit_documents"])
    pack = torch.load(directory / "contextual_head.pt", map_location="cpu", weights_only=False)
    head = nn.Linear(pack["dimension"], len(TAGS)); head.load_state_dict(pack["state_dict"]); head.eval()
    crf = joblib.load(directory / "crf.joblib")
    result = {}
    with torch.inference_mode():
        for key, feature in features.items():
            t = head(feature["hidden"].float()).softmax(-1).numpy()
            sequence = _crf_features(feature["tokens"])
            tags = crf.predict_single(sequence)
            c = np.asarray([[row.get(tag, 0.) for tag in TAGS] for row in crf.predict_marginals_single(sequence)])
            probability = _crf_sequence_confidence(crf, sequence, tags)
            result[key] = {"t": t, "c": c, "t_spans": _tag_spans([TAGS[i] for i in t.argmax(-1)]),
                           "c_spans": _tag_spans(tags), "sequence_confidence": probability}
    return result


def _candidate_boundaries(evidence):
    anchors = {(a, b) for a, b, _ in [*evidence["t_spans"], *evidence["c_spans"]]}
    candidates = set(anchors); length = len(evidence["t"])
    for values in (evidence["t"], evidence["c"]):
        start = None
        for i, active in enumerate([*(1 - values[:, 0] >= .72), False]):
            if active and start is None: start = i
            if not active and start is not None:
                for a in range(start, i, 12): candidates.add((a, min(i, a + 12)))
                start = None
    for a, b in list(candidates):
        for left, right in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1), (a - 1, b - 1), (a + 1, b + 1)):
            if 0 <= left < right <= length: candidates.add((left, right))
    assert anchors <= candidates
    return sorted(candidates)


def build_external_candidates(features, evidence, train_entities, gold_entities=None):
    """Gold-independent candidate union; target labels are joined only afterward."""
    surfaces = Counter(train_entities.entity_text.str.casefold())
    if gold_entities is not None:
        multiplicity = gold_entities.groupby(["document_id", "char_start", "char_end"]).entity_type.nunique()
        assert multiplicity.le(1).all(), "A boundary has conflicting gold types; a reviewed multilabel policy is required"
    truth = {} if gold_entities is None else {
        (r.document_id, int(r.char_start), int(r.char_end)): TYPES.index(r.entity_type)
        for r in gold_entities.itertuples()}
    rows, contexts, t_features, c_features, orthography, signals = [], [], [], [], [], []
    for key, feature in features.items():
        ev = evidence[key]; hidden = feature["hidden"].float().numpy(); zero = np.zeros(hidden.shape[1])
        t_boundaries = {(a, b) for a, b, _ in ev["t_spans"]}; c_boundaries = {(a, b) for a, b, _ in ev["c_spans"]}
        for left, right in _candidate_boundaries(ev):
            a, b = int(feature["offsets"][left, 0]), int(feature["offsets"][right - 1, 1])
            surface = feature["text"][a:b]; typed = []
            for probabilities, members in ((ev["t"], t_boundaries), (ev["c"], c_boundaries)):
                q = 1 - probabilities[left:right, 0]
                p = np.array([probabilities[left:right, 1:5].sum(-1).mean(), probabilities[left:right, 5:9].sum(-1).mean()])
                p /= max(p.sum(), 1e-8)
                typed.append([q.mean(), q.min(), probabilities[left, [1, 4, 5, 8]].sum(),
                    probabilities[right - 1, [3, 4, 7, 8]].sum(), float((left, right) in members), *p])
            t, c = np.asarray(typed[0]), np.asarray(typed[1])
            weights = np.exp((1 - ev["t"][left:right, 0]) - 1); weights /= weights.sum()
            context = np.concatenate([(hidden[left:right] * weights[:, None]).sum(0), hidden[left], hidden[right - 1],
                                      hidden[left - 1] if left else zero, hidden[right] if right < len(hidden) else zero])
            agree = (left, right) in t_boundaries & c_boundaries and t[-2:].argmax() == c[-2:].argmax()
            entropy = -float((t[-2:] * np.log(t[-2:] + 1e-8)).sum())
            seen = surface.casefold() in surfaces
            weak = entropy > .5 or not agree or not seen or right - left > 1 or min(t[0], c[0]) < .72
            contexts.append(context); t_features.append(t); c_features.append([*c, ev["sequence_confidence"]])
            orthography.append([np.log1p(right - left), surface.isupper(), surface.istitle(), any(x.isdigit() for x in surface),
                                "-" in surface, "/" in surface, len(surface) / 100, np.log1p(surfaces[surface.casefold()]) / 10])
            signals.append([float(agree), float(weak), float(seen), float(right - left > 1), entropy])
            target = truth.get((key, a, b), -1)
            rows.append({"document_id": key, "char_start": a, "char_end": b, "entity_text": surface,
                "token_start": left, "token_end": right, "span_label": int(target >= 0), "type_label": target,
                "origin_transformer": (left, right) in t_boundaries, "origin_crf": (left, right) in c_boundaries})
    frame = pd.DataFrame(rows, columns=["document_id", "char_start", "char_end", "entity_text",
        "token_start", "token_end", "span_label", "type_label", "origin_transformer", "origin_crf"])
    assert not frame.duplicated(["document_id", "char_start", "char_end"]).any()
    return {"frame": frame, **{key: torch.tensor(np.asarray(value), dtype=torch.float32) for key, value in
        (("context", contexts), ("transformer", t_features), ("crf", c_features), ("orthography", orthography), ("signals", signals))}}


def prepare_external_candidate_sets(project_root, train_features, dev_features, progress=None):
    root = _root(project_root); protocol = _verify_protocol(root)
    cache = _directory(root) / "candidate_sets.pt"
    if cache.exists():
        package = _sealed_package(cache)
        assert package["protocol_sha256"] == _hash(_directory(root) / "protocol.json")
        assert all(_hash(Path(path)) == digest for path, digest in package["upstream_sha256"].items())
        return package["sets"], package["coverage"]
    _, train_gold = _load_split(root, "train", True); _, dev_gold = _load_split(root, "validation", True)
    bundles = []
    for fold in range(3):
        held_ids = {key for key, value in protocol["expert_folds"].items() if value == fold}
        features = {key: train_features[key] for key in sorted(held_ids)}
        evidence = external_expert_evidence(root, features, fold)
        fit_gold = train_gold.loc[~train_gold.document_id.isin(held_ids)]
        bundles.append(build_external_candidates(features, evidence, fit_gold, train_gold))
        _notify(progress, "External OOF candidate construction", fold=fold, completed=fold + 1, total=3)
    train_bundle = {key: (pd.concat([b[key] for b in bundles], ignore_index=True) if key == "frame"
                         else torch.cat([b[key] for b in bundles])) for key in bundles[0]}
    sets = {"train": train_bundle}
    for role in ("selection", "calibration"):
        ids = set(protocol["roles"][role]); features = {key: dev_features[key] for key in sorted(ids)}
        evidence = external_expert_evidence(root, features)
        sets[role] = build_external_candidates(features, evidence, train_gold, dev_gold)
    coverage = []
    for role, bundle in sets.items():
        ids = set(protocol["roles"][role]); gold = train_gold if role == "train" else dev_gold.loc[dev_gold.document_id.isin(ids)]
        truth = set(gold[["document_id", "char_start", "char_end"]].itertuples(index=False, name=None))
        frame = bundle["frame"]
        for name, mask in (("Transformer candidates", frame.origin_transformer), ("CRF candidates", frame.origin_crf),
                           ("Candidate union", pd.Series(True, index=frame.index))):
            observed = set(frame.loc[mask, ["document_id", "char_start", "char_end"]].itertuples(index=False, name=None))
            coverage.append({"Role": role, "Source": name, "Gold boundaries": len(truth),
                "Candidates": len(observed), "Candidate recall": len(observed & truth) / max(1, len(truth))})
    coverage = pd.DataFrame(coverage)
    upstream = [_directory(root) / "experts" / name / "complete.json"
                for name in ("crossfit_0", "crossfit_1", "crossfit_2", "full")]
    _save_package(cache, {"sets": sets, "coverage": coverage,
                         "protocol_sha256": _hash(_directory(root) / "protocol.json"),
                         "upstream_sha256": {str(path): _hash(path) for path in upstream}})
    return sets, coverage


class ExternalACTNER(nn.Module):
    """Two-type conditional tri-expert residual span model, not model voting."""
    def __init__(self, context_dimension, hidden=128, dropout=.15):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(context_dimension, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.context_expert = nn.Sequential(nn.Linear(hidden + 7, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.sequence_expert = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.prototype_encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.prototypes = nn.Parameter(F.normalize(torch.randn(2, hidden), dim=-1))
        self.prototype_projection = nn.Linear(hidden, hidden, bias=False)
        self.orthographic = nn.Linear(8, hidden, bias=False)
        self.router = nn.Sequential(nn.Linear(hidden + 7 + 8 + 8 + 5, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.norm = nn.LayerNorm(hidden); self.span = nn.Linear(hidden, 1); self.subtype = nn.Linear(hidden, 2)
        self.prototype_type = nn.Linear(2, 2, bias=False)
        for layer in (self.context_expert[-1], self.sequence_expert[-1], self.prototype_projection,
                      self.orthographic, self.span, self.subtype, self.prototype_type):
            nn.init.normal_(layer.weight, std=.001)
            if layer.bias is not None: nn.init.zeros_(layer.bias)

    def forward(self, batch):
        context = self.context(batch["context"])
        t, c, o, signals = (batch[key] for key in ("transformer", "crf", "orthography", "signals"))
        logits = self.router(torch.cat([context, t, c, o, signals], -1))
        logits = logits.clone(); logits[signals[:, 1] < .5, 2] = -1e4
        dense = logits.softmax(-1); values, indices = dense.topk(2, dim=-1)
        weights = torch.zeros_like(dense).scatter(1, indices, values)
        weights = weights / weights.sum(-1, keepdim=True)
        embedding = torch.zeros_like(context); proto_logits = context.new_zeros((len(context), 2))
        active = torch.ones(len(context), dtype=torch.bool, device=context.device) if self.training else weights[:, 2] > 0
        if active.any():
            embedding[active] = F.normalize(self.prototype_encoder(context[active]), dim=-1)
            proto_logits[active] = embedding[active] @ F.normalize(self.prototypes, dim=-1).T / .2
        experts = torch.stack([self.context_expert(torch.cat([context, t], -1)),
                               self.sequence_expert(c), self.prototype_projection(embedding)], 1)
        z = self.norm(context + (weights[:, :, None] * experts).sum(1) + self.orthographic(o))
        boundary_prior = ((t[:, 0] + c[:, 0]) / 2).clamp(1e-5, 1 - 1e-5)
        type_prior = ((t[:, -2:] + c[:, 5:7]) / 2).clamp_min(1e-5)
        return {"span_logits": self.span(z).squeeze(-1) + torch.logit(boundary_prior),
            "type_logits": self.subtype(z) + type_prior.log() + weights[:, 2:3] * self.prototype_type(proto_logits),
            "embedding": embedding, "prototype_logits": proto_logits, "weights": weights, "dense_weights": dense}


def _batch(bundle, positions):
    return {key: value[positions] for key, value in bundle.items() if key != "frame"}


def _score_external(model, bundle):
    model.eval(); parts = []
    if bundle["frame"].empty:
        return bundle["frame"].assign(boundary_probability=pd.Series(dtype=float),
            type_probability=pd.Series(dtype=float), predicted_type=pd.Series(dtype=str),
            entity_probability=pd.Series(dtype=float), router_context=pd.Series(dtype=float),
            router_sequence=pd.Series(dtype=float), router_prototype=pd.Series(dtype=float))
    with torch.inference_mode():
        for start in range(0, len(bundle["frame"]), 1024):
            indices = np.arange(start, min(start + 1024, len(bundle["frame"])))
            result = model(_batch(bundle, indices)); types = result["type_logits"].softmax(-1)
            frame = bundle["frame"].iloc[indices].copy()
            frame["boundary_probability"] = result["span_logits"].sigmoid().numpy()
            frame["type_probability"] = types.max(-1).values.numpy()
            frame["predicted_type"] = [TYPES[i] for i in types.argmax(-1).tolist()]
            frame["entity_probability"] = frame.boundary_probability * frame.type_probability
            for j, expert in enumerate(("context", "sequence", "prototype")):
                frame[f"router_{expert}"] = result["weights"][:, j].numpy()
            parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def decode_external(scores, threshold):
    selected = []
    for _, group in scores.loc[scores.boundary_probability.ge(threshold)].groupby("document_id", sort=False):
        occupied = []
        for index, row in group.sort_values(["entity_probability", "char_start", "char_end"], ascending=[False, True, True]).iterrows():
            if any(row.char_start < end and start < row.char_end for start, end in occupied): continue
            occupied.append((row.char_start, row.char_end)); selected.append(index)
    return scores.loc[selected].copy().reset_index(drop=True)


def _contrastive(embedding, labels):
    if len(labels) < 2: return embedding.sum() * 0
    similarity = embedding @ embedding.T / .2
    diagonal = torch.eye(len(labels), dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positive.sum(1) > 0
    log_probability = similarity - torch.logsumexp(similarity.masked_fill(diagonal, -1e4), 1, keepdim=True)
    return -(log_probability * positive).sum(1)[valid].div(positive.sum(1)[valid]).mean() if valid.any() else embedding.sum() * 0


def fit_external_act(project_root, candidate_sets, progress=None):
    """Fit on OOF officialTrain only; selection does not consume calibration."""
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
    started = time.perf_counter()
    root = _root(project_root); directory = _directory(root)
    protocol = _verify_protocol(root); settings = protocol["act"]
    if (directory / "freeze.json").exists():
        seal = verify_external_freeze(root)
        temperature = json.loads((directory / "temperature.json").read_text())["temperature"]
        _notify(progress, "Verified frozen external ACT-NER", seed=SEED)
        return pd.read_csv(directory / "act_history.csv"), pd.DataFrame([{"Model": "ACT-NER (two-type architecture transfer)",
            "Seed": SEED, "Threshold": seal["threshold"], "Temperature": temperature,
            "Trainable parameters": seal["parameters"], "Status": "Hash-verified frozen artifact reused"}])
    assert not (directory / "act_two_type.pt").exists(), "Incomplete ACT attempt; preserve and use a new run"
    cached = _sealed_package(directory / "candidate_sets.pt")
    for role in ("train", "selection", "calibration"):
        assert candidate_sets[role]["frame"].equals(cached["sets"][role]["frame"])
        assert set(candidate_sets[role]["frame"].document_id) <= set(protocol["roles"][role])
        for key in ("context", "transformer", "crf", "orthography", "signals"):
            assert torch.equal(candidate_sets[role][key], cached["sets"][role][key])
    training = candidate_sets["train"]; frame = training["frame"]
    assert set(frame.document_id) <= set(protocol["roles"]["train"])
    _, dev_gold = _load_split(root, "validation", True)
    selected_gold = dev_gold.loc[dev_gold.document_id.isin(protocol["roles"]["selection"])]
    model = ExternalACTNER(training["context"].shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=1e-4)
    positive = np.flatnonzero(frame.span_label.to_numpy() == 1); negative = np.flatnonzero(frame.span_label.to_numpy() == 0)
    assert len(positive) and len(negative), "Both true and spurious OOF candidates are needed"
    hardness = (training["transformer"][:, 0] + training["crf"][:, 0]).numpy()
    negative = negative[np.argsort(-hardness[negative])[:settings["hard_negative_ratio"] * len(positive)]]
    positions = np.concatenate([positive, negative]); rng = np.random.default_rng(SEED)
    labels = torch.tensor(frame.type_label.to_numpy(), dtype=torch.long); span_target = torch.tensor(frame.span_label.to_numpy(), dtype=torch.float32)
    frequencies = torch.bincount(labels[positive], minlength=2).float()
    assert (frequencies > 0).all(), "Both external classes need Train-only prototype examples"
    class_weights = frequencies.sum() / frequencies; class_weights /= class_weights.mean()
    model.eval()
    with torch.no_grad():
        context = model.context(training["context"][positive]); embeddings = F.normalize(model.prototype_encoder(context), dim=-1)
        for label in range(2): model.prototypes[label].copy_(F.normalize(embeddings[labels[positive] == label].mean(0), dim=0))
    history, best, stale, state, chosen_threshold = [], -1., 0, None, None
    for epoch in range(1, settings["epochs"] + 1):
        model.train(); order = rng.permutation(positions); losses = []
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start:start + settings["batch_size"]]; output = model(_batch(training, indices)); optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(output["span_logits"], span_target[indices])
            selected = labels[indices] >= 0
            if selected.any():
                y = labels[indices][selected]
                loss += F.cross_entropy(output["type_logits"][selected], y, weight=class_weights)
                loss += settings["prototype_loss"] * (F.cross_entropy(output["prototype_logits"][selected], y)
                    + _contrastive(output["embedding"][selected], y))
            mean = output["dense_weights"].mean(0)
            entropy = -(output["dense_weights"] * output["dense_weights"].clamp_min(1e-8).log()).sum(-1).mean()
            loss += settings["router_loss"] * (3 * mean.square().sum() - .1 * entropy)
            assert torch.isfinite(loss); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach()))
        scores = _score_external(model, candidate_sets["selection"])
        tuning = pd.DataFrame([{"threshold": threshold, **external_metrics(selected_gold, decode_external(scores, threshold))}
                               for threshold in BOUNDARY_GRID]).sort_values(["Typed F1", "Boundary F1", "Typed recall", "threshold"], ascending=[False, False, False, True])
        selected = tuning.iloc[0]; metric = selected["Typed F1"]
        history.append({"Epoch": epoch, "Loss": np.mean(losses), "Selection Typed F1": metric, "Threshold": selected.threshold})
        _notify(progress, "External ACT-NER architecture transfer", seed=SEED, epoch=epoch, typed_f1=metric)
        if metric > best + 1e-6:
            best, stale, state, chosen_threshold = metric, 0, copy.deepcopy(model.state_dict()), float(selected.threshold)
        else: stale += 1
        if stale >= settings["patience"]: break
    model.load_state_dict(state); model.eval()
    torch.save({"state_dict": state, "context_dimension": training["context"].shape[1],
                "threshold": chosen_threshold, "types": TYPES}, directory / "act_two_type.pt")
    pd.DataFrame(history).to_csv(directory / "act_history.csv", index=False)
    calibration = _score_external(model, candidate_sets["calibration"])
    target = np.array([int(label >= 0 and TYPES[int(label)] == predicted) for label, predicted in
                       zip(calibration.type_label, calibration.predicted_type)])
    raw = np.clip(calibration.entity_probability.to_numpy(), 1e-6, 1 - 1e-6)
    def objective(log_temperature):
        p = np.clip(expit(logit(raw) / np.exp(log_temperature)), 1e-8, 1 - 1e-8)
        return -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))
    temperature = float(np.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))
    (directory / "temperature.json").write_text(json.dumps({"temperature": temperature, "role": "calibration", "N": len(target)}))
    files = ["act_two_type.pt", "temperature.json", "protocol.json", "act_history.csv", "candidate_sets.pt.sha256",
             "experts/full/contextual_head.pt", "experts/full/crf.joblib", "experts/full/complete.json"]
    seal = {"seed": SEED, "threshold": chosen_threshold, "types": TYPES, "frozen_at_ns": time.time_ns(),
            "files": {file: _hash(directory / file) for file in files}, "parameters": sum(p.numel() for p in model.parameters()),
            "module_sha256": _hash(Path(__file__)), "test_used_for_selection": False,
            "training_and_calibration_seconds": time.perf_counter() - started}
    (directory / "freeze.json").write_text(json.dumps(seal, indent=2))
    return pd.DataFrame(history), pd.DataFrame([{"Model": "ACT-NER (two-type architecture transfer)", "Seed": SEED,
        "Threshold": chosen_threshold, "Temperature": temperature, "Trainable parameters": seal["parameters"],
        "Status": "Frozen before external Test"}])


def verify_external_freeze(project_root):
    directory = _directory(project_root)
    seal = json.loads((directory / "freeze.json").read_text())
    assert all(_hash(directory / key) == value for key, value in seal["files"].items())
    assert _hash(Path(__file__)) == seal["module_sha256"]
    _verify_protocol(project_root)
    return seal


def evaluate_external_frozen(project_root, test_features, progress=None):
    """Explicit final inference only; calculate metrics only after predictions exist."""
    root = _root(project_root); seal = verify_external_freeze(root); directory = _directory(root)
    destination = root / "Predictions/Extended" / RUN / "BC5CDR"
    result_path = destination / "frozen_evaluation.pt"
    if result_path.exists():
        package = _sealed_package(result_path)
        assert package["freeze_sha256"] == _hash(directory / "freeze.json")
        assert all(_hash(destination / path) == digest for path, digest in package["files"].items())
        _notify(progress, "Hash-verified external Test results reused", seed=SEED)
        return package["result"]
    assert not destination.exists() or not any(destination.iterdir()), "Incomplete Test attempt requires review; do not overwrite"
    assert set(test_features) == set(_verify_protocol(root)["test_documents"])
    _, train_gold = _load_split(root, "train", True)
    evidence = external_expert_evidence(root, test_features)
    bundle = build_external_candidates(test_features, evidence, train_gold)
    pack = torch.load(directory / "act_two_type.pt", map_location="cpu", weights_only=False)
    model = ExternalACTNER(pack["context_dimension"]); model.load_state_dict(pack["state_dict"])
    scores = _score_external(model, bundle); act = decode_external(scores, seal["threshold"])
    temperature = json.loads((directory / "temperature.json").read_text())["temperature"]
    act["calibrated_confidence"] = expit(logit(np.clip(act.entity_probability, 1e-6, 1 - 1e-6)) / temperature)
    predictions = {"ACT-NER architecture transfer": act,
        "Contextual BIOES": _span_frame(test_features, {key: value["t_spans"] for key, value in evidence.items()}, "Contextual BIOES"),
        "Linear-chain CRF": _span_frame(test_features, {key: value["c_spans"] for key, value in evidence.items()}, "Linear-chain CRF")}
    destination.mkdir(parents=True, exist_ok=True)
    for name, prediction in predictions.items():
        prediction.to_parquet(destination / f"{name.lower().replace(' ', '_')}.parquet", index=False)
    scores.to_parquet(destination / "act_candidate_scores.parquet", index=False)
    # Evaluation labels are accessed after all three frozen predictions are formed.
    documents, gold = _load_split(root, "test", True)
    assert set(documents.document_id) == set(test_features)
    metrics = pd.DataFrame([{"Model": name, "Seed": SEED, **external_metrics(gold, prediction)} for name, prediction in predictions.items()])
    truth = set(gold[["document_id", "char_start", "char_end"]].itertuples(index=False, name=None))
    coverage = []
    for name, mask in (("Transformer", scores.origin_transformer), ("CRF", scores.origin_crf),
                       ("Candidate union", pd.Series(True, index=scores.index))):
        candidates = set(scores.loc[mask, ["document_id", "char_start", "char_end"]].itertuples(index=False, name=None))
        coverage.append({"Source": name, "Gold boundaries": len(truth), "Candidate recall": len(candidates & truth) / max(1, len(truth))})
    verify_external_freeze(root)
    _notify(progress, "External frozen evaluation complete", seed=SEED, documents=len(documents), models=len(predictions))
    result = {"metrics": metrics, "coverage": pd.DataFrame(coverage), "predictions": predictions,
            "documents": documents, "entities": gold, "candidate_scores": scores,
            "representation": external_representation_audit(test_features, gold)}
    paths = sorted(destination.glob("*.parquet"))
    _save_package(result_path, {"result": result, "freeze_sha256": _hash(directory / "freeze.json"),
        "evaluated_at_ns": time.time_ns(), "files": {path.name: _hash(path) for path in paths}})
    return result


def external_cluster_statistics(result, draws=5000, seed=9017):
    """Paired whole-document/dependency-cluster inference, one optimizer seed."""
    assert draws == 5000, "Use the preregistered resampling count"
    assert seed == 9017, "Use the preregistered statistical random seed"
    documents, gold = result["documents"], result["entities"]
    groups = _text_groups(documents); clusters = sorted(set(groups.values()))
    index = {group: i for i, group in enumerate(clusters)}
    truth = set(gold[["document_id", "char_start", "char_end", "entity_type"]].itertuples(index=False, name=None))
    counts, boundary_counts = {}, {}
    for name, frame in result["predictions"].items():
        predicted = set(frame[["document_id", "char_start", "char_end", "predicted_type"]].itertuples(index=False, name=None))
        assert {row[0] for row in predicted} <= set(documents.document_id)
        for destination, a, b in ((counts, truth, predicted),
                (boundary_counts, {row[:3] for row in truth}, {row[:3] for row in predicted})):
            values = np.zeros((len(clusters), 3), dtype=float)
            for column, collection in enumerate((a & b, a, b)):
                for row in collection: values[index[groups[row[0]]], column] += 1
            destination[name] = values
    def f1(values):
        return np.divide(2 * values[..., 0], values[..., 1] + values[..., 2],
            out=np.zeros(values.shape[:-1]), where=(values[..., 1] + values[..., 2]) > 0)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=draws)
    summaries = []
    for name in counts:
        for metric, values in (("Typed F1", counts[name]), ("Boundary F1", boundary_counts[name])):
            lower, upper = np.quantile(f1(weights @ values), [.025, .975])
            summaries.append({"Model": name, "Metric": metric, "Estimate": float(f1(values.sum(0))),
                "CI lower": lower, "CI upper": upper, "Documents": len(documents), "Independent clusters": len(clusters)})
    full = counts["ACT-NER architecture transfer"]; comparisons = []
    for name, other in counts.items():
        if name == "ACT-NER architecture transfer": continue
        delta = float(f1(full.sum(0)) - f1(other.sum(0)))
        lower, upper = np.quantile(f1(weights @ full) - f1(weights @ other), [.025, .975])
        swap = rng.integers(0, 2, size=(draws, len(clusters)))
        shuffled = full.sum(0) + swap @ (other - full)
        permuted = f1(shuffled) - f1(full.sum(0) + other.sum(0) - shuffled)
        p = (1 + np.count_nonzero(np.abs(permuted) >= abs(delta) - 1e-12)) / (draws + 1)
        comparisons.append({"Reference": "ACT-NER architecture transfer", "Comparator": name,
            "Delta Typed F1": delta, "CI lower": lower, "CI upper": upper, "Paired p": p})
    frame = pd.DataFrame(comparisons); assert len(frame) == 2
    order = np.argsort(frame["Paired p"].to_numpy()); adjusted = np.empty(2)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(frame["Paired p"].to_numpy()[order] * np.array([2, 1])))
    frame["Holm p (2 comparisons)"] = adjusted
    frame["Statistically supported ACT gain"] = (frame["CI lower"] > 0) & (adjusted < .05)
    protocol = pd.DataFrame([{"Resampling unit": "Transitive substantial-exact-text document cluster",
        "Documents": len(documents), "Clusters": len(clusters), "Optimizer seeds": 1,
        "Resampling random seed": seed,
        "Bootstrap": "5000 paired cluster draws; percentile 95% CI", "Paired test": "5000 within-cluster model swaps; two-sided add-one p",
        "Multiplicity": "Holm family of 2 ACT-versus-expert comparisons", "Metric": "2*sum(TP)/(sum(gold)+sum(predicted))",
        "Inference scope": "Conditional on one fitted seed; no optimizer-seed stability claim"}])
    return {"metric_intervals": pd.DataFrame(summaries), "comparisons": frame, "protocol": protocol}
