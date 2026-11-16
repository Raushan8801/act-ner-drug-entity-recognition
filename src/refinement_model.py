"""ACT-NER residual span model with one learned decoder for all candidates.

Frozen expert probabilities initialize residual logits, not acceptance rules.
No CRF/Transformer span bypasses the verifier or overrides the subtype head.
"""
from dataclasses import replace
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from prospective_model import ACTNER as SpanNetwork
from prospective_model import tensor_batch, composite_loss, score, strict_metrics
from core import finalize_prediction_frame
from models import set_seed


class ACTNER(SpanNetwork):
    def __init__(self, config):
        super().__init__(config)
        for layer in (self.span_head, self.type_head):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)

    @torch.no_grad()
    def initialize_prototypes(self, bundle, indices, device):
        # An excluded context expert must not survive in prototype initialization.
        if self.config.use_transformer_expert and self.config.use_prototype_expert:
            super().initialize_prototypes(bundle, indices, device)

    def forward(self, batch):
        output = super().forward(batch)
        experts = []
        if self.config.use_transformer_expert:
            experts.append(batch["transformer"])
        if self.config.use_crf_expert:
            experts.append(batch["crf"])
        boundary = torch.stack([value[:, 1] for value in experts]).mean(0)
        subtype = torch.stack([value[:, -4:] for value in experts]).mean(0)
        output["span_logits"] = output["span_logits"] + torch.logit(boundary.clamp(1e-5, 1 - 1e-5))
        output["type_logits"] = output["type_logits"] + subtype.clamp_min(1e-5).log()
        return output


def eligible_candidates(frame, config):
    """A fixed union isolates learned components from candidate coverage."""
    return frame.generated_candidate.to_numpy(dtype=bool, copy=True)


def training_indices(bundle, document_ids, config, seed):
    frame = bundle.frame
    allowed = frame.document_id.isin(document_ids).to_numpy() & eligible_candidates(frame, config)
    positive = np.flatnonzero(allowed & frame.span_label.eq(1).to_numpy())
    negative = np.flatnonzero(allowed & frame.span_label.eq(0).to_numpy())
    rng = np.random.default_rng(seed)
    limit = min(len(negative), config.hard_negative_ratio * len(positive))
    if config.hard_negative_mining:
        evidence = (["transformer_span_score"] if config.use_transformer_expert else [])
        evidence += ["crf_span_score"] if config.use_crf_expert else []
        hardness = frame.iloc[negative][evidence].max(axis=1).to_numpy()
        negative = negative[np.argsort(-(hardness + rng.uniform(0, 1e-6, len(negative))))[:limit]]
    else:
        negative = rng.choice(negative, limit, replace=False)
    return np.concatenate([positive, negative])


def decode(scores, sentences, config, threshold, seed, split="Validation"):
    selected = scores.loc[eligible_candidates(scores, config) & scores.span_probability.ge(threshold)]
    emitted = []
    for _, group in selected.groupby("sentence_id", sort=False):
        occupied = []
        group = group.sort_values(["entity_probability", "char_start", "char_end"], ascending=[False, True, True])
        for row in group.itertuples(index=False):
            if any(row.char_start < right and left < row.char_end for left, right in occupied):
                continue
            occupied.append((row.char_start, row.char_end))
            emitted.append({"document_id": row.document_id, "sentence_id": row.sentence_id,
                "char_start": int(row.char_start), "char_end": int(row.char_end),
                "token_start": int(row.token_start), "token_end": int(row.token_end),
                "predicted_type": row.predicted_type, "boundary_probability": row.span_probability,
                "type_probability": row.type_probability})
    result = finalize_prediction_frame(emitted, sentences, "ACT-NER", split, seed)
    result["calibrated_confidence"] = result.boundary_probability * result.type_probability
    assert result.entity_id.is_unique
    return result


def select_threshold(scores, sentences, entities, config, seed, grid):
    table = pd.DataFrame([{"threshold": threshold,
        **strict_metrics(entities, decode(scores, sentences, config, threshold, seed))} for threshold in grid])
    selected = table.sort_values(["typed_f1", "boundary_f1", "typed_recall", "macro_f1", "threshold"],
                                 ascending=[False, False, False, False, True]).iloc[0]
    return float(selected.threshold), table


def component_ablations(config):
    """One scientific mechanism per refit; inactive controls are not ablations."""
    variants = {
        "Without Context Expert": {"use_transformer_expert": False},
        "Without Sequence Expert": {"use_crf_expert": False},
        "Without Prototype Expert": {"use_prototype_expert": False},
        "Static routing": {"static_average": True},
        "Without orthographic residual": {"use_orthographic": False},
        "Without lexical reliability": {"use_lexical_reliability": False},
        "Without weak-signal recovery": {"weak_signal_recovery": False},
        "Without hard-negative mining": {"hard_negative_mining": False},
        "Without class balancing": {"class_balanced_type_loss": False},
        "Without source balancing": {"source_balancing": "none"},
        "Without domain adapter": {"domain_adapter": False},
        "Without router regularization": {"router_regularization": False},
        "Parameter-matched span MLP": {"parameter_matched_control": True},
    }
    return {name: replace(config, **change) for name, change in variants.items()}


def fit_model(bundle, selection_bundle, training_ids, sentences, entities,
              config, seed, settings, callback=None):
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ACTNER(config).to(device)
    positions = training_indices(bundle, training_ids, config, seed)
    rows = bundle.frame.iloc[positions]
    assert not set(rows.document_id) & set(sentences.document_id)
    model.initialize_prototypes(bundle, positions, device)
    labels = rows.type_label.to_numpy(int)
    counts = torch.tensor(np.maximum(np.bincount(labels[labels >= 0], minlength=4), 1),
                          dtype=torch.float32, device=device)
    weights = 1 / rows.document_id.map(rows.document_id.value_counts()).to_numpy(float)
    if config.source_balancing == "balanced_sampler":
        sources = rows[["document_id", "source"]].drop_duplicates().source.value_counts()
        weights /= rows.source.map(sources).to_numpy(float)
    if config.class_balanced_type_loss:
        weights *= np.where(np.isin(labels, [1, 3]), 2.0, 1.0)
    weights /= weights.sum()
    adapter = list(model.context_adapter.parameters())
    adapter_ids = {id(value) for value in adapter}
    other = [value for value in model.parameters() if value.requires_grad and id(value) not in adapter_ids]
    optimizer = torch.optim.AdamW([{"params": adapter, "lr": settings["learning_rate"] / 2},
        {"params": other, "lr": settings["learning_rate"]}], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, stale, history, state = -1.0, 0, [], None
    started = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        sampled = rng.choice(positions, len(positions), replace=True, p=weights)
        losses, terms = [], []
        for start in range(0, len(sampled), settings["batch_size"]):
            batch = tensor_batch(bundle, sampled[start:start + settings["batch_size"]], device)
            optimizer.zero_grad(set_to_none=True)
            loss, component = composite_loss(model, batch, model(batch), counts)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            terms.append(component)
        scores, _ = score(model, selection_bundle)
        threshold, tuning = select_threshold(scores, sentences, entities, config, seed,
                                             settings["threshold_grid"])
        metric = float(tuning.typed_f1.max())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "selection_typed_f1": metric, "threshold": threshold,
               **pd.DataFrame(terms).mean().to_dict()}
        history.append(row)
        if callback:
            callback(row)
        if metric > best + 1e-6:
            best, stale, best_threshold, best_epoch = metric, 0, threshold, epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    assert state is not None
    model.load_state_dict(state)
    return model, pd.DataFrame(history), best_threshold, best_epoch, time.perf_counter() - started
