"""Trainable ACT-NER span heads; no historical anchor-acceptance overrides.

Expert evidence is supplied by fold-local checkpoints. This module never opens
raw data, saved aggregate result tables or historical prediction artifacts.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from core import ENTITY_TYPES, finalize_prediction_frame, gold_for_metrics
from models import set_seed
from act_ner import ACTNERConfig, CandidateBundle, LowRankContextAdapter, TYPE_TO_ID


class ACTNER(nn.Module):
    """Residual context with conditional context/sequence/prototype experts."""

    def __init__(self, config: ACTNERConfig):
        super().__init__()
        self.config = config
        dim = config.fusion_dim
        self.context_projection = nn.Sequential(nn.Linear(5 * config.context_dim, dim),
            nn.GELU(), nn.LayerNorm(dim), nn.Dropout(config.dropout))
        self.context_adapter = LowRankContextAdapter(dim, rank=16)
        self.domain_embedding = nn.Embedding(2, dim)
        self.orthographic = nn.Linear(14, dim, bias=False)
        self.lexical = nn.Linear(7, dim, bias=False)
        self.context_expert = nn.Sequential(nn.Linear(dim + 10, dim), nn.GELU(), nn.Linear(dim, dim))
        self.sequence_expert = nn.Sequential(nn.Linear(12, dim), nn.GELU(), nn.Linear(dim, dim))
        self.prototype_encoder = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.prototype_projection = nn.Linear(dim, dim, bias=False)
        self.class_prototypes = nn.Parameter(torch.empty(4, dim))
        nn.init.orthogonal_(self.class_prototypes)
        self.router = nn.Sequential(nn.Linear(dim + 10 + 12 + 14 + 7 + 5, config.gate_hidden_size),
            nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.gate_hidden_size, 3))
        self.global_gate = nn.Parameter(torch.zeros(3))
        self.norm = nn.LayerNorm(dim)
        self.span_head = nn.Linear(dim, 1)
        self.type_head = nn.Linear(dim, 4)
        self.prototype_to_type = nn.Linear(4, 4, bias=False)
        self.control = None
        for layer in (self.context_expert[-1], self.sequence_expert[-1], self.prototype_projection,
                      self.orthographic, self.lexical, self.prototype_to_type):
            nn.init.normal_(layer.weight, std=0.001)
            if getattr(layer, "bias", None) is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.domain_embedding.weight)
        if config.parameter_matched_control:
            self._build_parameter_control()

    def _build_parameter_control(self):
        """Replace the routed branch with an equally sized active span MLP."""
        branches = [self.context_expert, self.sequence_expert, self.prototype_encoder,
                    self.prototype_projection, self.router, self.prototype_to_type]
        budget = sum(p.numel() for branch in branches for p in branch.parameters())
        budget += self.class_prototypes.numel() + self.global_gate.numel()
        for branch in branches:
            for parameter in branch.parameters():
                parameter.requires_grad_(False)
        self.class_prototypes.requires_grad_(False)
        self.global_gate.requires_grad_(False)
        dim = self.config.fusion_dim
        inputs = dim + 10 + 12 + 14 + 7 + 5
        width = max(1, round((budget - dim) / (inputs + 1 + dim)))
        self.control = nn.Sequential(nn.Linear(inputs, width), nn.GELU(),
                                     nn.Dropout(self.config.dropout), nn.Linear(width, dim))

    @torch.no_grad()
    def initialize_prototypes(self, bundle: CandidateBundle, indices, device):
        self.eval()
        labels = bundle.frame.iloc[indices].type_label.to_numpy(int)
        positive = np.asarray(indices)[labels >= 0]
        if not len(positive):
            return
        context = self.context_projection(bundle.context[positive].to(device))
        encoded = F.normalize(self.prototype_encoder(context), dim=-1)
        labels = bundle.frame.iloc[positive].type_label.to_numpy(int)
        for subtype in range(4):
            mask = torch.tensor(labels == subtype, device=device)
            if mask.any():
                self.class_prototypes[subtype].copy_(F.normalize(encoded[mask].mean(0), dim=0))

    def _route(self, inputs, weak):
        config = self.config
        available = torch.tensor([config.use_transformer_expert, config.use_crf_expert,
                                  config.use_prototype_expert], device=inputs.device)
        if config.parameter_matched_control:
            return torch.zeros((len(inputs), 3), device=inputs.device)
        if config.shared_gate:
            logits = self.global_gate.expand(len(inputs), -1)
        elif config.static_average:
            logits = inputs.new_zeros((len(inputs), 3))
        elif not config.dynamic_router:
            # Prespecified context-dominant control, without learned routing.
            logits = inputs.new_tensor([1.0, 0.0, 0.0]).expand(len(inputs), -1)
        else:
            logits = self.router(inputs)
        logits = logits.masked_fill(~available.unsqueeze(0), -1e4)
        if config.weak_signal_recovery:
            logits = logits.clone()
            logits[~weak, 2] = -1e4
        dense = logits.softmax(-1)
        if config.sparse_top2 and not (config.static_average or config.shared_gate):
            values, indices = dense.topk(2, dim=-1)
            dense = torch.zeros_like(dense).scatter(1, indices, values)
            dense = dense / dense.sum(-1, keepdim=True).clamp_min(1e-8)
        return dense

    def forward(self, batch):
        config = self.config
        context_input = batch["context"] if config.use_transformer_expert else torch.zeros_like(batch["context"])
        context = self.context_projection(context_input)
        if config.use_transformer_expert:
            context = context + self.context_adapter(context)
        if config.domain_adapter:
            context = context + self.domain_embedding(batch["source_id"])
        transformer = batch["transformer"] if config.use_transformer_expert else torch.zeros_like(batch["transformer"])
        sequence = batch["crf"] if config.use_crf_expert else torch.zeros_like(batch["crf"])
        orthographic = batch["orthographic"] if config.use_orthographic else torch.zeros_like(batch["orthographic"])
        lexical = batch["lexical"] if config.use_lexical_reliability else torch.zeros_like(batch["lexical"])
        agreement, weak_signal, rare_signal = batch["agreement"], batch["weak_signal"], batch["rare_signal"]
        if not (config.use_transformer_expert and config.use_crf_expert):
            evidence = sequence if config.use_crf_expert else transformer
            distribution = evidence[:, -4:]
            entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(-1)
            rare_signal = distribution[:, [TYPE_TO_ID["brand"], TYPE_TO_ID["drug_n"]]].max(1).values
            agreement = torch.zeros_like(agreement)
            weak_signal = ((entropy > 0.85) | (batch["seen_in_train"] == 0)
                | batch["multi_token"].bool() | (batch["source_id"] == 0)
                | (rare_signal > 0.25) | (evidence[:, 1] < 0.72)).float()
        signals = torch.stack([agreement, weak_signal, batch["seen_in_train"],
                               batch["multi_token"], rare_signal], dim=-1)
        inputs = torch.cat([context, transformer, sequence, orthographic, lexical, signals], dim=-1)
        weak = weak_signal.bool() if config.weak_signal_recovery else torch.ones(len(context), dtype=torch.bool, device=context.device)
        weights = self._route(inputs, weak)
        prototype = torch.zeros_like(context)
        prototype_logits = context.new_zeros((len(context), 4))
        active = (weights[:, 2] > 0) if config.early_exit and not self.training else torch.ones_like(weak)
        active = active & config.use_prototype_expert & ~torch.full_like(weak, config.parameter_matched_control)
        if active.any():
            # The expensive branch is evaluated only for routed candidates.
            encoded = F.normalize(self.prototype_encoder(context[active]), dim=-1)
            prototype[active] = encoded
            prototype_logits[active] = encoded @ F.normalize(self.class_prototypes, dim=-1).T / 0.2
        if self.control is not None:
            residual = self.control(inputs)
        else:
            outputs = torch.stack([self.context_expert(torch.cat([context, transformer], dim=-1)),
                                   self.sequence_expert(sequence), self.prototype_projection(prototype)], dim=1)
            residual = (outputs * weights.unsqueeze(-1)).sum(1)
        # No lexical boundary predictor: reliability enters routing and type residual.
        representation = self.norm(context + residual + self.orthographic(orthographic))
        span_logits = self.span_head(representation).squeeze(-1)
        type_logits = self.type_head(representation + self.lexical(lexical))
        if self.control is None and config.use_prototype_expert:
            type_logits = type_logits + self.prototype_to_type(prototype_logits) * weights[:, 2:3]
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
        return {"span_logits": span_logits, "type_logits": type_logits,
                "router_weights": weights, "router_entropy": entropy,
                "prototype_embedding": prototype, "prototype_logits": prototype_logits,
                "prototype_active": active, "early_exit": ~active, "weak_signal": weak_signal}


def tensor_batch(bundle: CandidateBundle, indices, device):
    positions = np.asarray(indices)
    rows = bundle.frame.iloc[positions]
    keys = ("context", "transformer", "crf", "orthographic", "lexical", "transformer_type", "crf_type")
    batch = {key: getattr(bundle, key)[positions].to(device) for key in keys}
    for key, values in {
        "span_label": rows.span_label, "type_label": rows.type_label,
        "source_id": rows.source.eq("DrugBank"), "agreement": rows.expert_agreement,
        "weak_signal": rows.weak_signal, "seen_in_train": rows.seen_in_train,
        "multi_token": rows.multi_token, "rare_signal": rows.rare_signal_probability,
    }.items():
        dtype = torch.long if key in ("source_id", "type_label") else torch.float32
        batch[key] = torch.tensor(values.to_numpy(), device=device, dtype=dtype)
    return batch


def supervised_contrastive(embeddings, labels, temperature=0.2):
    if len(embeddings) < 2:
        return embeddings.sum() * 0
    similarity = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positive.any(1)
    if not valid.any():
        return embeddings.sum() * 0
    log_prob = F.log_softmax(similarity.masked_fill(diagonal, -1e4), dim=-1)
    return -((log_prob * positive).sum(1) / positive.sum(1).clamp_min(1))[valid].mean()


def composite_loss(model, batch, output, counts):
    config = model.config
    span_losses = F.binary_cross_entropy_with_logits(output["span_logits"], batch["span_label"], reduction="none")
    positive = batch["type_label"] >= 0
    type_losses = torch.zeros_like(span_losses)
    zero = span_losses.sum() * 0
    prototype_loss = zero
    if positive.any():
        logits, labels = output["type_logits"][positive], batch["type_label"][positive]
        if config.class_balanced_type_loss:
            if config.type_loss == "effective_number":
                weights = (1 - 0.999) / (1 - 0.999 ** counts).clamp_min(1e-7)
                weights = weights / weights.mean()
                losses = F.cross_entropy(logits, labels, weight=weights, reduction="none")
            elif config.type_loss == "logit_adjusted":
                priors = counts / counts.sum()
                losses = F.cross_entropy(logits + priors.log().unsqueeze(0), labels, reduction="none")
            else:
                probability = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
                losses = (1 - probability).square() * F.cross_entropy(logits, labels, reduction="none")
        else:
            losses = F.cross_entropy(logits, labels, reduction="none")
        multiplier = 1 + 0.35 * output["weak_signal"][positive] if config.weak_signal_recovery else 1
        type_losses[positive] = losses * multiplier
        proto = positive & output["prototype_active"]
        if proto.any() and not config.parameter_matched_control:
            prototype_loss = F.cross_entropy(output["prototype_logits"][proto], batch["type_label"][proto])
            prototype_loss += supervised_contrastive(output["prototype_embedding"][proto][:128], batch["type_label"][proto][:128])
    type_loss = type_losses[positive].mean() if positive.any() else zero
    total_per_span = span_losses + config.lambda_type * type_losses
    domain_loss = zero
    if config.source_balancing == "group_dro":
        domain_loss = torch.stack([total_per_span[batch["source_id"] == source].mean()
            for source in (0, 1) if (batch["source_id"] == source).any()]).max()
    router_loss = zero
    if config.router_regularization and config.dynamic_router and not config.parameter_matched_control:
        weights = output["router_weights"]
        available = weights.new_tensor([config.use_transformer_expert,
            config.use_crf_expert, config.use_prototype_expert])
        target = available / available.sum().clamp_min(1)
        router_loss = (weights.mean(0) - target).square().mean()
        router_loss += F.relu(0.3 - output["router_entropy"].mean())
    loss = span_losses.mean() + config.lambda_type * type_loss
    loss += config.lambda_prototype * prototype_loss + config.lambda_domain * domain_loss + config.lambda_router * router_loss
    components = {"span_loss": span_losses.mean(), "type_loss": type_loss,
                  "prototype_loss": prototype_loss, "domain_loss": domain_loss, "router_loss": router_loss}
    return loss, {key: float(value.detach().cpu()) for key, value in components.items()}


def eligible_candidates(frame, config):
    mask = frame.generated_candidate.to_numpy(bool)
    if not config.use_transformer_expert:
        mask &= frame[["origin_crf", "origin_high_crf", "origin_shift_crf"]].any(axis=1).to_numpy()
    if not config.use_crf_expert:
        mask &= frame[["origin_transformer", "origin_high_transformer", "origin_shift_transformer"]].any(axis=1).to_numpy()
    return mask


def score(model, bundle, document_ids=None, batch_size=512):
    mask = np.ones(len(bundle.frame), dtype=bool) if document_ids is None else bundle.frame.document_id.isin(document_ids).to_numpy()
    indices = np.flatnonzero(mask)
    device = next(model.parameters()).device
    model.eval()
    results = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            positions = indices[start:start + batch_size]
            output = model(tensor_batch(bundle, positions, device))
            rows = bundle.frame.iloc[positions].copy()
            rows["span_probability"] = output["span_logits"].sigmoid().cpu().numpy()
            types = output["type_logits"].softmax(-1).cpu().numpy()
            rows["type_vector"] = list(types)
            rows["type_probability"] = types.max(1)
            rows["predicted_type"] = [ENTITY_TYPES[value] for value in types.argmax(1)]
            rows["entity_probability"] = rows.span_probability * rows.type_probability
            for index, expert in enumerate(("context", "sequence", "prototype")):
                rows[f"router_{expert}"] = output["router_weights"][:, index].cpu().numpy()
            rows["router_entropy"] = output["router_entropy"].cpu().numpy()
            rows["early_exit"] = output["early_exit"].cpu().numpy()
            results.append(rows)
    return pd.concat(results, ignore_index=True), time.perf_counter() - started


def decode(scores, sentences, config, threshold, seed, split="prospective"):
    """Same verifier threshold and non-overlap rule for every architecture."""
    selected = scores.loc[eligible_candidates(scores, config) & scores.span_probability.ge(threshold)].copy()
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
    predictions = finalize_prediction_frame(emitted, sentences, "ACT-NER", split, seed)
    predictions["calibrated_confidence"] = predictions.boundary_probability * predictions.type_probability
    assert predictions.entity_id.is_unique
    return predictions


def strict_metrics(entities, predictions):
    gold = gold_for_metrics(entities)
    truth = set(zip(gold.sentence_id, gold.char_start, gold.char_end, gold.entity_type))
    predicted = set(zip(predictions.sentence_id, predictions.char_start, predictions.char_end, predictions.predicted_type))
    boundary_gold, boundary_pred = {key[:3] for key in truth}, {key[:3] for key in predicted}
    def triple(a, b):
        correct = len(a & b)
        return correct / max(len(b), 1), correct / max(len(a), 1), 2 * correct / max(len(a) + len(b), 1)
    bp, br, bf = triple(boundary_gold, boundary_pred)
    tp, tr, tf = triple(truth, predicted)
    subtype = {label + "_f1": triple({x for x in truth if x[3] == label},
                                   {x for x in predicted if x[3] == label})[2] for label in ENTITY_TYPES}
    return {"N": len(truth), "predictions": len(predicted), "boundary_precision": bp,
            "boundary_recall": br, "boundary_f1": bf, "typed_precision": tp,
            "typed_recall": tr, "typed_f1": tf, "macro_f1": float(np.mean(list(subtype.values()))), **subtype}


def select_threshold(scores, sentences, entities, config, seed, grid):
    rows = []
    for threshold in grid:
        prediction = decode(scores, sentences, config, threshold, seed)
        metrics = strict_metrics(entities, prediction)
        rows.append({"threshold": threshold, **metrics})
    table = pd.DataFrame(rows)
    order = ["typed_f1", "boundary_f1", "typed_recall", "macro_f1"]
    selected = table.sort_values(order + ["threshold"], ascending=[False] * 4 + [True]).iloc[0]
    return float(selected.threshold), table


def training_indices(bundle, document_ids, config, seed):
    frame = bundle.frame
    allowed = frame.document_id.isin(document_ids).to_numpy()
    positive = np.flatnonzero(allowed & frame.span_label.eq(1).to_numpy())
    negative = np.flatnonzero(allowed & frame.span_label.eq(0).to_numpy() & eligible_candidates(frame, config))
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


def fit_head(bundle, selection_bundle, training_ids, selection_sentences, selection_entities,
             config, seed, settings, callback=None):
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ACTNER(config).to(device)
    positions = training_indices(bundle, training_ids, config, seed)
    assert not set(bundle.frame.iloc[positions].document_id) & set(selection_sentences.document_id)
    model.initialize_prototypes(bundle, positions, device)
    labels = bundle.frame.iloc[positions].type_label.to_numpy(int)
    counts = torch.tensor(np.maximum(np.bincount(labels[labels >= 0], minlength=4), 1), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings["learning_rate"], weight_decay=1e-4)
    sampling_frame = bundle.frame.iloc[positions]
    weights = 1 / sampling_frame.document_id.map(sampling_frame.document_id.value_counts()).to_numpy(float)
    if config.source_balancing == "balanced_sampler":
        doc_sources = sampling_frame[["document_id", "source"]].drop_duplicates().source.value_counts()
        weights /= sampling_frame.source.map(doc_sources).to_numpy(float)
    if config.class_balanced_type_loss:
        weights *= np.where(np.isin(labels, [TYPE_TO_ID["brand"], TYPE_TO_ID["drug_n"]]), 2.0, 1.0)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    best_score, best_state, stale, history = -1.0, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = rng.choice(positions, len(positions), replace=True, p=weights)
        losses, components = [], []
        for start in range(0, len(order), settings["batch_size"]):
            batch = tensor_batch(bundle, order[start:start + settings["batch_size"]], device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = composite_loss(model, batch, model(batch), counts)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            components.append(terms)
        candidates, _ = score(model, selection_bundle)
        threshold, tuning = select_threshold(candidates, selection_sentences, selection_entities,
                                            config, seed, settings["threshold_grid"])
        metric = float(tuning.typed_f1.max())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_typed_f1": metric,
               "threshold": threshold, **pd.DataFrame(components).mean().to_dict()}
        history.append(row)
        if callback:
            callback(row)
        if metric > best_score + 1e-6:
            best_score, stale = metric, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_threshold, best_epoch = threshold, epoch
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_threshold, best_epoch, time.perf_counter() - started


def configuration_candidates(protocol):
    base = ACTNERConfig(continual_adaptation=False, shift_triggered=False,
                        lambda_consistency=0.0, boundary_threshold=0.5)
    return [replace(base, **candidate) for candidate in protocol["head"]["joint_candidates"]]


def ablations(config):
    variants = {
        "Full ACT-NER": {}, "Without Context Expert": {"use_transformer_expert": False},
        "Without Sequence Expert": {"use_crf_expert": False},
        "Without Prototype Expert": {"use_prototype_expert": False},
        "Without orthographic residual": {"use_orthographic": False},
        "Without lexical reliability": {"use_lexical_reliability": False},
        "Without dynamic router": {"dynamic_router": False},
        "Static average": {"static_average": True}, "Global learned gate": {"shared_gate": True},
        "Without weak-signal recovery": {"weak_signal_recovery": False},
        "Without hard-negative mining": {"hard_negative_mining": False},
        "Without class balancing": {"class_balanced_type_loss": False},
        "Without source balancing": {"source_balancing": "none"},
        "Without domain adapter": {"domain_adapter": False},
        "Without router regularization": {"router_regularization": False},
        "Without calibration": {"calibration": False}, "Without early exit": {"early_exit": False},
        "Parameter-matched span MLP": {"parameter_matched_control": True},
    }
    return {name: replace(config, **changes) for name, changes in variants.items()}
