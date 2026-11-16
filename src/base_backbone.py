"""Pinned stronger-backbone feasibility check; no original artifact writes."""
from pathlib import Path
import gc
import json
import time
import shutil
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForTokenClassification, AutoTokenizer
from core import TYPED_TAGS

MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
RUN = "extended_20261016"


def acquire_backbone(root: Path):
    destination = root/"Models/Extended"/RUN/"biomedbert_base_pretrained"
    snapshot_download(MODEL, revision=REVISION, local_dir=destination,
        allow_patterns=["config.json", "pytorch_model.bin", "vocab.txt",
                        "tokenizer_config.json", "special_tokens_map.json", "README.md"])
    return destination


def preflight(root: Path, destination: Path, progress=None):
    """Benchmark real Train batches, without optimizer updates or held-out labels."""
    sentences = pd.read_parquet(root/"dataset/Processed/train_sentences.parquet")
    tokenizer = AutoTokenizer.from_pretrained(destination, use_fast=True, local_files_only=True)
    lengths = [len(tokenizer(t, add_special_tokens=True, truncation=False).input_ids)
               for t in sentences.text]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(destination,
        num_labels=len(TYPED_TAGS), local_files_only=True).to(device)
    model.train()
    records = []
    for batch_size in (4, 8):
        encoded = tokenizer(sentences.text.iloc[:batch_size].tolist(),
            padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        # Train-only proxy labels measure compute, not predictive performance.
        labels = torch.zeros_like(encoded.input_ids)
        labels[encoded.attention_mask == 0] = -100
        times = []
        for repeat in range(3):
            model.zero_grad(set_to_none=True)
            start = time.perf_counter()
            model(**encoded, labels=labels).loss.backward()
            if device.type == "mps": torch.mps.synchronize()
            times.append(time.perf_counter()-start)
        record = {"Model":MODEL, "Revision":REVISION, "Device":str(device),
            "Parameters":sum(p.numel() for p in model.parameters()),
            "Batch":batch_size, "Padded tokens":encoded.input_ids.shape[1],
            "Forward/backward seconds":sum(times[1:])/2,
            "MPS allocated GB":torch.mps.current_allocated_memory()/1e9 if device.type=="mps" else None,
            "Train sentences >512":sum(v>512 for v in lengths),
            "Maximum Train length":max(lengths), "Free disk GB":shutil.disk_usage(root).free/1e9,
            "Optimizer updates":0}
        records.append(record)
        if progress: progress(record)
    del model
    gc.collect()
    if device.type == "mps": torch.mps.empty_cache()
    return pd.DataFrame(records)
