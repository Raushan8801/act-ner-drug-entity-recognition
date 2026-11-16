"""Read-only R1.5 corpus checks; no model imports, training, or source writes.

Each public audit returns named DataFrames for visible Notebook 07 evidence.
The original manifests, raw XML, cached annotations, and lexical resources
are compared without rebuilding or replacing them.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "locked_test")
TYPES = ("drug", "brand", "group", "drug_n")
ORIGINAL_RUN = "refinement_20260907"


def normalized_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def normalized_surface(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower().strip()
    return re.sub(r"\s+", " ", re.sub("[‐‑‒–—−]", "-", value))


def _check(name: str, observed, expected, passed: bool) -> dict:
    return {"Check": name, "Observed": observed, "Expected": expected,
            "Status": "PASS" if passed else "FAIL"}


def _shared(root: Path) -> Path:
    return Path(root) / "dataset"


def _frames(root: Path, split: str):
    folder = _shared(root) / "Processed"
    return tuple(pd.read_parquet(folder / f"{split}_{name}.parquet")
                 for name in ("sentences", "entities", "tokens"))


def _xml_rows(shared: Path, manifest: pd.DataFrame):
    sentences, entities, identifiers = [], [], []
    for row in manifest.itertuples(index=False):
        path = shared / row.relative_xml
        tree = ET.parse(path).getroot()
        identifiers.append(tree.attrib["id"] == row.document_id
                           and path.parent.name == row.source)
        for sentence in tree.findall("sentence"):
            sid = sentence.attrib["id"]
            sentences.append((row.document_id, sid, sentence.attrib["text"]))
            for entity in sentence.findall("entity"):
                a = entity.attrib
                entities.append((row.document_id, sid, a["id"], a["charOffset"],
                                 a["text"], a["type"]))
    return sentences, entities, identifiers


def partition_audit(root: Path) -> dict[str, pd.DataFrame]:
    """Check raw/processed identity; retain documents with zero gold mentions."""
    shared = _shared(root)
    summaries, checks, empty_documents, anomalies = [], [], [], []
    manifests = {s: pd.read_csv(shared / "Splits" / f"{s}_documents.csv") for s in SPLITS}
    for split, manifest in manifests.items():
        sentences, entities, tokens = _frames(root, split)
        raw_s, raw_e, ids = _xml_rows(shared, manifest)
        sc = ["document_id", "sentence_id", "text"]
        ec = ["document_id", "sentence_id", "entity_id", "char_offset_xml", "entity_text", "entity_type"]
        gold = entities.loc[~entities.discontinuous & entities.offset_valid]
        texts = sentences.set_index("sentence_id").text.to_dict()
        mismatch = [r.entity_id for r in gold.itertuples()
                    if texts[r.sentence_id][r.char_start:r.char_end] != r.entity_text]
        token_mismatch = sum(texts[r.sentence_id][r.char_start:r.char_end] != r.surface_text
                             for r in tokens.itertuples())
        missing = sorted(set(manifest.document_id) - set(gold.document_id))
        empty_documents.extend({"Split": split, "Document": d, "Gold entities": 0} for d in missing)
        summaries.append({"Split": split, "Documents": len(manifest),
            "DrugBank": int(manifest.source.eq("DrugBank").sum()),
            "MedLine": int(manifest.source.eq("MedLine").sum()), "Sentences": len(sentences),
            "Raw entities": len(entities), "Discontinuous": int(entities.discontinuous.sum()),
            "Strict gold": len(gold), "Zero-gold documents": len(missing)})
        tests = {"XML identifiers/source": all(ids),
            "Manifest/sentence/token document universe": set(manifest.document_id) == set(sentences.document_id) == set(tokens.document_id),
            "Entity documents within manifest": set(entities.document_id) <= set(manifest.document_id),
            "Raw/processed sentences": set(raw_s) == set(sentences[sc].itertuples(index=False, name=None)),
            "Raw/processed annotations": set(raw_e) == set(entities[ec].itertuples(index=False, name=None)),
            "Unique sentence IDs": sentences.sentence_id.is_unique,
            "Unique entity IDs": entities.entity_id.is_unique,
            "Unique sentence/token coordinates": not tokens.duplicated(["sentence_id", "token_index"]).any(),
            "Valid annotation offsets": bool(entities.offset_valid.all()),
            "Eligible entity text restoration": not mismatch,
            "Token text restoration": token_mismatch == 0}
        checks.extend({"Split": split, **_check(k, bool(v), True, v)} for k, v in tests.items())
        for row in entities.loc[~entities.text_exact].itertuples():
            anomalies.append({"Split": split, "Entity": row.entity_id,
                "Discontinuous": row.discontinuous, "Eligible": not row.discontinuous and row.offset_valid,
                "Original offsets": row.char_offset_xml, "Original text": row.entity_text})
    for left, right in combinations(SPLITS, 2):
        n = len(set(manifests[left].document_id) & set(manifests[right].document_id))
        checks.append({"Split": f"{left} / {right}", **_check("Document-ID overlap", n, 0, n == 0)})
    return {"partitions": pd.DataFrame(summaries), "integrity": pd.DataFrame(checks),
            "zero_gold_documents": pd.DataFrame(empty_documents),
            "annotation_exceptions": pd.DataFrame(anomalies)}


def archive_audit(root: Path) -> pd.DataFrame:
    """Verify retained ZIP and every extracted XML against original bytes."""
    shared = _shared(root)
    record = json.loads((shared / "Raw/ACQUISITION.json").read_text())
    archive = shared / "Raw" / record["archive"]
    archive_ok = hashlib.sha256(archive.read_bytes()).hexdigest() == record["sha256"]
    with ZipFile(archive) as bundle:
        names = [n for n in bundle.namelist() if n.endswith(".xml") and not n.startswith("__MACOSX/")]
        failures = [n for n in names if not (shared / "Raw/DDICorpus-2013" / n).exists()
                    or (shared / "Raw/DDICorpus-2013" / n).read_bytes() != bundle.read(n)]
    return pd.DataFrame([_check("Original archive digest", archive_ok, True, archive_ok),
        _check("Extracted XML checked", len(names), 1017, len(names) == 1017),
        _check("XML byte mismatches", len(failures), 0, not failures)])


def split_reconstruction(root: Path) -> pd.DataFrame:
    """Independently reproduce the original seeded selection, without saving it."""
    shared = _shared(root)
    settings = json.loads((Path(root) / "src/experiment.json").read_text())
    documents, counts = [], {}
    for path in sorted((shared / "Raw/DDICorpus-2013/DDICorpus/Train").glob("*/*.xml")):
        tree = ET.parse(path).getroot()
        identifier = tree.attrib["id"]
        documents.append((identifier, path.parent.name))
        counts[identifier] = Counter(e.attrib["type"] for e in tree.iter("entity")
                                     if ";" not in e.attrib["charOffset"])
    profile = pd.DataFrame(documents, columns=["document_id", "source"]).set_index("document_id")
    for kind in TYPES:
        profile[kind] = [counts[d].get(kind, 0) for d in profile.index]
    rng = np.random.default_rng(settings["split_seed"])
    chosen, rows = set(), []
    fraction = settings["validation_fraction_of_original_train"]
    for source, group in profile.groupby("source"):
        identifiers, values = group.index.to_numpy(), group[list(TYPES)].to_numpy(float)
        n = round(len(identifiers) * fraction)
        target = values.sum(0) * fraction
        best_score, best = float("inf"), None
        for _ in range(4000):
            index = rng.choice(len(identifiers), n, replace=False)
            score = np.mean(((values[index].sum(0) - target) / np.maximum(target, 1)) ** 2)
            if score < best_score:
                best_score, best = score, index
        chosen.update(identifiers[best])
        rows.append({"Source": source, "Original documents": len(group),
                     "Validation documents": n, "Stratification objective": best_score})
    saved = set(pd.read_csv(shared / "Splits/validation_documents.csv").document_id)
    result = pd.DataFrame(rows)
    result["Seed"] = settings["split_seed"]
    result["Proposals per source"] = 4000
    result["Exact saved membership"] = chosen == saved
    result["Status"] = "PASS" if chosen == saved else "FAIL"
    return result


def lexicon_audit(root: Path) -> pd.DataFrame:
    """Reconstruct every learned lexical statistic strictly from original Train."""
    entities = _frames(root, "train")[1].query("not discontinuous").copy()
    entities["normalized_surface"] = entities.entity_text.map(normalized_surface)
    counts = pd.crosstab(entities.normalized_surface, entities.entity_type).reindex(columns=TYPES, fill_value=0).sort_index()
    probabilities = counts.div(counts.sum(1), axis=0)
    entropy = -(probabilities.where(probabilities > 0, 1) * np.log(probabilities.where(probabilities > 0, 1))).sum(1) / np.log(4)
    lexical = pd.read_parquet(_shared(root) / "Processed/training_lexicon.parquet").set_index("normalized_surface").sort_index()
    vocabulary_ok = set(lexical.index) == set(counts.index)
    tests = {"Train-only vocabulary": vocabulary_ok,
             "Train-only mention frequencies": vocabulary_ok and np.array_equal(lexical.frequency, counts.sum(1)),
             "Train-only majority types": vocabulary_ok and np.array_equal(lexical.majority_type, counts.idxmax(1)),
             "Train-only subtype probabilities": vocabulary_ok and np.allclose(lexical[[f"p_{k}" for k in TYPES]], probabilities),
             "Train-only normalized entropy": vocabulary_ok and np.allclose(lexical.ambiguity_score, entropy)}
    return pd.DataFrame([{**_check(k, bool(v), True, v), "Lexical forms": len(lexical),
                         "Train mentions": len(entities)} for k, v in tests.items()])


def _text_index(sentences: pd.DataFrame, split: str):
    index = defaultdict(list)
    for row in sentences.itertuples(index=False):
        key = normalized_sentence(row.text)
        if len(key) >= 40 and len(key.split()) >= 6:
            index[key].append((split, row.document_id, row.sentence_id))
    return index


def text_overlap_audit(root: Path) -> dict[str, pd.DataFrame]:
    """Use the exact NFKC/casefold/whitespace, >=40-char, >=6-token rule."""
    indices = {split: _text_index(_frames(root, split)[0], split) for split in SPLITS}
    summary, reasons = [], []
    for left, right in combinations(SPLITS, 2):
        keys = set(indices[left]) & set(indices[right])
        left_docs, right_docs = set(), set()
        for key in sorted(keys):
            for a in indices[left][key]:
                for b in indices[right][key]:
                    left_docs.add(a[1]); right_docs.add(b[1])
                    reasons.append({"Left split": left, "Right split": right,
                        "Left document": a[1], "Right document": b[1],
                        "Left sentence": a[2], "Right sentence": b[2],
                        "Characters": len(key), "Whitespace tokens": len(key.split()),
                        "Normalized-text digest": hashlib.sha256(key.encode()).hexdigest()})
        summary.append({"Left split": left, "Right split": right,
            "Shared text forms": len(keys), "Left documents": len(left_docs),
            "Right documents": len(right_docs), "Status": "DEPENDENCE PRESENT" if keys else "NO RULE MATCH"})
    return {"text_overlap": pd.DataFrame(summary), "text_overlap_reasons": pd.DataFrame(reasons)}


def reserve_audit(root: Path) -> dict[str, pd.DataFrame]:
    """Reproduce 191 -> 15 exclusions -> 176; never use predictions or tune."""
    root, shared = Path(root), _shared(root)
    path = root / "Reports" / ORIGINAL_RUN / "additional_holdout_plan.json"
    plan = json.loads(path.read_text())
    plan_ok = hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".sha256").read_text()
    known = defaultdict(list)
    for split in SPLITS:
        for key, values in _text_index(_frames(root, split)[0], split).items():
            known[key].extend(values)
    directory = shared / "Interim" / ORIGINAL_RUN / "additional_holdout"
    sentences = pd.read_parquet(directory / "sentences.parquet")
    entities = pd.read_parquet(directory / "entities.parquet")
    manifest = pd.read_csv(shared / "Splits" / f"{ORIGINAL_RUN}_additional_holdout_documents.csv")
    rows = []
    for row in sentences.itertuples(index=False):
        key = normalized_sentence(row.text)
        for split, doc, sid in known.get(key, []):
            rows.append({"Reserve document": row.document_id, "Reserve sentence": row.sentence_id,
                "Prior split": split, "Prior document": doc, "Prior sentence": sid,
                "Characters": len(key), "Whitespace tokens": len(key.split()),
                "Reason": "Exact substantial normalized sentence overlap"})
    reasons = pd.DataFrame(rows)
    excluded = {r["Reserve document"] for r in rows}
    retained = set(sentences.document_id) - excluded
    recorded = set(manifest.loc[manifest.eligible_independent_text, "document_id"])
    source_manifest = pd.DataFrame(plan["files"])
    raw_s, raw_e, identifiers = _xml_rows(shared, source_manifest)
    prior_ids = set().union(*(set(pd.read_csv(shared / "Splits" / f"{s}_documents.csv").document_id) for s in SPLITS))
    raw_paths = {str(p.relative_to(shared)) for p in (shared / "Raw/DDICorpus-2013/DDICorpus/Test/Test for DDI Extraction task").glob("*/*.xml")}
    sc = ["document_id", "sentence_id", "text"]
    ec = ["document_id", "sentence_id", "entity_id", "char_offset_xml", "entity_text", "entity_type"]
    digests_ok = all(hashlib.sha256((shared / r["relative_xml"]).read_bytes()).hexdigest() == r["sha256"] for r in plan["files"])
    tests = {"Plan seal": plan_ok, "Reserve XML digests": digests_ok,
        "Reserve identifiers/source": all(identifiers),
        "Complete original reserve XML inventory": raw_paths == set(source_manifest.relative_xml),
        "Unique reserve document IDs": manifest.document_id.is_unique,
        "Reserve/prior document-ID separation": not (set(manifest.document_id) & prior_ids),
        "Reserve raw/cached sentences": set(raw_s) == set(sentences[sc].itertuples(index=False, name=None)),
        "Reserve raw/cached annotations": set(raw_e) == set(entities[ec].itertuples(index=False, name=None)),
        "Exact retained membership": retained == recorded,
        "No retained qualifying text overlap": not (retained & excluded),
        "Original reserve count": len(manifest) == 191,
        "Excluded count": len(excluded) == 15, "Retained count": len(retained) == 176}
    gold = entities.loc[~entities.discontinuous & entities.offset_valid]
    population = pd.DataFrame([{"Stage": label, "Documents": len(ids),
        "DrugBank": int(manifest.loc[manifest.document_id.isin(ids), "source"].eq("DrugBank").sum()),
        "MedLine": int(manifest.loc[manifest.document_id.isin(ids), "source"].eq("MedLine").sum()),
        "Sentences": int(sentences.document_id.isin(ids).sum()),
        "Eligible gold": int(gold.document_id.isin(ids).sum())}
        for label, ids in (("Original DDI reserve", set(manifest.document_id)),
                           ("Rule exclusions", excluded), ("Text-disjoint same-corpus subset", retained))])
    return {"reserve_construction": population, "reserve_exclusions": reasons,
            "reserve_integrity": pd.DataFrame([_check(k, bool(v), True, v) for k, v in tests.items()])}
