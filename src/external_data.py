"""Pinned BC5CDR acquisition and data-only checks.

No model imports or training. Official partitions and all gold spans are retained.
Test annotations may be read only for declared schema/offset integrity auditing;
no test metric, example inspection, tuning, or DDI-to-CDR label mapping occurs.
All writes stay in the new External/extended_20261016/BC5CDR namespace.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

REVISION = "dd0b3165a94338874979a2a8aaedf74b67466b9c"
ARCHIVE_SHA256 = "0a359a7f038d283a7b05b084fa73de014e7410e1f5d7034bf3fd01f016fc2444"
ARCHIVE_GIT_BLOB = "d6f7468c14306d560ae333e12615d35e5f00b992"
BASE_URL = f"https://raw.githubusercontent.com/JHnlp/BioCreative-V-CDR-Corpus/{REVISION}/"
GOLD_PREFIX = "CDR_Data/CDR.Corpus.v010516/"
SPLIT_FILES = {"train": "Training", "validation": "Development", "test": "Test"}
LABELS = ("Chemical", "Disease")


def data_directory(root: Path) -> Path:
    return Path(root).resolve() / "dataset/External/extended_20261016/BC5CDR"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Existing external artifact differs; preserve it: {path.name}")
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ACT-NER-DataCheck/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def acquire_bc5cdr(root: Path) -> pd.DataFrame:
    """Download once; verify pinned bytes and retain actual archive rights notice."""
    folder = data_directory(root)
    archive = folder / "Raw/CDR_Data.zip"
    payload = archive.read_bytes() if archive.exists() else _read_url(BASE_URL + "CDR_Data.zip")
    if _sha(payload) != ARCHIVE_SHA256:
        raise ValueError("BC5CDR pinned archive SHA256 mismatch; no conversion permitted")
    blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    if blob != ARCHIVE_GIT_BLOB:
        raise ValueError("BC5CDR pinned Git blob mismatch")
    with ZipFile(io.BytesIO(payload)) as bundle:
        notice = bundle.read("CDR_Data/README.txt")
        if "PUBLIC DOMAIN NOTICE" not in notice.decode("utf-8"):
            raise ValueError("Expected source rights notice is absent")
        members = {name: _sha(bundle.read(name)) for name in bundle.namelist()
                   if name.startswith(GOLD_PREFIX) and name.endswith(".PubTator.txt")}
    _immutable_bytes(archive, payload)
    _immutable_bytes(folder / "Raw/README.archive.txt", notice)
    provenance = {"corpus": "BC5CDR", "repository": "https://github.com/JHnlp/BioCreative-V-CDR-Corpus",
        "revision": REVISION, "download_url": BASE_URL + "CDR_Data.zip", "sha256": _sha(payload),
        "git_blob_sha1": blob, "bytes": len(payload), "gold_members_sha256": members,
        "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4860626/",
        "rights": "NCBI public-domain notice in archive README; no Creative Commons claim",
        "test_role": "Schema/offset checks only until a separately frozen model is evaluated"}
    _immutable_bytes(folder / "provenance.json", json.dumps(provenance, indent=2).encode())
    return pd.DataFrame([{"Corpus": "BC5CDR", "Official partitions": "500 / 500 / 500",
        "Source revision": REVISION, "Archive SHA256": _sha(payload),
        "Rights evidence": "Archive NCBI public-domain notice", "Status": "VERIFIED"}])


def parse_pubtator(payload: str, split: str):
    """Preserve exact document text and unmodified half-open mention coordinates."""
    documents, entities, relations = [], [], []
    for block in re.split(r"\n\s*\n", payload.strip()):
        lines = block.splitlines()
        pmid, title_marker, title = lines[0].split("|", 2)
        abstract_pmid, abstract_marker, abstract = lines[1].split("|", 2)
        if (title_marker, abstract_marker, abstract_pmid) != ("t", "a", pmid):
            raise ValueError("Invalid PubTator title/abstract identifiers")
        documents.append({"document_id": pmid, "pmid": pmid, "split": split,
            "title": title, "abstract": abstract, "text": title + " " + abstract, "source": "BC5CDR"})
        for line in lines[2:]:
            fields = line.split("\t")
            if fields[0] != pmid:
                raise ValueError("Annotation belongs to another PMID")
            if fields[1] == "CID":
                relations.append({"document_id": pmid, "relation_type": "CID",
                                  "chemical_concept": fields[2], "disease_concept": fields[3]})
                continue
            if len(fields) < 6 or fields[4] not in LABELS:
                raise ValueError("Unexpected gold annotation schema; no automatic remapping")
            entities.append({"document_id": pmid, "entity_id": f"{pmid}.e{len(entities)}",
                "char_start": int(fields[1]), "char_end": int(fields[2]),
                "entity_text": fields[3], "entity_type": fields[4], "concept_id": fields[5],
                "composite_mentions": "\t".join(fields[6:])})
    return pd.DataFrame(documents), pd.DataFrame(entities), pd.DataFrame(relations)


def _check(name: str, value, expected, passed: bool, split: str = "all") -> dict:
    return {"Split": split, "Check": name, "Observed": value, "Expected": expected,
            "Status": "PASS" if passed else "FAIL"}


def _overlapping_gold(entities: pd.DataFrame) -> int:
    count = 0
    for _, group in entities.groupby("document_id"):
        spans = list(group[["char_start", "char_end"]].itertuples(index=False, name=None))
        count += sum(a < d and c < b for (a, b), (c, d) in combinations(spans, 2))
    return count


def _persist_frame(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def prepare_bc5cdr(root: Path) -> dict[str, pd.DataFrame]:
    """Convert official gold only and report structural checks, never performance.

    Test labels are accessed for integrity only, logged explicitly. No test text
    examples or class-specific outcome distributions are displayed by this API.
    """
    folder = data_directory(root)
    archive = folder / "Raw/CDR_Data.zip"
    payload = archive.read_bytes()
    if _sha(payload) != ARCHIVE_SHA256:
        raise ValueError("Acquisition verification is required before conversion")
    checks, summaries = [], []
    with ZipFile(io.BytesIO(payload)) as bundle:
        for split, marker in SPLIT_FILES.items():
            member = GOLD_PREFIX + f"CDR_{marker}Set.PubTator.txt"
            documents, entities, relations = parse_pubtator(bundle.read(member).decode("utf-8"), split)
            texts = documents.set_index("document_id").text.to_dict()
            offsets = [0 <= r.char_start < r.char_end <= len(texts[r.document_id])
                       and texts[r.document_id][r.char_start:r.char_end] == r.entity_text
                       for r in entities.itertuples()]
            duplicate = int(entities.duplicated(["document_id", "char_start", "char_end", "entity_type"]).sum())
            tests = [("Official document count", len(documents), 500, len(documents) == 500),
                ("Unique PMID", documents.pmid.nunique(), len(documents), documents.pmid.is_unique),
                ("Valid exact half-open offsets", sum(offsets), len(entities), all(offsets)),
                ("Gold categories unchanged", ", ".join(sorted(entities.entity_type.unique())),
                 "Chemical, Disease", set(entities.entity_type) == set(LABELS)),
                ("Duplicate typed spans", duplicate, 0, duplicate == 0)]
            checks.extend(_check(*test, split=split) for test in tests)
            if any(not test[3] for test in tests):
                raise ValueError(f"BC5CDR {split} integrity failure: {tests}")
            for kind, frame in zip(("documents", "entities", "relations"), (documents, entities, relations)):
                _persist_frame(folder / "Processed" / f"{split}_{kind}.parquet", frame)
            summaries.append({"Split": split, "Documents": len(documents), "Gold mentions": len(entities),
                "Overlapping gold pairs": _overlapping_gold(entities),
                "Composite mention rows": int(entities.composite_mentions.ne("").sum()),
                "Use": "Final evaluation only" if split == "test" else "Model development"})
    record = {"operation": "PubTator schema/offset integrity, no model or metric",
              "scope": "Official train/development/test; test examples not displayed",
              "archive_sha256": ARCHIVE_SHA256}
    _immutable_bytes(folder / "integrity_access.json", json.dumps(record, indent=2).encode())
    return {"partitions": pd.DataFrame(summaries), "integrity": pd.DataFrame(checks),
            "label_compatibility": label_compatibility()}


def load_bc5cdr_split(root: Path, split: str, *, include_annotations: bool = False):
    """Explicit annotation access; default returns document text and empty labels."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown official split: {split}")
    folder = data_directory(root) / "Processed"
    documents = pd.read_parquet(folder / f"{split}_documents.parquet")
    if not include_annotations:
        return documents, pd.DataFrame(), pd.DataFrame()
    return (documents, pd.read_parquet(folder / f"{split}_entities.parquet"),
            pd.read_parquet(folder / f"{split}_relations.parquet"))


def label_compatibility() -> pd.DataFrame:
    return pd.DataFrame([
        {"Corpus label": "Chemical", "DDI equivalence": "No one-to-one equivalence",
         "Reason": "Chemical scope is broader; drug/brand/group/drug_n are not supplied",
         "Protocol": "Retain Chemical in an explicitly adapted output head"},
        {"Corpus label": "Disease", "DDI equivalence": "Outside original DDI label space",
         "Reason": "Original ACT-NER has no Disease class",
         "Protocol": "Retain Disease; retrain applicable heads/experts on external Train only"}])


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _ddi_inventory(root: Path):
    shared = Path(root).resolve() / "dataset"
    roles = {}
    for split in ("train", "validation", "locked_test"):
        frame = pd.read_csv(shared / "Splits" / f"{split}_documents.csv")
        roles.update(dict(zip(frame.document_id, [split] * len(frame))))
    documents, sentences = [], []
    for path in sorted((shared / "Raw/DDICorpus-2013/DDICorpus").rglob("*.xml")):
        node = ET.parse(path).getroot()
        did = node.attrib["id"]
        pmid = path.stem if path.parent.name == "MedLine" and path.stem.isdigit() else None
        documents.append({"ddi_document": did, "ddi_pmid": pmid, "ddi_role": roles.get(did, "reserve"),
                          "ddi_source": path.parent.name})
        for item in node.findall("sentence"):
            text = _normalized(item.attrib["text"])
            if len(text) >= 40 and len(text.split()) >= 6:
                sentences.append({"ddi_document": did, "ddi_sentence": item.attrib["id"],
                                  "normalized_text": text, "ddi_role": roles.get(did, "reserve")})
    return pd.DataFrame(documents), pd.DataFrame(sentences)


def external_overlap_audit(root: Path) -> dict[str, pd.DataFrame]:
    """Check official CDR splits and DDI overlap; report rather than alter membership."""
    frames = {split: load_bc5cdr_split(root, split)[0] for split in SPLIT_FILES}
    checks, reasons = [], []
    for left, right in combinations(frames, 2):
        ids = set(frames[left].pmid) & set(frames[right].pmid)
        texts = set(frames[left].text.map(_normalized)) & set(frames[right].text.map(_normalized))
        checks.extend([_check("Cross-split PMID overlap", len(ids), 0, not ids, f"{left}/{right}"),
            _check("Cross-split normalized document overlap", len(texts), 0, not texts, f"{left}/{right}")])
    ddi_documents, ddi_sentences = _ddi_inventory(root)
    for split, frame in frames.items():
        for row in frame.itertuples():
            for match in ddi_documents.loc[ddi_documents.ddi_pmid.eq(row.pmid)].itertuples():
                reasons.append({"External split": split, "PMID": row.pmid, "Rule": "PMID identity",
                    "DDI document": match.ddi_document, "DDI role": match.ddi_role, "DDI sentence": ""})
            text = _normalized(row.text)
            for match in ddi_sentences.itertuples():
                if match.normalized_text in text:
                    reasons.append({"External split": split, "PMID": row.pmid,
                        "Rule": "DDI sentence containment (NFKC/casefold; >=40 chars, >=6 tokens)",
                        "DDI document": match.ddi_document, "DDI role": match.ddi_role,
                        "DDI sentence": match.ddi_sentence})
        matching = {r["PMID"] for r in reasons if r["External split"] == split}
        checks.append(_check("CDR documents matching DDI identity/text", len(matching), 0, not matching, split))
    columns = ["External split", "PMID", "Rule", "DDI document", "DDI role", "DDI sentence"]
    coverage = pd.DataFrame([{"DDI documents checked": len(ddi_documents),
        "DDI PMID available": int(ddi_documents.ddi_pmid.notna().sum()),
        "DDI PMID unavailable": int(ddi_documents.ddi_pmid.isna().sum()),
        "Text rule": "NFKC + casefold + whitespace; sentence containment >=40 chars and >=6 tokens",
        "Limitation": "No claim of fuzzy-semantic or pretraining-corpus independence"}])
    return {"overlap_integrity": pd.DataFrame(checks),
            "overlap_reasons": pd.DataFrame(reasons, columns=columns), "overlap_coverage": coverage}


def corpus_selection_evidence() -> pd.DataFrame:
    """Primary-source selection metadata, not external model results."""
    return pd.DataFrame([
        {"Corpus": "BC5CDR", "Annotation scope": "Chemical / Disease; 500/500/500 documents",
         "Availability / rights": "Pinned public mirror; archive NCBI public-domain notice",
         "Decision": "Selected: manageable two-type architecture-transfer benchmark",
         "Source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4860626/"},
        {"Corpus": "CHEMDNER", "Annotation scope": "Seven chemical mention-form classes; 10000 abstracts",
         "Availability / rights": "Official corpus endpoint returned HTTP 403; no downloaded archive license verified",
         "Decision": "Considered; chemical-form classes do not equal DDI drug subtypes",
         "Source": "https://pubmed.ncbi.nlm.nih.gov/25810773/"},
        {"Corpus": "GutBrainIE 2025", "Annotation scope": "13 entity types; multiple annotation-quality tiers",
         "Availability / rights": "Public Zenodo; API CC-BY-4.0 vs rendered MIT metadata need reconciliation",
         "Decision": "Considered; broader schema/tier protocol would require a separate adaptation",
         "Source": "https://zenodo.org/records/16845409"}])
