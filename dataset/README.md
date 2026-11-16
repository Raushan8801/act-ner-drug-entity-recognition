# Dataset: DDIExtraction 2013

This folder holds everything the notebooks read: the unchanged corpus files, the processed tables and the
document-level splits.

## Source and licence

- **Corpus.** DDIExtraction 2013 (SemEval-2013 Task 9).
- **Corpus description.** Herrero-Zazo, M., Segura-Bedmar, I., Martínez, P., & Declerck, T. (2013). The DDI corpus:
  An annotated corpus with pharmacological substances and drug–drug interactions. *Journal of Biomedical
  Informatics*, 46(5), 914–920. <https://doi.org/10.1016/j.jbi.2013.07.011>
- **Shared task.** Segura-Bedmar, I., Martínez, P., & Herrero-Zazo, M. (2013). SemEval-2013 Task 9: Extraction of
  drug-drug interactions from biomedical texts. <https://aclanthology.org/S13-2056/>
- **Distribution.** Author-maintained repository <https://github.com/isegura/DDICorpus>, commit
  `819b0fa538ed83ab2a8b4f5417bba3c03249b7f3`, archive `DDICorpus-2013.zip` with SHA-256
  `87fc63e32ebce20d9841c372e0c594ec72d60882ab5621550ca73c3eaf80ab60` (see `Raw/ACQUISITION.json`).
- **Licence.** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0), as stated by the author
  repository. The files in this folder, including the derived tables, are provided for non-commercial research under
  the same licence. Please cite the corpus description when you use them.

## Layout

```text
dataset/
├── Raw/
│   ├── ACQUISITION.json            source URL, commit and archive checksum
│   └── DDICorpus-2013/DDICorpus/   byte-identical extraction of the archive (XML)
│       ├── Train/DrugBank          572 documents
│       ├── Train/MedLine           142 documents
│       ├── Test/Test for DrugNER task/{DrugBank,MedLine}          54 + 58 documents
│       └── Test/Test for DDI Extraction task/{DrugBank,MedLine}   158 + 33 documents
├── Processed/                      sentence, token, entity, alignment and feature tables (Parquet)
└── Splits/                         document-level manifests of Train, Validation and the test sets
```

## Annotation scheme

Entities are annotated with sentence-relative character offsets and one of four types:

| Type | Meaning |
| :--- | :--- |
| `drug` | generic name of a drug approved for human use |
| `brand` | trade or brand name |
| `group` | pharmacological or chemical group |
| `drug_n` | active substance not approved for human use |

A small number of entities are discontinuous. The experiments address contiguous exact-span recognition:
discontinuous mentions are counted, excluded from model targets and strict metrics, and never rewritten.

## Splits

- The 714 original training documents are divided once, at document level, into Train (572) and Validation (142)
  with a fixed seed and stratification by source and entity type. A document never crosses splits.
- The 112 documents of the DrugNER test set form the official test set (`Splits/locked_test_documents.csv`).
- The 191 test documents of the relation-extraction task are used as a reserve; 176 of them share no text with the
  development data.

No XML file is corrected, normalised or overwritten. Folders created at run time (`Interim/`, `External/`) are
ignored by Git; the external BC5CDR corpus is downloaded by the code and is not redistributed here.
