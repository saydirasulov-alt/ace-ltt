# ACE-LTT reproducibility package (v1)

Frozen code, registrations, run records and archived model outputs for the manuscript

> **Fail-Closed Certification of Multi-Stage AI Policies**
> Norkobil Saydirasulov, Young-Im Cho, D. A. Davronbekov, B. A. Nazarov, Abdinabi Mukhamadiyev

ACE-LTT = Admissible Cascade Envelope via Learn-then-Test. Section numbers M*n* refer to the manuscript's
Supplementary Information. The manuscript is under peer review.

Archived release: Zenodo DOI to be added on the first release. Please cite the archived release (see
`CITATION.cff`).

## Contents

| folder | what it holds |
|---|---|
| `code/cascade/` | the frozen `cascade` package used for every reported result, with its own `MANIFEST.sha256` (`sha256sum -c MANIFEST.sha256` inside the folder). `ace_ltt.py` is the certification engine. `README_UZ.md` is the developers' internal note (Uzbek), kept because the manifest covers it. |
| `code/*.sh` | the Study 3 preparation, registration and one-shot execution scripts |
| `preregistrations/` | Monte Carlo preregistration R0; Study 3 preregistration R0 as frozen, and the same file with its dated post-run addendum of 28 September 2026 (the frozen text is a byte-prefix of the addended file) |
| `montecarlo/` | generator parameters, the harness and the complete run `run_R0/` (blocks MC-A to MC-D, summary, pilot record, MC-D witness) |
| `protocol_v249/` | the protocol instance used by Study 3: `protocol.yaml`, `splits.csv` (unit roles; no labels), `protocol.lock` |
| `scores/` | archived detector and verifier outputs (see "What is withheld"), their run metadata and record metadata |
| `study1/`, `study2/` | registrations and development-split run records of the two exploratory studies |
| `provenance/` | the provenance record of Studies 1 and 2 and of design audit 2A |
| `study3/` | registration receipt R1, audit receipt, run report and the logs of the v249 pipeline (self-tests, preflight, dry run, registration, execution) |
| `toy/` | runs of the dataset-free synthetic instance `cascade.ace_toy` |

## Digests that match the manuscript

| object | SHA-256 |
|---|---|
| engine `code/cascade/ace_ltt.py` | `ca93d2f3e4eafa85255f512318c0e7a350d259640518696ddbfed23e93960241` |
| `protocol_v249/protocol.yaml` | `f585e490d9bc4aa047e8c91fc3e7f3d482560b7d8e21fedfa78f152e9cc02fa4` |
| `protocol_v249/splits.csv` | `362d8689b193d61b3f97ff08ea1acd1a0956824cbbea3544f96921940994d27f` |
| `protocol_v249/protocol.lock` (file) | `01707a2314ce78f5d96cc45bf9f8a956355ce56692ba60316073c4de7f5e164c` |
| `analysis_code_sha256` (pinned in R1) | `aa461f6b8f1f2b556544e78680565ccc5cd4fbad56dbd8fe85185dd1069464b7` |
| registration receipt R1 `study3/ace3_v249.registration.json` | `d9bcf680f1374d8124fe7549147538af3722683feda6e72f8821dee8da0c6b42` |
| audit receipt `study3/ace3_audit_receipt.json` | `e52ade7dd6ebe5e2c07de3c700e77b1b5bb940706e88e91a5d79cfa4463e1182` |
| run report `study3/ace3_dev.json` | `dad2494631c6ad5b95f75b3a3acec4ca11bf306fda9994ee243cdbe3e32e697d` |
| Monte Carlo preregistration R0 | `a8e0b6fca234f043d7ead1e4a6f7d00b605a455888c1101cf504274c6271cedd` |
| Monte Carlo harness | `3c091eb22370232a50d85353dc06989685c6d9370edc719889372c530f0399df` |

Every file in the package is listed in `SHA256SUMS` (check with `sha256sum -c SHA256SUMS`).

## Models and data (not redistributed)

- Detector: YOLO26n trained by the authors on the FireSmoke-Clean v0.2 training split, Ultralytics 8.4.153,
  weights SHA-256 `b4dd9d7edf515401524af23d40c392bfbc600bf005c1188c5453909e4d640230` (see
  `scores/det_*.meta.json`). Weights are not included.
- Verifier: `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, zero-shot;
  prompts and their SHA-256 are in `scores/agent_*.meta.json`.
- FireSmoke-Clean v0.2 is a deduplicated union of D-Fire, FASDD_CV, Pyro-SDIS v1.0 (Hugging Face tag v1.0,
  commit dd86e151) and FLAME2. Images and labels remain under their source licences and are not included;
  the dataset manifest SHA-256 is `a05e06d5accdb68ca6db4e4d78feac3f760d3150eda3e6e97122e3ddfee5b6de`.

## What is withheld, and why

- **Sealed test set.** The sealed test set was never opened in this work. Rows of sealed-test units are removed
  from the three score files (`*.no_sealed.csv`: 12,196 detector rows and 7,361 rows per verifier view
  removed). The SHA-256 of the complete locked files is in `scores/FULL_FILES.sha256`; the complete files,
  needed to re-verify `protocol.lock`, are available from the corresponding author on request.
- **Labels and images.** No ground-truth labels or images are included (see above).
- Absolute paths inside the logs and metadata refer to the authors' machine and are kept unedited, since the
  files are digest-bound.

## Software

Python 3.12, PyTorch 2.11, Ultralytics 8.4.153 (detector run only), Transformers 5.17 (verifier run only).
The certification engine, `cascade.ace_toy` and the Monte Carlo harness need NumPy, SciPy and PyYAML.

## Licence

Code (`code/`, `montecarlo/*.py`): Apache License 2.0 (`LICENSE`).
All other files (registrations, records, logs, scores, preregistrations, documentation): Creative Commons
Attribution 4.0 International (`LICENSE-DATA-CC-BY-4.0.txt`).

## Contact

Norkobil Saydirasulov, AI and Smart City Laboratory, Department of Computer Engineering, Gachon University,
Seongnam-si, Republic of Korea — saydirasulov@gachon.ac.kr
