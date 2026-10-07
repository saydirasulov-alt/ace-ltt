# Licence scope

This file clarifies the licence boundaries of the ACE-LTT reproducibility package.

## Apache License 2.0

The repository-level Apache-2.0 `LICENSE` applies to author-created source code, including `code/` and
Python source files under `montecarlo/`, unless an individual file states otherwise.

## CC BY 4.0

`LICENSE-DATA-CC-BY-4.0.txt` applies to author-created non-code research artefacts where the authors hold
the necessary rights, including registrations, preregistrations, run records, documentation and derived
numerical outputs.

## Third-party and source-derived material

This repository does not relicense third-party datasets, model weights, source images/labels, or
third-party-derived metadata or identifiers. Such material remains subject to the original source terms.
FireSmoke-Clean v0.2 is assembled from external source datasets; source images and labels are not
redistributed here.

If a derived artefact contains material whose licensing status is uncertain, the repository-level CC BY
4.0 notice does not override the source terms. Add a file-specific notice or exclude the artefact from the
public release as appropriate.

## Zenodo metadata

This release contains mixed file-level licences. Zenodo's legacy `.zenodo.json` GitHub metadata format uses
a single licence field for the deposited files, so this repository does not ship an overriding
`.zenodo.json`. Zenodo deposition metadata should be reviewed at release time, with the mixed licence
boundaries preserved. `CITATION.cff` remains the canonical citation metadata in the repository.
