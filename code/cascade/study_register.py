"""Write a one-shot pre-registration for a dev study (closed variant list + decision rule + input hashes).

    python -m cascade.study_register --name acquisition_ladder --protocol P --det D --out REG.json \
        [--inputs name=path ...] [--note "..."]

The rule text lives in the study's own module docstring; this file records WHEN it was fixed and against WHICH
code and inputs. It refuses to overwrite an existing registration.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

RULES = {
    "acquisition_ladder": (
        "Closed list of 5 acquisition variants scored on dev frames only (3B, q85 round-trip): "
        "A crop; B overlay; C crop zoom x2; D crop context pad x3; E temporal pair (previous frame beside the "
        "crop, time-stamped sources only). Reference = A (crop). "
        "GO for a variant iff, versus the reference, it raises AUC by >= 0.05 or lowers FPR@TPR95 by >= 0.10 "
        "inside the small-smoke stratum S, AND the improvement holds in EVERY source separately "
        "(dfire, fasdd_cv, pyro_sdis), AND the 90% interval of the difference from a UNIT-level (cluster) "
        "bootstrap excludes 0. Frame-level standard errors are not used: in S, 2570 of 2951 positive frames "
        "come from 2 Pyro cameras. A Pyro-only improvement is descriptive and never generalized. "
        "If no variant passes, the acquisition axis is closed and the negative result is reported as such. "
        "The list is closed at registration; a later variant starts a new registration."),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, choices=sorted(RULES))
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="*", default=[], help="name=path pairs hashed into the registration")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    from cascade.make_protocol import analysis_code, sha256_file
    out = Path(a.out).expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists: a registration is written once")
    rec = {"study": a.name, "registered_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "rule": RULES[a.name], "note": a.note,
           "protocol_sha256": sha256_file(Path(a.protocol) / "protocol.yaml"),
           "splits_sha256": sha256_file(Path(a.protocol) / "splits.csv"),
           "analysis_code_sha256": analysis_code()[0],
           "det_sha256": sha256_file(a.det),
           "inputs": {k: sha256_file(v) for k, v in (x.split("=", 1) for x in a.inputs)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(out)
    print(json.dumps({k: rec[k] for k in ("study", "registered_utc", "protocol_sha256", "analysis_code_sha256")},
                     indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
