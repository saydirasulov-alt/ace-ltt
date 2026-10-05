"""Step 3 (CPU): merge manifest + detector scores + agent scores into one records.csv (+ records.meta.json).

    python -m cascade.build_records --dataset <ds> --det <det.csv> --agent <agent.csv> --out <rec.csv> --protocol <P>
    python -m cascade.build_records ... --protocol <P> --unseal      # only after `make_protocol freeze`, ONCE
    python -m cascade.build_records ... --protocol <P> --unseal --resume   # after a crash, same --out (audited)

With --protocol (fail-closed):
  * roles, units and events are taken from the FROZEN splits.csv (not recomputed);
  * the manifest hash must equal protocol.yaml; the eval UIDs must equal the UIDs of splits.csv;
  * every frame needs a detector score and every frame with s >= s_min an agent score, else it stops;
  * once the protocol is frozen, the detector / agent CSVs and metas must be byte-identical to the locked ones;
  * label blinding: before freeze only DEV labels are written; after freeze (valid lock) dev + calibration;
    sealed_test and external_shift labels only with --unseal (valid lock required);
  * order with --unseal: manifest FILE hash + lock (incl. analysis code) checks -> protocol receipt created
    atomically (state unsealing) -> only then manifest.csv (the labels) is read, interpreted and written ->
    records_sha256 recorded (state unsealed). A second --unseal is refused.
Without --protocol (legacy): units = split_group, events recomputed, missing scores are warned about.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.dataio import (EVAL_SPLITS, assign_events, external_tag, frame_label, read_csv_by_uid,  # noqa: E402
                            read_manifest)

FIELDS = ["uid", "source", "split", "role", "unit", "event", "tag", "y", "y_smoke", "y_fire", "s", "s_smoke",
          "s_fire", "g", "g_plain", "nbytes_frame", "nbytes_crop", "det_ms", "agent_ms", "agent_ms_plain", "agent"]


def sha256_file(p):
    return hashlib.sha256(Path(p).expanduser().read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--agent", default=None, help="agent CSV (omit for edge-only records)")
    ap.add_argument("--agent-plain", default=None, help="plain-view agent on every frame (pure cloud baseline)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--event-gap", type=float, default=30.0, help="minutes (legacy mode only)")
    ap.add_argument("--s-min", type=float, default=0.02, help="legacy mode only; protocol mode uses protocol.yaml")
    ap.add_argument("--protocol", default=None, help="protocol dir from cascade.make_protocol")
    ap.add_argument("--unseal", action="store_true", help="include sealed_test labels (frozen protocol only)")
    ap.add_argument("--resume", action="store_true", help="with --unseal: resume a crashed unseal (same --out)")
    a = ap.parse_args()

    det = read_csv_by_uid(a.det)
    agent = read_csv_by_uid(a.agent) if a.agent else {}
    plain = read_csv_by_uid(a.agent_plain) if a.agent_plain else {}
    det_meta = Path(a.det).expanduser().with_suffix(".meta.json")
    ag_meta = Path(a.agent).expanduser().with_suffix(".meta.json") if a.agent else None
    splits, sealed_ok, lock, P = {}, False, None, None
    s_min = a.s_min
    if a.protocol:
        from cascade.make_protocol import (begin_unseal, check_lock, load_protocol, lock_status, read_lock,
                                           records_guard, sha256_file as sha)
        P, splits = load_protocol(a.protocol)
        s_min = float(P["s_min"])
        if sha(Path(a.dataset).expanduser() / "manifest.csv") != P["manifest_sha256"]:
            raise SystemExit("manifest hash differs from protocol.yaml")
        if not a.agent:
            raise SystemExit("protocol mode needs --agent")
        if P.get("pure_cloud_baseline") and not a.agent_plain:
            raise SystemExit("protocol has pure_cloud_baseline: true -> --agent-plain is required")
        if read_lock(a.protocol) is not None and not check_lock(a.protocol):
            raise SystemExit("protocol.lock exists but no longer matches (something changed after freeze):\n  "
                             + "\n  ".join(lock_status(a.protocol)))
        if check_lock(a.protocol):
            lock = read_lock(a.protocol)
            same = (sha256_file(det_meta) == lock["det_meta_sha256"] and sha256_file(ag_meta) == lock["agent_meta_sha256"]
                    and sha256_file(a.det) == lock["det_csv_sha256"] and sha256_file(a.agent) == lock["agent_csv_sha256"]
                    and (not lock.get("agent_plain_csv") or
                         (a.agent_plain and sha256_file(a.agent_plain) == lock["agent_plain_csv_sha256"])))
            if not same:
                raise SystemExit("protocol is frozen, but these detector / agent files (contents) are not the locked ones")
        if a.unseal:
            if lock is None:
                raise SystemExit("refusing --unseal: no valid lock (not frozen, or protocol/splits/score files "
                                 "/ analysis code changed)")
            begin_unseal(a.protocol, a.out, lock["sha256"], resume=a.resume)   # atomic, BEFORE labels are read
            sealed_ok = True
        else:
            records_guard(a.protocol, a.out)
    elif a.unseal:
        raise SystemExit("--unseal needs --protocol")
    if a.resume and not a.unseal:
        raise SystemExit("--resume is only for --unseal")
    # the manifest (which holds the labels) is read only now: with --unseal the receipt already exists
    rows = read_manifest(a.dataset, EVAL_SPLITS)
    if a.protocol and {r["uid"] for r in rows} != set(splits):
        raise SystemExit("eval UIDs of the dataset differ from splits.csv")
    events = None if a.protocol else assign_events(rows, a.event_gap)

    out, miss_det, miss_agent = [], [], []
    for r in rows:
        d = det.get(r["uid"])
        if d is None:
            miss_det.append(r["uid"])
            continue
        g = agent.get(r["uid"], {})
        if a.agent and float(d["s"]) >= s_min and not g:
            miss_agent.append(r["uid"])
        y, ys, yf = frame_label(r)
        sp = splits.get(r["uid"])
        role = sp["role"] if sp else ""
        # label blinding: before freeze only dev; after freeze dev + calibration; --unseal everything
        visible = (not a.protocol) or role == "dev" or (role == "calibration" and lock is not None) or sealed_ok
        if not visible:
            y = ys = yf = ""
        out.append({
            "uid": r["uid"], "source": r["source"], "split": r["split"], "role": role,
            "unit": sp["unit"] if sp else (f"g{r['split_group']}" if r["split_group"] != "" else "u:" + r["uid"]),
            "event": sp["event"] if sp else events[r["uid"]],
            "tag": external_tag(r) if r["split"] == "external" else (r.get("tags") or ""),
            "y": y, "y_smoke": ys, "y_fire": yf, "s": d["s"], "s_smoke": d["s_smoke"], "s_fire": d["s_fire"],
            "g": g.get("g", ""), "g_plain": plain.get(r["uid"], {}).get("g", ""), "nbytes_frame": d["nbytes_frame"], "nbytes_crop": d["nbytes_crop"],
            "det_ms": d["det_ms"], "agent_ms": g.get("agent_ms", ""),
            "agent_ms_plain": plain.get(r["uid"], {}).get("agent_ms", ""), "agent": g.get("agent", ""),
        })
    miss_plain = [x["uid"] for x in out if a.agent_plain and x["g_plain"] == ""]
    if a.protocol and miss_plain:
        raise SystemExit(f"fail-closed: {len(miss_plain)} frames without plain-agent score")
    if a.protocol and (miss_det or miss_agent):
        raise SystemExit(f"fail-closed: {len(miss_det)} frames without detector score (e.g. {miss_det[:3]}), "
                         f"{len(miss_agent)} frames with s >= {s_min} without agent score (e.g. {miss_agent[:3]})")
    outp = Path(a.out).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_name(outp.name + ".partial")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(out)
    rec_sha = sha256_file(tmp)
    if sealed_ok:
        from cascade.make_protocol import read_receipt
        prev = read_receipt(a.protocol).get("records_sha256")
        if prev not in (None, rec_sha):
            raise SystemExit(f"resumed unseal does not reproduce the records in the receipt (kept as {tmp})")
    import os
    os.replace(tmp, outp)
    meta = {"records_sha256": rec_sha, "n_rows": len(out), "unsealed": sealed_ok,
            "protocol": str(Path(a.protocol).expanduser().resolve()) if a.protocol else None,
            "lock_sha256": lock["sha256"] if lock else None, "s_min": s_min,
            "det_csv": str(Path(a.det).expanduser().resolve()),
            "det_meta_sha256": sha256_file(det_meta) if det_meta.exists() else None,
            "det_csv_sha256": sha256_file(a.det),
            "agent_csv_sha256": sha256_file(a.agent) if a.agent else None,
            "agent_plain_csv_sha256": sha256_file(a.agent_plain) if a.agent_plain else None,
            "agent_csv": str(Path(a.agent).expanduser().resolve()) if a.agent else None,
            "agent_meta_sha256": sha256_file(ag_meta) if ag_meta and ag_meta.exists() else None}
    outp.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    if sealed_ok:
        from cascade.make_protocol import end_unseal
        end_unseal(a.protocol, rec_sha)
        print(f"protocol receipt: unsealed, records_sha256 {rec_sha}")
    c = Counter((x["role"] or x["split"], x["source"]) for x in out)
    for k in sorted(c):
        print(f"  {k[0]:15s} {k[1]:16s} {c[k]:7d}")
    shown = sorted({x["role"] for x in out if x["y"] != ""}) if a.protocol else ["all"]
    print(f"records: {len(out)} | labels written for roles: {shown}")
    if miss_det:
        print(f"WARNING: {len(miss_det)} frames have no detector score")
    if miss_agent:
        print(f"WARNING: {len(miss_agent)} frames with s >= {s_min} have no agent score")
    print("->", outp)


if __name__ == "__main__":
    main()
