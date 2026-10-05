"""Dev-only comparison and pre-registered selection of the detector head (or any detector variant).

    python -m cascade.dump_detector ... --splits val --nms false --out det_dev_nms-false.csv
    python -m cascade.dump_detector ... --splits val --nms none  --out det_dev_nms-none.csv
    python -m cascade.head_report --dataset <ds> --protocol <P> --det det_dev_nms-false.csv det_dev_nms-none.csv \
        [--weights best.pt --device 0]        # adds dev mAP50 with the deployment arguments of each meta

Metrics (dev = val role only):
    candidate recall at s_min: share of fire / smoke events (and frames) with at least one frame s >= s_min
                               (a detector miss below s_min can never be recovered by the VLM)
    negative escalation      : share of negative frames with s >= s_min (cloud load)
    mAP50                    : YOLO.val(split="val") with the same imgsz / conf / iou / max_det / nms as the meta
    det_ms                   : mean detector time per frame from the dump

Selection rule (protocol.yaml `head_rule`, deterministic):
    eligible = heads whose fire AND smoke event recall are each within RECALL_TOL of the best head on that class;
    among eligible: lowest negative escalation; heads within ESC_TOL of that minimum -> higher mAP50;
    within MAP_TOL -> lower det_ms; remaining tie -> nms "false". No eligible head -> max min(fire, smoke) recall.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.dataio import frame_label, read_csv_by_uid, read_manifest  # noqa: E402

RECALL_TOL, ESC_TOL, MAP_TOL = 0.010, 0.010, 0.005


def select_head(H):
    """H: list of dicts with name, nms, fire, smoke, neg_esc, map50 (may be nan), det_ms. Returns (name, reason)."""
    bf, bs = max(h["fire"] for h in H), max(h["smoke"] for h in H)
    elig = [h for h in H if h["fire"] >= bf - RECALL_TOL and h["smoke"] >= bs - RECALL_TOL]
    if not elig:
        h = max(H, key=lambda h: min(h["fire"], h["smoke"]))
        return h["name"], "no head within tolerance on both classes -> max min(fire, smoke) recall"
    m_esc = min(h["neg_esc"] for h in elig)
    c = [h for h in elig if h["neg_esc"] <= m_esc + ESC_TOL]
    if len(c) > 1 and all(np.isfinite(h["map50"]) for h in c):
        m_map = max(h["map50"] for h in c)
        c = [h for h in c if h["map50"] >= m_map - MAP_TOL]
    if len(c) > 1:
        m_ms = min(h["det_ms"] for h in c)
        c = sorted(c, key=lambda h: (h["det_ms"] > m_ms + 1e-9, str(h["nms"]) != "false"))
    return c[0]["name"], f"eligible={[h['name'] for h in elig]}"


def dev_map50(weights, meta, data_yaml, device):
    from ultralytics import YOLO
    kw = dict(data=data_yaml, split="val", imgsz=int(meta["imgsz"]), conf=float(meta["conf"]), iou=float(meta["iou"]),
              max_det=int(meta["max_det"]), batch=16, device=device, plots=False, verbose=False)
    if meta.get("nms", "none") != "none":
        kw["nms"] = meta["nms"] == "true"
    return float(YOLO(weights).val(**kw).box.map50)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--det", nargs="+", required=True)
    ap.add_argument("--weights", default=None, help="compute dev mAP50 per head (GPU)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--s-min", type=float, default=None, help="default: protocol s_min")
    ap.add_argument("--json", default=None, help="also write the table + selection as JSON (for figures dev)")
    a = ap.parse_args()
    from cascade.make_protocol import load_protocol
    P, splits = load_protocol(a.protocol)
    s_min = a.s_min if a.s_min is not None else float(P["s_min"])
    rows = [r for r in read_manifest(a.dataset, ("val",)) if splits.get(r["uid"], {}).get("role") == "dev"]
    print(f"dev frames: {len(rows)}  s_min = {s_min}\n")
    H = []
    for path in a.det:
        det = read_csv_by_uid(path)
        meta = json.loads(Path(path).expanduser().with_suffix(".meta.json").read_text())
        miss = [r["uid"] for r in rows if r["uid"] not in det]
        if miss:
            raise SystemExit(f"{path}: {len(miss)} dev frames without score")
        ev_hit = {"fire": defaultdict(bool), "smoke": defaultdict(bool)}
        fr = {"fire": [], "smoke": []}
        neg = []
        for r in rows:
            s = float(det[r["uid"]]["s"])
            _, ys, yf = frame_label(r)
            ev = splits[r["uid"]]["event"]
            for cls, yk in (("fire", yf), ("smoke", ys)):
                if yk == 1:
                    ev_hit[cls][ev] |= s >= s_min
                    fr[cls].append(s >= s_min)
            if ys == 0 and yf == 0:
                neg.append(s >= s_min)
        m50 = dev_map50(a.weights, meta, str(Path(a.dataset).expanduser() / "data.yaml"), a.device) \
            if a.weights else float("nan")
        H.append({"name": Path(path).name, "nms": meta.get("nms"), "head": meta.get("head", "?"),
                  "fire": float(np.mean(list(ev_hit["fire"].values()))),
                  "smoke": float(np.mean(list(ev_hit["smoke"].values()))),
                  "n_fire_ev": len(ev_hit["fire"]), "n_smoke_ev": len(ev_hit["smoke"]),
                  "fire_fr": float(np.mean(fr["fire"])), "smoke_fr": float(np.mean(fr["smoke"])),
                  "neg_esc": float(np.mean(neg)), "map50": m50,
                  "det_ms": float(np.mean([float(det[r["uid"]]["det_ms"]) for r in rows]))})
    print("| detector | head | fire event recall | smoke event recall | fire frame | smoke frame | "
          "neg escalation | dev mAP50 | det_ms |\n|---|---|---|---|---|---|---|---|---|")
    for h in H:
        print(f"| {h['name']} | {h['head']} | {h['fire']:.4f} (n={h['n_fire_ev']}) | {h['smoke']:.4f} "
              f"(n={h['n_smoke_ev']}) | {h['fire_fr']:.4f} | {h['smoke_fr']:.4f} | {h['neg_esc']:.4f} | "
              f"{h['map50']:.4f} | {h['det_ms']:.2f} |")
    name, why = select_head(H)
    sel = next(h for h in H if h["name"] == name)
    print(f"\nSELECTED by head_rule: {name}  (nms = {sel['nms']})  [{why}]")
    print(f'-> set  detector.nms: "{sel["nms"]}"  in protocol.yaml')
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(
            {"role": "dev", "s_min": s_min, "heads": H, "selected": name, "reason": why,
             "tolerances": {"recall": RECALL_TOL, "escalation": ESC_TOL, "map50": MAP_TOL}}, indent=1))


if __name__ == "__main__":
    main()
