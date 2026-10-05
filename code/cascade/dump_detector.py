"""Step 1 (server GPU): run the edge detector once over val / test / external and store per-frame scores.

    python -m cascade.dump_detector --weights ~/PROJECT/runs/fsc_v02/yolo26-base_s0/weights/best.pt \
        --dataset ~/PROJECT/lha-yolo26/datasets/FireSmoke-Clean_v0.2 \
        --out ~/PROJECT/runs/cascade/det_yolo26-base_s0.csv --device 0

Per frame: s = max box confidence (smoke or fire), s_smoke, s_fire, the top boxes (for the agent's
overlay), the uplink size of the frame (JPEG q85, long side 640) and of a crop around the top box,
and detector latency. Resumable: frames already in --out are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.dataio import EVAL_SPLITS, ResumableCSV, read_manifest  # noqa: E402

FIELDS = ["uid", "s", "s_smoke", "s_fire", "n_boxes_25", "boxes", "img_w", "img_h",
          "nbytes_frame", "nbytes_crop", "det_ms"]


def sha256(p):
    h = hashlib.sha256()
    with open(Path(p).expanduser(), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jpeg_bytes(img, long_side, quality=85):
    h, w = img.shape[:2]
    k = long_side / max(h, w)
    if k < 1:
        img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return len(buf) if ok else 0


def crop_box(img, xyxyn, pad=0.25, min_px=64):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxyn[0] * w, xyxyn[1] * h, xyxyn[2] * w, xyxyn[3] * h
    bw, bh = max(x2 - x1, min_px), max(y2 - y1, min_px)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    a, b = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
    c, d = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
    return img[b:d, a:c]


def result_row(res, top_k=3):
    b = res.boxes
    conf = b.conf.float().cpu().numpy() if len(b) else np.zeros(0)
    cls = b.cls.int().cpu().numpy() if len(b) else np.zeros(0, int)
    xyxyn = b.xyxyn.float().cpu().numpy() if len(b) else np.zeros((0, 4))
    s_smoke = float(conf[cls == 0].max()) if (cls == 0).any() else 0.0
    s_fire = float(conf[cls == 1].max()) if (cls == 1).any() else 0.0
    order = np.argsort(-conf)[:top_k]
    boxes = ";".join(f"{cls[i]}:{conf[i]:.4f}:" + ":".join(f"{v:.4f}" for v in xyxyn[i]) for i in order)
    img = res.orig_img
    h, w = img.shape[:2]
    nb_crop = jpeg_bytes(crop_box(img, xyxyn[order[0]]), 448) if len(order) else 0
    sp = res.speed or {}
    return {
        "uid": None, "s": f"{max(s_smoke, s_fire):.5f}", "s_smoke": f"{s_smoke:.5f}", "s_fire": f"{s_fire:.5f}",
        "n_boxes_25": int((conf >= 0.25).sum()), "boxes": boxes, "img_w": w, "img_h": h,
        "nbytes_frame": jpeg_bytes(img, 640), "nbytes_crop": nb_crop,
        "det_ms": f"{sum(sp.get(k, 0.0) for k in ('preprocess', 'inference', 'postprocess')):.2f}",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=list(EVAL_SPLITS))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true", help="fp16 inference")
    ap.add_argument("--conf", type=float, default=0.001, help="keep low: the cascade needs the full score range")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU (ignored by the NMS-free head)")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--nms", choices=["none", "true", "false"], default="none",
                    help="Ultralytics >= 8.4.1xx: none/true -> one-to-many head + NMS; false -> YOLO26 one-to-one "
                         "NMS-free head. Must match the deployment path in protocol.yaml")
    ap.add_argument("--limit", type=int, default=0, help="only N frames (dry run)")
    a = ap.parse_args()

    import lha26  # noqa: F401  (registers LHA modules so LHA checkpoints load too)
    from ultralytics import YOLO

    rows = read_manifest(a.dataset, a.splits, limit=a.limit)
    import torch
    import ultralytics
    meta = {"weights": str(Path(a.weights).expanduser().resolve()), "weights_sha256": sha256(a.weights),
            "ultralytics": ultralytics.__version__, "torch": torch.__version__, "python": platform.python_version(),
            "nms": a.nms, "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou, "max_det": a.max_det,
            "precision": "fp16" if a.half else "fp32", "dataset": str(Path(a.dataset).expanduser().resolve()),
            "manifest_sha256": sha256(Path(a.dataset).expanduser() / "manifest.csv"), "splits": a.splits,
            "limit": a.limit}
    meta_path = Path(a.out).expanduser().with_suffix(".meta.json")
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{a.out} was produced with a different configuration: {diff}. Use another --out.")
    out = ResumableCSV(a.out, FIELDS)
    todo = [r for r in rows if r["uid"] not in out.done]
    print(f"{len(rows)} frames, {len(rows) - len(todo)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    model = YOLO(a.weights)
    runtime_e2e = None
    t0, n = time.time(), 0
    chunk = a.batch * 25
    try:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            buf = []
            lst = Path(a.out).expanduser().with_suffix(".chunk.txt")
            lst.write_text("\n".join(r["path"] for r in part) + "\n")   # a .txt source keeps real file paths
            kw = dict(stream=True, batch=a.batch, imgsz=a.imgsz, conf=a.conf, iou=a.iou, device=a.device,
                      verbose=False, max_det=a.max_det, **({"half": True} if a.half else {}),
                      **({} if a.nms == "none" else {"nms": a.nms == "true"}))
            for r, res in zip(part, model.predict(str(lst), **kw)):
                assert Path(res.path).name == Path(r["path"]).name, (res.path, r["path"])
                row = result_row(res)
                row["uid"] = r["uid"]
                buf.append(row)
            if runtime_e2e is None:
                runtime_e2e = bool(getattr(model.predictor.model, "end2end", False))
                meta["runtime_end2end"] = runtime_e2e
                meta["head"] = "one-to-one (NMS-free)" if runtime_e2e else "one-to-many + NMS"
                meta_path.write_text(json.dumps(meta, indent=1))
                print("detector head:", meta["head"], flush=True)
            if len(buf) != len(part):
                raise SystemExit(f"detector returned {len(buf)} results for {len(part)} frames (unreadable image?)")
            out.write(buf)
            lst.unlink()
            n += len(part)
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {n / el:.1f} img/s  ETA {(len(todo) - n) / max(n / el, 1e-9) / 60:.1f} min",
                  flush=True)
    finally:
        out.close()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
