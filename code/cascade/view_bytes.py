"""Uplink bytes of exactly the image the VLM saw (label-free).

    python -m cascade.view_bytes --dataset <ds> --det <det.csv> --agent <agent_crop.csv> --out <viewbytes.csv>

For every frame in the agent CSV, the view image is rebuilt with agent_vlm.prepare_image (same boxes from the
detector CSV, same view and max_side as recorded in the agent meta) and JPEG-encoded at quality 85, the codec
used for nbytes_frame in dump_detector. nbytes_view is the uplink size if that view is produced on the edge.

Why: dump_detector's nbytes_crop is a 448-px crop, while agent_vlm (view=crop) sends up to max_side=640 px, so
the crop payload of the scored view was under-counted. The overlay view is drawn in the cloud from the box
coordinates, so its payload stays nbytes_frame (full frame, long side 640) and needs no recomputation.
No label is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

JPEG_QUALITY = 85


def jpeg_nbytes(im, quality=JPEG_QUALITY):
    import cv2
    a = np.asarray(im.convert("RGB"))[:, :, ::-1]
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(a), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return len(buf)


def _one(args):
    from cascade.agent_vlm import parse_boxes, prepare_image
    uid, path, boxes, view, max_side = args
    im = prepare_image(path, parse_boxes(boxes), view, max_side)
    return uid, jpeg_nbytes(im), im.width, im.height


def _sha(p):
    h = hashlib.sha256()
    with open(Path(p).expanduser(), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--agent", required=True, help="agent CSV whose view is re-encoded (normally view=crop)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    from cascade.dataio import read_csv_by_uid, read_manifest
    am = json.loads(Path(a.agent).expanduser().with_suffix(".meta.json").read_text())
    view, max_side = am["view"], int(am["max_side"])
    det_sha = _sha(a.det)
    if am.get("det_csv_sha256") and am["det_csv_sha256"] != det_sha:
        raise SystemExit("agent meta det_csv_sha256 != sha256(--det): the boxes would not be the ones the VLM saw")
    det, ag = read_csv_by_uid(a.det), read_csv_by_uid(a.agent)
    paths = {r["uid"]: r["path"] for r in read_manifest(a.dataset, ("train", "val", "test", "external"))}
    missing = [u for u in ag if u not in paths or u not in det]
    if missing:
        raise SystemExit(f"fail-closed: {len(missing)} agent uids without image/detector row, e.g. {missing[:3]}")
    jobs = [(u, paths[u], det[u].get("boxes", ""), view, max_side) for u in sorted(ag)]
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "nbytes_view", "view_w", "view_h"])
        with ProcessPoolExecutor(max(1, a.workers)) as ex:
            for i, row in enumerate(ex.map(_one, jobs, chunksize=64)):
                w.writerow(row)
                if (i + 1) % 5000 == 0:
                    print(f"  {i + 1}/{len(jobs)}", flush=True)
    tmp.replace(out)
    meta = {"view": view, "max_side": max_side, "jpeg_quality": JPEG_QUALITY, "codec": "cv2.imencode .jpg",
            "image_builder": "cascade.agent_vlm.prepare_image", "n_rows": len(jobs),
            "det_csv_sha256": det_sha, "agent_csv_sha256": _sha(a.agent),
            "agent_meta_sha256": _sha(Path(a.agent).expanduser().with_suffix(".meta.json")),
            "csv_sha256": _sha(out)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    nb = np.array([float(r[1]) for r in csv.reader(open(out)) if r[0] != "uid"])
    print(f"{len(nb)} frames, view={view}, max_side={max_side}: nbytes_view mean {nb.mean():.0f}, "
          f"median {np.median(nb):.0f}, p95 {np.quantile(nb, .95):.0f}")
    print("->", out)


if __name__ == "__main__":
    sys.exit(main())
