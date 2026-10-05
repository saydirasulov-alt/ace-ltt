"""Shared I/O: manifest reading, frame lists, resumable CSV output, Pyro-SDIS event ids."""
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

EVAL_SPLITS = ("val", "test", "external")
_PYRO = re.compile(r"^(?P<cam>.+)_(?P<t>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")


def read_manifest(dataset, splits=EVAL_SPLITS, sources=None, limit=0):
    """Rows of <dataset>/manifest.csv for the given splits, with absolute image paths in 'path'."""
    dataset = Path(dataset).expanduser()
    rows = []
    with open(dataset / "manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] not in splits or (sources and r["source"] not in sources):
                continue
            r["path"] = str(dataset / r["image"])
            rows.append(r)
    rows.sort(key=lambda r: (r["split"], r["image"]))
    if limit:
        rows = spread(rows, limit)
    return rows


def spread(rows, k):
    """k rows evenly spread over the list (keeps every split/source in a dry run)."""
    if k >= len(rows):
        return rows
    step = len(rows) / k
    return [rows[int(i * step)] for i in range(k)]


def frame_label(r):
    """(y, y_smoke, y_fire). External images only have an image-level class."""
    if r["split"] == "external":
        y = int(r.get("image_class", "") == "fire")
        return y, -1, -1
    ns, nf = int(r["n_smoke"] or 0), int(r["n_fire"] or 0)
    return int(ns + nf > 0), int(ns > 0), int(nf > 0)


def external_tag(r):
    """external/<tag>/<class>/file.jpg -> tag (fog, hazy, clear, clear2)."""
    p = Path(r["image"]).parts
    return p[1] if len(p) >= 4 and p[0] == "external" else (r.get("tags") or "")


def pyro_camera_time(stem):
    """'force-06_cabanelle-125_2024-01-04T08-34-18' -> ('force-06_cabanelle-125', datetime) or None."""
    m = _PYRO.match(stem)
    if not m:
        return None
    try:
        return m["cam"], datetime.strptime(m["t"], "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


def assign_events(rows, gap_min=30.0):
    """Event id per row: Pyro frames of one camera closer than gap_min minutes form one event;
    every other frame is its own event. Returns dict uid -> event id."""
    ev = {}
    cams = {}
    for r in rows:
        stem = Path(r["image"]).stem.split("__", 1)[-1]
        ct = pyro_camera_time(stem) if r["source"].startswith("pyro") else None
        if ct is None:
            ev[r["uid"]] = "img:" + r["uid"]
        else:
            cams.setdefault((r["split"], ct[0]), []).append((ct[1], r["uid"]))
    for (split, cam), lst in cams.items():
        lst.sort()
        k, prev = 0, None
        for t, uid in lst:
            if prev is not None and (t - prev).total_seconds() > gap_min * 60:
                k += 1
            ev[uid] = f"pyro:{split}:{cam}:{k}"
            prev = t
    return ev


class ResumableCSV:
    """Append rows keyed by 'uid'; rows already present are skipped on restart."""

    def __init__(self, path, fields):
        self.path = Path(path).expanduser()
        self.fields = list(fields)
        self.done = set()
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                if rd.fieldnames and rd.fieldnames != self.fields:
                    raise SystemExit(f"{self.path}: columns differ from this script version; use another --out")
                self.done = {r["uid"] for r in rd}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self.f = open(self.path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, self.fields)
        if new:
            self.w.writeheader()

    def write(self, rows):
        for r in rows:
            self.w.writerow({k: r.get(k, "") for k in self.fields})
        self.f.flush()

    def close(self):
        self.f.close()


def read_csv_by_uid(path):
    with open(Path(path).expanduser(), newline="", encoding="utf-8") as f:
        return {r["uid"]: r for r in csv.DictReader(f)}
