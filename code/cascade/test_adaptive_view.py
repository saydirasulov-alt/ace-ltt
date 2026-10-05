"""Fast, label-synthetic tests for RAVC-LTT.  No GPU or real labels are used."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .adaptive_view import RouterSpec, calibrate_adaptive_view, route_crop, routed_arrays, top_box_features
from .policy import metrics


def _synthetic(seed=9, n=2400):
    rng = np.random.default_rng(seed)
    unit = np.array([f"u{i}" for i in range(n)])
    y_fire = rng.random(n) < 0.22
    y_smoke = (~y_fire) & (rng.random(n) < 0.24)
    y = y_fire | y_smoke
    small = rng.random(n) < 0.55
    area = np.where(small, rng.uniform(0.002, 0.025, n), rng.uniform(0.08, 0.45, n))
    nbox = np.where(small, 1, rng.integers(1, 5, n))
    s = rng.uniform(0.24, 0.36, n)  # deliberately uninformative inside the ambiguous band
    sf = ss = s.copy()
    # Complementary views: each fixed view misses roughly half of the positives, while the
    # label-free scale router chooses the informative view.  Negatives stay low in both views.
    zo = -3.0 + 7.0 * (y & ~small) - 0.5 * (y & small) + rng.normal(0, 0.35, n)
    zc = -3.0 + 7.0 * (y & small) - 0.5 * (y & ~small) + rng.normal(0, 0.35, n)
    go, gc = 1 / (1 + np.exp(-zo)), 1 / (1 + np.exp(-zc))
    feat = {"top_area": area, "top_aspect": np.ones(n), "n_boxes_25": nbox, "s_fire": sf, "s_smoke": ss}
    return dict(s=s, go=go, gc=gc, y=y, yf=y_fire, ys=y_smoke, unit=unit, feat=feat,
                bf=np.full(n, 48000.0), bc=np.full(n, 8000.0))


def run_tests():
    f = top_box_features(["1:0.8:0.1:0.2:0.4:0.6", "bad"], [1, 0], [0.8, 0.0], [0.1, 0.0])
    assert np.isclose(f["top_area"][0], 0.12) and f["top_area"][1] == 1.0
    spec = RouterSpec("tiny", "small_isolated", area_max=0.15, max_boxes=1)
    assert route_crop(spec, f).tolist() == [True, False]
    r = routed_arrays(spec, f, [0.2, 0.3], [0.8, 0.9], [50, 50], [10, 10])
    assert np.allclose(r["g"], [0.8, 0.3]) and np.allclose(r["nbytes"], [10, 50])

    d = _synthetic()
    theta, info = calibrate_adaptive_view(
        d["s"], d["go"], d["gc"], d["feat"], d["y"], d["unit"],
        risk_labels={"fire": d["yf"], "smoke": d["ys"]}, nbytes_overlay=d["bf"], nbytes_crop=d["bc"],
        costs={"calls": 1.0, "fa": 8.0, "bytes": 1e-5}, seed=3,
    )
    assert info["certified"], info
    assert info["router"]["kind"] not in ("overlay", "crop"), info["router"]
    routed = routed_arrays(RouterSpec(**info["router"]), d["feat"], d["go"], d["gc"], d["bf"], d["bc"])
    rf = metrics(d["s"], routed["g"], d["yf"], theta, d["unit"])["miss_unit"]
    rs = metrics(d["s"], routed["g"], d["ys"], theta, d["unit"])["miss_unit"]
    assert max(rf, rs) < 0.08, (rf, rs, theta, info["router"])

    # Certification-fold score changes must not alter selection-fold router construction.
    rng = np.random.default_rng(3)
    uu = np.unique(d["unit"])
    sel = set(rng.choice(uu, int(round(0.30 * len(uu))), replace=False))
    cert = np.array([u not in sel for u in d["unit"]])
    gc2 = d["gc"].copy()
    gc2[cert] = 1 - gc2[cert]
    _, info2 = calibrate_adaptive_view(
        d["s"], d["go"], gc2, d["feat"], d["y"], d["unit"],
        risk_labels={"fire": d["yf"], "smoke": d["ys"]}, nbytes_overlay=d["bf"], nbytes_crop=d["bc"],
        costs={"calls": 1.0, "fa": 8.0, "bytes": 1e-5}, seed=3,
    )
    assert info["router_family"] == info2["router_family"]
    assert info["grid"] == info2["grid"]

    # handoff family: the human tier is available and never raises the certified risk definition
    theta_h, info_h = calibrate_adaptive_view(
        d["s"], d["go"], d["gc"], d["feat"], d["y"], d["unit"],
        risk_labels={"fire": d["yf"], "smoke": d["ys"]}, nbytes_overlay=d["bf"], nbytes_crop=d["bc"],
        costs={"calls": 1.0, "handoff": 5.0, "fa": 8.0, "bytes": 1e-5}, seed=3, handoff=True,
    )
    assert info_h["handoff"] is True and info_h["certified"], info_h

    # view_bytes: bytes of exactly the image agent_vlm.prepare_image builds (crop at max_side)
    from .agent_vlm import parse_boxes, prepare_image
    from .view_bytes import jpeg_nbytes
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        im = Image.fromarray((np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8))
        ip = Path(td) / "a.jpg"
        im.save(ip)
        bx = "1:0.9:0.40:0.40:0.70:0.90"
        v = prepare_image(str(ip), parse_boxes(bx), "crop", 640)
        assert max(v.size) <= 640 and v.size != im.size
        assert jpeg_nbytes(v) > 0

    # Dev command integration: frozen roles, two score contracts, crop bytes, registration, six methods
    from .ravc_dev import METHODS, acceptance, load_dev, run_trials, summarize, main as ravc_main
    from .selftest import make_fake
    import yaml
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ds = make_fake(root, n_units=160, seed=11)
        crop_csv = root / "agentcrop.csv"
        with open(root / "agent.csv", newline="") as f:
            ar = list(csv.DictReader(f))
        for row in ar:
            row["view"] = "crop"
        with open(crop_csv, "w", newline="") as f:
            w = csv.DictWriter(f, ar[0].keys())
            w.writeheader()
            w.writerows(ar)
        cm = json.loads((root / "agent.meta.json").read_text())
        cm["view"] = "crop"
        (root / "agentcrop.meta.json").write_text(json.dumps(cm))
        from .make_protocol import sha256_file
        for m in (root / "agent.meta.json", root / "agentcrop.meta.json"):   # fake metas: bind to det.csv
            mm = json.loads(m.read_text())
            mm["det_csv_sha256"] = sha256_file(root / "det.csv")
            m.write_text(json.dumps(mm))
        cb = root / "viewbytes.csv"
        with open(cb, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["uid", "nbytes_view", "view_w", "view_h"])
            for row in ar:
                w.writerow([row["uid"], 12000, 300, 200])
        cb.with_suffix(".meta.json").write_text(json.dumps({
            "view": "crop", "max_side": cm.get("max_side"), "agent_csv_sha256": sha256_file(crop_csv),
            "det_csv_sha256": sha256_file(root / "det.csv"), "csv_sha256": sha256_file(cb)}))
        proto = root / "proto"
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        p = subprocess.run(
            [sys.executable, "-m", "cascade.make_protocol", "init", "--dataset", str(ds), "--out", str(proto)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, env=env)
        assert p.returncode == 0, p.stdout + p.stderr
        args = ["--protocol", str(proto), "--det", str(root / "det.csv"), "--overlay", str(root / "agent.csv"),
                "--crop", str(crop_csv), "--crop-bytes", str(cb)]
        try:                                   # template has ravc: null -> refused (byte price not registered)
            load_dev(ds, proto, root / "det.csv", root / "agent.csv", crop_csv, cb)
            raise AssertionError("ravc: null must be refused")
        except SystemExit as e:
            assert "ravc" in str(e)
        pf = proto / "protocol.yaml"
        pf.write_text(pf.read_text().replace("ravc: null", "ravc: {byte_cost: 2.0e-5}"))
        dd, pp = load_dev(ds, proto, root / "det.csv", root / "agent.csv", crop_csv, cb)
        assert np.all(dd["bc"][dd["s"] >= 0.02] == 12000)
        dd["y_fire"][dd["source"] == "fasdd_cv"] = 0      # a source with no fire positives -> NaN, not a crash
        rr = run_trials(dd, pp, trials=2, seed=4, log=lambda m: None)
        assert {x["method"] for x in rr} == {m[0] for m in METHODS} and len(METHODS) == 6
        assert all(np.isnan(x["by_source"]["fasdd_cv"]["y_fire"]) for x in rr if "fasdd_cv" in x["by_source"])
        sm = summarize(rr)
        for fam in ("human", "no_human"):
            acceptance(sm, pp, fam)
        assert all(r["handoff"] == 0 for r in rr if r["family"] == "no_human")
        reg = root / "reg.json"
        ravc_main(["register", *args, "--out", str(reg)])
        try:
            ravc_main(["register", *args, "--out", str(reg)])
            raise AssertionError("second registration must be refused")
        except SystemExit:
            pass
        pf.write_text(pf.read_text().replace("byte_cost: 2.0e-5", "byte_cost: 1.0e-5"))
        try:
            ravc_main(["run", *args, "--dataset", str(ds), "--registration", str(reg), "--out", str(root / "o"),
                       "--trials", "1"])
            raise AssertionError("changed protocol after registration must be refused")
        except SystemExit as e:
            assert "registration mismatch" in str(e), e
        pf.write_text(pf.read_text().replace("byte_cost: 1.0e-5", "byte_cost: 2.0e-5"))
        ravc_main(["run", *args, "--dataset", str(ds), "--registration", str(reg), "--out", str(root / "o"),
                   "--trials", "1"])
        J = json.loads((root / "o" / "ravc_dev.json").read_text())
        assert J["acceptance"]["family"] == "human" and "edge_LTT" in (root / "o" / "ravc_dev.md").read_text()
        try:
            ravc_main(["run", *args, "--dataset", str(ds), "--registration", str(reg), "--out", str(root / "o"),
                       "--trials", "1"])
            raise AssertionError("second run must be refused")
        except SystemExit as e:
            assert "already made" in str(e)
    print("RAVC-LTT TESTS OK")


if __name__ == "__main__":
    run_tests()
