"""Reproducible self-tests of the statistics and of every protocol guard (no GPU, no real data).

    python -m cascade.selftest            # ~1 min, prints PASS/FAIL/SKIP per check, exit code 1 on any failure
The number of checks depends on the environment: the TikZ compile check needs pdflatex and the
RAVC-LTT view tests need OpenCV; each is SKIPPED, never FAILED, when its dependency is absent. Acceptance: exit code 0, no FAIL line, last lines "N/N checks passed" and "SELFTEST OK".

Builds a small fake FireSmoke-style dataset (manifest only) with fake detector / agent CSVs and meta files in a
temporary folder and drives the real command-line entry points.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.policy import Grid, event_miss_grid, evaluate_grid, metrics  # noqa: E402
from cascade.risk import unit_weights  # noqa: E402
from cascade.select import calibrate  # noqa: E402

RESULTS = []
SKIPPED = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not cond else ""), flush=True)


def run(*args, root=ROOT, env=None):
    p = subprocess.run([sys.executable, "-m", *args], cwd=root, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(root), **(env or {})})
    return p.returncode == 0, p.stdout + p.stderr


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ statistics
def test_statistics():
    rng = np.random.default_rng(1)
    N = 500
    unit = rng.integers(0, 40, N).astype(str)
    event = np.array([f"{u}:{rng.integers(0, 3)}" for u in unit])
    y = rng.random(N) < 0.5
    s = rng.random(N)
    g = np.where(rng.random(N) < 0.9, rng.random(N), np.nan)
    grid = Grid.default(s, n_s=8, n_g=6)
    w, _ = unit_weights(y, unit)
    E = evaluate_grid(s, g, grid, {"miss": w})["miss"]
    EM, _ = event_miss_grid(s, g, y, unit, event, grid)
    J, K = len(grid.s_edges), len(grid.g_edges)
    d1 = d2 = 0.0
    for j1, j2, k1, k2 in itertools.product(range(J), range(J), range(K), range(K)):
        if j1 > j2 or k1 > k2:
            continue
        th = (grid.s_edges[j1], grid.s_edges[j2], grid.g_edges[k1], grid.g_edges[k2])
        m = metrics(s, g, y, th, unit, None, event)
        d1 = max(d1, abs(m["miss_unit"] - E[j1, j2, k1, k2]))
        d2 = max(d2, abs(m["miss_event"] - EM[j1, j2, k1, k2]))
    check("grid evaluator == direct metrics (unit miss)", d1 < 1e-12, d1)
    check("event grid evaluator == direct metrics (event miss)", d2 < 1e-12, d2)

    from cascade.simulate import make
    d = make(n_units=1500, seed=4)
    t1, _ = calibrate(d["s"], d["g"], d["y"], d["unit"])
    t2, _ = calibrate(d["s"], d["g"], d["y"], d["unit"], risk_labels={"miss": d["y"]})
    check("single-risk call == explicit risk_labels", t1 == t2)
    # the certify part's (unlabelled) scores must not change the grid or the candidate order
    _, i1 = calibrate(d["s"], d["g"], d["y"], d["unit"], seed=7)
    uu = np.unique(d["unit"])
    sel = set(np.random.default_rng(7).choice(uu, int(round(0.3 * len(uu))), replace=False))
    cert = np.array([u not in sel for u in d["unit"]])
    s2 = d["s"].copy()
    s2[cert] = np.clip(s2[cert] ** 3, 0, 1)
    _, i2 = calibrate(s2, d["g"], d["y"], d["unit"], seed=7)
    check("grid depends on the select part only", i1["grid"] == i2["grid"])
    tb, ib = calibrate(d["s"], d["g"], d["y"], d["unit"], fwer="bonferroni")
    check("bonferroni uses the data-independent grid", ib["grid"] == Grid.fixed(0.02).to_dict())


# ------------------------------------------------------------------ fake dataset
def make_fake(root: Path, n_units=300, seed=0):
    rng = np.random.default_rng(seed)
    ds = root / "ds"
    ds.mkdir()
    fields = ["uid", "source", "orig_split", "orig_image", "split", "image", "label", "image_class", "md5", "w", "h",
              "n_smoke", "n_fire", "dup_cluster", "split_group", "test_unique", "tags", "ann_scope", "label_issues"]
    rows, det, ag = [], [], []
    gid = 0
    for u in range(n_units):
        split = rng.choice(["val", "test"], p=[0.4, 0.6])
        src = rng.choice(["dfire", "fasdd_cv", "pyro_sdis"], p=[0.45, 0.45, 0.10])
        k = 1 if src != "pyro_sdis" else int(rng.integers(5, 20))
        pos = rng.random() < 0.6
        for i in range(k):
            name = f"{src}__img{u}_{i}.jpg" if src != "pyro_sdis" else \
                f"pyro_sdis__cam{u}_2024-01-04T08-{i:02d}-00.jpg"
            ns, nf = (int(rng.random() < 0.6), int(rng.random() < 0.6)) if pos else (0, 0)
            if pos and ns + nf == 0:
                nf = 1
            uid = f"{src}:{split}:{u}_{i}"
            rows.append({"uid": uid, "source": src, "split": split, "image": f"images/{split}/{name}",
                         "image_class": "", "n_smoke": ns, "n_fire": nf, "split_group": gid, "tags": ""})
            z = 2.5 * (ns + nf > 0) - 1.5 + rng.normal(0, 1)
            s = 1 / (1 + np.exp(-z))
            det.append({"uid": uid, "s": f"{s:.5f}", "s_smoke": f"{s:.5f}", "s_fire": "0", "n_boxes_25": 0,
                        "boxes": "", "img_w": 640, "img_h": 480, "nbytes_frame": 50000, "nbytes_crop": 9000,
                        "det_ms": "5.0"})
            if s >= 0.02:
                gz = 3 * (ns + nf > 0) - 1.5 + rng.normal(0, 1)
                ag.append({"uid": uid, "g": f"{1 / (1 + np.exp(-gz)):.5f}", "logit_yes": 0, "logit_no": 0,
                           "agent_ms": "300", "n_tokens": 400, "agent": "fake", "view": "overlay"})
        gid += 1
    # a verified cross-source near-duplicate cluster in test: ONE split_group spanning dfire and fasdd_cv
    for src in ("dfire", "fasdd_cv", "fasdd_cv"):
        uid = f"{src}:test:cross_{len(rows)}"
        rows.append({"uid": uid, "source": src, "split": "test", "image": f"images/test/{src}__cross{len(rows)}.jpg",
                     "image_class": "", "n_smoke": 1, "n_fire": 0, "split_group": gid, "tags": ""})
        det.append({"uid": uid, "s": "0.7", "s_smoke": "0.7", "s_fire": "0", "n_boxes_25": 1, "boxes": "",
                    "img_w": 640, "img_h": 480, "nbytes_frame": 50000, "nbytes_crop": 9000, "det_ms": "5.0"})
        ag.append({"uid": uid, "g": "0.8", "logit_yes": 0, "logit_no": 0, "agent_ms": "300", "n_tokens": 400,
                   "agent": "fake", "view": "overlay"})
    gid += 1
    # a small source: one test video (1 unit) -> must go to external_shift
    for i in range(6):
        uid = f"flame2_det:test:v0_{i}"
        rows.append({"uid": uid, "source": "flame2_det", "split": "test", "image": f"images/test/flame2_det__{i}.jpg",
                     "image_class": "", "n_smoke": 0, "n_fire": 1, "split_group": gid, "tags": ""})
        det.append({"uid": uid, "s": "0.6", "s_smoke": "0", "s_fire": "0.6", "n_boxes_25": 1, "boxes": "",
                    "img_w": 640, "img_h": 480, "nbytes_frame": 50000, "nbytes_crop": 9000, "det_ms": "5.0"})
        ag.append({"uid": uid, "g": "0.7", "logit_yes": 0, "logit_no": 0, "agent_ms": "300", "n_tokens": 400,
                   "agent": "fake", "view": "overlay"})
    gid += 1
    for i in range(20):
        uid = f"ext:{i}"
        rows.append({"uid": uid, "source": "ext", "split": "external", "image": f"external/fog/fire/e{i}.jpg",
                     "image_class": "fire" if i % 2 else "nofire", "n_smoke": 0, "n_fire": 0, "split_group": gid,
                     "tags": ""})
        gid += 1
        det.append({"uid": uid, "s": "0.5", "s_smoke": "0.5", "s_fire": "0", "n_boxes_25": 0, "boxes": "",
                    "img_w": 640, "img_h": 480, "nbytes_frame": 50000, "nbytes_crop": 9000, "det_ms": "5.0"})
        ag.append({"uid": uid, "g": "0.6", "logit_yes": 0, "logit_no": 0, "agent_ms": "300", "n_tokens": 400,
                   "agent": "fake", "view": "overlay"})
    with open(ds / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def write(path, rs):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
    write(root / "det.csv", det)
    write(root / "agent.csv", ag)
    plain = [{"uid": d_["uid"], "g": f"{rng.random():.5f}", "logit_yes": 0, "logit_no": 0, "agent_ms": "250",
              "n_tokens": 400, "agent": "fake", "view": "plain"} for d_ in det]
    write(root / "agentplain.csv", plain)
    det_meta = {"weights_sha256": "ab" * 32, "ultralytics": "8.4.153", "nms": "false", "imgsz": 640, "conf": 0.001,
                "iou": 0.7, "max_det": 100, "precision": "fp32", "runtime_end2end": True}
    (root / "det.meta.json").write_text(json.dumps(det_meta))
    ag_meta = {"model": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "r" * 40, "quant": "none", "view": "overlay",
               "max_side": 640, "s_min": 0.02, "prompt_sha256": "fe" * 32, "dry": False,
               "det_meta_sha256": hashlib.sha256((root / "det.meta.json").read_bytes()).hexdigest(),
               "det_csv_sha256": hashlib.sha256((root / "det.csv").read_bytes()).hexdigest()}
    (root / "agent.meta.json").write_text(json.dumps(ag_meta))
    (root / "agentplain.meta.json").write_text(json.dumps(dict(ag_meta, view="plain", s_min=0.0)))
    return ds


def test_head_rule():
    from cascade.head_report import select_head
    nan = float("nan")
    H = [dict(name="A", nms="false", fire=0.95, smoke=0.90, neg_esc=0.30, map50=nan, det_ms=5.0),
         dict(name="B", nms="none", fire=0.955, smoke=0.905, neg_esc=0.35, map50=nan, det_ms=6.0)]
    check("head rule: recall within 1 pp -> lower escalation wins", select_head(H)[0] == "A")
    H = [dict(name="A", nms="false", fire=0.99, smoke=0.80, neg_esc=0.30, map50=0.7, det_ms=5.0),
         dict(name="B", nms="none", fire=0.90, smoke=0.95, neg_esc=0.30, map50=0.7, det_ms=5.0)]
    check("head rule: split winner -> max min(fire, smoke)", select_head(H)[0] == "B")
    H = [dict(name="A", nms="none", fire=0.95, smoke=0.95, neg_esc=0.30, map50=0.700, det_ms=5.0),
         dict(name="B", nms="false", fire=0.95, smoke=0.95, neg_esc=0.305, map50=0.702, det_ms=5.0)]
    check("head rule: full tie -> nms false", select_head(H)[0] == "B")


def test_protocol():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ds = make_fake(root)
        P = root / "proto"
        det, agent = str(root / "det.csv"), str(root / "agent.csv")
        ok, outp = run("cascade.make_protocol", "init", "--dataset", str(ds), "--out", str(P))
        check("init runs", ok, outp[-300:])
        lines = [ln for ln in outp.split("unit audit")[0].splitlines() if ln.startswith(("calibration", "sealed_test"))]
        check("init shows no positive counts for calibration / sealed_test",
              lines and all(ln.split()[-1] == "-" and ln.split()[-2] == "-" for ln in lines), lines[:2])
        import yaml
        check("protocol.yaml parses (dataset path safely quoted)",
              Path(yaml.safe_load((P / "protocol.yaml").read_text())["dataset"]) == ds)

        def records(out, *extra):
            return run("cascade.build_records", "--dataset", str(ds), "--det", det, "--agent", agent,
                       "--out", str(out), "--protocol", str(P), *extra)
        rec_pre = root / "rec_pre.csv"
        ok, outp = records(rec_pre)
        check("records before freeze (for dev)", ok, outp[-300:])
        with open(rec_pre) as f:
            R = list(csv.DictReader(f))
        def labelled(rows_, role):
            rr = [r for r in rows_ if r["role"] == role]
            return rr and all(r["y"] != "" for r in rr), rr and all(r["y"] == "" for r in rr)
        check("before freeze: only dev labels (calibration, sealed_test, external_shift blank)",
              labelled(R, "dev")[0] and labelled(R, "calibration")[1] and labelled(R, "sealed_test")[1]
              and labelled(R, "external_shift")[1])
        with open(P / "splits.csv") as f:
            sp = {r["uid"]: r for r in csv.DictReader(f)}
        check("units / events come from frozen splits.csv", all(r["unit"] == sp[r["uid"]]["unit"] and
                                                                  r["event"] == sp[r["uid"]]["event"] for r in R))
        cross = [r for r in sp.values() if "cross" in r["uid"]]
        with open(P / "unit_audit.csv") as f:
            ua = list(csv.DictReader(f))
        check("unit audit lists the cross-source unit (label-free)", len(ua) == 1
              and ua[0]["sources"].startswith("fasdd_cv:2;dfire:1"), ua)
        check("cross-source duplicate cluster stays ONE unit with ONE role",
              len({r["unit"] for r in cross}) == 1 and len({r["role"] for r in cross}) == 1
              and len({r["source"] for r in cross}) == 2)
        roles_by_unit = {}
        for r in sp.values():
            roles_by_unit.setdefault(r["unit"], set()).add(r["role"])
        check("no unit spans two roles", all(len(v) == 1 for v in roles_by_unit.values()))
        fl = [r for r in sp.values() if r["source"] == "flame2_det"]
        check("source with < min_units test units -> external_shift", fl and all(r["role"] == "external_shift"
                                                                               for r in fl)
              and "flame2_det" in (P / "protocol.yaml").read_text())
        with open(det) as f:
            dl = f.read().splitlines()
        (root / "det_missing.csv").write_text("\n".join(dl[:-1]) + "\n")
        (root / "det_missing.meta.json").write_text((root / "det.meta.json").read_text())
        ok, outp = run("cascade.build_records", "--dataset", str(ds), "--det", str(root / "det_missing.csv"),
                       "--agent", agent, "--out", str(root / "x.csv"), "--protocol", str(P))
        check("missing detector score -> refused", not ok and "fail-closed" in outp)
        rc = ["cascade.run_cascade", "--protocol", str(P), "--trials", "3"]
        ok, outp = run(*rc, "--records", str(rec_pre), "--mode", "dev", "--out-dir", str(root / "res"))
        check("dev mode runs before freeze", ok, outp[-400:])
        ok, outp = run("cascade.figures", "dev", "--protocol", str(P), "--dev-dir", str(root / "res"),
                       "--out", str(root / "figdev"))
        mf = json.loads((root / "figdev" / "figures_manifest.json").read_text()) if ok else {}
        tx = (root / "figdev" / "fig4_forest_dev.tex").read_text() if ok else ""
        check("figures dev: TikZ .tex + .csv, reads dev_* inputs only",
              ok and "\\begin{tikzpicture}" in tx and "\\begin{groupplot}" in tx
              and (root / "figdev" / "fig4_forest_dev.csv").exists() and (root / "figdev" / "figure_preamble.tex").exists()
              and mf["inputs"] and all(k.startswith("dev_") for k in mf["inputs"]) and mf["role"] == "dev",
              outp[-300:])
        ok, outp = run("cascade.figures", "final", "--protocol", str(P))
        check("figures final before any final -> refused", not ok and "not 'done'" in outp)
        ok, outp = run(*rc, "--records", str(rec_pre), "--mode", "validity", "--out-dir", str(root / "res"))
        check("validity before freeze -> refused", not ok and "validity refused" in outp)
        ok, outp = run(*rc, "--records", str(rec_pre), "--mode", "final", "--out-dir", str(root / "res"))
        check("final before freeze -> refused", not ok and "final refused" in outp)
        ok, outp = records(root / "u.csv", "--unseal")
        check("unseal before freeze -> refused", not ok)

        bad = json.loads((root / "det.meta.json").read_text())
        bad["nms"] = "none"
        (root / "detbad.csv").write_bytes((root / "det.csv").read_bytes())
        (root / "detbad.meta.json").write_text(json.dumps(bad))
        ok, outp = run("cascade.make_protocol", "freeze", "--out", str(P), "--det", str(root / "detbad.csv"),
                       "--agent", agent)
        check("freeze with agent not produced from this detector meta -> refused",
              not ok and "det_meta_sha256" in outp)
        am = json.loads((root / "agent.meta.json").read_text())
        am["det_meta_sha256"] = hashlib.sha256((root / "detbad.meta.json").read_bytes()).hexdigest()
        (root / "agentbad.csv").write_bytes((root / "agent.csv").read_bytes())
        (root / "agentbad.meta.json").write_text(json.dumps(am))
        ok, outp = run("cascade.make_protocol", "freeze", "--out", str(P), "--det", str(root / "detbad.csv"),
                       "--agent", str(root / "agentbad.csv"))
        check("freeze with detector nms != protocol nms -> refused", not ok and "detector.nms" in outp, outp[-200:])
        m_ok = (root / "det.meta.json").read_bytes()
        mm = json.loads(m_ok)
        mm["runtime_end2end"] = False                       # nms "false" but the NMS head actually ran
        (root / "det.meta.json").write_text(json.dumps(mm))
        am2 = json.loads((root / "agent.meta.json").read_text())
        a_ok = (root / "agent.meta.json").read_bytes()
        am2["det_meta_sha256"] = hashlib.sha256((root / "det.meta.json").read_bytes()).hexdigest()
        (root / "agent.meta.json").write_text(json.dumps(am2))
        ok, outp = run("cascade.make_protocol", "freeze", "--out", str(P), "--det", det, "--agent", agent)
        check("freeze with runtime_end2end inconsistent with nms -> refused", not ok and "runtime_end2end" in outp)
        (root / "det.meta.json").write_bytes(m_ok)
        (root / "agent.meta.json").write_bytes(a_ok)
        det_orig = Path(det).read_bytes()
        Path(det).write_bytes(det_orig.replace(b",0.", b",0.9", 1))
        ok, outp = run("cascade.make_protocol", "freeze", "--out", str(P), "--det", det, "--agent", agent)
        check("freeze after detector CSV edited post-agent -> refused", not ok and "det_csv_sha256" in outp)
        Path(det).write_bytes(det_orig)
        # fill-weights: the ONE path from the detector meta into protocol.yaml, before freeze
        ok, outp = run("cascade.make_protocol", "fill-weights", "--out", str(P), "--det", det)
        yml_now = (P / "protocol.yaml").read_text()
        check("fill-weights writes the meta digest into protocol.yaml", ok and "ab" * 32 in yml_now, outp[-300:])
        check("fill-weights keeps a snapshot of the previous protocol.yaml",
              any(x.name.endswith(".snapshot") for x in P.glob("protocol.yaml.*")))
        ok2, outp2 = run("cascade.make_protocol", "fill-weights", "--out", str(P), "--det", det)
        check("fill-weights is idempotent for the same digest", ok2 and "nothing to do" in outp2, outp2[-200:])
        (P / "protocol.yaml").write_text(yml_now.replace('"' + "ab" * 32 + '"', '"' + "cd" * 32 + '"'))
        ok3, outp3 = run("cascade.make_protocol", "fill-weights", "--out", str(P), "--det", det)
        check("fill-weights refuses to overwrite a DIFFERENT digest", not ok3 and "DIFFERENT" in outp3)
        ok4, outp4 = run("cascade.make_protocol", "freeze", "--out", str(P), "--det", det, "--agent", agent)
        check("freeze refuses a protocol digest that disagrees with the meta",
              not ok4 and "weights_sha256" in outp4)
        (P / "protocol.yaml").write_text(yml_now)
        ok, outp = run("cascade.make_protocol", "freeze", "--out", str(P), "--det", det, "--agent", agent)
        check("freeze with matching metas and CSVs (weights pre-filled)", ok, outp[-300:])
        ok5, outp5 = run("cascade.make_protocol", "fill-weights", "--out", str(P), "--det", det)
        check("fill-weights refused after freeze", not ok5 and "already frozen" in outp5)
        yml = (P / "protocol.yaml").read_text()
        check("freeze filled hashes into protocol.yaml", "ab" * 32 in yml and "r" * 40 in yml and "fe" * 32 in yml)

        ok, outp = run(*rc, "--records", str(rec_pre), "--mode", "validity", "--out-dir", str(root / "res"))
        check("validity on pre-freeze records -> refused", not ok and "different lock" in outp)
        rec_post = root / "rec_post.csv"
        ok, outp = records(rec_post)
        with open(rec_post) as f:
            R2 = list(csv.DictReader(f))
        check("after freeze: dev + calibration labels, sealed_test and external_shift blank",
              ok and labelled(R2, "dev")[0] and labelled(R2, "calibration")[0] and labelled(R2, "sealed_test")[1]
              and labelled(R2, "external_shift")[1], outp[-300:])
        ok, outp = run(*rc, "--records", str(rec_post), "--mode", "validity", "--out-dir", str(root / "res"))
        check("validity after freeze runs", ok, outp[-400:])
        for which, path in (("detector", det), ("agent", agent)):
            orig = Path(path).read_bytes()
            Path(path).write_bytes(orig.replace(b",0.", b",0.8", 1))
            ok1, o1 = records(root / "t.csv")
            ok2, o2 = records(root / "t.csv", "--unseal")
            Path(path).write_bytes(orig)
            check(f"{which} CSV edited after freeze -> records and unseal refused", not ok1 and not ok2, o1[-200:])
        uns = root / "unsealed.csv"
        ok, outp = records(uns, "--unseal")
        with open(uns) as f:
            R3 = list(csv.DictReader(f))
        check("unseal after freeze: labels for every role", ok and all(r["y"] != "" for r in R3), outp[-300:])
        rcpt = json.loads((P / "final_receipt.json").read_text())
        check("receipt created complete and atomically (no temp files left, valid JSON)",
              not list(P.glob("final_receipt.json.*")) and json.loads((P / "final_receipt.json").read_text()))
        check("receipt: state unsealed with the records_sha256 of the unsealed file",
              rcpt.get("state") == "unsealed" and rcpt.get("records_sha256") == sha(uns)
              and rcpt["events"][0]["event"] == "unseal_start")
        ok, outp = records(root / "unsealed_again.csv", "--unseal")
        check("second --unseal to another file -> refused", not ok and "receipt already exists" in outp
              and not (root / "unsealed_again.csv").exists())
        ok, outp = records(root / "unsealed_again.csv", "--unseal", "--resume")
        check("--unseal --resume to another file -> refused", not ok and "another records file" in outp)
        ok, outp = records(uns)
        check("sealed build over the receipt's records file -> refused", not ok and "receipt" in outp)
        ok, outp = records(uns, "--unseal", "--resume")
        rcpt = json.loads((P / "final_receipt.json").read_text())
        check("--unseal --resume, same file: allowed, logged, same hash",
              ok and rcpt["state"] == "unsealed" and rcpt["records_sha256"] == sha(uns)
              and [e["event"] for e in rcpt["events"]].count("unseal_resume") == 1, outp[-300:])
        # a different (but internally consistent) records file: only the receipt can catch it
        alt = root / "unsealed_alt.csv"
        alt.write_bytes(uns.read_bytes().replace(b",0.5,", b",0.51,", 1))
        am_ = json.loads(uns.with_suffix(".meta.json").read_text())
        am_["records_sha256"] = sha(alt)
        alt.with_suffix(".meta.json").write_text(json.dumps(am_))
        ok, outp = run(*rc, "--records", str(alt), "--mode", "final", "--out-dir", str(root / "fin_alt"))
        check("final with records other than the receipt's -> refused", not ok and "not the ones recorded" in outp,
              outp[-300:])
        ok, outp = run(*rc, "--records", str(rec_post), "--mode", "final", "--out-dir", str(root / "fin"))
        check("final with sealed records -> refused", not ok and "without --unseal" in outp)
        # crash after the results were written but before 'done'
        ok, outp = run(*rc, "--records", str(uns), "--mode", "final", "--out-dir", str(root / "fin"),
                       env={"CASCADE_SELFTEST_CRASH_FINAL": "1"})
        rcpt = json.loads((P / "final_receipt.json").read_text())
        check("crashed final leaves receipt state running", not ok and rcpt["state"] == "running", outp[-300:])
        ok, outp = run(*rc, "--records", str(uns), "--mode", "final", "--out-dir", str(root / "fin_x"))
        check("final after crash without --resume -> refused", not ok and "--resume" in outp)
        ok, outp = run(*rc, "--records", str(uns), "--mode", "final", "--out-dir", str(root / "fin_x"), "--resume")
        check("final --resume with another out-dir -> refused", not ok and "out_dir" in outp)
        ok, outp = records(uns, "--unseal", "--resume")
        check("--unseal --resume while final is running -> refused", not ok and "state is 'running'" in outp)
        ok, outp = run(*rc, "--records", str(uns), "--mode", "final", "--out-dir", str(root / "fin"), "--resume")
        fin = (root / "fin" / "final.md").read_text() if (root / "fin" / "final.md").exists() else ""
        rcpt = json.loads((P / "final_receipt.json").read_text())
        ev = [e["event"] for e in rcpt["events"]]
        check("final --resume (same out-dir): done, logged, partial output renamed not deleted",
              ok and rcpt["state"] == "done" and ev.count("final_resume") == 1
              and (root / "fin" / "final.md.crashed_1").exists() and "final resumes 1" in fin, outp[-400:])
        check("final runs once, with per-risk bootstrap CIs", ok and "y_fire:ci95" in fin and "5000-replicate" in fin,
              outp[-400:])
        rc_before = (P / "final_receipt.json").read_bytes()
        figdir = root / "fin" / "figures"
        okv, ov = run("cascade.make_protocol", "verify-final", "--out", str(P))
        check("verify-final: receipt, final.md, final.json, records, lock all match", okv and "verify-final: OK" in ov,
              ov[-300:])
        ok, outp = run("cascade.figures", "final", "--protocol", str(P))
        mf = json.loads((figdir / "figures_manifest.json").read_text()) if (figdir / "figures_manifest.json").exists() \
            else {}
        check("figures final (TikZ) written from final.json only (hash = receipt)",
              ok and "tikzpicture" in (figdir / "fig4_forest.tex").read_text() and (figdir / "fig4_forest.csv").exists()
              and mf.get("inputs", {}).get("final.json") == rcpt.get("final_json_sha256")
              and rcpt.get("final_json_sha256") == sha(root / "fin" / "final.json"))
        import shutil as _sh
        if _sh.which("pdflatex"):
            check("TikZ figures compile (pdflatex preview of every figure)",
                  all((figdir / "preview" / f"{n}.pdf").exists() for n in
                      ("fig3_risk_cost", "fig4_forest", "fig5_edge_cloud_cost", "fig6_by_source")))
        else:
            SKIPPED.append("TikZ compile check (pdflatex not installed; .tex/.csv still checked)")
            print("[SKIP] TikZ compile check: pdflatex not installed", flush=True)
        ok, outp = run("cascade.figures", "final", "--protocol", str(P), "--out", str(root / "fig_again"))
        check("figures final re-run: allowed, receipt untouched", ok and (P / "final_receipt.json").read_bytes()
              == rc_before, outp[-300:])
        fjb = (root / "fin" / "final.json").read_bytes()
        (root / "fin" / "final.json").write_bytes(fjb.replace(b'"risk"', b'"risk "', 1))
        ok, outp = run("cascade.figures", "final", "--protocol", str(P), "--out", str(root / "fig_bad"))
        okv, ov = run("cascade.make_protocol", "verify-final", "--out", str(P))
        (root / "fin" / "final.json").write_bytes(fjb)
        check("verify-final on an edited final.json -> FAILED", not okv and "final.json differs" in ov, ov[-300:])
        check("figures final on an edited final.json -> refused", not ok and "differ from the hashes" in outp)
        ok, outp = run(*rc, "--records", str(uns), "--mode", "final", "--out-dir", str(root / "fin"))
        check("second final (same out-dir) -> refused", not ok and "evaluated once" in outp)
        ok, outp = run(*rc, "--records", str(uns), "--mode", "final", "--out-dir", str(root / "fin_other"))
        check("second final (other out-dir) -> refused by protocol receipt", not ok and "already run" in outp)
        rcpt = json.loads((P / "final_receipt.json").read_text())
        check("receipt records state done + final.md hash", rcpt.get("state") == "done" and
              rcpt.get("final_md_sha256") == sha(root / "fin" / "final.md"))
        ok, outp = records(uns, "--unseal", "--resume")
        check("--unseal --resume after done -> refused", not ok and "state is 'done'" in outp)
        check("FLAME2 reported in external shift, not in sealed test", "flame2_det/" in fin)
        txt = uns.read_bytes()
        uns.write_bytes(txt.replace(b",0.5,", b",0.49,", 1))
        ok, outp = run(*rc, "--records", str(uns), "--mode", "final", "--out-dir", str(root / "fin2"))
        check("edited records -> refused", not ok and "changed after" in outp)
        uns.write_bytes(txt)
        # pure cloud baseline, full chain on a second protocol
        P2 = root / "proto2"
        ok, _ = run("cascade.make_protocol", "init", "--dataset", str(ds), "--out", str(P2))
        (P2 / "protocol.yaml").write_text((P2 / "protocol.yaml").read_text().replace(
            "pure_cloud_baseline: false", "pure_cloud_baseline: true"))
        plain_csv = str(root / "agentplain.csv")
        ok, outp = run("cascade.make_protocol", "freeze", "--out", str(P2), "--det", det, "--agent", agent)
        check("pure cloud: freeze without --agent-plain -> refused", not ok and "agent-plain" in outp)
        ok, outp = run("cascade.make_protocol", "freeze", "--out", str(P2), "--det", det, "--agent", agent,
                       "--agent-plain", plain_csv)
        check("pure cloud: freeze with --agent-plain", ok, outp[-300:])
        ok, outp = run("cascade.build_records", "--dataset", str(ds), "--det", det, "--agent", agent,
                       "--agent-plain", plain_csv, "--out", str(root / "u2.csv"), "--protocol", str(P2), "--unseal")
        ok2, outp2 = run("cascade.run_cascade", "--records", str(root / "u2.csv"), "--protocol", str(P2),
                         "--mode", "final", "--out-dir", str(root / "fin_pc"))
        f2 = (root / "fin_pc" / "final.md").read_text() if (root / "fin_pc" / "final.md").exists() else ""
        row = next((ln for ln in f2.splitlines() if ln.startswith("| pure_cloud_plain")), "")
        check("pure cloud: final has pure_cloud_plain with plain-agent latency (250 ms, no detector)",
              ok and ok2 and row and "| 250.0000 |" in row, (outp + outp2)[-300:])
        (P / "protocol.yaml").write_text(yml + "\n# edit\n")
        ok, outp = run(*rc, "--records", str(uns), "--mode", "final", "--out-dir", str(root / "fin3"))
        check("edited protocol.yaml -> refused", not ok and "lock" in outp)


def test_code_lock():
    """freeze locks analysis_code_sha256; one changed byte in policy.py (in a COPY of the package) must make
    records / validity / --unseal / final refuse. The real package is never modified."""
    import shutil
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pkg = root / "pkg"
        shutil.copytree(ROOT / "cascade", pkg / "cascade", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        ds = make_fake(root, n_units=160, seed=3)
        P = root / "proto"
        det, agent = str(root / "det.csv"), str(root / "agent.csv")

        def R(*args, **kw):
            return run(*args, root=pkg, **kw)
        ok1, o1 = R("cascade.make_protocol", "code-hash")
        ok2, o2 = run("cascade.make_protocol", "code-hash")
        check("analysis_code_sha256 is deterministic (copy == original)", ok1 and ok2 and
              o1.strip().splitlines()[-1] == o2.strip().splitlines()[-1], (o1 + o2)[-200:])
        R("cascade.make_protocol", "init", "--dataset", str(ds), "--out", str(P))
        ok, outp = R("cascade.make_protocol", "freeze", "--out", str(P), "--det", det, "--agent", agent)
        L = json.loads((P / "protocol.lock").read_text()) if ok else {}
        check("freeze writes analysis_code_sha256 into the lock", ok and len(L.get("analysis_code_sha256", "")) == 64
              and "policy.py" in L.get("analysis_code_files", {}), outp[-300:])

        def records(out, *extra):
            return R("cascade.build_records", "--dataset", str(ds), "--det", det, "--agent", agent,
                     "--out", str(out), "--protocol", str(P), *extra)
        rec = root / "rec_post.csv"
        ok, outp = records(rec)
        check("code lock: post-freeze records with unchanged code", ok, outp[-300:])
        pol = pkg / "cascade" / "policy.py"
        orig = pol.read_bytes()
        i = orig.index(b'"""') + 3                     # one byte inside the module docstring (code still imports)
        tampered = orig[:i] + bytes([orig[i] ^ 0x20]) + orig[i + 1:]
        rc = ["cascade.run_cascade", "--protocol", str(P), "--trials", "2"]
        pol.write_bytes(tampered)
        ok_r, o_r = records(root / "rec2.csv")
        ok_v, o_v = R(*rc, "--records", str(rec), "--mode", "validity", "--out-dir", str(root / "res"))
        ok_u, o_u = records(root / "uns.csv", "--unseal")
        check("policy.py 1 byte changed -> records refused", not ok_r and "analysis code" in o_r, o_r[-300:])
        check("policy.py 1 byte changed -> validity refused", not ok_v and "analysis code" in o_v, o_v[-300:])
        check("policy.py 1 byte changed -> --unseal refused, no receipt created",
              not ok_u and "analysis code" in o_u and not (P / "final_receipt.json").exists(), o_u[-300:])
        pol.write_bytes(orig)
        ok, outp = records(root / "uns.csv", "--unseal")
        check("code restored byte-exactly -> --unseal allowed", ok, outp[-300:])
        pol.write_bytes(tampered)
        ok_f, o_f = R(*rc, "--records", str(root / "uns.csv"), "--mode", "final", "--out-dir", str(root / "fin"))
        pol.write_bytes(orig)
        rcpt = json.loads((P / "final_receipt.json").read_text())
        check("policy.py 1 byte changed -> final refused, receipt stays 'unsealed'",
              not ok_f and "analysis code" in o_f and rcpt["state"] == "unsealed", o_f[-300:])
        ok, outp = R(*rc, "--records", str(root / "uns.csv"), "--mode", "final", "--out-dir", str(root / "fin"))
        check("code restored -> final runs (figures made automatically only via run_all)", ok, outp[-300:])
        figpy = pkg / "cascade" / "figures.py"
        forig = figpy.read_bytes()
        j = forig.index(b'"""') + 3
        figpy.write_bytes(forig[:j] + bytes([forig[j] ^ 0x20]) + forig[j + 1:])
        ok1, o1 = R("cascade.figures", "final", "--protocol", str(P))
        ok2, o2 = R("cascade.figures", "final", "--protocol", str(P), "--post-freeze-figure-fix")
        ok3, o3 = R(*rc, "--records", str(rec), "--mode", "validity", "--out-dir", str(root / "res"))
        mf = json.loads((root / "fin" / "figures" / "figures_manifest.json").read_text()) if ok2 else {}
        check("figures.py changed after freeze -> figures final refused without the flag",
              not ok1 and "figure code changed" in o1, o1[-300:])
        check("--post-freeze-figure-fix: allowed and recorded in the manifest",
              ok2 and list(mf.get("post_freeze_figure_code_change", {})) == ["figures.py"], o2[-300:])
        check("figures.py changed -> validity still refused (figure code is in analysis_code_sha256)",
              not ok3 and "analysis code" in o3)
        pol.write_bytes(tampered)
        ok4, o4 = R("cascade.figures", "final", "--protocol", str(P), "--post-freeze-figure-fix")
        pol.write_bytes(orig)
        figpy.write_bytes(forig)
        check("policy.py changed -> figures final refused even with the flag", not ok4 and "analysis code" in o4)
        check("real package untouched", sha(ROOT / "cascade" / "policy.py") == sha(pol))


def test_run_all():
    """run_all.sh itself (set -euo pipefail): every STAGE name parses and no variable is used before it is set.
    Uses a clean environment (no OUT / PROTO / AUDIT / N exported) and only side-effect-free stages."""
    sh = ROOT / "cascade" / "run_all.sh"
    import shutil
    if shutil.which("bash") is None or (os.name == "nt" and subprocess.run(
            ["bash", "-c", "true"], capture_output=True).returncode != 0):
        SKIPPED.append("run_all.sh checks (no usable bash, e.g. Windows without WSL/Git Bash)")
        return
    with tempfile.TemporaryDirectory() as td:
        env = {"PATH": os.environ.get("PATH", ""), "HOME": td, "PYTHONPATH": str(ROOT)}
        exe = Path(sys.executable).parent
        env["PATH"] = f"{exe}{os.pathsep}{env['PATH']}"
        p = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        check("run_all.sh: bash syntax", p.returncode == 0, p.stderr[-300:])
        p = subprocess.run(["bash", str(sh)], cwd=ROOT, capture_output=True, text=True,
                           env={**env, "STAGE": "codehash"})
        check("run_all.sh STAGE=codehash in a clean env (set -u: no unbound variable)",
              p.returncode == 0 and "analysis_code_sha256:" in p.stdout and "unbound" not in p.stderr,
              (p.stdout + p.stderr)[-300:])
        p = subprocess.run(["bash", str(sh)], cwd=ROOT, capture_output=True, text=True,
                           env={**env, "STAGE": "no_such_stage"})
        check("run_all.sh: unknown STAGE -> exit 1", p.returncode == 1 and "unknown STAGE" in p.stdout,
              (p.stdout + p.stderr)[-300:])
        stages = set()
        for ln in sh.read_text().splitlines():
            s = ln.strip()
            if s.endswith(")") and not s.startswith(("#", "if", "python", "for")) and "=" not in s and "(" not in s:
                stages.update(x.strip() for x in s[:-1].split("|"))
        need = {"selftest", "protocol", "audit", "heads", "scores", "records", "dev", "freeze", "validity",
                "unseal", "final", "verify-final", "figures-dev", "figures-final", "dataset-audit", "codehash",
                "crop-bytes", "ravc-register", "ravc-dev", "diagnose", "esva-register", "esva-dev",
                "probe", "study-register"}
        check("run_all.sh: every documented STAGE exists", need <= stages, sorted(need - stages))


def test_ravc():
    """RAVC-LTT: router, handoff family, view bytes, dev command with registration (cascade.test_adaptive_view)."""
    import io
    from contextlib import redirect_stdout
    from .test_adaptive_view import run_tests
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_tests()
        check("RAVC-LTT tests (test_adaptive_view)", "RAVC-LTT TESTS OK" in buf.getvalue())
    except ModuleNotFoundError as e:                                # optional image dependency
        # view_bytes / dump_detector need OpenCV to MEASURE payload bytes. No statistical guard depends on
        # it, so a machine without cv2 skips this check rather than failing: the alternative is a selftest
        # whose result depends on which packages happen to be installed.
        SKIPPED.append(f"RAVC-LTT tests (test_adaptive_view): {e.name} not installed")
    except (AssertionError, SystemExit, Exception) as e:            # noqa: BLE001
        check("RAVC-LTT tests (test_adaptive_view)", False, repr(e)[-300:])


def test_esva():
    """ESVA-LTT: stratum, discount, certification, dev_diagnose, esva_dev registration (cascade.test_esva)."""
    import io
    from contextlib import redirect_stdout
    from .test_esva import run_tests
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_tests()
        check("ESVA-LTT tests (test_esva)", "ESVA TESTS OK" in buf.getvalue())
    except (AssertionError, SystemExit, Exception) as e:            # noqa: BLE001
        check("ESVA-LTT tests (test_esva)", False, repr(e)[-300:])


def test_ace():
    """ACE-LTT: admissibility, chain certificate, tie-safe path, NOT CERTIFIED paths (cascade.test_ace_ltt)."""
    import io
    from contextlib import redirect_stdout
    from .test_ace_ltt import run_tests
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_tests()
        check("ACE-LTT tests (test_ace_ltt)", "ACE-LTT TESTS OK" in buf.getvalue())
    except (AssertionError, SystemExit, Exception) as e:            # noqa: BLE001
        check("ACE-LTT tests (test_ace_ltt)", False, repr(e)[-300:])


def test_ace_registration():
    """The ACE declaration must decompose as declared, and its preconditions must actually discriminate."""
    import numpy as _np
    from cascade.ace_ltt import allocate_delta, build_chains
    from cascade.ace_register import (DECLARATION, certify_mask, declared_family, events_spanning_units,
                                      membership_digests)
    pol, losses = declared_family()
    ch = build_chains(pol, losses)
    dm = allocate_delta(ch, DECLARATION["delta"], None)
    check("ACE declaration decomposes into the declared chains",
          len(ch) == DECLARATION["chains"]["M"] == 2
          and all(len(c) == len(DECLARATION["veto"]["lambda_grid"]) for c in ch)
          and all(abs(d - DECLARATION["chains"]["delta_m"]) < 1e-12 for d in dm), f"M={len(ch)} dm={dm}")
    lam = [float(p.veto_at("S")) for p in ch[0]]
    check("ACE chain runs from the safest candidate downwards", lam == sorted(lam, reverse=True), lam)
    check("ACE declares response as the catch set, not alarm",
          all(L["catch_set"] == "response" for L in DECLARATION["losses"]))
    check("ACE leaves the payload unconstrained in the dev study",
          "payload" in DECLARATION["diagnostic_only"] and not any(
              L.get("kind") == "payload" for L in DECLARATION["losses"]))

    with tempfile.TemporaryDirectory() as td:
        P = Path(td)
        head = "uid,source,unit,event,role\n"
        clean = head + "".join(f"a{i},dfire,u{i // 2},e{i // 2},dev\n" for i in range(8))
        (P / "splits.csv").write_text(clean)
        check("no spanning events in a clean split", events_spanning_units(P) == {})
        (P / "splits.csv").write_text(clean + "z1,dfire,u99,e0,dev\n")      # event e0 now in u0 and u99
        check("an event spanning two units is detected", events_spanning_units(P) == {"dfire": 1})

    unit = _np.array([f"u{i}" for i in _np.random.default_rng(3).integers(0, 200, 1200)])
    cert = certify_mask(unit, 0.3, 20260923)
    uu = _np.unique(unit)
    n_sel = len(set(unit[~cert]))
    check("registered split matches cascade.select.calibrate's rule",
          n_sel == max(1, int(round(0.3 * len(uu)))) and not (set(unit[cert]) & set(unit[~cert])),
          f"{n_sel} of {len(uu)}")
    D = {"unit": unit, "y_fire": (_np.random.default_rng(4).random(1200) < 0.4).astype(int),
         "y_smoke": (_np.random.default_rng(5).random(1200) < 0.4).astype(int)}
    d1, s1 = membership_digests(D, 0.3, 20260923)
    d2, s2 = membership_digests(dict(D), 0.3, 20260923)
    check("audit digests are reproducible and acquisition-independent", d1 == d2 and s1 == s2)
    check("the registration counts frames and units separately",
          s1["n_certify_frames"] > s1["n_certify_units"]
          and s1["n_certify_units"] + s1["n_select_units"] == s1["n_dev_units"]
          and all(d1[k]["n_units"] <= s1["n_certify_units"] for k in d1), s1)


def test_ace_end_to_end():
    """v2.3 regression: the Study-1 lambda family would be REFUSED for a NEW registration, and Study 2's
    b ladder is accepted and runs. The refusal is an admissibility statement about spec v2.3; it does not
    make the Study-1 experiment, registered and executed under v2.2, invalid."""
    import json as _json
    from cascade.ace_ltt import MARGIN_CAPABLE, MARGIN_INCAPABLE, SPEC_VERSION
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ds = make_fake(root, n_units=2500, seed=11)
        P = root / "P"
        run("cascade.make_protocol", "init", "--dataset", str(ds), "--out", str(P))
        det, ag = str(root / "det.csv"), str(root / "agent.csv")
        run("cascade.make_protocol", "fill-weights", "--out", str(P), "--det", det)
        args = ["--dataset", str(ds), "--protocol", str(P), "--det", det,
                "--agent", f"crop={ag}", "--agent", f"overlay={ag}"]

        ok1, o1 = run("cascade.ace_register", *args, "--out", str(root / "reg1.json"))
        check("v2.3 refuses the Study-1 lambda family for a NEW registration",
              not ok1 and MARGIN_INCAPABLE in o1 and not (root / "reg1.json").exists(), o1[-200:])
        check("the refusal is about margin capacity, not about validity",
              "chain_margin_capacity" in o1 and "invalid" not in o1.lower())

        reg = str(root / "reg2.json")
        ok, o = run("cascade.ace2_register", *args, "--out", reg)
        check("Study 2 (b ladder) registers under v2.3", ok and Path(reg).exists(), o[-300:])
        R = _json.loads(Path(reg).read_text())
        check("Study 2 records its spec version and exploratory status",
              R["spec_version"] == SPEC_VERSION and "exploratory" in R["declaration"]["status"])
        check("Study 2's chain coordinate is margin capable from DESIGN-time data",
              all(v["status"] == MARGIN_CAPABLE and v["usable_for_registration"]
                  for v in R["preconditions"]["P6_status"].values()), R["preconditions"]["P6_status"])
        sp = R["preconditions"]["P7_boundary_spanning"]
        check("Study 2's ladder crosses the joint boundary on the SELECT fold (Sec. 4.6)",
              all(v["status"] == "BOUNDARY_SPANNING" and v["safe_endpoint_feasible"]
                  and v["max_joint_slack"] >= 0 > v["min_joint_slack"] for v in sp.values()), sp)
        check("Study 2 freezes thresholds and the b ladder at registration",
              all("b_ladder" in v and len(v["b_ladder"]) >= 3
                  for v in R["preconditions"]["frozen_family"]["thresholds"].values()))
        ok2, _ = run("cascade.ace2_register", *args, "--out", reg)
        check("a second Study-2 registration is refused", not ok2)

        out = root / "run2"
        ok, o = run("cascade.ace2_dev", *args, "--registration", reg, "--out", str(out))
        check("Study 2 run executes the registered procedure", ok, o[-400:])
        rep = _json.loads((out / "ace2_dev.json").read_text())
        check("the run re-derives the registered cover before testing",
              rep["cover_id"] == R["preconditions"]["P2_chains"]["cover_id"]
              == _json.loads((out / "ace2_candidates.json").read_text())["cover_id"])
        check("realized capacity is reported as diagnostic only",
              all(v["source"] == "realized" and not
                  rep["capacity"]["status_realized"][k]["usable_for_registration"]
                  for k, v in rep["capacity"]["realized_DIAGNOSTIC_ONLY"].items()))
        check("the run is labelled exploratory", "exploratory" in rep["status"])
        ok3, _ = run("cascade.ace2_dev", *args, "--registration", reg, "--out", str(out))
        check("a second Study-2 run is refused", not ok3)
        # paired post-registration diagnostic: descriptive, outside ANALYSIS_CODE, cannot certify
        from cascade.make_protocol import ANALYSIS_CODE
        check("the diagnostic is outside ANALYSIS_CODE (it can never move a registered code hash)",
              "ace2_diag.py" not in ANALYSIS_CODE)
        nm = R["preconditions"]["frozen_family"]["policies"][3]["name"]
        okd, od = run("cascade.ace2_diag", *args, "--registration", reg, "--run", str(out), "--policy", nm)
        check("the paired diagnostic runs", okd, od[-300:])
        dg = _json.loads((out / "ace2_diag.json").read_text())
        check("the diagnostic carries no p-value and no decision rule",
              "p_value" not in od and "p_joint" not in od and "GO" not in od
              and "descriptive only" in dg["kind"])
        check("edge-only is the same object with the band collapsed (no calls, no hand-off)",
              dg["frames"]["edge_only"]["calls"] == 0.0 and dg["frames"]["edge_only"]["handoff"] == 0.0
              and dg["thresholds"]["t_high"] == dg["policy"]["t_high"])
        check("the comparison is paired over the same units",
              all(dg["risks"][k]["cascade"]["n_units"] == dg["risks"][k]["edge_only"]["n_units"]
                  == dg["paired_unit_differences"][k]["n_units"] for k in ("fire", "smoke")))
        check("the diagnostic labels its interval post-selection",
              all("POST-SELECTION" in dg["paired_unit_differences"][k]["kind"]
                  for k in ("fire", "smoke")))
        (P / "protocol.yaml").write_text((P / "protocol.yaml").read_text() + "\n# touched\n")
        ok4, o4 = run("cascade.ace2_dev", *args, "--registration", reg, "--out", str(root / "run3"))
        check("a changed protocol makes the registered run refuse",
              not ok4 and "protocol_sha256" in o4, o4[-200:])


def test_tree_identity():
    """MANIFEST.sha256 must describe the INSTALLED tree: an overlay unzip that leaves a stale module behind
    keeps the code hash current while the runtime tree differs, and only this check sees it."""
    import shutil
    from cascade.make_protocol import verify_tree, write_manifest
    r = verify_tree()
    check("runtime tree matches MANIFEST.sha256", r["ok"], f"extra={r['extra']} missing={r['missing']} "
                                                           f"changed={r['changed']}")
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "cascade"
        shutil.copytree(ROOT / "cascade", pkg, ignore=shutil.ignore_patterns("__pycache__"))
        write_manifest(pkg)
        check("fresh manifest verifies", verify_tree(pkg)["ok"])
        # runtime artefacts are ignored explicitly; anything NOT on that list must still fail
        (pkg / "__pycache__").mkdir()
        (pkg / "__pycache__" / "risk.cpython-311.pyc").write_bytes(b"\x00cache")
        (pkg / "risk.pyo").write_bytes(b"\x00cache")
        (pkg / ".DS_Store").write_bytes(b"\x00")
        check("runtime cache is not reported as a stale file", verify_tree(pkg)["ok"],
              f"extra={verify_tree(pkg)['extra']}")
        (pkg / "stale_module.py").write_text("# a file the new package no longer ships\n")
        check("stale file left by an overlay unzip is detected", verify_tree(pkg)["extra"] == ["stale_module.py"])
        (pkg / "stale_module.py").unlink()
        (pkg / "risk.py").write_text((pkg / "risk.py").read_text() + "\n")
        check("edited file is detected", verify_tree(pkg)["changed"] == ["risk.py"])
        (pkg / "MANIFEST.sha256").unlink()
        check("missing manifest is not silently OK", verify_tree(pkg)["ok"] is False)


def main():
    test_run_all()
    test_ravc()
    test_esva()
    test_ace()
    test_ace_registration()
    test_ace_end_to_end()
    test_tree_identity()
    test_statistics()
    test_head_rule()
    test_protocol()
    test_code_lock()
    n_fail = sum(not ok for _, ok in RESULTS)
    if SKIPPED:
        print(f"\nskipped (environment): {SKIPPED}")
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed" + (f", {len(SKIPPED)} skipped" if SKIPPED else ""))
    print("SELFTEST OK" if n_fail == 0 else f"SELFTEST FAILED ({n_fail})")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
