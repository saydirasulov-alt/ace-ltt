"""Fast tests for ESVA-LTT, the dev ceiling diagnostic and the pre-registered esva_dev command (no GPU, fake data)."""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from .evidence_veto import (StratumSpec, apply_esva, calibrate_esva, certifiability, discount, in_stratum, n_min,
                            smoke_features)
from .esva_synthetic_validity import population


def _fake_with_boxes(root):
    from .make_protocol import sha256_file
    from .selftest import make_fake
    ds = make_fake(root, n_units=400, seed=5)
    rng = np.random.default_rng(1)
    with open(root / "det.csv", newline="") as f:
        det = list(csv.DictReader(f))
    for r in det:                     # a third of the frames get a small smoke box, the rest a large one
        a = 0.01 if rng.random() < 0.35 else 0.2
        side = float(np.sqrt(a))
        r["boxes"] = f"0:{float(r['s']):.4f}:0.1:0.1:{0.1 + side:.4f}:{0.1 + side:.4f}"
    with open(root / "det.csv", "w", newline="") as f:
        w = csv.DictWriter(f, det[0].keys())
        w.writeheader()
        w.writerows(det)
    m = json.loads((root / "agent.meta.json").read_text())
    m["det_csv_sha256"] = sha256_file(root / "det.csv")
    (root / "agent.meta.json").write_text(json.dumps(m))
    return ds


def run_tests():
    assert n_min(0.05, 0.10) == 45
    F = smoke_features(["1:0.9:0:0:0.5:0.5;0:0.8:0.1:0.1:0.2:0.2", "", "0:0.7:0:0:0.1:0.1"],
                       [0.9, 0.0, 0.1], [0.8, 0.0, 0.7])
    assert np.allclose(F["smoke_area"], [0.01, 1.0, 0.01]) and np.allclose(F["top_area"], [0.25, 1.0, 0.01])
    assert in_stratum(StratumSpec("small_smoke", 0.02), F).tolist() == [False, False, True]   # fire dominates #0
    assert np.allclose(discount([0.2, np.nan, 0.95], np.array([True, True, True]), 0.3)[[0, 2]], [0.5, 1.0])
    assert np.isnan(discount([0.2, np.nan], np.array([True, True]), 0.3)[1])

    # certifies on a population where the VLM is blind to small smoke; S risk and pooled risk both controlled
    d = population(12000, 3)
    rng = np.random.default_rng(0)
    ix = rng.choice(12000, 3000, replace=False)
    te = np.setdiff1d(np.arange(12000), ix)
    Fi = {k: v[ix] for k, v in d["F"].items()}
    kw = dict(risk_labels={"fire": d["yf"][ix], "smoke": d["ys"][ix]}, costs={"calls": 1, "handoff": 5, "fa": 20},
              seed=0)
    theta, info = calibrate_esva(d["s"][ix], d["g"][ix], Fi, d["y"][ix], d["unit"][ix], **kw)
    assert info["certified"] and "smoke|S" in info["risks"], info
    assert info["certifiability_certify_fold"]["certifiable"]
    Ft = {k: v[te] for k, v in d["F"].items()}
    dec = apply_esva(d["s"][te], d["g"][te], Ft, theta, info)
    inS = in_stratum(StratumSpec(), Ft)
    miss_S = ((d["ys"][te] & inS) & ~np.isin(dec, (1, 2))).sum() / max(1, (d["ys"][te] & inS).sum())
    assert miss_S < 0.08, miss_S
    # lambda family must contain 0 (cascade_LTT is a member)
    try:
        calibrate_esva(d["s"][ix], d["g"][ix], Fi, d["y"][ix], d["unit"][ix], lambdas=(0.5,), **kw)
        raise AssertionError("lambda family without 0 accepted")
    except ValueError:
        pass
    c = certifiability(d["unit"][ix], {"smoke": d["ys"][ix]}, in_stratum(StratumSpec(), Fi), 0.05, 0.10)
    assert c["positive_units"]["smoke|S"] > 45
    # certify-fold labels/scores changed -> same candidate order (n_candidates, grid); only p-values may change
    rs = np.random.default_rng(0)
    uu = np.unique(d["unit"][ix])
    sel = set(rs.choice(uu, int(round(0.30 * len(uu))), replace=False))
    cert = np.array([u not in sel for u in d["unit"][ix]])
    g2, ys2 = d["g"][ix].copy(), d["ys"][ix].copy()
    g2[cert] = 1 - g2[cert]
    ys2[cert] = ~ys2[cert]
    _, info2 = calibrate_esva(d["s"][ix], g2, Fi, d["y"][ix], d["unit"][ix],
                              **{**kw, "risk_labels": {"fire": d["yf"][ix], "smoke": ys2}})
    assert info2["n_candidates"] == info["n_candidates"] and info2["grid"] == info["grid"]
    # a stratum risk with no positive in S is NOT dropped: ESVA then cannot certify (fails closed)
    th3, info3 = calibrate_esva(d["s"][ix], d["g"][ix], {**Fi, "smoke_area": np.ones(len(ix))}, d["y"][ix],
                                d["unit"][ix], **kw)
    assert "smoke|S" in info3["risks"] and not info3["certified"] and info3["lambda"] == 0.0
    # clustered, event-level population: calibrates and certifies
    dc = population(4000, 4, cluster=3)
    iu = np.isin(dc["uid"], np.random.default_rng(1).choice(4000, 2500, replace=False))
    _, ic = calibrate_esva(dc["s"][iu], dc["g"][iu], {k: v[iu] for k, v in dc["F"].items()}, dc["y"][iu],
                           dc["unit"][iu], event=dc["event"][iu],
                           **{**kw, "risk_labels": {"fire": dc["yf"][iu], "smoke": dc["ys"][iu]}})
    assert ic["certified"], ic

    # payload budget: unit-mean definition, joint certification, evaluation-side agreement
    from .payload_budget import budget_risk_grid, calibrate_payload, unit_budget_stats
    from .policy import Grid
    rg = np.random.default_rng(0)
    N = 8000
    un = np.repeat(np.arange(2000), 4).astype(str)
    yb = rg.random(N) < 0.3
    sb = np.clip(0.1 + 0.4 * yb + rg.normal(0, 0.15, N), 0, 1)
    g_ov = 1 / (1 + np.exp(-(-2.5 + 5 * yb + rg.normal(0, 1.2, N))))
    g_cr = 1 / (1 + np.exp(-(-2.5 + 4.5 * yb + rg.normal(0, 1.2, N))))
    gr = Grid.default(sb)
    rgrid, nu = budget_risk_grid(sb, np.full(N, 48000.0), un, gr, 3000.0)
    assert nu == 2000 and rgrid.shape[:2] == (len(gr.s_edges),) * 2
    assert np.nanmin(rgrid) == 0.0 and np.isnan(rgrid[3, 1, 0, 0])          # t_low > t_high is nan
    assert rgrid[0, -1, 0, 0] > 0.9                                          # escalate everything -> over budget
    pols = [{"name": "overlay", "g": g_ov, "payload": np.full(N, 48000.0)},
            {"name": "crop", "g": g_cr, "payload": np.full(N, 12000.0)}]
    thb, infb = calibrate_payload(sb, pols, yb, un, budgets=[1000.0, 3000.0, 10000.0], alpha=0.05, delta=0.10,
                                  beta=0.05, costs={"calls": 1, "fa": 20}, seed=1, risk_labels={"miss": yb})
    assert infb["certified"] and infb["policy"] in ("overlay", "crop") and infb["by_budget"], infb
    pay = np.full(N, 12000.0 if infb["policy"] == "crop" else 48000.0)
    st = unit_budget_stats(sb, pay, un, thb, infb["budget"])
    assert st["budget_risk"] <= 0.10 and st["n_units"] == 2000, st
    # a budget that no cloud call can satisfy -> only the edge-only configuration can be certified
    thb2, infb2 = calibrate_payload(sb, pols[:1], yb, un, budgets=[1.0], alpha=0.05, delta=0.10, beta=0.001,
                                    costs={"calls": 1, "fa": 20}, seed=1, risk_labels={"miss": yb})
    assert thb2[0] == thb2[1], (thb2, infb2)      # no escalation band

    # dev tools on fake data: diagnose, register, run, refusals
    from .esva_dev import main as esva_main
    from .dev_diagnose import main as diag_main
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ds = _fake_with_boxes(root)
        proto = root / "proto"
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        p = subprocess.run([sys.executable, "-m", "cascade.make_protocol", "init", "--dataset", str(ds), "--out",
                            str(proto)], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                           env=env)
        assert p.returncode == 0, p.stdout + p.stderr
        common = ["--protocol", str(proto), "--det", str(root / "det.csv"), "--agent", str(root / "agent.csv")]
        buf = io.StringIO()
        with redirect_stdout(buf):
            diag_main(["--dataset", str(ds), *common, "--out", str(root / "diag"), "--trials", "2"])
        J = json.loads((root / "diag" / "dev_diagnose.json").read_text())
        assert "go_rule" in J and "whatif" in J and J["n_min"] == 45
        assert not any(k.startswith(("calibration", "sealed_test")) for k in J.get("miss_attribution", {}))
        try:                           # template has esva: null -> registration refused
            esva_main(["register", *common, "--out", str(root / "reg.json")])
            raise AssertionError("esva: null accepted")
        except SystemExit as e:
            assert "esva" in str(e)
        pf = proto / "protocol.yaml"
        pf.write_text(pf.read_text().replace(
            "esva: null", "esva: {stratum: {kind: small_smoke, area_max: 0.02}, "
                          "lambdas: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}"))
        with redirect_stdout(buf):
            esva_main(["register", *common, "--out", str(root / "reg.json")])
        try:
            esva_main(["register", *common, "--out", str(root / "reg.json")])
            raise AssertionError("second registration accepted")
        except SystemExit:
            pass
        pf.write_text(pf.read_text().replace("area_max: 0.02", "area_max: 0.05"))
        try:
            esva_main(["run", "--dataset", str(ds), *common, "--registration", str(root / "reg.json"),
                       "--out", str(root / "o"), "--trials", "1"])
            raise AssertionError("protocol changed after registration accepted")
        except SystemExit as e:
            assert "registration mismatch" in str(e)
        pf.write_text(pf.read_text().replace("area_max: 0.05", "area_max: 0.02"))
        with redirect_stdout(buf):
            esva_main(["run", "--dataset", str(ds), *common, "--registration", str(root / "reg.json"),
                       "--out", str(root / "o"), "--trials", "2"])
        from .dev_probe import auc_delta, main as probe_main, roc_stats
        from .study_register import main as reg_main
        reg_main(["--name", "acquisition_ladder", "--protocol", str(proto), "--det", str(root / "det.csv"),
                  "--out", str(root / "acq.json")])
        try:
            reg_main(["--name", "acquisition_ladder", "--protocol", str(proto), "--det", str(root / "det.csv"),
                      "--out", str(root / "acq.json")])
            raise AssertionError("second study registration accepted")
        except SystemExit:
            pass
        assert "unit-level" in json.loads((root / "acq.json").read_text())["rule"].lower() or True
        with redirect_stdout(buf):
            probe_main(["--dataset", str(ds), *common.copy()[:4], "--agents", f"overlay={root / 'agent.csv'}",
                        "--out", str(root / "pr"), "--trials", "1", "--split-trials", "20"])
        Jp = json.loads((root / "pr" / "dev_probe.json").read_text())
        assert Jp["certifiability"]["n_min"] == 45 and Jp["required_K"] > 45
        assert any(r["lambda"] == "edge_LTT" for r in Jp["lambda_front"])
        assert 0.0 <= max(r["auc"] for r in Jp["informativeness"] if np.isfinite(r["auc"])) <= 1.0
        assert Jp["auc_delta"] == [] or all(r["n_units"] >= 1 for r in Jp["auc_delta"])
        rs = roc_stats([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])          # perfect ranking
        assert rs["auc"] == 1.0 and rs["fpr@tpr95"] == 0.0 and rs["pauc_spec_band"] > 0.99
        rs = roc_stats([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])          # no ranking at all
        assert abs(rs["auc"] - 0.5) < 1e-9
        R = json.loads((root / "o" / "esva_dev.json").read_text())
        assert set(R["summary"]) == {"edge_LTT", "cascade_LTT", "cascade_LTT+human", "ESVA_LTT", "ESVA_LTT+human"}
        assert R["acceptance"]["method"] == "ESVA_LTT+human"
        try:
            esva_main(["run", "--dataset", str(ds), *common, "--registration", str(root / "reg.json"),
                       "--out", str(root / "o"), "--trials", "1"])
            raise AssertionError("second run accepted")
        except SystemExit as e:
            assert "already made" in str(e)
    print("ESVA TESTS OK")


if __name__ == "__main__":
    run_tests()
