"""Step 4 (CPU): risk-controlled edge-agent cascade experiments.

Protocol mode (settings come from protocol.yaml, never from the command line):
    --mode dev        repeated random unit splits inside the dev role (val); the only place to compare variants
    --mode validity   ONLY AFTER `freeze` (nothing may change afterwards): repeated random unit splits inside the
                      calibration role (sealed_test untouched); certification rate, empirical risk distribution
                      and the cert-and-exceeds DIAGNOSTIC
    --mode final      once, after `make_protocol freeze`: calibrate each method on the whole calibration role,
                      evaluate once on sealed_test (+ unit-bootstrap CI) and on external_shift. Fail-closed checks:
                      valid lock, records built with --unseal from exactly the locked detector/agent files,
                      every split UID present.

Statistics self-tests (no data needed):
    --synthetic               repeated splits on simulated scores
    --synthetic-validity      the LTT guarantee itself: in a large simulated population the TRUE risk of each
                              certified configuration is known, so P(certified and true risk > alpha) is compared
                              with delta (Clopper-Pearson interval)

Legacy mode (no --protocol): repeated splits over --pool splits.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.experiment import (RISK_KEY, _take, data_for, evaluate, format_table, methods,  # noqa: E402
                                risk_value, run_trials, shift_loso, summarize)
from cascade.policy import Grid, unit_losses  # noqa: E402

MAIN = ("edge_LTT", "cascade_LTT", "cascade_LTT+human")


def load_records(path):
    with open(Path(path).expanduser(), newline="", encoding="utf-8") as f:
        R = list(csv.DictReader(f))

    def num(k):
        return np.array([float(r[k]) if r.get(k, "") not in ("", None) else np.nan for r in R])
    d = {k: np.array([r.get(k, "") for r in R]) for k in ("uid", "source", "split", "role", "unit", "event", "tag")}
    d.update({k: num(k) for k in ("y", "y_fire", "y_smoke", "s", "g", "g_plain", "nbytes_frame", "nbytes_crop",
                                  "det_ms", "agent_ms", "agent_ms_plain")})
    return d


def finalize(d):
    """Cast labels to int (external rows have y_fire = y_smoke = -1 -> 0)."""
    d = dict(d)
    for k in ("y", "y_fire", "y_smoke"):
        d[k] = np.where(np.isfinite(d[k]) & (d[k] > 0), 1, 0).astype(int)
    return d


def parse_costs(s):
    return {k: float(v) for k, v in (p.split("=") for p in s.split(",") if p)}


def md_table(rows, cols, fmt="{:.4f}"):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r.get(c), (float, np.floating))
                                         else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def clopper_pearson(k, n, conf=0.95):
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def e2e_ms(name, calls, cfg):
    """Mean end-to-end time per frame. Pure cloud: no detector, every frame goes to the plain agent."""
    if name == "pure_cloud_plain":
        return cfg.get("plain_ms", float("nan"))
    return cfg["det_ms"] + calls * cfg["agent_ms"]


def require_labels(d, what):
    if not np.isfinite(d["y"]).all():
        raise SystemExit(f"{what}: some labels are withheld in these records (wrong stage / not unsealed)")


def mkw_of(cfg):
    return dict(fwer=cfg["fwer"], select_fraction=cfg["select_fraction"], grid=cfg.get("grid"),
                fixed=cfg.get("fixed"), all_cloud=cfg.get("all_cloud", False), pure_cloud=cfg.get("pure_cloud", False))


def repeated(pool, cfg, out, tag, trials, seed, do_loso=True):
    """Repeated random unit splits (50/50) inside `pool`: tables, per-source view, LOSO."""
    target, risks, payload = cfg["target"], cfg["risks"], cfg["payload"]
    rows = run_trials(pool, trials, 0.5, cfg["alpha"], cfg["delta"], cfg["costs"], cfg["s_min"], seed,
                      target=target, payload=payload, by="source", risks=risks, **mkw_of(cfg))
    summ = summarize(rows, cfg["alpha"], target)
    for name, v in summ.items():
        v["e2e_ms_mean"] = e2e_ms(name, v["calls_mean"], cfg)
    extra = tuple(f"{k}:{RISK_KEY[target]}_mean" for k in (risks or [])) + ("e2e_ms_mean",)
    table = format_table(summ, cfg["alpha"], target, extra_cols=extra)
    print(table)
    (out / f"{tag}_table.md").write_text(table + "\n")
    (out / f"{tag}_summary.json").write_text(json.dumps(jsonable(summ), indent=1))
    keys = ["method", "trial", "theta", "certified", "risk", "miss_img", "miss_unit", "miss_event", "fa", "calls",
            "handoff", "bytes_per_frame", "n_pos", "n_units", "n_events"] + \
        [f"{k}:{RISK_KEY[target]}" for k in (risks or [])]
    with open(out / f"{tag}_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    src_rows = []
    for name in MAIN:
        R = [r for r in rows if r["method"] == name]
        for src in sorted({k for r in R for k in r["by"]}):
            vals = [r["by"][src] for r in R if src in r["by"]]
            rv = [risk_value(v, target, risks) for v in vals]
            src_rows.append({"method": name, "source": src, "trials": len(vals), "risk_mean": float(np.mean(rv)),
                             "fa": float(np.mean([v["fa"] for v in vals])),
                             "calls": float(np.mean([v["calls"] for v in vals]))})
    (out / f"{tag}_by_source.json").write_text(json.dumps(jsonable(src_rows), indent=1))
    (out / f"{tag}_by_source.md").write_text(
        "Pooled calibration; the guarantee is for the pool, not for each source.\n\n" +
        md_table(src_rows, ["method", "source", "trials", "risk_mean", "fa", "calls"]) + "\n")
    if do_loso and len(np.unique(pool["source"])) > 1:
        lo = shift_loso(pool, cfg["alpha"], cfg["delta"], cfg["costs"], cfg["s_min"], "source", target, payload,
                        MAIN, risks=risks, **mkw_of(cfg))
        lrows = [dict(v, theta=tuple(round(float(x), 4) for x in v["theta"])) for v in lo.values()]
        (out / f"{tag}_loso.md").write_text(
            "Calibrated without the held-out source (exchangeability broken on purpose).\n\n" +
            md_table(lrows, ["held_out", "method", "risk", "fa", "calls", "certified", "theta"]) + "\n")


def cluster_bootstrap(test, theta, cfg, reps, seed=0, chunk=250):
    """Unit (cluster) bootstrap of the sealed-test risk: resample test units with replacement; each risk is the
    mean of per-unit losses over resampled units that contain positives of that kind. Returns 95% percentile CIs
    for the controlled (max) risk and for each constrained risk separately."""
    target, keys = cfg["target"], (cfg["risks"] or ["y"])
    ev = test["event"] if target == "event" else None
    units = np.unique(test["unit"])
    pos = {u: i for i, u in enumerate(units)}
    L = np.full((len(keys), len(units)), np.nan)
    for j, k in enumerate(keys):
        yk = np.asarray(test[k]) > 0
        if yk.any():
            u, loss = unit_losses(test["s"], test["g"], yk, theta, test["unit"], ev)
            L[j, [pos[x] for x in u]] = loss
    rng = np.random.default_rng(seed)
    draws = []
    for b0 in range(0, reps, chunk):
        idx = rng.integers(0, len(units), (min(chunk, reps - b0), len(units)))
        Lb = L[:, idx]                                     # keys x reps x units
        cnt = np.isfinite(Lb).sum(-1)
        draws.append(np.where(cnt > 0, np.nansum(Lb, -1) / np.maximum(cnt, 1), np.nan))
    D = np.concatenate(draws, axis=1)                      # keys x reps
    ci = {k: (float(np.nanquantile(D[j], 0.025)), float(np.nanquantile(D[j], 0.975))) for j, k in enumerate(keys)}
    mx = np.nanmax(D, axis=0)
    ci["max"] = (float(np.quantile(mx, 0.025)), float(np.quantile(mx, 0.975)))
    return ci


def final(cal, test, ext, cfg, out, lock_sha, note=""):
    """Calibrate once on `cal`, evaluate once on `test` (sealed) and on `ext`."""
    target, risks, payload = cfg["target"], cfg["risks"], cfg["payload"]
    M = methods(cfg["alpha"], cfg["delta"], cfg["costs"], cfg["s_min"], target, payload, risks=risks,
                seed=cfg["calibrate_seed"], **mkw_of(cfg))
    rows, src_rows, ext_rows, res = [], [], [], {}
    ext0 = ext
    for name, f in M.items():
        theta, info = f(cal)
        test_n = data_for(name, test)
        ext = data_for(name, ext0) if ext0 is not None else None
        r = evaluate(test_n, theta, payload, risks)
        r.update(method=name, risk=risk_value(r, target, risks), certified=info.get("certified"),
                 theta=tuple(round(float(x), 4) for x in theta),
                 e2e_ms=e2e_ms(name, r["calls"], cfg))
        ci = cluster_bootstrap(test_n, theta, cfg, cfg["bootstrap_reps"])
        r["risk_ci95"] = tuple(round(x, 4) for x in ci["max"])
        for k in (risks or []):
            r[f"{k}:ci95"] = tuple(round(x, 4) for x in ci[k])
        rows.append(r)
        res[name] = {"theta": theta, "calibration": info, "sealed_test": r}
        for src in np.unique(test_n["source"]):
            m = test_n["source"] == src
            if test_n["y"][m].any():
                rs = evaluate(_take(test_n, m), theta, payload, risks)
                src_rows.append({"method": name, "source": str(src), "risk": risk_value(rs, target, risks),
                                 "fa": rs["fa"], "calls": rs["calls"]})
        if ext is not None and len(ext["y"]):
            keys = sorted(set(zip(ext["source"], ext["tag"])))
            for key in [None, *keys]:
                m = np.ones(len(ext["y"]), bool) if key is None else (ext["source"] == key[0]) & (ext["tag"] == key[1])
                if not m.any():
                    continue
                re_ = evaluate(_take(ext, m), theta, payload)
                ext_rows.append({"method": name, "subset": "all" if key is None else f"{key[0]}/{key[1] or '-'}",
                                 "n_pos": re_["n_pos"], "n_neg": re_["n_neg"], "miss_img": re_["miss_img"],
                                 "fa": re_["fa"], "calls": re_["calls"]})
    cols = ["method", "certified", "risk", "risk_ci95"] + \
           [c for k in (risks or []) for c in (f"{k}:{RISK_KEY[target]}", f"{k}:ci95")] + \
           ["miss_img", "fa", "calls", "handoff", "bytes_per_frame", "e2e_ms", "theta"]
    txt = (f"FINAL (sealed test, evaluated once). protocol lock {lock_sha}\n" + (note + "\n" if note else "") +
           f"alpha={cfg['alpha']} delta={cfg['delta']} target={target} risks={risks}\n"
           "Guarantee: P(certifying a configuration whose true risk > alpha) <= delta over the calibration draw. "
           f"The sealed-test risk below is an empirical estimate with a {cfg['bootstrap_reps']}-replicate unit (cluster) "
           "bootstrap 95% CI, for the controlled max risk and for each constrained risk.\n\n" + md_table(rows, cols) +
           "\n\n## by source (pooled guarantee, not per source)\n\n" +
           md_table(src_rows, ["method", "source", "risk", "fa", "calls"]))
    if ext_rows:
        txt += ("\n\n## external shift (El-Madafri fog/haze + sources with too few test units, e.g. FLAME2;"
                " frame-level any-positive labels, no guarantee claimed)\n\n" +
                md_table(ext_rows, ["method", "subset", "n_pos", "n_neg", "miss_img", "fa", "calls"]))
    print(txt)
    (out / "final.md").write_text(txt + "\n")
    (out / "final.json").write_text(json.dumps(jsonable({"lock_sha256": lock_sha, "config": cfg, "results": res,
                                                         "method_order": list(M), "by_source": src_rows,
                                                         "external_shift": ext_rows, "note": note}), indent=1))


def synthetic_validity(out, trials, n_cal_units, alpha, delta, seed, targets):
    """True-risk check of the guarantee. Population: 60k simulated units; each trial calibrates on n_cal_units
    and evaluates the chosen theta on the remaining population (~ the true risk)."""
    from cascade.simulate import make
    pop = make(n_units=60000, seed=seed)
    pop["nbytes"] = pop["nbytes"].astype(float)
    units = np.unique(pop["unit"])
    rng = np.random.default_rng(seed + 1)
    lines = []
    for target in targets:
        M = methods(alpha, delta, None, 0.02, target, "nbytes")
        keep = ("edge_LTT", "edge_LTT_image", "cascade_LTT_image", "cascade_LTT_bonf", "cascade_LTT")
        stats = {k: {"cert": 0, "bad": 0, "true": []} for k in keep}
        for t in range(trials):
            cu = set(rng.choice(units, n_cal_units, replace=False))
            cm = np.array([u in cu for u in pop["unit"]])
            cal, rest = _take(pop, cm), _take(pop, ~cm)
            for name in keep:
                theta, info = M[name](cal)
                r = evaluate(rest, theta, "nbytes")
                tr = r[RISK_KEY[target]]
                c = bool(info.get("certified"))
                stats[name]["cert"] += c
                stats[name]["bad"] += c and tr > alpha
                stats[name]["true"].append(tr)
            if (t + 1) % max(1, trials // 5) == 0:
                print(f"  [{target}] trial {t + 1}/{trials}", flush=True)
        lines.append(f"\n### target = {target} ({RISK_KEY[target]}), {n_cal_units} calibration units, "
                     f"{trials} trials, alpha={alpha}, delta={delta}\n")
        lines.append("| method | certification_rate | P(certified and TRUE risk > alpha) | 95% CI | "
                     "verdict (CP upper <= delta) | true risk mean |\n|---|---|---|---|---|---|")
        for name, s in stats.items():
            lo, hi = clopper_pearson(s["bad"], trials)
            ok = "supported" if hi <= delta else ("VIOLATED" if lo > delta else "inconclusive")
            lines.append(f"| {name} | {s['cert'] / trials:.3f} | {s['bad'] / trials:.3f} | [{lo:.3f}, {hi:.3f}] | "
                         f"{ok} | {np.mean(s['true']):.4f} |")
    txt = ("Synthetic check of the LTT statement with known (population) risk. Image-level variants assume "
           "i.i.d. frames, which the simulation violates on purpose (correlated frames within units).\n" +
           "\n".join(lines))
    print(txt)
    (out / "synthetic_validity.md").write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", help="records.csv from cascade.build_records")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-validity", action="store_true")
    ap.add_argument("--cal-units", type=int, default=1500, help="synthetic-validity: calibration units per trial")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--mode", choices=["dev", "validity", "final"], default=None)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--resume", action="store_true", help="final only: resume a crashed final (same --out-dir)")
    ap.add_argument("--seed", type=int, default=0)
    # legacy / synthetic settings (ignored in protocol mode)
    ap.add_argument("--pool", nargs="+", default=["val", "test"])
    ap.add_argument("--unit-col", choices=["unit", "event"], default="unit")
    ap.add_argument("--targets", nargs="+", default=["frame", "event"], choices=["frame", "event"])
    ap.add_argument("--risks", nargs="*", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--s-min", type=float, default=0.02)
    ap.add_argument("--payload", choices=["frame", "crop"], default="frame")
    ap.add_argument("--costs", default="calls=1,handoff=20,fa=10")
    a = ap.parse_args()
    out = Path(a.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if a.synthetic_validity:
        synthetic_validity(out, a.trials, a.cal_units, a.alpha, a.delta, a.seed, a.targets)
        return
    if a.synthetic:
        from cascade.simulate import make
        d = make(n_units=3000, seed=a.seed)
        d["nbytes_frame"] = d["nbytes_crop"] = d.pop("nbytes").astype(float)
        cfg = dict(alpha=a.alpha, delta=a.delta, costs=parse_costs(a.costs), s_min=a.s_min, payload="nbytes_frame",
                   risks=None, det_ms=5.0, agent_ms=300.0, fwer="pareto", select_fraction=0.3)
        for target in a.targets:
            repeated(d, dict(cfg, target=target), out, f"synthetic_{target}", a.trials, a.seed, do_loso=False)
        return

    D = load_records(a.records)
    det_ms = float(np.nanmean(D["det_ms"]))
    agent_ms = float(np.nanmean(D["agent_ms"])) if np.isfinite(D["agent_ms"]).any() else float("nan")
    rmeta_p = Path(a.records).expanduser().with_suffix(".meta.json")
    rmeta = json.loads(rmeta_p.read_text()) if rmeta_p.exists() else {}

    if a.protocol:
        from cascade.make_protocol import check_lock, load_protocol, lock_status, read_lock
        P, splits = load_protocol(a.protocol)
        if not a.mode:
            raise SystemExit("--protocol needs --mode dev|validity|final")
        if not rmeta or Path(rmeta.get("protocol") or "") != Path(a.protocol).expanduser().resolve():
            raise SystemExit("records were not built with this --protocol (records .meta.json missing/mismatch)")
        cfg = dict(alpha=P["alpha"], delta=P["delta"], target=P["target"], risks=P.get("risks"),
                   costs={k: float(v) for k, v in P["costs"].items()}, s_min=float(P["s_min"]),
                   payload="nbytes_" + P.get("payload", "frame"), fwer=P.get("fwer", "pareto"),
                   select_fraction=float(P.get("select_fraction", 0.3)), calibrate_seed=int(P.get("calibrate_seed", 1)),
                   grid=Grid.from_dict(P["grid"]) if P.get("grid") else None,
                   fixed=P.get("fixed_baselines"), bootstrap_reps=int(P.get("bootstrap_reps", 5000)),
                   all_cloud=bool(P.get("all_frames_cloud_overlay", False)),
                   pure_cloud=bool(P.get("pure_cloud_baseline", False)), det_ms=det_ms, agent_ms=agent_ms,
                   plain_ms=float(np.nanmean(D["agent_ms_plain"])) if np.isfinite(D["agent_ms_plain"]).any()
                   else float("nan"), mode=a.mode)
        role = D["role"]
        if cfg["all_cloud"] and not np.isfinite(D["g"]).all():
            raise SystemExit("all_frames_cloud_overlay needs agent scores for every frame (agent --s-min 0)")
        if cfg["pure_cloud"] and not np.isfinite(D["g_plain"]).all():
            raise SystemExit("pure_cloud_baseline needs plain-agent scores for every frame (records --agent-plain)")
        L = read_lock(a.protocol)

        def lock_problems():
            pr = []
            if not check_lock(a.protocol):
                pr.append("protocol lock missing or does not match: " + "; ".join(lock_status(a.protocol)))
                return pr
            if rmeta.get("lock_sha256") != L["sha256"]:
                pr.append("records were built before / under a different lock")
            for k in ("det_csv_sha256", "agent_csv_sha256", "det_meta_sha256", "agent_meta_sha256"):
                if rmeta.get(k) != L.get(k):
                    pr.append(f"records {k} differs from the lock")
            if L.get("agent_plain_csv_sha256") and rmeta.get("agent_plain_csv_sha256") != L["agent_plain_csv_sha256"]:
                pr.append("records agent_plain_csv_sha256 differs from the lock")
            from cascade.make_protocol import sha256_file
            if rmeta.get("records_sha256") != sha256_file(a.records):
                pr.append("records.csv changed after it was built")
            return pr
        if a.mode == "validity":
            pr = lock_problems()
            if pr:
                raise SystemExit("validity refused (run it only after freeze, on post-freeze records):\n  "
                                 + "\n  ".join(pr))
        if a.mode in ("dev", "validity"):
            raw = _take(D, role == ("dev" if a.mode == "dev" else "calibration"))
            require_labels(raw, a.mode)
            pool = finalize(raw)
            repeated(pool, cfg, out, a.mode, a.trials, a.seed)
            return
        # ---- final: fail-closed checks --------------------------------------------------------------------
        problems = lock_problems()
        if not rmeta.get("unsealed"):
            problems.append("records were built without --unseal")
        if set(D["uid"]) != set(splits):
            problems.append("records UIDs differ from splits.csv")
        if not np.isfinite(D["s"]).all():
            problems.append("missing detector scores")
        esc = D["s"] >= cfg["s_min"]
        if (esc & ~np.isfinite(D["g"])).any():
            problems.append("frames with s >= s_min lack agent scores")
        if problems:
            raise SystemExit("final refused:\n  " + "\n  ".join(problems))
        for rl in ("calibration", "sealed_test", "external_shift"):
            if (role == rl).any():
                require_labels(_take(D, role == rl), f"final/{rl}")
        if (out / "final.md").exists() and not a.resume:
            raise SystemExit(f"{out/'final.md'} exists: the sealed test is evaluated once.")
        from cascade.make_protocol import begin_final, finish_final, receipt_summary, sha256_file
        # protocol receipt: only the records hash recorded at --unseal is accepted; unsealed -> running
        R = begin_final(a.protocol, out, sha256_file(a.records), L["sha256"], resume=a.resume)
        final(finalize(_take(D, role == "calibration")), finalize(_take(D, role == "sealed_test")),
              finalize(_take(D, role == "external_shift")) if (role == "external_shift").any() else None,
              cfg, out, L["sha256"], note=receipt_summary(R))
        if os.environ.get("CASCADE_SELFTEST_CRASH_FINAL") == "1":     # selftest only: crash before 'done'
            raise SystemExit("simulated crash (selftest)")
        R = finish_final(a.protocol, out / "final.md")
        print(f"protocol receipt: done, final.md sha256 {R['final_md_sha256']}")
        return

    # legacy mode
    keep = np.isin(D["split"], a.pool) & np.isfinite(D["y"])
    if (np.isin(D["split"], a.pool) & ~np.isfinite(D["y"])).any():
        print("note: rows with withheld labels (sealed_test) are excluded")
    pool = finalize(_take(D, keep))
    if a.unit_col == "event":
        pool["unit"] = pool["event"].copy()
    cfg = dict(alpha=a.alpha, delta=a.delta, costs=parse_costs(a.costs), s_min=a.s_min,
               payload="nbytes_" + a.payload, risks=a.risks, det_ms=det_ms, agent_ms=agent_ms,
               fwer="pareto", select_fraction=0.3)
    for target in a.targets:
        repeated(pool, dict(cfg, target=target), out, f"legacy_{target}", a.trials, a.seed)


if __name__ == "__main__":
    main()
