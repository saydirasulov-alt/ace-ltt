"""Freeze split roles and the analysis protocol BEFORE any labelled calibration / sealed result is looked at.

    python -m cascade.make_protocol init   --dataset <ds> --out <P>          # label-free
    (dev work; edit <P>/protocol.yaml)
    python -m cascade.make_protocol freeze --out <P> --det <det.csv> --agent <agent.csv>

Roles (per unit = split_group; frames keep their dataset split):
    train           unchanged (the detector was trained on it)
    dev             current val: all variant choices (head, VLM, view, costs, grid) are made here
    calibration     half of the current test units, stratified by source. Inside it the calibrator uses a
                    'select' part (select_fraction of units: grid + Pareto order) and a disjoint 'certify' part
    sealed_test     the other half; its labels never enter records before the lock exists
    external_shift  fog / haze external set (image-level labels; no guarantee claimed)

init prints only label-free counts for calibration / sealed_test (positive counts are shown for dev only).
freeze
  * checks the dataset manifest hash against protocol.yaml,
  * checks that the detector / agent .meta.json match the deployment fields of protocol.yaml
    (nms, imgsz, conf, iou, max_det, precision, ultralytics; model, quant, view, max_side, s_min) and that the
    agent was run on exactly this detector output; fills weights_sha256 / revision / prompt_sha256 if empty,
  * checks that the agent was run on exactly this detector CSV (content hash),
  * writes protocol.lock = sha256(protocol.yaml, splits.csv, detector CSV + meta, agent CSV + meta
    [, plain agent CSV + meta], analysis_code_sha256). Any later change to any of these files -- including one
    byte of the analysis code (ANALYSIS_CODE below) -- invalidates the lock.

Protocol receipt (<P>/final_receipt.json), a one-way state machine:
    none -> unsealing -> unsealed(records_sha256) -> running -> done
  * `build_records --unseal` creates it atomically (O_EXCL) BEFORE any sealed label is read; a second --unseal
    (any file name) is refused. After a crash: `--unseal --resume` with the SAME --out; it is logged in the
    receipt and must reproduce the recorded records_sha256 if one was already written.
  * `run_cascade --mode final` accepts only records whose sha256 equals receipt.records_sha256; after a crash:
    `--resume` with the SAME --out-dir (logged; partial outputs are renamed, never deleted).
  * The receipt is never deleted by this code.

    python -m cascade.make_protocol audit --out <P>     # label-free unit audit (multi-source units per role)
    python -m cascade.make_protocol verify-final --out <P>   # final.md / final.json / records vs the receipt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.dataio import EVAL_SPLITS, assign_events, frame_label, read_manifest  # noqa: E402

TEMPLATE = """# FireSmoke-Clean cascade protocol. Edit BEFORE `freeze`; never after.
version: 3
created_utc: "{created}"
dataset: {dataset_json}
manifest_sha256: "{manifest_sha}"
split_seed: {seed}
calibration_fraction_of_test_units: {cal_frac}
event_gap_minutes: {gap}    # used at init to define Pyro-SDIS events (frozen in splits.csv)
# unit = split_group = GLOBAL connected component of the fsclean dedup graph over ALL sources; cross-source
# verified near-duplicate clusters (e.g. D-Fire pictures inside FASDD_CV) are deliberately ONE unit.
# Test groups are stratified by their primary source (most frames), each group drawn once.
min_test_units_per_source: {min_units}   # sources with fewer test units go to external_shift (no guarantee)
sources_moved_to_external: {moved}

# ---- guarantee ---------------------------------------------------------------------------------
# LTT statement: over the draw of the calibration data, P(a configuration whose TRUE risk exceeds alpha
# is certified) <= delta, provided calibration and future units are exchangeable.
alpha: 0.05
delta: 0.10
target: event               # 'event' = cluster-averaged event miss risk ('frame' = positive-frame miss)
risks: [y_fire, y_smoke]    # joint: certified only if every risk passes (p = max of HB p-values)
unit: split_group           # exchangeable unit (camera / video / near-duplicate group)
fwer: pareto                # pareto (select / certify split) or bonferroni (data-independent grid)
select_fraction: 0.3        # share of calibration units used for grid + Pareto order
calibrate_seed: 1           # seed of the select / certify split inside calibration
grid: null                  # optional pre-registered grid from dev: {{s_edges: [...], g_edges: [...]}}
costs: {{calls: 1.0, handoff: 20.0, fa: 10.0}}
s_min: 0.02                 # frames below this edge score are never escalated (agent must use the same)
payload: frame              # uplink accounting: frame | crop
bootstrap_reps: 5000        # final: cluster (unit) bootstrap for the sealed-test risk CIs
fixed_baselines:            # uncalibrated baselines (thresholds fixed a priori, not tuned)
  edge_threshold: 0.25
  cascade_t_low: 0.10
  cascade_t_high: 0.60
  agent_threshold: 0.50
all_frames_cloud_overlay: false   # agent (overlay view, detector boxes drawn) run on EVERY frame (--s-min 0)
pure_cloud_baseline: false        # separate plain-view agent run on every frame (detector-independent);
                                  # needs --agent-plain at freeze / records
human_tier_assumption: "hand-off frames are resolved correctly by the operator"
esva: null                  # ESVA-LTT (dev study): {{stratum: {{kind: small_smoke, area_max: 0.02}}, lambdas: [...]}}
ravc: null                  # RAVC-LTT (dev study): {{byte_cost: <objective units per uplink byte>}}; required by
                            # cascade.ravc_dev, never used by the main cascade objective (costs:)
figures: null               # null = cascade/figure_style.py DEFAULT_SPEC (code-hashed): method order, axis limits,
                            # main / supplement split. A dict here overrides it and is locked by freeze.

# ---- deployment path (the guarantee holds only for exactly this configuration) -----------------
detector:
  weights_sha256: ""        # filled by freeze from the detector .meta.json
  ultralytics: "8.4.153"
  nms: "false"              # "false" = YOLO26 one-to-one NMS-free head; "none"/"true" = one-to-many + NMS
  imgsz: 640
  conf: 0.001
  iou: 0.7
  max_det: 100
  precision: fp32
agent:
  model: "Qwen/Qwen2.5-VL-3B-Instruct"
  revision: ""              # filled by freeze from the agent .meta.json
  quant: none
  view: overlay
  max_side: 640
  prompt_sha256: ""         # filled by freeze from the agent .meta.json

# ---- pre-registered comparisons ----------------------------------------------------------------
primary_method: cascade_LTT
baselines: [edge_fixed, edge_LTT, edge_LTT_image, detector_gated_agent, cascade_fixed,
            cascade_LTT_image, cascade_LTT_bonf, cascade_LTT+human]
head_rule: |
  Dev only, deterministic (implemented in cascade.head_report): a head is eligible if its fire-event AND
  smoke-event candidate recall at s_min are each within 1.0 percentage point of the best head on that class.
  Among eligible heads: lowest negative-frame escalation share; heads within 1.0 pp of that minimum are
  compared by higher dev mAP50; within 0.5 pp mAP50, by lower detector latency; remaining tie -> nms "false".
  If no head is eligible, choose the head with the largest min(fire recall, smoke recall).
dev_decisions: |
  Chosen on dev only: detector head (head_rule), VLM (3B vs 7B-4bit), view (overlay vs crop), costs,
  optional grid. Order of work: protocol -> heads -> scores -> dev -> freeze -> records -> validity -> final.
  validity runs only after freeze and changes nothing.
prior_exposure: |
  Before freezing: the detector's overall test mAP (YOLO26n base s0: mAP50 0.731) was computed, and a
  300-frame smoke test with a colour-heuristic agent was run on a val+test mixture (v1 code, before roles
  existed). Neither was used for any cascade design choice. No calibration/sealed positive counts were shown.
"""

DET_KEYS = ("ultralytics", "nms", "imgsz", "conf", "iou", "max_det", "precision")
AGENT_KEYS = ("model", "quant", "view", "max_side")


def sha256_file(p):
    h = hashlib.sha256()
    with open(Path(p).expanduser(), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def meta_path(csv_path):
    return Path(csv_path).expanduser().with_suffix(".meta.json")


# files whose code can change a calibration / evaluation number after freeze. The score producers
# (dump_detector, agent_vlm) are covered through the locked CSV + meta contents; head_report / selftest /
# simulate are not used after freeze.
ANALYSIS_CODE = ("__init__.py", "dataio.py", "risk.py", "policy.py", "select.py", "adaptive_view.py", "experiment.py",
                 "build_records.py", "run_cascade.py", "make_protocol.py", "dataset_audit.py", "figure_style.py",
                 "figures.py", "ravc_dev.py", "ravc_synthetic_validity.py", "view_bytes.py",
                 "evidence_veto.py", "esva_dev.py", "esva_synthetic_validity.py", "dev_diagnose.py",
                 "dev_probe.py", "payload_budget.py", "study_register.py", "ace_ltt.py", "ace_register.py", "ace_dev.py",
                 "ace2_register.py", "ace2_dev.py", "ace3_register.py", "ace3_dev.py")
FIGURE_CODE = ("figure_style.py", "figures.py")    # only read final.json; see figures.py --post-freeze-figure-fix


MANIFEST = "MANIFEST.sha256"
# Runtime-generated artefacts: never shipped, never a reason to fail the tree check. The list is explicit so
# that anything NOT on it - a stale module left behind by an overlay install, above all - is reported.
TREE_IGNORE_DIRS = ("__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache")
TREE_IGNORE_SUFFIX = (".pyc", ".pyo", ".pyd", ".so", ".swp", ".orig", ".rej")
TREE_IGNORE_NAMES = (MANIFEST, ".DS_Store", "Thumbs.db")


def ignored_in_tree(rel):
    parts = rel.split("/")
    return (any(d in parts for d in TREE_IGNORE_DIRS) or rel.endswith(TREE_IGNORE_SUFFIX)
            or parts[-1] in TREE_IGNORE_NAMES)


def tree_files(pkg):
    """Every shipped file of the package, sorted, excluding runtime artefacts and the manifest itself."""
    return [rel for rel in (p.relative_to(pkg).as_posix() for p in sorted(Path(pkg).rglob("*")) if p.is_file())
            if not ignored_in_tree(rel)]


def write_manifest(pkg=None):
    """Build MANIFEST.sha256 (one 'sha256  relpath' line per shipped file). Run at packaging time."""
    pkg = Path(pkg) if pkg else Path(__file__).resolve().parent
    lines = [f"{sha256_file(pkg / rel)}  {rel}" for rel in tree_files(pkg)]
    (pkg / MANIFEST).write_text("\n".join(lines) + "\n")
    return len(lines)


def verify_tree(pkg=None):
    """Compare the INSTALLED tree against MANIFEST.sha256.

    An overlay unzip (`unzip -o`) updates and adds files but never removes one that the new package dropped,
    so the code hash can be current while a stale module still sits in the tree and gets imported. This is
    the check that catches it: extra / missing / changed, over every shipped file, not only ANALYSIS_CODE.
    """
    pkg = Path(pkg) if pkg else Path(__file__).resolve().parent
    mf = pkg / MANIFEST
    if not mf.exists():
        return {"ok": False, "extra": [], "missing": [MANIFEST], "changed": [], "n_files": 0,
                "manifest_sha256": ""}
    want = {}
    for line in mf.read_text().splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            want[rel] = h
    have = set(tree_files(pkg))
    extra = sorted(have - set(want))
    missing = sorted(set(want) - have)
    changed = sorted(rel for rel in (have & set(want)) if sha256_file(pkg / rel) != want[rel])
    return {"ok": not (extra or missing or changed), "extra": extra, "missing": missing, "changed": changed,
            "n_files": len(want), "manifest_sha256": sha256_file(mf)}


def analysis_code(pkg=None):
    """Deterministic hash of the analysis code: sha256 over (file name, sha256 of raw bytes) in fixed order."""
    pkg = Path(pkg) if pkg else Path(__file__).resolve().parent
    files = {n: sha256_file(pkg / n) for n in ANALYSIS_CODE}
    h = hashlib.sha256()
    for n in ANALYSIS_CODE:
        h.update(f"{n}\0{files[n]}\n".encode())
    return h.hexdigest(), files


def lock_digest(out, files, code_sha):
    """sha256 over protocol.yaml, splits.csv, the listed score / meta files (full contents) and the code hash."""
    h = hashlib.sha256()
    for p in (Path(out, "protocol.yaml"), Path(out, "splits.csv"), *[Path(f).expanduser() for f in files]):
        h.update(p.name.encode())
        h.update(sha256_file(p).encode())
    h.update(b"analysis_code_sha256")
    h.update(str(code_sha).encode())
    return h.hexdigest()


def locked_files(L):
    return [L[k] for k in ("det_csv", "det_meta", "agent_csv", "agent_meta", "agent_plain_csv", "agent_plain_meta")
            if L.get(k)]


def read_lock(out):
    lk = Path(out).expanduser() / "protocol.lock"
    return json.loads(lk.read_text()) if lk.exists() else None


def lock_status(out):
    """[] if protocol.lock exists and still matches protocol.yaml, splits.csv, the locked score files AND the
    analysis code now running; otherwise a list of human-readable problems."""
    L = read_lock(out)
    if L is None:
        return ["no protocol.lock (not frozen)"]
    pr = []
    if not L.get("analysis_code_sha256"):
        return ["protocol.lock has no analysis_code_sha256 (made by an older version): re-freeze a NEW protocol"]
    code_sha, files = analysis_code()
    if code_sha != L["analysis_code_sha256"]:
        diff = [n for n in ANALYSIS_CODE if files.get(n) != (L.get("analysis_code_files") or {}).get(n)]
        pr.append(f"analysis code differs from the code locked at freeze (analysis_code_sha256); changed: {diff}")
    try:
        if L["sha256"] != lock_digest(Path(out).expanduser(), locked_files(L), L["analysis_code_sha256"]):
            pr.append("protocol.yaml / splits.csv / locked score or meta files changed after freeze")
    except FileNotFoundError as e:
        pr.append(f"locked file missing: {e.filename}")
    return pr


def check_lock(out):
    return not lock_status(out)


def receipt_path(out):
    return Path(out).expanduser() / "final_receipt.json"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_receipt(out):
    p = receipt_path(out)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"protocol receipt {p} is not valid JSON ({e}); refusing (fail-closed). It is never "
                         "deleted by this code; inspect it by hand and report it.")


def _write_tmp(p, R):
    """Write the full JSON to a unique temp file next to p and fsync it (never a half-written receipt)."""
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    return tmp


def _write_receipt(out, R):
    p = receipt_path(out)
    os.replace(_write_tmp(p, R), p)                     # atomic update, the receipt file itself is never removed


def _create_receipt(p, R):
    """Atomic EXCLUSIVE creation of a COMPLETE receipt: the full JSON is written + fsynced to a temp file, then
    hard-linked to the receipt name (link fails if the name exists). A crash leaves either no receipt or a complete
    one, never a partial JSON. Filesystems without hard links: O_EXCL + write + fsync (documented fallback)."""
    tmp = _write_tmp(p, R)
    try:
        os.link(tmp, p)
    except FileExistsError:
        raise
    except OSError:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(R, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _event(R, what, **kw):
    R.setdefault("events", []).append({"event": what, "utc": _now(), **kw})


def begin_unseal(out, records_out, lock_sha, resume=False):
    """none -> unsealing (atomic, before any sealed label is read). Resume only from unsealing / unsealed, same
    records path, same lock."""
    p = receipt_path(out)
    rp = str(Path(records_out).expanduser().resolve())
    R = {"state": "unsealing", "lock_sha256": lock_sha, "records": rp, "records_sha256": None,
         "analysis_code_sha256": analysis_code()[0], "events": [{"event": "unseal_start", "utc": _now()}]}
    try:
        _create_receipt(p, R)
    except FileExistsError:
        R = read_receipt(out)
        if not resume:
            raise SystemExit(f"refusing --unseal: the protocol receipt already exists (state={R.get('state')}, "
                             f"records={R.get('records')}). Sealed labels are unsealed once per protocol; after a "
                             "crash use --unseal --resume with the same --out.")
        if R.get("state") not in ("unsealing", "unsealed"):
            raise SystemExit(f"--resume refused: receipt state is '{R.get('state')}' (only unsealing / unsealed "
                             "can be resumed by build_records)")
        if R.get("records") != rp:
            raise SystemExit(f"--resume refused: the receipt names another records file: {R.get('records')}")
        if R.get("lock_sha256") != lock_sha:
            raise SystemExit("--resume refused: the receipt was created under another lock")
        _event(R, "unseal_resume", from_state=R["state"])
        _write_receipt(out, R)
        return R
    return R


def end_unseal(out, records_sha):
    """unsealing -> unsealed(records_sha256). A resumed unseal must reproduce an already recorded hash."""
    R = read_receipt(out)
    if R.get("records_sha256") not in (None, records_sha):
        raise SystemExit("resumed unseal produced a different records file than the one in the receipt")
    R.update(state="unsealed", records_sha256=records_sha)
    _event(R, "unsealed", records_sha256=records_sha)
    _write_receipt(out, R)


def records_guard(out, records_out):
    """Sealed (non --unseal) builds may not overwrite the records file named in the receipt."""
    R = read_receipt(out)
    if R and R.get("records") == str(Path(records_out).expanduser().resolve()):
        raise SystemExit("refusing to overwrite the unsealed records file named in the protocol receipt")


def begin_final(out, out_dir, records_sha, lock_sha, resume=False):
    """unsealed -> running (or running -> running with --resume, same out-dir). Only the receipt's records hash
    is accepted."""
    R = read_receipt(out)
    od = str(Path(out_dir).expanduser().resolve())
    if R is None:
        raise SystemExit("final refused: no protocol receipt. Build the records with build_records --unseal "
                         "(it creates the receipt) and pass exactly that file.")
    st = R.get("state")
    if st == "done":
        raise SystemExit(f"final already run for this protocol (receipt state done, out_dir {R.get('out_dir')})")
    if R.get("lock_sha256") != lock_sha:
        raise SystemExit("final refused: the receipt was created under another lock")
    if st == "unsealing" or not R.get("records_sha256"):
        raise SystemExit("final refused: unseal did not complete (receipt state unsealing); rerun "
                         "build_records --unseal --resume with the same --out")
    if R["records_sha256"] != records_sha:
        raise SystemExit("final refused: these records are not the ones recorded in the protocol receipt "
                         f"(receipt records_sha256 {R['records_sha256'][:12]}..., file {records_sha[:12]}...)")
    if st == "unsealed":
        R.update(state="running", out_dir=od)
        _event(R, "final_start", out_dir=od)
    elif st == "running":
        if not resume:
            raise SystemExit(f"final refused: a final is running or crashed (receipt state running, out_dir "
                             f"{R.get('out_dir')}). After a crash use --resume with the same --out-dir.")
        if R.get("out_dir") != od:
            raise SystemExit(f"--resume refused: the receipt's out_dir is {R.get('out_dir')}")
        moved = []
        for n in ("final.md", "final.json"):
            f = Path(od) / n
            if f.exists():
                k = sum(1 for e in R.get("events", []) if e["event"] == "final_resume") + 1
                dst = f.with_name(f"{n}.crashed_{k}")
                os.replace(f, dst)
                moved.append({"file": dst.name, "sha256": sha256_file(dst)})
        _event(R, "final_resume", out_dir=od, renamed_partial=moved)
    else:
        raise SystemExit(f"final refused: unexpected receipt state {st!r}")
    _write_receipt(out, R)
    return R


def finish_final(out, final_md):
    R = read_receipt(out)
    fj = Path(final_md).with_name("final.json")
    R.update(state="done", final_md_sha256=sha256_file(final_md), final_json_sha256=sha256_file(fj))
    _event(R, "done", final_md_sha256=R["final_md_sha256"], final_json_sha256=R["final_json_sha256"])
    _write_receipt(out, R)
    return R


def verify_final(a):
    """Read-only audit of a finished final: receipt state, final.md / final.json / unsealed records hashes, lock."""
    R = read_receipt(a.out)
    pr = []
    if not R:
        raise SystemExit("verify-final: no protocol receipt")
    if R.get("state") != "done":
        pr.append(f"receipt state is {R.get('state')!r}, not 'done'")
    od = Path(R.get("out_dir") or ".")
    for n, k in (("final.md", "final_md_sha256"), ("final.json", "final_json_sha256")):
        f = od / n
        if not R.get(k):
            pr.append(f"receipt has no {k}")
        elif not f.exists():
            pr.append(f"{f} missing")
        elif sha256_file(f) != R[k]:
            pr.append(f"{n} differs from the receipt")
    rec = Path(R.get("records") or "")
    if not rec.exists() or sha256_file(rec) != R.get("records_sha256"):
        pr.append("unsealed records file missing or differs from receipt.records_sha256")
    pr += lock_status(a.out)
    ev = [e["event"] for e in R.get("events", [])]
    print(f"receipt: state={R.get('state')} out_dir={od}")
    print(f"  records_sha256    {R.get('records_sha256')}")
    print(f"  final_md_sha256   {R.get('final_md_sha256')}")
    print(f"  final_json_sha256 {R.get('final_json_sha256')}")
    print(f"  events: {ev}  (unseal resumes {ev.count('unseal_resume')}, final resumes {ev.count('final_resume')})")
    if pr:
        raise SystemExit("verify-final FAILED:\n  " + "\n  ".join(pr))
    print("verify-final: OK (receipt, final.md, final.json, unsealed records, lock and analysis code all match)")


def receipt_summary(R):
    ev = R.get("events", [])
    n = lambda k: sum(e["event"] == k for e in ev)   # noqa: E731
    return (f"receipt: records_sha256 {R.get('records_sha256')}, unseal resumes {n('unseal_resume')}, "
            f"final resumes {n('final_resume')}")


def load_protocol(out):
    import yaml
    out = Path(out).expanduser()
    P = yaml.safe_load((out / "protocol.yaml").read_text())
    with open(out / "splits.csv", newline="") as f:
        splits = {r["uid"]: r for r in csv.DictReader(f)}
    return P, splits


def init(a):
    ds = Path(a.dataset).expanduser()
    out = Path(a.out).expanduser()
    if (out / "protocol.lock").exists():
        raise SystemExit(f"{out} is frozen; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(ds, EVAL_SPLITS)
    ev = assign_events(rows, a.event_gap)
    rng = np.random.default_rng(a.seed)
    # unit = split_group: a GLOBAL connected component over all sources (fsclean dedup). A group that spans
    # sources is a verified cross-source near-duplicate cluster and must stay ONE unit. Each test group is put
    # in exactly one stratum: its primary source (most frames; ties -> name), so no group is drawn twice.
    grp_src = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["split"] == "test":
            grp_src[r["split_group"]][r["source"]] += 1
    primary = {gk: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for gk, c in grp_src.items()}
    by_src = defaultdict(list)
    for gk, s in primary.items():
        by_src[s].append(gk)
    moved = sorted(s for s, gs in by_src.items() if len(gs) < a.min_units)     # too few units to calibrate
    cal_units, ext_units = set(), set()
    for src in sorted(by_src):
        u = sorted(by_src[src])
        if src in moved:
            ext_units |= set(u)
            continue
        rng.shuffle(u)
        k = int(round(a.cal_frac * len(u)))
        if len(u) >= 2:
            k = min(max(k, 1), len(u) - 1)
        cal_units |= set(u[:k])
    role_of = {"val": "dev", "external": "external_shift"}
    table = defaultdict(lambda: {"frames": 0, "units": set(), "events": set(), "pos_units": set(), "pos_ev": set()})
    with open(out / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "split", "role", "source", "unit", "event"])
        for r in rows:
            role = role_of.get(r["split"]) or ("external_shift" if r["split_group"] in ext_units else
                                               "calibration" if r["split_group"] in cal_units else "sealed_test")
            unit = f"g{r['split_group']}" if r["split_group"] != "" else "u:" + r["uid"]
            w.writerow([r["uid"], r["split"], role, r["source"], unit, ev[r["uid"]]])
            t = table[(role, r["source"])]
            t["frames"] += 1
            t["units"].add(unit)
            t["events"].add(ev[r["uid"]])
            if role == "dev" and frame_label(r)[0]:           # labels are read for dev ONLY
                t["pos_units"].add(unit)
                t["pos_ev"].add(ev[r["uid"]])
    (out / "protocol.yaml").write_text(TEMPLATE.format(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"), dataset_json=json.dumps(str(ds)),
        manifest_sha=sha256_file(ds / "manifest.csv"), seed=a.seed, cal_frac=a.cal_frac, gap=a.event_gap,
        min_units=a.min_units, moved=json.dumps(moved)))
    # every unit must have exactly one role
    roles_of_unit = defaultdict(set)
    with open(out / "splits.csv", newline="") as f:
        for r in csv.DictReader(f):
            roles_of_unit[r["unit"]].add(r["role"])
    assert all(len(v) == 1 for v in roles_of_unit.values()), "a unit spans several roles"
    if moved:
        print(f"sources with < {a.min_units} test units moved to external_shift (no guarantee claimed): {moved}")
    print(f"{'role':15s} {'source':16s} {'frames':>7s} {'units':>6s} {'events':>7s} {'pos_units':>9s} {'pos_events':>10s}")
    for (role, src), t in sorted(table.items()):
        pu = str(len(t["pos_units"])) if role == "dev" else "-"
        pe = str(len(t["pos_ev"])) if role == "dev" else "-"
        print(f"{role:15s} {src:16s} {t['frames']:7d} {len(t['units']):6d} {len(t['events']):7d} {pu:>9s} {pe:>10s}")
    print("\n(positive counts are shown for dev only; calibration / sealed_test labels were not read)")
    print(f"wrote {out/'splits.csv'} and {out/'protocol.yaml'}")
    audit(a)


def audit(a):
    """Label-free unit audit from splits.csv: units whose frames come from more than one source (verified
    cross-source duplicate clusters), per role; writes <P>/unit_audit.csv."""
    out = Path(a.out).expanduser()
    comp = defaultdict(lambda: defaultdict(int))
    role = {}
    with open(out / "splits.csv", newline="") as f:
        for r in csv.DictReader(f):
            comp[r["unit"]][r["source"]] += 1
            role[r["unit"]] = r["role"]
    multi = {u: c for u, c in comp.items() if len(c) > 1}
    units_per_role = defaultdict(int)
    for u in comp:
        units_per_role[role[u]] += 1
    rows = []
    for u, c in multi.items():
        srcs = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        rows.append({"unit": u, "role": role[u], "primary_source": srcs[0][0], "n_frames": sum(c.values()),
                     "sources": ";".join(f"{s}:{n}" for s, n in srcs)})
    rows.sort(key=lambda r: (r["role"], -r["n_frames"], r["unit"]))
    with open(out / "unit_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["unit", "role", "primary_source", "n_frames", "sources"])
        w.writeheader()
        w.writerows(rows)
    print("\nunit audit (label-free): multi-source units = verified cross-source duplicate clusters, one unit each")
    print(f"{'role':15s} {'units':>7s} {'multi_src':>9s} {'frames_in_multi':>15s}")
    for rl in sorted(units_per_role):
        mr = [r for r in rows if r["role"] == rl]
        print(f"{rl:15s} {units_per_role[rl]:7d} {len(mr):9d} {sum(r['n_frames'] for r in mr):15d}")
    for r in rows[:10]:
        print(f"  {r['role']:13s} {r['unit']:10s} primary={r['primary_source']:10s} {r['sources']}")
    print(f"wrote {out/'unit_audit.csv'} ({len(rows)} multi-source units)")


def check_metas(P, det_meta, agent_meta, fill=False):
    """Compare meta files with protocol.yaml. Returns (problems, filled_yaml_fields)."""
    problems, filled = [], {}
    D, A = P["detector"], P["agent"]
    for k in DET_KEYS:
        if str(det_meta.get(k)) != str(D.get(k)):
            problems.append(f"detector.{k}: protocol={D.get(k)!r} meta={det_meta.get(k)!r}")
    want_e2e = str(det_meta.get("nms")) == "false"
    if det_meta.get("runtime_end2end") is None or bool(det_meta.get("runtime_end2end")) != want_e2e:
        problems.append(f"detector runtime_end2end={det_meta.get('runtime_end2end')!r} is inconsistent with "
                        f"nms={det_meta.get('nms')!r} (nms 'false' must run the one-to-one head)")
    for k in AGENT_KEYS:
        if str(agent_meta.get(k)) != str(A.get(k)):
            problems.append(f"agent.{k}: protocol={A.get(k)!r} meta={agent_meta.get(k)!r}")
    if float(agent_meta.get("s_min", -1)) != float(P["s_min"]) and not (P.get("all_frames_cloud_overlay") and
                                                                       float(agent_meta.get("s_min", -1)) == 0):
        problems.append(f"agent s_min {agent_meta.get('s_min')} != protocol s_min {P['s_min']}")
    if agent_meta.get("dry"):
        problems.append("agent is the --dry colour heuristic")
    for yk, mk, src in (("weights_sha256", "weights_sha256", D), ("revision", "revision", A),
                        ("prompt_sha256", "prompt_sha256", A)):
        mv = (det_meta if src is D else agent_meta).get(mk, "")
        if not mv:
            problems.append(f"meta has no {mk}")
        elif yk.endswith("_sha256") and not (len(str(mv)) == 64 and set(str(mv).lower()) <= set("0123456789abcdef")):
            problems.append(f"{yk} is not a sha256 hex digest: {mv!r}")   # a placeholder must never freeze
        elif not src.get(yk):
            filled[yk] = mv
        elif src.get(yk) != mv:
            problems.append(f"{yk}: protocol={src.get(yk)} meta={mv}")
    return problems, filled


def weights_hash(a):
    """Report the detector weights digest with ONE path semantics: the one the detector run itself used.

    protocol.yaml holds no weights path - only weights_sha256, which freeze copies from the detector
    .meta.json. That meta was written by dump_detector, which stored the ABSOLUTE resolved path of the
    checkpoint it actually loaded together with the digest of that same file. So the digest is never
    recomputed from a second, possibly differently-resolved path; it is read back and, if the file is still
    there, re-checked against it. A mismatch means the checkpoint on disk is no longer the one that produced
    the scores, which must be seen BEFORE freeze, not after.
    """
    m = meta_path(a.det)
    meta = json.loads(m.read_text())
    rec_path, rec_sha = meta.get("weights", ""), str(meta.get("weights_sha256", ""))
    print("meta:              ", m.resolve())
    print("weights_path:      ", rec_path, "(recorded by dump_detector, already absolute and resolved)")
    print("weights_sha256:    ", rec_sha)
    ok = len(rec_sha) == 64 and set(rec_sha.lower()) <= set("0123456789abcdef")
    print("digest_format:     ", "64-hex OK" if ok else "NOT A SHA256 DIGEST")
    p = Path(rec_path).expanduser() if rec_path else None
    if p and p.exists():
        now = sha256_file(p)
        print("on_disk_sha256:    ", now)
        print("checkpoint:        ", "MATCHES the recorded digest" if now == rec_sha else
              "DIFFERS - the file on disk is not the one that produced the scores")
        ok = ok and now == rec_sha
    else:
        print("on_disk_sha256:     file not present on this machine; the recorded digest is authoritative")
    print("\nfreeze fills protocol.yaml detector.weights_sha256 from this meta; do not edit it by hand.")
    raise SystemExit(0 if ok else 1)


def fill_weights(a):
    """Copy detector.weights_sha256 from the detector .meta.json into protocol.yaml, before freeze.

    freeze fills this field itself, but the pre-freeze checklist reads protocol.yaml, so the field has to be
    populated first; hand-editing it would create a second source for a value that must have exactly one.
    This command is that single path: it takes the digest from the meta the detector run wrote, refuses a
    non-hex value, refuses to overwrite a DIFFERENT value already in the file, refuses after freeze, and
    keeps a snapshot of the previous protocol.yaml. freeze then re-checks the field against the same meta.
    """
    import yaml
    out = Path(a.out).expanduser()
    if (out / "protocol.lock").exists():
        raise SystemExit("already frozen: protocol.yaml must not change after freeze")
    meta = json.loads(meta_path(a.det).read_text())
    mv = str(meta.get("weights_sha256", ""))
    if not (len(mv) == 64 and set(mv.lower()) <= set("0123456789abcdef")):
        raise SystemExit(f"the detector meta has no usable weights_sha256: {mv!r}")
    y = out / "protocol.yaml"
    txt = y.read_text()
    cur = str((yaml.safe_load(txt).get("detector") or {}).get("weights_sha256") or "")
    if cur == mv:
        print("already filled with exactly this digest; nothing to do")
        print("protocol.yaml sha256:", sha256_file(y))
        return
    if cur:
        raise SystemExit(f"protocol.yaml already has a DIFFERENT weights_sha256:\n  file: {cur}\n  meta: {mv}")
    old_sha = sha256_file(y)
    snap = y.with_suffix(f".yaml.{old_sha[:8]}.snapshot")
    if not snap.exists():
        snap.write_text(txt)
    new = txt.replace('weights_sha256: ""', f'weights_sha256: "{mv}"', 1)
    if new == txt:
        raise SystemExit('could not find the empty field `weights_sha256: ""` in protocol.yaml')
    y.write_text(new)
    assert (yaml.safe_load(new)["detector"]["weights_sha256"]) == mv
    print("weights_sha256:      ", mv, f"(from {meta_path(a.det).name})")
    print("snapshot of previous:", snap.name)
    print("protocol.yaml sha256:", old_sha, "->", sha256_file(y))
    print("\nprotocol.yaml has changed: take every registration hash again from here on.")


def freeze(a):
    import yaml
    out = Path(a.out).expanduser()
    if (out / "protocol.lock").exists():
        raise SystemExit("already frozen")
    P, splits = load_protocol(out)
    ds = Path(P["dataset"]).expanduser()
    if sha256_file(ds / "manifest.csv") != P["manifest_sha256"]:
        raise SystemExit("dataset manifest changed since init")
    dm, am = meta_path(a.det), meta_path(a.agent)
    det_meta, agent_meta = json.loads(dm.read_text()), json.loads(am.read_text())
    if agent_meta.get("det_meta_sha256") != hashlib.sha256(dm.read_bytes()).hexdigest():
        raise SystemExit("the agent CSV was not produced from this detector CSV (det_meta_sha256 mismatch)")
    if agent_meta.get("det_csv_sha256") != sha256_file(a.det):
        raise SystemExit("the detector CSV changed after the agent was run (det_csv_sha256 mismatch)")
    problems, filled = check_metas(P, det_meta, agent_meta)
    plain = {}
    if P.get("pure_cloud_baseline"):
        if not a.agent_plain:
            raise SystemExit("pure_cloud_baseline: true needs --agent-plain")
        pm_path = meta_path(a.agent_plain)
        pm = json.loads(pm_path.read_text())
        if pm.get("view") != "plain" or float(pm.get("s_min", -1)) != 0.0:
            problems.append("agent-plain must be run with --view plain --s-min 0")
        if pm.get("det_csv_sha256") != sha256_file(a.det):
            problems.append("agent-plain was not run on this detector CSV")
        for k in ("model", "quant", "max_side", "revision"):
            if str(pm.get(k)) != str(agent_meta.get(k)):
                problems.append(f"agent-plain.{k} differs from the main agent")
        plain = {"agent_plain_csv": str(Path(a.agent_plain).expanduser().resolve()),
                 "agent_plain_meta": str(pm_path.resolve()),
                 "agent_plain_csv_sha256": sha256_file(a.agent_plain), "agent_plain_meta_sha256": sha256_file(pm_path)}
    if problems:
        raise SystemExit("protocol / meta mismatch:\n  " + "\n  ".join(problems))
    if filled:                                     # write derived hashes into the yaml text before locking
        txt = (out / "protocol.yaml").read_text()
        for k, v in filled.items():
            txt = txt.replace(f'{k}: ""', f'{k}: "{v}"', 1)
        (out / "protocol.yaml").write_text(txt)
        P2 = yaml.safe_load(txt)
        assert all(P2["detector"].get(k) == v or P2["agent"].get(k) == v for k, v in filled.items())
    L = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "det_csv": str(Path(a.det).expanduser().resolve()), "agent_csv": str(Path(a.agent).expanduser().resolve()),
         "det_meta": str(dm.resolve()), "agent_meta": str(am.resolve()),
         "det_csv_sha256": sha256_file(a.det), "agent_csv_sha256": sha256_file(a.agent),
         "det_meta_sha256": sha256_file(dm), "agent_meta_sha256": sha256_file(am),
         "n_split_rows": len(splits), **plain}
    code_sha, code_files = analysis_code()
    L.update(analysis_code_sha256=code_sha, analysis_code_files=code_files)
    d = lock_digest(out, locked_files(L), code_sha)
    (out / "protocol.lock").write_text(json.dumps({"sha256": d, **L}, indent=1))
    print("filled:", filled or "nothing")
    print("analysis_code_sha256:", code_sha)
    print("frozen:", d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--cal-frac", type=float, default=0.5)
    p.add_argument("--event-gap", type=float, default=30.0, help="minutes (Pyro-SDIS events)")
    p.add_argument("--min-units", type=int, default=5,
                   help="sources with fewer test units are moved to external_shift (e.g. FLAME2: 1 video)")
    p = sub.add_parser("freeze")
    p.add_argument("--out", required=True)
    p.add_argument("--det", required=True, help="detector CSV of the deployment configuration")
    p.add_argument("--agent", required=True, help="agent CSV of the deployment configuration")
    p.add_argument("--agent-plain", default=None, help="plain-view, s_min=0 agent CSV (pure_cloud_baseline)")
    p = sub.add_parser("audit")
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify-final")
    p.add_argument("--out", required=True)
    p = sub.add_parser("weights-hash", help="the detector weights digest AS RECORDED, re-checked on disk")
    p.add_argument("--det", required=True, help="detector CSV (its .meta.json holds the path and the digest)")
    p = sub.add_parser("fill-weights", help="write detector.weights_sha256 into protocol.yaml from the meta")
    p.add_argument("--out", required=True)
    p.add_argument("--det", required=True)
    p = sub.add_parser("code-hash", help="print analysis_code_sha256 of the installed package")
    p.add_argument("--verify-tree", action="store_true",
                   help="also compare EVERY file of the installed package against MANIFEST.sha256 "
                        "(catches a stale file left behind by an overlay unzip)")
    a = ap.parse_args()
    if a.cmd == "weights-hash":
        return weights_hash(a)
    if a.cmd == "fill-weights":
        return fill_weights(a)
    if a.cmd == "code-hash":
        sha, files = analysis_code()
        for n in ANALYSIS_CODE:
            print(f"{files[n]}  {n}")
        print("analysis_code_sha256:", sha)
        if a.verify_tree:
            r = verify_tree()
            for k in ("extra", "missing", "changed"):
                if r[k]:
                    print(f"{k}: {r[k]}")
            print("runtime_tree:", "OK (matches MANIFEST.sha256)" if r["ok"] else "DIFFERS FROM THE PACKAGE",
                  f"[{r['n_files']} files, manifest_sha256 {r['manifest_sha256'][:16]}]")
            raise SystemExit(0 if r["ok"] else 1)
        return
    {"init": init, "freeze": freeze, "audit": audit, "verify-final": verify_final}[a.cmd](a)


if __name__ == "__main__":
    main()
