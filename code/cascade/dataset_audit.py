"""Label-free dataset audit for figure 2 (FireSmoke-Clean): before/after cleaning, duplicates, cross-source overlap,
split_group structure and the pHash threshold sensitivity of the leakage estimate.

    python -m cascade.dataset_audit --dataset <FireSmoke-Clean_v0.2> [--work <clean_work>] --out dataset_audit.json

--dataset : exported dataset (manifest.csv + provenance/*.json) -> counts, pipeline leakage, cross-source overlap
--work    : fsclean work dir (items.csv, split.csv, pairs_hash.npz, clusters.npz); only where fsclean ran (the PC).
            Adds the sensitivity curve: for pHash threshold t = 0..14, the share of held-out images that have a
            DIRECT hash neighbour (pHash <= t) in train
              * in the ORIGINAL split of the same source (D-Fire test, FASDD test/val, Pyro-SDIS val), and
              * in the FireSmoke-Clean split (any source),
            once for hash candidates (unverified look-alikes included) and once for pixel-verified duplicate edges
            (verification exists only up to the scene threshold, pHash <= 10).
            Direct neighbours only: the pipeline figure (dedup_report, transitive clusters + embedding + camera
            groups) is reported next to it and is higher.
No detector / label / sealed information is read.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

csv.field_size_limit(1 << 30)


def _json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def from_dataset(ds):
    ds = Path(ds).expanduser()
    A = {"dataset": ds.name}
    prov = ds / "provenance"
    dr = _json(prov / "dedup_report.json") or {}
    ss = _json(prov / "split_summary.json") or {}
    sc = _json(prov / "scan_summary.json") or {}
    A["pipeline_original_split_leakage"] = dr.get("original_split_leakage", {})
    A["cross_source_near_duplicates"] = dr.get("cross_source_near_duplicates(images)", {})
    A["exact_copies"] = dr.get("exact_copies", {})
    A["thresholds"] = {k: v for k, v in (dr.get("thresholds") or {}).items() if "hash" in k or "cos" in k}
    A["groups_spanning_splits"] = ss.get("groups_spanning_splits")
    A["test_unique"] = ss.get("test_unique")
    A["scan_status"] = sc.get("status", {})
    A["label_issues"] = sc.get("label_issues(images)", {})
    man = _rows(ds / "manifest.csv")
    kept = Counter((r["source"], r["split"]) for r in man)
    A["kept"] = {f"{s}/{sp}": n for (s, sp), n in sorted(kept.items())}
    g_src, g_split, g_n = defaultdict(set), defaultdict(set), Counter()
    for r in man:
        if r["split"] == "external" or r["split_group"] == "":
            continue
        g = r["split_group"]
        g_src[g].add(r["source"])
        g_split[g].add(r["split"])
        g_n[g] += 1
    sizes = np.array(list(g_n.values()))
    A["split_group"] = {"n": int(len(sizes)), "singletons": int((sizes == 1).sum()),
                        "max": int(sizes.max()) if len(sizes) else 0,
                        "frames_in_groups>1": int(sizes[sizes > 1].sum()),
                        "multi_source_groups": int(sum(len(v) > 1 for v in g_src.values())),
                        "groups_spanning_splits_in_manifest": int(sum(len(v) > 1 for v in g_split.values())),
                        "size_hist": {str(k): int(v) for k, v in sorted(Counter(sizes.tolist()).items())}}
    by_src_sizes = defaultdict(list)
    for g, n in g_n.items():
        by_src_sizes["multi-source" if len(g_src[g]) > 1 else next(iter(g_src[g]))].append(int(n))
    A["split_group_sizes_by_source"] = {k: sorted(v) for k, v in by_src_sizes.items()}
    return A


def sensitivity(work, tmax=14):
    work = Path(work).expanduser()
    items = _rows(work / "items.csv")
    sp = {r["uid"]: r for r in _rows(work / "split.csv")}
    N = len(items)
    src = np.array([r["source"] for r in items])
    osplit = np.array([r["orig_split"] for r in items])
    pool = np.array([r["role"] == "pool" and r["status"] == "ok" for r in items])
    csplit = np.array([(sp.get(r["uid"]) or {}).get("split", "") for r in items])
    removed = np.array([bool((sp.get(r["uid"]) or {}).get("removed")) for r in items])
    P = np.load(work / "pairs_hash.npz")
    I, J, pd = P["i"].astype(np.int64), P["j"].astype(np.int64), P["p"].astype(np.int32)
    C = np.load(work / "clusters.npz")
    assert len(C["ok"]) == N and len(C["hash_dup"]) == len(I), "clusters.npz / pairs_hash.npz / items.csv mismatch"
    verified = C["hash_dup"].astype(bool)
    both_pool = pool[I] & pool[J]
    heldout = [(s, h) for s in ("dfire", "fasdd_cv", "pyro_sdis", "flame2_det") for h in ("test", "val")
               if (pool & (src == s) & (osplit == h)).any() and (pool & (src == s) & (osplit == "train")).any()]
    rows = []
    for t in range(tmax + 1):
        at = both_pool & (pd <= t)
        for kind, edge in (("hash_candidate", at), ("pixel_verified", at & verified)):
            if kind == "pixel_verified" and t > 10:
                continue                                   # no verification beyond the scene threshold
            for s, h in heldout:
                q = (src == s) & (osplit == h) & pool
                tr = (src == s) & (osplit == "train") & pool
                hit = np.zeros(N, bool)
                e = edge & ((q[I] & tr[J]) | (q[J] & tr[I]))
                hit[np.where(q[I[e]], I[e], J[e])] = True
                rows.append({"split": "original", "subset": f"{s}/{h}", "phash_t": t, "kind": kind,
                             "n_heldout": int(q.sum()), "n_with_neighbour": int(hit[q].sum()),
                             "pct": round(100 * hit[q].sum() / max(1, q.sum()), 2)})
            qt = (csplit == "test") & ~removed & pool
            tr = (csplit == "train") & ~removed & pool
            hit = np.zeros(N, bool)
            e = edge & ((qt[I] & tr[J]) | (qt[J] & tr[I]))
            hit[np.where(qt[I[e]], I[e], J[e])] = True
            rows.append({"split": "FireSmoke-Clean", "subset": "test (all sources)", "phash_t": t, "kind": kind,
                         "n_heldout": int(qt.sum()), "n_with_neighbour": int(hit[qt].sum()),
                         "pct": round(100 * hit[qt].sum() / max(1, qt.sum()), 2)})
    per_src = defaultdict(Counter)
    for i, r in enumerate(items):
        if r["role"] != "pool":
            per_src[r["source"]]["external_or_tag"] += 1
            continue
        per_src[r["source"]]["raw"] += 1
        if r["status"] != "ok":
            per_src[r["source"]]["invalid"] += 1
        elif removed[i]:
            per_src[r["source"]]["exact_copy_removed"] += 1
        else:
            per_src[r["source"]][f"kept_{csplit[i]}"] += 1
    return rows, {k: dict(v) for k, v in per_src.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A = from_dataset(a.dataset)
    if a.work:
        rows, per_src = sensitivity(a.work)
        A["phash_sensitivity"] = rows
        A["before_after_by_source"] = per_src
        A["work"] = Path(a.work).name
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(A, indent=1))
    print(json.dumps({k: A[k] for k in ("kept", "split_group", "groups_spanning_splits", "exact_copies")
                      if k in A}, indent=1)[:3000])
    if a.work:
        for r in A["phash_sensitivity"]:
            if r["phash_t"] in (4, 8, 10, 14):
                print(r)
    print("->", out)


if __name__ == "__main__":
    sys.exit(main())
