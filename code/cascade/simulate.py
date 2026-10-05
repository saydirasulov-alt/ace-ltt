"""Synthetic frames with the structure of the real problem (for testing the statistics only)."""
from __future__ import annotations

import numpy as np


def _sig(x):
    return 1 / (1 + np.exp(-x))


def make(n_units=3000, frames_per_unit=(1, 30), pos_unit_frac=0.5, seed=0, hard_neg=0.15,
         sources=("dfire", "fasdd_cv", "pyro_sdis"), agent_quality=2.2, source_shift=None, event_len=8):
    """Returns dict of arrays: source, unit, event, y, s, g, nbytes, tag.
    Unit random effects make frames of one unit correlated (like a camera or a video)."""
    rng = np.random.default_rng(seed)
    src_of_unit = rng.choice(len(sources), n_units)
    sizes = rng.integers(frames_per_unit[0], frames_per_unit[1] + 1, n_units)
    pos_unit = rng.random(n_units) < pos_unit_frac
    u_eff = rng.normal(0, 0.8, n_units)                      # unit difficulty
    shift = np.zeros(len(sources)) if source_shift is None else np.asarray(source_shift, float)
    unit = np.repeat(np.arange(n_units), sizes)
    N = len(unit)
    su = src_of_unit[unit]
    pos_in_unit = np.arange(N) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    event = np.char.add(np.char.add(unit.astype(str), ":"), (pos_in_unit // event_len).astype(str))
    y = pos_unit[unit] & (rng.random(N) < 0.75)             # positive units still contain empty frames
    hard = (~y) & (rng.random(N) < hard_neg)                  # fog / cloud / sunset look-alikes
    z = 2.8 * y - 2.2 + 1.8 * hard + u_eff[unit] + shift[su] + rng.normal(0, 1.1, N)
    s = _sig(z)
    # agent: sees the frame independently; hard negatives fool it less than the edge
    zg = agent_quality * (2 * y - 1) + 0.6 * hard + 0.5 * u_eff[unit] + rng.normal(0, 1.2, N)
    g = _sig(zg)
    nbytes = rng.lognormal(np.log(60_000), 0.4, N).astype(int)
    tag = np.where(hard, "hard_negative", "")
    return {"source": np.array(sources)[su], "unit": unit.astype(str), "event": event, "y": y.astype(int), "s": s, "g": g,
            "nbytes": nbytes, "tag": tag}
