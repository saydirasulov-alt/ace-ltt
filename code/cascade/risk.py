"""Distribution-free risk control (Learn-then-Test, Angelopoulos et al. 2021).

A configuration theta is *valid* if we can reject H0(theta): R(theta) > alpha, where R is the
expected loss of one exchangeable unit (loss in [0, 1]). p-values: Hoeffding-Bentkus (HB) for
bounded losses (Bates et al., 2021), computed from the empirical mean and the number of units.
Family-wise error control over the grid: Bonferroni (default) or fixed-sequence testing.
With probability >= 1 - delta over the calibration draw, every selected configuration has
true risk <= alpha.
"""
from __future__ import annotations

import math

import numpy as np


def _log_binom_pmf(n, p):
    k = np.arange(n + 1)
    lg = np.array([math.lgamma(n + 1)]) - np.vectorize(math.lgamma)(k + 1) - np.vectorize(math.lgamma)(n - k + 1)
    with np.errstate(divide="ignore"):
        return lg + k * math.log(p) + (n - k) * math.log1p(-p)


class BinomCDF:
    """P(Bin(n, p) <= k) for many k with fixed n, p."""

    def __init__(self, n, p):
        lp = _log_binom_pmf(n, p)
        m = lp.max()
        self.cdf = np.minimum(1.0, np.cumsum(np.exp(lp - m)) * math.exp(m))
        self.n = n

    def __call__(self, k):
        k = np.asarray(k)
        out = np.where(k < 0, 0.0, self.cdf[np.clip(k, 0, self.n)])
        return np.where(k >= self.n, 1.0, out)


def _h1(a, b):
    a = np.clip(a, 1e-12, 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hb_pvalue(rhat, n, alpha, cdf=None):
    """Hoeffding-Bentkus p-value for H0: R > alpha, given empirical risk rhat over n units."""
    rhat = np.asarray(rhat, dtype=float)
    if n <= 0:
        return np.ones_like(rhat)
    hoeff = np.exp(-n * _h1(np.minimum(rhat, alpha), alpha))
    cdf = cdf or BinomCDF(n, alpha)
    bentkus = math.e * cdf(np.ceil(n * rhat - 1e-9).astype(int))
    p = np.minimum(hoeff, bentkus)
    return np.where(rhat >= alpha, 1.0, np.minimum(p, 1.0))


def ltt_valid(pvals, delta, method="bonferroni", order=None):
    """Boolean mask of configurations certified to have risk <= alpha (FWER <= delta).
    fixed_sequence: test configurations along `order` (most-likely-safe first) until the first
    failure; every config before it is valid."""
    pvals = np.asarray(pvals).ravel()
    if method == "bonferroni":
        return pvals <= delta / max(1, pvals.size)
    if method == "fixed_sequence":
        order = np.arange(pvals.size) if order is None else np.asarray(order)
        ok = np.zeros(pvals.size, bool)
        for i in order:
            if pvals[i] > delta:
                break
            ok[i] = True
        return ok
    raise ValueError(method)


def unit_weights(y, unit):
    """Weights so that sum(w * miss) = mean over units (with >=1 positive) of the fraction of that
    unit's positives that are missed. Returns (w, n_units)."""
    y = np.asarray(y).astype(bool)
    unit = np.asarray(unit)
    w = np.zeros(len(y))
    if not y.any():
        return w, 0
    u, inv, cnt = np.unique(unit[y], return_inverse=True, return_counts=True)
    w[np.flatnonzero(y)] = 1.0 / (len(u) * cnt[inv])
    return w, len(u)
