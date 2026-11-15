#!/usr/bin/env python3
"""Empirical-heavy forecaster (jump memory) -- polymarket-faa-attribution-01, wallet w3.

Idea: Polymarket Yes prices mostly sit still, then tick, then occasionally jump.
A fixed Gaussian random walk (the house baseline) mis-prices both the quiet
minutes and the jumps. This forecaster therefore leans on the market's own
history instead of an assumed shape:

  * For each anchor horizon H in {1,3,6,12,30,60,90,180,240} minutes it takes
    the EMPIRICAL quantiles of the H-step price changes observed over the last
    2880 minutes (48 h) of the case, and blends their SHAPE (each quantile's
    deviation from the empirical median) 75% empirical / 25% Gaussian
    (sigma from median absolute 1-minute move, see below). The median itself
    stays at the last price (persistence): the empirical median of past
    returns proved a worse centre than the last price on the practice
    proxy (pooled pinball 0.399 -> 0.363), so the history informs the
    spread, not the drift.
  * Between anchors it interpolates the empirical quantiles linearly in
    sqrt(h), so the forecast widens smoothly like a random walk.
  * THIN-ANCHOR SHRINKAGE: an anchor built from few returns (a young
    market) has its empirical weight shrunk by cnt / (cnt + 25) toward the
    Gaussian, so a 13-sample anchor cannot dominate the forecast.
  * A REGIME factor compares the last 10 h of volatility with the 48 h
    window and scales the whole spread, clamped to [0.75, 1.55].
  * JUMP MEMORY: if the last 1-minute move exceeds 6 sigma, the outer
    quantiles (5% / 95%) are widened by up to 12% -- a market that just
    jumped is more likely to jump again than a quiet one.

Median forecast: the last price x (persistence). The history only shapes
the spread around it: q_tau(h) = x + 0.75 * w(h) * (emp_q_tau - emp_median)
+ 0.25 * z_tau * sigma * sqrt(h), with w(h) the thin-anchor shrinkage.

Python 3.13 standard library only. Deterministic: no randomness, no time,
no network; sorted inputs, fixed interpolation. Run per case:

    python3 main.py <case.in> [<forecast.out>]
    python3 main.py            # directory mode: $MAKEFUN_INPUT_DIR/*.in
                               # -> $MAKEFUN_OUTPUT_DIR/*.out
                               # (defaults /input and /output)
"""

import json
import math
import os
import sys

# --- fixed parameters (the w3 variant) -------------------------------------
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
ANCHORS = [1, 3, 6, 12, 30, 60, 90, 180, 240]
EMP_WEIGHT = 0.75            # weight on the empirical quantiles
WINDOW = 2880                # minutes of history used for empirical quantiles
SIGMA_FLOOR = 0.0013         # median-abs sigma, price units per sqrt(minute)
SIGMA_CAP = 0.0055
SIGMA_DEFAULT = 0.0025       # fallback when too few price changes exist
REGIME_MINUTES = 600         # 10 h recent window for the regime factor
REGIME_LO = 0.75
REGIME_HI = 1.55
JUMP_SIGMAS = 6.0            # last 1-min move above this many sigma = a jump
JUMP_MAX_BOOST = 0.12        # max widening of the outer quantiles
JUMP_SLOPE = 0.03            # extra widening per sigma beyond the threshold


def _median(sorted_vals):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _quantile(sorted_vals, tau):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _median_abs_sigma(diffs):
    """Robust 1-minute sigma from median absolute move.

    For a normal move, median(|d|) = 0.6745 * sigma. Clamped to the
    floor/cap so a dead-quiet market still gets a minimal spread and a
    wild one does not explode the forecast.
    """
    if len(diffs) < 5:
        return SIGMA_DEFAULT
    abs_sorted = sorted(abs(d) for d in diffs)
    sigma = _median(abs_sorted) / 0.67448975
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def _round6(x):
    v = round(min(1.0, max(0.0, x)), 6)
    return v + 0.0  # never -0.0


def forecast_market(prices, horizon):
    """Return `horizon` rows of 5 quantiles for one market's price list."""
    n = len(prices)
    if n == 0:
        raise ValueError("market has no prices")
    x = prices[-1]

    # 1-minute changes over the estimation window (last WINDOW minutes).
    diffs = [prices[i] - prices[i - 1] for i in range(max(1, n - WINDOW), n)]
    sigma = _median_abs_sigma(diffs)

    # Regime: recent (10 h) volatility vs window volatility.
    recent = diffs[-REGIME_MINUTES:] if len(diffs) > REGIME_MINUTES else diffs
    sigma_recent = _median_abs_sigma(recent)
    if sigma > 0:
        regime = min(REGIME_HI, max(REGIME_LO, sigma_recent / sigma))
    else:
        regime = 1.0

    # Jump memory: a large last-minute move widens the outer quantiles.
    jump_boost = 0.0
    if n >= 2 and sigma > 0:
        ratio = abs(prices[-1] - prices[-2]) / sigma
        if ratio > JUMP_SIGMAS:
            jump_boost = min(JUMP_MAX_BOOST, JUMP_SLOPE * (ratio - JUMP_SIGMAS))

    # Empirical quantiles of H-step returns at each anchor, from the window.
    # emp[H] = [q05, q25, q50, q75, q95] of (p[t] - p[t-H]), or None if the
    # market is too young for that anchor. cnt[H] counts the returns, for
    # the thin-anchor shrinkage (few samples -> trust the Gaussian more).
    emp = {}
    cnt = {}
    start = max(0, n - WINDOW)
    for h_anchor in ANCHORS:
        rets = [prices[t] - prices[t - h_anchor]
                for t in range(max(start, h_anchor), n)]
        cnt[h_anchor] = len(rets)
        if len(rets) >= 8:
            rets.sort()
            emp[h_anchor] = [_quantile(rets, tau) for tau in TAUS]
        else:
            emp[h_anchor] = None

    anchors_avail = [h for h in ANCHORS if emp[h] is not None]

    def _anchor_weight(h_anchor):
        c = cnt[h_anchor]
        return c / (c + 25.0)

    def emp_at(h):
        """(empirical quantiles, empirical weight) interpolated in sqrt(h)."""
        if not anchors_avail:
            return None, 0.0
        s = math.sqrt(h)
        if h <= anchors_avail[0]:
            a = anchors_avail[0]
            return emp[a], _anchor_weight(a)
        if h >= anchors_avail[-1]:
            a = anchors_avail[-1]
            return emp[a], _anchor_weight(a)
        for a, b in zip(anchors_avail, anchors_avail[1:]):
            if a <= h <= b:
                sa, sb = math.sqrt(a), math.sqrt(b)
                w = (s - sa) / (sb - sa) if sb > sa else 0.0
                ea, eb = emp[a], emp[b]
                eq = [ea[j] * (1.0 - w) + eb[j] * w for j in range(5)]
                ew = (_anchor_weight(a) * (1.0 - w)
                      + _anchor_weight(b) * w)
                return eq, ew
        a = anchors_avail[-1]
        return emp[a], _anchor_weight(a)

    rows = []
    for h in range(1, horizon + 1):
        sq = math.sqrt(h)
        eq, ew = emp_at(h)
        if eq is None:
            # No empirical anchor at all (very young market): pure Gaussian.
            vals = [x + Z[j] * sigma * sq for j in range(5)]
        else:
            # Persistence median; 75% empirical SHAPE (deviation from the
            # empirical median) + 25% Gaussian deviation, empirical part
            # shrunk when its anchor is thinly sampled.
            vals = [x + EMP_WEIGHT * ew * (eq[j] - eq[2])
                    + (1.0 - EMP_WEIGHT) * Z[j] * sigma * sq
                    for j in range(5)]
        # Regime scales the spread around the median, never the median.
        centre = vals[2]
        vals = [centre + (vals[j] - centre) * regime for j in range(5)]
        # Jump memory widens only the outer quantiles.
        if jump_boost:
            vals[0] = centre + (vals[0] - centre) * (1.0 + jump_boost)
            vals[4] = centre + (vals[4] - centre) * (1.0 + jump_boost)
        # Clamp, round, enforce non-decreasing quantiles.
        vals = [_round6(v) for v in vals]
        for j in range(1, 5):
            if vals[j] < vals[j - 1]:
                vals[j] = vals[j - 1]
        rows.append(vals)
    return rows


def forecast_case(case):
    horizon = int(case.get("horizon_minutes", 240))
    forecasts = []
    for m in case["markets"]:
        forecasts.append({"id": m["id"], "q": forecast_market(m["prices"], horizon)})
    return {"forecasts": forecasts}


def _run_single(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        case = json.load(f)
    out = forecast_case(case)
    text = json.dumps(out, separators=(",", ":")) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


def _run_dir(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    if not names:
        raise SystemExit("no .in case files in %s" % in_dir)
    for fn in names:
        case_id = fn[:-3]
        _run_single(os.path.join(in_dir, fn),
                    os.path.join(out_dir, case_id + ".out"))


def main(argv):
    if len(argv) >= 2:
        in_path = argv[1]
        out_path = argv[2] if len(argv) >= 3 else None
        if os.path.isdir(in_path):
            _run_dir(in_path, out_path or os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
        else:
            _run_single(in_path, out_path)
    else:
        in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
        out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
        _run_dir(in_dir, out_dir)


if __name__ == "__main__":
    main(sys.argv)
