# -*- coding: utf-8 -*-
"""
Unit tests for src/qm_advanced.py.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calibration.qm_advanced import (
    _fit_gpd,
    _gpd_cdf,
    _gpd_quantile,
    apply_advanced_qm,
    apply_pop_mask,
    fit_advanced_qm,
    fit_pop_threshold,
)


# ------------------------------------------------------------------ #
# A. PoP threshold
# ------------------------------------------------------------------ #
def test_pop_zero_when_station_dry():
    sat = np.array([0.0, 0.5, 1.0, 2.0, 0.0])
    st = np.zeros_like(sat)
    p_th = fit_pop_threshold(sat, st)
    assert p_th == float("inf")


def test_pop_threshold_matches_frequency():
    """If station is wet 30% of the time, the IMERG threshold should
    equal the 70th percentile of IMERG so that 30% of IMERG remains wet."""
    rng = np.random.default_rng(0)
    n = 5000
    # Station: 30% wet (Exp(1) scaled)
    st_wet_idx = rng.choice(n, size=int(0.3 * n), replace=False)
    st = np.zeros(n)
    st[st_wet_idx] = rng.exponential(1.0, size=len(st_wet_idx))
    # Satellite: 50% wet (Exp(0.7))
    sat_wet_idx = rng.choice(n, size=int(0.5 * n), replace=False)
    sat = np.zeros(n)
    sat[sat_wet_idx] = rng.exponential(0.7, size=len(sat_wet_idx))

    p_th = fit_pop_threshold(sat, st)
    # Expected: 70th percentile of sat
    expected = float(np.quantile(sat, 0.70))
    assert math.isclose(p_th, expected, abs_tol=1e-9)


def test_apply_pop_mask_zeroes_below_threshold():
    arr = np.array([0.0, 0.1, 0.5, 1.2, 5.0])
    out = apply_pop_mask(arr, p_th=0.5)
    # values <= 0.5 zeroed
    assert (out == np.array([0.0, 0.0, 0.0, 1.2, 5.0])).all()


# ------------------------------------------------------------------ #
# B. GPD tail
# ------------------------------------------------------------------ #
def test_gpd_fit_recovers_known_exponential():
    """Pure Exponential(1) sample: excesses above any threshold are
    Exponential(1) by memorylessness, so GPD MLE -> xi ~ 0, sigma ~ 1."""
    rng = np.random.default_rng(2)
    sample = rng.exponential(1.0, size=8000)
    tail = _fit_gpd(sample, threshold_q=0.70)
    assert tail.valid
    # xi for memoryless exponential -> 0
    assert -0.25 < tail.xi < 0.25, f"xi={tail.xi}"
    # sigma -> mean of excesses ~ 1
    assert 0.7 < tail.sigma < 1.4, f"sigma={tail.sigma}"


def test_gpd_quantile_monotone_above_threshold():
    rng = np.random.default_rng(3)
    sample = np.concatenate([rng.exponential(1.0, 5000)])
    tail = _fit_gpd(sample, threshold_q=0.95)
    assert tail.valid
    Fs = np.linspace(0.96, 0.999, 30)
    qs = np.array([_gpd_quantile(F, tail) for F in Fs])
    assert (np.diff(qs) >= -1e-9).all(), "GPD quantile not monotone"


def test_gpd_cdf_inverse_consistency():
    rng = np.random.default_rng(4)
    sample = rng.exponential(1.5, 5000)
    tail = _fit_gpd(sample, threshold_q=0.90)
    assert tail.valid
    for F in [0.92, 0.95, 0.99, 0.995]:
        v = _gpd_quantile(F, tail)
        F_back = _gpd_cdf(v, tail)
        assert math.isclose(F, F_back, abs_tol=1e-3)


# ------------------------------------------------------------------ #
# Combined fit + apply
# ------------------------------------------------------------------ #
def test_fit_advanced_qm_returns_model():
    rng = np.random.default_rng(5)
    n = 5000
    st = np.zeros(n)
    st_wet = rng.choice(n, size=2000, replace=False)
    st[st_wet] = rng.exponential(1.0, size=2000)

    sat = np.zeros(n)
    sat_wet = rng.choice(n, size=3000, replace=False)
    sat[sat_wet] = rng.exponential(0.7, size=3000)

    model = fit_advanced_qm(sat, st)
    assert model is not None
    assert model.q_sat.size == 1000
    assert model.q_station.size == 1000
    assert model.p_th >= 0
    assert model.tail_sat.valid
    assert model.tail_station.valid


def test_apply_returns_zero_below_pop_threshold():
    rng = np.random.default_rng(6)
    n = 5000
    st = rng.exponential(1.0, n) * (rng.random(n) < 0.3)  # 30% wet
    sat = rng.exponential(0.7, n) * (rng.random(n) < 0.5)  # 50% wet
    model = fit_advanced_qm(sat, st)
    assert model is not None
    test_vals = np.array([0.0, model.p_th * 0.5, model.p_th, model.p_th * 1.1, 100.0])
    out = apply_advanced_qm(test_vals, model)
    # Strictly below or equal to p_th -> 0
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == 0.0
    # Above threshold -> non-zero
    assert out[3] > 0.0
    assert out[4] > 0.0


def test_apply_monotone_in_input():
    """Mapping should be (weakly) monotonically increasing in input."""
    rng = np.random.default_rng(7)
    n = 6000
    st = rng.exponential(1.0, n) * (rng.random(n) < 0.4)
    sat = rng.exponential(0.7, n) * (rng.random(n) < 0.6)
    model = fit_advanced_qm(sat, st)
    assert model is not None
    grid = np.concatenate([np.array([0.0]), np.linspace(0.01, 50.0, 300)])
    out = apply_advanced_qm(grid, model)
    diffs = np.diff(out)
    # Allow tiny negatives due to float; -1e-6 is generous
    assert (diffs >= -1e-6).all(), f"non-monotone: min diff = {diffs.min()}"


def test_volume_factor_anchors_mean_or_tail():
    """When VF blend_w=0 (mean only), mean(corrected wet) should approx
    mean(station wet); when blend_w=1 (tail only), tail mean should match."""
    rng = np.random.default_rng(8)
    n = 6000
    st = np.zeros(n)
    idx = rng.choice(n, 2400, replace=False)
    st[idx] = rng.exponential(1.0, size=2400)
    sat = np.zeros(n)
    idx = rng.choice(n, 3500, replace=False)
    sat[idx] = rng.exponential(0.5, size=3500)

    # Without tail blending
    m0 = fit_advanced_qm(sat, st, vf_tail_blend_w=0.0)
    out0 = apply_advanced_qm(sat, m0)
    pbias_mean = (out0.sum() - st.sum()) / st.sum() * 100
    assert abs(pbias_mean) < 30.0, f"mean-anchored PBIAS = {pbias_mean:.1f}%"

    # With tail-only weighting we anchor the upper tail; mean PBIAS is allowed to drift
    m1 = fit_advanced_qm(sat, st, vf_tail_blend_w=1.0)
    out1 = apply_advanced_qm(sat, m1)
    st_tail = st[st > np.quantile(st[st > 0], 0.95)]
    out_tail = out1[out1 > np.quantile(out1[out1 > 0], 0.95)]
    if out_tail.size and st_tail.size:
        pbias_tail = (out_tail.mean() - st_tail.mean()) / st_tail.mean() * 100
        # Tail anchor isn't perfect because GPD-extrapolated values still
        # depend on satellite ranks; accept within 25% of station tail mean.
        assert abs(pbias_tail) < 25.0, f"tail PBIAS = {pbias_tail:.1f}%"


def test_fallback_when_pop_disabled():
    rng = np.random.default_rng(9)
    n = 4000
    st = rng.exponential(1.0, n) * (rng.random(n) < 0.4)
    sat = rng.exponential(0.7, n) * (rng.random(n) < 0.6)
    model = fit_advanced_qm(sat, st, use_pop=False, use_gpd_tail=False)
    assert model is not None
    assert model.p_th == 0.0
    assert not model.tail_sat.valid
    assert not model.tail_station.valid


# ------------------------------------------------------------------ #
# Standalone runner
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import inspect
    failed = 0; passed = 0
    for name, fn in dict(globals()).items():
        if name.startswith("test_") and callable(fn) and len(inspect.signature(fn).parameters) == 0:
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as exc:
                print(f"  FAIL  {name}: {exc}")
                failed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {name}: {exc!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
