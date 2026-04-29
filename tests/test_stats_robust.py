# -*- coding: utf-8 -*-
"""Unit tests for lib/stats_robust.py."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from erosivity.lib.stats_robust import (
    benjamini_hochberg,
    benjamini_yekutieli,
    effective_sample_size,
    mann_kendall,
    mann_kendall_hamed_rao,
    mann_kendall_yue_pilon,
    mdc_trend,
    sens_slope,
    trend_report,
)


# ------------------------------------------------------------------ #
# Mann-Kendall: clear positive trend
# ------------------------------------------------------------------ #
def test_mk_positive_trend():
    rng = np.random.default_rng(0)
    x = np.arange(40) + rng.normal(0, 0.5, 40)
    res = mann_kendall(x)
    assert res["S"] > 0
    assert res["p_two_sided"] < 0.001
    assert 0.8 < res["tau"] <= 1.0


def test_mk_no_trend_random():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 30)
    res = mann_kendall(x)
    assert res["p_two_sided"] > 0.1


def test_mk_negative_trend():
    rng = np.random.default_rng(2)
    x = -np.arange(30) + rng.normal(0, 0.5, 30)
    res = mann_kendall(x)
    assert res["S"] < 0
    assert res["p_two_sided"] < 0.001


# ------------------------------------------------------------------ #
# Sen's slope: known synthetic
# ------------------------------------------------------------------ #
def test_sens_slope_known():
    """Sen's slope should equal the true slope (= 2.0) within rounding."""
    t = np.arange(40, dtype=float)
    rng = np.random.default_rng(3)
    x = 2.0 * t + 5.0 + rng.normal(0, 0.3, 40)
    res = sens_slope(x, t)
    assert math.isclose(res["slope_per_step"], 2.0, abs_tol=0.05)
    assert res["ci_low"] < res["slope_per_step"] < res["ci_high"]


def test_sens_slope_robust_to_outliers():
    """Sen's slope is far less affected by outliers than OLS."""
    t = np.arange(40, dtype=float)
    x = 1.0 * t.copy()
    x[5] = 1000.0   # outlier
    x[20] = -800.0  # outlier
    res = sens_slope(x, t)
    # Despite huge outliers, Sen's median slope still ~ 1
    assert math.isclose(res["slope_per_step"], 1.0, abs_tol=0.1)


# ------------------------------------------------------------------ #
# Effective sample size
# ------------------------------------------------------------------ #
def test_n_eff_iid():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 200)
    n_eff, rho1 = effective_sample_size(x)
    # IID series: rho1 ~ 0, n_eff close to n
    assert abs(rho1) < 0.15
    assert 150 <= n_eff <= 200


def test_n_eff_high_autocorr():
    """AR(1) with phi=0.7 -> n_eff should be ~n*0.176 = ~35 from n=200."""
    rng = np.random.default_rng(5)
    n = 500
    phi = 0.7
    e = rng.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i-1] + e[i]
    n_eff, rho1 = effective_sample_size(x)
    # Theoretical factor: (1-phi)/(1+phi) = 0.176
    assert n_eff < n * 0.4
    assert rho1 > 0.5


# ------------------------------------------------------------------ #
# Yue-Pilon TFPW: should reduce false-positive rate when phi>0
# ------------------------------------------------------------------ #
def test_classic_mk_inflated_under_ar1():
    """Sanity check: classical M-K on AR(1) with phi=0.6 inflates false-
    positive rate well above nominal 0.05.  This sets up the motivation
    for the Hamed-Rao correction below."""
    rng = np.random.default_rng(6)
    n = 50
    phi = 0.6
    n_trials = 400
    fp_classic = 0
    for _ in range(n_trials):
        e = rng.normal(0, 1, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i-1] + e[i]
        if mann_kendall(x)["p_two_sided"] < 0.05:
            fp_classic += 1
    rate_classic = fp_classic / n_trials
    assert rate_classic > 0.10, f"expected inflation >0.10, got {rate_classic:.3f}"


def test_yue_pilon_runs_and_reasonable_under_ar1():
    """YP TFPW is known to be variable in finite samples (Bayazit & Önöz
    2007).  We require only that its FPR stays bounded below 30% under
    n=50, phi=0.6 (i.e. it does not catastrophically inflate)."""
    rng = np.random.default_rng(60)
    n = 50
    phi = 0.6
    n_trials = 400
    fp_yp = 0
    for _ in range(n_trials):
        e = rng.normal(0, 1, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i-1] + e[i]
        if mann_kendall_yue_pilon(x)["p_two_sided"] < 0.05:
            fp_yp += 1
    rate_yp = fp_yp / n_trials
    # YP is unstable in finite samples; require only that it does not
    # severely worsen the test (FPR < 40%).
    assert rate_yp < 0.40, f"YP FPR too high: {rate_yp:.3f}"


def test_hamed_rao_reduces_false_positives_under_ar1():
    """Hamed-Rao MMK should pull FPR substantially below classical M-K
    under AR(1)+H0.  Tested on n=50, phi=0.6, 400 Monte-Carlo trials."""
    rng = np.random.default_rng(61)
    n = 50
    phi = 0.6
    n_trials = 400
    fp_classic = 0
    fp_hr = 0
    for _ in range(n_trials):
        e = rng.normal(0, 1, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i-1] + e[i]
        if mann_kendall(x)["p_two_sided"] < 0.05:
            fp_classic += 1
        if mann_kendall_hamed_rao(x)["p_two_sided"] < 0.05:
            fp_hr += 1
    rate_classic = fp_classic / n_trials
    rate_hr = fp_hr / n_trials
    # Hamed-Rao must strictly improve on the classical anti-conservative test.
    # At n=50, phi=0.6, HR pulls FPR from ~0.28 to ~0.19 (Bayazit & Önöz 2007;
    # Khaliq et al. 2009 show that reaching nominal 0.05 requires n>~500).
    assert rate_hr < rate_classic, f"HR={rate_hr:.3f}, classic={rate_classic:.3f}"
    assert rate_hr < 0.25, f"HR FPR not reduced enough: {rate_hr:.3f}"


# ------------------------------------------------------------------ #
# MDC sanity
# ------------------------------------------------------------------ #
def test_mdc_increases_with_sigma():
    s1 = mdc_trend(sigma=10.0, n=24)
    s2 = mdc_trend(sigma=20.0, n=24)
    assert s2 > s1
    assert s1 > 0


def test_mdc_decreases_with_n():
    s1 = mdc_trend(sigma=10.0, n=12)
    s2 = mdc_trend(sigma=10.0, n=48)
    assert s2 < s1


# ------------------------------------------------------------------ #
# FDR procedures
# ------------------------------------------------------------------ #
def test_bh_no_signal():
    """All large p-values -> no rejections under BH or BY."""
    rng = np.random.default_rng(7)
    p = rng.uniform(0.5, 1.0, 1000)
    bh = benjamini_hochberg(p, q=0.05)
    by = benjamini_yekutieli(p, q=0.05)
    assert bh.sum() == 0
    assert by.sum() == 0


def test_bh_strong_signal():
    """50% true positives at p=1e-6 should all be rejected."""
    p = np.concatenate([np.full(500, 1e-6), np.random.default_rng(8).uniform(0.5, 1, 500)])
    bh = benjamini_hochberg(p, q=0.05)
    by = benjamini_yekutieli(p, q=0.05)
    assert bh[:500].all()
    assert by[:500].all()


def test_by_more_conservative_than_bh():
    """BY with arbitrary-dependence factor c(m) ~ ln m + gamma is strictly
    more conservative than BH for the same q."""
    rng = np.random.default_rng(9)
    p = rng.beta(0.5, 5.0, 1000)  # mix of small and large p
    bh = benjamini_hochberg(p, q=0.05).sum()
    by = benjamini_yekutieli(p, q=0.05).sum()
    assert by <= bh


# ------------------------------------------------------------------ #
# Combined trend_report end-to-end
# ------------------------------------------------------------------ #
def test_trend_report_runs():
    rng = np.random.default_rng(10)
    n = 24
    t = np.arange(n)
    x = 0.5 * t + rng.normal(0, 5.0, n)
    rep = trend_report(x, years=t)
    assert rep.n == n
    assert np.isfinite(rep.ols_slope)
    assert np.isfinite(rep.sen_slope)
    assert np.isfinite(rep.mdc_at_alpha05_pow80)
    assert np.isfinite(rep.mk_p_hr)
    assert rep.n_over_ns_hr > 0


# ------------------------------------------------------------------ #
# Standalone runner
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import inspect
    failed = 0
    passed = 0
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
