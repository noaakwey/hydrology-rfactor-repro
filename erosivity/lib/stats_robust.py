# -*- coding: utf-8 -*-
"""
lib/stats_robust.py
===================
Robust statistical tools for trend detection in climatic / R-factor time
series.  Self-contained (no pymannkendall dependency) and seeded for
reproducibility where applicable.

Provided functions
------------------
* mann_kendall(x)             -> dict (S, Var(S), Z, p_two_sided, tau, n)
* sens_slope(x)               -> dict (slope_per_step, intercept_med, ci_low, ci_high)
* effective_sample_size(x)    -> n_eff via lag-1 autocorrelation correction
* mann_kendall_yue_pilon(x)   -> M-K with Yue-Pilon trend-free pre-whitening
* mdc_trend(sigma, n, alpha=0.05, power=0.80)  -> minimum detectable slope
* benjamini_yekutieli(p, q=0.05) -> rejected mask (handles spatially-correlated tests)
* benjamini_hochberg(p, q=0.05)  -> rejected mask (independent tests)

References
----------
Hamed, K.H., Rao, A.R. (1998). A modified Mann-Kendall trend test for
    autocorrelated data. J. Hydrol. 204, 182-196.
Yue, S., Pilon, P., Phinney, B., Cavadias, G. (2002). The influence of
    autocorrelation on the ability to detect trend in hydrological series.
    Hydrol. Process. 16, 1807-1829.
Sen, P.K. (1968). Estimates of the regression coefficient based on
    Kendall's tau. JASA 63, 1379-1389.
Benjamini, Y., Yekutieli, D. (2001). The control of the false discovery
    rate in multiple testing under dependency. Ann. Stat. 29, 1165-1188.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats as sp_stats


# ------------------------------------------------------------------ #
# Mann-Kendall (classical, for ties)
# ------------------------------------------------------------------ #
def mann_kendall(x: np.ndarray) -> dict:
    """
    Mann-Kendall trend test with tie-corrected variance.

    Returns dict with:
        S, var_S, Z, p_two_sided, tau, n
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return {"S": 0.0, "var_S": np.nan, "Z": np.nan, "p_two_sided": np.nan,
                "tau": np.nan, "n": n}

    # Compute S = sum_{i<j} sign(x_j - x_i)
    diff = x[None, :] - x[:, None]
    S = np.sign(diff[np.triu_indices(n, k=1)]).sum()

    # Tie correction
    _, counts = np.unique(x, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_S = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_S <= 0:
        Z = 0.0
    elif S > 0:
        Z = (S - 1) / math.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / math.sqrt(var_S)
    else:
        Z = 0.0
    p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(Z)))

    n_pairs = n * (n - 1) / 2.0
    tau = S / n_pairs if n_pairs > 0 else np.nan

    return {"S": float(S), "var_S": float(var_S), "Z": float(Z),
            "p_two_sided": float(p), "tau": float(tau), "n": int(n)}


# ------------------------------------------------------------------ #
# Sen's slope estimator + 90/95% CI
# ------------------------------------------------------------------ #
def sens_slope(x: np.ndarray, t: np.ndarray | None = None,
               alpha: float = 0.05) -> dict:
    """
    Sen's robust slope estimator.

    Returns dict with:
        slope_per_step, intercept_med, ci_low, ci_high
    """
    x = np.asarray(x, dtype=float)
    if t is None:
        t = np.arange(len(x), dtype=float)
    else:
        t = np.asarray(t, dtype=float)

    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    n = len(x)
    if n < 3:
        return {"slope_per_step": np.nan, "intercept_med": np.nan,
                "ci_low": np.nan, "ci_high": np.nan}

    # All pairwise slopes
    slopes = []
    for i in range(n - 1):
        dt = t[i+1:] - t[i]
        dx = x[i+1:] - x[i]
        good = dt > 0
        if good.any():
            slopes.append(dx[good] / dt[good])
    slopes = np.concatenate(slopes)
    slope = float(np.median(slopes))

    # Median intercept (Conover form)
    intercept = float(np.median(x - slope * t))

    # CI on slope: rank-based
    z_alpha = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    # Variance of S under M-K (no tie correction here for CI)
    var_S = n * (n - 1) * (2 * n + 5) / 18.0
    C = z_alpha * math.sqrt(var_S)
    N_pairs = len(slopes)
    M1 = int(round((N_pairs - C) / 2.0))
    M2 = int(round((N_pairs + C) / 2.0))
    s_sorted = np.sort(slopes)
    if 0 <= M1 < N_pairs and 0 <= M2 < N_pairs:
        ci_low = float(s_sorted[M1])
        ci_high = float(s_sorted[M2])
    else:
        ci_low = ci_high = np.nan

    return {"slope_per_step": slope, "intercept_med": intercept,
            "ci_low": ci_low, "ci_high": ci_high}


# ------------------------------------------------------------------ #
# Effective sample size from lag-1 autocorrelation
# ------------------------------------------------------------------ #
def effective_sample_size(x: np.ndarray) -> Tuple[int, float]:
    """
    Standard Bayley-Hammersley n_eff = n * (1 - rho1) / (1 + rho1).
    Returns (n_eff_int, rho1).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return n, np.nan
    x0 = x - x.mean()
    rho1 = float(np.corrcoef(x0[:-1], x0[1:])[0, 1]) if n > 2 else 0.0
    if not np.isfinite(rho1):
        rho1 = 0.0
    rho1 = max(min(rho1, 0.99), -0.99)
    factor = (1.0 - rho1) / (1.0 + rho1)
    # Cap below at 0.05*n (avoid degenerate n_eff for near-1 autocorr) and
    # above at n (negative rho would otherwise give n_eff>n which has no
    # sensible interpretation for trend testing).
    factor = min(max(factor, 0.05), 1.0)
    n_eff = max(2, int(round(n * factor)))
    return n_eff, rho1


# ------------------------------------------------------------------ #
# Yue-Pilon trend-free pre-whitening + Mann-Kendall
# ------------------------------------------------------------------ #
def mann_kendall_yue_pilon(x: np.ndarray) -> dict:
    """
    Mann-Kendall after Yue-Pilon trend-free pre-whitening (TFPW):
        1. Estimate slope by Sen's slope.
        2. Detrend.
        3. Estimate lag-1 autocorrelation rho1 of detrended series.
        4. Pre-whiten:  y_t = x_t - rho1 * x_{t-1}
        5. Add the trend back.
        6. Run M-K on the resulting series.

    Returns the M-K dict plus rho1 and the Sen slope used for detrending.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        out = mann_kendall(x)
        out.update({"rho1": np.nan, "slope_used_for_tfpw": np.nan})
        return out

    sen = sens_slope(x)
    slope = sen["slope_per_step"]
    if not np.isfinite(slope):
        slope = 0.0

    # Detrend
    t = np.arange(n, dtype=float)
    detr = x - slope * t

    # Lag-1 autocorr of detrended
    d0 = detr - detr.mean()
    rho1 = float(np.corrcoef(d0[:-1], d0[1:])[0, 1]) if n > 2 else 0.0
    if not np.isfinite(rho1):
        rho1 = 0.0
    rho1 = max(min(rho1, 0.99), -0.99)

    # Pre-whiten detrended series, then add back the trend.
    # Variance-preserving form:  y_pw = (Y_t - rho*Y_{t-1}) / sqrt(1 - rho^2).
    # Without this rescale the residual variance shrinks by (1-rho^2), which
    # makes the subsequent M-K Z artificially large and the test
    # anti-conservative under AR(1) noise.
    if abs(rho1) < 1e-3:
        y = x.copy()
    else:
        denom = math.sqrt(max(1.0 - rho1 * rho1, 1e-9))
        y_pw = (detr[1:] - rho1 * detr[:-1]) / denom
        y = y_pw + slope * t[1:]

    out = mann_kendall(y)
    out["rho1"] = rho1
    out["slope_used_for_tfpw"] = slope
    return out


# ------------------------------------------------------------------ #
# Hamed-Rao Modified Mann-Kendall (variance-correction approach)
# ------------------------------------------------------------------ #
def mann_kendall_hamed_rao(x: np.ndarray, max_lag: int | None = None) -> dict:
    """
    Modified Mann-Kendall test of Hamed & Rao (1998).

    Variance of S is multiplied by a correction factor n/n_s* derived from
    the significant autocorrelation coefficients of the rank-detrended
    series.  This is generally more reliable than Yue-Pilon TFPW for
    moderate-length hydrological series.

        Var*(S) = Var(S) * n / n_s*
        n / n_s* = 1 + (2/(n*(n-1)*(n-2))) *
                   sum_{k=1}^{m} (n-k)(n-k-1)(n-k-2) * rho_k

    Only autocorrelations with |rho_k| > z_{0.025}/sqrt(n - k) (at 5%) are
    retained.

    Returns the standard mann_kendall dict with corrected p-value, plus
    `n_over_ns` (the variance correction multiplier).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        out = mann_kendall(x)
        out["n_over_ns"] = 1.0
        return out

    base = mann_kendall(x)

    # Detrend by Sen's slope
    sen = sens_slope(x)
    slope = sen["slope_per_step"] if np.isfinite(sen["slope_per_step"]) else 0.0
    t = np.arange(n, dtype=float)
    detr = x - slope * t

    # Sample autocorrelations of detrended series (use ranks per Hamed-Rao)
    ranks = sp_stats.rankdata(detr)
    r0 = ranks - ranks.mean()
    var_r = float(np.dot(r0, r0))
    if var_r <= 0:
        out = dict(base)
        out["n_over_ns"] = 1.0
        return out

    if max_lag is None:
        max_lag = max(1, min(n - 4, int(round(n / 4.0))))

    z_crit = sp_stats.norm.ppf(0.975)
    correction = 0.0
    for k in range(1, max_lag + 1):
        rho_k = float(np.dot(r0[:-k], r0[k:]) / var_r)
        # Significance test for rho_k under H0: rho=0
        ci_half = z_crit / math.sqrt(n - k)
        if abs(rho_k) > ci_half:
            correction += (n - k) * (n - k - 1) * (n - k - 2) * rho_k

    n_over_ns = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * correction
    n_over_ns = max(n_over_ns, 1e-3)  # safety

    var_S_corr = base["var_S"] * n_over_ns
    if var_S_corr <= 0 or not np.isfinite(var_S_corr):
        Z = 0.0
    elif base["S"] > 0:
        Z = (base["S"] - 1) / math.sqrt(var_S_corr)
    elif base["S"] < 0:
        Z = (base["S"] + 1) / math.sqrt(var_S_corr)
    else:
        Z = 0.0
    p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(Z)))

    out = dict(base)
    out.update({
        "Z": float(Z),
        "p_two_sided": float(p),
        "var_S_corrected": float(var_S_corr),
        "n_over_ns": float(n_over_ns),
    })
    return out


# ------------------------------------------------------------------ #
# Minimum detectable change (MDC) for a trend
# ------------------------------------------------------------------ #
def mdc_trend(sigma: float, n: int, alpha: float = 0.05,
              power: float = 0.80) -> float:
    """
    Approximate minimum detectable trend slope for OLS regression on n
    equally-spaced years, given residual sigma.

    Formula (Lettenmaier 1976; Kendall & Stuart):
        slope_MDC = (z_{1-alpha/2} + z_{power}) * sigma * sqrt(12 / (n^3 - n))

    Returns slope in units of x per unit step (yr if t is in yr).
    """
    if n < 4 or not np.isfinite(sigma):
        return np.nan
    z_a = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    z_b = sp_stats.norm.ppf(power)
    return float((z_a + z_b) * sigma * math.sqrt(12.0 / (n ** 3 - n)))


# ------------------------------------------------------------------ #
# Benjamini-Hochberg FDR (assumes independent tests)
# ------------------------------------------------------------------ #
def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Return boolean mask of rejections under BH FDR control."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p, dtype=bool)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pf = p[finite]
    n = pf.size
    order = np.argsort(pf)
    ranked = pf[order]
    thresh = np.arange(1, n + 1) * q / n
    pass_mask = ranked <= thresh
    if pass_mask.any():
        max_k = np.where(pass_mask)[0].max()
        rejected_in_pf = np.zeros_like(pf, dtype=bool)
        rejected_in_pf[order[: max_k + 1]] = True
        out[finite] = rejected_in_pf
    return out


# ------------------------------------------------------------------ #
# Benjamini-Yekutieli FDR (handles arbitrary dependence; conservative)
# ------------------------------------------------------------------ #
def benjamini_yekutieli(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """
    BY FDR control under arbitrary dependence (e.g. spatially-correlated
    pixel-wise tests).  More conservative than BH by factor c(m) = sum 1/i.
    """
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p, dtype=bool)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pf = p[finite]
    n = pf.size
    cm = float(np.sum(1.0 / np.arange(1, n + 1)))
    order = np.argsort(pf)
    ranked = pf[order]
    thresh = np.arange(1, n + 1) * q / (n * cm)
    pass_mask = ranked <= thresh
    if pass_mask.any():
        max_k = np.where(pass_mask)[0].max()
        rejected_in_pf = np.zeros_like(pf, dtype=bool)
        rejected_in_pf[order[: max_k + 1]] = True
        out[finite] = rejected_in_pf
    return out


# ------------------------------------------------------------------ #
# Convenience wrapper: full trend report
# ------------------------------------------------------------------ #
@dataclass
class TrendReport:
    n: int
    n_eff: int
    rho1: float
    sigma_resid: float
    ols_slope: float
    ols_p: float
    mk_p: float
    mk_tau: float
    mk_p_yp: float
    mk_p_hr: float
    n_over_ns_hr: float
    sen_slope: float
    sen_ci_low: float
    sen_ci_high: float
    mdc_at_alpha05_pow80: float

    def as_text(self) -> str:
        return (
            f"n={self.n}, n_eff={self.n_eff}, lag-1 rho={self.rho1:+.3f}\n"
            f"OLS:                slope={self.ols_slope:+.3f}/yr, p={self.ols_p:.3f}\n"
            f"Mann-Kendall:       p={self.mk_p:.3f}, tau={self.mk_tau:+.3f}\n"
            f"M-K Yue-Pilon TFPW: p={self.mk_p_yp:.3f}\n"
            f"M-K Hamed-Rao MMK:  p={self.mk_p_hr:.3f}, var-correction n/n*={self.n_over_ns_hr:.3f}\n"
            f"Sen's slope:        {self.sen_slope:+.3f}/yr  [{self.sen_ci_low:+.3f}, {self.sen_ci_high:+.3f}] (95% CI)\n"
            f"MDC (alpha=0.05, power=0.80): |slope| >= {self.mdc_at_alpha05_pow80:.3f}/yr"
        )


def trend_report(x: np.ndarray, years: np.ndarray | None = None) -> TrendReport:
    x = np.asarray(x, dtype=float)
    n = int(np.sum(np.isfinite(x)))
    if years is None:
        t = np.arange(len(x), dtype=float)
    else:
        t = np.asarray(years, dtype=float)

    n_eff, rho1 = effective_sample_size(x)

    # OLS
    s, b0, _r, p_ols, _se = sp_stats.linregress(t, x)
    resid = x - (s * t + b0)
    sigma = float(np.std(resid, ddof=1))

    mk = mann_kendall(x)
    mk_yp = mann_kendall_yue_pilon(x)
    mk_hr = mann_kendall_hamed_rao(x)
    sen = sens_slope(x, t)
    mdc = mdc_trend(sigma, n_eff if np.isfinite(n_eff) and n_eff >= 4 else n, alpha=0.05, power=0.80)

    return TrendReport(
        n=n, n_eff=int(n_eff), rho1=float(rho1) if np.isfinite(rho1) else 0.0,
        sigma_resid=sigma, ols_slope=float(s), ols_p=float(p_ols),
        mk_p=float(mk["p_two_sided"]), mk_tau=float(mk["tau"]),
        mk_p_yp=float(mk_yp["p_two_sided"]),
        mk_p_hr=float(mk_hr["p_two_sided"]),
        n_over_ns_hr=float(mk_hr.get("n_over_ns", 1.0)),
        sen_slope=float(sen["slope_per_step"]),
        sen_ci_low=float(sen["ci_low"]), sen_ci_high=float(sen["ci_high"]),
        mdc_at_alpha05_pow80=float(mdc),
    )
