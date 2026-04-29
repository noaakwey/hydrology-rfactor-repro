# -*- coding: utf-8 -*-
"""
src/qm_advanced.py
==================
Source-side improvements to the IMERG quantile-mapping calibration.

The legacy IMERG fit (``qm_calibration.fit_qm_transfer``) is essentially
a wet-day full-distribution QM with a coarse linear tail extrapolation
and no probability-of-precipitation (PoP) frequency adaptation.  Three
gaps are addressed here:

  A. PoP / frequency adaptation
       IMERG V07 has a documented light-precipitation Probability of
       Detection (POD) deficit (Tan et al. 2019; Pradhan et al. 2022):
       station-recorded events at <0.3 mm/3h are often missed.  Without
       PoP-matching, downstream QM amplifies the few small IMERG values
       that do exist into station-distribution percentiles, biasing the
       light-rain mode upward.  We compute the IMERG threshold p_th such
       that
           P(P_sat > p_th) = P(P_station > 0)
       on the training window, then map only values > p_th, sending
       sub-threshold values to zero.  This restores correct wet-day
       frequency before any quantile mapping.

  B. GPD upper-tail extrapolation
       The legacy method uses the slope of the last 5 of 1000 quantiles
       (i.e. 99.5–99.9 percentile band) to extrapolate values above
       q_sat[-1].  At small wet-day samples this slope is noisy, and at
       extreme intensities the linear assumption diverges from the
       observed power-law / Pareto behaviour.  We instead fit a
       Generalized Pareto Distribution to the upper tail of station and
       satellite (above the 95th percentile of wet days), and define a
       distribution-matching extrapolation
           q_station(F) = u_station + (sigma_st/xi_st) * ((1-F)^(-xi_st) - 1)
           q_sat(F)     = u_sat     + (sigma_sat/xi_sat) * ((1-F)^(-xi_sat) - 1)
       so that for any input value v >= q_sat[-1]:
           F = 1 - (1 + xi_sat * (v - u_sat) / sigma_sat)^(-1/xi_sat)
           v_corrected = q_station(F)
       This is theoretically grounded by the Pickands-Balkema-de Haan
       theorem and is far more stable for the upper tail that drives R.

  F. Tail-quantile-anchored Volume Factor
       The legacy multiplicative VF anchors mean(corrected) to
       mean(station), but R-relevant intensities lie in P95-P99.99.
       We add an alternative VF that anchors quantile-95-99 means.
       Optionally we use a weighted blend of mean-VF and tail-VF, with
       weights specified at fit time.

A regression test against the legacy fit is in
``tests/test_qm_advanced.py`` (cross-repository).

References
----------
Tan, J., Petersen, W.A., Kirstetter, P.E., Tian, Y. (2019). Performance
    of IMERG as a function of spatiotemporal scale. *J. Hydrometeor.*
    18, 1819-1839.
Pickands, J. (1975). Statistical inference using extreme order
    statistics. *Ann. Stat.* 3, 119-131.
Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme
    Values*.  Springer.
Cannon, A.J., Sobie, S.R., Murdock, T.Q. (2015).  Bias correction of
    GCM precipitation by quantile mapping. *J. Climate* 28, 6938-6959.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


# ------------------------------------------------------------------ #
# A. PoP / frequency adaptation
# ------------------------------------------------------------------ #
def fit_pop_threshold(p_sat: np.ndarray, p_station: np.ndarray) -> float:
    """
    Return the IMERG threshold p_th [mm] such that the fraction of
    IMERG values exceeding p_th equals the fraction of station values
    exceeding zero.  Below p_th IMERG should be zeroed before any QM
    is applied.

    If the station record is universally dry (PoP_st = 0), p_th = +inf.
    If IMERG is universally wet at the station threshold, p_th = 0.
    """
    p_sat = np.asarray(p_sat, dtype=float)
    p_station = np.asarray(p_station, dtype=float)
    if p_sat.size == 0 or p_station.size == 0:
        return 0.0
    pop_st = float(np.mean(p_station > 0.0))
    if pop_st <= 0.0:
        return float(np.inf)
    if pop_st >= 1.0:
        return 0.0
    p_th = float(np.quantile(p_sat, 1.0 - pop_st))
    return max(p_th, 0.0)


def apply_pop_mask(p_sat: np.ndarray, p_th: float) -> np.ndarray:
    """Zero values <= p_th; keep others.  Pre-QM step."""
    out = np.asarray(p_sat, dtype=float).copy()
    if not np.isfinite(p_th):
        out[:] = 0.0
        return out
    out[out <= p_th] = 0.0
    return out


# ------------------------------------------------------------------ #
# B. GPD upper-tail extrapolation
# ------------------------------------------------------------------ #
@dataclass
class GPDTail:
    """Fitted GPD tail parameters for one side (station or satellite)."""
    threshold_u: float        # location (top-quantile threshold above which GPD holds)
    sigma: float              # scale
    xi: float                 # shape
    F_at_u: float             # CDF value at threshold (fraction <= u)
    n_exceed: int             # number of points used in the fit
    valid: bool               # True if the fit succeeded


def _fit_gpd(values: np.ndarray, threshold_q: float = 0.95,
             min_exceed: int = 30) -> GPDTail:
    """Fit a GPD to exceedances over the threshold_q-quantile."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < min_exceed * 2:
        return GPDTail(0.0, 0.0, 0.0, threshold_q, 0, valid=False)

    u = float(np.quantile(values, threshold_q))
    excesses = values[values > u] - u
    if excesses.size < min_exceed:
        return GPDTail(u, 0.0, 0.0, threshold_q, int(excesses.size), valid=False)

    # MLE for GPD
    try:
        xi, _loc, sigma = sp_stats.genpareto.fit(excesses, floc=0.0)
    except Exception:  # noqa: BLE001
        return GPDTail(u, 0.0, 0.0, threshold_q, int(excesses.size), valid=False)

    if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
        return GPDTail(u, 0.0, 0.0, threshold_q, int(excesses.size), valid=False)

    # Cap xi to a physically sensible range:
    #   xi < -0.5 :  short tail (rarely seen for precipitation)
    #   xi >  0.7 :  super-Pareto, often a fitting artefact
    xi = float(np.clip(xi, -0.5, 0.7))

    return GPDTail(threshold_u=u, sigma=float(sigma), xi=xi,
                   F_at_u=threshold_q, n_exceed=int(excesses.size),
                   valid=True)


def _gpd_quantile(F: float, tail: GPDTail) -> float:
    """Return the value at cumulative probability F under the GPD tail."""
    if not tail.valid or F <= tail.F_at_u:
        return tail.threshold_u
    F = float(min(max(F, tail.F_at_u + 1e-9), 1.0 - 1e-12))
    p_excess = (F - tail.F_at_u) / (1.0 - tail.F_at_u)
    if abs(tail.xi) < 1e-6:
        return tail.threshold_u + tail.sigma * (-np.log1p(-p_excess))
    return tail.threshold_u + (tail.sigma / tail.xi) * ((1.0 - p_excess) ** (-tail.xi) - 1.0)


def _gpd_cdf(value: float, tail: GPDTail) -> float:
    """Return F at value under the GPD tail (exceedance side)."""
    if not tail.valid or value <= tail.threshold_u:
        return tail.F_at_u
    z = (value - tail.threshold_u) / max(tail.sigma, 1e-9)
    if abs(tail.xi) < 1e-6:
        p_excess = 1.0 - np.exp(-z)
    else:
        base = 1.0 + tail.xi * z
        if base <= 0.0:
            p_excess = 1.0  # outside support; treat as far tail
        else:
            p_excess = 1.0 - base ** (-1.0 / tail.xi)
    return float(tail.F_at_u + p_excess * (1.0 - tail.F_at_u))


# ------------------------------------------------------------------ #
# Combined fit: PoP + EQM body + GPD tail
# ------------------------------------------------------------------ #
@dataclass
class AdvancedQMModel:
    p_th: float                                  # PoP-derived IMERG threshold
    q_levels: np.ndarray = field(repr=False)
    q_sat: np.ndarray = field(repr=False)
    q_station: np.ndarray = field(repr=False)
    tail_sat: GPDTail = field(default_factory=lambda: GPDTail(0, 0, 0, 1.0, 0, False))
    tail_station: GPDTail = field(default_factory=lambda: GPDTail(0, 0, 0, 1.0, 0, False))
    vf_mean: float = 1.0
    vf_tail: float = 1.0
    vf_blend_w: float = 0.5      # 0 = mean-only, 1 = tail-only

    def effective_vf(self) -> float:
        return float((1.0 - self.vf_blend_w) * self.vf_mean
                     + self.vf_blend_w * self.vf_tail)


def fit_advanced_qm(
    p_sat: np.ndarray,
    p_station: np.ndarray,
    num_quantiles: int = 1000,
    tail_threshold_q: float = 0.95,
    use_pop: bool = True,
    use_gpd_tail: bool = True,
    vf_clip: Tuple[float, float] = (0.5, 2.0),
    vf_tail_blend_w: float = 0.5,
) -> Optional[AdvancedQMModel]:
    """
    Fit the advanced QM model (PoP + body EQM + GPD tail) on a
    (p_sat, p_station) sample.

    Returns None if the sample is too small to fit.
    """
    p_sat = np.asarray(p_sat, dtype=float)
    p_station = np.asarray(p_station, dtype=float)
    if p_sat.size < 50 or p_station.size < 50:
        return None

    # A. PoP threshold
    p_th = fit_pop_threshold(p_sat, p_station) if use_pop else 0.0
    if not np.isfinite(p_th):
        return None
    sat_eff = p_sat.copy()
    if use_pop:
        sat_eff[sat_eff <= p_th] = 0.0

    sat_wet = sat_eff[sat_eff > 0.0]
    st_wet = p_station[p_station > 0.0]
    if sat_wet.size < 30 or st_wet.size < 30:
        return None

    # Body EQM
    q_levels = np.linspace(0.001, 0.999, num_quantiles)
    q_sat = np.quantile(sat_wet, q_levels)
    q_station = np.quantile(st_wet, q_levels)

    # B. GPD tails
    if use_gpd_tail:
        tail_sat = _fit_gpd(sat_wet, threshold_q=tail_threshold_q)
        tail_station = _fit_gpd(st_wet, threshold_q=tail_threshold_q)
    else:
        tail_sat = GPDTail(0, 0, 0, 1.0, 0, False)
        tail_station = GPDTail(0, 0, 0, 1.0, 0, False)

    # F. Volume factors
    sat_corr_body = _apply_qm_body(sat_eff, q_sat, q_station, p_th)
    mean_st = float(np.mean(p_station))
    mean_corr = float(np.mean(sat_corr_body))
    vf_mean = float(np.clip(mean_st / mean_corr, vf_clip[0], vf_clip[1])) \
        if mean_corr > 0 else 1.0

    # tail-anchored: use mean over the joint upper tail (>=q95) of both series
    upper_q = 0.95
    if st_wet.size >= 50 and sat_wet.size >= 50:
        st_tail = st_wet[st_wet >= np.quantile(st_wet, upper_q)]
        sat_corr_wet = sat_corr_body[sat_corr_body > 0]
        if sat_corr_wet.size > 0:
            sat_tail_corr = sat_corr_wet[sat_corr_wet >= np.quantile(sat_corr_wet, upper_q)]
            if sat_tail_corr.size > 0 and st_tail.size > 0:
                vf_tail = float(np.clip(st_tail.mean() / sat_tail_corr.mean(),
                                        vf_clip[0], vf_clip[1]))
            else:
                vf_tail = vf_mean
        else:
            vf_tail = vf_mean
    else:
        vf_tail = vf_mean

    return AdvancedQMModel(
        p_th=p_th, q_levels=q_levels, q_sat=q_sat, q_station=q_station,
        tail_sat=tail_sat, tail_station=tail_station,
        vf_mean=vf_mean, vf_tail=vf_tail,
        vf_blend_w=float(np.clip(vf_tail_blend_w, 0.0, 1.0)),
    )


# ------------------------------------------------------------------ #
# Apply: scalar core
# ------------------------------------------------------------------ #
def _apply_qm_body(values: np.ndarray, q_sat: np.ndarray, q_station: np.ndarray,
                   p_th: float) -> np.ndarray:
    """Body-only QM application (no GPD tail, no VF) — used internally."""
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr)
    pos = arr > p_th
    if not pos.any():
        return out
    a = arr[pos]
    below = a < q_sat[0]
    above = a > q_sat[-1]
    inside = ~below & ~above
    out_buf = np.zeros_like(a)
    if below.any():
        dx = q_sat[0] - p_th
        if dx > 0:
            out_buf[below] = q_station[0] * ((a[below] - p_th) / dx)
        else:
            out_buf[below] = q_station[0]
    if inside.any():
        idx = np.searchsorted(q_sat, a[inside])
        idx = np.clip(idx, 1, len(q_sat) - 1)
        x0 = q_sat[idx - 1]; x1 = q_sat[idx]
        y0 = q_station[idx - 1]; y1 = q_station[idx]
        denom = np.where(x1 > x0, x1 - x0, 1.0)
        out_buf[inside] = y0 + (y1 - y0) * (a[inside] - x0) / denom
    if above.any():
        # Default linear extrapolation by last-5 slope (used here only as
        # fallback when no GPD; the full pipeline overrides this with GPD).
        if (q_sat[-1] - q_sat[-5]) > 0:
            slope = (q_station[-1] - q_station[-5]) / (q_sat[-1] - q_sat[-5])
        else:
            slope = 1.0
        out_buf[above] = q_station[-1] + slope * (a[above] - q_sat[-1])

    out[pos] = np.maximum(out_buf, 0.0)
    return out


def apply_advanced_qm(
    values: np.ndarray,
    model: AdvancedQMModel,
    alpha_low: float | None = None,
    alpha_high: float | None = None,
    alpha_break_q: float = 0.90,
) -> np.ndarray:
    """
    Apply the full advanced QM (PoP -> body EQM -> GPD-extrapolated tail
    -> blended Volume Factor).

    Optional quantile-dependent blending: when ``alpha_low`` and
    ``alpha_high`` are both given, the QM correction strength varies by
    rank quantile:
        alpha(q) = alpha_low                       if q < alpha_break_q
        alpha(q) = alpha_high                      if q >= alpha_break_q
    Reasoning: under non-stationarity between training and target periods,
    the IMERG body distribution can already match station, while the
    upper tail still needs full correction (where R-factor signal lives).
    Setting alpha_low=0.2 (legacy-equivalent) and alpha_high=1.0 with
    alpha_break_q=0.90 resolves the volume/tail trade-off.

    Pass alpha_low=alpha_high (or both = None) to disable quantile
    blending; the legacy ``blend_alpha`` argument of
    ``calibrate_station_advanced`` still applies a uniform blend.
    """
    if model is None:
        return np.zeros_like(np.asarray(values, dtype=float))

    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr)

    # PoP mask
    pos = arr > model.p_th
    if not pos.any():
        return out
    a = arr[pos]

    # Split into body and upper-tail (above last quantile or above tail threshold u)
    use_gpd = (model.tail_sat.valid and model.tail_station.valid)
    body_mask_in_a = ~(use_gpd & (a >= model.tail_sat.threshold_u))
    out_buf = np.zeros_like(a)

    # Body
    if body_mask_in_a.any():
        body_a = a[body_mask_in_a]
        out_buf[body_mask_in_a] = _apply_qm_body(
            body_a, model.q_sat, model.q_station, model.p_th
        )

    # Tail via GPD
    if use_gpd and (~body_mask_in_a).any():
        tail_a = a[~body_mask_in_a]
        F_vals = np.array([_gpd_cdf(v, model.tail_sat) for v in tail_a])
        out_buf[~body_mask_in_a] = np.array([
            _gpd_quantile(F, model.tail_station) for F in F_vals
        ])

    # Volume factor (blended)
    out_buf *= model.effective_vf()

    # Quantile-dependent blend with raw values (optional).
    # We define rank as quantile of the input value within the IMERG
    # training distribution (q_sat), so the alpha mapping is consistent
    # across application calls.
    if alpha_low is not None and alpha_high is not None and alpha_low != alpha_high:
        # Rank lookup against q_sat
        ranks = np.searchsorted(model.q_sat, a, side="right") / float(model.q_sat.size)
        alphas = np.where(ranks < alpha_break_q, float(alpha_low), float(alpha_high))
        blended = a + alphas * (out_buf - a)
        out[pos] = np.maximum(blended, 0.0)
    else:
        out[pos] = np.maximum(out_buf, 0.0)
    return out


# ------------------------------------------------------------------ #
# Convenience: full station calibration with seasonal partition
# ------------------------------------------------------------------ #
def calibrate_station_advanced(
    paired_df,
    train_start: str = "2001-01-01",
    train_end: str = "2015-12-31",
    seasons=("DJF", "MAM", "JJA", "SON"),
    use_pop: bool = True,
    use_gpd_tail: bool = True,
    vf_tail_blend_w: float = 0.0,
    blend_alpha: float = 1.0,
    alpha_low: float | None = 0.2,
    alpha_high: float | None = 0.6,
    alpha_break_q: float = 0.95,
):
    """
    Drop-in replacement for ``qm_calibration.calibrate_station`` that
    uses the advanced fit (PoP + GPD tail + blended VF) per season.
    Returns a copy of paired_df with a 'P_corrected_mm' column.

    Parameter ``blend_alpha`` controls the strength of the QM correction
    (P_out = P_raw + alpha * (P_qm - P_raw)).  blend_alpha = 1.0 applies
    the full QM (recommended for IMERG V06 / older satellite products
    with persistent volume bias); blend_alpha = 0.2-0.4 is appropriate
    for IMERG V07, which has substantially smaller raw bias and where
    full QM can over-correct the mean under non-stationarity between
    the training and target periods.

    Note: this routine is meant for cross-validation and methodological
    comparison, not for replacing the operational v5_year_anchor
    pipeline (which has additional daily / annual guards and is already
    tuned to volume balance).  The advantage of the advanced fit
    surfaces in the upper tail (P95-P99) and in event count.
    """
    import pandas as pd

    df = paired_df.copy()
    if "season" not in df.columns:
        from .qm_calibration import get_season
        df["season"] = df["datetime"].dt.month.apply(get_season)

    train_mask = (df["datetime"] >= train_start) & (df["datetime"] <= train_end)
    df["P_corrected_mm"] = df["P_sat_mm"].astype(float)

    for season in seasons:
        sd = df[train_mask & (df["season"] == season)]
        if len(sd) < 80:
            continue
        model = fit_advanced_qm(
            sd["P_sat_mm"].values, sd["P_station_mm"].values,
            use_pop=use_pop, use_gpd_tail=use_gpd_tail,
            vf_tail_blend_w=vf_tail_blend_w,
        )
        if model is None:
            continue
        season_mask = df["season"] == season
        raw = df.loc[season_mask, "P_sat_mm"].values
        qm = apply_advanced_qm(
            raw, model,
            alpha_low=alpha_low, alpha_high=alpha_high,
            alpha_break_q=alpha_break_q,
        )
        # Uniform soft blend only when quantile-dependent blending is OFF
        if blend_alpha < 1.0 and (alpha_low is None or alpha_high is None):
            blended = raw + float(blend_alpha) * (qm - raw)
            df.loc[season_mask, "P_corrected_mm"] = np.maximum(blended, 0.0)
        else:
            df.loc[season_mask, "P_corrected_mm"] = qm

    return df
