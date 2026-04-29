# -*- coding: utf-8 -*-
"""
09_uncertainty.py — Full uncertainty budget for the domain-mean R-factor.

Five components, all combined in quadrature for the total 1-sigma:
  1. Sampling / climate uncertainty   — Monte-Carlo bootstrap on the
                                         24 annual rasters (seeded).
  2. Calibration uncertainty          — analytical PBIAS propagation from
                                         the 201 station-level cross-
                                         validation summary (alpha=1.5).
  3. Energy-formula uncertainty       — spread across the e(i) ensemble
                                         (BF k=0.05/0.082, van Dijk,
                                         Wischmeier, Sanchez-Moreno),
                                         evaluated at the storm-relevant
                                         intensity i = 25.4 mm/h.
  4. Sub-bin / sub-pixel I_30 bias    — measured empirically from the
                                         Kazan AWS pluviograph
                                         (CF mean = 1.05, std ~ 0.05).
  5. Phase-mask uncertainty           — 5% multiplicative term to allow
                                         for ERA5-Land mixed-phase
                                         misclassification.

Outputs
-------
docs/figures/fig15_uncertainty_bootstrap.png
docs/figures/fig16_uncertainty_budget.png
output/tables/uncertainty_summary.csv
"""
from __future__ import annotations
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from erosivity.lib.energy_models import unit_energy_np

REPO = ROOT

YEARS = list(range(2001, 2025))
DPI = 220
SEED = 42
BLUE, RED, GREEN, ORANGE, PURPLE = "#2166ac", "#d6604d", "#1a9641", "#fdae61", "#762a83"


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate the uncertainty budget for the domain-mean R-factor.")
    parser.add_argument("--annual-dir", default=os.path.join(ROOT, "output", "v6_rfactor", "annual"))
    parser.add_argument("--pbias-csv", default=os.path.join(ROOT, "data", "external", "station_annual_pbias_summary_imerg.csv"))
    parser.add_argument("--compare-csv", default=os.path.join(ROOT, "output", "tables", "compare_k05_k082.csv"))
    parser.add_argument("--sensitivity-csv", default=os.path.join(ROOT, "output", "tables", "sensitivity_pack.csv"))
    parser.add_argument("--fig-dir", default=os.path.join(ROOT, "docs", "figures"))
    parser.add_argument("--table-dir", default=os.path.join(ROOT, "output", "tables"))
    return parser.parse_args()


def load_stack(annual_dir: str):
    stack = []
    for y in YEARS:
        p = os.path.join(annual_dir, f"R_imerg_{y}.tif")
        with rasterio.open(p) as ds:
            band = ds.read(1).astype(np.float64)
        band[band == 0] = np.nan
        stack.append(band)
    return np.array(stack)


def domain_means(stack):
    return np.array([np.nanmean(stack[i]) for i in range(len(stack))])


# ------------------------------------------------------------------ #
# Components
# ------------------------------------------------------------------ #
def bootstrap_sampling(means, n_boot=50_000, ci=0.90):
    rng = np.random.default_rng(SEED)
    n = len(means)
    boot = np.array([rng.choice(means, n, replace=True).mean()
                     for _ in range(n_boot)])
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return boot, lo, hi


def bootstrap_spatial(stack, n_boot=5_000):
    rng = np.random.default_rng(SEED)
    n = stack.shape[0]
    sums = np.zeros(stack.shape[1:])
    sq = np.zeros(stack.shape[1:])
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = np.nanmean(stack[idx], axis=0)
        sums += s
        sq += s ** 2
    mean_b = sums / n_boot
    std_b = np.sqrt(np.maximum(sq / n_boot - mean_b ** 2, 0.0))
    return std_b


def calibration_uncertainty(mean_R: float, pbias_csv: str):
    df = pd.read_csv(pbias_csv)
    pb = df["annual_pbias_corr_abs_med"].values
    pb = pb[pb < 50]  # drop edge stations
    sigma_pbias_med = float(np.median(pb))
    sigma_pbias_p84 = float(np.percentile(pb, 84))

    alpha = 1.5    # E ~ p^alpha (sub-quadratic; conservative)
    s_rel_med = alpha * sigma_pbias_med / 100.0
    s_rel_p84 = alpha * sigma_pbias_p84 / 100.0

    return {
        "pbias_med": sigma_pbias_med,
        "pbias_p84": sigma_pbias_p84,
        "sigma_rel_med": s_rel_med,
        "sigma_rel_p84": s_rel_p84,
        "sigma_calib_med": s_rel_med * mean_R,
    }


def parametric_k_uncertainty(mean_R: float, compare_csv: str):
    df = pd.read_csv(compare_csv)
    df = df[df["year"] != "MEAN"]
    ratio = df["ratio"].astype(float)
    return {
        "ratio_mean": float(ratio.mean()),
        "ratio_std": float(ratio.std()),
        "sigma_param": float(ratio.std() / ratio.mean() * mean_R),
        "sigma_rel": float(ratio.std() / ratio.mean()),
    }


def energy_ensemble_uncertainty(mean_R: float):
    """Spread of e(i) across the BF-k=0.082 (baseline used here for R) and
    alternative parameterisations, evaluated at the storm-relevant
    intensity i = 25.4 mm/h.  The ensemble variance is propagated as a
    relative uncertainty since e enters multiplicatively into R for any
    fixed precipitation series."""
    i_ref = 25.4
    e_values = {
        "bf_k082": float(unit_energy_np(i_ref, model="bf", exp_k=0.082)),
        "bf_k05":  float(unit_energy_np(i_ref, model="bf", exp_k=0.05)),
        "vd":      float(unit_energy_np(i_ref, model="vd")),
        "ws":      float(unit_energy_np(i_ref, model="ws")),
        "sm":      float(unit_energy_np(i_ref, model="sm")),
    }
    e_ref = e_values["bf_k082"]   # nominal model used for R
    rel_devs = np.array([(v - e_ref) / e_ref for v in e_values.values()])
    sigma_rel = float(np.std(rel_devs, ddof=1))
    return {
        "e_values": e_values,
        "sigma_rel": sigma_rel,
        "sigma_abs": sigma_rel * mean_R,
    }


def subbin_cf_uncertainty(mean_R: float, sensitivity_csv: str, cf_mean=1.05, cf_std=0.05):
    """Sub-bin I_30 correction factor from AWS Kazan diagnostic.
    R scales linearly with I_30 (per event EI30 = E*I30), so propagating
    the CF distribution into R gives mean shift = (CF-1)*R and
    sigma = cf_std * R.

    Read measured values from sensitivity_pack.csv if present; otherwise
    use the defaults (mean=1.05, std=0.05) from the documented diagnostic.
    """
    if os.path.exists(sensitivity_csv):
        try:
            df = pd.read_csv(sensitivity_csv)
            sub = df[df["category"] == "subbin_CF"]
            cf_mean = float(sub.loc[sub["metric"] == "CF_mean", "value"].iloc[0])
            cf_std_obs = (
                (float(sub.loc[sub["metric"] == "CF_P75", "value"].iloc[0])
                 - float(sub.loc[sub["metric"] == "CF_P25", "value"].iloc[0])) / 1.349
            )
            if np.isfinite(cf_std_obs) and cf_std_obs > 0:
                cf_std = cf_std_obs
        except Exception:
            pass
    return {
        "cf_mean": cf_mean,
        "cf_std": cf_std,
        "sigma_rel": float(cf_std / max(cf_mean, 1e-6)),
        "sigma_abs": float(cf_std / max(cf_mean, 1e-6) * mean_R),
        "bias_correction_pct": 100.0 * (cf_mean - 1.0),
    }


def phase_mask_uncertainty(mean_R: float, sigma_rel=0.05):
    """ERA5-Land air-temperature-based phase mask vs wet-bulb
    discrimination differs by ~5–10% in mid-latitude transition seasons.
    A 5% relative uncertainty is a conservative lower bound; can be
    refined by recomputing R with both masks if data permit."""
    return {"sigma_rel": float(sigma_rel),
            "sigma_abs": float(sigma_rel * mean_R)}


# ------------------------------------------------------------------ #
# Plots
# ------------------------------------------------------------------ #
def fig15_bootstrap(means, boot, lo, hi, ci, std_spatial, annual_dir: str, fig_dir: str):
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)
    mean_R = float(np.mean(means))

    ax1 = fig.add_subplot(gs[0])
    ax1.hist(boot, bins=80, color=BLUE, alpha=0.7, edgecolor="none", density=True)
    ax1.axvline(mean_R, color="black", lw=2, label=f"Observed mean = {mean_R:.0f}")
    ax1.axvline(lo, color=RED, lw=1.5, ls="--", label=f"{int(ci*100)}% CI: [{lo:.0f}, {hi:.0f}]")
    ax1.axvline(hi, color=RED, lw=1.5, ls="--")
    ax1.set_xlabel("R-factor, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    ax1.set_ylabel("Density")
    ax1.set_title(f"(a) Bootstrap distribution of mean R\nseed={SEED}, n_boot=50 000",
                  fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(YEARS, means, color=plt.cm.YlOrRd(np.interp(means,
            [means.min(), means.max()], [0.25, 0.92])),
            edgecolor="gray", lw=0.4, alpha=0.85)
    ax2.axhline(mean_R, color="black", lw=1.5, ls="--", label=f"Mean = {mean_R:.0f}")
    ax2.fill_between(YEARS, lo, hi, alpha=0.15, color=BLUE,
                     label=f"{int(ci*100)}% bootstrap CI")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("R-factor, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    ax2.set_title(f"(b) Annual means + CI band\nhalf-width = {(hi-lo)/2:.0f} ({(hi-lo)/2/mean_R*100:.1f}%)",
                  fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(gs[2])
    p = os.path.join(annual_dir, "R_imerg_2001.tif")
    with rasterio.open(p) as ds:
        t = ds.transform
        ext = [t.c, t.c + t.a * ds.width, t.f + t.e * ds.height, t.f]
    im = ax3.imshow(std_spatial, extent=ext, cmap="Oranges",
                    vmin=0, vmax=np.nanpercentile(std_spatial, 98),
                    interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax3, shrink=0.85, pad=0.02,
                 label="Bootstrap σ, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    ax3.set_title("(c) Spatial sampling σ (bootstrap)", fontweight="bold")
    ax3.set_xlabel("Longitude, °E"); ax3.set_ylabel("Latitude, °N")
    ax3.grid(alpha=0.2, lw=0.5)

    out = os.path.join(fig_dir, "fig15_uncertainty_bootstrap.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved:", out)


def fig16_budget(components: dict, mean_R: float, fig_dir: str):
    labels = list(components.keys())
    vals_abs = [components[k]["sigma_abs"] for k in labels]
    combined = float(np.sqrt(sum(v ** 2 for v in vals_abs)))
    colors = [BLUE, RED, GREEN, ORANGE, PURPLE][: len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    bars = axes[0].barh(labels, vals_abs, color=colors, alpha=0.85, edgecolor="gray", lw=0.5)
    axes[0].axvline(combined, color="black", lw=2, ls="--",
                    label=f"RSS total = {combined:.0f} MJ·mm·ha⁻¹·h⁻¹·yr⁻¹  ({combined/mean_R*100:.1f}%)")
    for bar, v in zip(bars, vals_abs):
        axes[0].text(v + 1, bar.get_y() + bar.get_height() / 2,
                     f"{v:.1f}", va="center", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("σ, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    axes[0].set_title(f"(a) Uncertainty components (1σ); mean R = {mean_R:.0f}",
                      fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="x", alpha=0.3)

    pie_vals = [v ** 2 for v in vals_abs]
    wedges, texts, auto = axes[1].pie(
        pie_vals, labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        colors=colors, startangle=90, pctdistance=0.65,
        wedgeprops=dict(edgecolor="white", lw=1.5),
    )
    for t in auto:
        t.set_fontsize(11); t.set_fontweight("bold")
    axes[1].set_title("(b) Variance contributions\n(quadrature sum)", fontweight="bold")

    out = os.path.join(fig_dir, "fig16_uncertainty_budget.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved:", out)


# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    print("Loading stack...")
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)
    stack = load_stack(args.annual_dir)
    means = domain_means(stack)
    mean_R = float(np.mean(means))
    print(f"Domain mean R = {mean_R:.1f}")

    print("\n[1/5] Bootstrap sampling (50k draws, seed=42)...")
    boot, lo, hi = bootstrap_sampling(means, n_boot=50_000, ci=0.90)
    sigma_samp = (hi - lo) / 2.0
    print(f"  90% CI [{lo:.1f}, {hi:.1f}], half-width {sigma_samp:.1f} ({sigma_samp/mean_R*100:.1f}%)")

    print("\n[2/5] Spatial bootstrap sigma (5k draws)...")
    std_spatial = bootstrap_spatial(stack, n_boot=5_000)
    print(f"  mean spatial sigma = {np.nanmean(std_spatial):.1f}")

    print("\n[3/5] Calibration uncertainty from PBIAS...")
    u_calib = calibration_uncertainty(mean_R, args.pbias_csv)
    print(f"  PBIAS median = {u_calib['pbias_med']:.1f}%, sigma_rel_med = {u_calib['sigma_rel_med']*100:.1f}%")
    print(f"  sigma_calib_med = {u_calib['sigma_calib_med']:.1f}")

    print("\n[4a/5] Parametric uncertainty (BF k=0.05 vs 0.082)...")
    u_param = parametric_k_uncertainty(mean_R, args.compare_csv)
    print(f"  ratio std = {u_param['ratio_std']:.4f}, sigma_param = {u_param['sigma_param']:.1f}")

    print("\n[4b/5] Energy-ensemble uncertainty (BF/vD/WS/SM)...")
    u_ens = energy_ensemble_uncertainty(mean_R)
    for k, v in u_ens["e_values"].items():
        print(f"    e_{k}(25.4 mm/h) = {v:.4f}")
    print(f"  sigma_ensemble_rel = {u_ens['sigma_rel']*100:.1f}%, sigma_ensemble = {u_ens['sigma_abs']:.1f}")

    print("\n[5a/5] Sub-bin / sub-pixel I_30 correction factor...")
    u_cf = subbin_cf_uncertainty(mean_R, args.sensitivity_csv)
    print(f"  CF mean = {u_cf['cf_mean']:.3f} ± {u_cf['cf_std']:.3f}")
    print(f"  bias correction = {u_cf['bias_correction_pct']:+.1f}%, sigma_CF = {u_cf['sigma_abs']:.1f}")

    print("\n[5b/5] Phase-mask uncertainty (assumed 5%)...")
    u_phase = phase_mask_uncertainty(mean_R)
    print(f"  sigma_phase = {u_phase['sigma_abs']:.1f}")

    components = {
        "Climatic (bootstrap σ)":           {"sigma_abs": sigma_samp},
        "Calibration (PBIAS, α=1.5)":       {"sigma_abs": u_calib["sigma_calib_med"]},
        "e(i) ensemble (5 models)":         {"sigma_abs": u_ens["sigma_abs"]},
        "Sub-bin I_30 (AWS Kazan CF)":      {"sigma_abs": u_cf["sigma_abs"]},
        "Phase mask (T2m vs T_wet)":        {"sigma_abs": u_phase["sigma_abs"]},
    }

    print("\n[plots] fig15...")
    fig15_bootstrap(means, boot, lo, hi, 0.90, std_spatial, args.annual_dir, args.fig_dir)

    print("[plots] fig16...")
    fig16_budget(components, mean_R, args.fig_dir)

    # ----- Save table -----
    rss = float(np.sqrt(sum(c["sigma_abs"] ** 2 for c in components.values())))
    rows = [
        ["Mean R (baseline, BF k=0.082)", f"{mean_R:.1f}", "MJ·mm·ha⁻¹·h⁻¹·yr⁻¹"],
        ["Climatic 90% CI (bootstrap)", f"[{lo:.1f}, {hi:.1f}]", "—"],
        ["σ climatic (CI/2)", f"{sigma_samp:.1f} ({sigma_samp/mean_R*100:.1f}%)", "—"],
        ["σ calibration (PBIAS×α=1.5)", f"{u_calib['sigma_calib_med']:.1f} "
            f"({u_calib['sigma_rel_med']*100:.1f}%)", "—"],
        ["σ k (BF 0.05 vs 0.082)", f"{u_param['sigma_param']:.1f} "
            f"({u_param['sigma_rel']*100:.2f}%)", "—"],
        ["σ e(i)-ensemble (5 models)", f"{u_ens['sigma_abs']:.1f} "
            f"({u_ens['sigma_rel']*100:.1f}%)", "BF/vD/WS/SM at i=25.4"],
        ["σ sub-bin I_30 CF", f"{u_cf['sigma_abs']:.1f} "
            f"({u_cf['sigma_rel']*100:.1f}%)", f"CF={u_cf['cf_mean']:.3f}±{u_cf['cf_std']:.3f}"],
        ["σ phase mask", f"{u_phase['sigma_abs']:.1f} "
            f"({u_phase['sigma_rel']*100:.1f}%)", "T2m vs wet-bulb assumed 5%"],
        ["Total 1σ (RSS)", f"{rss:.1f} ({rss/mean_R*100:.1f}%)", "quadrature sum"],
    ]
    out_csv = os.path.join(args.table_dir, "uncertainty_summary.csv")
    pd.DataFrame(rows, columns=["Component", "Value", "Note"]).to_csv(
        out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {out_csv}")

    print("\n=== UNCERTAINTY SUMMARY (1-sigma) ===")
    print(f"  R           = {mean_R:.1f}")
    print(f"  sigma_climate   = {sigma_samp:.1f}  ({sigma_samp/mean_R*100:.1f}%)")
    print(f"  sigma_calib     = {u_calib['sigma_calib_med']:.1f}  ({u_calib['sigma_rel_med']*100:.1f}%)")
    print(f"  sigma_e(i)      = {u_ens['sigma_abs']:.1f}  ({u_ens['sigma_rel']*100:.1f}%)")
    print(f"  sigma_subbin    = {u_cf['sigma_abs']:.1f}  ({u_cf['sigma_rel']*100:.1f}%)")
    print(f"  sigma_phase     = {u_phase['sigma_abs']:.1f}  ({u_phase['sigma_rel']*100:.1f}%)")
    print(f"  sigma_total RSS = {rss:.1f}  ({rss/mean_R*100:.1f}%)")


if __name__ == "__main__":
    main()
