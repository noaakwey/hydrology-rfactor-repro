from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

YEARS = list(range(2001, 2025))
DPI = 220
NODATA = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v6 paper figures and summary tables.")
    parser.add_argument("--v6-annual-dir", default=str(ROOT / "output" / "v6_rfactor" / "annual"))
    parser.add_argument("--v5-annual-dir", default=str(ROOT / "output" / "v6_rfactor_area" / "annual"))
    parser.add_argument("--diagnostics-dir", default=str(ROOT / "output" / "v6_diagnostics"))
    parser.add_argument("--fig-dir", default=str(ROOT / "docs" / "figures"))
    return parser.parse_args()


def load_annual_stack(annual_dir: str) -> tuple[np.ndarray, dict]:
    stack = []
    profile = None
    for year in YEARS:
        path = Path(annual_dir) / f"R_imerg_{year}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        with rasterio.open(path) as ds:
            band = ds.read(1).astype(np.float64)
            if profile is None:
                profile = ds.profile.copy()
        band[band == NODATA] = np.nan
        stack.append(band)
    return np.stack(stack, axis=0), profile


def get_extent(profile: dict) -> list[float]:
    tr = profile["transform"]
    return [tr.c, tr.c + tr.a * profile["width"], tr.f + tr.e * profile["height"], tr.f]


def domain_stat(stack: np.ndarray, func) -> np.ndarray:
    out = []
    for i in range(stack.shape[0]):
        vals = stack[i][np.isfinite(stack[i])]
        out.append(float(func(vals)) if vals.size else np.nan)
    return np.asarray(out, dtype=np.float64)


def save_figure(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_domain_stats(v6_stack: np.ndarray, v5_stack: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = np.asarray(YEARS, dtype=np.int32)
    v6_mean = domain_stat(v6_stack, np.nanmean)
    v6_median = domain_stat(v6_stack, np.nanmedian)
    v6_p25 = domain_stat(v6_stack, lambda x: np.percentile(x, 25))
    v6_p75 = domain_stat(v6_stack, lambda x: np.percentile(x, 75))
    v5_mean = domain_stat(v5_stack, np.nanmean)

    year_df = pd.DataFrame(
        {
            "year": years,
            "v6_domain_mean": v6_mean,
            "v6_domain_median": v6_median,
            "v6_domain_p25": v6_p25,
            "v6_domain_p75": v6_p75,
            "v5_domain_mean": v5_mean,
            "delta_mean_pct": 100.0 * (v6_mean - v5_mean) / np.maximum(v5_mean, 1e-6),
        }
    )

    v6_mean_map = np.nanmean(v6_stack, axis=0)
    v5_mean_map = np.nanmean(v5_stack, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pixel_cv = np.where(v6_mean_map > 0.0, np.nanstd(v6_stack, axis=0) / v6_mean_map * 100.0, np.nan)
        delta_pct = 100.0 * (v6_mean_map - v5_mean_map) / np.maximum(v5_mean_map, 1e-6)

    slope, intercept, r_value, p_value, _ = sp_stats.linregress(years.astype(np.float64), v6_mean)

    summary_df = pd.DataFrame(
        [
            {"metric": "v6_domain_mean_r", "value": float(np.nanmean(v6_mean_map))},
            {"metric": "v6_domain_median_r", "value": float(np.nanmedian(v6_mean_map))},
            {"metric": "v6_domain_p05_r", "value": float(np.nanpercentile(v6_mean_map, 5))},
            {"metric": "v6_domain_p95_r", "value": float(np.nanpercentile(v6_mean_map, 95))},
            {"metric": "v6_domain_cv_pct", "value": float(np.nanstd(v6_mean) / np.nanmean(v6_mean) * 100.0)},
            {"metric": "v6_pixel_cv_mean_pct", "value": float(np.nanmean(pixel_cv))},
            {"metric": "v6_trend_slope_per_year", "value": float(slope)},
            {"metric": "v6_trend_p_value", "value": float(p_value)},
            {"metric": "v6_trend_r2", "value": float(r_value**2)},
            {"metric": "v5_domain_mean_r", "value": float(np.nanmean(v5_mean_map))},
            {"metric": "v6_minus_v5_mean_pct", "value": float(100.0 * (np.nanmean(v6_mean_map) - np.nanmean(v5_mean_map)) / np.nanmean(v5_mean_map))},
            {"metric": "v6_minus_v5_pixel_median_pct", "value": float(np.nanmedian(delta_pct))},
            {"metric": "v6_minus_v5_pixel_p95_pct", "value": float(np.nanpercentile(delta_pct, 95))},
        ]
    )
    return year_df, summary_df


def fig_mean_pdf(v6_stack: np.ndarray, profile: dict, fig_dir: Path) -> None:
    mean_r = np.nanmean(v6_stack, axis=0)
    vals = mean_r[np.isfinite(mean_r)]
    ext = get_extent(profile)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, gridspec_kw={"width_ratios": [2.1, 1, 1]})
    ax = axes[0]
    im = ax.imshow(mean_r, extent=ext, cmap="YlOrRd", vmin=np.nanpercentile(mean_r, 2), vmax=np.nanpercentile(mean_r, 98), interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="MJ mm ha-1 h-1 yr-1")
    ax.set_title("Mean annual R-factor, 2001-2024")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.15)

    ax = axes[1]
    ax.hist(vals, bins=32, color="#d95f02", alpha=0.8, edgecolor="none", density=True)
    ax.axvline(np.mean(vals), color="black", linestyle="--", linewidth=1.1, label=f"mean={np.mean(vals):.1f}")
    ax.axvline(np.median(vals), color="#1b9e77", linestyle=":", linewidth=1.4, label=f"median={np.median(vals):.1f}")
    ax.set_title("Distribution of mean R")
    ax.set_xlabel("MJ mm ha-1 h-1 yr-1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100.0
    ax.plot(sorted_vals, cdf, color="#2c7fb8", linewidth=2.0)
    ax.set_title("Empirical CDF")
    ax.set_xlabel("MJ mm ha-1 h-1 yr-1")
    ax.set_ylabel("% of pixels")
    ax.grid(alpha=0.25)

    save_figure(fig, fig_dir, "fig21_v6_mean_pdf_cdf.png")


def fig_temporal(year_df: pd.DataFrame, fig_dir: Path) -> None:
    years = year_df["year"].to_numpy(dtype=np.int32)
    mean_v6 = year_df["v6_domain_mean"].to_numpy(dtype=np.float64)
    med_v6 = year_df["v6_domain_median"].to_numpy(dtype=np.float64)
    p25 = year_df["v6_domain_p25"].to_numpy(dtype=np.float64)
    p75 = year_df["v6_domain_p75"].to_numpy(dtype=np.float64)
    mean_v5 = year_df["v5_domain_mean"].to_numpy(dtype=np.float64)

    slope, intercept, r_value, p_value, _ = sp_stats.linregress(years.astype(np.float64), mean_v6)
    mu = float(np.mean(mean_v6))
    anomaly = mean_v6 - mu

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 1.1, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(years, mean_v6, color="#d95f02", alpha=0.8, edgecolor="white", linewidth=0.3, label="v6")
    ax1.plot(years, mean_v5, color="#2c7fb8", linewidth=1.8, marker="o", markersize=3, label="v5 baseline")
    ax1.plot(years, med_v6, color="#1b9e77", linewidth=1.2, linestyle="--", label="v6 median pixel")
    ax1.fill_between(years, p25, p75, color="#1b9e77", alpha=0.12, label="v6 IQR")
    ax1.plot(years, slope * years + intercept, color="black", linestyle="--", linewidth=1.2, label=f"OLS slope={slope:+.2f}, p={p_value:.3f}")
    ax1.set_title("Interannual dynamics of domain-mean R")
    ax1.set_ylabel("MJ mm ha-1 h-1 yr-1")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, ncol=2, fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    colors = ["#d95f02" if a >= 0 else "#2c7fb8" for a in anomaly]
    ax2.bar(years, anomaly, color=colors, alpha=0.85)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Anomaly")
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(years, year_df["delta_mean_pct"], color="#756bb1", marker="o", linewidth=1.8)
    ax3.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_ylabel("v6-v5, %")
    ax3.set_xlabel("Year")
    ax3.set_title("Relative shift vs v5 baseline")
    ax3.grid(alpha=0.25)

    save_figure(fig, fig_dir, "fig22_v6_temporal_dynamics.png")


def fig_cv_trend(v6_stack: np.ndarray, profile: dict, fig_dir: Path) -> None:
    ext = get_extent(profile)
    mean_r = np.nanmean(v6_stack, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(mean_r > 0.0, np.nanstd(v6_stack, axis=0) / mean_r * 100.0, np.nan)

    years_arr = np.asarray(YEARS, dtype=np.float64)
    slope_map = np.full(mean_r.shape, np.nan, dtype=np.float64)
    pval_map = np.full(mean_r.shape, np.nan, dtype=np.float64)
    for i in range(mean_r.shape[0]):
        for j in range(mean_r.shape[1]):
            ts = v6_stack[:, i, j]
            mask = np.isfinite(ts)
            if np.sum(mask) < 6:
                continue
            slope, _, _, pval, _ = sp_stats.linregress(years_arr[mask], ts[mask])
            slope_map[i, j] = slope
            pval_map[i, j] = pval
    sig = np.where(pval_map < 0.05, slope_map, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    im0 = axes[0].imshow(cv, extent=ext, cmap="viridis", interpolation="nearest", aspect="auto")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02, label="%")
    axes[0].set_title("Pixel-scale coefficient of variation")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    vmax = np.nanpercentile(np.abs(sig), 95) if np.any(np.isfinite(sig)) else 1.0
    im1 = axes[1].imshow(sig, extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="auto")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.02, label="MJ mm ha-1 h-1 yr-2")
    axes[1].set_title("Significant linear trends (p < 0.05)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    save_figure(fig, fig_dir, "fig23_v6_cv_trend.png")


def fig_v5_v6_comparison(v6_stack: np.ndarray, v5_stack: np.ndarray, profile: dict, fig_dir: Path) -> None:
    v6_mean = np.nanmean(v6_stack, axis=0)
    v5_mean = np.nanmean(v5_stack, axis=0)
    delta = v6_mean - v5_mean
    with np.errstate(invalid="ignore", divide="ignore"):
        delta_pct = 100.0 * delta / np.maximum(v5_mean, 1e-6)
    ext = get_extent(profile)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, field, title, cmap in [
        (axes[0, 0], v5_mean, "Baseline v5 mean R", "YlOrRd"),
        (axes[0, 1], v6_mean, "Hybrid v6 mean R", "YlOrRd"),
    ]:
        im = ax.imshow(field, extent=ext, cmap=cmap, vmin=np.nanpercentile(v6_mean, 2), vmax=np.nanpercentile(v6_mean, 98), interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    vmax = np.nanpercentile(np.abs(delta), 95)
    im = axes[1, 0].imshow(delta, extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.82, pad=0.02)
    axes[1, 0].set_title("Absolute mean shift, v6 - v5")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")

    vals = delta_pct[np.isfinite(delta_pct)]
    axes[1, 1].hist(vals, bins=30, color="#756bb1", alpha=0.85, edgecolor="none")
    axes[1, 1].axvline(np.median(vals), color="black", linestyle="--", linewidth=1.1, label=f"median={np.median(vals):.1f}%")
    axes[1, 1].set_title("Distribution of relative change")
    axes[1, 1].set_xlabel("%")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False, fontsize=8)

    save_figure(fig, fig_dir, "fig24_v5_v6_comparison.png")


def fig_annual_multiples(v6_stack: np.ndarray, profile: dict, fig_dir: Path) -> None:
    ext = get_extent(profile)
    means = domain_stat(v6_stack, np.nanmean)
    vmin = np.nanpercentile(v6_stack, 2)
    vmax = np.nanpercentile(v6_stack, 98)

    fig, axes = plt.subplots(4, 6, figsize=(18, 12), constrained_layout=True)
    for i, year in enumerate(YEARS):
        ax = axes[i // 6, i % 6]
        im = ax.imshow(v6_stack[i], extent=ext, cmap="YlOrRd", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_title(f"{year}\n{means[i]:.0f}", fontsize=8, color="darkred" if means[i] > np.mean(means) else "navy")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.42, pad=0.01, orientation="horizontal", aspect=40, label="MJ mm ha-1 h-1 yr-1")
    fig.suptitle("Annual v6 R-factor maps, 2001-2024", fontsize=14, fontweight="bold")
    save_figure(fig, fig_dir, "fig25_v6_annual_multiples.png")


def copy_supporting_figures(diagnostics_dir: Path, fig_dir: Path) -> None:
    mapping = {
        diagnostics_dir / "v6_station_verification.png": fig_dir / "fig26_v6_station_verification.png",
        diagnostics_dir / "v6_peak_model_diagnostics.png": fig_dir / "fig27_v6_peak_model_diagnostics.png",
        diagnostics_dir / "v6_highfreq_summary.png": fig_dir / "fig28_v6_highfreq_summary.png",
    }
    fig_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    diagnostics_dir = Path(args.diagnostics_dir)

    v6_stack, profile = load_annual_stack(args.v6_annual_dir)
    v5_stack, _ = load_annual_stack(args.v5_annual_dir)
    year_df, summary_df = build_domain_stats(v6_stack, v5_stack)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    year_df.to_csv(diagnostics_dir / "v6_domain_year_stats.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(diagnostics_dir / "v6_domain_summary.csv", index=False, encoding="utf-8-sig")

    fig_mean_pdf(v6_stack, profile, fig_dir)
    fig_temporal(year_df, fig_dir)
    fig_cv_trend(v6_stack, profile, fig_dir)
    fig_v5_v6_comparison(v6_stack, v5_stack, profile, fig_dir)
    fig_annual_multiples(v6_stack, profile, fig_dir)
    copy_supporting_figures(diagnostics_dir, fig_dir)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
