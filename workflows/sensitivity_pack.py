# -*- coding: utf-8 -*-
"""
sensitivity_pack.py — Sensitivity analysis of the event detector and a
direct empirical check of the sub-30-min I-bias.

Three diagnostics
-----------------
1. Sub-pixel / sub-bin I_30 correction factor (CF)
   Using the 1-min pluviograph at AWS Kazan (Pluvio2 bucket-cumulative
   trace).  For each storm, compute the "true" I_30 (rolling 30-min
   maximum from minute data) and the "bin-aligned" I_30 (resample to
   30-min sums, then maximum / 0.5).  Report
        CF = E[I_30_true / I_30_binned]
   This is the multiplicative correction to apply to IMERG-derived
   I_max and hence to R per event.

2. Activation rate of the 25.4 mm/h erosivity criterion
   Fraction of events whose erosivity is triggered by peak intensity
   alone (vs. by 12.7 mm depth) at IMERG dt=30min, on the same AWS
   record resampled to 30-min bins.

3. Sensitivity of annual R to the gap_intensity threshold
   Re-run the event detector on the 30-min Kazan series with
   gap_intensity ∈ {0.5, 1.0, 1.27, 2.0, 3.0} mm/h and compute the
   relative change in annual R.

Outputs:
  output/tables/sensitivity_pack.csv
  output/plots/sensitivity_pack.png
"""
from __future__ import annotations
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import OUTPUT_DIR
from erosivity.lib.event_detector import DetectorConfig, detect_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the event-detector sensitivity pack.")
    parser.add_argument("--aws-csv", default=os.path.join(ROOT, "data", "external", "aws310_pluvio_v2.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def load_kazan_minute_series(aws_csv: str) -> pd.Series:
    """Return 1-min precipitation rate (mm/min) from the Pluvio2 bucket
    cumulative reading.  Negative differences (bucket emptying) are
    zeroed; below-noise (<0.01 mm) is also zeroed."""
    df = pd.read_csv(aws_csv, parse_dates=["datetime_utc"])
    df = df.set_index("datetime_utc").sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    bucket = df["Pluvio2_1.value1"].astype(float).rolling("10min").median()
    rate = bucket.diff().fillna(0.0)
    rate[rate < 0.01] = 0.0
    rate[bucket.diff() < -2.0] = 0.0
    # Liquid-only filter: HMP155 air temp > 0 C ; April-Oct
    air = df.get("HMP155.T", pd.Series(20.0, index=df.index))
    valid_temp = air > 0.0
    valid_month = df.index.month.isin([4, 5, 6, 7, 8, 9, 10])
    rate.loc[~(valid_temp & valid_month)] = 0.0
    return rate.resample("1min").sum().fillna(0.0)


def sub_bin_cf_diagnostic(rate_mm_per_min: pd.Series) -> dict:
    """Compute the 'true' I_30 (rolling 30-min) vs the bin-aligned I_30
    on storm-segments of the AWS record.  Storm = >=15 min of >0
    minute-rate within a 6-h window with cumulative >= 5 mm.
    Returns aggregated CF and per-storm distribution.
    """
    # Convert minute-rate (mm/min) to mm at minute resolution:
    p_min = rate_mm_per_min.copy()
    # True I_30: rolling 30-min sum / 0.5 h => mm/h  (rolling on 30 minutes)
    rolling30 = p_min.rolling("30min").sum() / 0.5
    # Bin-aligned I_30: resample to 30-min sums => mm/h  (sum / 0.5 h)
    binned30_mm = p_min.resample("30min").sum()
    binned30 = binned30_mm / 0.5

    # Group into storms by 6-h dry gaps with cum >= 5 mm
    p30 = binned30_mm.fillna(0.0)
    storms = []
    in_storm = False
    cur = []
    cur_dry = 0
    for ts, v in p30.items():
        if v > 0:
            if not in_storm:
                in_storm = True
                cur = []
                cur_dry = 0
            cur.append(ts)
            cur_dry = 0
        else:
            if in_storm:
                cur_dry += 1
                if cur_dry >= 12:  # 12 * 30min = 6 h
                    if cur:
                        storms.append((cur[0], cur[-1]))
                    in_storm = False
                    cur = []
                    cur_dry = 0
    if in_storm and cur:
        storms.append((cur[0], cur[-1]))

    cfs = []
    for t0, t1 in storms:
        seg_min = p_min.loc[t0: t1 + pd.Timedelta("30min")]
        if seg_min.sum() < 5.0:
            continue
        rolling_max = float(rolling30.loc[t0: t1 + pd.Timedelta("30min")].max())
        binned_max = float(binned30.loc[t0: t1].max())
        if binned_max > 0:
            cfs.append(rolling_max / binned_max)
    cfs = np.array(cfs)
    if cfs.size == 0:
        return {"n_storms": 0, "cf_mean": np.nan, "cf_median": np.nan,
                "cf_p25": np.nan, "cf_p75": np.nan, "cfs": cfs}
    return {"n_storms": int(cfs.size),
            "cf_mean": float(cfs.mean()),
            "cf_median": float(np.median(cfs)),
            "cf_p25": float(np.percentile(cfs, 25)),
            "cf_p75": float(np.percentile(cfs, 75)),
            "cfs": cfs}


def activation_rate(rate_30min_series: pd.Series, cfg: DetectorConfig) -> dict:
    """Run the detector and report which criterion (peak vs depth)
    activated each erosive event."""
    intens_mm_h = (rate_30min_series.to_numpy() / cfg.dt_hours)
    res = detect_events(intens_mm_h, cfg)
    n_total = len(res.events)
    n_erosive = sum(1 for e in res.events if e.erosive)
    n_by_peak_only = sum(
        1 for e in res.events
        if e.erosive and e.has_peak and e.depth_mm < cfg.erosive_depth_mm
    )
    n_by_depth_only = sum(
        1 for e in res.events
        if e.erosive and (not e.has_peak) and e.depth_mm >= cfg.erosive_depth_mm
    )
    n_by_both = sum(
        1 for e in res.events
        if e.erosive and e.has_peak and e.depth_mm >= cfg.erosive_depth_mm
    )
    return {
        "n_total_events": n_total,
        "n_erosive": n_erosive,
        "n_by_peak_only": n_by_peak_only,
        "n_by_depth_only": n_by_depth_only,
        "n_by_both": n_by_both,
        "annual_R": float(res.annual_R),
    }


def gap_intensity_sensitivity(rate_30min_series: pd.Series,
                              gap_grid=(0.5, 1.0, 1.27, 2.0, 3.0)) -> pd.DataFrame:
    intens_mm_h = (rate_30min_series.to_numpy() / 0.5)
    rows = []
    for g in gap_grid:
        cfg = DetectorConfig(gap_intensity_mm_h=g, energy_model="bf", exp_k=0.082)
        res = detect_events(intens_mm_h, cfg)
        rows.append({
            "gap_intensity_mm_h": g,
            "n_events": len(res.events),
            "n_erosive": sum(1 for e in res.events if e.erosive),
            "annual_R": float(res.annual_R),
        })
    df = pd.DataFrame(rows)
    if (df["annual_R"] > 0).any():
        ref = df[df["gap_intensity_mm_h"] == 1.27]["annual_R"].iloc[0]
        df["rel_to_default_pct"] = 100.0 * (df["annual_R"] - ref) / ref
    return df


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    print("=" * 60)
    print("Event-detector sensitivity pack")
    print("=" * 60)

    if not os.path.exists(args.aws_csv):
        print(f"AWS CSV not found: {args.aws_csv}")
        return

    print("Loading 1-min Pluvio2 series from Kazan AWS...")
    minute_series = load_kazan_minute_series(args.aws_csv)
    print(f"  {len(minute_series)} 1-min samples; {minute_series.sum():.1f} mm cumulative liquid")

    # 1. Sub-bin CF
    print("\n[1] Sub-bin I_30 correction factor (true 30-min rolling vs 30-min binned):")
    cf = sub_bin_cf_diagnostic(minute_series)
    print(f"  n_storms = {cf['n_storms']}")
    print(f"  CF mean   = {cf['cf_mean']:.3f}")
    print(f"  CF median = {cf['cf_median']:.3f} (P25 = {cf['cf_p25']:.3f}, P75 = {cf['cf_p75']:.3f})")

    # 2. Activation rate (event detector at 30-min, k=0.082)
    print("\n[2] Erosivity-criterion activation rate (at IMERG 30-min equivalent):")
    rate30 = minute_series.resample("30min").sum().fillna(0.0)
    cfg = DetectorConfig(energy_model="bf", exp_k=0.082)
    act = activation_rate(rate30, cfg)
    print(f"  total events:             {act['n_total_events']}")
    print(f"  erosive events:           {act['n_erosive']}")
    print(f"    triggered by peak only: {act['n_by_peak_only']}")
    print(f"    triggered by depth only:{act['n_by_depth_only']}")
    print(f"    triggered by both:      {act['n_by_both']}")
    print(f"  annual R (full record):   {act['annual_R']:.1f}")

    # 3. gap_intensity sensitivity
    print("\n[3] Sensitivity to gap_intensity threshold (mm/h):")
    sens = gap_intensity_sensitivity(rate30)
    print(sens.to_string(index=False))

    # ------------------------------------------------------------- #
    # Save
    # ------------------------------------------------------------- #
    out_csv = os.path.join(args.output_dir, "tables", "sensitivity_pack.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_rows = [
        {"category": "subbin_CF", "metric": "n_storms", "value": cf["n_storms"]},
        {"category": "subbin_CF", "metric": "CF_mean", "value": cf["cf_mean"]},
        {"category": "subbin_CF", "metric": "CF_median", "value": cf["cf_median"]},
        {"category": "subbin_CF", "metric": "CF_P25", "value": cf["cf_p25"]},
        {"category": "subbin_CF", "metric": "CF_P75", "value": cf["cf_p75"]},
        {"category": "activation", "metric": "n_total_events", "value": act["n_total_events"]},
        {"category": "activation", "metric": "n_erosive", "value": act["n_erosive"]},
        {"category": "activation", "metric": "n_by_peak_only", "value": act["n_by_peak_only"]},
        {"category": "activation", "metric": "n_by_depth_only", "value": act["n_by_depth_only"]},
        {"category": "activation", "metric": "n_by_both", "value": act["n_by_both"]},
        {"category": "activation", "metric": "annual_R", "value": act["annual_R"]},
    ]
    for _, r in sens.iterrows():
        for col, val in r.items():
            summary_rows.append({"category": f"gap_intensity={r['gap_intensity_mm_h']}",
                                 "metric": col, "value": val})
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\n  saved: {out_csv}")

    # ------------------------------------------------------------- #
    # Plot
    # ------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if cf["n_storms"] > 0:
        axes[0].hist(cf["cfs"], bins=20, color="#4575b4", alpha=0.8, edgecolor="white")
        axes[0].axvline(cf["cf_median"], color="red", lw=2,
                        label=f"median CF = {cf['cf_median']:.2f}")
        axes[0].axvline(1.0, color="gray", ls="--", lw=1, label="CF = 1 (no bias)")
    axes[0].set_xlabel("CF = I_30(true rolling) / I_30(bin-aligned)")
    axes[0].set_ylabel("Storms")
    axes[0].set_title(f"(a) Sub-bin I_30 correction factor (n={cf['n_storms']} storms, Kazan AWS)",
                      fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if not sens.empty:
        x = sens["gap_intensity_mm_h"].values
        y_R = sens["annual_R"].values
        y_E = sens["n_erosive"].values
        ax2 = axes[1]
        ax2.plot(x, y_R, "o-", color="#d73027", label="annual R")
        ax2.set_xlabel("gap_intensity (mm/h)")
        ax2.set_ylabel("annual R, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹", color="#d73027")
        ax2.tick_params(axis="y", labelcolor="#d73027")
        ax2.axvline(1.27, color="gray", ls=":", label="default 1.27 mm/h")
        ax3 = ax2.twinx()
        ax3.plot(x, y_E, "s-", color="#1a9641", label="N erosive events")
        ax3.set_ylabel("N erosive events", color="#1a9641")
        ax3.tick_params(axis="y", labelcolor="#1a9641")
        ax2.set_title("(b) Sensitivity of annual R and event count to gap_intensity",
                      fontweight="bold")
        ax2.grid(alpha=0.3)

    out_png = os.path.join(args.output_dir, "plots", "sensitivity_pack.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_png}")


if __name__ == "__main__":
    main()
