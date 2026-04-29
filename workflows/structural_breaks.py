# -*- coding: utf-8 -*-
"""
structural_breaks.py — Structural-break detection in the domain-mean
R-factor time series.

Two tests:
  * Pettitt's test (single change-point, distribution-free)
  * Buishand range test (mean shift, normal-distribution assumption)

Particular attention is paid to whether a break falls in 2014–2017,
which would coincide with the IMERG V07 sensor-constellation transition
(SSMI/S retiring, GMI dominant) — a known potential source of artificial
inhomogeneity that any user of IMERG-derived climatology should rule out.

Outputs:
  output/tables/domain_annual_rfactor.csv
  output/tables/structural_break_summary.csv
  output/plots/structural_break_pettitt.png
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import DATA_DIR, OUTPUT_DIR

try:
    import pyhomogeneity as hg  # type: ignore
except ImportError:
    hg = None


def load_annual_means(data_dir: str) -> pd.DataFrame:
    rows = []
    for tif in sorted(glob.glob(os.path.join(data_dir, "R_imerg_*.tif"))):
        base = os.path.basename(tif)
        year_str = base.replace("R_imerg_", "").replace(".tif", "")
        if not year_str.isdigit():
            continue
        with rasterio.open(tif) as src:
            data = src.read(1).astype(np.float64)
        valid = (data > 0) & np.isfinite(data)
        rows.append({"year": int(year_str),
                     "r_factor": float(np.mean(data[valid])) if valid.any() else np.nan})
    return pd.DataFrame(rows).dropna().sort_values("year").reset_index(drop=True)


def pettitt_via_pyhom(series: pd.Series):
    if hg is None:
        return None
    return hg.pettitt_test(series)


def pettitt_manual(x: np.ndarray):
    """Manual Pettitt rank-statistic test (fallback when pyhomogeneity is
    unavailable).  Returns (k, U_max, p_two_sided)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    U = np.zeros(n)
    for k in range(1, n):
        s = 0.0
        for i in range(k):
            for j in range(k, n):
                s += np.sign(x[i] - x[j])
        U[k] = s
    k_star = int(np.argmax(np.abs(U)))
    Uk = float(U[k_star])
    p = 2.0 * np.exp((-6.0 * Uk * Uk) / (n ** 3 + n ** 2))
    p = float(min(max(p, 0.0), 1.0))
    return k_star, Uk, p


def buishand_range(x: np.ndarray):
    """Buishand range test for mean shift (assumes Gaussian noise)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = (x - x.mean()).cumsum()
    R = s.max() - s.min()
    R_norm = R / x.std(ddof=1)
    # Approx. critical value for Q/sqrt(n) at p=0.05 from Buishand 1982 tables;
    # we report Rstat for context only.
    return {"R_range": float(R), "R_normalised": float(R_norm), "n": int(n)}


def main():
    parser = argparse.ArgumentParser(description="Detect structural breaks in the domain-mean annual R-factor series.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    df = load_annual_means(args.data_dir)
    print(f"loaded {len(df)} years from {args.data_dir}")
    series = pd.Series(df["r_factor"].values, index=df["year"].values)
    print(series.describe())

    out_csv = os.path.join(args.output_dir, "tables", "domain_annual_rfactor.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv}")

    # --- Pettitt's test ---
    res = pettitt_via_pyhom(series)
    if res is not None:
        cp_idx = int(res.cp)
        cp_year = int(series.index[cp_idx])
        p_val = float(res.p)
        u_stat = float(getattr(res, "U", np.nan))
    else:
        cp_idx, u_stat, p_val = pettitt_manual(series.values)
        cp_year = int(series.index[cp_idx])

    mean_before = float(series.iloc[:cp_idx].mean()) if cp_idx > 0 else np.nan
    mean_after = float(series.iloc[cp_idx:].mean())
    print(f"\nPettitt test: cp_year={cp_year}, p={p_val:.4f}, U={u_stat}")
    print(f"  mean before {cp_year}: {mean_before:.1f}")
    print(f"  mean from   {cp_year}: {mean_after:.1f}")

    # --- Buishand ---
    bui = buishand_range(series.values)
    print(f"\nBuishand range stat: R/sigma = {bui['R_normalised']:.2f}  (R = {bui['R_range']:.1f})")

    # --- Sensor-transition flag (2014-2017) ---
    in_imerg_v07_window = 2014 <= cp_year <= 2017
    flag = "WARNING: cp falls in IMERG V07 sensor-transition window 2014-2017" \
           if in_imerg_v07_window else "OK: cp outside 2014-2017 sensor window"
    print(f"\n{flag}")

    # --- Save summary ---
    summary = pd.DataFrame([
        {"metric": "n_years", "value": len(series)},
        {"metric": "Pettitt change-point year", "value": cp_year},
        {"metric": "Pettitt p-value (two-sided)", "value": p_val},
        {"metric": "Pettitt significant at 0.05", "value": bool(p_val < 0.05)},
        {"metric": "Mean before change-point", "value": mean_before},
        {"metric": "Mean from change-point", "value": mean_after},
        {"metric": "Buishand R/sigma", "value": bui["R_normalised"]},
        {"metric": "Cp in IMERG V07 sensor window 2014-2017", "value": in_imerg_v07_window},
    ])
    out_sum = os.path.join(args.output_dir, "tables", "structural_break_summary.csv")
    summary.to_csv(out_sum, index=False)
    print(f"  saved: {out_sum}")

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values, marker="o", label="Domain-mean R-factor")
    plt.axvline(cp_year, color="red", linestyle="--",
                label=f"Pettitt cp: {cp_year} (p={p_val:.3f})")
    if cp_idx > 0:
        plt.hlines(mean_before, series.index[0], cp_year - 1, color="green",
                   label=f"Mean before {cp_year}: {mean_before:.1f}")
    plt.hlines(mean_after, cp_year, series.index[-1], color="purple",
               label=f"Mean from {cp_year}: {mean_after:.1f}")
    if 2014 <= cp_year <= 2017:
        plt.axvspan(2014, 2017, color="orange", alpha=0.15,
                    label="IMERG V07 sensor-transition window")

    if p_val < 0.05:
        plt.title("Statistically-significant structural break (Pettitt)")
    else:
        plt.title(f"No statistically-significant break (Pettitt p={p_val:.3f})")
    plt.xlabel("Year")
    plt.ylabel("R-factor (MJ·mm/(ha·h·yr))")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(args.output_dir, "plots", "structural_break_pettitt.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"  saved: {out_png}")


if __name__ == "__main__":
    main()
