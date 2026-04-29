#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_rfactor_v7.py  — fast pre-loaded in-memory R-factor computation
from v7 advanced-calibrated IMERG rasters.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from erosivity.r_factor_rusle2 import (
    compute_R_year_preloaded,
    compute_period_mean,
    list_tifs_from_input,
    group_files_by_year,
    group_mask_quarters_by_year,
    build_year_mask_sequence,
    DatasetSpec,
    RConfig,
)
from erosivity.lib.energy_models import ENERGY_BROWN_FOSTER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute annual R-factor rasters from calibrated 30-min precipitation stacks.")
    parser.add_argument("--precip-dir", default=os.path.join(ROOT, "output", "v6_imerg_calibrated"))
    parser.add_argument("--mask-dir", default=os.path.join(ROOT, "data", "external", "era5_phase_masks"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "output", "v6_rfactor"))
    parser.add_argument("--year-start", type=int, default=2001)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--exp-k", type=float, default=0.082)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out_dir, "annual"), exist_ok=True)

    precip_files = list_tifs_from_input(args.precip_dir)
    mask_files   = list_tifs_from_input(args.mask_dir)
    if not precip_files:
        sys.exit(f"No calibrated tifs in {args.precip_dir}")
    if not mask_files:
        sys.exit(f"No mask tifs in {args.mask_dir}")

    precip_by_year = group_files_by_year(precip_files)
    mask_q_by_year = group_mask_quarters_by_year(mask_files)

    spec = DatasetSpec("imerg", dt_hours=0.5, mask_step_hours=1.0, cf_to_ei30=1.0)
    cfg  = RConfig(event_split_hours=6.0, event_split_sum_mm=1.27,
                   gap_intensity_mm_h=1.27, erosive_depth_mm=12.7,
                   erosive_peak_mm_h=25.4)

    annual_outputs = []
    for y in range(args.year_start, args.year_end + 1):
        p_list = precip_by_year.get(y, [])
        if not p_list:
            print(f"  {y}: no precip, skip")
            continue
        best = p_list[0]
        year_masks = build_year_mask_sequence(mask_q_by_year, y)
        if not year_masks:
            print(f"  {y}: no masks, skip")
            continue

        out_year = os.path.join(args.out_dir, "annual", f"R_imerg_{y}.tif")
        if os.path.exists(out_year) and args.overwrite:
            os.remove(out_year)
        elif os.path.exists(out_year):
            print(f"  {y}: exists, skip")
            annual_outputs.append(out_year)
            continue

        t0 = time.time()
        compute_R_year_preloaded(
            precip_year_file=best, mask_quarters=year_masks,
            out_path=out_year, year=y, spec=spec, cfg=cfg,
            energy_model=ENERGY_BROWN_FOSTER, exp_k=args.exp_k,
        )
        with rasterio.open(out_year) as ds:
            d = ds.read(1).astype(float); d[d == 0] = float("nan")
        m = float(np.nanmean(d))
        print(f"  {y}: mean_R={m:.1f}  ({time.time()-t0:.0f}s)")
        annual_outputs.append(out_year)

    if annual_outputs:
        out_mean = os.path.join(args.out_dir, f"R_imerg_{args.year_start}_{args.year_end}_MEAN.tif")
        if os.path.exists(out_mean) and args.overwrite:
            os.remove(out_mean)
        elif os.path.exists(out_mean):
            print(f"  mean exists: {out_mean}")
            return
        compute_period_mean(annual_outputs, out_mean)
        print(f"  mean: {out_mean}")


if __name__ == "__main__":
    main()
