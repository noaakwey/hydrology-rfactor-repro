from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from erosivity.lib.v6_hybrid import (
    apply_weighted_annual_sanity,
    apply_weighted_annual_transfer,
    apply_weighted_daily_constraint,
    blended_qm_field,
    build_station_weight_maps,
    extract_pixel_series,
    fit_peak_model,
    fit_station_models_for_year,
    get_season,
    load_aws_series,
    load_biomet_series,
    load_calibration_tables,
    load_station_metadata,
    read_year_from_quarters,
    redistribute_window_with_peaks,
)


IMERG_RE = re.compile(r"IMERG_V07_P30min_mmh_(\d{4})_(Q\d)_permanent\.tif$", re.IGNORECASE)
BIOMET_COORDS = {"lon": 49.2802, "lat": 55.2694}
AWS310_COORDS = {"lon": 48.81, "lat": 55.84}


def parse_args() -> argparse.Namespace:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Build IMERG v6 hybrid calibrated 30-min annual stacks.")
    parser.add_argument(
        "--zip",
        default=os.path.join(root, "data", "raw", "imerg_quarterly", "imerg_30min_quarters.zip"),
        help="Path to source IMERG ZIP archive with quarterly 30-min stacks.",
    )
    parser.add_argument(
        "--calib-dir",
        default=os.path.join(root, "data", "interim", "calib_imerg"),
        help="Directory with *_calib.csv station calibration tables.",
    )
    parser.add_argument(
        "--meteo-dir",
        default=os.path.join(root, "data", "raw", "meteo"),
        help="Directory with meteo station CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(root, "output", "v6_imerg_calibrated"),
        help="Directory for annual calibrated GeoTIFF outputs.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default=os.path.join(root, "output", "v6_diagnostics"),
        help="Directory for training diagnostics and metadata tables.",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2001,
        help="First year to process.",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2024,
        help="Last year to process.",
    )
    parser.add_argument(
        "--half-window",
        type=int,
        default=7,
        help="Half-width of the rolling training window in years.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=4,
        help="How many nearest stations to blend for each pixel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annual outputs.",
    )
    parser.add_argument(
        "--biomet-csv",
        default=os.path.join(root, "data", "external", "biomet_highfreq.csv"),
        help="Optional Biomet high-frequency CSV for peak-model training.",
    )
    parser.add_argument(
        "--aws-csv",
        default=os.path.join(root, "data", "external", "aws310_pluvio_v2.csv"),
        help="Optional AWS310 processed CSV for peak-model training.",
    )
    return parser.parse_args()


def group_tifs_by_year(root_dir: str) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for path in sorted(
        os.path.join(dp, f)
        for dp, _, files in os.walk(root_dir)
        for f in files
        if f.lower().endswith(".tif")
    ):
        match = IMERG_RE.search(os.path.basename(path))
        if match:
            grouped[int(match.group(1))].append(path)
    return {year: paths for year, paths in sorted(grouped.items())}


def save_float_stack(
    calib_mmh: np.ndarray,
    dts_arr: list[pd.Timestamp],
    profile: dict,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    profile_out = profile.copy()
    profile_out.update(
        dtype="float32",
        count=int(calib_mmh.shape[0]),
        compress="lzw",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    long_name = str(tuple(f"{i}_P_{dt.strftime('%Y%m%d_%H%M')}" for i, dt in enumerate(dts_arr)))
    with rasterio.open(out_path, "w", **profile_out) as dst:
        for band_idx in range(calib_mmh.shape[0]):
            dst.write(calib_mmh[band_idx].astype(np.float32), band_idx + 1)
        dst.update_tags(long_name=long_name, methodology="v6_hybrid_multi_station_peak_aware")


def build_peak_model(
    tif_paths_by_year: dict[int, list[str]],
    biomet_csv: str,
    aws_csv: str,
    diagnostics_dir: str,
) -> dict:
    raw_site_series: dict[str, pd.Series] = {}
    ground_site_series: dict[str, pd.Series] = {}

    if os.path.exists(aws_csv):
        ground_site_series["aws310"] = load_aws_series(aws_csv)
        raw_site_series["aws310"] = extract_pixel_series(
            tif_paths_by_year,
            lon=AWS310_COORDS["lon"],
            lat=AWS310_COORDS["lat"],
            years=ground_site_series["aws310"].index.year.unique(),
        )

    if os.path.exists(biomet_csv):
        ground_site_series["biomet"] = load_biomet_series(biomet_csv)
        raw_site_series["biomet"] = extract_pixel_series(
            tif_paths_by_year,
            lon=BIOMET_COORDS["lon"],
            lat=BIOMET_COORDS["lat"],
            years=ground_site_series["biomet"].index.year.unique(),
        )

    peak_model = fit_peak_model(raw_site_series=raw_site_series, ground_site_series=ground_site_series)
    training_table = peak_model.pop("training_table", pd.DataFrame())
    if not training_table.empty:
        os.makedirs(diagnostics_dir, exist_ok=True)
        training_table.to_csv(os.path.join(diagnostics_dir, "v6_peak_model_training.csv"), index=False, encoding="utf-8-sig")
    return peak_model


def process_year(
    year: int,
    tif_paths: list[str],
    out_dir: str,
    year_models,
    weight_maps,
    peak_model,
    overwrite: bool,
) -> str:
    out_path = os.path.join(out_dir, f"IMERG_V07_P30min_mmh_{year}_v6_qm.tif")
    if os.path.exists(out_path) and not overwrite:
        return out_path

    raw_mmh, dts_arr, profile = read_year_from_quarters(tif_paths)
    step_h = 0.5
    raw_mm = raw_mmh.astype(np.float64) * step_h
    calib_mm = np.zeros_like(raw_mm, dtype=np.float64)

    slots_per_3h = 6
    n_windows = raw_mm.shape[0] // slots_per_3h
    remainder = raw_mm.shape[0] % slots_per_3h

    for win in tqdm(range(n_windows), desc=f"v6 {year}", leave=False):
        i0 = win * slots_per_3h
        i1 = i0 + slots_per_3h
        raw_window = raw_mm[i0:i1]
        mid_dt = dts_arr[i0 + slots_per_3h // 2]
        season = get_season(mid_dt.month)
        corrected_total = blended_qm_field(raw_window.sum(axis=0), season, weight_maps, year_models)
        calib_mm[i0:i1] = redistribute_window_with_peaks(raw_window, corrected_total, mid_dt.month, peak_model)

    if remainder > 0:
        tail_start = n_windows * slots_per_3h
        calib_mm[tail_start:] = raw_mm[tail_start:]

    calib_mm = apply_weighted_daily_constraint(calib_mm, raw_mm, dts_arr, weight_maps, year_models)
    calib_mm = apply_weighted_annual_transfer(calib_mm, raw_mm, weight_maps, year_models)
    calib_mm = apply_weighted_annual_sanity(calib_mm, raw_mm, weight_maps, year_models)
    calib_mmh = np.maximum(calib_mm / step_h, 0.0).astype(np.float32)

    save_float_stack(calib_mmh, dts_arr, profile, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.diagnostics_dir, exist_ok=True)

    tmp_root = tempfile.mkdtemp(prefix="imerg_v6_")
    try:
        with zipfile.ZipFile(args.zip, "r") as zf:
            zf.extractall(tmp_root)

        tif_paths_by_year = group_tifs_by_year(tmp_root)
        years = [year for year in range(args.year_start, args.year_end + 1) if year in tif_paths_by_year]
        if not years:
            raise RuntimeError("No IMERG quarterly TIFFs found for requested year range.")

        station_meta = load_station_metadata(args.meteo_dir)
        station_tables = load_calibration_tables(args.calib_dir)

        sample_year = years[0]
        _, _, profile = read_year_from_quarters(tif_paths_by_year[sample_year])
        weight_maps = build_station_weight_maps(
            profile["transform"],
            profile["width"],
            profile["height"],
            station_meta,
            k_neighbors=args.k_neighbors,
        )

        peak_model = build_peak_model(
            tif_paths_by_year=tif_paths_by_year,
            biomet_csv=args.biomet_csv,
            aws_csv=args.aws_csv,
            diagnostics_dir=args.diagnostics_dir,
        )
        pd.DataFrame(
            {
                "key": list(peak_model.keys()),
                "value": [str(value) for value in peak_model.values()],
            }
        ).to_csv(os.path.join(args.diagnostics_dir, "v6_peak_model_meta.csv"), index=False, encoding="utf-8-sig")

        summary_rows: list[dict[str, float | int | str]] = []
        for year in years:
            year_models = fit_station_models_for_year(
                station_tables=station_tables,
                target_year=year,
                half_window_years=args.half_window,
            )
            out_path = process_year(
                year=year,
                tif_paths=tif_paths_by_year[year],
                out_dir=args.out_dir,
                year_models=year_models,
                weight_maps=weight_maps,
                peak_model=peak_model,
                overwrite=args.overwrite,
            )
            alpha_values = [bundle.blend_alpha for bundle in year_models.values()]
            summary_rows.append(
                {
                    "year": year,
                    "n_station_models": len(year_models),
                    "blend_alpha_median": float(np.median(alpha_values)) if alpha_values else np.nan,
                    "blend_alpha_mean": float(np.mean(alpha_values)) if alpha_values else np.nan,
                    "output_path": out_path,
                }
            )

        pd.DataFrame(summary_rows).to_csv(
            os.path.join(args.diagnostics_dir, "v6_year_model_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
