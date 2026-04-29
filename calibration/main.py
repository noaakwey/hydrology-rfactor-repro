import argparse
import glob
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_loader import load_meteo_station, load_satellite_data, pair_datasets
from qm_calibration import calibrate_station, calibrate_station_moving_window
from validation import aggregate_and_validate

def main():
    parser = argparse.ArgumentParser(description="Calibrate precipitation data.")
    parser.add_argument("--dataset", type=str, choices=['imerg', 'era5land'], default='imerg',
                        help="Choose which dataset to calibrate (default: imerg)")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--meteo-dir", default=str(repo_root / "data" / "raw" / "meteo"))
    parser.add_argument("--sat-dir", default=None,
                        help="Optional explicit satellite directory. If omitted, a dataset-specific default is used.")
    parser.add_argument("--out-dir", default=None,
                        help="Optional explicit output directory. If omitted, a dataset-specific default is used.")
    parser.add_argument("--sat-pattern", default=None,
                        help="Optional filename pattern for satellite CSV files.")
    parser.add_argument("--train-start", default="2001-01-01")
    parser.add_argument("--train-end", default="2015-12-31")
    parser.add_argument("--val-start", default="2016-01-01")
    parser.add_argument("--val-end", default="2021-12-31")
    args = parser.parse_args()

    meteo_dir = args.meteo_dir
    
    if args.dataset == 'imerg':
        sat_dir = args.sat_dir or str(repo_root / "data" / "raw" / "imerg_stations_3h")
        sat_pattern = args.sat_pattern or "IMERG_V07_P3H_mm_*_permanent_trailing.csv"
        out_dir = args.out_dir or str(repo_root / "data" / "interim" / "calib_imerg")
    elif args.dataset == 'era5land':
        sat_dir = args.sat_dir or str(repo_root / "data" / "raw" / "era5land_stations_3h")
        sat_pattern = args.sat_pattern or "ERA5Land_P3H_mm_*.csv"
        out_dir = args.out_dir or str(repo_root / "data" / "interim" / "calib_era5land")
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {args.dataset.upper()} dataset (this may take a minute)...")
    sat_all = load_satellite_data(sat_dir, sat_pattern)
    print(f"{args.dataset.upper()} dataset loaded: {len(sat_all)} non-NaN records.")
    
    meteo_files = glob.glob(os.path.join(meteo_dir, "*.csv"))
    print(f"Found {len(meteo_files)} meteo stations.")
    
    all_metrics = []
    
    # Process all stations
    for mf in tqdm(meteo_files, desc="Processing stations"):
        st_name = os.path.splitext(os.path.basename(mf))[0]
        
        # 1. Load meteo
        try:
            meteo_df = load_meteo_station(mf)
        except Exception as e:
            print(f"Error loading {st_name}: {e}")
            continue
            
        if len(meteo_df) == 0:
            continue
            
        wmo_idx = meteo_df['wmo_index'].iloc[0]
        
        # 2. Filter Satellite to this station
        sat_st = sat_all[sat_all['wmo_index'] == wmo_idx]
        if sat_st.empty:
            continue
            
        # 3. Pair datasets
        paired_df = pair_datasets(meteo_df, sat_st, force_full_3h_grid=True)
        
        # 4. Calibrate
        if args.dataset == 'era5land':
            calibrated_df = calibrate_station_moving_window(
                paired_df, half_window=15, val_start=args.val_start, val_end=args.val_end)
        else:
            calibrated_df = calibrate_station(
                paired_df, train_start=args.train_start, train_end=args.train_end, dataset=args.dataset)
        
        # 5. Save calibration results
        out_csv = os.path.join(out_dir, f"{st_name}_{wmo_idx}_calib.csv")
        calibrated_df.to_csv(out_csv, index=False)
        
        # 6. Evaluate metrics
        metrics = aggregate_and_validate(calibrated_df, date_start=args.val_start, date_end=args.val_end)
        
        # Flatten metrics for logging
        row = {
            'wmo_index': wmo_idx,
            'station_name': st_name,
            'daily_KGE_raw': metrics['daily']['KGE_raw'],
            'daily_KGE_corr': metrics['daily']['KGE_corr'],
            'daily_PBIAS_raw': metrics['daily']['PBIAS_raw'],
            'daily_PBIAS_corr': metrics['daily']['PBIAS_corr'],
            'monthly_KGE_raw': metrics['monthly']['KGE_raw'],
            'monthly_KGE_corr': metrics['monthly']['KGE_corr'],
            'monthly_PBIAS_raw': metrics['monthly']['PBIAS_raw'],
            'monthly_PBIAS_corr': metrics['monthly']['PBIAS_corr'],
        }
        all_metrics.append(row)
        
    # Save overall metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_csv = os.path.join(out_dir, f"validation_metrics_{args.dataset}.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Metrics saved to {metrics_csv}")
        print(f"\nAverage metrics context run ({args.dataset.upper()}):")
        print(metrics_df[['daily_KGE_raw', 'daily_KGE_corr', 'monthly_KGE_raw', 'monthly_KGE_corr']].mean())

if __name__ == '__main__':
    main()
