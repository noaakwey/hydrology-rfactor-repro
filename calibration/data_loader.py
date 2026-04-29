import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

ALL_3H_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)

def load_meteo_station(filepath):
    """
    Loads meteo station data from CSV.
    Retains only 3-hour accumulation intervals for direct comparison with IMERG 
    (hours: 3, 6, 15, 18).
    Keeps 00 and 12 UTC for daily/monthly controls.
    """
    df = pd.read_csv(filepath, sep=';', encoding='cp866')
    
    # Required columns: Index, Year, Month, Day, Hour, Precipitation, StationName, X, Y, H
    df['datetime'] = pd.to_datetime(
        dict(year=df['Year'], month=df['Month'], day=df['Day'], hour=df['Hour'])
    )
    
    df = df.rename(columns={
        'Index': 'wmo_index',
        'Precipitation': 'P_station_mm',
        'StationName': 'station_name'
    })
    
    # Extract only necessary columns
    df = df[['wmo_index', 'station_name', 'datetime', 'Hour', 'P_station_mm', 'X', 'Y', 'H']]
    return df

def load_satellite_data(folder_path, file_pattern):
    """
    Loads all satellite/reanalysis CSVs from the provided folder.
    Concatenates them into a single dataframe.
    """
    all_files = glob.glob(os.path.join(folder_path, file_pattern))
    
    df_list = []
    for f in all_files:
        _df = pd.read_csv(f)
        df_list.append(_df)
        
    sat_df = pd.concat(df_list, ignore_index=True)
    
    sat_df['datetime'] = pd.to_datetime(sat_df['time_utc'])
    sat_df = sat_df.rename(columns={
        'Index': 'wmo_index',
        'Name': 'station_name',
        'P_mm_3h': 'P_sat_mm'
    })
    
    # Drop NaNs: missing slots should be excluded from pairwise comparison
    sat_df = sat_df.dropna(subset=['P_sat_mm'])
    
    return sat_df[['wmo_index', 'datetime', 'P_sat_mm']]

def pair_datasets(meteo_df, sat_df, force_full_3h_grid=True):
    """
    Pairs satellite data to station data on wmo_index and datetime.
    Filters to 3-hour synoptic terms.
    Crucially, creates a COMPLETE time grid between min and max dates.
    Missing records for both Satellite and Meteo are filled with P = 0.
    This prevents severe QM overestimation on continuous raster data.
    """
    if force_full_3h_grid:
        target_hours = set(ALL_3H_HOURS)
    else:
        # Legacy behavior: only hours observed in station file.
        target_hours = set(meteo_df['Hour'].unique()).intersection(set(ALL_3H_HOURS))
        if not target_hours:
            target_hours = set(ALL_3H_HOURS)

    sat_filtered = sat_df[sat_df['datetime'].dt.hour.isin(target_hours)].copy()
    meteo_filtered = meteo_df[meteo_df['Hour'].isin(target_hours)][['wmo_index', 'datetime', 'P_station_mm']]
    # Defensive de-duplication before merge to avoid accidental row multiplication.
    sat_filtered = sat_filtered.groupby('datetime', as_index=False)['P_sat_mm'].sum()
    meteo_filtered = meteo_filtered.groupby('datetime', as_index=False)['P_station_mm'].sum()

    wmo = meteo_df['wmo_index'].iloc[0]
    
    # Define complete time range
    min_time = min(sat_filtered['datetime'].min() if len(sat_filtered) else pd.Timestamp('2099-01-01'), 
                   meteo_filtered['datetime'].min() if len(meteo_filtered) else pd.Timestamp('2099-01-01'))
    max_time = max(sat_filtered['datetime'].max() if len(sat_filtered) else pd.Timestamp('1900-01-01'), 
                   meteo_filtered['datetime'].max() if len(meteo_filtered) else pd.Timestamp('1900-01-01'))
                   
    if min_time > max_time:
        return pd.DataFrame(columns=['wmo_index', 'datetime', 'P_sat_mm', 'P_station_mm'])

    # Create complete 3-hour grid and keep canonical synoptic terms
    full_idx = pd.date_range(min_time, max_time, freq='3h')
    full_idx = full_idx[full_idx.hour.isin(target_hours)]
    
    full_df = pd.DataFrame({'datetime': full_idx, 'wmo_index': wmo})
    
    # Merge Satellite
    merged = pd.merge(full_df, sat_filtered[['datetime', 'P_sat_mm']], on='datetime', how='left')
    # Merge Meteo
    merged = pd.merge(merged, meteo_filtered[['datetime', 'P_station_mm']], on='datetime', how='left')
    
    # Fill NAs with 0.0 (no precipitation reported)
    merged['P_sat_mm'] = merged['P_sat_mm'].fillna(0.0)
    merged['P_station_mm'] = merged['P_station_mm'].fillna(0.0)
    
    return merged
