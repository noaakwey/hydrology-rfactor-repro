import numpy as np
import pandas as pd

def calc_pbias(sim, obs):
    """
    Percent Bias (PBIAS)
    PBIAS = 100 * sum(sim - obs) / sum(obs)
    """
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    sim = sim[mask]
    obs = obs[mask]
    if len(obs) == 0 or np.sum(obs) == 0:
        return np.nan
    return 100.0 * np.sum(sim - obs) / np.sum(obs)

def calc_kge(sim, obs):
    """
    Kling-Gupta Efficiency (KGE)
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where:
    r = Pearson correlation coefficient between sim and obs
    alpha = std(sim) / std(obs)
    beta = mean(sim) / mean(obs)
    """
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    sim = sim[mask]
    obs = obs[mask]
    
    if len(obs) < 2 or np.std(obs) == 0:
        return np.nan
        
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    
    kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

def evaluate_metrics(df, obs_col='P_station_mm', raw_sim_col='P_sat_mm', corr_sim_col='P_corrected_mm'):
    """
    Evaluates PBIAS and KGE on the provided dataframe.
    """
    metrics = {}
    
    # Raw IMERG
    metrics['PBIAS_raw'] = calc_pbias(df[raw_sim_col].values, df[obs_col].values)
    metrics['KGE_raw'] = calc_kge(df[raw_sim_col].values, df[obs_col].values)
    
    # Corrected IMERG
    metrics['PBIAS_corr'] = calc_pbias(df[corr_sim_col].values, df[obs_col].values)
    metrics['KGE_corr'] = calc_kge(df[corr_sim_col].values, df[obs_col].values)
    
    return metrics

def aggregate_and_validate(paired_df, date_start='2016-01-01', date_end='2021-12-31'):
    """
    Aggregates to daily and monthly sums for the validation period and computes metrics.
    Note: To aggregate station data properly, we should ensure we are using all available data 
    for the station for accurate monthly totals, but for direct validation against IMERG, 
    we aggregate only the paired slots. For the 12-hour sums, we could merge them here,
    but keeping it simple by aggregating the paired 3h slots.
    """
    df = paired_df.copy()
    val_mask = (df['datetime'] >= date_start) & (df['datetime'] <= date_end)
    val_df = df[val_mask].copy()
    
    # Daily aggregation
    val_df['date'] = val_df['datetime'].dt.date
    daily_df = val_df.groupby('date').agg({
        'P_station_mm': 'sum',
        'P_sat_mm': 'sum',
        'P_corrected_mm': 'sum'
    }).reset_index()
    
    daily_metrics = evaluate_metrics(daily_df)
    
    # Monthly aggregation
    val_df['year_month'] = val_df['datetime'].dt.to_period('M')
    monthly_df = val_df.groupby('year_month').agg({
        'P_station_mm': 'sum',
        'P_sat_mm': 'sum',
        'P_corrected_mm': 'sum'
    }).reset_index()
    
    monthly_metrics = evaluate_metrics(monthly_df)
    
    return {
        'daily': daily_metrics,
        'monthly': monthly_metrics
    }
