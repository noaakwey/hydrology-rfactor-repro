import numpy as np
import pandas as pd
import scipy.stats as stats

def get_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    else:
        return 'SON'

def compute_quantiles(series, q_range):
    """
    Computes exact quantiles for a series.
    series: pandas Series of non-zero values.
    q_range: array of probabilities [0.01, 0.02, ..., 0.99]
    """
    if len(series) == 0:
        return np.zeros_like(q_range)
    return np.quantile(series, q_range)

def fit_highres_qm(p_sat_all, p_st_all, num_quantiles=1000):
    """
    Fits High-Resolution Empirical Quantile Mapping (EQM) for Big Data (e.g. ERA5).
    Uses a dense array of quantiles to map the shape without parametric smoothing.
    """
    # 1. Frequency Adaptation (Calculate PoP)
    pop_st = np.sum(p_st_all > 0) / len(p_st_all) if len(p_st_all) > 0 else 0
    
    if pop_st == 0:
        return None, None, 1.0, 0.0
        
    p_th = np.quantile(p_sat_all, 1.0 - pop_st)
    if p_th < 0:
        p_th = 0
        
    p_sat_wet = p_sat_all[p_sat_all > p_th]
    p_st_wet = p_st_all[p_st_all > 0]
    
    if len(p_sat_wet) < 20 or len(p_st_wet) < 20:
        return None, None, 1.0, p_th
        
    # Create high-res linear space (e.g., 0.001 to 0.999)
    # Using linspace avoids hard 0 and 1 which can cause infinity in parametric, though eqm handles it fine
    quantiles_list = np.linspace(0.001, 0.999, num_quantiles)
    
    q_sat = compute_quantiles(p_sat_wet, quantiles_list)
    q_station = compute_quantiles(p_st_wet, quantiles_list)
    
    # Linear extrapolation for the extreme right tail (beyond 99.9%)
    if (q_sat[-1] - q_sat[-5]) > 0:
        dy = q_station[-1] - q_station[-5]
        dx = q_sat[-1] - q_sat[-5]
        slope_extrap = dy / dx if dx > 0 else 1.0
    else:
        slope_extrap = 1.0
        
    return q_sat, q_station, slope_extrap, p_th

def fit_qm_transfer(p_sat_all, p_st_all, quantiles_list=np.arange(0.01, 1.00, 0.01)):
    """
    Fits standard Wet-Day QM transfer function between Satellite and station (For IMERG).
    """
    p_th = 0.0
    p_sat_wet = p_sat_all[p_sat_all > 0]
    p_st_wet = p_st_all[p_st_all > 0]
    
    if len(p_sat_wet) < 5 or len(p_st_wet) < 5:
        return None, None, 1.0, p_th
        
    # Calculate quantiles
    q_sat = compute_quantiles(p_sat_wet, quantiles_list)
    q_station = compute_quantiles(p_st_wet, quantiles_list)
    
    # Linear extrapolation for values above 99th percentile
    if len(p_sat_wet) > 20 and len(p_st_wet) > 20 and (q_sat[-1] - q_sat[-5]) > 0:
        dy = q_station[-1] - q_station[-5]
        dx = q_sat[-1] - q_sat[-5]
        slope_extrap = dy / dx if dx > 0 else 1.0
    else:
        slope_extrap = 1.0
        
    return q_sat, q_station, slope_extrap, p_th

def apply_qm(val, q_sat, q_station, slope_extrap, p_th):
    """
    Applies the fitted QM transformation with precipitation thresholding (p_th).
    """
    if isinstance(val, (int, float)):
        return _apply_qm_scalar(val, q_sat, q_station, slope_extrap, p_th)
    elif isinstance(val, pd.Series):
        val_arr = val.values
        res = np.zeros_like(val_arr, dtype=float)
        
        # Values strictly greater than threshold get mapped
        pos_mask = val_arr > p_th
        res[pos_mask] = [_apply_qm_scalar(v, q_sat, q_station, slope_extrap, p_th) for v in val_arr[pos_mask]]
        return res
    else:
        # np array
        res = np.zeros_like(val, dtype=float)
        pos_mask = val > p_th
        res[pos_mask] = [_apply_qm_scalar(v, q_sat, q_station, slope_extrap, p_th) for v in val[pos_mask]]
        return res

def _apply_qm_scalar(v, q_sat, q_station, slope_extrap, p_th):
    if v <= p_th:
        return 0.0
    if v <= q_sat[0]:
        # Interp between threshold and 1st quantile
        # Map values between p_th and q_sat[0] linearly to 0 -> q_station[0]
        dx = q_sat[0] - p_th
        if dx > 0:
            return q_station[0] * ((v - p_th) / dx)
        else:
            return q_station[0]
            
    if v >= q_sat[-1]:
        # Extrapolate beyond 99th
        return q_station[-1] + slope_extrap * (v - q_sat[-1])
    
    # Interpolate using binary search
    idx = np.searchsorted(q_sat, v)
    if idx == 0: return q_station[0] # Fallback
    
    x0, x1 = q_sat[idx-1], q_sat[idx]
    y0, y1 = q_station[idx-1], q_station[idx]
    
    if x1 == x0:
        return y0
    
    return y0 + (y1 - y0) * ((v - x0) / (x1 - x0))


def _fit_season_models(train_df, seasons=('DJF', 'MAM', 'JJA', 'SON'),
                       num_q=1000, min_samples=30, vf_clip=(0.7, 1.3)):
    """Fit seasonal full-distribution QM + bounded volume factor."""
    q_levels = np.linspace(0.0, 1.0, num_q + 2)[1:-1]
    models = {}

    for season in seasons:
        sd = train_df[train_df['season'] == season]
        p_sat = sd['P_sat_mm'].values
        p_st = sd['P_station_mm'].values
        if len(p_sat) < min_samples or len(p_st) < min_samples:
            continue

        q_sat = np.quantile(p_sat, q_levels)
        q_st = np.quantile(p_st, q_levels)
        if (q_sat[-1] - q_sat[-5]) > 0:
            slope = (q_st[-1] - q_st[-5]) / (q_sat[-1] - q_sat[-5])
        else:
            slope = 1.0

        train_corr = np.maximum(apply_qm(sd['P_sat_mm'], q_sat, q_st, slope, p_th=0.0), 0.0)
        mean_st = float(np.mean(p_st))
        mean_corr = float(np.mean(train_corr))
        vf = float(np.clip(mean_st / mean_corr, vf_clip[0], vf_clip[1])) if mean_corr > 0 else 1.0

        models[season] = {
            'q_sat': q_sat,
            'q_st': q_st,
            'slope': slope,
            'p_th': 0.0,
            'vf': vf,
        }

    return models


def _apply_season_models(df, models):
    corrected = df['P_sat_mm'].to_numpy(dtype=float, copy=True)
    for season, params in models.items():
        mask = (df['season'] == season).to_numpy()
        if not np.any(mask):
            continue
        vals = df.loc[mask, 'P_sat_mm']
        mapped = apply_qm(vals, params['q_sat'], params['q_st'], params['slope'], params['p_th'])
        corrected[mask] = np.maximum(mapped, 0.0) * params['vf']
    return corrected


def _calc_pbias(sim, obs):
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = np.asarray(sim)[mask]
    obs = np.asarray(obs)[mask]
    if len(obs) == 0:
        return np.nan
    denom = float(np.sum(obs))
    if denom == 0:
        return np.nan
    return 100.0 * float(np.sum(sim - obs)) / denom


def _calc_kge(sim, obs):
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = np.asarray(sim)[mask]
    obs = np.asarray(obs)[mask]
    if len(obs) < 2 or np.std(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def _score_candidate(df_tune, candidate):
    tmp = df_tune[['datetime', 'P_station_mm']].copy()
    tmp['P_candidate_mm'] = candidate

    tmp['date'] = tmp['datetime'].dt.floor('D')
    daily = tmp.groupby('date', as_index=False)[['P_station_mm', 'P_candidate_mm']].sum()
    daily_kge = _calc_kge(daily['P_candidate_mm'].values, daily['P_station_mm'].values)
    daily_pbias = _calc_pbias(daily['P_candidate_mm'].values, daily['P_station_mm'].values)

    tmp['ym'] = tmp['datetime'].dt.to_period('M')
    monthly = tmp.groupby('ym', as_index=False)[['P_station_mm', 'P_candidate_mm']].sum()
    monthly_kge = _calc_kge(monthly['P_candidate_mm'].values, monthly['P_station_mm'].values)
    monthly_pbias = _calc_pbias(monthly['P_candidate_mm'].values, monthly['P_station_mm'].values)

    tmp['year'] = tmp['datetime'].dt.year
    annual = tmp.groupby('year', as_index=False)[['P_station_mm', 'P_candidate_mm']].sum()
    annual_pbias = _calc_pbias(annual['P_candidate_mm'].values, annual['P_station_mm'].values)

    dk = -2.0 if pd.isna(daily_kge) else daily_kge
    mk = -2.0 if pd.isna(monthly_kge) else monthly_kge
    dp = 200.0 if pd.isna(daily_pbias) else abs(daily_pbias)
    mp = 200.0 if pd.isna(monthly_pbias) else abs(monthly_pbias)
    ap = 200.0 if pd.isna(annual_pbias) else abs(annual_pbias)

    return (0.65 * mk + 0.45 * dk - 0.003 * mp - 0.002 * dp - 0.001 * ap)


def _select_blend_alpha(df, train_mask, default_alpha=0.2):
    """Blocked split on train period: subtrain fit -> tune choose alpha."""
    train_df = df[train_mask].copy()
    if len(train_df) < 2500:
        return float(default_alpha)

    split_date = train_df['datetime'].quantile(0.75)
    subtrain_mask = train_mask & (df['datetime'] <= split_date)
    tune_mask = train_mask & (df['datetime'] > split_date)
    if int(np.sum(subtrain_mask)) < 1500 or int(np.sum(tune_mask)) < 500:
        return float(default_alpha)

    models_sub = _fit_season_models(df[subtrain_mask], min_samples=25, vf_clip=(0.75, 1.25))
    if len(models_sub) < 2:
        return float(default_alpha)

    qm_sub = _apply_season_models(df, models_sub)
    tune_df = df[tune_mask][['datetime', 'P_station_mm', 'P_sat_mm']].copy()
    raw_tune = tune_df['P_sat_mm'].to_numpy(dtype=float)
    qm_tune = qm_sub[tune_mask]

    alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4]
    best_alpha = float(default_alpha)
    best_score = -1e18
    for alpha in alpha_grid:
        cand = raw_tune + alpha * (qm_tune - raw_tune)
        score = _score_candidate(tune_df, cand)
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    return best_alpha


def _apply_annual_soft_guard(df, corrected_col, train_mask):
    """Soft annual ratio guard to prevent runaway annual sums."""
    train = df[train_mask].copy()
    if train.empty:
        return df

    train['year'] = train['datetime'].dt.year
    annual_train = train.groupby('year', as_index=False)[['P_station_mm', 'P_sat_mm']].sum()
    annual_train = annual_train[annual_train['P_sat_mm'] > 0]
    if len(annual_train) < 3:
        return df

    ratio = (annual_train['P_station_mm'] / annual_train['P_sat_mm']).values
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) < 3:
        return df

    low = max(0.0, float(np.percentile(ratio, 10)) * 0.9)
    high = max(low + 1e-6, float(np.percentile(ratio, 90)) * 1.1)

    out = df.copy()
    out['year'] = out['datetime'].dt.year
    for _, idx in out.groupby('year').groups.items():
        y_raw = float(out.loc[idx, 'P_sat_mm'].sum())
        y_corr = float(out.loc[idx, corrected_col].sum())
        if y_raw <= 0 or y_corr <= 0:
            continue

        lower = y_raw * low
        upper = y_raw * high
        if y_corr < lower:
            factor = lower / y_corr
        elif y_corr > upper:
            factor = upper / y_corr
        else:
            continue

        factor = float(np.clip(factor, 0.85, 1.15))
        out.loc[idx, corrected_col] = out.loc[idx, corrected_col] * factor

    out.drop(columns=['year'], inplace=True)
    return out

def calibrate_station(paired_df, train_start='2001-01-01', train_end='2015-12-31', dataset='imerg'):
    """
    Station calibration.
    IMERG: adaptive soft-QM (close to synoptic series + annual soft guard).
    ERA5-Land: classic full-distribution seasonal QM.
    """
    df = paired_df.copy()
    if 'season' not in df.columns:
        df['season'] = df['datetime'].dt.month.apply(get_season)

    train_mask = (df['datetime'] >= train_start) & (df['datetime'] <= train_end)
    df['P_corrected_mm'] = df['P_sat_mm']

    if dataset == 'era5land':
        models = _fit_season_models(df[train_mask], min_samples=20, vf_clip=(0.5, 2.0))
        if models:
            df['P_corrected_mm'] = np.maximum(_apply_season_models(df, models), 0.0)
        return df

    # IMERG adaptive branch
    models_full = _fit_season_models(df[train_mask], min_samples=25, vf_clip=(0.75, 1.25))
    if not models_full:
        return df

    qm_full = np.maximum(_apply_season_models(df, models_full), 0.0)
    alpha = _select_blend_alpha(df, train_mask, default_alpha=0.2)
    blended = df['P_sat_mm'].values + alpha * (qm_full - df['P_sat_mm'].values)
    df['P_corrected_mm'] = np.maximum(blended, 0.0)

    df = _apply_annual_soft_guard(df, corrected_col='P_corrected_mm', train_mask=train_mask)
    return df

def calibrate_station_moving_window(paired_df, half_window=15, val_start=None, val_end=None):
    """
    Moving Window Full-Distribution QM для ERA5-Land.
    Для каждого года Y строит QM на данных из окна [Y-half_window, Y+half_window].
    При кросс-валидации (val_start/val_end заданы) данные валидационного периода
    исключаются из обучающего окна.
    """
    df = paired_df.copy()
    if 'season' not in df.columns:
        df['season'] = df['datetime'].dt.month.apply(get_season)
    
    df['P_corrected_mm'] = df['P_sat_mm']  # fallback
    df['year'] = df['datetime'].dt.year
    
    all_years = sorted(df['year'].unique())
    min_year, max_year = all_years[0], all_years[-1]
    
    num_q = 1000
    quantiles_list = np.linspace(0.0, 1.0, num_q + 2)[1:-1]
    
    for target_year in all_years:
        # Определяем границы окна
        win_start = max(min_year, target_year - half_window)
        win_end = min(max_year, target_year + half_window)
        
        # Маска обучающего окна
        window_mask = (df['year'] >= win_start) & (df['year'] <= win_end)
        
        # При кросс-валидации: исключаем валидационный период из обучения
        if val_start is not None and val_end is not None:
            val_s = int(val_start[:4])
            val_e = int(val_end[:4])
            window_mask = window_mask & ~((df['year'] >= val_s) & (df['year'] <= val_e))
        
        target_mask = df['year'] == target_year
        
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            train_data = df[window_mask & (df['season'] == season)]
            p_sat_win = train_data['P_sat_mm'].values
            p_st_win = train_data['P_station_mm'].values
            
            if len(p_sat_win) < 30 or len(p_st_win) < 30:
                continue
            
            target_season_mask = target_mask & (df['season'] == season)
            if target_season_mask.sum() == 0:
                continue
            
            # Full-Distribution QM (1000 квантилей)
            q_sat = np.quantile(p_sat_win, quantiles_list)
            q_st = np.quantile(p_st_win, quantiles_list)
            
            # Экстраполяция хвоста
            if (q_sat[-1] - q_sat[-5]) > 0:
                slope_extrap = (q_st[-1] - q_st[-5]) / (q_sat[-1] - q_sat[-5])
            else:
                slope_extrap = 1.0
            
            # Применяем к целевому году
            corrected = apply_qm(df.loc[target_season_mask, 'P_sat_mm'],
                                 q_sat, q_st, slope_extrap, p_th=0.0)
            corrected = np.maximum(corrected, 0.0)
            
            # Volume Scaling по окну обучения
            train_corr = apply_qm(train_data['P_sat_mm'],
                                  q_sat, q_st, slope_extrap, p_th=0.0)
            train_corr = np.maximum(train_corr, 0.0)
            mean_st = np.mean(p_st_win)
            mean_corr = np.mean(train_corr)
            
            if mean_corr > 0:
                vf = np.clip(mean_st / mean_corr, 0.5, 2.0)
            else:
                vf = 1.0
            
            df.loc[target_season_mask, 'P_corrected_mm'] = corrected * vf
    
    df.drop(columns=['year'], inplace=True)
    return df
