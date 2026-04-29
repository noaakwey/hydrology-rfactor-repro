from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree


SEASONS = ("DJF", "MAM", "JJA", "SON")
_BAND_RE = re.compile(r"P_(\d{8})_(\d{4})")
PEAK_GAMMA_MIN = 0.70
PEAK_GAMMA_MAX = 3.50


@dataclass(frozen=True)
class StationModels:
    seasonal: Dict[str, Dict[str, np.ndarray | float]]
    daily: Optional[Dict[str, np.ndarray | float]]
    annual_transfer: Optional[Dict[str, np.ndarray | float]]
    annual_envelope: Optional[Dict[str, float]]
    blend_alpha: float


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def compute_quantiles(series: np.ndarray, q_levels: np.ndarray) -> np.ndarray:
    if series.size == 0:
        return np.zeros_like(q_levels, dtype=np.float64)
    return np.quantile(series, q_levels)


def apply_qm(
    values: np.ndarray | pd.Series | float,
    q_sat: np.ndarray,
    q_station: np.ndarray,
    slope_extrap: float,
    p_th: float,
) -> np.ndarray | float:
    if isinstance(values, (int, float, np.floating)):
        return _apply_qm_scalar(float(values), q_sat, q_station, slope_extrap, p_th)

    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    pos_mask = arr > p_th
    if not np.any(pos_mask):
        return out

    out[pos_mask] = np.array(
        [_apply_qm_scalar(v, q_sat, q_station, slope_extrap, p_th) for v in arr[pos_mask]],
        dtype=np.float64,
    )
    return out


def _apply_qm_scalar(
    value: float,
    q_sat: np.ndarray,
    q_station: np.ndarray,
    slope_extrap: float,
    p_th: float,
) -> float:
    if value <= p_th:
        return 0.0
    if value <= q_sat[0]:
        dx = q_sat[0] - p_th
        return q_station[0] * ((value - p_th) / dx) if dx > 0 else q_station[0]
    if value >= q_sat[-1]:
        return q_station[-1] + slope_extrap * (value - q_sat[-1])

    idx = int(np.searchsorted(q_sat, value))
    if idx == 0:
        return q_station[0]

    x0, x1 = q_sat[idx - 1], q_sat[idx]
    y0, y1 = q_station[idx - 1], q_station[idx]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * ((value - x0) / (x1 - x0))


def _fit_empirical_qm(
    p_sat: np.ndarray,
    p_station: np.ndarray,
    q_levels: np.ndarray,
    min_samples: int,
) -> Optional[Dict[str, np.ndarray | float]]:
    if p_sat.size < min_samples or p_station.size < min_samples:
        return None

    q_sat = compute_quantiles(p_sat, q_levels)
    q_station = compute_quantiles(p_station, q_levels)
    dx = q_sat[-1] - q_sat[-5]
    slope = float((q_station[-1] - q_station[-5]) / dx) if dx > 0 else 1.0
    return {
        "q_sat": q_sat.astype(np.float64),
        "q_station": q_station.astype(np.float64),
        "slope": slope,
        "p_th": 0.0,
    }


def _calc_pbias(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = np.asarray(sim, dtype=np.float64)[mask]
    obs = np.asarray(obs, dtype=np.float64)[mask]
    if obs.size == 0:
        return np.nan
    denom = float(np.sum(obs))
    if denom == 0.0:
        return np.nan
    return 100.0 * float(np.sum(sim - obs)) / denom


def _calc_kge(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = np.asarray(sim, dtype=np.float64)[mask]
    obs = np.asarray(obs, dtype=np.float64)[mask]
    if obs.size < 2 or float(np.std(obs)) == 0.0:
        return np.nan
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(np.std(sim) / np.std(obs))
    beta = float(np.mean(sim) / np.mean(obs))
    return 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def _score_candidate(df_tune: pd.DataFrame, candidate: np.ndarray) -> float:
    tmp = df_tune[["datetime", "P_station_mm"]].copy()
    tmp["P_candidate_mm"] = np.asarray(candidate, dtype=np.float64)

    tmp["date"] = tmp["datetime"].dt.floor("D")
    daily = tmp.groupby("date", as_index=False)[["P_station_mm", "P_candidate_mm"]].sum()
    daily_kge = _calc_kge(daily["P_candidate_mm"].values, daily["P_station_mm"].values)
    daily_pbias = _calc_pbias(daily["P_candidate_mm"].values, daily["P_station_mm"].values)

    tmp["ym"] = tmp["datetime"].dt.to_period("M")
    monthly = tmp.groupby("ym", as_index=False)[["P_station_mm", "P_candidate_mm"]].sum()
    monthly_kge = _calc_kge(monthly["P_candidate_mm"].values, monthly["P_station_mm"].values)
    monthly_pbias = _calc_pbias(monthly["P_candidate_mm"].values, monthly["P_station_mm"].values)

    dk = -2.0 if pd.isna(daily_kge) else float(daily_kge)
    mk = -2.0 if pd.isna(monthly_kge) else float(monthly_kge)
    dp = 200.0 if pd.isna(daily_pbias) else abs(float(daily_pbias))
    mp = 200.0 if pd.isna(monthly_pbias) else abs(float(monthly_pbias))

    obs = tmp["P_station_mm"].to_numpy(dtype=np.float64)
    sim = tmp["P_candidate_mm"].to_numpy(dtype=np.float64)
    proxy_obs = np.sum(obs * (obs / 3.0))
    proxy_sim = np.sum(sim * (sim / 3.0))
    proxy_pbias = 100.0 * (proxy_sim - proxy_obs) / proxy_obs if proxy_obs > 0.0 else np.nan

    wet_obs = obs[obs > 0.05]
    wet_sim = sim[sim > 0.05]
    if wet_obs.size > 10 and wet_sim.size > 10:
        q99_obs = float(np.percentile(wet_obs / 3.0, 99))
        q99_sim = float(np.percentile(wet_sim / 3.0, 99))
        q99_bias = 100.0 * (q99_sim - q99_obs) / q99_obs if q99_obs > 0.0 else np.nan
    else:
        q99_bias = np.nan

    rp = 200.0 if pd.isna(proxy_pbias) else abs(float(proxy_pbias))
    qp = 200.0 if pd.isna(q99_bias) else abs(float(q99_bias))
    return 0.70 * mk + 0.40 * dk - 0.003 * mp - 0.002 * dp - 0.006 * rp - 0.003 * qp


def _fit_seasonal_models(
    train_df: pd.DataFrame,
    q_levels: np.ndarray,
    min_samples: int = 30,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    models: Dict[str, Dict[str, np.ndarray | float]] = {}
    for season in SEASONS:
        season_df = train_df[train_df["season"] == season]
        params = _fit_empirical_qm(
            season_df["P_sat_mm"].to_numpy(dtype=np.float64),
            season_df["P_station_mm"].to_numpy(dtype=np.float64),
            q_levels=q_levels,
            min_samples=min_samples,
        )
        if params is not None:
            models[season] = params
    return models


def _apply_seasonal_models(
    df: pd.DataFrame,
    seasonal_models: Mapping[str, Mapping[str, np.ndarray | float]],
) -> np.ndarray:
    corrected = df["P_sat_mm"].to_numpy(dtype=np.float64, copy=True)
    for season, params in seasonal_models.items():
        mask = (df["season"] == season).to_numpy()
        if not np.any(mask):
            continue
        mapped = apply_qm(
            df.loc[mask, "P_sat_mm"].to_numpy(dtype=np.float64),
            np.asarray(params["q_sat"], dtype=np.float64),
            np.asarray(params["q_station"], dtype=np.float64),
            float(params["slope"]),
            float(params["p_th"]),
        )
        corrected[mask] = np.maximum(np.asarray(mapped, dtype=np.float64), 0.0)
    return corrected


def _select_blend_alpha(
    train_df: pd.DataFrame,
    seasonal_models: Mapping[str, Mapping[str, np.ndarray | float]],
    default_alpha: float = 0.45,
) -> float:
    if len(train_df) < 2500 or len(seasonal_models) < 2:
        return float(default_alpha)

    split_date = train_df["datetime"].quantile(0.75)
    subtrain = train_df[train_df["datetime"] <= split_date].copy()
    tune = train_df[train_df["datetime"] > split_date].copy()
    if len(subtrain) < 1500 or len(tune) < 500:
        return float(default_alpha)

    q_levels = np.linspace(0.0, 1.0, 301 + 2)[1:-1]
    sub_models = _fit_seasonal_models(subtrain, q_levels=q_levels, min_samples=25)
    if len(sub_models) < 2:
        return float(default_alpha)

    qm_tune = _apply_seasonal_models(tune, sub_models)
    raw_tune = tune["P_sat_mm"].to_numpy(dtype=np.float64)

    alpha_grid = [0.25, 0.35, 0.45, 0.55, 0.65]
    best_alpha = float(default_alpha)
    best_score = -1e18
    for alpha in alpha_grid:
        candidate = raw_tune + alpha * (qm_tune - raw_tune)
        score = _score_candidate(tune, candidate)
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def _fit_annual_transfer_model(annual_df: pd.DataFrame) -> Optional[Dict[str, np.ndarray | float]]:
    if annual_df.empty:
        return None
    data = annual_df[
        np.isfinite(annual_df["P_sat_mm"].values)
        & np.isfinite(annual_df["P_station_mm"].values)
        & (annual_df["P_sat_mm"].values > 0.0)
    ].copy()
    if len(data) < 5:
        return None

    x = data["P_sat_mm"].to_numpy(dtype=np.float64)
    y = data["P_station_mm"].to_numpy(dtype=np.float64)
    q_levels = np.linspace(0.10, 0.90, 17)
    xq = np.quantile(x, q_levels)
    yq = np.quantile(y, q_levels)
    xq_u, idx_u = np.unique(xq, return_index=True)
    yq_u = np.maximum.accumulate(yq[idx_u])

    if len(xq_u) < 4:
        return {
            "kind": "ratio",
            "ratio": float(np.median(np.clip(y / np.maximum(x, 1e-6), 0.2, 5.0))),
            "raw_p10": float(np.percentile(x, 10)),
            "raw_p50": float(np.percentile(x, 50)),
            "raw_p90": float(np.percentile(x, 90)),
        }

    n_tail = min(4, len(xq_u))
    dx_low = xq_u[n_tail - 1] - xq_u[0]
    dx_high = xq_u[-1] - xq_u[-n_tail]
    slope_low = float((yq_u[n_tail - 1] - yq_u[0]) / dx_low) if dx_low > 0 else 1.0
    slope_high = float((yq_u[-1] - yq_u[-n_tail]) / dx_high) if dx_high > 0 else 1.0
    slope_low = float(np.clip(slope_low, 0.1, 6.0))
    slope_high = float(np.clip(slope_high, 0.1, 6.0))

    return {
        "kind": "empirical",
        "xq": xq_u.astype(np.float64),
        "yq": yq_u.astype(np.float64),
        "slope_low": slope_low,
        "slope_high": slope_high,
        "raw_p10": float(np.percentile(x, 10)),
        "raw_p50": float(np.percentile(x, 50)),
        "raw_p90": float(np.percentile(x, 90)),
    }


def _map_annual_raw_to_target(raw_vals: np.ndarray, model: Mapping[str, np.ndarray | float]) -> np.ndarray:
    raw = np.asarray(raw_vals, dtype=np.float64)
    out = np.zeros_like(raw, dtype=np.float64)
    if model.get("kind") == "ratio":
        ratio = float(model.get("ratio", 1.0))
        out = raw * ratio
        out[raw <= 0.0] = 0.0
        return np.maximum(out, 0.0)

    xq = np.asarray(model["xq"], dtype=np.float64)
    yq = np.asarray(model["yq"], dtype=np.float64)
    slope_low = float(model.get("slope_low", 1.0))
    slope_high = float(model.get("slope_high", 1.0))

    out = np.interp(raw, xq, yq)
    low = raw < xq[0]
    high = raw > xq[-1]
    if np.any(low):
        out[low] = yq[0] + slope_low * (raw[low] - xq[0])
    if np.any(high):
        out[high] = yq[-1] + slope_high * (raw[high] - xq[-1])

    out[raw <= 0.0] = 0.0
    out[~np.isfinite(out)] = 0.0
    return np.maximum(out, 0.0)


def _fit_annual_envelope(annual_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    annual_df = annual_df[annual_df["P_sat_mm"] > 0.0].copy()
    if len(annual_df) < 4:
        return None

    ratio = np.clip(
        annual_df["P_station_mm"].to_numpy(dtype=np.float64)
        / np.maximum(annual_df["P_sat_mm"].to_numpy(dtype=np.float64), 1e-6),
        0.2,
        5.0,
    )
    return {
        "ratio_p10": float(np.percentile(ratio, 10)),
        "ratio_p50": float(np.percentile(ratio, 50)),
        "ratio_p90": float(np.percentile(ratio, 90)),
        "station_p90": float(np.percentile(annual_df["P_station_mm"].to_numpy(dtype=np.float64), 90)),
    }


def load_station_metadata(meteo_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in glob.glob(os.path.join(meteo_dir, "*.csv")):
        try:
            df = pd.read_csv(path, sep=";", encoding="cp866", nrows=1)
        except Exception:
            continue
        if {"Index", "StationName", "X", "Y"}.issubset(df.columns):
            rows.append(
                {
                    "wmo_index": int(df["Index"].iloc[0]),
                    "station_name": str(df["StationName"].iloc[0]),
                    "lon": float(df["Y"].iloc[0]),
                    "lat": float(df["X"].iloc[0]),
                }
            )
    if not rows:
        raise RuntimeError(f"No meteo stations found in {meteo_dir}")
    return pd.DataFrame(rows).drop_duplicates("wmo_index").sort_values("wmo_index").reset_index(drop=True)


def load_calibration_tables(calib_dir: str) -> Dict[int, pd.DataFrame]:
    tables: Dict[int, pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(calib_dir, "*_calib.csv"))):
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c in {"datetime", "wmo_index", "P_sat_mm", "P_station_mm", "P_corrected_mm", "season"},
            )
        except ValueError:
            df = pd.read_csv(path)
            needed = ["datetime", "wmo_index", "P_sat_mm", "P_station_mm"]
            if not set(needed).issubset(df.columns):
                continue
            if "season" not in df.columns:
                df["season"] = pd.to_datetime(df["datetime"]).dt.month.apply(get_season)
            cols = ["datetime", "wmo_index", "P_sat_mm", "P_station_mm"]
            if "P_corrected_mm" in df.columns:
                cols.append("P_corrected_mm")
            cols.append("season")
            df = df[cols]

        df["datetime"] = pd.to_datetime(df["datetime"])
        df["year"] = df["datetime"].dt.year
        df["season"] = df["datetime"].dt.month.apply(get_season)
        wmo = int(df["wmo_index"].iloc[0])
        tables[wmo] = df.sort_values("datetime").reset_index(drop=True)
    if not tables:
        raise RuntimeError(f"No calibration CSV files found in {calib_dir}")
    return tables


def fit_station_models_for_year(
    station_tables: Mapping[int, pd.DataFrame],
    target_year: int,
    half_window_years: int = 7,
    num_quantiles: int = 399,
    min_train_years: int = 5,
) -> Dict[int, StationModels]:
    q_levels = np.linspace(0.0, 1.0, num_quantiles + 2)[1:-1]
    year_models: Dict[int, StationModels] = {}

    for wmo, df in station_tables.items():
        years = df["year"].to_numpy()
        train_mask = (years >= target_year - half_window_years) & (years <= target_year + half_window_years) & (years != target_year)
        if np.unique(years[train_mask]).size < min_train_years:
            train_mask = years != target_year

        train_df = df.loc[train_mask].copy()
        if train_df.empty:
            continue

        seasonal_models = _fit_seasonal_models(train_df, q_levels=q_levels, min_samples=30)
        if len(seasonal_models) < 2:
            continue

        blend_alpha = _select_blend_alpha(train_df, seasonal_models, default_alpha=0.45)

        daily_df = (
            train_df.assign(date=train_df["datetime"].dt.floor("D"))
            .groupby("date", as_index=False)[["P_sat_mm", "P_station_mm"]]
            .sum()
        )
        daily_model = _fit_empirical_qm(
            daily_df["P_sat_mm"].to_numpy(dtype=np.float64),
            daily_df["P_station_mm"].to_numpy(dtype=np.float64),
            q_levels=q_levels,
            min_samples=25,
        )

        annual_df = (
            train_df.assign(year=train_df["datetime"].dt.year)
            .groupby("year", as_index=False)[["P_sat_mm", "P_station_mm"]]
            .sum()
        )
        annual_transfer = _fit_annual_transfer_model(annual_df)
        annual_envelope = _fit_annual_envelope(annual_df)

        year_models[wmo] = StationModels(
            seasonal=seasonal_models,
            daily=daily_model,
            annual_transfer=annual_transfer,
            annual_envelope=annual_envelope,
            blend_alpha=blend_alpha,
        )

    return year_models


def _project_lon_lat(lon: np.ndarray, lat: np.ndarray, lat0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    lat0 = math.radians(lat0_deg)
    x = lon * math.cos(lat0)
    y = lat
    return x, y


def build_station_weight_maps(
    transform,
    width: int,
    height: int,
    station_meta: pd.DataFrame,
    k_neighbors: int = 4,
    power: float = 1.75,
) -> Dict[int, np.ndarray]:
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    cc, rr = np.meshgrid(cols, rows)
    xs, ys = rasterio.transform.xy(transform, rr, cc, offset="center")
    lon = np.asarray(xs, dtype=np.float64)
    lat = np.asarray(ys, dtype=np.float64)

    lat0 = float(np.mean(station_meta["lat"].to_numpy(dtype=np.float64)))
    pix_x, pix_y = _project_lon_lat(lon.ravel(), lat.ravel(), lat0)
    st_x, st_y = _project_lon_lat(
        station_meta["lon"].to_numpy(dtype=np.float64),
        station_meta["lat"].to_numpy(dtype=np.float64),
        lat0,
    )

    tree = cKDTree(np.column_stack([st_x, st_y]))
    k_neighbors = min(k_neighbors, len(station_meta))
    dist, idx = tree.query(np.column_stack([pix_x, pix_y]), k=k_neighbors)
    if k_neighbors == 1:
        dist = dist[:, np.newaxis]
        idx = idx[:, np.newaxis]

    weights = 1.0 / np.maximum(dist, 1e-6) ** power
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    wmos = station_meta["wmo_index"].to_numpy(dtype=np.int64)
    weight_maps: Dict[int, np.ndarray] = {
        int(wmo): np.zeros(width * height, dtype=np.float64) for wmo in wmos
    }
    for nbr in range(k_neighbors):
        station_ids = wmos[idx[:, nbr]]
        for wmo in np.unique(station_ids):
            mask = station_ids == wmo
            weight_maps[int(wmo)][mask] += weights[mask, nbr]

    return {wmo: arr.reshape(height, width) for wmo, arr in weight_maps.items()}


def blended_qm_field(
    raw_field_mm: np.ndarray,
    season: str,
    weight_maps: Mapping[int, np.ndarray],
    year_models: Mapping[int, StationModels],
) -> np.ndarray:
    corrected = np.zeros_like(raw_field_mm, dtype=np.float64)
    raw_field = np.asarray(raw_field_mm, dtype=np.float64)

    for wmo, weight_map in weight_maps.items():
        if not np.any(weight_map > 0.0):
            continue
        station_bundle = year_models.get(int(wmo))
        if station_bundle is None:
            corrected += weight_map * raw_field
            continue

        params = station_bundle.seasonal.get(season)
        if params is None:
            corrected += weight_map * raw_field
            continue

        mapped = apply_qm(
            raw_field.ravel(),
            np.asarray(params["q_sat"], dtype=np.float64),
            np.asarray(params["q_station"], dtype=np.float64),
            float(params["slope"]),
            float(params["p_th"]),
        )
        mapped = np.maximum(np.asarray(mapped, dtype=np.float64).reshape(raw_field.shape), 0.0)
        blended = raw_field + station_bundle.blend_alpha * (mapped - raw_field)
        corrected += weight_map * np.maximum(blended, 0.0)

    return np.maximum(corrected, 0.0)


def parse_band_datetimes(long_name_value: str) -> List[pd.Timestamp]:
    matches = _BAND_RE.findall(str(long_name_value))
    return [pd.to_datetime(f"{date}{hhmm}", format="%Y%m%d%H%M") for date, hhmm in matches]


def read_tif_stack(path: str) -> Tuple[np.ndarray, List[pd.Timestamp], dict]:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        long_name_value = src.tags().get("long_name", "")
        dts = parse_band_datetimes(long_name_value)
        if len(dts) != data.shape[0]:
            desc_blob = " ".join([d for d in src.descriptions if d])
            dts = parse_band_datetimes(desc_blob)
    if len(dts) != data.shape[0]:
        raise ValueError(f"Failed to parse datetimes for {path}")
    data = np.where(np.isfinite(data), np.maximum(data, 0.0), 0.0)
    return data, dts, profile


def read_year_from_quarters(tif_paths: Sequence[str]) -> Tuple[np.ndarray, List[pd.Timestamp], dict]:
    stacks: List[np.ndarray] = []
    dts_all: List[pd.Timestamp] = []
    profile: Optional[dict] = None

    for path in sorted(tif_paths):
        data, dts, prof = read_tif_stack(path)
        stacks.append(data)
        dts_all.extend(dts)
        if profile is None:
            profile = prof

    order = np.argsort([dt.value for dt in dts_all])
    stack = np.concatenate(stacks, axis=0)[order]
    dts_sorted = [dts_all[i] for i in order]
    if profile is None:
        raise ValueError("No raster profile available")
    return stack, dts_sorted, profile


def apply_weighted_daily_constraint(
    calib_mm: np.ndarray,
    raw_mm: np.ndarray,
    dts_arr: Sequence[pd.Timestamp],
    weight_maps: Mapping[int, np.ndarray],
    year_models: Mapping[int, StationModels],
) -> np.ndarray:
    if calib_mm.size == 0:
        return calib_mm

    out = np.asarray(calib_mm, dtype=np.float64).copy()
    day_values = pd.Series(pd.to_datetime(list(dts_arr))).dt.floor("D").to_numpy()
    unique_days = np.unique(day_values)

    for day in unique_days:
        mask_idx = np.where(day_values == day)[0]
        raw_day = raw_mm[mask_idx].sum(axis=0)
        calib_day = out[mask_idx].sum(axis=0)
        target_day = np.zeros_like(raw_day, dtype=np.float64)

        for wmo, weight_map in weight_maps.items():
            station_bundle = year_models.get(int(wmo))
            if station_bundle is None or station_bundle.daily is None:
                target_day += weight_map * raw_day
                continue
            params = station_bundle.daily
            mapped = apply_qm(
                raw_day.ravel(),
                np.asarray(params["q_sat"], dtype=np.float64),
                np.asarray(params["q_station"], dtype=np.float64),
                float(params["slope"]),
                float(params["p_th"]),
            )
            mapped = np.maximum(np.asarray(mapped, dtype=np.float64).reshape(raw_day.shape), 0.0)
            target_day += weight_map * mapped

        pos = calib_day > 0.0
        factor = np.ones_like(calib_day, dtype=np.float64)
        needed = np.ones_like(calib_day, dtype=np.float64)
        needed[pos] = target_day[pos] / calib_day[pos]
        miss = (~pos) & (target_day > 0.0) & (raw_day > 0.0)
        needed[miss] = target_day[miss] / raw_day[miss]

        mismatch = np.abs(needed - 1.0)
        active = mismatch > 0.15
        factor[active] = 1.0 + 0.45 * (needed[active] - 1.0)
        factor = np.clip(factor, 0.80, 1.20)
        out[mask_idx] *= factor

    return out


def apply_weighted_annual_transfer(
    calib_mm: np.ndarray,
    raw_mm: np.ndarray,
    weight_maps: Mapping[int, np.ndarray],
    year_models: Mapping[int, StationModels],
) -> np.ndarray:
    out = np.asarray(calib_mm, dtype=np.float64).copy()
    raw_annual = raw_mm.sum(axis=0)
    calib_annual = out.sum(axis=0)
    target_annual = np.zeros_like(raw_annual, dtype=np.float64)
    gain_field = np.zeros_like(raw_annual, dtype=np.float64)

    for wmo, weight_map in weight_maps.items():
        station_bundle = year_models.get(int(wmo))
        if station_bundle is None or station_bundle.annual_transfer is None:
            target_annual += weight_map * raw_annual
            gain_field += weight_map * 0.25
            continue

        model = station_bundle.annual_transfer
        target = _map_annual_raw_to_target(raw_annual.ravel(), model).reshape(raw_annual.shape)
        target_annual += weight_map * target

        spread = max(float(model["raw_p90"]) - float(model["raw_p10"]), 1.0)
        z = np.abs(raw_annual - float(model["raw_p50"])) / spread
        gain = 0.25 + 0.35 * np.clip(z, 0.0, 2.0) / 2.0
        gain_field += weight_map * gain

    pos = calib_annual > 0.0
    factor = np.ones_like(calib_annual, dtype=np.float64)
    needed = np.ones_like(calib_annual, dtype=np.float64)
    needed[pos] = target_annual[pos] / calib_annual[pos]
    miss = (~pos) & (target_annual > 0.0) & (raw_annual > 0.0)
    needed[miss] = target_annual[miss] / raw_annual[miss]

    mismatch = np.abs(needed - 1.0)
    active = mismatch > 0.08
    factor[active] = 1.0 + gain_field[active] * (needed[active] - 1.0)
    factor = np.clip(factor, 0.75, 1.30)
    out *= factor
    return out


def apply_weighted_annual_sanity(
    calib_mm: np.ndarray,
    raw_mm: np.ndarray,
    weight_maps: Mapping[int, np.ndarray],
    year_models: Mapping[int, StationModels],
) -> np.ndarray:
    out = np.asarray(calib_mm, dtype=np.float64).copy()
    raw_annual = raw_mm.sum(axis=0)
    calib_annual = out.sum(axis=0)

    low_ratio = np.zeros_like(raw_annual, dtype=np.float64)
    high_ratio = np.zeros_like(raw_annual, dtype=np.float64)
    abs_upper = np.zeros_like(raw_annual, dtype=np.float64)

    for wmo, weight_map in weight_maps.items():
        station_bundle = year_models.get(int(wmo))
        env = None if station_bundle is None else station_bundle.annual_envelope
        if env is None:
            low_ratio += weight_map * 0.7
            high_ratio += weight_map * 1.3
            abs_upper += weight_map * np.nanpercentile(raw_annual, 99)
            continue

        low_ratio += weight_map * max(0.0, float(env["ratio_p10"]) * 0.9)
        high_ratio += weight_map * max(float(env["ratio_p10"]) * 0.9 + 1e-6, float(env["ratio_p90"]) * 1.1)
        abs_upper += weight_map * (float(env["station_p90"]) * 1.2)

    lower = raw_annual * low_ratio
    upper = np.minimum(raw_annual * high_ratio, abs_upper)

    needed = np.ones_like(calib_annual, dtype=np.float64)
    over = calib_annual > upper
    under = (calib_annual < lower) & (calib_annual > 0.0)
    needed[over] = upper[over] / calib_annual[over]
    needed[under] = lower[under] / calib_annual[under]

    factor = np.ones_like(calib_annual, dtype=np.float64)
    mismatch = np.abs(needed - 1.0)
    active = mismatch > 0.12
    factor[active] = 1.0 + 0.45 * (needed[active] - 1.0)
    factor = np.clip(factor, 0.85, 1.15)
    out *= factor
    return out


def load_biomet_series(csv_path: str, utc_offset_hours: float = 5.0) -> pd.Series:
    df = pd.read_csv(csv_path, skiprows=[1])
    df["dt"] = pd.to_datetime(df["TIMESTAMP_1"], format="%Y-%m-%d %H%M", errors="coerce")
    if utc_offset_hours:
        # Campbell/Biomet exports are logged in local station time; convert to UTC first.
        df["dt"] = df["dt"] - pd.to_timedelta(float(utc_offset_hours), unit="h")
    rain_mm = df["P_RAIN_1_1_1"].replace(-9999, 0.0).astype(np.float64) * 1000.0
    temp_c = df["TA_1_1_1"].replace(-9999, np.nan).astype(np.float64) - 273.15
    rain_mm[temp_c < 2.0] = 0.0
    return pd.Series(rain_mm.values, index=df["dt"]).resample("30min").sum().fillna(0.0)


def load_aws_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["datetime_utc"])
    df = df.set_index("datetime_utc")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df["P_smooth"] = df["Pluvio2_1.value1"].rolling("10min").median()
    df["P_1min"] = df["P_smooth"].diff().fillna(0.0)
    df.loc[df["P_1min"] < 0.01, "P_1min"] = 0.0
    df.loc[df["P_smooth"].diff() < -2.0, "P_1min"] = 0.0
    valid_month = df.index.month.isin([4, 5, 6, 7, 8, 9, 10])
    valid_temp = df["HMP155.T"] > 0.0 if "HMP155.T" in df.columns else pd.Series(True, index=df.index)
    df.loc[~(valid_month & valid_temp), "P_1min"] = 0.0
    return df["P_1min"].resample("30min").sum().fillna(0.0)


def extract_pixel_series(
    tif_paths_by_year: Mapping[int, Sequence[str]],
    lon: float,
    lat: float,
    years: Optional[Iterable[int]] = None,
    dt_hours: float = 0.5,
) -> pd.Series:
    parts: List[pd.Series] = []
    target_years = sorted(tif_paths_by_year) if years is None else sorted({int(y) for y in years if int(y) in tif_paths_by_year})

    for year in target_years:
        for path in sorted(tif_paths_by_year[year]):
            with rasterio.open(path) as src:
                row, col = src.index(lon, lat)
                dts = parse_band_datetimes(src.tags().get("long_name", ""))
                if len(dts) != src.count:
                    desc_blob = " ".join([d for d in src.descriptions if d])
                    dts = parse_band_datetimes(desc_blob)
                vals = src.read(window=((row, row + 1), (col, col + 1))).astype(np.float64)[:, 0, 0]
            mm = np.maximum(vals, 0.0) * dt_hours
            parts.append(pd.Series(mm, index=pd.to_datetime(dts)))

    if not parts:
        return pd.Series(dtype=np.float64)
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="first")]


def _windowize_3h(series_30m: pd.Series) -> pd.DataFrame:
    if series_30m.empty:
        return pd.DataFrame()

    series = series_30m.resample("30min").sum().fillna(0.0).sort_index()
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_convert("UTC").tz_localize(None)
    start = series.index.min().floor("3h")
    end = series.index.max().ceil("3h")
    full_idx = pd.date_range(start, end, freq="30min")
    series = series.reindex(full_idx, fill_value=0.0)

    anchors = pd.date_range(full_idx.min().floor("3h"), full_idx.max().ceil("3h"), freq="3h")
    rows: List[Dict[str, object]] = []
    for ts in anchors:
        slots = series.reindex(pd.date_range(ts, periods=6, freq="30min"), fill_value=0.0).to_numpy(dtype=np.float64)
        rows.append(
            {
                "window_start": ts,
                "slots": slots,
                "total_mm": float(np.sum(slots)),
            }
        )
    return pd.DataFrame(rows)


def _sharpen_weights(weights: np.ndarray, gamma: np.ndarray | float) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 1e-9)
    sharp = w ** gamma
    sharp_sum = np.sum(sharp, axis=0, keepdims=True)
    sharp_sum = np.where(sharp_sum > 0.0, sharp_sum, 1.0)
    return sharp / sharp_sum


def _solve_gamma_star(raw_weights: np.ndarray, target_peak_share: float) -> float:
    gamma_grid = np.linspace(PEAK_GAMMA_MIN, PEAK_GAMMA_MAX, 57)
    best_gamma = 1.0
    best_err = 1e18
    for gamma in gamma_grid:
        sharpened = _sharpen_weights(raw_weights[:, np.newaxis], gamma).ravel()
        err = abs(float(np.max(sharpened)) - target_peak_share)
        if err < best_err:
            best_err = err
            best_gamma = float(gamma)
    return best_gamma


def fit_peak_model(
    raw_site_series: Mapping[str, pd.Series],
    ground_site_series: Mapping[str, pd.Series],
    min_ground_total_mm: float = 5.0,
) -> Dict[str, object]:
    rows: List[Dict[str, float | str]] = []
    templates: Dict[str, List[np.ndarray]] = {season: [] for season in SEASONS}

    for site_name, raw_series in raw_site_series.items():
        ground_series = ground_site_series.get(site_name)
        if ground_series is None or raw_series.empty or ground_series.empty:
            continue

        raw_df = _windowize_3h(raw_series)
        ground_df = _windowize_3h(ground_series)
        merged = raw_df.merge(ground_df, on="window_start", suffixes=("_raw", "_ground"))

        for _, row in merged.iterrows():
            raw_slots = np.asarray(row["slots_raw"], dtype=np.float64)
            ground_slots = np.asarray(row["slots_ground"], dtype=np.float64)
            raw_total = float(np.sum(raw_slots))
            ground_total = float(np.sum(ground_slots))
            if raw_total <= 0.0 or ground_total < min_ground_total_mm:
                continue

            raw_weights = raw_slots / raw_total
            ground_peak_share = float(np.max(ground_slots) / ground_total)
            gamma_star = _solve_gamma_star(raw_weights, ground_peak_share)
            season = get_season(int(row["window_start"].month))
            templates[season].append(ground_slots / ground_total)

            rows.append(
                {
                    "site": site_name,
                    "season": season,
                    "ground_total_mm": ground_total,
                    "raw_peak_share": float(np.max(raw_weights)),
                    "wet_slots": float(np.sum(raw_slots > 0.05)),
                    "gamma_star": gamma_star,
                }
            )

    if not rows:
        return {
            "beta": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "templates": {season: [1.0 / 6.0] * 6 for season in SEASONS},
            "gamma_min": PEAK_GAMMA_MIN,
            "gamma_max": PEAK_GAMMA_MAX,
            "n_samples": 0,
            "training_table": pd.DataFrame(),
        }

    train_df = pd.DataFrame(rows)
    x = np.column_stack(
        [
            np.ones(len(train_df), dtype=np.float64),
            np.log1p(train_df["ground_total_mm"].to_numpy(dtype=np.float64)),
            train_df["raw_peak_share"].to_numpy(dtype=np.float64),
            train_df["wet_slots"].to_numpy(dtype=np.float64),
            (train_df["season"] == "MAM").to_numpy(dtype=np.float64),
            (train_df["season"] == "JJA").to_numpy(dtype=np.float64),
            (train_df["season"] == "SON").to_numpy(dtype=np.float64),
        ]
    )
    y = train_df["gamma_star"].to_numpy(dtype=np.float64)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)

    template_dict: Dict[str, List[float]] = {}
    for season in SEASONS:
        if templates[season]:
            template = np.mean(np.stack(templates[season], axis=0), axis=0)
            template = template / np.sum(template)
        else:
            template = np.full(6, 1.0 / 6.0, dtype=np.float64)
        template_dict[season] = template.astype(float).tolist()

    return {
        "beta": beta.astype(float).tolist(),
        "templates": template_dict,
        "gamma_min": PEAK_GAMMA_MIN,
        "gamma_max": PEAK_GAMMA_MAX,
        "n_samples": int(len(train_df)),
        "training_table": train_df,
    }


def predict_peak_gamma(
    corrected_total_mm: np.ndarray,
    raw_peak_share: np.ndarray,
    wet_slots: np.ndarray,
    season: str,
    peak_model: Mapping[str, object],
) -> np.ndarray:
    beta = np.asarray(peak_model["beta"], dtype=np.float64)
    season_flags = {
        "DJF": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "MAM": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "JJA": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "SON": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }[season]

    x = np.vstack(
        [
            np.ones_like(corrected_total_mm, dtype=np.float64),
            np.log1p(np.asarray(corrected_total_mm, dtype=np.float64)),
            np.asarray(raw_peak_share, dtype=np.float64),
            np.asarray(wet_slots, dtype=np.float64),
            np.full_like(corrected_total_mm, season_flags[0], dtype=np.float64),
            np.full_like(corrected_total_mm, season_flags[1], dtype=np.float64),
            np.full_like(corrected_total_mm, season_flags[2], dtype=np.float64),
        ]
    ).T
    gamma = x @ beta
    gamma = np.where(corrected_total_mm < 2.0, 1.0, gamma)
    return np.clip(gamma, float(peak_model["gamma_min"]), float(peak_model["gamma_max"]))


def redistribute_window_with_peaks(
    raw_window_mm: np.ndarray,
    corrected_total_mm: np.ndarray,
    month: int,
    peak_model: Optional[Mapping[str, object]],
) -> np.ndarray:
    out = np.zeros_like(raw_window_mm, dtype=np.float64)
    season = get_season(int(month))
    if peak_model is None:
        raw_sum = raw_window_mm.sum(axis=0)
        pos = (raw_sum > 0.0) & (corrected_total_mm > 0.0)
        if np.any(pos):
            out[:, pos] = corrected_total_mm[pos] * (raw_window_mm[:, pos] / raw_sum[pos])
        return out

    raw_sum = raw_window_mm.sum(axis=0)
    pos = corrected_total_mm > 0.0
    with_raw = pos & (raw_sum > 0.0)
    without_raw = pos & (raw_sum <= 0.0)

    if np.any(with_raw):
        weights = raw_window_mm[:, with_raw] / raw_sum[with_raw]
        raw_peak_share = np.max(weights, axis=0)
        wet_slots = np.sum(raw_window_mm[:, with_raw] > 0.05, axis=0)
        gamma = predict_peak_gamma(corrected_total_mm[with_raw], raw_peak_share, wet_slots, season, peak_model)
        sharp = _sharpen_weights(weights, gamma[np.newaxis, :])
        out[:, with_raw] = corrected_total_mm[with_raw] * sharp

    if np.any(without_raw):
        template = np.asarray(peak_model["templates"].get(season, [1.0 / 6.0] * raw_window_mm.shape[0]), dtype=np.float64)
        template = template / np.sum(template)
        out[:, without_raw] = corrected_total_mm[without_raw] * template[:, np.newaxis]

    return out


def apply_station_models_to_target_year(
    station_df: pd.DataFrame,
    target_year: int,
    station_models: StationModels,
) -> pd.DataFrame:
    year_df = station_df[station_df["year"] == target_year].copy()
    if year_df.empty:
        return year_df

    qm = _apply_seasonal_models(year_df, station_models.seasonal)
    year_df["P_v6_mm"] = np.maximum(
        year_df["P_sat_mm"].to_numpy(dtype=np.float64)
        + station_models.blend_alpha * (qm - year_df["P_sat_mm"].to_numpy(dtype=np.float64)),
        0.0,
    )

    if station_models.daily is not None:
        tmp = year_df.assign(date=year_df["datetime"].dt.floor("D"))
        daily = tmp.groupby("date", as_index=False)[["P_sat_mm", "P_v6_mm"]].sum()
        target = apply_qm(
            daily["P_sat_mm"].to_numpy(dtype=np.float64),
            np.asarray(station_models.daily["q_sat"], dtype=np.float64),
            np.asarray(station_models.daily["q_station"], dtype=np.float64),
            float(station_models.daily["slope"]),
            float(station_models.daily["p_th"]),
        )
        daily["target"] = np.maximum(np.asarray(target, dtype=np.float64), 0.0)
        scale = np.ones(len(daily), dtype=np.float64)
        pos = daily["P_v6_mm"].to_numpy(dtype=np.float64) > 0.0
        needed = np.ones(len(daily), dtype=np.float64)
        needed[pos] = daily.loc[pos, "target"].to_numpy(dtype=np.float64) / daily.loc[pos, "P_v6_mm"].to_numpy(dtype=np.float64)
        mismatch = np.abs(needed - 1.0)
        active = mismatch > 0.15
        scale[active] = 1.0 + 0.45 * (needed[active] - 1.0)
        scale = np.clip(scale, 0.80, 1.20)
        scale_map = dict(zip(daily["date"], scale))
        year_df["P_v6_mm"] *= year_df["datetime"].dt.floor("D").map(scale_map).astype(np.float64)

    annual_total = float(year_df["P_v6_mm"].sum())
    raw_total = float(year_df["P_sat_mm"].sum())
    if station_models.annual_transfer is not None and annual_total > 0.0 and raw_total > 0.0:
        target_total = float(_map_annual_raw_to_target(np.array([raw_total], dtype=np.float64), station_models.annual_transfer)[0])
        needed = target_total / annual_total
        factor = float(np.clip(1.0 + 0.50 * (needed - 1.0), 0.75, 1.30))
        year_df["P_v6_mm"] *= factor

    if station_models.annual_envelope is not None:
        raw_total = float(year_df["P_sat_mm"].sum())
        corr_total = float(year_df["P_v6_mm"].sum())
        low = max(0.0, station_models.annual_envelope["ratio_p10"] * 0.9) * raw_total
        high = min(
            max(low + 1e-6, station_models.annual_envelope["ratio_p90"] * 1.1 * raw_total),
            station_models.annual_envelope["station_p90"] * 1.2,
        )
        if corr_total > 0.0:
            if corr_total < low:
                factor = low / corr_total
            elif corr_total > high:
                factor = high / corr_total
            else:
                factor = 1.0
            factor = float(np.clip(1.0 + 0.45 * (factor - 1.0), 0.85, 1.15))
            year_df["P_v6_mm"] *= factor

    return year_df
