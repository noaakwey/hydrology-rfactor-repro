#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
r_factor_rusle2.py  -  v8 (parameterised energy + ensemble support)
===================================================================
Universal R-factor calculator (RUSLE2-like workflow) for raster
precipitation stacks.

This is the canonical, fully-parameterised implementation. The previous
in-repo copy hard-coded k=0.05 in `unit_energy`; that variant produced
results inconsistent with the supplementary k=0.082 archive that was
in fact computed by an external copy of this file.  The two paths are
now consolidated into a single CLI:

    --energy_model {bf, vd, ws, salles, sm}
    --exp_k FLOAT   (used only with --energy_model bf)

Default --energy_model bf --exp_k 0.05 reproduces the legacy classical
result.  --energy_model bf --exp_k 0.082 reproduces RUSLE2 / Foster (2003).
The other models exist for ensemble uncertainty analysis.

The full RUSLE2-like event logic (event split, pending-buffer for weak
steps, year-end flush) is unchanged from v7. See `lib/energy_models.py`
for the energy parameterisations.

Main features
-------------
1) Datasets:
   --dataset imerg  : 30-min precipitation intensity stacks (mm/h)
   --dataset era5   : 1-hour precipitation intensity stacks (mm/h)

2) Input organisation:
   - precip can be:  directory / glob / single file
   - masks: quarterly (..._YYYY_Q1..Q4.tif) or single file (fallback)

3) Pending-buffer logic for weak steps (i < gap_intensity):
   - weak steps accumulated in pending_*
   - committed to event when "gap" proven wet (dry_sum >= split_sum_mm)
   - discarded when "gap" proven dry  (dry_sum <  split_sum_mm)

4) Output annual R raster, units: MJ mm ha-1 h-1 yr-1

Note on the 30-min depth criterion (RUSLE2):
    For dt = 0.5 h, the condition `i_step >= erosive_peak_mm_h (=25.4)`
    is exactly equivalent to "12.7 mm in 30 min", so the 30-min
    erosivity criterion of RUSLE2 is satisfied without an extra check.
"""

from __future__ import annotations

import os
import re
import glob
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from numba import njit, prange
from tqdm import tqdm

from erosivity.lib.energy_models import (
    unit_energy as _unit_energy_dispatch,
    parse_model,
    ENERGY_BROWN_FOSTER,
)


# =============================================================================
# Specifications
# =============================================================================

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dt_hours: float
    mask_step_hours: float
    cf_to_ei30: float

    def describe(self) -> str:
        true_i30 = self.dt_hours <= 0.5
        lines = [
            f"Dataset : {self.name}",
            f"  dt    : {self.dt_hours} h  "
            f"({'bin-aligned 30-min I30 (sub-bin convective peaks may be smeared)' if true_i30 else 'I1h proxy, calibrate cf_to_ei30'})",
            f"  mask  : {self.mask_step_hours} h",
            f"  CF    : {self.cf_to_ei30}",
        ]
        if (not true_i30) and self.cf_to_ei30 == 1.0:
            lines.append(
                "  WARNING: ERA5 dt=1h => I1h != I30. Use --cf_to_ei30 (local calibration recommended)."
            )
        return "\n".join(lines)


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "imerg": DatasetSpec("imerg", dt_hours=0.5, mask_step_hours=1.0, cf_to_ei30=1.0),
    "era5":  DatasetSpec("era5",  dt_hours=1.0, mask_step_hours=1.0, cf_to_ei30=1.0),
}


@dataclass(frozen=True)
class RConfig:
    # Inter-event gap: cumulative depth < event_split_sum_mm over event_split_hours
    event_split_hours: float = 6.0
    event_split_sum_mm: float = 1.27

    # Classification boundary for weak steps (pending buffer)
    gap_intensity_mm_h: float = 1.27

    # Erosive event thresholds
    erosive_depth_mm: float = 12.7
    erosive_peak_mm_h: float = 25.4   # at dt=0.5h equivalent to 12.7 mm in 30 min

    # Data gap handling
    gap_close_event: bool = True

    # Optional intensity cap
    use_intensity_cap: bool = False
    intensity_cap_mm_h: float = 300.0

    # Output
    compress: str = "deflate"
    predictor: int = 3
    bigtiff: str = "IF_SAFER"


# =============================================================================
# Numba kernel
# =============================================================================

@njit(cache=True, fastmath=True)
def unit_energy(i_mm_h: float, model_id: int, exp_k: float) -> float:
    """Inline shim that forwards to the central dispatcher in lib.energy_models."""
    return _unit_energy_dispatch(i_mm_h, model_id, exp_k)


@njit(cache=True, parallel=True, fastmath=True)
def process_step(
    i_band: np.ndarray,           # (H,W) float32  mm/h
    valid: np.ndarray,            # (H,W) uint8    1=data present
    liquid: np.ndarray,           # (H,W) uint8    1=liquid
    dt_hours: float,
    split_steps: int,
    split_sum_mm: float,
    gap_intensity: float,
    erosive_depth: float,
    erosive_peak: float,
    use_cap: bool,
    cap_mm_h: float,
    gap_close: bool,
    energy_model: int,
    exp_k: float,
    # Event state
    annual_R: np.ndarray,
    in_event: np.ndarray,
    event_E: np.ndarray,
    event_Imax: np.ndarray,
    event_P: np.ndarray,
    event_has_peak: np.ndarray,
    # Gap / pending state
    dry_steps: np.ndarray,
    dry_sum: np.ndarray,
    pending_P: np.ndarray,
    pending_E: np.ndarray,
    pending_Imax: np.ndarray,
    pending_has_peak: np.ndarray,
) -> None:
    H, W = i_band.shape

    for row in prange(H):
        for col in range(W):

            # No-data / masked band
            if valid[row, col] == 0:
                if gap_close and in_event[row, col] == 1:
                    is_valid_gap_like = (
                        (dry_steps[row, col] >= split_steps) and
                        (dry_sum[row, col] < split_sum_mm)
                    )

                    # Flush pending unless current weak window already qualifies as gap
                    if (pending_P[row, col] > 0.0) and (not is_valid_gap_like):
                        event_P[row, col] += pending_P[row, col]
                        event_E[row, col] += pending_E[row, col]
                        if pending_Imax[row, col] > event_Imax[row, col]:
                            event_Imax[row, col] = pending_Imax[row, col]
                        if pending_has_peak[row, col] == 1:
                            event_has_peak[row, col] = 1

                    erosive = (
                        (event_P[row, col] >= erosive_depth) or
                        (event_has_peak[row, col] == 1)
                    )
                    if erosive:
                        annual_R[row, col] += event_E[row, col] * event_Imax[row, col]

                    in_event[row, col]       = 0
                    event_E[row, col]        = 0.0
                    event_Imax[row, col]     = 0.0
                    event_P[row, col]        = 0.0
                    event_has_peak[row, col] = 0

                dry_steps[row, col]        = 0
                dry_sum[row, col]          = 0.0
                pending_P[row, col]        = 0.0
                pending_E[row, col]        = 0.0
                pending_Imax[row, col]     = 0.0
                pending_has_peak[row, col] = 0
                continue

            # Clip + phase mask
            i = i_band[row, col]
            if (not np.isfinite(i)) or (i < 0.0):
                i = 0.0
            if use_cap and i > cap_mm_h:
                i = cap_mm_h
            if liquid[row, col] == 0:
                i = 0.0

            p = i * dt_hours  # mm per step

            # Weak step
            if i < gap_intensity:
                # Event may start on weak liquid rain
                if in_event[row, col] == 0 and p > 0.0:
                    in_event[row, col]       = 1
                    event_E[row, col]        = 0.0
                    event_Imax[row, col]     = 0.0
                    event_P[row, col]        = 0.0
                    event_has_peak[row, col] = 0

                dry_steps[row, col] += 1
                dry_sum[row, col]   += p

                if p > 0.0:
                    e = unit_energy(i, energy_model, exp_k)
                    pending_P[row, col] += p
                    pending_E[row, col] += e * p
                    if i > pending_Imax[row, col]:
                        pending_Imax[row, col] = i
                    if i >= erosive_peak:
                        pending_has_peak[row, col] = 1

                if in_event[row, col] == 1 and dry_steps[row, col] >= split_steps:
                    if dry_sum[row, col] < split_sum_mm:
                        # Valid gap: close event, discard pending
                        erosive = (
                            (event_P[row, col] >= erosive_depth) or
                            (event_has_peak[row, col] == 1)
                        )
                        if erosive:
                            annual_R[row, col] += event_E[row, col] * event_Imax[row, col]

                        in_event[row, col]       = 0
                        event_E[row, col]        = 0.0
                        event_Imax[row, col]     = 0.0
                        event_P[row, col]        = 0.0
                        event_has_peak[row, col] = 0
                    else:
                        # Wet gap: commit pending into event
                        if pending_P[row, col] > 0.0:
                            event_P[row, col] += pending_P[row, col]
                            event_E[row, col] += pending_E[row, col]
                            if pending_Imax[row, col] > event_Imax[row, col]:
                                event_Imax[row, col] = pending_Imax[row, col]
                            if pending_has_peak[row, col] == 1:
                                event_has_peak[row, col] = 1

                    dry_steps[row, col]        = 0
                    dry_sum[row, col]          = 0.0
                    pending_P[row, col]        = 0.0
                    pending_E[row, col]        = 0.0
                    pending_Imax[row, col]     = 0.0
                    pending_has_peak[row, col] = 0

                continue

            # Significant step
            if in_event[row, col] == 0:
                in_event[row, col]       = 1
                event_E[row, col]        = 0.0
                event_Imax[row, col]     = 0.0
                event_P[row, col]        = 0.0
                event_has_peak[row, col] = 0

            # Flush pending
            if pending_P[row, col] > 0.0:
                event_P[row, col] += pending_P[row, col]
                event_E[row, col] += pending_E[row, col]
                if pending_Imax[row, col] > event_Imax[row, col]:
                    event_Imax[row, col] = pending_Imax[row, col]
                if pending_has_peak[row, col] == 1:
                    event_has_peak[row, col] = 1

                pending_P[row, col]        = 0.0
                pending_E[row, col]        = 0.0
                pending_Imax[row, col]     = 0.0
                pending_has_peak[row, col] = 0

            # Reset gap counters
            dry_steps[row, col] = 0
            dry_sum[row, col]   = 0.0

            # Add current step
            e = unit_energy(i, energy_model, exp_k)
            event_E[row, col] += e * p
            event_P[row, col] += p
            if i > event_Imax[row, col]:
                event_Imax[row, col] = i
            if i >= erosive_peak:
                event_has_peak[row, col] = 1


# =============================================================================
# Path / filename helpers
# =============================================================================

_YEAR_RE = re.compile(r"(?:^|_)(19\d{2}|20\d{2})(?:_|$)")
_QUARTER_RE = re.compile(r"_Q([1-4])(?:\D|$)", re.IGNORECASE)


def parse_year(path: str) -> Optional[int]:
    m = _YEAR_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_quarter(path: str) -> Optional[int]:
    m = _QUARTER_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def list_tifs_from_input(path_or_glob: str) -> List[str]:
    p = path_or_glob.strip().strip('"').strip("'")
    if os.path.isdir(p):
        files = []
        files.extend(glob.glob(os.path.join(p, "*.tif")))
        files.extend(glob.glob(os.path.join(p, "*.tiff")))
        return sorted(set(files))
    if os.path.isfile(p):
        return [p]
    files = glob.glob(p)
    return sorted(files)


def group_files_by_year(paths: List[str]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for p in paths:
        y = parse_year(p)
        if y is not None:
            out.setdefault(y, []).append(p)
    return {y: sorted(v) for y, v in out.items()}


def group_mask_quarters_by_year(paths: List[str]) -> Dict[int, Dict[int, str]]:
    out: Dict[int, Dict[int, str]] = {}
    for p in sorted(paths):
        y = parse_year(p)
        q = parse_quarter(p)
        if y is None or q is None:
            continue
        out.setdefault(y, {})
        if q in out[y]:
            warnings.warn(f"Duplicate mask for year={y}, Q{q}; using later file:\n  {p}")
        out[y][q] = p
    return out


def build_year_mask_sequence(mask_q_by_year: Dict[int, Dict[int, str]], year: int) -> List[str]:
    qmap = mask_q_by_year.get(year, {})
    return [qmap[q] for q in (1, 2, 3, 4) if q in qmap]


def quarter_band_counts_for_precip_year(pds: rasterio.DatasetReader, spec: DatasetSpec, year: int) -> List[int]:
    leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    q_days = [sum(days[0:3]), sum(days[3:6]), sum(days[6:9]), sum(days[9:12])]

    steps_per_day = int(round(24.0 / spec.dt_hours))
    q_counts = [d * steps_per_day for d in q_days]

    total_expected = sum(q_counts)
    if pds.count != total_expected:
        warnings.warn(
            f"Precip annual band count for {year} is {pds.count}, expected {total_expected}. "
            "Will slice quarterly masks greedily; verify temporal completeness."
        )
        rem = pds.count
        out = []
        for qc in q_counts:
            take = max(0, min(qc, rem))
            out.append(take)
            rem -= take
        return out

    return q_counts


def quarter_slices_from_counts(q_counts: List[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    s = 1
    for n in q_counts:
        e = s + n - 1
        out.append((s, e))
        s = e + 1
    return out


# =============================================================================
# Raster IO helpers (Numba-safe)
# =============================================================================

def read_band_masked(
    ds: rasterio.DatasetReader, b: int, win: Window
) -> Tuple[np.ndarray, np.ndarray]:
    arr_ma = ds.read(b, window=win, out_dtype="float32", masked=True)
    if isinstance(arr_ma, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(arr_ma)
        valid = (~mask).astype(np.uint8)
        data = np.asarray(arr_ma.filled(0.0), dtype=np.float32)
    else:
        data = np.asarray(arr_ma, dtype=np.float32)
        valid = np.ones(data.shape, dtype=np.uint8)
    return data, valid


def read_mask_band(
    ds: rasterio.DatasetReader, b: int, win: Window
) -> np.ndarray:
    m = ds.read(b, window=win, out_dtype="uint8", resampling=Resampling.nearest)
    m = np.asarray(m)
    return (m > 0).astype(np.uint8)


def mask_band_for_step(step_1based: int, spec: DatasetSpec) -> int:
    steps_per_mask = max(1, int(round(spec.mask_step_hours / spec.dt_hours)))
    return ((step_1based - 1) // steps_per_mask) + 1


# =============================================================================
# Fast in-memory annual R-factor (pre-loads full stack for small domains)
# =============================================================================

def compute_R_year_preloaded(
    precip_year_file: str,
    mask_quarters: List[str],
    out_path: str,
    year: int,
    spec: DatasetSpec,
    cfg: RConfig,
    energy_model: int = ENERGY_BROWN_FOSTER,
    exp_k: float = 0.05,
) -> None:
    """
    Faster version of compute_R_year_annual_precip_and_quarter_masks that
    loads the entire annual precipitation stack and all mask stacks into RAM
    before the processing loop.  For domains <= ~500 MB this is 10-30x faster
    than band-by-band rasterio reads.

    Use this when the domain is small (< ~1000 pixels); fall back to the
    tiled band-by-band version for large rasters.
    """
    split_steps = max(1, int(round(cfg.event_split_hours / spec.dt_hours)))

    with rasterio.open(precip_year_file) as ref:
        profile = ref.profile.copy()
        profile.update(
            dtype="float32", count=1,
            compress=cfg.compress, predictor=cfg.predictor,
            bigtiff=cfg.bigtiff, nodata=None,
        )
        H, W = ref.height, ref.width
        q_counts = quarter_band_counts_for_precip_year(ref, spec, year)
        q_slices = quarter_slices_from_counts(q_counts)
        # Load entire precip stack at once — (N_bands, H, W) float32
        precip_all = ref.read().astype(np.float32)
        # Build valid mask: 1 where not nodata/nan
        nodata = ref.nodata
        if nodata is not None:
            valid_all = (precip_all != nodata).astype(np.uint8)
        else:
            valid_all = np.isfinite(precip_all).astype(np.uint8)
        precip_all = np.where(valid_all, np.clip(precip_all, 0.0, None), 0.0)

    # Load all quarterly masks at once
    mask_stacks = []
    for qm_path in mask_quarters[:4]:
        with rasterio.open(qm_path) as mds:
            m = mds.read().astype(np.uint8)
            m = (m > 0).astype(np.uint8)
            mask_stacks.append(m)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    annual_R       = np.zeros((H, W), np.float32)
    in_event       = np.zeros((H, W), np.uint8)
    event_E        = np.zeros((H, W), np.float32)
    event_Imax     = np.zeros((H, W), np.float32)
    event_P        = np.zeros((H, W), np.float32)
    event_has_peak = np.zeros((H, W), np.uint8)
    dry_steps        = np.zeros((H, W), np.uint32)
    dry_sum          = np.zeros((H, W), np.float32)
    pending_P        = np.zeros((H, W), np.float32)
    pending_E        = np.zeros((H, W), np.float32)
    pending_Imax     = np.zeros((H, W), np.float32)
    pending_has_peak = np.zeros((H, W), np.uint8)

    steps_per_mask = max(1, int(round(spec.mask_step_hours / spec.dt_hours)))

    for q_idx in range(min(len(mask_quarters), 4)):
        if q_idx >= len(mask_stacks):
            break
        q_start, q_end = q_slices[q_idx]
        if q_start > precip_all.shape[0]:
            continue
        q_end = min(q_end, precip_all.shape[0])
        if q_end < q_start:
            continue
        mstack = mask_stacks[q_idx]

        for local_step, b in enumerate(range(q_start, q_end + 1), start=1):
            i_band = precip_all[b - 1]          # (H, W) float32
            valid  = valid_all[b - 1]            # (H, W) uint8
            mb = ((local_step - 1) // steps_per_mask)
            mb = min(mb, mstack.shape[0] - 1)
            liq = mstack[mb]                     # (H, W) uint8

            process_step(
                i_band=i_band, valid=valid, liquid=liq,
                dt_hours=spec.dt_hours,
                split_steps=split_steps,
                split_sum_mm=cfg.event_split_sum_mm,
                gap_intensity=cfg.gap_intensity_mm_h,
                erosive_depth=cfg.erosive_depth_mm,
                erosive_peak=cfg.erosive_peak_mm_h,
                use_cap=cfg.use_intensity_cap,
                cap_mm_h=cfg.intensity_cap_mm_h,
                gap_close=cfg.gap_close_event,
                energy_model=energy_model, exp_k=exp_k,
                annual_R=annual_R, in_event=in_event,
                event_E=event_E, event_Imax=event_Imax,
                event_P=event_P, event_has_peak=event_has_peak,
                dry_steps=dry_steps, dry_sum=dry_sum,
                pending_P=pending_P, pending_E=pending_E,
                pending_Imax=pending_Imax, pending_has_peak=pending_has_peak,
            )

    # Year-end flush + close
    mask_flush = (in_event == 1) & (pending_P > 0.0)
    if np.any(mask_flush):
        event_P[mask_flush] += pending_P[mask_flush]
        event_E[mask_flush] += pending_E[mask_flush]
        event_Imax[mask_flush] = np.maximum(event_Imax[mask_flush], pending_Imax[mask_flush])
        event_has_peak[mask_flush] = np.maximum(event_has_peak[mask_flush], pending_has_peak[mask_flush])

    open_ev = in_event == 1
    if np.any(open_ev):
        erosive = open_ev & (
            (event_P >= cfg.erosive_depth_mm) | (event_has_peak == 1)
        )
        annual_R[erosive] += event_E[erosive] * event_Imax[erosive]

    if spec.cf_to_ei30 != 1.0:
        annual_R *= spec.cf_to_ei30

    with rasterio.open(out_path, "w", **profile) as out_ds:
        out_ds.write(annual_R, 1)

    del precip_all, valid_all, mask_stacks


# =============================================================================
# Annual R-factor computation (tiled, disk-streaming, for large rasters)
# =============================================================================

def compute_R_year_annual_precip_and_quarter_masks(
    precip_year_file: str,
    mask_quarters: List[str],
    out_path: str,
    year: int,
    spec: DatasetSpec,
    cfg: RConfig,
    energy_model: int = ENERGY_BROWN_FOSTER,
    exp_k: float = 0.05,
    tile: int = 256,
) -> None:
    split_steps = max(1, int(round(cfg.event_split_hours / spec.dt_hours)))

    with rasterio.open(precip_year_file) as ref:
        profile = ref.profile.copy()
        profile.update(
            dtype="float32", count=1,
            compress=cfg.compress, predictor=cfg.predictor,
            bigtiff=cfg.bigtiff, nodata=None,
        )
        H, W = ref.height, ref.width

        q_counts = quarter_band_counts_for_precip_year(ref, spec, year)
        q_slices = quarter_slices_from_counts(q_counts)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as out_ds:
        for r0 in tqdm(range(0, H, tile), desc=os.path.basename(out_path)):
            for c0 in range(0, W, tile):
                h = min(tile, H - r0)
                w = min(tile, W - c0)
                win = Window(c0, r0, w, h)

                annual_R       = np.zeros((h, w), np.float32)
                in_event       = np.zeros((h, w), np.uint8)
                event_E        = np.zeros((h, w), np.float32)
                event_Imax     = np.zeros((h, w), np.float32)
                event_P        = np.zeros((h, w), np.float32)
                event_has_peak = np.zeros((h, w), np.uint8)

                dry_steps        = np.zeros((h, w), np.uint32)
                dry_sum          = np.zeros((h, w), np.float32)
                pending_P        = np.zeros((h, w), np.float32)
                pending_E        = np.zeros((h, w), np.float32)
                pending_Imax     = np.zeros((h, w), np.float32)
                pending_has_peak = np.zeros((h, w), np.uint8)

                with rasterio.open(precip_year_file) as pds:
                    for q_idx, qm in enumerate(mask_quarters, start=1):
                        if q_idx > 4:
                            break
                        q_start, q_end = q_slices[q_idx - 1]
                        if q_start > pds.count:
                            continue
                        q_end = min(q_end, pds.count)
                        if q_end < q_start:
                            continue

                        with rasterio.open(qm) as mds:
                            if (pds.transform != mds.transform or
                                pds.width  != mds.width or
                                pds.height != mds.height):
                                raise ValueError(
                                    f"Grid mismatch:\n  precip: {precip_year_file}\n  mask:   {qm}"
                                )
                            local_step = 0
                            for b in range(q_start, q_end + 1):
                                local_step += 1

                                i_band, valid = read_band_masked(pds, b, win)

                                mb = mask_band_for_step(local_step, spec)
                                mb = max(1, min(mb, mds.count))
                                liq = read_mask_band(mds, mb, win)

                                process_step(
                                    i_band=i_band,
                                    valid=valid,
                                    liquid=liq,
                                    dt_hours=spec.dt_hours,
                                    split_steps=split_steps,
                                    split_sum_mm=cfg.event_split_sum_mm,
                                    gap_intensity=cfg.gap_intensity_mm_h,
                                    erosive_depth=cfg.erosive_depth_mm,
                                    erosive_peak=cfg.erosive_peak_mm_h,
                                    use_cap=cfg.use_intensity_cap,
                                    cap_mm_h=cfg.intensity_cap_mm_h,
                                    gap_close=cfg.gap_close_event,
                                    energy_model=energy_model,
                                    exp_k=exp_k,
                                    annual_R=annual_R,
                                    in_event=in_event,
                                    event_E=event_E,
                                    event_Imax=event_Imax,
                                    event_P=event_P,
                                    event_has_peak=event_has_peak,
                                    dry_steps=dry_steps,
                                    dry_sum=dry_sum,
                                    pending_P=pending_P,
                                    pending_E=pending_E,
                                    pending_Imax=pending_Imax,
                                    pending_has_peak=pending_has_peak,
                                )

                # Year-end: flush pending into event before scoring
                mask_flush = (in_event == 1) & (pending_P > 0.0)
                if np.any(mask_flush):
                    event_P[mask_flush] += pending_P[mask_flush]
                    event_E[mask_flush] += pending_E[mask_flush]
                    event_Imax[mask_flush] = np.maximum(event_Imax[mask_flush], pending_Imax[mask_flush])
                    event_has_peak[mask_flush] = np.maximum(event_has_peak[mask_flush], pending_has_peak[mask_flush])
                    pending_P[mask_flush] = 0.0
                    pending_E[mask_flush] = 0.0
                    pending_Imax[mask_flush] = 0.0
                    pending_has_peak[mask_flush] = 0

                open_ev = in_event == 1
                if np.any(open_ev):
                    erosive = open_ev & (
                        (event_P >= cfg.erosive_depth_mm) |
                        (event_has_peak == 1)
                    )
                    annual_R[erosive] += event_E[erosive] * event_Imax[erosive]

                if spec.cf_to_ei30 != 1.0:
                    annual_R *= spec.cf_to_ei30

                out_ds.write(annual_R, 1, window=win)

    energy_label_map = {
        0: f"BrownFoster_k{exp_k}",
        1: "vanDijk_2002",
        2: "WischmeierSmith_1958",
        3: "Salles_2002",
        4: "SanchezMoreno_2014",
    }
    formula_map = {
        0: f"e=0.29*(1-0.72*exp(-{exp_k}*i))",
        1: "e=0.283*(1-0.52*exp(-0.042*i))",
        2: "e=0.119+0.0873*log10(i), saturated at i>=76 mm/h",
        3: "e=0.359*(1-0.56*exp(-0.034*i))",
        4: "e=0.353*(1-0.65*exp(-0.044*i))",
    }
    with rasterio.open(out_path, "r+") as ds:
        ds.update_tags(
            units="MJ mm ha-1 h-1 yr-1",
            dataset=spec.name,
            dt_hours=str(spec.dt_hours),
            cf_to_ei30=str(spec.cf_to_ei30),
            event_split_hours=str(cfg.event_split_hours),
            event_split_sum_mm=str(cfg.event_split_sum_mm),
            gap_intensity_mm_h=str(cfg.gap_intensity_mm_h),
            erosive_depth_mm=str(cfg.erosive_depth_mm),
            erosive_peak_mm_h=str(cfg.erosive_peak_mm_h),
            precip_layout="annual_stack_per_year",
            mask_layout="quarterly_stacks_per_year",
            pending_note=(
                "Pending buffer stores weak steps until gap criterion is resolved; "
                "flushed into event for wet gaps and at year-end; discarded for valid gaps"
            ),
            energy_model=energy_label_map.get(energy_model, str(energy_model)),
            energy_formula=formula_map.get(energy_model, "unknown"),
            methodology="RUSLE2-like event logic with pending weak-step buffer",
        )


# =============================================================================
# Multi-year mean
# =============================================================================

def compute_period_mean(
    annual_paths: List[str],
    out_path: str,
    tile: int = 256,
) -> None:
    with rasterio.open(annual_paths[0]) as ref:
        profile = ref.profile.copy()
        profile.update(dtype="float32", count=1, nodata=None)
        H, W = ref.height, ref.width
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_wh = (ref.width, ref.height)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as out_ds:
        for r0 in tqdm(range(0, H, tile), desc="Period mean"):
            for c0 in range(0, W, tile):
                h = min(tile, H - r0)
                w = min(tile, W - c0)
                win = Window(c0, r0, w, h)

                acc = np.zeros((h, w), np.float64)
                n = 0

                for ap in annual_paths:
                    with rasterio.open(ap) as ds:
                        if ds.transform != ref_transform or ds.crs != ref_crs or (ds.width, ds.height) != ref_wh:
                            warnings.warn(f"Grid mismatch in mean, skipping: {ap}", UserWarning)
                            continue
                        arr = ds.read(1, window=win, out_dtype="float32")
                        acc += np.where(np.isfinite(arr), arr, 0.0)
                        n += 1

                out_ds.write((acc / max(n, 1)).astype(np.float32), 1, window=win)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="RUSLE2 R-factor from annual precip stacks + quarterly liquid/solid masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python r_factor_rusle2.py --dataset imerg "
            "--precip_path D:/.../imerg_rfactor_calib "
            "--mask_path \\\\Carbon-NAS\\Artur\\googledrive\\ERA5LAND_PHASEMASK_TW "
            "--out_dir D:/.../rfactor_imerg --year_start 2001 --year_end 2025 "
            "--energy_model bf --exp_k 0.082\n"
        ),
    )

    ap.add_argument("--dataset",      required=True, choices=["imerg", "era5"])
    ap.add_argument("--precip_path",  required=True)
    ap.add_argument("--mask_path",    required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--year_start",   type=int, required=True)
    ap.add_argument("--year_end",     type=int, required=True)
    ap.add_argument("--tile",         type=int, default=256)

    ap.add_argument("--cf_to_ei30",   type=float, default=None,
                    help="EI proxy -> EI30. IMERG: 1.0. ERA5: calibrate locally.")
    ap.add_argument("--erosive_depth", type=float, default=12.7)
    ap.add_argument("--erosive_peak",  type=float, default=25.4)

    ap.add_argument("--split_hours",  type=float, default=6.0)
    ap.add_argument("--split_sum_mm", type=float, default=1.27)

    ap.add_argument("--gap_intensity", type=float, default=1.27,
                    help="Classification boundary [mm/h] for weak steps (pending buffer).")
    ap.add_argument("--cap_intensity", action="store_true")
    ap.add_argument("--intensity_cap", type=float, default=300.0)

    ap.add_argument("--energy_model", default="bf",
                    choices=["bf", "vd", "ws", "salles", "sm"],
                    help="Kinetic-energy parameterisation (see lib/energy_models.py).")
    ap.add_argument("--exp_k", type=float, default=0.05,
                    help="k in BF: e=0.29*(1-0.72*exp(-k*i)). "
                         "Used only with --energy_model bf. Defaults: classical 0.05; "
                         "RUSLE2/Foster 2003: 0.082.")

    args = ap.parse_args()

    energy_model = parse_model(args.energy_model)
    print(f"  Energy model = {args.energy_model}  (exp_k={args.exp_k} only used for bf)")

    spec = DATASET_SPECS[args.dataset]
    if args.cf_to_ei30 is not None:
        spec = DatasetSpec(spec.name, spec.dt_hours, spec.mask_step_hours, cf_to_ei30=args.cf_to_ei30)

    min_split_hours = 2.0 * spec.dt_hours
    if args.split_hours < min_split_hours:
        raise ValueError(
            f"--split_hours must be >= {min_split_hours} for dataset {spec.name} (dt={spec.dt_hours} h)."
        )

    if args.dataset == "era5" and spec.cf_to_ei30 == 1.0:
        warnings.warn(
            "ERA5 dt=1h: I1h != I30. R will be underestimated. Use --cf_to_ei30.",
            UserWarning, stacklevel=2
        )

    print(spec.describe())

    cfg = RConfig(
        event_split_hours=args.split_hours,
        event_split_sum_mm=args.split_sum_mm,
        gap_intensity_mm_h=args.gap_intensity,
        erosive_depth_mm=args.erosive_depth,
        erosive_peak_mm_h=args.erosive_peak,
        use_intensity_cap=args.cap_intensity,
        intensity_cap_mm_h=args.intensity_cap,
    )

    precip_files = list_tifs_from_input(args.precip_path)
    mask_files   = list_tifs_from_input(args.mask_path)

    if not precip_files:
        raise FileNotFoundError(f"No precip files found from: {args.precip_path}")
    if not mask_files:
        raise FileNotFoundError(f"No mask files found from: {args.mask_path}")

    precip_by_year = group_files_by_year(precip_files)
    mask_q_by_year = group_mask_quarters_by_year(mask_files)

    annual_outputs: List[str] = []

    for y in range(args.year_start, args.year_end + 1):
        p_candidates = precip_by_year.get(y, [])
        if not p_candidates:
            print(f"  skip {y}: no precip annual file")
            continue

        if len(p_candidates) > 1:
            counts = []
            for p in p_candidates:
                try:
                    with rasterio.open(p) as ds:
                        counts.append((ds.count, p))
                except Exception:
                    counts.append((0, p))
            counts.sort(reverse=True)
            precip_year_file = counts[0][1]
            warnings.warn(
                f"Year {y}: multiple precip files ({len(p_candidates)}). "
                f"Selected by max band count:\n  {precip_year_file}",
                UserWarning
            )
        else:
            precip_year_file = p_candidates[0]

        year_masks = build_year_mask_sequence(mask_q_by_year, y)
        if len(year_masks) == 0:
            print(f"  skip {y}: no quarterly masks for year")
            continue
        if len(year_masks) < 4:
            warnings.warn(
                f"Year {y}: only {len(year_masks)} mask quarter(s) found. "
                "Will process available quarters only.",
                UserWarning
            )

        out_year = os.path.join(args.out_dir, "annual", f"R_{args.dataset}_{y}.tif")
        compute_R_year_annual_precip_and_quarter_masks(
            precip_year_file=precip_year_file,
            mask_quarters=year_masks,
            out_path=out_year,
            year=y,
            spec=spec,
            cfg=cfg,
            energy_model=energy_model,
            exp_k=args.exp_k,
            tile=args.tile,
        )
        annual_outputs.append(out_year)
        print(f"  written: {out_year}")

    if annual_outputs:
        out_mean = os.path.join(args.out_dir, f"R_{args.dataset}_{args.year_start}_{args.year_end}_MEAN.tif")
        compute_period_mean(annual_outputs, out_mean, tile=args.tile)
        print(f"  written: {out_mean}")
    else:
        print("No annual outputs produced. Check year range and input paths.")


if __name__ == "__main__":
    main()
