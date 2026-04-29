# -*- coding: utf-8 -*-
"""
lib/event_detector.py
=====================
Reference, pure-Python (numpy) implementation of the RUSLE2-like event
detector.  Used as a verification ground-truth for the numba kernel in
`r_factor_rusle2.py`, and as a single-pixel oracle for sensitivity tests.

The logic is exactly the same as the kernel:
    * Weak step:   i < gap_intensity
    * Significant: i >= gap_intensity
    * Inter-event split: dry_steps >= split_steps AND dry_sum < split_sum_mm
    * Pending buffer: weak steps accumulate; flushed into event on a wet
      gap (dry_sum >= split_sum_mm) and at year-end; discarded on a valid
      (dry) gap.
    * Erosive event: event_P >= erosive_depth OR event_has_peak (i >= erosive_peak)
    * Annual R = sum over erosive events of E_event * I_max_event.

The reference implementation is written for clarity, not speed.  It also
returns per-event diagnostics, which the numba kernel does not expose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from erosivity.lib.energy_models import unit_energy_np


@dataclass
class EventRecord:
    start_idx: int
    end_idx: int               # inclusive
    depth_mm: float
    energy_MJ_ha: float        # cumulative E
    i_max_mm_h: float
    has_peak: bool             # any step with i >= erosive_peak
    erosive: bool              # depth >= erosive_depth OR has_peak
    EI30: float                # E * I_max (set 0 if not erosive)


@dataclass
class DetectorConfig:
    dt_hours: float = 0.5
    event_split_hours: float = 6.0
    event_split_sum_mm: float = 1.27
    gap_intensity_mm_h: float = 1.27
    erosive_depth_mm: float = 12.7
    erosive_peak_mm_h: float = 25.4
    energy_model: str = "bf"
    exp_k: float = 0.05

    @property
    def split_steps(self) -> int:
        return max(1, int(round(self.event_split_hours / self.dt_hours)))


@dataclass
class DetectorResult:
    annual_R: float
    events: List[EventRecord] = field(default_factory=list)


def _e(i: float, cfg: DetectorConfig) -> float:
    return float(unit_energy_np(i, model=cfg.energy_model, exp_k=cfg.exp_k))


def detect_events(
    intensities_mm_h: Sequence[float],
    cfg: DetectorConfig | None = None,
    liquid_mask: Sequence[int] | None = None,
) -> DetectorResult:
    """
    Walk through a 1-D intensity series (one pixel) and return the annual
    R-factor plus per-event diagnostics.

    Parameters
    ----------
    intensities_mm_h
        Sequence of step-mean intensities (mm/h), one per dt_hours-long step.
    cfg
        Detector configuration (defaults match RUSLE2 + IMERG 30-min).
    liquid_mask
        Optional 0/1 mask of same length; 0 zeroes the intensity (treated
        as solid precipitation, no contribution to R).

    Returns
    -------
    DetectorResult with `annual_R` and `events`.
    """
    cfg = cfg or DetectorConfig()
    arr = np.asarray(intensities_mm_h, dtype=np.float64)
    if liquid_mask is None:
        liq = np.ones_like(arr, dtype=np.int8)
    else:
        liq = np.asarray(liquid_mask, dtype=np.int8)
    n = len(arr)

    annual_R = 0.0
    events: List[EventRecord] = []

    # Event state
    in_event = False
    ev_start = 0
    ev_E = 0.0
    ev_I = 0.0
    ev_P = 0.0
    ev_has_peak = False
    ev_last_idx = -1   # last significant step index

    # Gap / pending state
    dry_steps = 0
    dry_sum = 0.0
    pend_P = 0.0
    pend_E = 0.0
    pend_I = 0.0
    pend_has_peak = False
    pend_last_idx = -1

    def _close_event(end_idx: int):
        nonlocal annual_R, in_event, ev_start, ev_E, ev_I, ev_P, ev_has_peak, ev_last_idx
        erosive = (ev_P >= cfg.erosive_depth_mm) or ev_has_peak
        EI30 = ev_E * ev_I if erosive else 0.0
        if erosive:
            annual_R += EI30
        events.append(
            EventRecord(
                start_idx=ev_start,
                end_idx=end_idx,
                depth_mm=ev_P,
                energy_MJ_ha=ev_E,
                i_max_mm_h=ev_I,
                has_peak=ev_has_peak,
                erosive=erosive,
                EI30=EI30,
            )
        )
        in_event = False
        ev_E = ev_I = ev_P = 0.0
        ev_has_peak = False

    def _flush_pending():
        nonlocal ev_E, ev_I, ev_P, ev_has_peak
        nonlocal pend_P, pend_E, pend_I, pend_has_peak, pend_last_idx
        if pend_P > 0.0:
            ev_E += pend_E
            ev_P += pend_P
            if pend_I > ev_I:
                ev_I = pend_I
            ev_has_peak = ev_has_peak or pend_has_peak
        pend_P = 0.0
        pend_E = 0.0
        pend_I = 0.0
        pend_has_peak = False
        pend_last_idx = -1

    for k in range(n):
        i = arr[k]
        if liq[k] == 0 or not np.isfinite(i) or i < 0.0:
            i = 0.0
        p = i * cfg.dt_hours

        if i < cfg.gap_intensity_mm_h:
            # Weak step
            if not in_event and p > 0.0:
                in_event = True
                ev_start = k
                ev_E = ev_I = ev_P = 0.0
                ev_has_peak = False

            dry_steps += 1
            dry_sum += p

            if p > 0.0:
                pend_E += _e(i, cfg) * p
                pend_P += p
                if i > pend_I:
                    pend_I = i
                if i >= cfg.erosive_peak_mm_h:
                    pend_has_peak = True
                pend_last_idx = k

            if in_event and dry_steps >= cfg.split_steps:
                if dry_sum < cfg.event_split_sum_mm:
                    # Valid (dry) gap: close event, discard pending
                    _close_event(end_idx=ev_last_idx if ev_last_idx >= 0 else k)
                else:
                    # Wet gap: commit pending into event
                    if pend_last_idx >= 0:
                        ev_last_idx = max(ev_last_idx, pend_last_idx)
                    _flush_pending()
                dry_steps = 0
                dry_sum = 0.0
                pend_P = 0.0
                pend_E = 0.0
                pend_I = 0.0
                pend_has_peak = False
                pend_last_idx = -1
            continue

        # Significant step
        if not in_event:
            in_event = True
            ev_start = k
            ev_E = ev_I = ev_P = 0.0
            ev_has_peak = False

        # Flush any pending; track latest contributing index
        if pend_last_idx >= 0:
            ev_last_idx = max(ev_last_idx, pend_last_idx)
        _flush_pending()
        dry_steps = 0
        dry_sum = 0.0

        ev_E += _e(i, cfg) * p
        ev_P += p
        if i > ev_I:
            ev_I = i
        if i >= cfg.erosive_peak_mm_h:
            ev_has_peak = True
        ev_last_idx = k

    # End-of-record: flush pending into open event, then close
    if in_event:
        if pend_P > 0.0:
            if pend_last_idx >= 0:
                ev_last_idx = max(ev_last_idx, pend_last_idx)
            _flush_pending()
        _close_event(end_idx=ev_last_idx if ev_last_idx >= 0 else (n - 1))

    return DetectorResult(annual_R=annual_R, events=events)
