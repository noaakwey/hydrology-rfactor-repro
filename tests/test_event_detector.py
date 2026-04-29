# -*- coding: utf-8 -*-
"""
Synthetic-storm tests for the RUSLE2-like event detector.

Each test specifies an intensity series (mm/h on a fixed dt grid) and an
analytically-computed expected R-factor.  We then compare against:
  (a) the pure-Python reference detector (lib.event_detector.detect_events),
  (b) the numba kernel from r_factor_rusle2.process_step driven on a
      single-pixel raster.

The two implementations must agree to floating-point tolerance.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from erosivity.lib.energy_models import (
    ENERGY_BROWN_FOSTER,
    parse_model,
    unit_energy_np,
)
from erosivity.lib.event_detector import DetectorConfig, detect_events
from erosivity.r_factor_rusle2 import process_step


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _kernel_R(
    intens: np.ndarray,
    cfg: DetectorConfig,
    energy_model: int,
    exp_k: float,
    liquid: np.ndarray | None = None,
) -> float:
    """Drive the numba kernel on a single pixel and return annual_R."""
    n = len(intens)
    i_band = np.zeros((1, 1), dtype=np.float32)
    valid = np.ones((1, 1), dtype=np.uint8)
    liq = np.ones((1, 1), dtype=np.uint8) if liquid is None else None

    annual_R = np.zeros((1, 1), dtype=np.float32)
    in_event = np.zeros((1, 1), dtype=np.uint8)
    event_E = np.zeros((1, 1), dtype=np.float32)
    event_Imax = np.zeros((1, 1), dtype=np.float32)
    event_P = np.zeros((1, 1), dtype=np.float32)
    event_has_peak = np.zeros((1, 1), dtype=np.uint8)

    dry_steps = np.zeros((1, 1), dtype=np.uint32)
    dry_sum = np.zeros((1, 1), dtype=np.float32)
    pending_P = np.zeros((1, 1), dtype=np.float32)
    pending_E = np.zeros((1, 1), dtype=np.float32)
    pending_Imax = np.zeros((1, 1), dtype=np.float32)
    pending_has_peak = np.zeros((1, 1), dtype=np.uint8)

    for k in range(n):
        i_band[0, 0] = float(intens[k])
        if liquid is not None:
            liq = np.array([[int(liquid[k])]], dtype=np.uint8)
        process_step(
            i_band=i_band,
            valid=valid,
            liquid=liq,
            dt_hours=cfg.dt_hours,
            split_steps=cfg.split_steps,
            split_sum_mm=cfg.event_split_sum_mm,
            gap_intensity=cfg.gap_intensity_mm_h,
            erosive_depth=cfg.erosive_depth_mm,
            erosive_peak=cfg.erosive_peak_mm_h,
            use_cap=False,
            cap_mm_h=300.0,
            gap_close=True,
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

    # End-of-record flush + close (mirror raster end-of-year code)
    if in_event[0, 0] == 1 and pending_P[0, 0] > 0.0:
        event_P[0, 0] += pending_P[0, 0]
        event_E[0, 0] += pending_E[0, 0]
        if pending_Imax[0, 0] > event_Imax[0, 0]:
            event_Imax[0, 0] = pending_Imax[0, 0]
        if pending_has_peak[0, 0] == 1:
            event_has_peak[0, 0] = 1

    if in_event[0, 0] == 1:
        if (event_P[0, 0] >= cfg.erosive_depth_mm) or (event_has_peak[0, 0] == 1):
            annual_R[0, 0] += event_E[0, 0] * event_Imax[0, 0]

    return float(annual_R[0, 0])


def _expected_E_I_for_constant_storm(i: float, n_steps: int, cfg: DetectorConfig) -> float:
    """Analytical expected EI30 for a uniform-intensity storm of n_steps duration."""
    e = float(unit_energy_np(i, model=cfg.energy_model, exp_k=cfg.exp_k))
    p = i * cfg.dt_hours
    E = e * p * n_steps
    return E * i


def _series_with_baseline(values, total_len, fill=0.0):
    """Embed `values` at index 0 of a length-`total_len` zero-padded series."""
    out = np.full(total_len, fill, dtype=np.float64)
    out[: len(values)] = values
    return out


# ------------------------------------------------------------------ #
# 1. Trivial: no rain -> R = 0
# ------------------------------------------------------------------ #
def test_no_rain():
    cfg = DetectorConfig()
    arr = np.zeros(48, dtype=np.float64)
    res = detect_events(arr, cfg)
    R_kern = _kernel_R(arr, cfg, ENERGY_BROWN_FOSTER, cfg.exp_k)
    assert res.annual_R == 0.0
    assert R_kern == 0.0
    assert len(res.events) == 0


# ------------------------------------------------------------------ #
# 2. Sub-erosive light rain (below depth and peak) -> R = 0
# ------------------------------------------------------------------ #
def test_sub_erosive_drizzle():
    cfg = DetectorConfig()
    # 4 hours of i=2 mm/h: depth = 2*4 = 8 mm < 12.7, peak 2 < 25.4 -> not erosive
    arr = np.full(8, 2.0)  # 8 steps * 0.5 h = 4 h
    pad = _series_with_baseline(arr, 100)
    res = detect_events(pad, cfg)
    R_kern = _kernel_R(pad, cfg, ENERGY_BROWN_FOSTER, cfg.exp_k)
    assert res.annual_R == 0.0
    assert R_kern == 0.0
    assert len(res.events) == 1
    assert not res.events[0].erosive


# ------------------------------------------------------------------ #
# 3. Single uniform storm above depth threshold -> exact analytical R
# ------------------------------------------------------------------ #
def test_single_uniform_storm():
    cfg = DetectorConfig(energy_model="bf", exp_k=0.082)
    # 2 h at 10 mm/h: depth = 20 mm > 12.7, peak 10 < 25.4 -> erosive by depth
    n_storm = 4
    intens = 10.0
    arr = _series_with_baseline([intens] * n_storm, 60)

    res = detect_events(arr, cfg)
    expected = _expected_E_I_for_constant_storm(intens, n_storm, cfg)

    assert math.isclose(res.annual_R, expected, rel_tol=1e-9)
    R_kern = _kernel_R(arr, cfg, ENERGY_BROWN_FOSTER, cfg.exp_k)
    assert math.isclose(R_kern, expected, rel_tol=1e-5)


# ------------------------------------------------------------------ #
# 4. Single high-intensity short burst -> erosive by peak
# ------------------------------------------------------------------ #
def test_single_high_peak_burst():
    cfg = DetectorConfig(energy_model="bf", exp_k=0.082)
    # 1 step at i = 30 mm/h: depth = 15 mm > 12.7 AND peak 30 >= 25.4
    arr = _series_with_baseline([30.0], 60)
    res = detect_events(arr, cfg)
    expected = _expected_E_I_for_constant_storm(30.0, 1, cfg)
    assert math.isclose(res.annual_R, expected, rel_tol=1e-9)
    R_kern = _kernel_R(arr, cfg, ENERGY_BROWN_FOSTER, cfg.exp_k)
    assert math.isclose(R_kern, expected, rel_tol=1e-5)


# ------------------------------------------------------------------ #
# 5. Two storms separated by a long, dry gap -> two independent events
# ------------------------------------------------------------------ #
def test_two_storms_dry_gap():
    cfg = DetectorConfig(energy_model="bf", exp_k=0.082)
    # storm A: 2h@10 mm/h ; gap = 14 steps (7 h) of zero ; storm B: 2h@15 mm/h
    arr = np.concatenate([
        np.full(4, 10.0),
        np.zeros(14),
        np.full(4, 15.0),
        np.zeros(20),
    ])
    res = detect_events(arr, cfg)
    assert len(res.events) == 2
    expA = _expected_E_I_for_constant_storm(10.0, 4, cfg)
    expB = _expected_E_I_for_constant_storm(15.0, 4, cfg)
    assert math.isclose(res.annual_R, expA + expB, rel_tol=1e-9)
    R_kern = _kernel_R(arr, cfg, ENERGY_BROWN_FOSTER, cfg.exp_k)
    assert math.isclose(R_kern, expA + expB, rel_tol=1e-5)


# ------------------------------------------------------------------ #
# 6. Two storms with a wet gap (drizzle) -> single merged event
# ------------------------------------------------------------------ #
def test_two_storms_wet_gap_merge():
    """A 6-h gap that contains > 1.27 mm of drizzle should NOT split the
    event (wet gap merges via pending-buffer flush)."""
    cfg = DetectorConfig(energy_model="bf", exp_k=0.082)
    # storm A: 2h@10 ; 12 weak steps of 0.4 mm/h (=0.2 mm/step => 2.4 mm > 1.27 wet) ; storm B: 2h@15
    arr = np.concatenate([
        np.full(4, 10.0),
        np.full(12, 0.4),  # 6 h at 0.4 mm/h, dry_sum = 0.4*0.5*12 = 2.4 mm > 1.27
        np.full(4, 15.0),
        np.zeros(20),
    ])
    res = detect_events(arr, cfg)
    assert len(res.events) == 1, f"expected single event, got {len(res.events)}"
    R_kern = _kernel_R(arr, cfg, ENERGY_BROWN_FOSTER, cfg.exp_k)
    assert math.isclose(R_kern, res.annual_R, rel_tol=1e-5)


# ------------------------------------------------------------------ #
# 7. Phase mask: solid precipitation produces zero R
# ------------------------------------------------------------------ #
def test_phase_mask_zeroes_solid():
    cfg = DetectorConfig(energy_model="bf", exp_k=0.082)
    arr = _series_with_baseline([15.0] * 4, 30)
    liq = np.zeros_like(arr, dtype=np.int8)  # all solid
    res = detect_events(arr, cfg, liquid_mask=liq)
    assert res.annual_R == 0.0


# ------------------------------------------------------------------ #
# 8. Pending-buffer asymmetry: leading drizzle should NOT split a single storm.
#
# This is the asymmetry I flagged on review: a weak step BEFORE the burst
# opens an event (in_event=1), and is then either flushed (wet gap) or
# discarded (dry gap).  Verify that the natural case "5 steps of 0.5 mm/h
# leading into a 2h burst at 10 mm/h" yields ONE event whose R equals the
# burst-only R plus the small contribution of the leading drizzle, NOT
# two events.
# ------------------------------------------------------------------ #
def test_pending_leading_drizzle_no_split():
    cfg = DetectorConfig(energy_model="bf", exp_k=0.082)
    # 5 steps@0.5 mm/h: dry_sum after 5 = 5*0.5*0.5 = 1.25 mm < 1.27 -> still pending
    # then immediate burst -> pending must flush into event, not start a 2nd.
    arr = np.concatenate([
        np.full(5, 0.5),
        np.full(4, 10.0),
        np.zeros(20),
    ])
    res = detect_events(arr, cfg)
    assert len(res.events) == 1
    # Total depth: 5*0.5*0.5 + 4*10*0.5 = 1.25 + 20.0 = 21.25 mm; peak=10
    assert math.isclose(res.events[0].depth_mm, 21.25, abs_tol=1e-6)
    assert math.isclose(res.events[0].i_max_mm_h, 10.0, abs_tol=1e-9)


# ------------------------------------------------------------------ #
# 9. Pending-buffer asymmetry: leading drizzle THEN long dry gap THEN burst
#
# Expected: leading drizzle should NOT count as a separate (sub-erosive)
# event; instead the dry gap closes the would-be drizzle event (it never
# became erosive), and the burst is its own event.
# ------------------------------------------------------------------ #
def test_pending_leading_drizzle_with_dry_gap():
    cfg = DetectorConfig(energy_model="bf", exp_k=0.082)
    arr = np.concatenate([
        np.full(2, 0.5),     # 2 weak steps; pending; opens an event
        np.zeros(13),        # 13 zeros = 6.5 h dry; split_steps=12
        np.full(4, 10.0),    # main burst
        np.zeros(20),
    ])
    res = detect_events(arr, cfg)
    # Drizzle event: depth = 0.5 mm < 12.7, peak 0.5 < 25.4 -> not erosive
    # Burst event:   depth = 20 mm > 12.7 -> erosive by depth
    erosive_events = [e for e in res.events if e.erosive]
    assert len(erosive_events) == 1
    assert math.isclose(erosive_events[0].depth_mm, 20.0, abs_tol=1e-6)


# ------------------------------------------------------------------ #
# 10. Numba kernel agrees with reference detector on a complex random series
# ------------------------------------------------------------------ #
def test_kernel_matches_reference_random():
    rng = np.random.default_rng(7)
    cfg = DetectorConfig(energy_model="bf", exp_k=0.082)
    # 7 days of 30-min steps with intermittent storms
    n = 24 * 2 * 7
    arr = np.zeros(n, dtype=np.float64)
    for _ in range(20):
        start = rng.integers(0, n - 8)
        length = rng.integers(2, 8)
        peak = rng.uniform(0.5, 60.0)
        arr[start:start + length] = peak * rng.uniform(0.5, 1.0, size=length)
    R_ref = detect_events(arr, cfg).annual_R
    R_kern = _kernel_R(arr, cfg, ENERGY_BROWN_FOSTER, cfg.exp_k)
    assert math.isclose(R_ref, R_kern, rel_tol=1e-4, abs_tol=1e-3), (
        f"reference R = {R_ref:.6f}, kernel R = {R_kern:.6f}"
    )


# ------------------------------------------------------------------ #
# 11. Energy ensemble agreement across the same storm (sanity ordering)
# ------------------------------------------------------------------ #
def test_energy_models_ordering():
    """For a moderate storm, BF k=0.082 should give the highest R, BF
    k=0.05 should be lower, vD lower still, and W&S in between BF k=0.05
    and BF k=0.082 (in this regime).  Used as monotone-in-parameterisation
    sanity, not a strict numerical check."""
    cfg_base = DetectorConfig()
    arr = np.concatenate([
        np.full(2, 5.0),
        np.full(4, 20.0),
        np.full(2, 8.0),
        np.zeros(20),
    ])
    Rs = {}
    for label, model, k in [
        ("bf_k05", "bf", 0.05),
        ("bf_k082", "bf", 0.082),
        ("vd", "vd", 0.0),
        ("ws", "ws", 0.0),
    ]:
        cfg = DetectorConfig(energy_model=model, exp_k=k)
        Rs[label] = detect_events(arr, cfg).annual_R
    assert Rs["bf_k082"] > Rs["bf_k05"], Rs
    assert Rs["bf_k082"] > Rs["vd"], Rs
    # All positive and finite
    assert all(np.isfinite(v) and v > 0 for v in Rs.values())


# ------------------------------------------------------------------ #
# Standalone runner
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import inspect
    failed = 0
    passed = 0
    for name, fn in dict(globals()).items():
        if name.startswith("test_") and callable(fn) and len(inspect.signature(fn).parameters) == 0:
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as exc:
                print(f"  FAIL  {name}: {exc}")
                failed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {name}: {exc!r}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
