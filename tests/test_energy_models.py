# -*- coding: utf-8 -*-
"""
Unit tests for lib/energy_models.py.

Run:
    python -m pytest tests/test_energy_models.py -v
or directly:
    python tests/test_energy_models.py
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
    DEFAULT_ENSEMBLE,
    ENERGY_BROWN_FOSTER,
    ENERGY_SALLES,
    ENERGY_SANCHEZ_MORENO,
    ENERGY_VAN_DIJK,
    ENERGY_WISCHMEIER,
    ensemble_curves,
    parse_model,
    unit_energy,
    unit_energy_np,
)


# ------------------------------------------------------------------ #
# 1. Boundary conditions
# ------------------------------------------------------------------ #
def test_zero_intensity_yields_zero_energy():
    for label, model, k in DEFAULT_ENSEMBLE:
        e = unit_energy_np(0.0, model=model, exp_k=k)
        assert e == 0.0, f"{label}: e(0) != 0"


def test_negative_intensity_yields_zero():
    for label, model, k in DEFAULT_ENSEMBLE:
        e = unit_energy_np(-3.5, model=model, exp_k=k)
        assert e == 0.0, f"{label}: e(<0) != 0"


# ------------------------------------------------------------------ #
# 2. Reference values at the RUSLE 25.4 mm/h erosivity threshold
# ------------------------------------------------------------------ #
def test_reference_values_at_25_4():
    """Hand-computed reference values; tolerance ±2e-3 MJ/ha/mm."""
    refs = {
        "bf_k05":          0.2314,
        "bf_k082":         0.2640,
        "van_dijk":        0.2324,
        "wischmeier":      0.2416,
        "sanchez_moreno":  0.2780,
    }
    for label, model, k in DEFAULT_ENSEMBLE:
        e = unit_energy_np(25.4, model=model, exp_k=k)
        assert math.isclose(e, refs[label], abs_tol=2e-3), (
            f"{label}: expected {refs[label]:.4f}, got {e:.4f}"
        )


# ------------------------------------------------------------------ #
# 3. Saturation behaviour (BF / vD / Salles / SM all asymptote to e_max)
# ------------------------------------------------------------------ #
def test_brown_foster_saturation():
    e = unit_energy_np(1000.0, model="bf", exp_k=0.082)
    assert math.isclose(e, 0.29, abs_tol=1e-4)


def test_van_dijk_saturation():
    e = unit_energy_np(1000.0, model="vd")
    assert math.isclose(e, 0.283, abs_tol=1e-4)


def test_sanchez_moreno_saturation():
    e = unit_energy_np(1000.0, model="sm")
    assert math.isclose(e, 0.353, abs_tol=1e-4)


def test_wischmeier_saturation_at_76():
    """W&S saturates at i=76 mm/h. We use a continuous cap (the log-curve's
    value at i=76, ~0.2832) so the function is strictly monotone."""
    expected = 0.119 + 0.0873 * math.log10(76.0)
    e_low = unit_energy_np(75.99, model="ws")
    e_at = unit_energy_np(76.0, model="ws")
    e_hi = unit_energy_np(200.0, model="ws")
    assert math.isclose(e_low, expected, abs_tol=2e-3)
    assert math.isclose(e_at, expected, abs_tol=1e-12)
    assert math.isclose(e_hi, expected, abs_tol=1e-12)


# ------------------------------------------------------------------ #
# 4. Monotonicity (continuous models)
# ------------------------------------------------------------------ #
def test_monotonic_increasing():
    """e(i) must be non-decreasing in i for all parameterisations."""
    i_grid = np.linspace(0.05, 200.0, 500)
    for label, model, k in DEFAULT_ENSEMBLE:
        e = unit_energy_np(i_grid, model=model, exp_k=k)
        diffs = np.diff(e)
        # Allow tiny negative due to floating-point; tolerance 1e-6
        assert (diffs >= -1e-6).all(), f"{label}: not monotone"


# ------------------------------------------------------------------ #
# 5. Numba dispatcher matches numpy reference exactly
# ------------------------------------------------------------------ #
def test_njit_matches_numpy():
    """The njit `unit_energy` must be byte-identical to the numpy reference
    for a representative grid of intensities."""
    i_grid = np.array([0.0, 0.05, 0.5, 1.0, 1.27, 5.0, 12.7, 25.4, 50.0, 100.0, 200.0])
    cases = [
        (ENERGY_BROWN_FOSTER, 0.05),
        (ENERGY_BROWN_FOSTER, 0.082),
        (ENERGY_VAN_DIJK, 0.0),
        (ENERGY_WISCHMEIER, 0.0),
        (ENERGY_SALLES, 0.0),
        (ENERGY_SANCHEZ_MORENO, 0.0),
    ]
    np_label = {
        ENERGY_BROWN_FOSTER: "bf",
        ENERGY_VAN_DIJK: "vd",
        ENERGY_WISCHMEIER: "ws",
        ENERGY_SALLES: "salles",
        ENERGY_SANCHEZ_MORENO: "sm",
    }
    for mid, k in cases:
        for i in i_grid:
            e_njit = unit_energy(float(i), mid, k)
            e_np = unit_energy_np(float(i), model=np_label[mid], exp_k=k)
            assert math.isclose(e_njit, e_np, abs_tol=1e-12, rel_tol=1e-12), (
                f"model_id={mid}, k={k}, i={i}: njit={e_njit}, np={e_np}"
            )


# ------------------------------------------------------------------ #
# 6. Ensemble spread sanity
# ------------------------------------------------------------------ #
def test_ensemble_spread_at_threshold():
    """Spread of e at i=25.4 mm/h across the ensemble should be ~10-20%."""
    curves = ensemble_curves(np.array([25.4]))
    vals = np.array([v[0] for v in curves.values()])
    cv = vals.std() / vals.mean()
    assert 0.05 < cv < 0.25, f"unexpected ensemble CV at i=25.4: {cv:.3f}"


# ------------------------------------------------------------------ #
# 7. parse_model
# ------------------------------------------------------------------ #
def test_parse_model_aliases():
    assert parse_model("bf") == ENERGY_BROWN_FOSTER
    assert parse_model("Brown_Foster") == ENERGY_BROWN_FOSTER
    assert parse_model("vd") == ENERGY_VAN_DIJK
    assert parse_model("ws") == ENERGY_WISCHMEIER
    assert parse_model("sm") == ENERGY_SANCHEZ_MORENO
    assert parse_model("salles") == ENERGY_SALLES


def test_parse_model_invalid():
    try:
        parse_model("foster_2003")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ------------------------------------------------------------------ #
# Standalone runner
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import inspect
    failed = 0
    passed = 0
    g = dict(globals())
    for name, fn in g.items():
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
