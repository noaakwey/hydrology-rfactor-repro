# -*- coding: utf-8 -*-
"""
lib/energy_models.py
====================
Ensemble of rainfall kinetic energy parameterisations e(i)
[MJ ha^-1 mm^-1] used in the RUSLE family.

Each model returns kinetic energy per unit precipitation as a function of
intensity i [mm/h]. All models are njit-compatible (pure float math, no
Python objects) so they can be called from the numba kernel.

Models
------
  id=0  Brown & Foster (1987)           e = 0.29  * (1 - 0.72 * exp(-k * i))
        - default classical:  k = 0.05
        - RUSLE2 / Foster (2003):  k = 0.082
  id=1  van Dijk et al. (2002)          e = 0.283 * (1 - 0.52 * exp(-0.042 * i))
        - global mean fit, lower e at low i; widely used in Europe.
  id=2  Wischmeier & Smith (1958)       e = 0.119 + 0.0873 * log10(i),  i in [0.05, 76]
                                        e = 0.283 for i > 76 mm/h (saturation)
        - original USLE; tends to overestimate at low i, saturates at high i.
  id=3  Salles et al. (2002, tropical)  e = 0.359 * (1 - 0.56 * exp(-0.034 * i))
        - tropical climate; here used purely as ensemble bound.
  id=4  Sanchez-Moreno et al. (2014, oceanic)
                                        e = 0.353 * (1 - 0.65 * exp(-0.044 * i))
        - oceanic / boreal climate fit; relevant ensemble member for ETR.

The classical (id=0, k=0.05) and RUSLE2 (id=0, k=0.082) Brown-Foster forms
share e_max = 0.29; they differ only in the rate at which the asymptote is
approached. All other forms have different e_max.

Why an ensemble?
----------------
Brown-Foster coefficients were calibrated against drop-size distributions in
the US Midwest. For mid-latitude continental climates the actual DSD differs,
and Russia has no published disdrometric calibration.  Using an ensemble of
parameterisations bounds the resulting e-induced uncertainty in R without
adopting an unsupported single value.

Reference behaviour at i = 25.4 mm/h (RUSLE erosive intensity threshold):

    BF k=0.05 :  0.231 MJ/ha/mm
    BF k=0.082:  0.264 MJ/ha/mm
    van Dijk  :  0.232 MJ/ha/mm
    Salles    :  0.274 MJ/ha/mm
    Sanchez-M :  0.278 MJ/ha/mm
    W-S       :  0.242 MJ/ha/mm
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

# Energy-model integer ids (kept stable; do NOT renumber once raster metadata
# starts to reference them).
ENERGY_BROWN_FOSTER = 0
ENERGY_VAN_DIJK = 1
ENERGY_WISCHMEIER = 2
ENERGY_SALLES = 3
ENERGY_SANCHEZ_MORENO = 4

ENERGY_MODEL_IDS = {
    "bf": ENERGY_BROWN_FOSTER,
    "brown_foster": ENERGY_BROWN_FOSTER,
    "vd": ENERGY_VAN_DIJK,
    "van_dijk": ENERGY_VAN_DIJK,
    "ws": ENERGY_WISCHMEIER,
    "wischmeier": ENERGY_WISCHMEIER,
    "salles": ENERGY_SALLES,
    "sm": ENERGY_SANCHEZ_MORENO,
    "sanchez_moreno": ENERGY_SANCHEZ_MORENO,
}


def parse_model(name: str) -> int:
    key = str(name).strip().lower()
    if key not in ENERGY_MODEL_IDS:
        raise ValueError(
            f"Unknown energy model '{name}'. "
            f"Allowed: {sorted(set(ENERGY_MODEL_IDS.keys()))}"
        )
    return ENERGY_MODEL_IDS[key]


# ------------------------------------------------------------------ #
# Numba-compiled unit-energy dispatcher
# ------------------------------------------------------------------ #
@njit(cache=True, fastmath=True)
def unit_energy(i_mm_h: float, model_id: int, exp_k: float) -> float:
    """
    Return e(i) [MJ ha^-1 mm^-1] for a chosen parameterisation.

    `exp_k` is only used for model_id = ENERGY_BROWN_FOSTER (Brown & Foster).
    For other models it is ignored; callers may pass 0.0.
    """
    if i_mm_h <= 0.0:
        return 0.0

    if model_id == 0:  # Brown & Foster
        return 0.29 * (1.0 - 0.72 * math.exp(-exp_k * i_mm_h))

    if model_id == 1:  # van Dijk 2002
        return 0.283 * (1.0 - 0.52 * math.exp(-0.042 * i_mm_h))

    if model_id == 2:  # Wischmeier-Smith 1958
        if i_mm_h < 0.05:
            return 0.0
        if i_mm_h >= 76.0:
            # Saturation value taken from the log-curve at the cap so that the
            # function is continuous (W&S spec rounds e_max to 0.283; using the
            # exact value at i=76 mm/h removes a sub-mJ discontinuity that
            # otherwise breaks strict monotonicity tests).
            return 0.119 + 0.0873 * math.log10(76.0)
        return 0.119 + 0.0873 * math.log10(i_mm_h)

    if model_id == 3:  # Salles 2002 (tropical)
        return 0.359 * (1.0 - 0.56 * math.exp(-0.034 * i_mm_h))

    if model_id == 4:  # Sanchez-Moreno 2014 (oceanic)
        return 0.353 * (1.0 - 0.65 * math.exp(-0.044 * i_mm_h))

    # Unknown model -> 0 to remain numerically safe
    return 0.0


# ------------------------------------------------------------------ #
# Pure-numpy reference (for tests, plotting, sensitivity)
# ------------------------------------------------------------------ #
def unit_energy_np(i_mm_h, model: str = "bf", exp_k: float = 0.05) -> np.ndarray:
    """
    Vectorised, plain-numpy reference matching the njit kernel exactly.
    Used for tests and figures so we do not depend on numba at import time.
    """
    arr = np.atleast_1d(np.asarray(i_mm_h, dtype=np.float64))
    out = np.zeros_like(arr)
    pos = arr > 0.0
    a = arr[pos]

    mid = parse_model(model)
    if mid == ENERGY_BROWN_FOSTER:
        out[pos] = 0.29 * (1.0 - 0.72 * np.exp(-exp_k * a))
    elif mid == ENERGY_VAN_DIJK:
        out[pos] = 0.283 * (1.0 - 0.52 * np.exp(-0.042 * a))
    elif mid == ENERGY_WISCHMEIER:
        e_cap = 0.119 + 0.0873 * np.log10(76.0)  # ~0.28320, continuous with log curve
        e = np.where(a < 0.05, 0.0, 0.119 + 0.0873 * np.log10(np.maximum(a, 1e-12)))
        e = np.where(a >= 76.0, e_cap, e)
        out[pos] = e
    elif mid == ENERGY_SALLES:
        out[pos] = 0.359 * (1.0 - 0.56 * np.exp(-0.034 * a))
    elif mid == ENERGY_SANCHEZ_MORENO:
        out[pos] = 0.353 * (1.0 - 0.65 * np.exp(-0.044 * a))

    if np.isscalar(i_mm_h):
        return float(out[0])
    return out


# ------------------------------------------------------------------ #
# Convenience: ensemble evaluation
# ------------------------------------------------------------------ #
DEFAULT_ENSEMBLE = (
    ("bf_k05", "bf", 0.05),
    ("bf_k082", "bf", 0.082),
    ("van_dijk", "vd", 0.0),
    ("wischmeier", "ws", 0.0),
    ("sanchez_moreno", "sm", 0.0),
)


def ensemble_curves(intensities_mm_h: np.ndarray) -> "dict[str, np.ndarray]":
    """Return dict {label: e(i)} for the standard ensemble."""
    out = {}
    for label, model, k in DEFAULT_ENSEMBLE:
        out[label] = unit_energy_np(intensities_mm_h, model=model, exp_k=k)
    return out
