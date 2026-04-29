# Hydrology R-factor Reproducibility Repository

This repository packages the source code used to calibrate IMERG precipitation against station observations and to compute the rainfall erosivity (`R`) products discussed in the manuscript prepared for *Hydrology*.

The repository is intentionally data-free. It contains:

- calibration code adapted from `imerg2meteo_calib`
- erosivity and diagnostics code adapted from `rfactor-analysis`
- workflow scripts for calibration, annual stack generation, R-factor calculation, sensitivity diagnostics, structural-break checks, and figure production
- unit tests covering the core statistical and erosivity logic

It does **not** contain station observations, raw IMERG archives, ERA5 phase masks, or derived rasters.

## Repository layout

- `calibration/` station-satellite calibration code
- `erosivity/` RUSLE2-like event detection and R-factor computation
- `workflows/` end-to-end scripts for the publication pipeline
- `tests/` unit tests
- `data/` documentation of expected external inputs
- `docs/` reproducibility notes
- `config/` path defaults used by selected workflows

## Minimal workflow

1. Prepare the external inputs described in `data/README.md`.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Build station-level calibration tables:

```bash
python calibration/main.py \
  --dataset imerg \
  --meteo-dir data/raw/meteo \
  --sat-dir data/raw/imerg_stations_3h \
  --out-dir data/interim/calib_imerg
```

4. Build annual calibrated 30-min IMERG stacks:

```bash
python workflows/run_v6_imerg_pipeline.py \
  --zip data/raw/imerg_quarterly/imerg_30min_quarters.zip \
  --calib-dir data/interim/calib_imerg \
  --meteo-dir data/raw/meteo \
  --out-dir output/v6_imerg_calibrated \
  --diagnostics-dir output/v6_diagnostics
```

5. Compute annual R-factor rasters:

```bash
python workflows/compute_rfactor_v7.py \
  --precip-dir output/v6_imerg_calibrated \
  --mask-dir data/external/era5_phase_masks \
  --out-dir output/v6_rfactor \
  --exp-k 0.082
```

6. Run checks and build publication figures as needed:

```bash
pytest
python workflows/sensitivity_pack.py --aws-csv data/external/aws310_pluvio_v2.csv
python workflows/structural_breaks.py --data-dir output/v6_rfactor/annual
python workflows/plot_v6_paper.py
```

## Reproducibility scope

The code path is fully reproducible from source, but exact raster and tabular outputs still depend on access to:

- station precipitation observations used for calibration
- local IMERG quarterly 30-min stacks
- ERA5-based rain/snow phase masks
- site-specific high-frequency validation series used for peak diagnostics

Those inputs are intentionally excluded from version control. The manuscript Data Availability Statement should therefore distinguish between:

- public code: this repository
- restricted or large inputs: described in `data/README.md`
- derived publication outputs: optionally releasable via Zenodo or a separate archive
