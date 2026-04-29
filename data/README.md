# External Data Manifest

This repository excludes all input data. To rerun the full workflow, place external files under the paths below or pass explicit CLI arguments.

## Expected directory conventions

- `data/raw/meteo/`
  - station precipitation CSV files
- `data/raw/imerg_stations_3h/`
  - station-matched IMERG 3-hourly CSV files used during calibration
- `data/raw/era5land_stations_3h/`
  - optional ERA5-Land station-matched 3-hourly CSV files
- `data/raw/imerg_quarterly/imerg_30min_quarters.zip`
  - ZIP archive containing quarterly IMERG 30-min GeoTIFF stacks
- `data/external/era5_phase_masks/`
  - quarterly rain/snow masks aligned to the precipitation stacks
- `data/external/aws310_pluvio_v2.csv`
  - high-frequency AWS record used in sensitivity diagnostics
- `data/external/biomet_highfreq.csv`
  - optional second high-frequency record used in peak-model training
- `data/external/station_annual_pbias_summary_imerg.csv`
  - station-level validation summary used in uncertainty propagation

## What can be public

- code: yes
- manuscript figures generated from derived outputs: yes
- small summary tables without sensitive station information: usually yes

## What may need access restrictions

- raw station observations
- large intermediate rasters
- locally curated site records from operational archives
