# Reproducibility Notes

This repository was extracted from the working research codebases and normalized for publication.

## Main design choices

- All workflow scripts use repository-relative defaults or explicit CLI arguments.
- Absolute developer-specific paths were removed.
- No data files are tracked in the repository.
- Tests were copied with the scientific core logic so that readers can verify the main numerical components independently from the full raster workflow.

## Recommended archival strategy

For manuscript submission, the most robust setup is:

1. publish this repository on GitHub
2. create a Zenodo archive for the exact tagged release
3. separately archive non-public or large derived outputs if journal policy allows

## Remaining non-code dependencies

- calibration station data
- IMERG quarterly rasters
- ERA5 phase masks
- optional teleconnection downloads from NOAA PSL
- optional high-frequency site series used for sub-bin diagnostics
