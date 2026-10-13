# compound-heat-drought

Code for:

> Bhattarai, S., Pradhan, N. R., and Talchabhadel, R. *Compound heatwave and drought hazards accelerate fastest in drylands while human exposure rises.*
> (submitted to npj Natural Hazards)

Archived code (this release, v1.0.0): Zenodo, doi:[10.5281/zenodo.23352920](https://doi.org/10.5281/zenodo.23352920).
Derived data: Zenodo, doi:[10.5281/zenodo.23350013](https://doi.org/10.5281/zenodo.23350013).
Both Zenodo records become public when the paper is published.

The code builds monthly heatwave, drought and compound heatwave–drought flags on the
global 0.25° ERA5 grid for 1950–2025. It then computes per-pixel trends and produces
every number, table and figure in the paper.

## Definition used in the paper

| | |
|---|---|
| Heatwave month | ≥1 day in a run of ≥3 consecutive days with daily T<sub>max</sub> > T90(day of year). T90 is the 90th percentile of 1961–1990 T<sub>max</sub> in a ±15-day window. |
| Drought month | SPEI-1 ≤ −1. SPEI uses D = P − PET, with PET from FAO-56 Penman–Monteith. The log-logistic (GLO) distribution is fitted by L-moments per calendar month over 1950–2025. |
| Compound month | heatwave month AND drought month |
| Trends | Annual event-month counts per pixel → Mann–Kendall (tie-corrected) + Theil–Sen |

`config.py` also supports other HW baselines and SPEI calibrations and timescales. The
paper uses only the combination above.

## Layout

```
config.py                 every path, period and method constant (single source of truth)
src/chd/                  library
  daily, deaccum          L1  6-hourly ERA5 -> daily fields
  pet, water_balance      L2  FAO-56 PET, monthly P, PET, D
  spei                    L3  SPEI-1/3/6/12
  thresholds, heatwave    L4  T90(doy) and heatwave metrics
  compound                L5  monthly compound flags
  trends                  L6  Mann–Kendall + Theil–Sen per pixel
  drylands, regions,      ancillary layers: aridity index and Köppen, IPCC AR6 regions,
  agriculture, population,  MIRCA2000 cropland, LandScan population, RONI (ENSO),
  enso, leading             heat- vs drought-leading onset order
  manuscript_figures      all results tables (figures/manuscript/tables/*.csv)
  nature_figures          the paper's figures (cache stage + drawing stage)
scripts/01..06            pipeline stages L1-L6
scripts/11                tables + ancillary layers
scripts/12                paper figures
scripts/13                export of the Zenodo deposit
analysis/robustness       aridity-class acceleration, P vs PET variance share, AI robustness
analysis/sampling_validation  6-hourly vs 24-hourly ERA5 check at 120 cells (Supplementary Fig. 1)
analysis/finalization_checks  South Asia per-person trend since 1979; SPEI fit vs lmoments3
tests/                    plain-assert unit tests: python tests/test_*.py
```

Each stage reads the previous stage's Zarr store and writes its own. Stages skip
outputs that already exist, so every script is safe to re-run.

## Install

```bash
conda create -n chd python=3.12 && conda activate chd
pip install -r requirements.txt
python tests/test_pet.py && python tests/test_spei.py && python tests/test_l4_kernels.py && python tests/test_leading.py
```

## Reproduce

### A. Figures from the archived data (minutes, laptop)

Download the Zenodo deposit, then:

```bash
mkdir -p figures/manuscript/nature
cp -r <zenodo>/tables              figures/manuscript/tables
cp -r <zenodo>/figure_source_data  figures/manuscript/nature/cache
python scripts/12_nature_figures.py          # -> figures/manuscript/nature/*.pdf and figures/paper/
```

### B. Full pipeline from ERA5 (days; large-memory machine)

1. **ERA5 input.** Download one NetCDF per month, named `era5_daily_YYYY_MM.nc`, to
   `$CHD_ERA5_DIR`. Each file holds 6-hourly (00/06/12/18 UTC) fields for 1950–2025:
   `t2m, d2m, u10, v10, sp` (instantaneous) and `tp, ssrd, strd` (accumulated).
   The source is ERA5 single levels on the Copernicus Climate Data Store.
2. **Output location.** `export CHD_DATA_ROOT=/path/with/~600GB/free` (the L1 daily stores alone are ~485 GB).
3. **Ancillary inputs.** These go in `data_aux/`, next to `config.py`. The Köppen–Geiger
   map (Beck et al. 2018), MIRCA2000 and RONI download automatically on first use.
   LandScan Global 2000 and 2024 need a free ORNL account: download them from
   https://landscan.ornl.gov/ into `data_aux/population/` (see `src/chd/population.py`).
4. Run:

```bash
./run_all.sh            # L1 -> L6, tables, figure cache; logs to $CHD_DATA_ROOT/logs
python scripts/12_nature_figures.py
```

Resource notes. The pipeline is I/O-bound, and `config.py` worker counts are tuned
for spinning disks. L1 reads the whole ERA5 archive (~1.5 TB). L4 holds the
full-record daily T<sub>max</sub> in memory (~110 GB).

## Known limitations

- Daily T<sub>max</sub> comes from 4 synoptic samples, which biases it low. Heatwave
  detection is still internally consistent because the threshold and the values share
  the same sampling. `analysis/sampling_validation/` reruns the pipeline at 120 cells
  with all 24 hours, and the trends agree (Supplementary Fig. 1).
- Precipitation is reconstructed from 4 of 24 hourly accumulations. It is unbiased but
  noisy.
- Ocean cells are computed but have no meaning. Every statistic uses the analysis land
  mask (land excluding Antarctica).

## Licence

Code: MIT (see `LICENSE`). ERA5-derived data: Copernicus licence. Credit: Copernicus
Climate Change Service.
