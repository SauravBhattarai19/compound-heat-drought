# Verification checks, 2026-10-13 (finalization round)

| Script | Output (2026-10-13) | Used in manuscript |
|---|---|---|
| `south_asia_perperson.py` | 1950–2025: −0.069/dec, Hamed–Rao p=0.029; 1979–2025: −0.045/dec, p=0.271; means 1.35 (1950–79), 1.02 (1979–95), 0.86 (1996–2025) | Results, "Human exposure" (South Asia since 1979) |
| `spei_vs_lmoments3.py` | 480 cell-months, max \|ΔSPEI\| 2.23e-05, median 3.8e-07 | Methods, "Reference evapotranspiration and SPEI-1" |

Ad-hoc country check (Natural Earth 110m countries on the ERA5 grid, LandScan 2024 weights): per-person
compound frequency 1950–79 → 1996–2025: India 1.36 → 0.86, Pakistan 1.64 → 0.80; share of population in
pixels with a significant per-pixel decrease: India 35%, Pakistan 61%. Share of the global significantly
decreasing land area that lies in AR6 South Asia: 51%. (Quoted in words only, not as numbers, in the text.)

Citation check: Ridder et al. 2020 does not identify Sub-Saharan Africa/Indus "clear, dry" hotspots; it places
heatwave–dry-condition hotspots at midlatitudes (1980–2014). The Introduction sentence was corrected.
