# Outputs

- `reference/` — artefacts committed before the refactor, produced by the
  notebooks now in `notebooks/archive/`. Kept as a baseline to compare new runs
  against. `Fakedata_PDF*.png` are real-vs-synthetic marginal plots;
  `generated_fake_data*.csv` are 100-patient cohorts over the 149-column layout.
- `generated/` — CSVs written by `scripts/generate.py` (git-ignored).
- `figures/` — plots written by `scripts/generate.py` (git-ignored).
