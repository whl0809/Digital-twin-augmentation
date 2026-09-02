# Notebooks

`archive/` holds the three original exploratory notebooks, kept for provenance
with their outputs stripped. Their logic now lives in the `dtaug` package and
is reachable from `scripts/generate.py`; they are **not** maintained and are not
expected to run as-is (they read CSVs from the old flat repository layout).

| Archived notebook | Superseded by |
| --- | --- |
| `01_core35_vae.ipynb` (was `NNtest_original.ipynb`) | `dtaug.schema.CORE35` |
| `02_all149_vae.ipynb` (was `NNtest._allfeatures.ipynb`) | `dtaug.schema.ALL149` |
| `03_mixed149_vae.ipynb` (was `NNtest_allfeatureswithcategorical.ipynb`) | `dtaug.schema.MIXED149` |

To work interactively against the current code, use `demo.ipynb`.
