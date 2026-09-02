# Digital-twin augmentation

Variational autoencoders that learn the joint distribution of **cardiovascular
digital-twin parameters** from a real patient cohort, then sample synthetic
patients from the latent space. The synthetic cohorts are meant to augment a
small clinical dataset (248 patients) for downstream TriSeg circulation
simulations.

Three model variants are supported, from the narrowest to the most complete:

| Variant | Source table | Features | Discrete fields |
| --- | --- | --- | --- |
| `core35` | `input_VAE_final.csv` | 35 hemodynamic / anatomical parameters | treated as continuous |
| `all149` | `input_VAE_allpara.csv` | 149 parameters incl. labs, comorbidities, medications | treated as continuous |
| `mixed149` | `input_VAE_allpara.csv` | same 149 | 57 modelled as categoricals |

`mixed149` is the one to use. It gives each discrete clinical field
(sex, NYHA class, HF type, 49 comorbidity/medication flags, …) its own
Gumbel-softmax decoder head, so a generated patient has *"diabetes: yes"*
rather than *"diabetes: 0.63, rounded"*.

## Quick start

```bash
pip install -r requirements.txt          # or: pip install -e ".[dev]"

python scripts/generate.py mixed149 --epochs 300 --samples 100
```

That trains the model, writes `outputs/generated/mixed149_synthetic.csv`, and
saves a real-vs-synthetic marginal comparison to
`outputs/figures/mixed149_marginals.png`.

```
usage: dtaug-generate [-h] [--epochs N] [--batch-size N] [--latent-dim N]
                      [--learning-rate LR] [--kl-weight W] [--samples N]
                      [--temperature T] [--stop-below LOSS] [--seed N]
                      [--device DEVICE] [--out PATH] [--figure PATH] [--no-plot]
                      {all149,core35,mixed149}
```

`notebooks/demo.ipynb` walks through the same pipeline step by step.

## How it works

### Preprocessing (`dtaug/data.py`)

1. **Label encoding** — `HFtype`, `Smoking`, `Alcohol`, `Drug` are strings in
   the CSV and become integer codes. Matching is case- and
   whitespace-insensitive; anything unrecognised raises a warning instead of
   silently turning into a missing value.
2. **Log transform** — six resistance parameters (`R_t_c`, `R_p_o`, `R_p_c`,
   `R_m_c`, `R_a_o`, `R_a_c`) span five or more orders of magnitude and are
   modelled in log10 space, so min-max scaling does not crush them into a
   sliver of [0, 1].
3. **Imputation** — the 149-column table is sparse; missing values are filled
   by `KNNImputer(n_neighbors=5)`. Categorical codes are then rounded and
   clipped back to valid indices.
4. **Scaling** — `MinMaxScaler` on the continuous columns only.

`prepare()` returns a `PreparedData` holding the scaled matrix, the untouched
raw slice, the fitted scaler and the column layout, so `inverse_transform()`
can map generated samples all the way back to the units of the input CSV —
including turning categorical codes back into `"HFpEF"` / `"Yes"` / `Sex = 2`.

### Models (`dtaug/models.py`)

All variants share an encoder trunk of `input → 256 → latent(16)`.

- `VAE` — decoder `16 → 256 → 128 → input`, Gaussian latent, MSE + KL loss.
- `MixedVAE` — shared decoder trunk, then a linear head for the 92 continuous
  columns and one softmax head per categorical column. The model input is the
  continuous block concatenated with every one-hot block: 92 + 118 = **210**.
  Loss is MSE + one cross-entropy per head + KL.
- `ConditionalVAE` — label-conditioned variant, carried over from the original
  notebooks; not wired into the CLI.

Categorical heads use the Gumbel-softmax trick: soft samples during training
(so gradients flow), hard one-hot samples at generation time.

### Physics-informed penalty (`dtaug/physics.py`)

`matlab/simulate_from_python.m` runs a parameter vector through the TriSeg
circulation model and scores how far the resulting hemodynamics (SBP, LVEDV,
cardiac output, …) fall outside physiologically plausible ranges. `MatlabPenalty`
wraps it so generated cohorts can be scored or filtered for plausibility.

This is **optional and off by default** — it needs a MATLAB install with the
Python engine plus a checkout of the TriSeg digital-twin model. The penalty is
non-differentiable, so it cannot simply be added to the loss and
backpropagated; use it to score or filter cohorts, or as the reward in a
score-function estimator.

### Expanding to simulator input (`dtaug/postprocess.py`)

The VAE only models the columns it was trained on. `input_VAE.csv` has the
wider 52-column layout the simulator reads, where some columns are fixed per
cohort and others are algebraic functions of generated columns
(`T = 60 / HR`, `LvSepR = Vw_LV / (Vw_LV + Vw_SEP)`, …).
`expand_to_full_format()` fills those in. Columns that are neither generated,
carried over, nor derived — currently just `Vh0` — come out as NaN and must be
supplied before simulating.

## Layout

```
dtaug/              the package
  schema.py         column groups, categorical cardinalities, dataset variants
  data.py           loading, preprocessing, inverse transform
  models.py         AutoEncoder / VAE / MixedVAE / ConditionalVAE
  losses.py         KL, Gaussian ELBO, mixed ELBO
  train.py          training loops
  sampling.py       latent sampling -> original units
  plotting.py       real-vs-synthetic marginal grids
  postprocess.py    expand a cohort to the simulator's input format
  physics.py        optional MATLAB/TriSeg plausibility penalty
  cli.py            argument parsing and orchestration
scripts/generate.py runnable wrapper for a source checkout
matlab/             simulate_from_python.m
data/raw/           input CSVs (see below)
notebooks/          demo.ipynb, plus the original notebooks under archive/
outputs/            reference/ (pre-refactor artefacts), generated/, figures/
tests/              pytest suite
```

### Input tables

| File | Rows × cols | Used by |
| --- | --- | --- |
| `input_VAE_allpara.csv` | 408 × 149 | `all149`, `mixed149` (first 248 rows) |
| `input_VAE_final.csv` | 248 × 35 | `core35` |
| `input_VAE.csv` | 64 × 52 | reference layout for `expand_to_full_format` |
| `input_VAE_cleaned.csv` | 31 × 35 | earlier revision, unused |
| `input_VAE_modified.csv` | 64 × 35 | earlier revision, unused |

Only the first 248 rows are used for training; later rows are incomplete. Set
`n_rows` on a `DatasetSchema` to change that.

## Tests

```bash
pytest
```

The suite runs a couple of epochs on the real CSVs and checks shapes, column
order, categorical validity, the preprocessing round-trip, and the derived
columns in `expand_to_full_format`. It also asserts that the column *names* in
`dtaug/schema.py` resolve to the same positions the original notebooks
hard-coded, so a column reordering in a CSV fails loudly.

<!--
## Differences from the original notebooks

The refactor is behaviour-preserving apart from these deliberate changes:

- **Losses are per-sample.** The notebooks summed squared error over the batch
  and then multiplied by the batch size again when accumulating; losses are now
  averaged per sample, so reported values are smaller and comparable across
  batch sizes. Adam makes the training dynamics essentially unchanged.
- **Early stopping is off by default.** The notebooks stopped at hard-coded
  thresholds (`loss < 5`, `loss < 20`) tuned to the old loss scale. Pass
  `--stop-below` to re-enable it.
- **Categorical columns round-trip.** Generated CSVs now come back in the
  input's own coding (`Sex ∈ {1,2}`, `HFtype = "HFrEF"`), not as 0-based codes.
- **The 1-based shift is tied to categorical modelling.** `Sex` and `NYHAClass`
  are shifted to 0-based only when they are actually one-hot encoded, so
  `core35` and `all149` no longer see a spurious offset.
- **Case-insensitive label matching.** The raw table contains both `"Yes"` and
  `"yes"` for `Alcohol`; the notebooks' map only had `"Yes"`, so those rows
  became missing and were imputed.
- **Column indices are named.** `[32, 35, 38, 39, 40, 44] + range(46, 96) + [148]`
  is now a name→cardinality mapping in `dtaug/schema.py`, verified by a test.
- **The MNIST download and `.ipynb_checkpoints` are gone.** 64 MB of torchvision
  demo data left over from a commented-out cell; still in git history.
-->
