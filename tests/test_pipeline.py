"""End-to-end checks over the preprocessing / training / sampling pipeline.

These run a couple of epochs on the real CSVs -- enough to catch shape,
column-order and round-trip mistakes without being a training run.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtaug import data, sampling, schema, train  # noqa: E402
from dtaug.postprocess import expand_to_full_format  # noqa: E402


@pytest.fixture(scope="module")
def mixed():
    return data.prepare(schema.MIXED149)


def test_schema_matches_notebook_indices():
    """The notebooks hard-coded these positions; the names must resolve to them."""
    columns = list(pd.read_csv(schema.ALL149.csv_path, nrows=0).columns)
    index_of = {name: i for i, name in enumerate(columns)}

    assert sorted(index_of[c] for c in schema.LOG10_COLUMNS) == [10, 11, 12, 14, 15, 16]
    assert sorted(index_of[c] for c in schema.CATEGORICAL_CARDINALITY) == (
        [32, 35, 38, 39, 40, 44] + list(range(46, 96)) + [148]
    )


def test_mixed_dimensions(mixed):
    assert mixed.X.shape == (248, 149)
    assert mixed.continuous_dim == 92
    assert len(mixed.categorical_idx) == 57
    assert mixed.model_input_size == 210  # the notebooks' hard-coded input_size


@pytest.mark.parametrize("name", sorted(schema.SCHEMAS))
def test_prepare_produces_a_finite_scaled_matrix(name):
    prep = data.prepare(schema.SCHEMAS[name])
    assert prep.X.shape[1] == schema.SCHEMAS[name].n_features
    assert torch.isfinite(prep.X).all()
    continuous = prep.X[:, prep.continuous_idx]
    assert continuous.min() >= 0.0 and continuous.max() <= 1.0


def test_categorical_codes_are_valid_indices(mixed):
    for col, dim in zip(mixed.categorical_idx, mixed.categorical_dims, strict=True):
        values = mixed.X[:, col]
        assert values.min() >= 0 and values.max() <= dim - 1
        assert torch.equal(values, values.round())


def test_split_widths_match_the_model(mixed):
    continuous, one_hots = mixed.split()
    assert continuous.shape[1] == mixed.continuous_dim
    assert [h.shape[1] for h in one_hots] == mixed.categorical_dims
    for one_hot in one_hots:
        assert torch.equal(one_hot.sum(dim=1), torch.ones(len(one_hot)))


def test_inverse_transform_round_trips(mixed):
    """Feeding the scaled training matrix back must recover the raw values."""
    recovered = data.inverse_transform(
        mixed, mixed.X.numpy(), decode_categories=False
    )
    original = data.prepare(schema.MIXED149).raw

    for name in schema.LOG10_COLUMNS:
        np.testing.assert_allclose(
            recovered[name].to_numpy(), original[name].to_numpy(), rtol=1e-4
        )
    np.testing.assert_allclose(
        recovered["HR"].to_numpy(), original["HR"].to_numpy(), rtol=1e-5
    )


def test_decoded_categories_match_the_raw_vocabulary(mixed):
    decoded = data.inverse_transform(mixed, mixed.X.numpy())
    assert set(decoded["Sex"].unique()) <= {1, 2}
    assert set(decoded["NYHAClass"].unique()) <= {1, 2, 3, 4}
    assert set(decoded["HFtype"].dropna().unique()) <= set(
        schema.CATEGORY_CODES["HFtype"]
    )


def test_label_encoding_is_case_insensitive():
    """The raw table mixes 'Yes' and 'yes'; neither may become a missing value."""
    encoded = data._encode_labels(
        pd.Series(["Yes", "yes", " No "], name="Alcohol"),
        schema.CATEGORY_CODES["Alcohol"],
    )
    assert encoded.tolist() == [1.0, 1.0, 0.0]


def test_core35_does_not_shift_one_based_codes():
    """`Sex` is continuous in core35, so it must stay in raw 1/2 coding."""
    prep = data.prepare(schema.CORE35)
    recovered = data.inverse_transform(prep, prep.X.numpy())
    assert set(recovered["Sex"].round().unique()) <= {1.0, 2.0}


def test_train_and_sample_continuous():
    prep = data.prepare(schema.CORE35)
    model = train.train_vae(prep.X, num_epochs=2, on_epoch=lambda *_: None)
    fake = sampling.sample_vae(model, prep, num_samples=7)
    assert fake.shape == (7, 35)
    assert list(fake.columns) == prep.columns
    assert np.isfinite(fake.to_numpy(dtype=float)).all()


def test_train_and_sample_mixed(mixed):
    continuous, one_hots = mixed.split()
    model = train.train_mixed_vae(
        continuous, one_hots, num_epochs=2, on_epoch=lambda *_: None
    )
    fake = sampling.sample_mixed_vae(model, mixed, num_samples=7)

    assert fake.shape == (7, 149)
    assert list(fake.columns) == mixed.columns
    assert set(fake["Sex"].unique()) <= {1, 2}
    for col in mixed.categorical_idx:
        name = mixed.columns[col]
        if name in schema.CATEGORY_CODES:
            continue
        codes = fake[name]
        assert codes.min() >= 0


def test_early_stopping_triggers():
    prep = data.prepare(schema.CORE35)
    seen: list[int] = []
    train.train_vae(
        prep.X,
        num_epochs=50,
        stop_below=1e9,  # always true, so it must stop after one epoch
        on_epoch=lambda epoch, *_: seen.append(epoch),
    )
    assert seen == [1]


def test_missing_values_without_imputation_are_rejected():
    strict = replace(schema.ALL149, impute=False)
    with pytest.raises(ValueError, match="missing values"):
        data.prepare(strict)


def test_expand_to_full_format():
    prep = data.prepare(schema.MIXED149)
    fake = data.inverse_transform(prep, prep.X.numpy())
    reference = pd.read_csv(schema.RAW_DIR / "input_VAE.csv")

    full = expand_to_full_format(fake, reference)
    assert list(full.columns) == list(reference.columns)
    assert len(full) == len(fake)
    np.testing.assert_allclose(full["T"], 60.0 / full["HR"])
    np.testing.assert_allclose(full["R_PV"], full["R_SV"])
    np.testing.assert_allclose(full["RAV0c"], full["LAV0c"])
    assert full["tau_TS"].notna().all()


def test_inverse_transform_rejects_wrong_width(mixed):
    with pytest.raises(ValueError, match="expected 149 columns"):
        data.inverse_transform(mixed, np.zeros((3, 10)))
