"""Loading and preprocessing of the raw digital-twin parameter tables.

The pipeline mirrors what the original notebooks did, in this order:

1. map string categoricals to integer codes (and, for schemas that model
   them as categoricals, shift 1-based codes such as ``Sex`` to 0-based);
2. slice the leading ``n_rows x n_features`` block;
3. take ``log10`` of the wide-range resistance parameters;
4. KNN-impute missing values (and snap categorical codes back to valid ints);
5. min-max scale the continuous columns.

:func:`prepare` returns a :class:`PreparedData`, which carries the fitted
scaler and column layout so that generated samples can be mapped back to the
original units by :func:`inverse_transform`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from dtaug.schema import (
    CATEGORICAL_CARDINALITY,
    CATEGORY_CODES,
    ONE_BASED_COLUMNS,
    DatasetSchema,
)

__all__ = ["PreparedData", "prepare", "inverse_transform"]


def _encode_labels(values: pd.Series, codes: dict[str, int]) -> pd.Series:
    """Map string labels to integer codes, tolerating case and stray whitespace.

    The raw tables are hand-maintained and contain variants such as ``"yes"``
    alongside ``"Yes"``.  A plain ``Series.map`` turns every miss into NaN,
    which the imputer then silently invents a value for, so unmapped labels are
    warned about instead.
    """
    lookup = {label.strip().casefold(): code for label, code in codes.items()}
    cleaned = values.astype("string").str.strip().str.casefold()
    mapped = cleaned.map(lookup)

    unmapped = sorted(set(cleaned[cleaned.notna() & mapped.isna()]))
    if unmapped:
        warnings.warn(
            f"{values.name!r}: unrecognised label(s) {unmapped} treated as "
            f"missing; add them to CATEGORY_CODES in dtaug/schema.py",
            RuntimeWarning,
            stacklevel=3,
        )
    return mapped.astype("Float64").astype("float64")


@dataclass
class PreparedData:
    """A training matrix plus everything needed to undo the preprocessing.

    Attributes:
        X: Scaled feature matrix, shape ``(n_rows, n_features)``.  Continuous
            columns are in [0, 1]; categorical columns hold integer codes.
        raw: The unmodified slice of the CSV, for real-vs-fake comparisons.
        columns: Column names, in CSV order.
        continuous_idx: Positions of the continuous columns.
        categorical_idx: Positions of the categorical columns (empty unless the
            schema sets ``categorical=True``).
        categorical_dims: Cardinality of each categorical column, aligned with
            ``categorical_idx``.
        scaler: The :class:`MinMaxScaler` fitted on the continuous columns.
        log10_idx: Positions of the columns held in log10 space.
    """

    X: torch.Tensor
    raw: pd.DataFrame
    columns: list[str]
    continuous_idx: list[int]
    categorical_idx: list[int]
    categorical_dims: list[int]
    scaler: MinMaxScaler
    log10_idx: list[int]

    @property
    def continuous_dim(self) -> int:
        return len(self.continuous_idx)

    @property
    def model_input_size(self) -> int:
        """Width of the tensor the model sees (one-hot expands categoricals)."""
        return self.continuous_dim + sum(self.categorical_dims)

    def split(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Split ``X`` into a continuous block and a list of one-hot blocks."""
        continuous = self.X[:, self.continuous_idx]
        one_hots = [
            torch.nn.functional.one_hot(
                self.X[:, col].long(), num_classes=dim
            ).float()
            for col, dim in zip(self.categorical_idx, self.categorical_dims, strict=True)
        ]
        return continuous, one_hots


def prepare(schema: DatasetSchema, *, csv_path=None) -> PreparedData:
    """Read and preprocess a raw CSV according to ``schema``."""
    df = pd.read_csv(csv_path or schema.csv_path, header=0)
    raw = df.iloc[: schema.n_rows, : schema.n_features].copy()

    # String fields must become numeric whatever the schema; the 1-based shift
    # exists only to turn codes into valid one-hot indices, so it is tied to
    # categorical modelling and undone again by `inverse_transform`.
    for column, codes in CATEGORY_CODES.items():
        if column in df.columns:
            df[column] = _encode_labels(df[column], codes)
    if schema.categorical:
        for column in ONE_BASED_COLUMNS:
            if column in df.columns:
                df[column] = df[column] - 1

    data = df.iloc[: schema.n_rows, : schema.n_features].copy()
    columns = list(data.columns)
    index_of = {name: i for i, name in enumerate(columns)}

    log10_idx = [index_of[c] for c in schema.log10_columns if c in index_of]
    for col in log10_idx:
        values = data.iloc[:, col]
        n_bad = int((values <= 0).sum())
        if n_bad:
            warnings.warn(
                f"{columns[col]!r}: {n_bad} non-positive value(s) become "
                f"-inf/NaN under log10",
                RuntimeWarning,
                stacklevel=2,
            )
        data.iloc[:, col] = np.log10(values)

    if schema.categorical:
        categorical_idx = [
            index_of[c] for c in CATEGORICAL_CARDINALITY if c in index_of
        ]
        categorical_dims = [
            CATEGORICAL_CARDINALITY[columns[i]] for i in categorical_idx
        ]
    else:
        categorical_idx, categorical_dims = [], []
    continuous_idx = [i for i in range(len(columns)) if i not in set(categorical_idx)]

    if schema.impute:
        data = pd.DataFrame(
            KNNImputer(n_neighbors=5).fit_transform(data), columns=columns
        )
        # Imputation returns floats; categorical codes must stay valid indices.
        for col, dim in zip(categorical_idx, categorical_dims, strict=True):
            data.iloc[:, col] = (
                data.iloc[:, col].round().clip(lower=0, upper=dim - 1).astype(int)
            )
    elif data.isna().to_numpy().any():
        raise ValueError(
            f"{schema.csv} has missing values but {schema.name!r} does not "
            f"enable imputation; set impute=True on the schema."
        )

    scaler = MinMaxScaler()
    data.iloc[:, continuous_idx] = scaler.fit_transform(data.iloc[:, continuous_idx])

    return PreparedData(
        X=torch.tensor(data.to_numpy(dtype=np.float32), dtype=torch.float32),
        raw=raw,
        columns=columns,
        continuous_idx=continuous_idx,
        categorical_idx=categorical_idx,
        categorical_dims=categorical_dims,
        scaler=scaler,
        log10_idx=log10_idx,
    )


def inverse_transform(
    prep: PreparedData, scaled: np.ndarray, *, decode_categories: bool = True
) -> pd.DataFrame:
    """Map a generated matrix back to the units of the raw CSV.

    Undoes the min-max scaling on the continuous columns and the log10
    transform.  With ``decode_categories`` (the default) the categorical
    columns are also restored to how they appear in the CSV -- 1-based codes
    shifted back, string-valued fields turned back into their labels -- so a
    generated table is directly comparable with the input table.  Pass
    ``False`` to keep them as 0-based integer codes.
    """
    if scaled.shape[1] != len(prep.columns):
        raise ValueError(
            f"expected {len(prep.columns)} columns, got {scaled.shape[1]}"
        )

    out = np.asarray(scaled, dtype=np.float64).copy()
    out[:, prep.continuous_idx] = prep.scaler.inverse_transform(
        out[:, prep.continuous_idx]
    )
    for col in prep.log10_idx:
        out[:, col] = 10 ** out[:, col]

    frame = pd.DataFrame(out, columns=prep.columns)
    if not decode_categories:
        return frame

    for col in prep.categorical_idx:
        name = prep.columns[col]
        codes = frame[name].round().astype(int)
        if name in ONE_BASED_COLUMNS:
            frame[name] = codes + 1
        elif name in CATEGORY_CODES:
            labels = {code: label for label, code in CATEGORY_CODES[name].items()}
            frame[name] = codes.map(labels)
        else:
            frame[name] = codes
    return frame
