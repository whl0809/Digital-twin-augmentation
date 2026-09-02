"""Drawing synthetic patients from a trained model and returning them in the
original units and column order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtaug.data import PreparedData, inverse_transform
from dtaug.models import VAE, MixedVAE

__all__ = ["sample_vae", "sample_mixed_vae"]


def sample_vae(model: VAE, prep: PreparedData, num_samples: int) -> pd.DataFrame:
    """Sample from an all-continuous VAE and undo the preprocessing."""
    samples = model.sample(num_samples).cpu().numpy()
    return inverse_transform(prep, samples)


def sample_mixed_vae(
    model: MixedVAE,
    prep: PreparedData,
    num_samples: int,
    temperature: float = 0.1,
) -> pd.DataFrame:
    """Sample from a mixed VAE and undo the preprocessing.

    Continuous heads are inverse-scaled; categorical heads are collapsed from
    one-hot back to the integer codes used in the raw CSV, and each column is
    put back at its original position.
    """
    continuous, categoricals = model.sample(num_samples, temperature=temperature)
    continuous = continuous.cpu().numpy()
    codes = [c.argmax(dim=1).cpu().numpy() for c in categoricals]

    combined = np.empty((num_samples, len(prep.columns)), dtype=np.float64)
    combined[:, prep.continuous_idx] = continuous
    for position, column_codes in zip(prep.categorical_idx, codes, strict=True):
        combined[:, position] = column_codes

    return inverse_transform(prep, combined)
