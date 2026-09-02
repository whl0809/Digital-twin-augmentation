"""Expand a generated cohort into the full simulator input format.

The VAE only models the parameters it was trained on.  ``input_VAE.csv``
carries the wider layout the downstream simulator reads, in which some columns
are fixed per-cohort and others are algebraic functions of generated columns.
:func:`expand_to_full_format` fills those in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["DERIVED_COLUMNS", "expand_to_full_format"]

#: Columns taken verbatim from the reference table rather than generated.
CARRIED_OVER = [
    "tau_TS", "tau_TR", "R_Veins", "R_t_o", "V0u_coeff", "LEa", "REa",
    "V0c_coeff", "LAV1c", "RAV1c", "LEp", "REp", "Pc",
]

#: Columns computed from other columns, as ``name -> f(frame) -> Series``.
DERIVED_COLUMNS = {
    # Cycle length is the reciprocal of heart rate, in seconds.
    "T": lambda d: 60.0 / d["HR"],
    # Pulmonary venous resistance is set equal to systemic venous resistance.
    "R_PV": lambda d: d["R_SV"],
    # Septum-to-free-wall volume ratio.
    "LvSepR": lambda d: d["Vw_LV"] / (d["Vw_LV"] + d["Vw_SEP"]).replace(0, np.nan),
    # Right atrial contracted volume mirrors the left.
    "RAV0c": lambda d: d["LAV0c"],
}


def expand_to_full_format(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    carried_over: list[str] | None = None,
) -> pd.DataFrame:
    """Build a simulator-ready table from a generated cohort.

    Args:
        generated: Synthetic samples in original units.
        reference: A real table in the target layout (e.g. ``input_VAE.csv``);
            its column order defines the output and supplies ``carried_over``.
        carried_over: Columns copied from ``reference``.  Defaults to
            :data:`CARRIED_OVER`.

    Returns:
        A frame with ``reference``'s columns and ``generated``'s row count.
        Columns that are neither generated, carried over, nor derived are NaN.
    """
    carried_over = CARRIED_OVER if carried_over is None else carried_over
    n_rows = len(generated)

    out = pd.DataFrame(index=range(n_rows))
    for column in reference.columns:
        if column in generated.columns:
            out[column] = generated[column].to_numpy()
        elif column in carried_over and column in reference.columns:
            # Recycle the reference values if the cohort is longer than it.
            values = reference[column].to_numpy()
            out[column] = np.resize(values, n_rows) if len(values) else np.nan
        else:
            out[column] = np.nan

    for column, formula in DERIVED_COLUMNS.items():
        if column in out.columns:
            out[column] = formula(out)

    return out[list(reference.columns)]
