"""Real-vs-synthetic marginal distribution plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import pandas as pd

__all__ = ["plot_marginals"]


def plot_marginals(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    output: Path | str,
    *,
    columns: Sequence[str] | None = None,
    ncols: int = 5,
    dpi: int = 300,
    show: bool = False,
):
    """Overlay real and synthetic kernel density estimates, one panel per column.

    Args:
        real: The training data, in original units.
        fake: Generated samples, in original units and with the same columns.
        output: Where to write the figure; the suffix picks the format.
        columns: Subset to plot.  Defaults to every column of ``real``.
        show: Call ``plt.show()`` after saving (for interactive use).

    Returns:
        The matplotlib ``Figure``.
    """
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    columns = list(columns if columns is not None else real.columns)
    # Decoded categoricals come back as labels; a KDE needs numbers.
    columns = [
        c for c in columns
        if pd.api.types.is_numeric_dtype(real[c])
        and pd.api.types.is_numeric_dtype(fake[c])
    ]
    if not columns:
        raise ValueError("no numeric columns in common between real and fake")
    nrows = -(-len(columns) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    # `axes` is padded out to a full grid, so it can be longer than `columns`.
    for ax, column in zip(axes, columns, strict=False):
        # Constant columns (`expPeri`, flags nobody in the cohort has) have no
        # density to draw; seaborn skips them, which is the wanted behaviour.
        sns.kdeplot(
            real[column], ax=ax, label="Real",
            color="blue", linewidth=2, warn_singular=False,
        )
        sns.kdeplot(
            fake[column], ax=ax, label="Synthetic", color="red",
            linestyle="--", linewidth=2, warn_singular=False,
        )
        ax.set_title(column, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    for ax in axes[len(columns):]:
        fig.delaxes(ax)

    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig
