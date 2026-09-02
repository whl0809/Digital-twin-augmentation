"""Optional physics-informed penalty backed by the TriSeg MATLAB model.

``matlab/simulate_from_python.m`` runs a decoded parameter vector through the
TriSeg circulation model and scores how far the resulting hemodynamics fall
outside physiologically plausible ranges.  Feeding that score back into
training is what makes the VAE "physics-informed" (PINN) -- generated patients
are pushed towards parameter sets that actually simulate.

This lives behind :class:`MatlabPenalty` because it needs a MATLAB install with
the Python engine plus a checkout of the TriSeg digital-twin model, neither of
which is a dependency of the rest of the package.  In the original notebooks
this code was commented out; it is kept here so it can be switched on without
being reconstructed from scratch.

Usage::

    penalty = MatlabPenalty(triseg_path="/path/to/TriSeg-Digital-Twins")
    model = train_vae(prep.X, ...)          # unchanged
    score = penalty(decoded_original_units)  # float, 0.0 == fully plausible

The penalty is not differentiable, so it cannot simply be added to the loss and
backpropagated.  Use it to score or filter generated cohorts, or wire it into a
score-function/REINFORCE-style estimator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["MatlabPenalty", "MATLAB_COLUMNS"]

#: Column order `simulate_from_python.m` expects, per its own `colNames`.
MATLAB_COLUMNS = [
    "HR", "C_SA", "C_SV", "C_PA", "C_PV", "R_SA", "R_tSA", "R_PA", "R_tPA",
    "R_SV", "R_t_c", "R_p_o", "R_p_c", "R_m_o", "R_m_c", "R_a_o", "R_a_c",
    "k_pas_LV", "k_pas_RV", "k_act_LV", "k_act_RV",
    "Amref_LV", "Amref_SEP", "Amref_RV", "Vw_LV", "Vw_SEP", "Vw_RV",
    "RAV0u", "LAV0u", "LAV0c", "K1", "Vh0", "Height", "Weight", "Sex",
]

FAILURE_PENALTY = 1e3


class MatlabPenalty:
    """Score parameter vectors by simulating them in MATLAB.

    Args:
        triseg_path: Directory holding ``ProcessParamsFromVAE`` and
            ``runSimonFakeDT``.
        matlab_dir: Directory holding ``simulate_from_python.m``; defaults to
            the ``matlab/`` folder of this repository.
        failure_penalty: Score returned when a simulation errors out.
    """

    def __init__(
        self,
        triseg_path: str | Path,
        matlab_dir: str | Path | None = None,
        failure_penalty: float = FAILURE_PENALTY,
    ):
        try:
            import matlab.engine
        except ImportError as exc:  # pragma: no cover - needs a MATLAB install
            raise ImportError(
                "MatlabPenalty needs the MATLAB Engine API for Python; see "
                "https://www.mathworks.com/help/matlab/matlab-engine-for-python.html"
            ) from exc

        self._matlab = matlab
        self.failure_penalty = failure_penalty
        self.engine = matlab.engine.start_matlab()
        self.engine.addpath(str(Path(triseg_path).resolve()), nargout=0)
        matlab_dir = matlab_dir or Path(__file__).resolve().parent.parent / "matlab"
        self.engine.addpath(str(Path(matlab_dir).resolve()), nargout=0)

    def score_row(self, row: np.ndarray) -> float:
        """Score one parameter vector, ordered as :data:`MATLAB_COLUMNS`."""
        try:
            vector = self._matlab.double(np.asarray(row, dtype=float).tolist())
            return float(self.engine.simulate_from_python(vector))
        except Exception as exc:  # pragma: no cover - MATLAB-side failures
            print(f"simulate_from_python failed: {exc}")
            return self.failure_penalty

    def __call__(self, rows) -> float:
        """Total penalty over a batch of parameter vectors in original units."""
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        return float(sum(self.score_row(row) for row in rows))

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
