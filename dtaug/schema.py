"""Column layout of the digital-twin parameter tables.

The raw CSVs are wide and positional: the notebooks referred to columns by bare
index (`[32, 35, 38, ...]`).  Everything positional is collected here so the
rest of the package can stay index-free, and so a column reordering in the CSV
only has to be fixed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
GENERATED_DIR = REPO_ROOT / "outputs" / "generated"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures"

#: Resistance/compliance parameters spanning several orders of magnitude.
#: Modelled in log10 space so the min-max scaler does not collapse them.
LOG10_COLUMNS = ["R_t_c", "R_p_o", "R_p_c", "R_m_c", "R_a_o", "R_a_c"]

#: String-valued clinical fields, mapped to 0-based integer codes.
CATEGORY_CODES = {
    "HFtype": {"HFpEF": 0, "HFrEF": 1, "PH-Suspected": 2},
    "Smoking": {"Never": 0, "Former": 1, "Current": 2},
    "Alcohol": {"No": 0, "Yes": 1},
    "Drug": {"No": 0, "Yes": 1},
}

#: Numeric fields stored 1-based in the CSV that need shifting to 0-based codes.
ONE_BASED_COLUMNS = ["Sex", "NYHAClass"]

#: Binary yes/no comorbidity and medication flags (contiguous block in the CSV).
_FLAG_COLUMNS = [
    "UsageOfERA", "UsageOfCCB", "UsageOfSGLT2", "UsageOfStatin", "UsageOfAsprin",
    "UsageOfNitrates", "UsageOfDiuretic", "O2Supp", "UsageOfBetaBlocker",
    "UsageOfACEi_ARB", "HTN", "PHD", "PH", "CAD", "SHD", "MI", "HCM", "NCCM",
    "RCM", "DCM", "Amyloidosis", "ValvularDisease", "ValveRepair",
    "PericardiumDisease", "Pericardiectomy", "SeptalMyomectomy", "AF",
    "OtherArrhythmia", "HLD", "DM", "COPD", "CTD", "HTh", "LTh", "ICD", "CRT",
    "Pacemaker", "LiverFailure", "RenalFailure", "RespiratoryFailure", "AD_AA",
    "HeartTransplant", "KidneyTransplant", "LiverTransplant", "LungTransplant",
    "PancreasTransplant", "ASD", "VSD", "OtherShunt",
]

#: Discrete columns and their cardinality, for the mixed continuous/categorical
#: model.  Order fixes the order of the decoder's categorical heads.
CATEGORICAL_CARDINALITY = {
    "Sex": 2,
    "HFtype": 3,
    "Smoking": 3,
    "Alcohol": 2,
    "Drug": 2,
    "NYHAClass": 4,
    "FirstDiagnosisOfHeartFailureInThePast18Months": 2,
    **{name: 2 for name in _FLAG_COLUMNS},
    "EndEvent": 2,
}


@dataclass(frozen=True)
class DatasetSchema:
    """One way of reading a raw CSV into a training matrix.

    Attributes:
        name: Short identifier, used for default output filenames.
        csv: Raw CSV, relative to ``data/raw``.
        n_features: Number of leading columns kept from the CSV.
        n_rows: Number of leading rows kept (the tables have trailing blanks).
        categorical: Whether discrete columns get their own softmax heads.  When
            False every column is treated as continuous and min-max scaled.
        impute: Whether to KNN-impute missing values before scaling.
    """

    name: str
    csv: str
    n_features: int
    n_rows: int = 248
    categorical: bool = False
    impute: bool = False
    log10_columns: list[str] = field(default_factory=lambda: list(LOG10_COLUMNS))

    @property
    def csv_path(self) -> Path:
        return RAW_DIR / self.csv


#: 35 hemodynamic/anatomical parameters only; complete table, no imputation.
CORE35 = DatasetSchema(
    name="core35",
    csv="input_VAE_final.csv",
    n_features=35,
)

#: All 149 parameters, discrete fields treated as ordinary continuous values.
ALL149 = DatasetSchema(
    name="all149",
    csv="input_VAE_allpara.csv",
    n_features=149,
    impute=True,
)

#: All 149 parameters with the 57 discrete fields modelled as categoricals.
MIXED149 = DatasetSchema(
    name="mixed149",
    csv="input_VAE_allpara.csv",
    n_features=149,
    categorical=True,
    impute=True,
)

SCHEMAS = {s.name: s for s in (CORE35, ALL149, MIXED149)}
