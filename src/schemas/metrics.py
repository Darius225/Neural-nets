"""Pydantic schema for forecast quality metrics.

Pydantic (rather than ``@dataclass``) here because the metrics are
routinely serialised — appended to CSV reports, dumped as JSON
artefacts from experiment scripts. ``.model_dump_json()`` works out
of the box. Field bounds (``mae >= 0``, ``mape >= 0``,
``out_of_train_range_pct in [0, 100]``) also guard against accidental
sign errors in metric computation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionMetrics(BaseModel):
    """Bundle of regression + trading + risk metrics for a forecast.

    All errors are in the price units (e.g. USD); ``mape`` and
    ``directional_accuracy`` are percentages. Instances are frozen —
    a metrics snapshot semantically can't be mutated after the report
    it was written to is on disk.
    """
    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=0)
    mae: float = Field(ge=0)
    rmse: float = Field(ge=0)
    mape: float = Field(ge=0)
    r2: float
    directional_accuracy: float
    mean_signed_error: float
    worst_day_error: float = Field(ge=0)
    skill_vs_persistence: Optional[float] = None
    out_of_train_range_pct: Optional[float] = Field(default=None, ge=0, le=100)

    def as_row(self) -> dict:
        return {
            "n": self.n,
            "MAE": round(self.mae, 4),
            "RMSE": round(self.rmse, 4),
            "MAPE%": round(self.mape, 3),
            "R2": round(self.r2, 4),
            "DirAcc%": round(self.directional_accuracy, 2),
            "MeanSignedErr": round(self.mean_signed_error, 4),
            "WorstDayErr": round(self.worst_day_error, 4),
            "Skill_vs_persist": None if self.skill_vs_persistence is None
                else round(self.skill_vs_persistence, 4),
            "OutOfRange%": None if self.out_of_train_range_pct is None
                else round(self.out_of_train_range_pct, 2),
        }
