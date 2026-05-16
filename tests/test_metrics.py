"""Tests for src.metrics — the most critical module to get right.

If skill_score / directional_accuracy / persistence baseline have bugs,
every experiment result in the repo is suspect. These tests check the
math directly against hand-computed expectations on small arrays.
"""

import math

import numpy as np
import pytest

from src.metrics import (
    compute_metrics,
    directional_accuracy,
    naive_persistence_forecast,
    skill_score,
)


class TestPersistenceForecast:
    def test_returns_yesterdays_close_unchanged(self):
        yprev = np.array([100.0, 101.5, 99.8, 102.0])
        out = naive_persistence_forecast(yprev)
        assert np.array_equal(out, yprev)
        assert out.dtype == np.float64

    def test_accepts_list_input_and_preserves_values(self):
        out = naive_persistence_forecast([1, 2, 3])
        assert isinstance(out, np.ndarray)
        # Lists must be cast to float arrays with the values intact.
        assert out.dtype.kind == "f"
        assert np.array_equal(out, np.array([1.0, 2.0, 3.0]))


class TestDirectionalAccuracy:
    def test_perfect_directional_match(self):
        # actual goes up/down/up; predictions agree on direction (not magnitude)
        prev = np.array([100, 100, 100])
        actual = np.array([105, 95, 110])    # up, down, up
        pred = np.array([101, 99, 200])      # up, down, up
        assert directional_accuracy(actual, pred, prev) == 100.0

    def test_zero_directional_match(self):
        prev = np.array([100, 100, 100])
        actual = np.array([105, 95, 110])    # up, down, up
        pred = np.array([99, 105, 90])       # down, up, down
        assert directional_accuracy(actual, pred, prev) == 0.0

    def test_persistence_baseline_yields_nan(self):
        """Persistence has pred == prev, so sign(pred - prev) == 0 — no
        directional information by construction. Spec: return NaN."""
        prev = np.array([100, 101, 102])
        actual = np.array([105, 99, 110])
        pred = prev.copy()
        assert math.isnan(directional_accuracy(actual, pred, prev))

    def test_partial_match(self):
        prev = np.array([100, 100, 100, 100])
        actual = np.array([105, 95, 110, 90])   # up, down, up, down
        pred = np.array([101, 99, 99, 105])     # up, down, down, up — 2/4 = 50%
        assert directional_accuracy(actual, pred, prev) == 50.0


class TestSkillScore:
    def test_perfect_model_skill_one(self):
        assert skill_score(mse_model=0.0, mse_baseline=1.0) == 1.0

    def test_tie_with_baseline_skill_zero(self):
        assert skill_score(mse_model=0.5, mse_baseline=0.5) == 0.0

    def test_worse_than_baseline_negative(self):
        assert skill_score(mse_model=2.0, mse_baseline=1.0) == -1.0

    def test_zero_baseline_returns_nan(self):
        assert math.isnan(skill_score(mse_model=0.5, mse_baseline=0.0))


class TestComputeMetrics:
    def test_perfect_prediction_zero_error(self):
        y = np.array([10.0, 20.0, 30.0])
        m = compute_metrics(y, y)
        assert m.mae == 0.0
        assert m.rmse == 0.0
        assert m.mape == 0.0
        assert m.r2 == 1.0
        assert m.worst_day_error == 0.0
        assert m.mean_signed_error == 0.0
        assert m.n == 3

    def test_mae_rmse_mape_against_hand_values(self):
        actual = np.array([100.0, 100.0])
        pred = np.array([110.0, 90.0])
        m = compute_metrics(actual, pred)
        # residual = [10, -10] → abs = [10, 10]
        assert m.mae == 10.0
        # rmse = sqrt(mean([100, 100])) = 10
        assert m.rmse == 10.0
        # mape = mean([10/100, 10/100]) * 100 = 10
        assert m.mape == 10.0
        # mean signed error = mean([10, -10]) = 0 (no bias)
        assert m.mean_signed_error == 0.0
        # worst-day error = max abs = 10
        assert m.worst_day_error == 10.0

    def test_systematic_overprediction_positive_bias(self):
        actual = np.array([100.0, 100.0, 100.0])
        pred = np.array([105.0, 102.0, 108.0])
        m = compute_metrics(actual, pred)
        assert m.mean_signed_error > 0
        # MAE here equals signed error mean since all residuals positive
        assert m.mae == pytest.approx(m.mean_signed_error)

    def test_skill_vs_persistence_zero_when_pred_equals_prev(self):
        # If model predicts exactly yesterday's close, MSE == baseline MSE,
        # so skill score == 0.
        actual = np.array([105.0, 99.0, 110.0])
        prev = np.array([100.0, 101.0, 102.0])
        pred = prev.copy()
        m = compute_metrics(actual, pred, y_prev=prev)
        assert m.skill_vs_persistence == pytest.approx(0.0)

    def test_out_of_range_flag(self):
        actual = np.array([5.0, 50.0, 200.0])
        pred = actual.copy()
        m = compute_metrics(actual, pred, train_min=10.0, train_max=100.0)
        # 5 < 10 and 200 > 100 → 2 out of 3 = 66.66...%
        assert m.out_of_train_range_pct == pytest.approx(200 / 3)

    def test_out_of_range_none_when_no_bounds(self):
        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
        assert m.out_of_train_range_pct is None
        assert m.skill_vs_persistence is None  # no y_prev given

    def test_pydantic_rejects_negative_error_fields(self):
        """MAE / RMSE / MAPE / worst_day_error are non-negative by
        construction; PredictionMetrics enforces this via Field(ge=0)."""
        from pydantic import ValidationError

        from src.metrics import PredictionMetrics

        with pytest.raises(ValidationError):
            PredictionMetrics(
                n=3, mae=-0.1, rmse=0.0, mape=0.0, r2=1.0,
                directional_accuracy=50.0, mean_signed_error=0.0,
                worst_day_error=0.0,
            )

    def test_pydantic_metrics_are_frozen(self):
        """Metrics snapshot is immutable — mutating after the fact would
        invalidate any report it was already written to."""
        from pydantic import ValidationError

        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            m.mae = 999.0

    def test_as_row_has_correct_keys_and_values(self):
        # actual=[1, 2], pred=[1.1, 1.9] -> residuals [+0.1, -0.1]
        # MAE = 0.1, mean signed = 0.0, worst = 0.1
        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
        row = m.as_row()

        for key in ("n", "MAE", "RMSE", "MAPE%", "R2", "DirAcc%",
                    "MeanSignedErr", "WorstDayErr",
                    "Skill_vs_persist", "OutOfRange%"):
            assert key in row, f"missing key {key}"

        # Spot-check the actual values too — the rounding alone is non-trivial.
        assert row["n"] == 2
        assert row["MAE"] == pytest.approx(0.1)
        assert row["MeanSignedErr"] == pytest.approx(0.0, abs=1e-9)
        assert row["WorstDayErr"] == pytest.approx(0.1)
        # No y_prev / no bounds given -> these stay None in the row.
        assert row["Skill_vs_persist"] is None
        assert row["OutOfRange%"] is None
