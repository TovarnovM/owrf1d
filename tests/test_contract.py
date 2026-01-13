import math
import pytest

from owrf1d import OnlineWindowRegressor1D
from owrf1d.flags import (
    FLAG_PREDICT_ONLY,
    FLAG_INSUFFICIENT_DATA,
)

REQUIRED_KEYS = {
    "mu",
    "trend",
    "sigma2",
    "n_star",
    "score_star",
    "score_second",
    "delta_score",
    "nu",
    "pred_mu",
    "pred_s2",
    "resid",
    "t",
    "dt",
    "flags",
}


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def test_step_dict_contract_and_types():
    f = OnlineWindowRegressor1D(max_window=32, min_window=4, history=0)

    # warmup points
    for i in range(10):
        out = f.update(1.0 + 0.1 * i, t=float(i))
        assert isinstance(out, dict)
        assert REQUIRED_KEYS.issubset(out.keys())

        # required numeric fields should be finite (when y is provided)
        assert _is_finite(out["mu"])
        assert _is_finite(out["trend"])
        assert _is_finite(out["sigma2"])
        assert out["sigma2"] >= 0.0
        assert isinstance(out["n_star"], int)
        assert out["n_star"] >= 0
        assert isinstance(out["nu"], int)

        assert _is_finite(out["pred_mu"])
        assert _is_finite(out["pred_s2"])
        assert out["pred_s2"] >= 0.0

        assert _is_finite(out["resid"])
        assert _is_finite(out["t"])
        assert _is_finite(out["dt"])

        assert isinstance(out["flags"], int)


def test_insufficient_data_flag_before_min_window():
    f = OnlineWindowRegressor1D(max_window=32, min_window=6, history=0)

    # With less than min_window prior points, must set INSUFFICIENT flag.
    for i in range(1, 6):
        out = f.update(1.0, t=float(i))
        assert (out["flags"] & FLAG_INSUFFICIENT_DATA) != 0


def test_predict_only_y_none_sets_flag_and_does_not_change_state_shape():
    f = OnlineWindowRegressor1D(max_window=32, min_window=4, history=0)

    # First feed enough points
    for i in range(8):
        f.update(1.0 + 0.01 * i, t=float(i))

    st0 = f.get_state()
    out = f.update(None, t=8.0)  # predict-only / time advance

    assert (out["flags"] & FLAG_PREDICT_ONLY) != 0

    # State should remain valid and finite
    st1 = f.get_state()
    assert isinstance(st1, dict)
    assert "mu" in st1 and "trend" in st1 and "sigma2" in st1

    # Minimal invariants: no NaN/Inf in state after y=None
    assert _is_finite(st1["mu"])
    assert _is_finite(st1["trend"])
    assert _is_finite(st1["sigma2"])


def test_history_controls_storage():
    f0 = OnlineWindowRegressor1D(max_window=16, min_window=4, history=0)
    for i in range(10):
        f0.update(1.0 + 0.1 * i, t=float(i))
    assert f0.get_history() == []

    fN = OnlineWindowRegressor1D(max_window=16, min_window=4, history=5)
    for i in range(10):
        fN.update(1.0 + 0.1 * i, t=float(i))
    h = fN.get_history()
    assert isinstance(h, list)
    assert len(h) == 5
    # last entry should have required keys
    assert REQUIRED_KEYS.issubset(h[-1].keys())
