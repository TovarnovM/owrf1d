import math

from owrf1d import OnlineWindowRegressor1D


def test_dt_interface_works_without_timestamps():
    f = OnlineWindowRegressor1D(max_window=32, min_window=4, history=0)

    for i in range(20):
        out = f.update(1.0 + 0.1 * i, dt=1.0)
        assert math.isfinite(out["dt"])
        assert out["dt"] == 1.0
        assert math.isfinite(out["t"])  # internal accumulated time is acceptable
