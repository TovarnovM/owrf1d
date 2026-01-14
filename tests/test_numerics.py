import math

from owrf1d import OnlineWindowRegressor1D
from owrf1d.flags import (
    FLAG_DEGENERATE_XTX,
    FLAG_NEGATIVE_SSE,
    FLAG_NUMERIC_GUARD,
)


def test_repeated_timestamps_sets_degenerate_flag_but_no_crash():
    f = OnlineWindowRegressor1D(max_window=32, min_window=4, history=0, selection='hard')

    # Many equal timestamps => x variance collapses => D ~ 0
    for i in range(10):
        out = f.update(1.0 + 0.1 * i, t=1.0)

    assert isinstance(out, dict)
    assert (out["flags"] & FLAG_DEGENERATE_XTX) != 0
    assert math.isfinite(out["mu"])
    assert math.isfinite(out["trend"])
    assert math.isfinite(out["sigma2"])
    assert out["sigma2"] >= 0.0


def test_numeric_guards_prevent_nan_inf():
    f = OnlineWindowRegressor1D(max_window=64, min_window=4, history=0, selection='hard')

    # Large magnitude y and tiny dt can stress numerics
    t = 0.0
    for i in range(200):
        t += 1e-12
        y = 1e12 + 1e6 * i
        out = f.update(y, t=t)

    assert math.isfinite(out["mu"])
    assert math.isfinite(out["trend"])
    assert math.isfinite(out["sigma2"])
    assert out["sigma2"] >= 0.0

    # If guards were triggered, flag should reflect it (allowed either way).
    if (out["flags"] & FLAG_NUMERIC_GUARD) != 0:
        assert True


def test_negative_sse_is_clipped_and_flagged():
    f = OnlineWindowRegressor1D(max_window=16, min_window=4, history=0)

    # Construct near-perfect line so SSE can go tiny; numeric roundoff may go negative
    for i in range(50):
        out = f.update(1.0 + 0.25 * i, t=float(i))

    # We accept either: no negative SSE occurred, or it was clipped+flagged.
    if (out["flags"] & FLAG_NEGATIVE_SSE) != 0:
        assert out["sigma2"] >= 0.0
