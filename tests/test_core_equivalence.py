import math
import random
import pytest

from owrf1d import OnlineWindowRegressor1D
from owrf1d.filter import HAVE_CYTHON_CORE


def _gen_piecewise_linear(n: int, *, t0: int, a0: float, b1: float, b2: float, sigma: float, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = float(i)
        b = b1 if i < t0 else b2
        y = a0 + b * t + rng.gauss(0.0, sigma)
        out.append((t, y))
    return out


@pytest.mark.skipif(not HAVE_CYTHON_CORE, reason="Cython core not built")
def test_cython_selection_matches_python_reference():
    W = 64
    seq = _gen_piecewise_linear(240, t0=120, a0=0.0, b1=0.05, b2=0.25, sigma=0.08, seed=2)

    f_py = OnlineWindowRegressor1D(max_window=W, min_window=4, history=0, selection="hard")
    f_cy = OnlineWindowRegressor1D(max_window=W, min_window=4, history=0, selection="hard")

    # Force modes
    f_py._use_core = False
    f_cy._use_core = True

    for t, y in seq:
        out_py = f_py.update(y, t=t)
        out_cy = f_cy.update(y, t=t)

        # Selection-sensitive fields should match
        for k in ("n_star", "nu", "flags"):
            assert out_py[k] == out_cy[k]

        for k in ("pred_mu", "pred_s2", "score_star", "score_second", "delta_score"):
            assert math.isfinite(out_py[k])
            assert math.isfinite(out_cy[k])
            assert abs(out_py[k] - out_cy[k]) <= 1e-9

        # State estimates should match (post-update is Python in both cases)
        for k in ("mu", "trend", "sigma2"):
            assert abs(out_py[k] - out_cy[k]) <= 1e-9
