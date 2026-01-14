import math
import random

import pytest

from owrf1d import OnlineWindowRegressor1D
from owrf1d.filter import HAVE_CYTHON_CORE


def _gen_piecewise_linear_sigma(
    n: int,
    *,
    t0: int,
    a0: float,
    b1: float,
    b2: float,
    s1: float,
    s2: float,
    seed: int = 0,
):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = float(i)
        b = b1 if i < t0 else b2
        sigma = s1 if i < t0 else s2
        y = a0 + b * t + rng.gauss(0.0, sigma)
        out.append((t, y))
    return out


@pytest.mark.skipif(not HAVE_CYTHON_CORE, reason="Cython core not built")
def test_cython_soft_matches_python_reference_constants():
    W = 64
    seq = _gen_piecewise_linear_sigma(
        320,
        t0=160,
        a0=0.0,
        b1=0.05,
        b2=0.25,
        s1=0.06,
        s2=0.12,
        seed=7,
    )

    f_py = OnlineWindowRegressor1D(max_window=W, min_window=4, history=0, selection="soft")
    f_cy = OnlineWindowRegressor1D(max_window=W, min_window=4, history=0, selection="soft")

    # Force modes
    f_py._use_core = False
    f_cy._use_core = True

    atol = 1e-8
    float_keys = (
        "pred_mu",
        "pred_s2",
        "score_star",
        "score_second",
        "delta_score",
        "mu",
        "trend",
        "sigma2",
        "n_eff",
        "w_star",
        "entropy",
        "entropy_norm",
        "tau",
        "sigma2_total",
        "d_used",
    )
    int_keys = (
        "n_star",
        "nu",
        "flags",
        "cap",
        "cap_before",
        "cap_target",
        "n_star_hard",
    )

    for t, y in seq:
        out_py = f_py.update(y, t=t)
        out_cy = f_cy.update(y, t=t)

        for k in int_keys:
            assert out_py[k] == out_cy[k]

        for k in float_keys:
            assert math.isfinite(out_py[k])
            assert math.isfinite(out_cy[k])
            assert abs(out_py[k] - out_cy[k]) <= atol

        # core flag is expected to differ (py forced, cy forced)
        assert out_py["core"] is False
        assert out_cy["core"] is True
