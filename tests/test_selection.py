import random

import pytest

from owrf1d import OnlineWindowRegressor1D


def _gen_piecewise_linear(
    n: int,
    *,
    t0: int,
    a0: float,
    b1: float,
    b2: float,
    sigma: float,
    seed: int = 0,
):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = float(i)
        b = b1 if i < t0 else b2
        y = a0 + b * t + rng.gauss(0.0, sigma)
        out.append((t, y))
    return out


@pytest.mark.parametrize("sigma", [0.05, 0.10])
def test_stationary_line_prefers_large_window(sigma):
    W = 64
    f = OnlineWindowRegressor1D(max_window=W, min_window=4, history=0, selection='hard')

    seq = _gen_piecewise_linear(
        200, t0=10_000, a0=1.0, b1=0.1, b2=0.1, sigma=sigma, seed=1
    )

    nstars = []
    for t, y in seq:
        out = f.update(y, t=t)
        nstars.append(out["n_star"])

    # После прогрева n_star должен быть "близко" к max_window.
    tail = nstars[-50:]
    assert sum(1 for x in tail if x >= int(0.8 * W)) >= 35


def test_slope_change_forces_window_drop_then_recover():
    W = 80
    cp = 120
    f = OnlineWindowRegressor1D(max_window=W, min_window=4, history=0, selection='hard')

    seq = _gen_piecewise_linear(
        260, t0=cp, a0=0.0, b1=0.05, b2=0.25, sigma=0.08, seed=2
    )

    nstars = []
    for t, y in seq:
        out = f.update(y, t=t)
        nstars.append(out["n_star"])

    # До change-point окно должно быть относительно большим
    pre = nstars[cp - 30 : cp - 5]
    assert sum(1 for x in pre if x >= int(0.6 * W)) >= 15

    # Сразу после change-point — заметное падение окна
    post = nstars[cp : cp + 20]
    assert min(post) <= int(0.4 * W)

    # Дальше — восстановление окна (адаптация к новому наклону)
    late = nstars[-40:]
    assert sum(1 for x in late if x >= int(0.6 * W)) >= 20
