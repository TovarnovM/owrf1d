from __future__ import annotations

import math

from owrf1d import OnlineWindowRegressor1D


def test_dumps_loads_roundtrip_next_step_identical():
    f = OnlineWindowRegressor1D(max_window=32, min_window=4, history=0)

    for i in range(30):
        f.update(1.0 + 0.1 * i, t=float(i))

    blob = f.dumps()
    g = OnlineWindowRegressor1D.loads(blob)

    out_f = f.update(4.2, t=30.0)
    out_g = g.update(4.2, t=30.0)

    # Strict equality (target: deterministic + stable serialization)
    assert out_f["n_star"] == out_g["n_star"]
    assert out_f["flags"] == out_g["flags"]

    assert out_f["mu"] == out_g["mu"]
    assert out_f["trend"] == out_g["trend"]
    assert out_f["sigma2"] == out_g["sigma2"]

    assert out_f["pred_mu"] == out_g["pred_mu"]
    assert out_f["pred_s2"] == out_g["pred_s2"]
    assert out_f["score_star"] == out_g["score_star"]
    assert out_f["score_second"] == out_g["score_second"]
    assert out_f["delta_score"] == out_g["delta_score"]

    # sanity
    assert math.isfinite(out_g["mu"])
