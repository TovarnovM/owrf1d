def test_determinism_same_inputs_same_outputs():
    from owrf1d import OnlineWindowRegressor1D

    seq = [(float(i), 1.0 + 0.01 * i) for i in range(50)]

    f1 = OnlineWindowRegressor1D(max_window=32, min_window=4, history=0)
    f2 = OnlineWindowRegressor1D(max_window=32, min_window=4, history=0)

    outs1 = [f1.update(y, t=t) for (t, y) in seq]
    outs2 = [f2.update(y, t=t) for (t, y) in seq]

    # Strict: all key fields equal bitwise for deterministic arithmetic path.
    # If you later introduce math library differences, relax to approx.
    for a, b in zip(outs1, outs2, strict=True):
        assert a["n_star"] == b["n_star"]
        assert a["flags"] == b["flags"]
        assert a["mu"] == b["mu"]
        assert a["trend"] == b["trend"]
        assert a["sigma2"] == b["sigma2"]
        assert a["pred_mu"] == b["pred_mu"]
        assert a["pred_s2"] == b["pred_s2"]
        assert a["score_star"] == b["score_star"]
        assert a["score_second"] == b["score_second"]
        assert a["delta_score"] == b["delta_score"]
