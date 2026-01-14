import math
import importlib.metadata as md

from owrf1d import OnlineWindowRegressor1D
from owrf1d.filter import HAVE_CYTHON_CORE


def assert_finite(step: dict, keys: tuple[str, ...]) -> None:
    for k in keys:
        v = step[k]
        if not isinstance(v, (int, float)):
            raise AssertionError((k, type(v), v))
        if not math.isfinite(float(v)):
            raise AssertionError((k, v))


def main() -> None:
    print("Installed owrf1d:", md.version("owrf1d"))
    print("HAVE_CYTHON_CORE:", HAVE_CYTHON_CORE)

    # 1) Contract / finiteness (hard selection)
    f = OnlineWindowRegressor1D(max_window=64, min_window=4, history=0, selection="hard")
    last = None
    for i in range(200):
        t = float(i)
        slope = 0.05 if i < 120 else 0.25
        y = slope * t
        out = f.update(y, t=t)

        assert_finite(
            out,
            ("mu", "trend", "sigma2", "pred_mu", "pred_s2", "score_star", "score_second", "delta_score", "t", "dt"),
        )
        if out["sigma2"] < 0:
            raise AssertionError(("sigma2 negative", out["sigma2"]))
        if not isinstance(out["n_star"], int) or out["n_star"] < 0:
            raise AssertionError(("bad n_star", out["n_star"]))

        last = out

    assert last is not None
    print("Hard selection OK. last:", {k: last[k] for k in ("mu", "trend", "sigma2", "n_star", "flags")})

    # 2) dumps/loads determinism (soft)
    g = OnlineWindowRegressor1D(max_window=32, min_window=4, history=0, selection="soft")
    for i in range(30):
        g.update(1.0 + 0.1 * i, t=float(i))

    blob = g.dumps()
    h = OnlineWindowRegressor1D.loads(blob)

    a = g.update(4.2, t=30.0)
    b = h.update(4.2, t=30.0)

    keys = (
        "n_star",
        "flags",
        "mu",
        "trend",
        "sigma2",
        "pred_mu",
        "pred_s2",
        "score_star",
        "score_second",
        "delta_score",
    )
    for k in keys:
        if a[k] != b[k]:
            raise AssertionError((k, a[k], b[k]))

    print("dumps/loads deterministic OK.")
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
