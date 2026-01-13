# examples/example1.py
"""
Example 1: “real-ish” 1D time series with:
- piecewise linear drift (slope change)
- variance jump
- occasional outliers
- occasional missing observations (y=None)

Runs OnlineWindowRegressor1D strictly online and saves PNG plots.

Usage:
  python examples/example1.py
  python examples/example1.py --n 600 --cp 250 --max-window 128 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, List

try:
    import matplotlib

    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for this example. Install optional deps, e.g.\n"
        "  pip install -U matplotlib\n"
        f"Original import error: {e}"
    )

from owrf1d import OnlineWindowRegressor1D


@dataclass(frozen=True)
class Scenario:
    n: int = 600
    cp: int = 200  # change-point index (slope)
    a0: float = 0.0
    b1: float = -0.7
    b2: float = 0.2
    sigma1: float = 2.08
    sigma2: float = 7.16  # noise std after cp2 (variance jump)
    cp2: int = 400  # variance change index
    dt_mean: float = 1.0
    dt_jitter: float = 0.10
    p_outlier: float = 0.02
    outlier_scale: float = 0.70
    p_missing: float = 0.02


def gen_series(
    seed: int, sc: Scenario
) -> Iterator[Tuple[float, Optional[float], int, float, float, float]]:
    """
    Yields strictly in time order:
        (t, y_or_none, i, mu_true, trend_true, sigma_true)

    mu_true: latent noise-free level at time t
    trend_true: latent slope (per unit time) at time t
    sigma_true: observation noise std used at time t
    """
    rng = random.Random(seed)

    t = 0.0
    mu = sc.a0
    slope = sc.b1

    for i in range(sc.n):
        dt = max(1e-6, sc.dt_mean + sc.dt_jitter * rng.uniform(-1.0, 1.0))
        t += dt

        # regime changes (slope)
        if i == sc.cp:
            slope = sc.b2

        # regime changes (noise)
        sigma = sc.sigma1 if i < sc.cp2 else sc.sigma2

        # latent update
        mu = mu + slope * dt

        mu_true = mu
        trend_true = slope
        sigma_true = sigma

        # missing observation
        if rng.random() < sc.p_missing:
            yield t, None, i, mu_true, trend_true, sigma_true
            continue

        # observation with noise
        y = mu_true + rng.gauss(0.0, sigma_true)

        # occasional outlier
        if rng.random() < sc.p_outlier:
            y += rng.choice([-1.0, 1.0]) * sc.outlier_scale * (1.0 + rng.random())

        yield t, y, i, mu_true, trend_true, sigma_true


def save_overview_png(
    outdir: Path,
    t: List[float],
    y: List[float],
    mu: List[float],
    mu_true: List[float],
    trend: List[float],
    trend_true: List[float],
    sigma: List[float],
    sigma_true: List[float],
    n_star: List[int],
    cp_t: float,
    cp2_t: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, height_ratios=[3,1,1,1], figsize=(12, 16), sharex=True)

    # 1) y (points) vs mu + true mu
    ax = axes[0]
    # scatter only finite points (missing are NaN)
    t_pts = [ti for ti, yi in zip(t, y) if math.isfinite(yi)]
    y_pts = [yi for yi in y if math.isfinite(yi)]
    ax.scatter(t_pts, y_pts, s=5, label="y (points)", alpha=0.7, color='C01')
    ax.plot(t, mu, linewidth=2.0, label="mu (filtered)", zorder=10)
    ax.plot(t, mu_true, linewidth=1.5, label="mu_true")
    ax.axvline(cp_t, linestyle="--", linewidth=1.0)
    ax.axvline(cp2_t, linestyle="--", linewidth=1.0)
    ax.set_title("Signal: y (points), filtered level, and true level")
    ax.legend(loc="best")

    # 2) trend + true trend
    ax = axes[1]
    ax.plot(t, trend, linewidth=1.8, label="trend (filtered)")
    ax.plot(t, trend_true, linewidth=1.5, label="trend_true")
    ax.axvline(cp_t, linestyle="--", linewidth=1.0)
    ax.axvline(cp2_t, linestyle="--", linewidth=1.0)
    ax.set_title("Estimated trend vs true trend")
    ax.legend(loc="best")

    # 3) sigma (sqrt(sigma2)) + true sigma
    ax = axes[2]
    ax.plot(t, sigma, linewidth=1.8, label="sigma_hat = sqrt(sigma2)")
    ax.plot(t, sigma_true, linewidth=1.5, label="sigma_true")
    ax.axvline(cp_t, linestyle="--", linewidth=1.0)
    ax.axvline(cp2_t, linestyle="--", linewidth=1.0)
    ax.set_title("Estimated noise scale vs true sigma")
    ax.legend(loc="best")

    # 4) n_star
    ax = axes[3]
    ax.plot(t, n_star, linewidth=1.5)
    ax.axvline(cp_t, linestyle="--", linewidth=1.0)
    ax.axvline(cp2_t, linestyle="--", linewidth=1.0)
    ax.set_title("Selected window length n*")
    ax.set_xlabel("t")

    fig.tight_layout()
    fig.savefig(outdir / "example1_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--cp", type=int, default=100)
    p.add_argument("--cp2", type=int, default=350)
    p.add_argument("--seed", type=int, default=2)

    p.add_argument("--max-window", type=int, default=96)
    p.add_argument("--min-window", type=int, default=4)
    p.add_argument("--history", type=int, default=0)

    p.add_argument("--outdir", type=str, default=str(Path("examples") / "out"))
    p.add_argument("--dump-jsonl", action="store_true", help="also write steps to jsonl")

    args = p.parse_args()

    sc = Scenario(n=args.n, cp=args.cp, cp2=args.cp2)
    outdir = Path(args.outdir)

    f = OnlineWindowRegressor1D(
        max_window=args.max_window,
        min_window=args.min_window,
        history=args.history,
        selection='soft'
    )

    ts: List[float] = []
    ys: List[float] = []
    mus: List[float] = []
    trends: List[float] = []
    sigmas: List[float] = []
    nstars: List[int] = []
    flags: List[int] = []

    mu_true: List[float] = []
    trend_true: List[float] = []
    sigma_true: List[float] = []

    cp_t = None
    cp2_t = None

    jsonl_fp = None
    if args.dump_jsonl:
        outdir.mkdir(parents=True, exist_ok=True)
        jsonl_fp = (outdir / "example1_steps.jsonl").open("w", encoding="utf-8")

    for t, y, i, mu_t, b_t, sig_t in gen_series(args.seed, sc):
        if i == sc.cp:
            cp_t = t
        if i == sc.cp2:
            cp2_t = t

        step = f.update(y, t=t)

        ts.append(float(t))
        ys.append(float("nan") if y is None else float(y))
        mus.append(float(step["mu"]))
        trends.append(float(step["trend"]))
        sigmas.append(math.sqrt(max(float(step["sigma2"]), 0.0)))
        nstars.append(int(step["n_star"]))
        flags.append(int(step["flags"]))

        mu_true.append(float(mu_t))
        trend_true.append(float(b_t))
        sigma_true.append(float(sig_t))

        if jsonl_fp is not None:
            jsonl_fp.write(json.dumps(step, ensure_ascii=False) + "\n")

    if jsonl_fp is not None:
        jsonl_fp.close()

    cp_t = float(cp_t) if cp_t is not None else ts[len(ts) // 2]
    cp2_t = float(cp2_t) if cp2_t is not None else ts[int(len(ts) * 0.7)]

    save_overview_png(
        outdir=outdir,
        t=ts,
        y=ys,
        mu=mus,
        mu_true=mu_true,
        trend=trends,
        trend_true=trend_true,
        sigma=sigmas,
        sigma_true=sigma_true,
        n_star=nstars,
        cp_t=cp_t,
        cp2_t=cp2_t,
    )

    print("Saved:", outdir / "example1_overview.png")
    if args.dump_jsonl:
        print("Saved:", outdir / "example1_steps.jsonl")
    print("Final state:", f.get_state())


if __name__ == "__main__":
    main()
