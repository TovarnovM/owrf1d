# examples/example_nonlinear.py
"""
Nonlinear drift + nonlinear sigma example (smooth regime changes).

- trend(t) changes smoothly via logistic transition + sinusoidal modulation
- sigma(t) changes smoothly via logistic transition + periodic modulation
- occasional outliers
- occasional missing observations (y=None)

Runs OnlineWindowRegressor1D strictly online and saves PNG plots.

Usage:
  python examples/example_nonlinear.py --dump-jsonl
  python examples/example_nonlinear.py --n 800 --max-window 128 --seed 3
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

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for this example. Install optional deps, e.g.\n"
        "  pip install -U matplotlib\n"
        f"Original import error: {e}"
    )

from owrf1d import OnlineWindowRegressor1D


def _sigmoid(z: float) -> float:
    # Stable logistic
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class Scenario:
    n: int = 700

    # Smooth transition centers in index space (we’ll convert to time via generated t)
    cp: int = 200       # trend transition center
    cp2: int = 450      # sigma transition center

    # Time step process
    dt_mean: float = 1.0
    dt_jitter: float = 0.10

    # Trend model parameters
    a0: float = 0.0
    b1: float = -0.25          # initial drift (slope)
    b2: float = 0.15           # final drift (slope)
    trend_width_t: float = 30.0  # transition width in "time units"
    trend_wave_amp: float = 0.08
    trend_wave_period_t: float = 120.0

    # Sigma model parameters (noise std)
    sigma1: float = 1.0
    sigma2: float = 3.5
    sigma_width_t: float = 35.0
    sigma_wave_amp: float = 0.6
    sigma_wave_period_t: float = 30.0
    sigma_floor: float = 1e-3

    # Contamination
    p_outlier: float = 0.02
    outlier_scale: float = 2.0
    p_missing: float = 0.02


def gen_series(
    seed: int, sc: Scenario
) -> Iterator[Tuple[float, Optional[float], int, float, float, float, float, float]]:
    """
    Yields strictly in time order:
        (t, y_or_none, i, mu_true, trend_true, sigma_true, cp_t, cp2_t)

    cp_t/cp2_t returned for plotting vertical markers (constant after computed).
    """
    rng = random.Random(seed)

    # 1) Precompute timestamps (to define smooth transitions in *time* domain)
    dts: List[float] = []
    ts: List[float] = []
    t = 0.0
    for _ in range(sc.n):
        dt = max(1e-6, sc.dt_mean + sc.dt_jitter * rng.uniform(-1.0, 1.0))
        t += dt
        dts.append(dt)
        ts.append(t)

    cp_t = ts[min(max(sc.cp, 0), sc.n - 1)]
    cp2_t = ts[min(max(sc.cp2, 0), sc.n - 1)]

    # 2) Define nonlinear trend(t) and sigma(t)
    def trend_fn(tt: float) -> float:
        # logistic transition from b1 to b2 around cp_t
        z = (tt - cp_t) / max(sc.trend_width_t, 1e-6)
        base = sc.b1 + (sc.b2 - sc.b1) * _sigmoid(z)
        # sinusoidal modulation
        wave = sc.trend_wave_amp * math.sin(2.0 * math.pi * tt / max(sc.trend_wave_period_t, 1e-6))
        return base + wave

    def sigma_fn(tt: float) -> float:
        # logistic transition from sigma1 to sigma2 around cp2_t
        z = (tt - cp2_t) / max(sc.sigma_width_t, 1e-6)
        base = sc.sigma1 + (sc.sigma2 - sc.sigma1) * _sigmoid(z)
        # periodic modulation (kept non-negative)
        wave = sc.sigma_wave_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * tt / max(sc.sigma_wave_period_t, 1e-6)))
        return max(sc.sigma_floor, base + wave)

    # 3) Generate mu_true by integrating trend(t) over dt (midpoint rule)
    mu = sc.a0
    for i, (tt, dt) in enumerate(zip(ts, dts, strict=True)):
        t_mid = tt - 0.5 * dt
        trend_true = trend_fn(tt)
        mu = mu + trend_fn(t_mid) * dt  # keep mu continuous and smooth
        mu_true = mu
        sigma_true = sigma_fn(tt)

        # missing observation
        if rng.random() < sc.p_missing:
            yield tt, None, i, mu_true, trend_true, sigma_true, cp_t, cp2_t
            continue

        # observation with nonlinear sigma
        y = mu_true + rng.gauss(0.0, sigma_true)

        # occasional outlier (scaled by sigma for realism)
        if rng.random() < sc.p_outlier:
            y += rng.choice([-1.0, 1.0]) * sc.outlier_scale * sigma_true * (1.0 + rng.random())

        yield tt, y, i, mu_true, trend_true, sigma_true, cp_t, cp2_t


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

    fig, axes = plt.subplots(
        4, 1, height_ratios=[3, 1, 1, 1], figsize=(12, 16), sharex=True
    )

    # 1) y points + mu + mu_true
    ax = axes[0]
    t_pts = [ti for ti, yi in zip(t, y) if math.isfinite(yi)]
    y_pts = [yi for yi in y if math.isfinite(yi)]
    ax.scatter(t_pts, y_pts, s=5, label="y (points)", alpha=0.7)
    ax.plot(t, mu, linewidth=2.0, label="mu (filtered)", zorder=10)
    ax.plot(t, mu_true, linewidth=1.5, label="mu_true")
    ax.axvline(cp_t, linestyle="--", linewidth=1.0)
    ax.axvline(cp2_t, linestyle="--", linewidth=1.0)
    ax.set_title("Nonlinear drift: y, filtered mu, and true mu")
    ax.legend(loc="best")

    # 2) trend + true trend
    ax = axes[1]
    ax.plot(t, trend, linewidth=1.8, label="trend (filtered)")
    ax.plot(t, trend_true, linewidth=1.5, label="trend_true")
    ax.axvline(cp_t, linestyle="--", linewidth=1.0)
    ax.axvline(cp2_t, linestyle="--", linewidth=1.0)
    ax.set_title("Trend vs true trend")
    ax.legend(loc="best")

    # 3) sigma_hat + sigma_true
    ax = axes[2]
    ax.plot(t, sigma, linewidth=1.8, label="sigma_hat = sqrt(sigma2)")
    ax.plot(t, sigma_true, linewidth=1.5, label="sigma_true")
    ax.axvline(cp_t, linestyle="--", linewidth=1.0)
    ax.axvline(cp2_t, linestyle="--", linewidth=1.0)
    ax.set_title("Nonlinear sigma: estimated vs true")
    ax.legend(loc="best")

    # 4) n_star
    ax = axes[3]
    ax.plot(t, n_star, linewidth=1.5)
    ax.axvline(cp_t, linestyle="--", linewidth=1.0)
    ax.axvline(cp2_t, linestyle="--", linewidth=1.0)
    ax.set_title("Selected window length n* (soft)")
    ax.set_xlabel("t")

    fig.tight_layout()
    fig.savefig(outdir / "example_nonlinear_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=700)
    p.add_argument("--cp", type=int, default=200)
    p.add_argument("--cp2", type=int, default=450)
    p.add_argument("--seed", type=int, default=2)

    p.add_argument("--max-window", type=int, default=128)
    p.add_argument("--min-window", type=int, default=4)
    p.add_argument("--history", type=int, default=0)

    p.add_argument("--outdir", type=str, default=str(Path("examples") / "out"))
    p.add_argument("--dump-jsonl", action="store_true")

    args = p.parse_args()

    sc = Scenario(n=args.n, cp=args.cp, cp2=args.cp2)
    outdir = Path(args.outdir)

    f = OnlineWindowRegressor1D(
        max_window=args.max_window,
        min_window=args.min_window,
        history=args.history,
        selection="soft",
    )

    ts: List[float] = []
    ys: List[float] = []
    mus: List[float] = []
    trends: List[float] = []
    sigmas: List[float] = []
    nstars: List[int] = []

    mu_true: List[float] = []
    trend_true: List[float] = []
    sigma_true: List[float] = []

    cp_t = 0.0
    cp2_t = 0.0

    jsonl_fp = None
    if args.dump_jsonl:
        outdir.mkdir(parents=True, exist_ok=True)
        jsonl_fp = (outdir / "example_nonlinear_steps.jsonl").open("w", encoding="utf-8")

    for t, y, i, mu_t, b_t, sig_t, cp_tt, cp2_tt in gen_series(args.seed, sc):
        cp_t, cp2_t = cp_tt, cp2_tt

        step = f.update(y, t=t)

        ts.append(float(t))
        ys.append(float("nan") if y is None else float(y))
        mus.append(float(step["mu"]))
        trends.append(float(step["trend"]))
        sigmas.append(math.sqrt(max(float(step["sigma2"]), 0.0)))
        nstars.append(int(step["n_star"]))

        mu_true.append(float(mu_t))
        trend_true.append(float(b_t))
        sigma_true.append(float(sig_t))

        if jsonl_fp is not None:
            jsonl_fp.write(json.dumps(step, ensure_ascii=False) + "\n")

    if jsonl_fp is not None:
        jsonl_fp.close()

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
        cp_t=float(cp_t),
        cp2_t=float(cp2_t),
    )

    print("Saved:", outdir / "example_nonlinear_overview.png")
    if args.dump_jsonl:
        print("Saved:", outdir / "example_nonlinear_steps.jsonl")
    print("Final state:", f.get_state())


if __name__ == "__main__":
    main()
