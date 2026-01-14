# examples/bench.py
"""
Micro-benchmarks for owrf1d.

What it measures
- End-to-end OnlineWindowRegressor1D.update() throughput for:
  - selection: hard vs soft
  - backend: Cython core vs pure-Python fallback

Optional
- Core-only microbench (fixed ring state):
  - two-step: select_student_t + soft_step_from_packed
  - fused:    soft_update_step (if available)

Usage
  python examples/bench.py
  python examples/bench.py --n 300000 --max-window 128 --repeats 5
  python examples/bench.py --dt-jitter 0.1
  python examples/bench.py --micro-core --core-iters 200000

Notes
- This is a microbenchmark. For stable numbers:
  - run on AC power, disable CPU scaling/turbo if you care,
  - pin to one core (Linux): taskset -c 0 python examples/bench.py ...
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from owrf1d import OnlineWindowRegressor1D
from owrf1d.filter import HAVE_CYTHON_CORE


@dataclass(frozen=True)
class SeriesCfg:
    n: int
    seed: int
    dt_mean: float = 1.0
    dt_jitter: float = 0.0

    # piecewise drift
    cp: Optional[int] = None
    a0: float = 0.0
    b1: float = -0.7
    b2: float = 0.2

    # variance jump
    cp2: Optional[int] = None
    sigma1: float = 2.08
    sigma2: float = 7.16


def _gen_series(cfg: SeriesCfg) -> Tuple[List[float], List[float]]:
    """
    Returns (dts, ys). Always returns dts list for simplicity.
    """
    rng = random.Random(cfg.seed)

    cp = cfg.cp if cfg.cp is not None else cfg.n // 3
    cp2 = cfg.cp2 if cfg.cp2 is not None else (2 * cfg.n) // 3

    t = 0.0
    mu = cfg.a0
    slope = cfg.b1

    dts: List[float] = [0.0] * cfg.n
    ys: List[float] = [0.0] * cfg.n

    for i in range(cfg.n):
        if cfg.dt_jitter != 0.0:
            dt = max(1e-6, cfg.dt_mean + cfg.dt_jitter * rng.uniform(-1.0, 1.0))
        else:
            dt = cfg.dt_mean

        t += dt

        if i == cp:
            slope = cfg.b2

        sigma = cfg.sigma1 if i < cp2 else cfg.sigma2

        mu = mu + slope * dt
        y = mu + rng.gauss(0.0, sigma)

        dts[i] = dt
        ys[i] = y

    return dts, ys


def _now_ns() -> int:
    return time.perf_counter_ns()


def _fmt_ns_per_op(ns_per_op: float) -> str:
    if ns_per_op < 1e3:
        return f"{ns_per_op:.1f} ns/op"
    if ns_per_op < 1e6:
        return f"{ns_per_op/1e3:.2f} µs/op"
    return f"{ns_per_op/1e6:.2f} ms/op"


def _bench_update(
    *,
    make_filter: Callable[[], OnlineWindowRegressor1D],
    dts: List[float],
    ys: List[float],
    warmup: int,
) -> float:
    """
    Returns ns/op for a single run.
    """
    f = make_filter()

    # warmup (not measured)
    w = min(warmup, len(ys))
    for i in range(w):
        f.update(ys[i], dt=dts[i])

    t0 = _now_ns()
    for i in range(w, len(ys)):
        f.update(ys[i], dt=dts[i])
    t1 = _now_ns()

    n_ops = max(1, len(ys) - w)
    return (t1 - t0) / float(n_ops)


def _print_header(args: argparse.Namespace) -> None:
    print("==== owrf1d bench ====")
    print(f"python:     {sys.version.split()[0]}  ({sys.executable})")
    print(f"platform:   {platform.platform()}")
    print(f"pid:        {os.getpid()}")
    print(f"HAVE_CYTHON_CORE: {HAVE_CYTHON_CORE}")
    print("")
    print("config:")
    print(f"  n:         {args.n}")
    print(f"  repeats:   {args.repeats}")
    print(f"  warmup:    {args.warmup}")
    print(f"  max_window:{args.max_window}")
    print(f"  min_window:{args.min_window}")
    print(f"  history:   {args.history}")
    print(f"  dt_mean:   {args.dt_mean}")
    print(f"  dt_jitter: {args.dt_jitter}")
    print("")


def _run_matrix(
    *,
    dts: List[float],
    ys: List[float],
    args: argparse.Namespace,
) -> None:
    cases: List[Tuple[str, str, bool]] = []

    def add(name: str, selection: str, use_core: bool) -> None:
        cases.append((name, selection, use_core))

    # default matrix
    add("hard-py", "hard", False)
    add("soft-py", "soft", False)
    add("hard-core", "hard", True)
    add("soft-core", "soft", True)

    # allow filtering
    if args.modes:
        wanted = {m.strip() for m in args.modes.split(",") if m.strip()}
        cases = [c for c in cases if c[0] in wanted]

    results: Dict[str, List[float]] = {}

    for name, selection, use_core in cases:
        if use_core and not HAVE_CYTHON_CORE:
            continue

        def make_filter() -> OnlineWindowRegressor1D:
            f = OnlineWindowRegressor1D(
                max_window=args.max_window,
                min_window=args.min_window,
                history=args.history,
                selection=selection,
            )
            # Force backend for *this* instance (no need to restart process)
            f._use_core = bool(use_core) and HAVE_CYTHON_CORE  # type: ignore[attr-defined]
            return f

        ns_ops: List[float] = []
        for _ in range(args.repeats):
            ns_ops.append(
                _bench_update(make_filter=make_filter, dts=dts, ys=ys, warmup=args.warmup)
            )
        results[name] = ns_ops

    if not results:
        print("No benchmark cases to run (check --modes and HAVE_CYTHON_CORE).")
        return

    # aggregate + print table
    rows: List[Tuple[str, float, float]] = []
    for name, xs in results.items():
        med = statistics.median(xs)
        # robust-ish dispersion: median absolute deviation (MAD) scaled
        mad = statistics.median([abs(v - med) for v in xs])
        rows.append((name, med, mad))

    best = min(r[1] for r in rows)

    print("end-to-end update() throughput:")
    print(f"{'case':<12} {'ns/op':>12} {'updates/s':>12} {'rel':>8}  {'spread(MAD)':>14}")
    for name, med, mad in sorted(rows, key=lambda r: r[1]):
        upd_s = 1e9 / med if med > 0 else float("inf")
        rel = med / best if best > 0 else float("nan")
        print(
            f"{name:<12} {med:>12.1f} {upd_s:>12.0f} {rel:>8.2f}  {_fmt_ns_per_op(mad):>14}"
        )
    print("")


def _microbench_core(args: argparse.Namespace) -> None:
    """
    Core-only microbench for soft path (fixed ring state):
      - two-step: select_student_t + soft_step_from_packed
      - fused:    soft_update_step (if available)
    """
    if not HAVE_CYTHON_CORE:
        print("core microbench skipped: HAVE_CYTHON_CORE is False")
        return

    import owrf1d._core as core  # type: ignore

    if not hasattr(core, "select_student_t") or not hasattr(core, "soft_step_from_packed"):
        print("core microbench skipped: core functions not found")
        return

    has_fused = hasattr(core, "soft_update_step")

    # Fill a ring with some data once
    f = OnlineWindowRegressor1D(
        max_window=args.max_window,
        min_window=args.min_window,
        history=0,
        selection="soft",
    )
    f._use_core = True  # type: ignore[attr-defined]

    # seed ring
    rng = random.Random(args.seed)
    for _ in range(max(args.max_window + 8, 64)):
        y = rng.gauss(0.0, 1.0)
        f.update(y, dt=args.dt_mean)

    t_buf = f._t_buf.view()  # type: ignore[attr-defined]
    y_buf = f._y_buf.view()  # type: ignore[attr-defined]
    head = int(f._t_buf.head)  # type: ignore[attr-defined]
    size = int(f._t_buf.size)  # type: ignore[attr-defined]

    d_used = float(args.dt_mean)
    y_t = 0.123456789

    min_w = int(args.min_window)
    W = int(args.max_window)
    max_eff = W
    soft_cap = W

    tau_min, tau_max, tau_boot = 2.0, 3.0, 2.0
    r_min, r_max, cap_beta = 0.7, 3.0, 0.010

    def bench_two_step(iters: int) -> float:
        t0 = _now_ns()
        for _ in range(iters):
            (n_h, sc1, sc2, pm, ps2, nu, f_sel, K, packed) = core.select_student_t(
                t_buf, y_buf, head, size, float(y_t), float(d_used), min_w, max_eff, True
            )
            if K > 0:
                core.soft_step_from_packed(
                    packed,
                    int(K),
                    float(d_used),
                    float(y_t),
                    int(soft_cap),
                    int(W),
                    int(min_w),
                    float(tau_min),
                    float(tau_max),
                    float(tau_boot),
                    float(r_min),
                    float(r_max),
                    float(cap_beta),
                )
        t1 = _now_ns()
        return (t1 - t0) / float(max(1, iters))

    def bench_fused(iters: int) -> float:
        t0 = _now_ns()
        for _ in range(iters):
            core.soft_update_step(
                t_buf,
                y_buf,
                head,
                size,
                float(y_t),
                float(d_used),
                int(min_w),
                int(max_eff),
                int(soft_cap),
                int(W),
                float(tau_min),
                float(tau_max),
                float(tau_boot),
                float(r_min),
                float(r_max),
                float(cap_beta),
            )
        t1 = _now_ns()
        return (t1 - t0) / float(max(1, iters))

    print("core-only soft microbench (fixed ring state):")
    iters = int(args.core_iters)

    ns_two = bench_two_step(iters)
    print(f"  two-step: {_fmt_ns_per_op(ns_two)}  ({1e9/ns_two:,.0f} it/s)")

    if has_fused:
        ns_f = bench_fused(iters)
        speedup = ns_two / ns_f if ns_f > 0 else float("inf")
        print(f"  fused:    {_fmt_ns_per_op(ns_f)}  ({1e9/ns_f:,.0f} it/s)  speedup={speedup:.2f}x")
    else:
        print("  fused:    not available (no soft_update_step in core)")
    print("")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2_000)

    p.add_argument("--max-window", type=int, default=128)
    p.add_argument("--min-window", type=int, default=4)
    p.add_argument("--history", type=int, default=0)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dt-mean", type=float, default=1.0)
    p.add_argument("--dt-jitter", type=float, default=0.0)

    p.add_argument(
        "--modes",
        type=str,
        default="",
        help='Comma-separated cases to run (e.g. "hard-py,soft-core"). Empty = all.',
    )

    p.add_argument("--micro-core", action="store_true", help="also run core-only microbench")
    p.add_argument("--core-iters", type=int, default=200_000, help="iterations for --micro-core")

    args = p.parse_args()

    _print_header(args)

    cfg = SeriesCfg(
        n=int(args.n),
        seed=int(args.seed),
        dt_mean=float(args.dt_mean),
        dt_jitter=float(args.dt_jitter),
    )
    dts, ys = _gen_series(cfg)

    _run_matrix(dts=dts, ys=ys, args=args)

    if args.micro_core:
        _microbench_core(args)


if __name__ == "__main__":
    main()
