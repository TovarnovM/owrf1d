from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Deque

import math
import os
from array import array
from collections import deque

from .flags import (
    FLAG_PREDICT_ONLY,
    FLAG_INSUFFICIENT_DATA,
    FLAG_DEGENERATE_XTX,
    FLAG_NEGATIVE_SSE,
    FLAG_NUMERIC_GUARD,
    FLAG_HISTORY_TRUNC,
)

_EPS = 1e-12
_TIE_TOL = 1e-12
_WINDOW_PRIOR_WEIGHT = 0.5

# Soft selection internals (kept for Python fallback only)
_TAU_MIN = 2.0
_TAU_MAX = 3.0
_TAU_BOOT = 2.0
_CAP_R_MIN = 0.7
_CAP_R_MAX = 3.0
_CAP_BETA = 0.010

# Packed candidates layout (must match src/owrf1d/_core.pyx)
_PACK_STRIDE = 10
_P_K = 0
_P_SCORE = 1
_P_PRED_MU = 2
_P_PRED_S2 = 3
_P_SX = 4
_P_SXX = 5
_P_SY = 6
_P_SXY = 7
_P_SYY = 8
_P_FLAGS = 9

try:
    from ._core import select_student_t as _cy_select_student_t  # type: ignore
    from ._core import soft_step_from_packed as _cy_soft_step_from_packed  # type: ignore
    from ._core import soft_update_step as _cy_soft_update_step  # type: ignore
except Exception:  # pragma: no cover
    _cy_select_student_t = None
    _cy_soft_step_from_packed = None
    _cy_soft_update_step = None

HAVE_CYTHON_CORE: bool = _cy_select_student_t is not None


@dataclass
class _State:
    mu: float = 0.0
    trend: float = 0.0
    sigma2: float = 1.0
    n_star: int = 0
    t: float = 0.0
    has_t: bool = False


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


class _RingBufferD:
    __slots__ = ("cap", "buf", "head", "size")

    def __init__(self, cap: int) -> None:
        if cap <= 0:
            raise ValueError("cap must be positive")
        self.cap = int(cap)
        self.buf = array("d", [0.0]) * self.cap
        self.head = 0
        self.size = 0

    def append(self, x: float) -> None:
        self.buf[self.head] = float(x)
        self.head = (self.head + 1) % self.cap
        if self.size < self.cap:
            self.size += 1

    def last(self) -> float:
        if self.size <= 0:
            raise IndexError("empty")
        return float(self.buf[(self.head - 1) % self.cap])

    def get_last_n_pair(self, other: "_RingBufferD", n: int) -> Tuple[List[float], List[float]]:
        if self.cap != other.cap:
            raise ValueError("cap mismatch")
        n = int(n)
        if n <= 0:
            return [], []
        n = min(n, self.size, other.size)
        start = (self.head - n) % self.cap
        ts: List[float] = []
        ys: List[float] = []
        for i in range(n):
            idx = (start + i) % self.cap
            ts.append(float(self.buf[idx]))
            ys.append(float(other.buf[idx]))
        return ts, ys

    def view(self) -> memoryview:
        return memoryview(self.buf)


def _ols_from_sums(
    n: int, sx: float, sxx: float, sy: float, sxy: float, syy: float
) -> Tuple[float, float, float, float, int]:
    flags = 0
    if n <= 0:
        return 0.0, 0.0, 1.0, 0.0, FLAG_NUMERIC_GUARD

    D = n * sxx - sx * sx
    if (not _finite(D)) or D <= _EPS:
        return 0.0, 0.0, 1.0, 0.0, FLAG_DEGENERATE_XTX

    b = (n * sxy - sx * sy) / D
    a = (sy - b * sx) / n

    sse = (syy - (sy * sy) / n) - (b * b) * (sxx - (sx * sx) / n)
    if not _finite(sse):
        flags |= FLAG_NUMERIC_GUARD
        sse = 0.0
    if sse < 0.0:
        flags |= FLAG_NEGATIVE_SSE
        sse = 0.0

    df = n - 2
    sigma2 = sse / df if df > 0 else max(sse, 0.0)
    if (not _finite(sigma2)) or sigma2 < _EPS:
        flags |= FLAG_NUMERIC_GUARD
        sigma2 = _EPS

    if (not _finite(a)) or (not _finite(b)):
        flags |= FLAG_NUMERIC_GUARD
        a = 0.0 if not _finite(a) else a
        b = 0.0 if not _finite(b) else b

    return float(a), float(b), float(sigma2), float(D), int(flags)


def _student_t_loglik(e: float, s2: float, nu: int) -> float:
    if (not _finite(e)) or (not _finite(s2)) or s2 <= 0.0 or nu <= 0:
        return -1e300
    half = 0.5 * float(nu)
    log_norm = (
        math.lgamma(half + 0.5)
        - math.lgamma(half)
        - 0.5 * math.log(float(nu) * math.pi * float(s2))
    )
    quad = 1.0 + (e * e) / (float(nu) * float(s2))
    if (not _finite(quad)) or quad <= 0.0:
        return -1e300
    return float(log_norm - (half + 0.5) * math.log(quad))


def _softmax_weights(log_scores: List[float], tau: float) -> Tuple[List[float], float, float]:
    if not log_scores:
        return [], 0.0, 0.0
    if (not _finite(tau)) or tau <= 0.0:
        tau = 1.0
    m = max(log_scores)
    exps = [math.exp((s - m) / tau) for s in log_scores]
    z = sum(exps)
    if (not _finite(z)) or z <= 0.0:
        return [], 0.0, 0.0
    ws = [v / z for v in exps]
    w_star = max(ws) if ws else 0.0
    ent = 0.0
    for w in ws:
        if w > 0.0:
            ent -= w * math.log(w)
    return ws, float(w_star), float(ent)


def _entropy_norm(entropy: float, k: int) -> float:
    if k <= 1:
        return 0.0
    denom = math.log(float(k))
    if denom <= 0.0:
        return 0.0
    h = float(entropy) / denom
    if not _finite(h):
        return 0.0
    return float(min(1.0, max(0.0, h)))


def _post_params_from_pre_sums(
    *,
    k: int,
    d: float,
    y_t: float,
    sx: float,
    sxx: float,
    sy: float,
    sxy: float,
    syy: float,
) -> Tuple[float, float, float, int]:
    if k <= 0:
        return 0.0, 0.0, _EPS, FLAG_NUMERIC_GUARD

    d_f = float(d)
    if (not _finite(d_f)) or d_f <= 0.0:
        d_f = 1.0

    sx_post = sx - float(k) * d_f
    sxx_post = sxx - 2.0 * d_f * sx + float(k) * d_f * d_f
    sxy_post = sxy - d_f * sy

    N = k + 1
    sy_N = sy + y_t
    syy_N = syy + y_t * y_t

    a, b, sigma2, _, f = _ols_from_sums(N, sx_post, sxx_post, sy_N, sxy_post, syy_N)
    return float(a), float(b), float(max(sigma2, _EPS)), int(f)


def _py_select_student_t_from_ring(
    *,
    t_buf: _RingBufferD,
    y_buf: _RingBufferD,
    y_t: float,
    d: float,
    min_window: int,
    max_window_effective: int,
    collect_candidates: bool,
) -> Tuple[int, float, float, float, float, int, int, int, array]:
    flags = 0
    packed_list: List[float] = []

    n_avail = y_buf.size
    n_max = min(max_window_effective, n_avail)
    if n_max < min_window:
        return 0, 0.0, 0.0, 0.0, _EPS, 0, flags, 0, array("d")

    x_origin = t_buf.last()
    d_f = float(d)
    if (not _finite(d_f)) or d_f <= 0.0:
        flags |= FLAG_NUMERIC_GUARD
        d_f = 1.0

    sx = sxx = sy = sxy = syy = 0.0

    best_n = 0
    best_score = -1e300
    best_second = -1e300
    best_pred_mu = 0.0
    best_pred_s2 = _EPS
    best_nu = 0
    best_flags = 0

    any_valid = False
    any_degenerate = False

    cap = t_buf.cap
    head = t_buf.head

    for k in range(1, n_max + 1):
        idx = (head - k) % cap
        t_i = float(t_buf.buf[idx])
        y_i = float(y_buf.buf[idx])

        x_i = t_i - x_origin
        sx += x_i
        sxx += x_i * x_i
        sy += y_i
        sxy += x_i * y_i
        syy += y_i * y_i

        if k < min_window:
            continue

        a, b, sigma2, D, f_ols = _ols_from_sums(k, sx, sxx, sy, sxy, syy)
        if (f_ols & FLAG_DEGENERATE_XTX) != 0:
            any_degenerate = True
            continue

        nu = k - 2
        if nu <= 0:
            continue

        any_valid = True
        pred_mu = a + b * d_f

        h_t = (sxx - 2.0 * sx * d_f + float(k) * d_f * d_f) / D
        if (not _finite(h_t)) or h_t < 0.0:
            f_ols |= FLAG_NUMERIC_GUARD
            h_t = 0.0

        pred_s2 = sigma2 * (1.0 + h_t)
        if (not _finite(pred_s2)) or pred_s2 <= _EPS:
            f_ols |= FLAG_NUMERIC_GUARD
            pred_s2 = max(sigma2, _EPS)

        e = y_t - pred_mu
        score = _student_t_loglik(e=e, s2=pred_s2, nu=nu) + _WINDOW_PRIOR_WEIGHT * math.log(float(k))

        if collect_candidates:
            packed_list.extend(
                [
                    float(k),
                    float(score),
                    float(pred_mu),
                    float(pred_s2),
                    float(sx),
                    float(sxx),
                    float(sy),
                    float(sxy),
                    float(syy),
                    float(int(f_ols)),
                ]
            )

        if (score > best_score + _TIE_TOL) or (abs(score - best_score) <= _TIE_TOL and k > best_n):
            best_second = best_score
            best_score = score
            best_n = k
            best_pred_mu = pred_mu
            best_pred_s2 = pred_s2
            best_nu = nu
            best_flags = f_ols
        elif score > best_second + _TIE_TOL:
            best_second = score

    if not any_valid:
        if any_degenerate:
            flags |= FLAG_DEGENERATE_XTX
        return 0, 0.0, 0.0, 0.0, _EPS, 0, flags, 0, array("d")

    flags |= best_flags
    if best_second <= -1e299:
        best_second = best_score

    packed = array("d", packed_list) if collect_candidates else array("d")
    K = (len(packed) // _PACK_STRIDE) if collect_candidates else 0

    return (
        int(best_n),
        float(best_score),
        float(best_second),
        float(best_pred_mu),
        float(best_pred_s2),
        int(best_nu),
        int(flags),
        int(K),
        packed,
    )


class OnlineWindowRegressor1D:
    def __init__(
        self,
        *,
        max_window: int = 128,
        min_window: int = 4,
        history: int = 0,
        selection: str = "soft",
    ) -> None:
        if max_window < 1:
            raise ValueError("max_window must be >= 1")
        if min_window < 3:
            raise ValueError("min_window must be >= 3 (recommended >= 4)")
        if history < -1:
            raise ValueError("history must be -1, 0, or positive int")
        if selection not in ("hard", "soft"):
            raise ValueError('selection must be "hard" or "soft"')

        self.max_window = int(max_window)
        self.min_window = int(min_window)
        self.history = int(history)
        self.selection = str(selection)

        cap = self.max_window + 1
        self._t_buf = _RingBufferD(cap)
        self._y_buf = _RingBufferD(cap)

        self._state = _State()
        self._hist: Deque[Dict[str, Any]] = deque(maxlen=None if self.history == -1 else self.history)

        self._soft_cap: int = int(self.max_window)
        self._use_core: bool = HAVE_CYTHON_CORE and (os.getenv("OWRF1D_FORCE_PY", "0") != "1")

    def get_state(self) -> Dict[str, Any]:
        return {
            "mu": self._state.mu,
            "trend": self._state.trend,
            "sigma2": self._state.sigma2,
            "n_star": self._state.n_star,
            "t": self._state.t,
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._hist)

    def dumps(self) -> bytes:
        import cloudpickle
        return cloudpickle.dumps(self)

    @staticmethod
    def loads(data: bytes) -> "OnlineWindowRegressor1D":
        import cloudpickle
        obj = cloudpickle.loads(data)
        if not isinstance(obj, OnlineWindowRegressor1D):
            raise TypeError("Loaded object is not OnlineWindowRegressor1D")
        return obj

    def _push_history(self, step: Dict[str, Any]) -> None:
        if self.history == 0:
            return
        if self.history > 0 and len(self._hist) == self._hist.maxlen:
            step["flags"] = int(step.get("flags", 0)) | FLAG_HISTORY_TRUNC
        self._hist.append(step)

    def _advance_time(self, *, t: Optional[float], dt: Optional[float]) -> Tuple[float, float, int]:
        flags = 0
        if dt is not None and t is not None:
            flags |= FLAG_NUMERIC_GUARD  # deterministic: dt wins

        if dt is not None:
            dt_f = float(dt)
            if (not _finite(dt_f)) or dt_f <= 0.0:
                flags |= FLAG_NUMERIC_GUARD
                dt_f = 1.0
            if self._state.has_t:
                t_f = self._state.t + dt_f
            else:
                t_f = dt_f
                self._state.has_t = True
            self._state.t = t_f
            return t_f, dt_f, flags

        if t is not None:
            t_f = float(t)
            if not _finite(t_f):
                flags |= FLAG_NUMERIC_GUARD
                t_f = self._state.t + 1.0 if self._state.has_t else 0.0
            if self._state.has_t:
                dt_f = t_f - self._state.t
                if (not _finite(dt_f)) or dt_f <= 0.0:
                    flags |= FLAG_NUMERIC_GUARD
                    dt_f = 1.0
            else:
                dt_f = 1.0
                self._state.has_t = True
            self._state.t = t_f
            return t_f, dt_f, flags

        dt_f = 1.0
        t_f = self._state.t + dt_f if self._state.has_t else 0.0
        self._state.has_t = True
        self._state.t = t_f
        return t_f, dt_f, flags

    def update(
        self,
        y: float | None,
        *,
        t: float | None = None,
        dt: float | None = None,
    ) -> Dict[str, Any]:
        t_now, dt_now, time_flags = self._advance_time(t=t, dt=dt)
        flags = int(time_flags)

        if y is None:
            flags |= FLAG_PREDICT_ONLY
            step = self._make_step(
                y=None,
                t=t_now,
                dt=dt_now,
                flags=flags,
                pred_mu=self._state.mu,
                pred_s2=max(self._state.sigma2, _EPS),
                score_star=0.0,
                score_second=0.0,
                n_star=self._state.n_star,
                nu=max(int(self._state.n_star) - 2, 0),
                extras={"selection": self.selection},
            )
            self._push_history(step)
            return step

        y_f = float(y)
        if not _finite(y_f):
            flags |= FLAG_NUMERIC_GUARD
            y_f = 0.0

        d_used = float(dt_now)
        if self._t_buf.size > 0:
            dt_obs = float(t_now) - self._t_buf.last()
            if _finite(dt_obs) and dt_obs > 0.0:
                d_used = dt_obs
        if (not _finite(d_used)) or d_used <= 0.0:
            flags |= FLAG_NUMERIC_GUARD
            d_used = 1.0

        pred_mu = self._state.mu + self._state.trend * d_used
        pred_s2 = max(self._state.sigma2, _EPS)
        score_star = 0.0
        score_second = 0.0
        nu_sel = 0
        n_hard = 0

        n_eff = 0.0
        tau_t = 0.0
        cap_before = int(self._soft_cap)
        cap_target = int(self._soft_cap)
        h_norm = 0.0
        w_star = 0.0
        ent = 0.0

        K = 0
        packed = array("d")

        used_core_soft = False
        mu_post = float(y_f)
        trend_post = 0.0
        sigma2_post = float(max(self._state.sigma2, _EPS))
        sigma2_total = float(max(self._state.sigma2, _EPS))
        n_final = 0

        n_avail = self._y_buf.size
        if n_avail < self.min_window:
            flags |= FLAG_INSUFFICIENT_DATA
        else:
            max_eff = self.max_window
            if self.selection == "soft":
                max_eff = min(self.max_window, max(self.min_window, int(self._soft_cap)))

            if self._use_core and HAVE_CYTHON_CORE:
                (n_hard, score_star, score_second, pred_mu_star, pred_s2_star, nu_sel, f_sel, K, packed) = _cy_select_student_t(
                    self._t_buf.view(),
                    self._y_buf.view(),
                    int(self._t_buf.head),
                    int(self._t_buf.size),
                    float(y_f),
                    float(d_used),
                    int(self.min_window),
                    int(max_eff),
                    bool(self.selection == "soft"),
                )
            else:
                (n_hard, score_star, score_second, pred_mu_star, pred_s2_star, nu_sel, f_sel, K, packed) = _py_select_student_t_from_ring(
                    t_buf=self._t_buf,
                    y_buf=self._y_buf,
                    y_t=y_f,
                    d=d_used,
                    min_window=self.min_window,
                    max_window_effective=max_eff,
                    collect_candidates=(self.selection == "soft"),
                )

            flags |= int(f_sel)

            if n_hard > 0:
                if self.selection == "hard":
                    pred_mu = float(pred_mu_star)
                    pred_s2 = float(pred_s2_star)
                    n_eff = float(n_hard)
                else:
                    # Preferred: full soft path in Cython (no Python loops on candidates)
                    if self._use_core and (_cy_soft_update_step is not None):
                        (
                            n_hard,
                            score_star,
                            score_second,
                            pred_mu,
                            pred_s2,
                            nu_sel,
                            f_core,
                            n_eff,
                            tau_t,
                            w_star,
                            ent,
                            h_norm,
                            cap_target,
                            cap_new,
                            mu_post,
                            trend_post,
                            sigma2_post,
                            sigma2_total,
                            n_final,
                        ) = _cy_soft_update_step(
                            self._t_buf.view(),
                            self._y_buf.view(),
                            int(self._t_buf.head),
                            int(self._t_buf.size),
                            float(y_f),
                            float(d_used),
                            int(self.min_window),
                            int(max_eff),
                            int(self._soft_cap),
                            int(self.max_window),
                            float(_TAU_MIN),
                            float(_TAU_MAX),
                            float(_TAU_BOOT),
                            float(_CAP_R_MIN),
                            float(_CAP_R_MAX),
                            float(_CAP_BETA),
                        )
                        flags |= int(f_core)
                        self._soft_cap = int(cap_new)
                        used_core_soft = True
                    elif (
                        self._use_core
                        and (_cy_soft_step_from_packed is not None)
                        and K > 0
                    ):
                        (
                            pred_mu,
                            pred_s2,
                            n_eff,
                            tau_t,
                            w_star,
                            ent,
                            h_norm,
                            cap_target,
                            cap_new,
                            mu_post,
                            trend_post,
                            sigma2_post,
                            sigma2_total,
                            f_add,
                            n_final,
                        ) = _cy_soft_step_from_packed(
                            packed,
                            int(K),
                            float(d_used),
                            float(y_f),
                            int(self._soft_cap),
                            int(self.max_window),
                            int(self.min_window),

                            float(_TAU_MIN),
                            float(_TAU_MAX),
                            float(_TAU_BOOT),
                            float(_CAP_R_MIN),
                            float(_CAP_R_MAX),
                            float(_CAP_BETA),
                        )
                        flags |= int(f_add)
                        self._soft_cap = int(cap_new)
                        used_core_soft = True
                    else:
                        # Python fallback
                        if K > 0:
                            log_scores = [packed[i * _PACK_STRIDE + _P_SCORE] for i in range(K)]
                            ws0, _, ent0 = _softmax_weights(log_scores, tau=_TAU_BOOT)
                            h0 = _entropy_norm(ent0, len(ws0)) if ws0 else 0.0
                            tau_t = _TAU_MIN + (_TAU_MAX - _TAU_MIN) * h0

                            ws, w_star, ent = _softmax_weights(log_scores, tau=tau_t)
                            h_norm = _entropy_norm(ent, len(ws)) if ws else 0.0

                            if ws:
                                pred_mu = 0.0
                                for i, w in enumerate(ws):
                                    pred_mu += w * packed[i * _PACK_STRIDE + _P_PRED_MU]
                                pred_s2 = 0.0
                                for i, w in enumerate(ws):
                                    mu_i = packed[i * _PACK_STRIDE + _P_PRED_MU]
                                    dm = mu_i - pred_mu
                                    pred_s2 += w * (packed[i * _PACK_STRIDE + _P_PRED_S2] + dm * dm)
                                pred_s2 = float(max(pred_s2, _EPS))
                                n_eff = 0.0
                                for i, w in enumerate(ws):
                                    n_eff += w * packed[i * _PACK_STRIDE + _P_K]
                            else:
                                pred_mu = float(pred_mu_star)
                                pred_s2 = float(pred_s2_star)
                                n_eff = float(n_hard)

                            r = _CAP_R_MIN + (_CAP_R_MAX - _CAP_R_MIN) * (1.0 - h_norm)
                            cap_target = int(round(n_eff * r))
                            cap_target = int(min(self.max_window, max(self.min_window, cap_target)))
                            cap_new = int(round((1.0 - _CAP_BETA) * float(self._soft_cap) + _CAP_BETA * float(cap_target)))
                            cap_new = int(min(self.max_window, max(self.min_window, cap_new)))
                            self._soft_cap = cap_new
                        else:
                            pred_mu = float(pred_mu_star)
                            pred_s2 = float(pred_s2_star)
                            n_eff = float(n_hard)

        # append current point
        self._t_buf.append(float(t_now))
        self._y_buf.append(float(y_f))

        # post update
        if self.selection == "hard":
            if n_hard > 0:
                N = min(n_hard + 1, self._y_buf.size)
                ts_post, ys_post = self._t_buf.get_last_n_pair(self._y_buf, N)
                x_origin_post = ts_post[-1]
                sx = sxx = sy = sxy = syy = 0.0
                for ti, yi in zip(ts_post, ys_post, strict=True):
                    x = float(ti) - float(x_origin_post)
                    sx += x
                    sxx += x * x
                    sy += float(yi)
                    sxy += x * float(yi)
                    syy += float(yi) * float(yi)

                a_post2, b_post2, sigma2_fit, _, f3 = _ols_from_sums(N, sx, sxx, sy, sxy, syy)
                flags |= int(f3)

                mu_post = float(a_post2)
                trend_post = float(b_post2)
                sigma2_post = float(max(sigma2_fit, _EPS))
                sigma2_total = sigma2_post
                n_final = int(n_hard)
            else:
                mu_post = float(y_f)
                trend_post = 0.0
                sigma2_post = float(max(self._state.sigma2, _EPS))
                sigma2_total = sigma2_post
                n_final = 0
        else:
            if used_core_soft:
                # already computed in Cython
                pass
            else:
                # Python soft post (fallback)
                if n_avail >= self.min_window and K > 0:
                    log_scores = [packed[i * _PACK_STRIDE + _P_SCORE] for i in range(K)]
                    if not _finite(tau_t) or tau_t <= 0.0:
                        ws0, _, ent0 = _softmax_weights(log_scores, tau=_TAU_BOOT)
                        h0 = _entropy_norm(ent0, len(ws0)) if ws0 else 0.0
                        tau_t = _TAU_MIN + (_TAU_MAX - _TAU_MIN) * h0

                    ws, w_star, ent = _softmax_weights(log_scores, tau=tau_t)
                    h_norm = _entropy_norm(ent, len(ws)) if ws else h_norm

                    if ws:
                        mu_mix = 0.0
                        b_mix = 0.0
                        s2_noise = 0.0
                        mus_local: List[float] = []

                        for i, w in enumerate(ws):
                            base = i * _PACK_STRIDE
                            k = int(packed[base + _P_K])
                            sx_k = packed[base + _P_SX]
                            sxx_k = packed[base + _P_SXX]
                            sy_k = packed[base + _P_SY]
                            sxy_k = packed[base + _P_SXY]
                            syy_k = packed[base + _P_SYY]

                            a_k, b_k, s2_k, f_k = _post_params_from_pre_sums(
                                k=k,
                                d=d_used,
                                y_t=y_f,
                                sx=sx_k,
                                sxx=sxx_k,
                                sy=sy_k,
                                sxy=sxy_k,
                                syy=syy_k,
                            )
                            flags |= int(f_k) | int(packed[base + _P_FLAGS])
                            mu_mix += w * a_k
                            b_mix += w * b_k
                            s2_noise += w * s2_k
                            mus_local.append(a_k)

                        s2_model = 0.0
                        for w, a_k in zip(ws, mus_local, strict=True):
                            dm = a_k - mu_mix
                            s2_model += w * dm * dm

                        mu_post = float(mu_mix)
                        trend_post = float(b_mix)
                        sigma2_post = float(max(s2_noise, _EPS))
                        sigma2_total = float(max(s2_noise + s2_model, _EPS))
                        n_final = int(round(n_eff)) if n_eff > 0.0 else int(n_hard if n_hard > 0 else 0)
                    else:
                        mu_post = float(y_f)
                        trend_post = 0.0
                        sigma2_post = float(max(self._state.sigma2, _EPS))
                        sigma2_total = sigma2_post
                        n_final = int(n_hard if n_hard > 0 else 0)
                else:
                    mu_post = float(y_f)
                    trend_post = 0.0
                    sigma2_post = float(max(self._state.sigma2, _EPS))
                    sigma2_total = sigma2_post
                    n_final = 0

        if (not _finite(mu_post)) or (not _finite(trend_post)) or (not _finite(sigma2_post)):
            flags |= FLAG_NUMERIC_GUARD
            mu_post = float(y_f) if not _finite(mu_post) else float(mu_post)
            trend_post = 0.0 if not _finite(trend_post) else float(trend_post)
            sigma2_post = _EPS if not _finite(sigma2_post) else float(max(sigma2_post, _EPS))

        self._state.mu = float(mu_post)
        self._state.trend = float(trend_post)
        self._state.sigma2 = float(max(sigma2_post, _EPS))
        self._state.n_star = int(n_final)

        extras: Dict[str, Any] = {"selection": self.selection}
        if self.selection == "soft":
            extras.update(
                {
                    "n_star_hard": int(n_hard),
                    "n_eff": float(n_eff) if n_eff > 0.0 else float(n_final),
                    "w_star": float(w_star),
                    "entropy": float(ent),
                    "entropy_norm": float(h_norm),
                    "tau": float(tau_t) if _finite(tau_t) else 0.0,
                    "cap": int(self._soft_cap),
                    "cap_before": int(cap_before),
                    "cap_target": int(cap_target),
                    "sigma2_total": float(sigma2_total),
                    "d_used": float(d_used),
                    "core": bool(self._use_core and HAVE_CYTHON_CORE),
                }
            )

        step = self._make_step(
            y=y_f,
            t=t_now,
            dt=dt_now,
            flags=flags,
            pred_mu=float(pred_mu),
            pred_s2=float(max(pred_s2, _EPS)),
            score_star=float(score_star),
            score_second=float(score_second),
            n_star=int(self._state.n_star),
            nu=int(nu_sel),
            extras=extras,
        )
        self._push_history(step)
        return step

    def _make_step(
        self,
        *,
        y: Optional[float],
        t: float,
        dt: float,
        flags: int,
        pred_mu: float,
        pred_s2: float,
        score_star: float,
        score_second: float,
        n_star: int,
        nu: int,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resid = 0.0 if y is None else float(y) - float(self._state.mu)

        if not _finite(score_star):
            flags |= FLAG_NUMERIC_GUARD
            score_star = 0.0
        if not _finite(score_second):
            score_second = 0.0

        out: Dict[str, Any] = {
            "mu": float(self._state.mu),
            "trend": float(self._state.trend),
            "sigma2": float(max(self._state.sigma2, _EPS)),
            "n_star": int(n_star),
            "score_star": float(score_star),
            "score_second": float(score_second),
            "delta_score": float(score_star - score_second),
            "nu": int(max(int(nu), 0)),
            "pred_mu": float(pred_mu),
            "pred_s2": float(max(pred_s2, _EPS)),
            "resid": float(resid),
            "t": float(t),
            "dt": float(dt),
            "flags": int(flags),
        }
        if extras:
            out.update(extras)
        return out
