# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.math cimport log, lgamma, isfinite, fabs
cimport cython

cdef double _EPS = 1e-12
cdef double _TIE_TOL = 1e-12
cdef double _WINDOW_PRIOR_WEIGHT = 0.5

# IMPORTANT: must match src/owrf1d/flags.py
cdef int FLAG_DEGENERATE_XTX = 1 << 2
cdef int FLAG_NEGATIVE_SSE   = 1 << 3
cdef int FLAG_NUMERIC_GUARD  = 1 << 4


cdef inline int _finite(double x) nogil:
    return isfinite(x) != 0


cdef inline int _wrap_idx(int idx, int cap) nogil:
    """
    Python-style wrap for ring-buffer indices, but in C semantics.
    Assumption in our loops: idx in [-(cap-1), cap-1], so one +cap is enough.
    """
    if idx < 0:
        idx += cap
    elif idx >= cap:
        idx -= cap
    return idx


cdef inline double _student_t_loglik(double e, double s2, int nu) nogil:
    cdef double half, log_norm, quad
    if (not _finite(e)) or (not _finite(s2)) or s2 <= 0.0 or nu <= 0:
        return -1e300
    half = 0.5 * nu
    log_norm = lgamma(half + 0.5) - lgamma(half) - 0.5 * log(nu * 3.141592653589793 * s2)
    quad = 1.0 + (e * e) / (nu * s2)
    if (not _finite(quad)) or quad <= 0.0:
        return -1e300
    return log_norm - (half + 0.5) * log(quad)


@cython.cfunc
@cython.inline
cdef void _ols_from_sums(
    int n,
    double sx, double sxx, double sy, double sxy, double syy,
    double* a_out, double* b_out, double* sigma2_out, double* D_out,
    int* flags_out,
) nogil:
    cdef int flags = 0
    cdef double D, b, a, sse, df, sigma2

    if n <= 0:
        flags_out[0] |= FLAG_NUMERIC_GUARD
        a_out[0] = 0.0
        b_out[0] = 0.0
        sigma2_out[0] = 1.0
        D_out[0] = 0.0
        return

    D = n * sxx - sx * sx
    if (not _finite(D)) or D <= _EPS:
        flags_out[0] |= FLAG_DEGENERATE_XTX
        a_out[0] = 0.0
        b_out[0] = 0.0
        sigma2_out[0] = 1.0
        D_out[0] = 0.0
        return

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
    sigma2 = sse / df if df > 0.0 else sse
    if (not _finite(sigma2)) or sigma2 < _EPS:
        flags |= FLAG_NUMERIC_GUARD
        sigma2 = _EPS

    if (not _finite(a)) or (not _finite(b)):
        flags |= FLAG_NUMERIC_GUARD
        if not _finite(a):
            a = 0.0
        if not _finite(b):
            b = 0.0

    a_out[0] = a
    b_out[0] = b
    sigma2_out[0] = sigma2
    D_out[0] = D
    flags_out[0] |= flags


cpdef select_student_t(
    double[:] t_buf,
    double[:] y_buf,
    int head,
    int size,
    double y_t,
    double d,
    int min_window,
    int max_window_effective,
    bint collect_candidates,
):
    """
    Selection phase (pre): choose n* by maximizing predictive Student-t log-likelihood.

    Buffer semantics:
      - ring buffer arrays t_buf / y_buf of identical length (cap)
      - 'head' is next write position
      - 'size' is number of valid items (<= cap)
      - current y_t is NOT in buffers yet (strict online)

    Returns:
      (n_star, score_star, score_second, pred_mu_star, pred_s2_star, nu, flags, candidates)

    candidates: list of tuples
      (k, score, pred_mu, pred_s2, sx, sxx, sy, sxy, syy, flags_k)
      only when collect_candidates is True, else [].
    """
    cdef int flags = 0
    cdef int cap = t_buf.shape[0]
    cdef int n_avail = size
    cdef int n_max = max_window_effective
    cdef int k, idx_raw, idx
    cdef double x_origin, t_i, y_i, x_i
    cdef double sx = 0.0
    cdef double sxx = 0.0
    cdef double sy = 0.0
    cdef double sxy = 0.0
    cdef double syy = 0.0

    cdef int best_n = 0
    cdef double best_score = -1e300
    cdef double best_second = -1e300
    cdef double best_pred_mu = 0.0
    cdef double best_pred_s2 = _EPS
    cdef int best_nu = 0
    cdef int best_flags = 0

    cdef bint any_valid = False
    cdef bint any_degenerate = False

    cdef double a, b, sigma2, D
    cdef int f_ols
    cdef int nu
    cdef double pred_mu, pred_s2, h_t, e, score

    cdef object candidates = []
    if n_max > n_avail:
        n_max = n_avail
    if n_max < min_window:
        return 0, 0.0, 0.0, 0.0, _EPS, 0, 0, candidates

    if (not _finite(d)) or d <= 0.0:
        flags |= FLAG_NUMERIC_GUARD
        d = 1.0

    # last prior timestamp index = head-1 (Python wrap, not C %)
    idx_raw = head - 1
    idx = _wrap_idx(idx_raw, cap)
    x_origin = t_buf[idx]
    if not _finite(x_origin):
        flags |= FLAG_NUMERIC_GUARD
        x_origin = 0.0

    for k in range(1, n_max + 1):
        # idx = (head - k) mod cap in Python sense
        idx_raw = head - k
        idx = _wrap_idx(idx_raw, cap)

        t_i = t_buf[idx]
        y_i = y_buf[idx]

        x_i = t_i - x_origin
        sx += x_i
        sxx += x_i * x_i
        sy += y_i
        sxy += x_i * y_i
        syy += y_i * y_i

        if k < min_window:
            continue

        f_ols = 0
        _ols_from_sums(k, sx, sxx, sy, sxy, syy, &a, &b, &sigma2, &D, &f_ols)
        if (f_ols & FLAG_DEGENERATE_XTX) != 0:
            any_degenerate = True
            continue

        nu = k - 2
        if nu <= 0:
            continue

        any_valid = True
        pred_mu = a + b * d

        h_t = (sxx - 2.0 * sx * d + k * d * d) / D
        if (not _finite(h_t)) or h_t < 0.0:
            f_ols |= FLAG_NUMERIC_GUARD
            h_t = 0.0

        pred_s2 = sigma2 * (1.0 + h_t)
        if (not _finite(pred_s2)) or pred_s2 <= _EPS:
            f_ols |= FLAG_NUMERIC_GUARD
            pred_s2 = sigma2 if sigma2 > _EPS else _EPS

        e = y_t - pred_mu
        score = _student_t_loglik(e, pred_s2, nu) + _WINDOW_PRIOR_WEIGHT * log(<double>k)

        if collect_candidates:
            candidates.append((k, score, pred_mu, pred_s2, sx, sxx, sy, sxy, syy, f_ols))

        if (score > best_score + _TIE_TOL) or (fabs(score - best_score) <= _TIE_TOL and k > best_n):
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
        return 0, 0.0, 0.0, 0.0, _EPS, 0, flags, candidates

    flags |= best_flags
    if best_second <= -1e299:
        best_second = best_score

    return best_n, best_score, best_second, best_pred_mu, best_pred_s2, best_nu, flags, candidates
