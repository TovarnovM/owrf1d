# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import array as pyarray
from libc.math cimport log, lgamma, isfinite, fabs, exp, floor
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport cython

cdef double _EPS = 1e-12
cdef double _TIE_TOL = 1e-12
cdef double _WINDOW_PRIOR_WEIGHT = 0.5

# Packed layout: STRIDE=10 doubles per candidate
# [0]=k
# [1]=score
# [2]=pred_mu
# [3]=pred_s2
# [4]=sx
# [5]=sxx
# [6]=sy
# [7]=sxy
# [8]=syy
# [9]=flags_k  (stored as double; cast to int on Python side)
cdef int _STRIDE = 10

# IMPORTANT: must match src/owrf1d/flags.py
cdef int FLAG_DEGENERATE_XTX = 1 << 2
cdef int FLAG_NEGATIVE_SSE   = 1 << 3
cdef int FLAG_NUMERIC_GUARD  = 1 << 4


cdef inline int _finite(double x) nogil:
    return isfinite(x) != 0


cdef inline int _wrap_idx(int idx, int cap) nogil:
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

    Returns:
      (n_star, score_star, score_second, pred_mu_star, pred_s2_star, nu, flags, K, packed)

    packed: array('d') with STRIDE=10 per candidate, only when collect_candidates=True, else empty.
    K: number of candidates packed.
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

    cdef object packed
    cdef double[:] pview
    cdef int max_cands = 0
    cdef int m = 0
    cdef int base

    if n_max > n_avail:
        n_max = n_avail
    if n_max < min_window:
        packed = pyarray.array("d")
        return 0, 0.0, 0.0, 0.0, _EPS, 0, 0, 0, packed

    if (not _finite(d)) or d <= 0.0:
        flags |= FLAG_NUMERIC_GUARD
        d = 1.0

    # allocate packed for candidates if needed (upper bound)
    if collect_candidates:
        max_cands = n_max - min_window + 1
        if max_cands < 0:
            max_cands = 0
        packed = pyarray.array("d", [0.0]) * (max_cands * _STRIDE)
        pview = packed
    else:
        packed = pyarray.array("d")

    # last prior timestamp index = head-1 (Python wrap, not C %)
    idx_raw = head - 1
    idx = _wrap_idx(idx_raw, cap)
    x_origin = t_buf[idx]
    if not _finite(x_origin):
        flags |= FLAG_NUMERIC_GUARD
        x_origin = 0.0

    for k in range(1, n_max + 1):
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
            base = m * _STRIDE
            pview[base + 0] = <double>k
            pview[base + 1] = score
            pview[base + 2] = pred_mu
            pview[base + 3] = pred_s2
            pview[base + 4] = sx
            pview[base + 5] = sxx
            pview[base + 6] = sy
            pview[base + 7] = sxy
            pview[base + 8] = syy
            pview[base + 9] = <double>f_ols
            m += 1

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
        if collect_candidates and max_cands > 0:
            del packed[:]  # empty
        return 0, 0.0, 0.0, 0.0, _EPS, 0, flags, 0, packed

    flags |= best_flags
    if best_second <= -1e299:
        best_second = best_score

    if collect_candidates:
        # shrink to actual length
        if max_cands > 0 and m < max_cands:
            del packed[m * _STRIDE :]
        return best_n, best_score, best_second, best_pred_mu, best_pred_s2, best_nu, flags, m, packed

    return best_n, best_score, best_second, best_pred_mu, best_pred_s2, best_nu, flags, 0, packed



cpdef soft_step_from_packed(
    object packed,
    int K,
    double d,
    double y_t,
    int soft_cap,
    int max_window,
    int min_window,

    double tau_min=2.0,
    double tau_max=3.0,
    double tau_boot=2.0,
    double r_min=0.7,
    double r_max=3.0,
    double cap_beta=0.01,

):
    """
    Returns:
      (pred_mu, pred_s2, n_eff, tau_t, w_star, entropy, entropy_norm,
       cap_target, cap_new,
       mu_post, trend_post, sigma2_noise, sigma2_total,
       flags_add, n_final)
    """
    cdef double[:] p = packed
    cdef int flags_add = 0

    cdef int stride = 10  # packed stride

    # declare ALL cdef vars up-front (Cython requirement)
    cdef int i, base, k, fk, N, best_i, f_post
    cdef int cap_target, cap_new, n_final
    cdef double d_used, s, m, sumexp, tmp, w
    cdef double ent, logK, best_score
    cdef double h0, tau_t, h_norm, w_star

    cdef double pred_mu, pred_s2, n_eff
    cdef double Emu_pred, Emu2_pred, Evar_pred
    cdef double Ea, Ea2, Eb, Es2
    cdef double mu_i, s2_i
    cdef double sx, sxx, sy, sxy, syy
    cdef double sx_post, sxx_post, sxy_post, syN, syyN
    cdef double a_post, b_post, s2_post, Dpost
    cdef double s2_model, sigma2_noise, sigma2_total
    cdef double r

    # guard d
    d_used = d
    if (not _finite(d_used)) or d_used <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        d_used = 1.0

    # guard soft params (Python passes these; keep defensive for direct core calls)
    if (not _finite(tau_boot)) or tau_boot <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        tau_boot = 2.0
    if (not _finite(tau_min)) or tau_min <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        tau_min = 1.0
    if (not _finite(tau_max)) or tau_max <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        tau_max = tau_min
    if tau_max < tau_min:
        tmp = tau_max
        tau_max = tau_min
        tau_min = tmp

    if (not _finite(r_min)) or r_min <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        r_min = 1.0
    if (not _finite(r_max)) or r_max <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        r_max = r_min
    if r_max < r_min:
        tmp = r_max
        r_max = r_min
        r_min = tmp

    if (not _finite(cap_beta)):
        flags_add |= FLAG_NUMERIC_GUARD
        cap_beta = 0.010
    if cap_beta < 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        cap_beta = 0.0
    if cap_beta > 1.0:
        flags_add |= FLAG_NUMERIC_GUARD
        cap_beta = 1.0

    # trivial
    if K <= 0:
        return (
            0.0, _EPS, 0.0, tau_boot, 0.0, 0.0, 0.0,
            min_window, min_window,
            0.0, 0.0, _EPS, _EPS,
            flags_add, 0
        )

    logK = log(<double>K) if K > 1 else 0.0

    # find best_i for hard fallback
    best_score = -1e300
    best_i = 0
    for i in range(K):
        base = i * stride
        s = p[base + 1]  # score
        if s > best_score:
            best_score = s
            best_i = i

    # --------------- bootstrap softmax (tau_boot) to get entropy_norm -> tau_t
    m = -1e300
    for i in range(K):
        base = i * stride
        s = p[base + 1]
        if s > m:
            m = s

    sumexp = 0.0
    for i in range(K):
        base = i * stride
        s = p[base + 1]
        tmp = exp((s - m) / tau_boot)
        sumexp += tmp

    if (not _finite(sumexp)) or sumexp <= 0.0:
        # hard fallback (no cdef declarations here!)
        base = best_i * stride
        k = <int>p[base + 0]
        fk = <int>p[base + 9]
        flags_add |= fk

        pred_mu = p[base + 2]
        pred_s2 = p[base + 3]
        if (not _finite(pred_s2)) or pred_s2 <= _EPS:
            flags_add |= FLAG_NUMERIC_GUARD
            pred_s2 = _EPS

        sx = p[base + 4]
        sxx = p[base + 5]
        sy = p[base + 6]
        sxy = p[base + 7]
        syy = p[base + 8]

        sx_post = sx - (<double>k) * d_used
        sxx_post = sxx - 2.0 * d_used * sx + (<double>k) * d_used * d_used
        sxy_post = sxy - d_used * sy
        N = k + 1
        syN = sy + y_t
        syyN = syy + y_t * y_t

        f_post = 0
        _ols_from_sums(N, sx_post, sxx_post, syN, sxy_post, syyN, &a_post, &b_post, &s2_post, &Dpost, &f_post)
        flags_add |= f_post
        if (not _finite(s2_post)) or s2_post <= _EPS:
            flags_add |= FLAG_NUMERIC_GUARD
            s2_post = _EPS

        n_eff = <double>k
        r = r_max
        cap_target = <int>(n_eff * r + 0.5)
        if cap_target < min_window:
            cap_target = min_window
        if cap_target > max_window:
            cap_target = max_window

        cap_new = <int>(((1.0 - cap_beta) * soft_cap + cap_beta * cap_target) + 0.5)
        if cap_new < min_window:
            cap_new = min_window
        if cap_new > max_window:
            cap_new = max_window

        return (
            pred_mu, pred_s2, n_eff, tau_boot, 1.0, 0.0, 0.0,
            cap_target, cap_new,
            a_post, b_post, s2_post, s2_post,
            flags_add, k
        )

    # entropy under tau_boot
    ent = 0.0
    for i in range(K):
        base = i * stride
        s = p[base + 1]
        w = exp((s - m) / tau_boot) / sumexp
        if w > 0.0:
            ent -= w * log(w)

    h0 = (ent / logK) if (K > 1 and logK > 0.0) else 0.0
    if (not _finite(h0)) or h0 < 0.0:
        h0 = 0.0
    if h0 > 1.0:
        h0 = 1.0

    tau_t = tau_min + (tau_max - tau_min) * h0
    if (not _finite(tau_t)) or tau_t <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        tau_t = tau_boot

    # --------------- main softmax with tau_t
    m = -1e300
    for i in range(K):
        base = i * stride
        s = p[base + 1]
        if s > m:
            m = s

    sumexp = 0.0
    for i in range(K):
        base = i * stride
        s = p[base + 1]
        tmp = exp((s - m) / tau_t)
        sumexp += tmp

    if (not _finite(sumexp)) or sumexp <= 0.0:
        # hard fallback again
        base = best_i * stride
        k = <int>p[base + 0]
        fk = <int>p[base + 9]
        flags_add |= fk

        pred_mu = p[base + 2]
        pred_s2 = p[base + 3]
        if (not _finite(pred_s2)) or pred_s2 <= _EPS:
            flags_add |= FLAG_NUMERIC_GUARD
            pred_s2 = _EPS

        sx = p[base + 4]
        sxx = p[base + 5]
        sy = p[base + 6]
        sxy = p[base + 7]
        syy = p[base + 8]

        sx_post = sx - (<double>k) * d_used
        sxx_post = sxx - 2.0 * d_used * sx + (<double>k) * d_used * d_used
        sxy_post = sxy - d_used * sy
        N = k + 1
        syN = sy + y_t
        syyN = syy + y_t * y_t

        f_post = 0
        _ols_from_sums(N, sx_post, sxx_post, syN, sxy_post, syyN, &a_post, &b_post, &s2_post, &Dpost, &f_post)
        flags_add |= f_post
        if (not _finite(s2_post)) or s2_post <= _EPS:
            flags_add |= FLAG_NUMERIC_GUARD
            s2_post = _EPS

        n_eff = <double>k
        r = r_max
        cap_target = <int>(n_eff * r + 0.5)
        if cap_target < min_window:
            cap_target = min_window
        if cap_target > max_window:
            cap_target = max_window

        cap_new = <int>(((1.0 - cap_beta) * soft_cap + cap_beta * cap_target) + 0.5)
        if cap_new < min_window:
            cap_new = min_window
        if cap_new > max_window:
            cap_new = max_window

        return (
            pred_mu, pred_s2, n_eff, tau_t, 1.0, 0.0, 0.0,
            cap_target, cap_new,
            a_post, b_post, s2_post, s2_post,
            flags_add, k
        )

    # --------------- single pass: weights + pred moments + post moments
    Emu_pred = 0.0
    Emu2_pred = 0.0
    Evar_pred = 0.0
    n_eff = 0.0

    Ea = 0.0
    Ea2 = 0.0
    Eb = 0.0
    Es2 = 0.0

    ent = 0.0
    w_star = 0.0

    for i in range(K):
        base = i * stride
        s = p[base + 1]
        w = exp((s - m) / tau_t) / sumexp
        if w > w_star:
            w_star = w
        if w > 0.0:
            ent -= w * log(w)

        fk = <int>p[base + 9]
        flags_add |= fk

        mu_i = p[base + 2]
        s2_i = p[base + 3]
        if (not _finite(s2_i)) or s2_i <= _EPS:
            flags_add |= FLAG_NUMERIC_GUARD
            s2_i = _EPS

        Emu_pred += w * mu_i
        Emu2_pred += w * mu_i * mu_i
        Evar_pred += w * s2_i

        k = <int>p[base + 0]
        n_eff += w * (<double>k)

        sx = p[base + 4]
        sxx = p[base + 5]
        sy = p[base + 6]
        sxy = p[base + 7]
        syy = p[base + 8]

        sx_post = sx - (<double>k) * d_used
        sxx_post = sxx - 2.0 * d_used * sx + (<double>k) * d_used * d_used
        sxy_post = sxy - d_used * sy

        N = k + 1
        syN = sy + y_t
        syyN = syy + y_t * y_t

        f_post = 0
        _ols_from_sums(N, sx_post, sxx_post, syN, sxy_post, syyN, &a_post, &b_post, &s2_post, &Dpost, &f_post)
        flags_add |= f_post

        if (not _finite(s2_post)) or s2_post <= _EPS:
            flags_add |= FLAG_NUMERIC_GUARD
            s2_post = _EPS
        if (not _finite(a_post)) or (not _finite(b_post)):
            flags_add |= FLAG_NUMERIC_GUARD
            if not _finite(a_post):
                a_post = 0.0
            if not _finite(b_post):
                b_post = 0.0

        Ea += w * a_post
        Ea2 += w * a_post * a_post
        Eb += w * b_post
        Es2 += w * s2_post

    h_norm = (ent / logK) if (K > 1 and logK > 0.0) else 0.0
    if (not _finite(h_norm)) or h_norm < 0.0:
        h_norm = 0.0
    if h_norm > 1.0:
        h_norm = 1.0

    pred_mu = Emu_pred
    pred_s2 = Evar_pred + (Emu2_pred - Emu_pred * Emu_pred)
    if (not _finite(pred_s2)) or pred_s2 <= _EPS:
        flags_add |= FLAG_NUMERIC_GUARD
        pred_s2 = _EPS

    sigma2_noise = Es2
    if (not _finite(sigma2_noise)) or sigma2_noise <= _EPS:
        flags_add |= FLAG_NUMERIC_GUARD
        sigma2_noise = _EPS

    s2_model = Ea2 - Ea * Ea
    if (not _finite(s2_model)) or s2_model < 0.0:
        if not _finite(s2_model):
            flags_add |= FLAG_NUMERIC_GUARD
        s2_model = 0.0

    sigma2_total = sigma2_noise + s2_model
    if (not _finite(sigma2_total)) or sigma2_total <= _EPS:
        flags_add |= FLAG_NUMERIC_GUARD
        sigma2_total = _EPS

    r = r_min + (r_max - r_min) * (1.0 - h_norm)
    cap_target = <int>(n_eff * r + 0.5)
    if cap_target < min_window:
        cap_target = min_window
    if cap_target > max_window:
        cap_target = max_window

    cap_new = <int>(((1.0 - cap_beta) * soft_cap + cap_beta * cap_target) + 0.5)
    if cap_new < min_window:
        cap_new = min_window
    if cap_new > max_window:
        cap_new = max_window

    n_final = <int>(n_eff + 0.5)
    if n_final < 0:
        n_final = 0
    if n_final > max_window:
        n_final = max_window

    return (
        pred_mu, pred_s2, n_eff, tau_t, w_star, ent, h_norm,
        cap_target, cap_new,
        Ea, Eb, sigma2_noise, sigma2_total,
        flags_add, n_final
    )


cpdef soft_update_step(
    double[:] t_buf,
    double[:] y_buf,
    int head,
    int size,
    double y_t,
    double d,
    int min_window,
    int max_window_effective,
    int soft_cap,
    int max_window,

    double tau_min=2.0,
    double tau_max=3.0,
    double tau_boot=2.0,
    double r_min=0.7,
    double r_max=3.0,
    double cap_beta=0.01,
):
    """
    Fused soft path: selection + soft aggregation + post moments, without packed allocation.

    Returns:
      (n_hard, score_star, score_second, pred_mu, pred_s2, nu_sel, flags_core,
       n_eff, tau_t, w_star, entropy, entropy_norm,
       cap_target, cap_new,
       mu_post, trend_post, sigma2_noise, sigma2_total,
       n_final)
    """
    cdef int flags_sel = 0
    cdef int flags_add = 0
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
    cdef int best_nu = 0
    cdef int best_flags = 0
    cdef int best_i = 0
    cdef double best_score = -1e300
    cdef double best_second = -1e300

    cdef bint any_valid = False
    cdef bint any_degenerate = False

    cdef double a, b, sigma2, D
    cdef int f_ols, nu
    cdef double pred_mu, pred_s2, h_t, e, score

    # arrays for candidates (upper bound)
    cdef int max_cands = 0
    cdef int m = 0
    cdef int* ks = NULL
    cdef int* fks = NULL
    cdef double* scores = NULL
    cdef double* pred_mus = NULL
    cdef double* pred_s2s = NULL
    cdef double* a_posts = NULL
    cdef double* b_posts = NULL
    cdef double* s2_posts = NULL

    # post computations
    cdef int N, f_post
    cdef double sx_post, sxx_post, sxy_post, syN, syyN
    cdef double a_post, b_post, s2_post, Dpost
    cdef double d_used, tmp

    # soft accumulators
    cdef int i
    cdef double mscore, sumexp, w, ent, logK, h0, tau_t, h_norm, w_star
    cdef double Emu_pred, Emu2_pred, Evar_pred, n_eff
    cdef double Ea, Ea2, Eb, Es2
    cdef double mu_i, s2_i
    cdef double sigma2_noise, sigma2_total, s2_model
    cdef double r
    cdef int cap_target, cap_new, n_final

    # ---------------- guards / trivial
    if n_max > n_avail:
        n_max = n_avail
    if n_max < min_window:
        return (
            0, 0.0, 0.0, 0.0, _EPS, 0, 0,
            0.0, tau_boot, 0.0, 0.0, 0.0,
            min_window, min_window,
            0.0, 0.0, _EPS, _EPS,
            0
        )

    d_used = d
    if (not _finite(d_used)) or d_used <= 0.0:
        flags_sel |= FLAG_NUMERIC_GUARD
        d_used = 1.0

    # guard soft params (same as soft_step_from_packed)
    if (not _finite(tau_boot)) or tau_boot <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        tau_boot = 2.0
    if (not _finite(tau_min)) or tau_min <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        tau_min = 1.0
    if (not _finite(tau_max)) or tau_max <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        tau_max = tau_min
    if tau_max < tau_min:
        tmp = tau_max
        tau_max = tau_min
        tau_min = tmp

    if (not _finite(r_min)) or r_min <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        r_min = 1.0
    if (not _finite(r_max)) or r_max <= 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        r_max = r_min
    if r_max < r_min:
        tmp = r_max
        r_max = r_min
        r_min = tmp

    if (not _finite(cap_beta)):
        flags_add |= FLAG_NUMERIC_GUARD
        cap_beta = 0.010
    if cap_beta < 0.0:
        flags_add |= FLAG_NUMERIC_GUARD
        cap_beta = 0.0
    if cap_beta > 1.0:
        flags_add |= FLAG_NUMERIC_GUARD
        cap_beta = 1.0

    # last prior timestamp index = head-1
    idx_raw = head - 1
    idx = _wrap_idx(idx_raw, cap)
    x_origin = t_buf[idx]
    if not _finite(x_origin):
        flags_sel |= FLAG_NUMERIC_GUARD
        x_origin = 0.0

    max_cands = n_max - min_window + 1
    if max_cands < 0:
        max_cands = 0

    if max_cands == 0:
        return (
            0, 0.0, 0.0, 0.0, _EPS, 0, int(flags_sel | flags_add),
            0.0, tau_boot, 0.0, 0.0, 0.0,
            min_window, min_window,
            0.0, 0.0, _EPS, _EPS,
            0
        )

    # allocate candidate buffers
    ks = <int*>PyMem_Malloc(max_cands * sizeof(int))
    fks = <int*>PyMem_Malloc(max_cands * sizeof(int))
    scores = <double*>PyMem_Malloc(max_cands * sizeof(double))
    pred_mus = <double*>PyMem_Malloc(max_cands * sizeof(double))
    pred_s2s = <double*>PyMem_Malloc(max_cands * sizeof(double))
    a_posts = <double*>PyMem_Malloc(max_cands * sizeof(double))
    b_posts = <double*>PyMem_Malloc(max_cands * sizeof(double))
    s2_posts = <double*>PyMem_Malloc(max_cands * sizeof(double))

    if (ks is NULL) or (fks is NULL) or (scores is NULL) or (pred_mus is NULL) or (pred_s2s is NULL) or (a_posts is NULL) or (b_posts is NULL) or (s2_posts is NULL):
        if ks is not NULL: PyMem_Free(ks)
        if fks is not NULL: PyMem_Free(fks)
        if scores is not NULL: PyMem_Free(scores)
        if pred_mus is not NULL: PyMem_Free(pred_mus)
        if pred_s2s is not NULL: PyMem_Free(pred_s2s)
        if a_posts is not NULL: PyMem_Free(a_posts)
        if b_posts is not NULL: PyMem_Free(b_posts)
        if s2_posts is not NULL: PyMem_Free(s2_posts)
        raise MemoryError()

    try:
        # ---------------- selection pass: compute candidates + store minimal stats + post params
        for k in range(1, n_max + 1):
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

            pred_mu = a + b * d_used

            h_t = (sxx - 2.0 * sx * d_used + k * d_used * d_used) / D
            if (not _finite(h_t)) or h_t < 0.0:
                f_ols |= FLAG_NUMERIC_GUARD
                h_t = 0.0

            pred_s2 = sigma2 * (1.0 + h_t)
            if (not _finite(pred_s2)) or pred_s2 <= _EPS:
                f_ols |= FLAG_NUMERIC_GUARD
                pred_s2 = sigma2 if sigma2 > _EPS else _EPS

            e = y_t - pred_mu
            score = _student_t_loglik(e, pred_s2, nu) + _WINDOW_PRIOR_WEIGHT * log(<double>k)

            # post params for this candidate (same math as soft_step_from_packed)
            sx_post = sx - (<double>k) * d_used
            sxx_post = sxx - 2.0 * d_used * sx + (<double>k) * d_used * d_used
            sxy_post = sxy - d_used * sy

            N = k + 1
            syN = sy + y_t
            syyN = syy + y_t * y_t

            f_post = 0
            _ols_from_sums(N, sx_post, sxx_post, syN, sxy_post, syyN, &a_post, &b_post, &s2_post, &Dpost, &f_post)

            if (not _finite(s2_post)) or s2_post <= _EPS:
                f_post |= FLAG_NUMERIC_GUARD
                s2_post = _EPS
            if (not _finite(a_post)) or (not _finite(b_post)):
                f_post |= FLAG_NUMERIC_GUARD
                if not _finite(a_post): a_post = 0.0
                if not _finite(b_post): b_post = 0.0

            # store
            ks[m] = k
            scores[m] = score
            pred_mus[m] = pred_mu
            pred_s2s[m] = pred_s2
            a_posts[m] = a_post
            b_posts[m] = b_post
            s2_posts[m] = s2_post
            fks[m] = f_ols | f_post
            m += 1

            # update best selection (tie-break identical to select_student_t)
            if (score > best_score + _TIE_TOL) or (fabs(score - best_score) <= _TIE_TOL and k > best_n):
                best_second = best_score
                best_score = score
                best_n = k
                best_nu = nu
                best_flags = f_ols
                best_i = m - 1
            elif score > best_second + _TIE_TOL:
                best_second = score

        if not any_valid or m <= 0:
            if any_degenerate:
                flags_sel |= FLAG_DEGENERATE_XTX
            return (
                0, 0.0, 0.0, 0.0, _EPS, 0, int(flags_sel | flags_add),
                0.0, tau_boot, 0.0, 0.0, 0.0,
                min_window, min_window,
                0.0, 0.0, _EPS, _EPS,
                0
            )

        flags_sel |= best_flags
        if best_second <= -1e299:
            best_second = best_score

        # ---------------- bootstrap entropy (tau_boot) -> tau_t
        logK = log(<double>m) if m > 1 else 0.0
        mscore = best_score  # max score among candidates

        sumexp = 0.0
        for i in range(m):
            sumexp += exp((scores[i] - mscore) / tau_boot)

        if (not _finite(sumexp)) or sumexp <= 0.0:
            # hard fallback
            k = ks[best_i]
            flags_add |= fks[best_i]

            pred_mu = pred_mus[best_i]
            pred_s2 = pred_s2s[best_i]
            if (not _finite(pred_s2)) or pred_s2 <= _EPS:
                flags_add |= FLAG_NUMERIC_GUARD
                pred_s2 = _EPS

            n_eff = <double>k
            tau_t = tau_boot
            w_star = 1.0
            ent = 0.0
            h_norm = 0.0

            r = r_max
            cap_target = <int>(n_eff * r + 0.5)
            if cap_target < min_window: cap_target = min_window
            if cap_target > max_window: cap_target = max_window

            cap_new = <int>(((1.0 - cap_beta) * soft_cap + cap_beta * cap_target) + 0.5)
            if cap_new < min_window: cap_new = min_window
            if cap_new > max_window: cap_new = max_window

            return (
                best_n, best_score, best_second,
                pred_mu, pred_s2,
                best_nu,
                int(flags_sel | flags_add),
                n_eff, tau_t, w_star, ent, h_norm,
                cap_target, cap_new,
                a_posts[best_i], b_posts[best_i], s2_posts[best_i], s2_posts[best_i],
                k
            )

        ent = 0.0
        for i in range(m):
            w = exp((scores[i] - mscore) / tau_boot) / sumexp
            if w > 0.0:
                ent -= w * log(w)

        h0 = (ent / logK) if (m > 1 and logK > 0.0) else 0.0
        if (not _finite(h0)) or h0 < 0.0: h0 = 0.0
        if h0 > 1.0: h0 = 1.0

        tau_t = tau_min + (tau_max - tau_min) * h0
        if (not _finite(tau_t)) or tau_t <= 0.0:
            flags_add |= FLAG_NUMERIC_GUARD
            tau_t = tau_boot

        # ---------------- main softmax + moments
        sumexp = 0.0
        for i in range(m):
            sumexp += exp((scores[i] - mscore) / tau_t)

        if (not _finite(sumexp)) or sumexp <= 0.0:
            # hard fallback (as in soft_step_from_packed)
            k = ks[best_i]
            flags_add |= fks[best_i]

            pred_mu = pred_mus[best_i]
            pred_s2 = pred_s2s[best_i]
            if (not _finite(pred_s2)) or pred_s2 <= _EPS:
                flags_add |= FLAG_NUMERIC_GUARD
                pred_s2 = _EPS

            n_eff = <double>k
            w_star = 1.0
            ent = 0.0
            h_norm = 0.0

            r = r_max
            cap_target = <int>(n_eff * r + 0.5)
            if cap_target < min_window: cap_target = min_window
            if cap_target > max_window: cap_target = max_window

            cap_new = <int>(((1.0 - cap_beta) * soft_cap + cap_beta * cap_target) + 0.5)
            if cap_new < min_window: cap_new = min_window
            if cap_new > max_window: cap_new = max_window

            return (
                best_n, best_score, best_second,
                pred_mu, pred_s2,
                best_nu,
                int(flags_sel | flags_add),
                n_eff, tau_t, w_star, ent, h_norm,
                cap_target, cap_new,
                a_posts[best_i], b_posts[best_i], s2_posts[best_i], s2_posts[best_i],
                k
            )

        Emu_pred = 0.0
        Emu2_pred = 0.0
        Evar_pred = 0.0
        n_eff = 0.0

        Ea = 0.0
        Ea2 = 0.0
        Eb = 0.0
        Es2 = 0.0

        ent = 0.0
        w_star = 0.0

        for i in range(m):
            w = exp((scores[i] - mscore) / tau_t) / sumexp
            if w > w_star:
                w_star = w
            if w > 0.0:
                ent -= w * log(w)

            flags_add |= fks[i]

            mu_i = pred_mus[i]
            s2_i = pred_s2s[i]
            if (not _finite(s2_i)) or s2_i <= _EPS:
                flags_add |= FLAG_NUMERIC_GUARD
                s2_i = _EPS

            Emu_pred += w * mu_i
            Emu2_pred += w * mu_i * mu_i
            Evar_pred += w * s2_i

            n_eff += w * (<double>ks[i])

            Ea += w * a_posts[i]
            Ea2 += w * a_posts[i] * a_posts[i]
            Eb += w * b_posts[i]
            Es2 += w * s2_posts[i]

        h_norm = (ent / logK) if (m > 1 and logK > 0.0) else 0.0
        if (not _finite(h_norm)) or h_norm < 0.0: h_norm = 0.0
        if h_norm > 1.0: h_norm = 1.0

        pred_mu = Emu_pred
        pred_s2 = Evar_pred + (Emu2_pred - Emu_pred * Emu_pred)
        if (not _finite(pred_s2)) or pred_s2 <= _EPS:
            flags_add |= FLAG_NUMERIC_GUARD
            pred_s2 = _EPS

        sigma2_noise = Es2
        if (not _finite(sigma2_noise)) or sigma2_noise <= _EPS:
            flags_add |= FLAG_NUMERIC_GUARD
            sigma2_noise = _EPS

        s2_model = Ea2 - Ea * Ea
        if (not _finite(s2_model)) or s2_model < 0.0:
            if not _finite(s2_model):
                flags_add |= FLAG_NUMERIC_GUARD
            s2_model = 0.0

        sigma2_total = sigma2_noise + s2_model
        if (not _finite(sigma2_total)) or sigma2_total <= _EPS:
            flags_add |= FLAG_NUMERIC_GUARD
            sigma2_total = _EPS

        r = r_min + (r_max - r_min) * (1.0 - h_norm)
        cap_target = <int>(n_eff * r + 0.5)
        if cap_target < min_window: cap_target = min_window
        if cap_target > max_window: cap_target = max_window

        cap_new = <int>(((1.0 - cap_beta) * soft_cap + cap_beta * cap_target) + 0.5)
        if cap_new < min_window: cap_new = min_window
        if cap_new > max_window: cap_new = max_window

        n_final = <int>(n_eff + 0.5)
        if n_final < 0: n_final = 0
        if n_final > max_window: n_final = max_window

        return (
            best_n, best_score, best_second,
            pred_mu, pred_s2,
            best_nu,
            int(flags_sel | flags_add),
            n_eff, tau_t, w_star, ent, h_norm,
            cap_target, cap_new,
            Ea, Eb, sigma2_noise, sigma2_total,
            n_final
        )

    finally:
        PyMem_Free(ks)
        PyMem_Free(fks)
        PyMem_Free(scores)
        PyMem_Free(pred_mus)
        PyMem_Free(pred_s2s)
        PyMem_Free(a_posts)
        PyMem_Free(b_posts)
        PyMem_Free(s2_posts)
