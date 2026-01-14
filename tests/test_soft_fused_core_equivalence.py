import math
import pytest

from owrf1d import OnlineWindowRegressor1D
from owrf1d.filter import HAVE_CYTHON_CORE

pytestmark = pytest.mark.skipif(not HAVE_CYTHON_CORE, reason="Cython core not built")


def _fill_ring(f: OnlineWindowRegressor1D, n: int = 80) -> None:
    # детерминированная линия, чтобы не было шума/рандома
    t = 0.0
    for i in range(n):
        t += 1.0
        y = 1.0 + 0.1 * t
        f.update(y, t=t)


def test_soft_update_step_matches_two_step_core():
    import owrf1d._core as core

    W = 64
    min_w = 4

    f = OnlineWindowRegressor1D(max_window=W, min_window=min_w, history=0, selection="hard")
    f._use_core = True  # не критично, нам нужен ring
    _fill_ring(f, n=90)

    t_buf = f._t_buf.view()
    y_buf = f._y_buf.view()
    head = int(f._t_buf.head)
    size = int(f._t_buf.size)

    # "текущая" точка, которая еще НЕ добавлена в ring
    d = 1.0
    y_t = 3.14159

    max_eff = W
    soft_cap = W

    # old two-step
    (n_h, sc1, sc2, pm_star, ps2_star, nu, f_sel, K, packed) = core.select_student_t(
        t_buf, y_buf, head, size, float(y_t), float(d), int(min_w), int(max_eff), True
    )
    (
        pm,
        ps2,
        n_eff,
        tau_t,
        w_star,
        ent,
        h_norm,
        cap_target,
        cap_new,
        mu_post,
        trend_post,
        s2_noise,
        s2_total,
        f_add,
        n_final,
    ) = core.soft_step_from_packed(
        packed,
        int(K),
        float(d),
        float(y_t),
        int(soft_cap),
        int(W),
        int(min_w),
        2.0, 3.0, 2.0, 0.7, 3.0, 0.010,
    )

    # new fused
    out = core.soft_update_step(
        t_buf,
        y_buf,
        head,
        size,
        float(y_t),
        float(d),
        int(min_w),
        int(max_eff),
        int(soft_cap),
        int(W),
        2.0, 3.0, 2.0, 0.7, 3.0, 0.010,
    )
    (
        n_h2,
        sc1_2,
        sc2_2,
        pm_2,
        ps2_2,
        nu_2,
        f_core_2,
        n_eff_2,
        tau_t_2,
        w_star_2,
        ent_2,
        h_norm_2,
        cap_target_2,
        cap_new_2,
        mu_post_2,
        trend_post_2,
        s2_noise_2,
        s2_total_2,
        n_final_2,
    ) = out

    assert n_h2 == n_h
    assert nu_2 == nu
    assert f_core_2 == (int(f_sel) | int(f_add))

    # Численные поля — допускаем микроскопическое отличие
    atol = 1e-12
    for a, b in [
        (sc1_2, sc1),
        (sc2_2, sc2),
        (pm_2, pm),
        (ps2_2, ps2),
        (n_eff_2, n_eff),
        (tau_t_2, tau_t),
        (w_star_2, w_star),
        (ent_2, ent),
        (h_norm_2, h_norm),
        (mu_post_2, mu_post),
        (trend_post_2, trend_post),
        (s2_noise_2, s2_noise),
        (s2_total_2, s2_total),
    ]:
        assert math.isfinite(a)
        assert math.isfinite(b)
        assert abs(a - b) <= atol

    assert cap_target_2 == cap_target
    assert cap_new_2 == cap_new
    assert n_final_2 == n_final
