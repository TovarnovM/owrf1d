cdef extern from *:
    pass

# public Python-level function (for import in filter.py)
# signature duplicated for type checkers / future cimports
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
)

cpdef soft_step_from_packed(
    object packed,
    int K,
    double d,
    double y_t,
    int soft_cap,
    int max_window,
    int min_window,
    double tau_min=*,
    double tau_max=*,
    double tau_boot=*,
    double r_min=*,
    double r_max=*,
    double cap_beta=*,
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
    double tau_min=*,
    double tau_max=*,
    double tau_boot=*,
    double r_min=*,
    double r_max=*,
    double cap_beta=*,
)