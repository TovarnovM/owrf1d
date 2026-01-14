from owrf1d import OnlineWindowRegressor1D
from owrf1d.filter import HAVE_CYTHON_CORE


def main() -> None:
    # В этом процессе выставлен OWRF1D_FORCE_PY=1 (см. run_smoke.sh)
    print("HAVE_CYTHON_CORE (forced):", HAVE_CYTHON_CORE)

    f = OnlineWindowRegressor1D(max_window=32, min_window=4, history=0, selection="soft")
    for i in range(50):
        f.update(0.1 * i, t=float(i))

    print("Forced-Python path OK.")


if __name__ == "__main__":
    main()
