# owrf1d — Online Window Regression Filter (1D)

A strictly-online 1D filter that separates **local linear drift** (trend) from **noise** by fitting linear regression on a sliding window and **selecting an effective window length** via **predictive Student-t log-likelihood**.

Key properties:

- **Strictly online:** processes one observation at a time (`update()`), no future access.
- **Adaptive window:** automatically selects how far back to look (effective window).
- **Minimal API:** only `max_window`, `min_window`, `history`, and `selection`.
- **Deterministic:** same inputs → bitwise-identical outputs (within the same backend path).
- **Fast path:** optional Cython core (auto-used when available), with a pure-Python fallback.
- **Serialization:** `dumps()` / `loads()` via `cloudpickle`.

![overview](https://github.com/TovarnovM/owrf1d/raw/main/docs/owrf1d_readme_overview.png)

---

## Installation

```bash
pip install owrf1d
````

Optional dependencies for examples (plots):

```bash
pip install "owrf1d[examples]"
```

Notes on performance:

* The package can use a Cython extension (`owrf1d._core`) when present.
* You can force the pure-Python path via environment variable:

```bash
OWRF1D_FORCE_PY=1 python your_script.py
```

---

## Quick start

### Minimal online loop (timestamps)

```python
from owrf1d import OnlineWindowRegressor1D

f = OnlineWindowRegressor1D(max_window=128, min_window=4, selection="soft")

for t, y in stream:               # strictly in time order
    step = f.update(y, t=t)
    mu = step["mu"]
    trend = step["trend"]
    sigma = step["sigma2"] ** 0.5
```

### Minimal online loop (fixed dt, no timestamps)

```python
from owrf1d import OnlineWindowRegressor1D

f = OnlineWindowRegressor1D(max_window=128, min_window=4)

for y in stream:
    step = f.update(y, dt=1.0)    # dt interface
```

### Missing observations / “predict-only”

If `y is None`, the filter advances time but does **not** update the regression buffers:

```python
step = f.update(None, t=t)        # or dt=...
assert step["flags"] != 0         # includes FLAG_PREDICT_ONLY
```

---

## API

```python
OnlineWindowRegressor1D(
    *,
    max_window: int = 128,
    min_window: int = 4,
    history: int = 0,          # 0 disables history; -1 keeps all; N keeps last N
    selection: str = "soft",   # "soft" (default) or "hard"
)
```

### `update()`

```python
update(
    y: float | None,
    *,
    t: float | None = None,
    dt: float | None = None,
) -> dict
```

Time rules:

* If `dt` is provided, it is used to advance time.
* If `t` is provided, `dt` is inferred as `t - previous_t`.
* If neither is provided, the filter assumes `dt = 1.0`.
* If both `t` and `dt` are provided, `dt` wins (and a numeric-guard flag may be set).

### `get_state()`, `get_history()`

```python
state = f.get_state()
history = f.get_history()
```

### Serialization

```python
blob = f.dumps()
f2 = OnlineWindowRegressor1D.loads(blob)
```

---

## Output contract (step dict)

Every `update()` returns a dictionary with at least the following keys:

* `mu` — filtered level at current time
* `trend` — filtered slope (per unit time)
* `sigma2` — estimated noise variance (non-negative, guarded)
* `n_star` — selected effective window length (integer)
* `score_star`, `score_second`, `delta_score` — predictive log-likelihood scores (selection phase)
* `nu` — Student-t degrees of freedom used in selection (typically `n_star - 2`)
* `pred_mu`, `pred_s2` — one-step-ahead predictive mean/variance (selection phase)
* `resid` — `y - mu` (after update)
* `t`, `dt` — time and step used
* `flags` — bitmask (see below)

For `selection="soft"`, additional diagnostic fields may be present, such as:
`n_star_hard`, `n_eff`, `w_star`, `entropy_norm`, `tau`, `cap`, `sigma2_total`, etc.

---

## Flags

Bitmask values are exposed in `owrf1d.flags`:

```python
from owrf1d.flags import (
    FLAG_PREDICT_ONLY,
    FLAG_INSUFFICIENT_DATA,
    FLAG_DEGENERATE_XTX,
    FLAG_NEGATIVE_SSE,
    FLAG_NUMERIC_GUARD,
    FLAG_HISTORY_TRUNC,
)
```

Meaning:

* `FLAG_PREDICT_ONLY` — `y is None` (time advanced, no update)
* `FLAG_INSUFFICIENT_DATA` — fewer than `min_window` prior points
* `FLAG_DEGENERATE_XTX` — regression matrix became (near-)singular for some candidates
* `FLAG_NEGATIVE_SSE` — numeric artifact caused SSE < 0 (clipped)
* `FLAG_NUMERIC_GUARD` — NaN/Inf / non-positive dt / clipping / fallback guards triggered
* `FLAG_HISTORY_TRUNC` — history ring buffer truncated (when `history > 0`)

---

## Model and scoring (math)

At time `t`, an observation `(T_t, y_t)` arrives.

### Selection phase (predictive scoring)

For each candidate window size `k` (number of *previous* points) in:

`k ∈ [min_window, max_window_effective]`

we fit OLS on the `k` points **before** `t`:

`D_t^(k) = {(T_{t-k}, y_{t-k}), ..., (T_{t-1}, y_{t-1})}`

We set the regressor relative to the last pre-point:

* `x_i = T_i - T_{t-1}`  ⇒ `x_{t-1} = 0`
* current-step `d = T_t - T_{t-1}` (or `dt` interface) ⇒ `x_t = d`

Using sums:

* `Sx = Σ x_i`
* `Sxx = Σ x_i^2`
* `Sy = Σ y_i`
* `Sxy = Σ x_i y_i`
* `Syy = Σ y_i^2`

Define:

* `D = k*Sxx - Sx^2`  (degenerate if too small)

OLS:

* `b = (k*Sxy - Sx*Sy) / D`
* `a = (Sy - b*Sx) / k`

SSE (numerically-guarded / clipped):

* `SSE = (Syy - Sy^2/k) - b^2 * (Sxx - Sx^2/k)`

Noise variance estimate:

* `nu = k - 2`
* `sigma2 = SSE / nu`  (guarded to be ≥ eps)

Leverage for prediction at `x_t = d`:

* `h = (Sxx - 2*Sx*d + k*d^2) / D`

Predictive variance:

* `pred_s2 = sigma2 * (1 + h)`

Predictive distribution (approx):

`y_t | D_t^(k) ~ StudentT_df=nu(mean = pred_mu, variance = pred_s2)`

where:

* `pred_mu = a + b*d`

Score is the Student-t log-pdf plus a mild prior favoring larger windows:

* `score(k) = log p_t(y_t | D_t^(k)) + w * log(k)`
* current implementation uses `w = 0.5`

The best window is:

* `k* = argmax_k score(k)` (ties resolved toward larger `k`)

### Update phase (state estimation)

* **Hard (`selection="hard"`):** re-fit OLS on the window of `k* + 1` points including `(T_t, y_t)`,
  using `x_i = T_i - T_t` so the intercept is the current level estimate `mu_t`.
* **Soft (`selection="soft"`):** compute weights `w_k ∝ exp(score(k)/tau)` (entropy-adaptive `tau`),
  and mix the post-update parameters across candidate windows. This makes estimates smoother and
  provides diagnostics such as `n_eff` and `entropy_norm`. The effective scan limit is also adapted
  via an internal “cap” to reduce per-step work when the model is confident.

---

## Examples

Generate an overview PNG similar to the one shown above:

```bash
python examples/example1.py --n 600 --cp 200 --cp2 400 --max-window 128
```

Micro-benchmark:

```bash
python examples/bench.py --n 200000 --max-window 128 --repeats 5
```

---

## Practical guidance

* Use `selection="soft"` (default) when you want **stable estimates** and smoother adaptation.
* Use `selection="hard"` when you want the **most interpretable discrete window length** `n_star`.
* If timestamps are noisy or unavailable, prefer the `dt=` interface.
* The method assumes locally linear dynamics and approximately iid noise within the selected window.
  Heavy autocorrelation or strong seasonality may require preprocessing or a different model class.

---

## License

```MIT License

Copyright (c) 2026 Tovarnov Mikhail

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.```



```


