"""Bitmask flags for Online Window Regression Filter 1D (OWRF-1D).

Часть публичного контракта: значения должны быть стабильны.
"""

FLAG_PREDICT_ONLY: int = 1 << 0
FLAG_INSUFFICIENT_DATA: int = 1 << 1
FLAG_DEGENERATE_XTX: int = 1 << 2
FLAG_NEGATIVE_SSE: int = 1 << 3
FLAG_NUMERIC_GUARD: int = 1 << 4
FLAG_HISTORY_TRUNC: int = 1 << 5
