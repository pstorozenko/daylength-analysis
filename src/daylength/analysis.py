"""Analysis helpers: harmonic approximations and 'halfway' dates."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

TROPICAL_YEAR_DAYS = 365.2422


class HarmonicFit(NamedTuple):
    """Least-squares fit ``y ~ mean + sum_k a_k sin(k w t) + b_k cos(k w t)``."""

    mean: float
    amplitudes: tuple[float, ...]  # sqrt(a_k^2 + b_k^2) per harmonic
    fitted: pd.Series
    residuals: pd.Series

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals.to_numpy() ** 2)))

    @property
    def max_abs_error(self) -> float:
        return float(np.max(np.abs(self.residuals.to_numpy())))


def fit_harmonics(series: pd.Series, n_harmonics: int = 1, period_days: float = TROPICAL_YEAR_DAYS) -> HarmonicFit:
    """Fit a yearly Fourier series to a daily, ``DatetimeIndex``-ed series.

    Unlike scaling a ``sin`` by ``(max - min) / 2`` around the mean, the
    least-squares fit also estimates the phase and the offset, so it does not
    depend on the series happening to start exactly at an equinox. NaNs (polar
    day/night azimuths) are ignored during fitting.
    """
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")

    t = (series.index - series.index[0]) / pd.Timedelta(days=1)
    t = np.asarray(t, dtype=float)
    w = 2 * np.pi / period_days
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        cols += [np.sin(k * w * t), np.cos(k * w * t)]
    design = np.column_stack(cols)

    y = series.to_numpy(dtype=float)
    ok = np.isfinite(y)
    coef, *_ = np.linalg.lstsq(design[ok], y[ok], rcond=None)

    fitted = pd.Series(design @ coef, index=series.index, name=f"{series.name}_fit")
    amps = tuple(float(np.hypot(coef[2 * k - 1], coef[2 * k])) for k in range(1, n_harmonics + 1))
    return HarmonicFit(float(coef[0]), amps, fitted, (series - fitted).rename(f"{series.name}_residual"))


def halfway_date(series: pd.Series, start, end) -> pd.Timestamp:
    """Date in ``[start, end]`` where ``series`` is closest to the midpoint of
    its values at ``start`` and ``end``.

    Restricting the search to the window matters: a yearly curve hits every
    intermediate value twice, so a global ``argmin`` may return the matching
    date from the wrong half of the year.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError("start must not be after end")
    window = series.loc[start:end]
    target = (series.loc[start] + series.loc[end]) / 2
    return (window - target).abs().idxmin()
