"""Matplotlib plots for sun tables produced by :func:`daylength.sun_table`."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def plot_series_with_fit(
    actual: pd.Series, fitted: pd.Series, *, title: str, ylabel: str, ax: Axes | None = None
) -> Axes:
    ax = ax or plt.subplots(figsize=(9, 4))[1]
    ax.plot(actual.index, actual, label="actual")
    ax.plot(fitted.index, fitted, label="sinusoidal fit", linestyle="--")
    ax.set(title=title, xlabel="Date", ylabel=ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_residuals(residuals: pd.Series, *, title: str, ylabel: str, ax: Axes | None = None) -> Axes:
    ax = ax or plt.subplots(figsize=(9, 3))[1]
    ax.plot(residuals.index, residuals)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(title=title, xlabel="Date", ylabel=ylabel)
    ax.grid(alpha=0.3)
    return ax


def plot_azimuth_compass(
    table: pd.DataFrame,
    days: Mapping[str, pd.Timestamp],
    *,
    title: str,
    ax: Axes | None = None,
) -> Axes:
    """Draw sunrise/sunset directions for selected days on a compass rose.

    The plot is a *map view* (looking down, North up, East right), so azimuth
    increases clockwise. Each label in ``days`` gets one colour; the sunrise
    ray is solid and the sunset ray dashed.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise => 90 deg is East, as on a map

    for label, day in days.items():
        row = table.loc[pd.Timestamp(day)]
        (line,) = ax.plot([np.radians(row.sunrise_azimuth)] * 2, [0, 1], label=f"{label} ({day:%Y-%m-%d})")
        ax.plot([np.radians(row.sunset_azimuth)] * 2, [0, 1], color=line.get_color(), linestyle="--")

    ax.set_rlim(0, 1)
    ax.set_rticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, title="solid = sunrise, dashed = sunset")
    return ax
