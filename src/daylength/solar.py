"""Core solar computations built on top of PyEphem.

All public functions work in the *local civil time* of the location, so a row
for ``2024-06-21`` describes the sunrise and sunset that happen on that local
calendar day (not on a UTC day, which in Warsaw starts at 01:00 or 02:00).

Conventions
-----------
* Sunrise/sunset follow the standard USNO/NOAA definition: the upper limb of
  the Sun touches the horizon, with 34' of atmospheric refraction folded into
  the horizon altitude (``pressure=0``, ``horizon=-0:34``). This makes results
  independent of PyEphem's weather-dependent refraction model and comparable
  with published almanacs.
* Azimuths are in degrees, measured from North through East
  (0 = N, 90 = E, 180 = S, 270 = W).
* Day length is in hours and is computed with ``timedelta.total_seconds()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import NamedTuple
from zoneinfo import ZoneInfo

import ephem
import pandas as pd

# Standard refraction + semi-diameter correction used by USNO / NOAA.
_STANDARD_HORIZON = "-0:34"


@dataclass(frozen=True)
class Location:
    """A point on Earth together with its IANA time zone."""

    name: str
    latitude: float  # degrees, north positive
    longitude: float  # degrees, east positive
    timezone: str  # IANA name, e.g. "Europe/Warsaw"
    elevation: float = 0.0  # metres above sea level

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude must be in [-90, 90], got {self.latitude}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude must be in [-180, 180], got {self.longitude}")
        # Fail early on a typo in the zone name rather than deep inside a loop.
        ZoneInfo(self.timezone)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def observer(self) -> ephem.Observer:
        obs = ephem.Observer()
        # ephem parses *strings* as degrees and *floats* as radians.
        obs.lat = str(self.latitude)
        obs.lon = str(self.longitude)
        obs.elevation = self.elevation
        obs.pressure = 0  # disable ephem's own refraction model ...
        obs.horizon = _STANDARD_HORIZON  # ... and use the standard correction
        return obs


WARSAW = Location("Warsaw", latitude=52.2297, longitude=21.0122, timezone="Europe/Warsaw")


class SunEvents(NamedTuple):
    """Solar events for a single local calendar day.

    For polar day/night ``sunrise``/``sunset`` are ``None``, azimuths are NaN
    and ``day_length`` is 24.0 or 0.0 respectively.
    """

    date: date
    sunrise: datetime | None  # tz-aware, local time
    sunset: datetime | None  # tz-aware, local time
    solar_noon: datetime  # tz-aware, local time
    day_length: float  # hours
    sunrise_azimuth: float  # degrees from North
    sunset_azimuth: float  # degrees from North
    noon_altitude: float  # degrees above horizon at solar noon

    @property
    def azimuth_span(self) -> float:
        """Degrees of horizon swept between sunrise and sunset (through South)."""
        return (self.sunset_azimuth - self.sunrise_azimuth) % 360.0


def _to_local(d: ephem.Date, tz: ZoneInfo) -> datetime:
    return d.datetime().replace(tzinfo=UTC).astimezone(tz)


def _local_midnight_utc(day: date, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, time(0), tzinfo=tz).astimezone(UTC).replace(tzinfo=None)


def _azimuth_deg(obs: ephem.Observer, when: ephem.Date) -> float:
    obs.date = when
    sun = ephem.Sun(obs)
    return math.degrees(float(sun.az))


def sun_events(location: Location, day: date | datetime | pd.Timestamp) -> SunEvents:
    """Compute sunrise, sunset, day length and azimuths for one local day."""
    if isinstance(day, datetime):  # includes pd.Timestamp
        day = day.date()
    tz = location.tz
    start = _local_midnight_utc(day, tz)

    obs = location.observer()
    obs.date = start
    sun = ephem.Sun()

    noon = obs.next_transit(sun, start=start)
    obs.date = noon
    sun.compute(obs)
    noon_altitude = math.degrees(float(sun.alt))

    try:
        rise = obs.next_rising(sun, start=start)
        # Take the setting that follows *this* sunrise so that the difference is
        # always the length of one continuous daylight period.
        set_ = obs.next_setting(sun, start=rise)
    except ephem.AlwaysUpError:
        return SunEvents(day, None, None, _to_local(noon, tz), 24.0, math.nan, math.nan, noon_altitude)
    except ephem.NeverUpError:
        return SunEvents(day, None, None, _to_local(noon, tz), 0.0, math.nan, math.nan, noon_altitude)

    sunrise, sunset = _to_local(rise, tz), _to_local(set_, tz)
    day_length = (sunset - sunrise).total_seconds() / 3600.0
    return SunEvents(
        date=day,
        sunrise=sunrise,
        sunset=sunset,
        solar_noon=_to_local(noon, tz),
        day_length=day_length,
        sunrise_azimuth=_azimuth_deg(obs, rise),
        sunset_azimuth=_azimuth_deg(obs, set_),
        noon_altitude=noon_altitude,
    )


def sun_table(location: Location, start: date | str, end: date | str) -> pd.DataFrame:
    """Return one row of :class:`SunEvents` per local day in ``[start, end]``.

    The frame is indexed by a daily ``DatetimeIndex`` and has an extra
    ``azimuth_span`` column.
    """
    days = pd.date_range(start=start, end=end, freq="D")
    rows = [sun_events(location, d) for d in days]
    df = pd.DataFrame(rows, columns=SunEvents._fields).drop(columns="date")
    df.index = days
    df.index.name = "date"
    df["azimuth_span"] = [r.azimuth_span for r in rows]
    return df


class SeasonMarkers(NamedTuple):
    """Instants of equinoxes and solstices, tz-aware in the location's zone."""

    march_equinox: datetime
    june_solstice: datetime
    september_equinox: datetime
    december_solstice: datetime


def season_markers(year: int, tz: str | ZoneInfo = "UTC") -> SeasonMarkers:
    """Compute equinoxes and solstices of ``year`` instead of hard-coding them.

    Calendar dates of these events drift between years (e.g. the September
    equinox was on 22 Sep in 2024 but is often on 23 Sep), so never hard-code
    them.
    """
    zone = ZoneInfo(tz) if isinstance(tz, str) else tz
    start = ephem.Date(f"{year}/1/1")
    mar = ephem.next_equinox(start)
    jun = ephem.next_solstice(mar)
    sep = ephem.next_equinox(jun)
    dec = ephem.next_solstice(sep)
    return SeasonMarkers(*(_to_local(e, zone) for e in (mar, jun, sep, dec)))
