import math
from datetime import date, datetime

import pandas as pd
import pytest

from daylength import WARSAW, Location, fit_harmonics, halfway_date, season_markers, sun_events, sun_table
from daylength.cli import main


def _minutes(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


# Reference values: published almanac times for Warsaw (local time, rounded to the minute).
@pytest.mark.parametrize(
    ("day", "sunrise", "sunset"),
    [
        (date(2024, 6, 21), "04:14", "21:01"),  # summer, CEST (UTC+2)
        (date(2024, 12, 21), "07:43", "15:25"),  # winter, CET (UTC+1)
    ],
)
def test_warsaw_matches_almanac(day, sunrise, sunset):
    ev = sun_events(WARSAW, day)
    to_min = lambda s: int(s[:2]) * 60 + int(s[3:])  # noqa: E731
    assert abs(_minutes(ev.sunrise) - to_min(sunrise)) <= 1
    assert abs(_minutes(ev.sunset) - to_min(sunset)) <= 1
    assert ev.sunrise.date() == day and ev.sunset.date() == day


def test_times_are_local_and_dst_aware():
    summer = sun_events(WARSAW, date(2024, 7, 1))
    winter = sun_events(WARSAW, date(2024, 1, 1))
    assert summer.sunrise.utcoffset().total_seconds() == 2 * 3600
    assert winter.sunrise.utcoffset().total_seconds() == 1 * 3600


def test_day_length_and_azimuths_are_consistent():
    ev = sun_events(WARSAW, date(2024, 6, 21))
    assert ev.day_length == pytest.approx((ev.sunset - ev.sunrise).total_seconds() / 3600)
    assert 16.7 < ev.day_length < 16.8
    # Midsummer sun rises in the north-east and sets in the north-west.
    assert 40 < ev.sunrise_azimuth < 50
    assert 310 < ev.sunset_azimuth < 320
    # Solar noon altitude = 90 - lat + declination(+23.44)
    assert ev.noon_altitude == pytest.approx(90 - 52.2297 + 23.44, abs=0.2)


def test_equinox_is_roughly_symmetric():
    ev = sun_events(WARSAW, date(2024, 9, 22))
    assert ev.sunrise_azimuth == pytest.approx(90, abs=2)
    assert ev.sunset_azimuth == pytest.approx(270, abs=2)
    assert ev.azimuth_span == pytest.approx(180, abs=3)
    # Day is slightly > 12h at the equinox because of refraction and the solar disk.
    assert 12.0 < ev.day_length < 12.3


def test_polar_day_and_night():
    tromso = Location("Tromso", 69.6492, 18.9553, "Europe/Oslo")
    summer = sun_events(tromso, date(2024, 6, 21))
    winter = sun_events(tromso, date(2024, 12, 21))
    assert summer.day_length == 24.0 and summer.sunrise is None
    assert winter.day_length == 0.0 and winter.sunset is None
    assert math.isnan(summer.sunrise_azimuth)


def test_southern_hemisphere():
    sydney = Location("Sydney", -33.8688, 151.2093, "Australia/Sydney")
    assert sun_events(sydney, date(2024, 6, 21)).day_length < 10.5
    assert sun_events(sydney, date(2024, 12, 21)).day_length > 14


def test_season_markers_2024():
    m = season_markers(2024, "Europe/Warsaw")
    assert m.march_equinox.date() == date(2024, 3, 20)
    assert m.june_solstice.date() == date(2024, 6, 20)  # not the 21st!
    assert m.september_equinox.date() == date(2024, 9, 22)  # not the 23rd!
    assert m.december_solstice.date() == date(2024, 12, 21)


def test_sun_table_shape():
    df = sun_table(WARSAW, "2024-01-01", "2024-12-31")
    assert len(df) == 366
    assert df.day_length.idxmax() == pd.Timestamp("2024-06-20")  # longest day around the solstice
    assert df.day_length.between(7.5, 16.9).all()
    assert (df.azimuth_span > 0).all()


def test_fit_harmonics_recovers_sine():
    idx = pd.date_range("2024-01-01", periods=730, freq="D")
    t = (idx - idx[0]).days.to_numpy()
    s = pd.Series(12 + 4 * pd.Series(t).map(lambda x: math.sin(2 * math.pi * x / 365.2422 + 0.3)).to_numpy(), idx)
    fit = fit_harmonics(s)
    assert fit.mean == pytest.approx(12)
    assert fit.amplitudes[0] == pytest.approx(4)
    assert fit.rmse < 1e-9


def test_fit_on_real_day_length_needs_third_harmonic():
    df = sun_table(WARSAW, "2024-03-20", "2025-03-19")
    one, three = fit_harmonics(df.day_length, 1), fit_harmonics(df.day_length, 3)
    assert one.max_abs_error < 0.25  # a single sine is within 15 minutes
    assert three.max_abs_error < 1 / 60  # three harmonics are within a minute


def test_halfway_date_restricts_to_window():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    s = pd.Series([0, 1, 2, 3, 4, 5, 4, 3, 2, 1], idx, dtype=float)
    # Midpoint of 0 and 4 is 2: present on Jan 3 *and* Jan 9; only Jan 3 is in the window.
    assert halfway_date(s, "2024-01-01", "2024-01-05") == pd.Timestamp("2024-01-03")
    with pytest.raises(ValueError):
        halfway_date(s, "2024-01-05", "2024-01-01")


@pytest.mark.parametrize("kwargs", [{"latitude": 91}, {"longitude": -181}, {"timezone": "Mars/Base"}])
def test_location_validation(kwargs):
    base = {"name": "x", "latitude": 0.0, "longitude": 0.0, "timezone": "UTC"} | kwargs
    with pytest.raises((ValueError, KeyError)):
        Location(**base)


def test_cli(tmp_path, capsys):
    out = tmp_path / "w.csv"
    assert main(["--start", "2024-01-01", "--end", "2024-12-31", "--csv", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Longest day:  2024-06-20" in text
    assert len(pd.read_csv(out)) == 366
    assert main(["--tz", "Not/AZone"]) == 2
