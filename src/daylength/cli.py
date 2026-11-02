"""Command line interface: ``daylength --help``."""

from __future__ import annotations

import argparse
import sys
from datetime import date

from daylength.analysis import fit_harmonics
from daylength.solar import WARSAW, Location, season_markers, sun_table


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    today = date.today()
    p = argparse.ArgumentParser(description="Sunrise, sunset and day length for a location.")
    p.add_argument("--name", default=WARSAW.name)
    p.add_argument("--lat", type=float, default=WARSAW.latitude, help="latitude in degrees (north positive)")
    p.add_argument("--lon", type=float, default=WARSAW.longitude, help="longitude in degrees (east positive)")
    p.add_argument("--tz", default=WARSAW.timezone, help="IANA time zone, e.g. Europe/Warsaw")
    p.add_argument("--start", type=date.fromisoformat, default=date(today.year, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date(today.year, 12, 31))
    p.add_argument("--csv", metavar="PATH", help="write the full daily table to a CSV file")
    args = p.parse_args(argv)
    if args.start > args.end:
        p.error("--start must not be after --end")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        loc = Location(args.name, args.lat, args.lon, args.tz)
    except (ValueError, KeyError) as exc:  # ZoneInfoNotFoundError is a KeyError
        print(f"error: {exc}", file=sys.stderr)
        return 2

    df = sun_table(loc, args.start, args.end)
    if args.csv:
        df.to_csv(args.csv)

    longest, shortest = df.day_length.idxmax(), df.day_length.idxmin()
    print(f"{loc.name} ({loc.latitude:.4f}, {loc.longitude:.4f}, {loc.timezone})")
    print(f"Period: {args.start} .. {args.end} ({len(df)} days)")
    print(f"Longest day:  {longest:%Y-%m-%d}  {_hm(df.day_length.max())}")
    print(f"Shortest day: {shortest:%Y-%m-%d}  {_hm(df.day_length.min())}")
    if len(df) >= 365:
        fit = fit_harmonics(df.day_length)
        print(f"Sinusoid fit: RMSE {fit.rmse * 60:.1f} min, max error {fit.max_abs_error * 60:.1f} min")
    for name, when in season_markers(args.start.year, loc.tz)._asdict().items():
        print(f"{name.replace('_', ' ').title():18s} {when:%Y-%m-%d %H:%M %Z}")
    return 0


def _hm(hours: float) -> str:
    minutes = round(hours * 60)
    return f"{minutes // 60:d}h {minutes % 60:02d}m"


if __name__ == "__main__":
    raise SystemExit(main())
