# Warsaw day length analysis

Day length, sunrise/sunset times and sunrise/sunset directions for Warsaw, Poland
(any other location works too).

Check out the [Jupyter notebook](analysis.ipynb) for visualizations and details.

## Highlights (March 2024 – March 2025)

- Day length ranges from **7 h 42 m** (21 Dec) to **16 h 47 m** (20 Jun).
- A single least-squares sine is within ~13 min of the true day length; adding the
  **3rd** harmonic (not the 2nd) brings the error under one minute.
- The sunrise→sunset horizon span shrinks from ~182° at the September equinox to ~102° at the
  December solstice; the halfway point (~142°) is reached on **24 October**.
- Earliest sunset (12 Dec) and latest sunrise (30 Dec) do *not* fall on the shortest day.

## Layout

```
src/daylength/   tested library: solar events, harmonic fits, plots, CLI
tests/           pytest suite (checked against published almanac times)
analysis.ipynb   the analysis, built on top of the library
```

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # create .venv from uv.lock
uv run pytest                 # run tests
uv run ruff check . && uv run ruff format --check .
uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # re-run the notebook

# CLI: summary for any location / period, optional CSV export
uv run daylength --start 2025-01-01 --end 2025-12-31 --csv warsaw-2025.csv
uv run daylength --name Tromso --lat 69.6492 --lon 18.9553 --tz Europe/Oslo
```

```python
from daylength import WARSAW, sun_events
sun_events(WARSAW, "2024-06-21")  # local, DST-aware sunrise/sunset, day length, azimuths
```

## Conventions

- Times are local to the location's IANA time zone (DST-aware), and each row is a local calendar day.
- Sunrise/sunset: upper limb on the horizon with 34′ standard refraction (USNO/NOAA convention).
- Azimuths: degrees from North through East (90° = E, 270° = W).
- Equinoxes/solstices are computed with PyEphem, never hard-coded.
