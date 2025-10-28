"""Validation suite for OOR duration calculations across typical scenarios."""

from datetime import timedelta
from typing import List, Tuple

import pandas as pd

from oor import _compute_oor_events


def make_series(start: str, points: List[Tuple[int, float]], channel: str = "Value") -> pd.DataFrame:
    """Build a timeseries dataframe for tests.

    Parameters
    ----------
    start:
        Base timestamp (string) used for the first reading.
    points:
        Sequence of ``(minutes_offset, value)`` tuples relative to ``start``.
    channel:
        Name of the numeric channel column to populate.
    """

    base = pd.to_datetime(start)
    rows = [
        {"DateTime": base + timedelta(minutes=minutes), channel: value}
        for minutes, value in points
    ]
    return pd.DataFrame(rows)


def total_minutes(ev_df: pd.DataFrame) -> float:
    """Return rounded total duration for an event dataframe."""

    if ev_df.empty:
        return 0.0
    return round(float(ev_df["Duration(min)"].sum()), 2)


def test_room_humidity_90_minutes() -> None:
    points = [(minutes, 65.0) for minutes in range(0, 90, 5)]
    points.append((90, 50.0))  # recovery sample
    df = make_series("2025-09-01 12:00", points, channel="Humidity")

    events = _compute_oor_events(df, "Humidity", lo=None, hi=60.0, default_source="room")

    assert total_minutes(events) == 90.0


def test_room_temperature_50_minutes() -> None:
    points = [(minutes, 27.0) for minutes in range(0, 50, 5)]
    points.append((50, 24.0))
    df = make_series("2025-09-01 14:00", points, channel="Temperature")

    events = _compute_oor_events(
        df,
        "Temperature",
        lo=None,
        hi=26.0,
        default_source="room",
    )

    assert total_minutes(events) == 50.0


def test_olympus_humidity_30_minutes() -> None:
    points = [(0, 75.0), (5, 75.0), (10, 75.0), (15, 75.0), (20, 75.0), (25, 75.0), (30, 50.0)]
    df = make_series("2025-09-02 10:00", points, channel="Humidity")

    events = _compute_oor_events(df, "Humidity", lo=None, hi=70.0, default_source="olympus")

    assert total_minutes(events) == 30.0


def test_freezer_temperature_30_minutes() -> None:
    points = [(0, -5.0), (5, -5.0), (10, -5.0), (15, -5.0), (20, -5.0), (25, -5.0), (30, -20.0)]
    df = make_series("2025-09-03 09:00", points, channel="Temperature")

    events = _compute_oor_events(df, "Temperature", lo=None, hi=-10.0, default_source="freezer")

    assert total_minutes(events) == 30.0


def test_fridge_two_runs_total_35_minutes() -> None:
    points = [
        (10, 12.0),
        (15, 12.0),
        (20, 12.0),
        (25, 12.0),
        (30, 5.0),
        (40, 0.0),
        (45, 0.0),
        (50, 0.0),
        (55, 4.0),
    ]
    df = make_series("2025-09-04 14:00", points, channel="Temperature")

    events = _compute_oor_events(df, "Temperature", lo=2.0, hi=8.0, default_source="fridge")

    assert total_minutes(events) == 35.0


def test_ult_minus_80_duration_45_minutes() -> None:
    points = [(minutes, -65.0) for minutes in range(0, 45, 5)]
    points.append((45, -75.0))
    df = make_series("2025-09-05 10:00", points, channel="Temperature")

    events = _compute_oor_events(
        df,
        "Temperature",
        lo=-86.0,
        hi=-70.0,
        default_source="ult",
    )

    assert total_minutes(events) == 45.0


def test_single_sample_defaults_to_one_minute() -> None:
    df = make_series("2025-09-06 08:00", [(0, 70.0)], channel="Humidity")

    events = _compute_oor_events(df, "Humidity", lo=None, hi=60.0, default_source="room")

    assert total_minutes(events) == 1.0


def test_exact_threshold_is_in_range() -> None:
    df = make_series(
        "2025-09-07 12:00",
        [(0, 8.0), (5, 8.0), (10, 8.0)],
        channel="Temperature",
    )

    events = _compute_oor_events(df, "Temperature", lo=2.0, hi=8.0, default_source="fridge")

    assert total_minutes(events) == 0.0
