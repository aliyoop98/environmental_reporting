import pandas as pd

from oor import OOR_MIN_SINGLE_SAMPLE_MINUTES, _compute_oor_events, _is_oor_strict


def test_is_oor_strict_enforces_strict_bounds() -> None:
    lo, hi = -80.0, -60.0
    assert not _is_oor_strict(-80.0, lo, hi)
    assert not _is_oor_strict(-60.0, lo, hi)
    assert _is_oor_strict(-81.0, lo, hi)
    assert _is_oor_strict(-59.9, lo, hi)


def test_compute_oor_events_duration_and_single_sample_minimum() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:05",
            "2024-01-01 00:10",
            "2024-01-01 00:10",
            "2024-01-01 00:20",
            "2024-01-01 00:25",
            "2024-01-01 01:00",
            "2024-01-01 01:05",
        ]
    )
    values = [-70.0, -95.0, -92.0, -91.5, -91.0, -65.0, -95.0, -70.0]
    sources = [
        "primary",
        "primary",
        "overlay",
        "primary",
        "primary",
        "primary",
        None,
        "primary",
    ]
    df = pd.DataFrame({"DateTime": timestamps, "Temp": values, "Source": sources})

    events = _compute_oor_events(df, "Temp", lo=-90.0, hi=-60.0, default_source="primary")

    assert len(events) == 2
    first, second = events.iloc[0], events.iloc[1]

    # First run: three unique timestamps -> gaps (5 + 10) minutes = 15 total.
    assert first["Duration(min)"] == 15.0
    assert first["Start"] == pd.Timestamp("2024-01-01 00:05")
    assert first["End"] == pd.Timestamp("2024-01-01 00:20")
    assert first["Source"] == "primary"

    # Second run: single sample should be rounded up to policy minimum.
    assert second["Duration(min)"] == OOR_MIN_SINGLE_SAMPLE_MINUTES
    assert second["Start"] == pd.Timestamp("2024-01-01 01:00")
    assert second["End"] == pd.Timestamp("2024-01-01 01:00")
    assert second["Source"] == "primary"

    # Total duration respects sum of individual runs.
    assert events["Duration(min)"].sum() == 15.0 + OOR_MIN_SINGLE_SAMPLE_MINUTES
