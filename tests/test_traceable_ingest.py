import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from traceable_ingest import ingest_traceable_csv


def test_ingest_traceable_csv_pivots_to_wide(tmp_path):
    csv_text = "\n".join(
        [
            "Timestamp,Serial Number,Channel,Data,Unit of Measure",
            "2024-01-01 00:00:00,SN-1,Sensor 2,4.5,°C",
            "2024-01-01 00:00:00,SN-1,Sensor 1,60,% RH",
            "2024-01-01 00:05:00,SN-1,Sensor 2,4.7,°C",
            "2024-01-01 00:05:00,SN-1,Sensor 1,61,% RH",
            "",
        ]
    )
    csv_path = tmp_path / "traceable.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    results = ingest_traceable_csv(str(csv_path))

    assert "SN-1" in results
    df = results["SN-1"]
    assert list(df.columns) == ["DateTime", "Temperature", "Humidity"]
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["DateTime"])
    assert df["Temperature"].tolist() == [4.5, 4.7]
    assert df["Humidity"].tolist() == [60.0, 61.0]


def test_ingest_traceable_csv_includes_missing_channel(tmp_path):
    csv_text = "\n".join(
        [
            "Timestamp,Serial Number,Channel,Data,Unit of Measure",
            "2024-02-01 00:00:00,SN-2,Sensor 2,5.0,°C",
            "2024-02-01 00:05:00,SN-2,Sensor 2,5.1,°C",
            "",
        ]
    )
    csv_path = tmp_path / "temperature_only.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    results = ingest_traceable_csv(str(csv_path))

    assert "SN-2" in results
    df = results["SN-2"]
    assert list(df.columns) == ["DateTime", "Temperature", "Humidity"]
    assert df["Humidity"].isna().all()
    assert df["Temperature"].tolist() == [5.0, 5.1]


def test_ingest_traceable_respects_serial_overrides(tmp_path):
    csv_text = "\n".join(
        [
            "Timestamp,Serial Number,Channel,Data,Unit of Measure",
            "2024-03-01 00:00:00,250269656,Sensor-2,7.0,",  # noisy channel, blank unit
            "2024-03-01 00:00:00,250269656,Sensor-1,40,",  # noisy channel, blank unit
            "",
        ]
    )
    csv_path = tmp_path / "override.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    results = ingest_traceable_csv(str(csv_path))

    df = results["250269656"]
    assert list(df.columns) == ["DateTime", "Temperature", "Humidity"]
    assert df["Temperature"].tolist() == [7.0]
    assert df["Humidity"].tolist() == [40.0]
