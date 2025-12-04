import sys
from io import BytesIO
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing import parse_serial_csv


def test_serial_overrides_apply_with_noisy_channels_and_units():
    csv_text = "\n".join(
        [
            "Timestamp,Serial Number,Channel,Data,Unit of Measure",
            "2025-Oct-01 00:01,250269656,Sensor-2,19.84,Â°C",  # noisy degree symbol
            "2025-Oct-01 00:01,250269656,Sensor-1,66.2,%",
            "",
        ]
    )
    file_obj = BytesIO(csv_text.encode("utf-8"))
    file_obj.name = "traceable.csv"

    frames = parse_serial_csv([file_obj])

    assert "250269656" in frames
    df = frames["250269656"]["df"]
    assert list(df.columns)[:3] == ["DateTime", "Temperature", "Humidity"]
    assert df["Temperature"].iloc[0] == 19.84
    assert df["Humidity"].iloc[0] == 66.2


def test_serial_overrides_with_comma_decimal_values():
    csv_text = "\n".join(
        [
            "Timestamp;Serial Number;Channel;Data;Unit of Measure",
            "2025-Oct-01 00:01;250269655;sensor2;19,84;°C",  # comma decimal
            "2025-Oct-01 00:01;250269655;sensor1;66,2;%",
        ]
    )
    file_obj = BytesIO(csv_text.encode("utf-8"))
    file_obj.name = "traceable.csv"

    frames = parse_serial_csv([file_obj])

    assert "250269655" in frames
    df = frames["250269655"]["df"]
    assert df["Temperature"].iloc[0] == 19.84
    assert df["Humidity"].iloc[0] == 66.2


def test_serial_overrides_allow_leading_decimal_values():
    csv_text = "\n".join(
        [
            "Timestamp,Serial Number,Channel,Data,Unit of Measure",
            "2025-Oct-01 00:01,250259653,sensor2,.84,°C",  # leading decimal without zero
            "2025-Oct-01 00:01,250259653,sensor1,.62,%",  # leading decimal without zero
        ]
    )
    file_obj = BytesIO(csv_text.encode("utf-8"))
    file_obj.name = "traceable.csv"

    frames = parse_serial_csv([file_obj])

    assert "250259653" in frames
    df = frames["250259653"]["df"]
    assert df["Temperature"].iloc[0] == 0.84
    assert df["Humidity"].iloc[0] == 0.62
