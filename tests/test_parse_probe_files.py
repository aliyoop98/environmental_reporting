import io

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing import _parse_probe_files


class InMemoryFile(io.BytesIO):
    def __init__(self, text: str, name: str):
        super().__init__(text.encode('utf-8'))
        self.name = name


    def read(self, *args, **kwargs):  # type: ignore[override]
        return super().read(*args, **kwargs)


def test_parse_legacy_probe_file():
    csv_text = "\n".join(
        ["Date,Time,CH1,CH2", "2024-01-01,12:00:00,55,4.2", ""]
    )
    file_obj = InMemoryFile(csv_text, "legacy.csv")

    dfs, ranges = _parse_probe_files([file_obj])

    assert "legacy.csv" in dfs
    df = dfs["legacy.csv"]
    assert set(["Date", "Time", "DateTime", "Humidity", "Temperature"]).issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["DateTime"])
    assert ranges["legacy.csv"]["Temperature"] == (-35, -5) or ranges["legacy.csv"]["Temperature"] == (15, 25)


def test_parse_consolidated_probe_file():
    csv_text = "\n".join(
        [
            "Timestamp,Serial Number,Channel,Data,Unit of Measure",
            "2024-01-01 12:00:00,SN-123,Temperature,4.5,°C",
            "2024-01-01 12:05:00,SN-123,Humidity,60,%",
            "2024-01-01 12:00:00,SN-999,Temperature,5.1,°C",
            "2024-01-01 12:05:00,SN-999,Humidity,58,%",
            "",
        ]
    )
    file_obj = InMemoryFile(csv_text, "consolidated.csv")

    dfs, ranges = _parse_probe_files([file_obj])

    assert "consolidated.csv [SN-123]" in dfs
    assert "consolidated.csv [SN-999]" in dfs

    df_serial = dfs["consolidated.csv [SN-123]"]
    assert list(df_serial.columns) == ["Date", "Time", "DateTime", "Temperature", "Humidity"]
    assert pd.api.types.is_datetime64_any_dtype(df_serial["DateTime"])
    assert df_serial["Temperature"].dtype.kind in {"f", "i"}
    assert df_serial["Humidity"].dtype.kind in {"f", "i"}
    assert ranges["consolidated.csv [SN-123]"]["Temperature"] == (15, 25)
    assert ranges["consolidated.csv [SN-123]"]["Humidity"] == (0, 60)


def test_parse_consolidated_probe_file_with_space_assignments():
    csv_text = "\n".join(
        [
            "Timestamp,Serial Number,Channel,Data,Unit of Measure,Space Type,Space Name",
            "2024-01-01 12:00:00,EQ-001,Equipment Temperature,4.5,°C,Equipment,Freezer A",
            "2024-01-01 12:05:00,EQ-001,Equipment Humidity,62,%,Equipment,Freezer A",
            "2024-01-01 12:00:00,AMB-002,Ambient Temperature,5.5,°C,Ambient,Freezer A Ambient",
            "2024-01-01 12:05:00,AMB-002,Ambient Humidity,58,%,Ambient,Freezer A Ambient",
            "",
        ]
    )
    file_obj = InMemoryFile(csv_text, "assigned_spaces.csv")

    dfs, ranges = _parse_probe_files([file_obj])

    equipment_key = "assigned_spaces.csv - Freezer A (Equipment) [EQ-001]"
    ambient_key = "assigned_spaces.csv - Freezer A Ambient (Ambient) [AMB-002]"

    assert equipment_key in dfs
    assert ambient_key in dfs

    df_equipment = dfs[equipment_key]
    df_ambient = dfs[ambient_key]

    assert list(df_equipment.columns) == ["Date", "Time", "DateTime", "Temperature", "Humidity"]
    assert list(df_ambient.columns) == ["Date", "Time", "DateTime", "Temperature", "Humidity"]

    assert ranges[equipment_key]["Temperature"] == (15, 25)
    assert ranges[ambient_key]["Humidity"] == (0, 60)


def test_parse_consolidated_probe_file_with_serial_num_alias():
    csv_text = "\n".join(
        [
            "Timestamp,Serial Num,Channel,Data,Unit of Measure",
            "2025-09-11 15:36,250269594,sensor1,67.2,%,",
            "2025-09-11 15:41,250269594,sensor1,68.5,%,",
            "",
        ]
    )
    file_obj = InMemoryFile(csv_text, "alias.csv")

    dfs, ranges = _parse_probe_files([file_obj])

    key = "alias.csv [250269594]"
    assert key in dfs
    df_alias = dfs[key]
    assert list(df_alias.columns) == ["Date", "Time", "DateTime", "Humidity"]
    assert not df_alias.empty
    assert ranges[key]["Humidity"] == (0, 60)
