from io import BytesIO
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing import parse_serial_csv


class _BytesFile(BytesIO):
    def __init__(self, data: str, name: str = "report.csv"):
        super().__init__(data.encode("utf-8"))
        self.name = name


def test_traceable_report_infers_freezer_limits():
    report = """Device ID: 12345\nDevice Name: Walk-in Freezer\nReport Timezone: UTC\n\nTimestamp,Data,Range Low,Range High\n2024-01-01 00:00,-20,,\n2024-01-01 01:00,-18,,\n"""

    serials = parse_serial_csv([_BytesFile(report, name="Walk-in Freezer.csv")])
    assert serials

    info = next(iter(serials.values()))
    df = info.get("df")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    range_map = info.get("range_map")
    assert isinstance(range_map, dict)
    assert range_map.get("Temperature") == (-35, -5)
