from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing import merge_serial_data


def _frame(values):
    df = pd.DataFrame(values)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df


def test_merge_serial_data_combines_and_deduplicates():
    existing = {}
    first = _frame(
        [
            {'DateTime': '2024-09-10T00:00:00', 'Temperature': 5.0},
            {'DateTime': '2024-09-11T00:00:00', 'Temperature': 6.0},
        ]
    )
    second = _frame(
        [
            {'DateTime': '2024-09-11T00:00:00', 'Temperature': 6.5},
            {'DateTime': '2024-09-12T00:00:00', 'Temperature': 7.0},
        ]
    )

    merged_first = merge_serial_data(existing, first, 'SER123')
    assert len(merged_first) == 2
    assert existing['SER123'] is merged_first

    merged_second = merge_serial_data(existing, second, 'SER123')

    assert existing['SER123'] is merged_second
    assert list(merged_second['DateTime']) == [
        datetime(2024, 9, 10, 0, 0),
        datetime(2024, 9, 11, 0, 0),
        datetime(2024, 9, 12, 0, 0),
    ]
    # The duplicate timestamp keeps the most recent value (6.5).
    value_series = merged_second.loc[
        merged_second['DateTime'] == datetime(2024, 9, 11, 0, 0), 'Temperature'
    ]
    assert len(value_series) == 1
    assert pytest.approx(value_series.iloc[0], rel=1e-9) == 6.5
