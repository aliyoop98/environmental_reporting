"""Traceable CSV ingestion utilities.

This module implements an alternative ingest path that yields columns such as
``datetime``, ``temp_c`` and ``rh_percent``. The Streamlit app (`app.py`) does
not import this module and instead expects the canonical schema produced by
``data_processing.py`` (``DateTime``, ``Temperature``, ``Humidity``). The note
exists to avoid wiring the two together inadvertently in the future.
"""

from __future__ import annotations

import io
from typing import Dict, Optional

import pandas as pd


def _read_text(path: str) -> str:
    """Read a CSV file as text handling BOMs and stray null bytes."""

    with open(path, "rb") as file:
        raw = file.read()
    cleaned = raw.replace(b"\x00", b"")
    return cleaned.decode("utf-8-sig", errors="replace")


def _detect_legacy_format(text: str) -> bool:
    """Return True when the provided CSV text matches the legacy schema."""

    lower_text = text.lower()
    return "ch1 (%)".lower() in lower_text and "time,date" in lower_text


def _select_first_available(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first column name from ``candidates`` present in ``df``."""

    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _legacy_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the legacy Traceable CSV structure into the canonical schema."""

    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Legacy Traceable format is missing Date or Time columns")

    humidity_col = _select_first_available(
        df,
        "CH1 (%)",
        "CH1 %",
        "CH1",
        "CH3 (%)",
        "CH3 %",
        "CH3",
    )
    temperature_col = _select_first_available(
        df,
        "CH2 (C)",
        "CH2",
        "CH4 (C)",
        "CH4",
    )

    result = pd.DataFrame()
    result["datetime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        errors="coerce",
    )

    if temperature_col is not None:
        result["temp_c"] = pd.to_numeric(df[temperature_col], errors="coerce")
    else:
        result["temp_c"] = pd.Series(dtype="float64")

    if humidity_col is not None:
        result["rh_percent"] = pd.to_numeric(df[humidity_col], errors="coerce")
    else:
        result["rh_percent"] = pd.Series(dtype="float64")

    return result


def _find_column(columns: pd.Index, *keywords: str) -> Optional[str]:
    """Return the first column containing any of the provided keywords."""

    lowered = {str(col).lower(): col for col in columns}
    for col_lower, original in lowered.items():
        for keyword in keywords:
            if keyword in col_lower:
                return original
    return None


def _identify_temp_column(columns: pd.Index) -> Optional[str]:
    """Return the column name that most resembles a temperature measurement."""

    candidates = []
    for col in columns:
        lower = str(col).lower()
        stripped = lower.replace("°", "").replace("deg", "").strip()
        if "temp" in lower or stripped in {"c", "celsius"}:
            candidates.append(col)
        elif "c" in stripped.split():
            candidates.append(col)
    return candidates[0] if candidates else None


def _identify_humidity_column(columns: pd.Index) -> Optional[str]:
    """Return the column name that most resembles a humidity measurement."""

    candidates = []
    for col in columns:
        lower = str(col).lower()
        if "%" in lower or "humidity" in lower or "rh" in lower or "humid" in lower:
            candidates.append(col)
    return candidates[0] if candidates else None


def _new_format_to_dataframes(
    df: pd.DataFrame, serial_alias: Optional[Dict[str, str]]
) -> Dict[str, pd.DataFrame]:
    """Transform the modern Traceable export into per-serial DataFrames."""

    timestamp_col = _find_column(df.columns, "timestamp", "date", "time")
    serial_col = _find_column(df.columns, "serial")
    data_col = _find_column(df.columns, "data", "value", "reading")
    unit_col = _find_column(df.columns, "unit")
    channel_col = _find_column(df.columns, "channel", "sensor")

    if not timestamp_col or not serial_col or not data_col:
        raise ValueError("Traceable CSV missing required columns for new format")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, serial_col])

    pivot_column = unit_col if unit_col and df[unit_col].notna().any() else channel_col

    grouped: Dict[str, pd.DataFrame] = {}
    for serial, serial_df in df.groupby(serial_col):
        if pivot_column is None:
            serial_pivot = serial_df.set_index(timestamp_col)[[data_col]]
        else:
            serial_pivot = serial_df.pivot_table(
                index=timestamp_col,
                columns=pivot_column,
                values=data_col,
                aggfunc="first",
            )
        serial_pivot = serial_pivot.sort_index()

        if isinstance(serial_pivot, pd.Series):  # only one pivot column
            serial_pivot = serial_pivot.to_frame()

        temp_col = _identify_temp_column(serial_pivot.columns)
        humidity_col = _identify_humidity_column(serial_pivot.columns)

        result = pd.DataFrame()
        result["datetime"] = serial_pivot.index
        if temp_col is not None:
            result["temp_c"] = pd.to_numeric(serial_pivot[temp_col], errors="coerce")
        if humidity_col is not None:
            result["rh_percent"] = pd.to_numeric(
                serial_pivot[humidity_col], errors="coerce"
            )

        alias = serial_alias.get(serial, serial) if serial_alias else serial
        grouped[str(alias)] = result.reset_index(drop=True)

    return grouped


def ingest_traceable_csv(
    path: str, serial_alias: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """Parse Traceable probe CSV exports into standardized data frames.

    Parameters
    ----------
    path:
        Path to the Traceable CSV export file.
    serial_alias:
        Optional mapping from probe serial numbers to friendly names.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A mapping of alias/serial identifiers to data frames with the columns
        ``datetime``, ``temp_c``, and ``rh_percent``.
    """

    text = _read_text(path)
    if not text.strip():
        return {}

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise ValueError("Failed to read Traceable CSV") from exc

    df.columns = [str(col).strip() for col in df.columns]

    if _detect_legacy_format(text):
        legacy_df = _legacy_to_dataframe(df)
        return {"legacy_single": legacy_df}

    return _new_format_to_dataframes(df, serial_alias)


__all__ = ["ingest_traceable_csv"]
