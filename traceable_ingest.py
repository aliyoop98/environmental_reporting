"""Traceable CSV ingestion utilities.

This module implements an alternative ingest path that yields canonical
columns (``DateTime``, ``Temperature``, ``Humidity``) similar to the
``data_processing`` pipeline. The Streamlit app (`app.py`) does not import this
module and instead expects the canonical schema produced elsewhere; the note
exists to avoid wiring the two together inadvertently in the future.
"""

from __future__ import annotations

import io
from typing import Dict, Optional

import pandas as pd


SERIAL_KIND_OVERRIDES: Dict[str, Dict[str, str]] = {
    "250269655": {"sensor1": "Humidity", "sensor2": "Temperature"},
    "250269656": {"sensor1": "Humidity", "sensor2": "Temperature"},
    "250259653": {"sensor1": "Humidity", "sensor2": "Temperature"},
}


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


def _normalize_text(value: object) -> str:
    """Return a stripped, lower-case representation of arbitrary text."""

    if value is None:
        return ""
    text = str(value).strip()
    return text.lower()


def _normalize_channel_key(value: object) -> str:
    """Return a compact, alphanumeric-only channel key for comparisons."""

    normalized = _normalize_text(value)
    return "".join(char for char in normalized if char.isalnum())


def _classify_channel(channel: str, unit: str) -> Optional[str]:
    """Classify a Traceable measurement into Temperature or Humidity."""

    channel_norm = _normalize_text(channel)
    channel_key = _normalize_channel_key(channel)
    unit_norm = _normalize_text(unit)

    direct_map = {
        "sensor1": "Humidity",
        "sensor2": "Temperature",
        "sensor01": "Humidity",
        "sensor02": "Temperature",
    }
    if channel_key in direct_map:
        return direct_map[channel_key]

    if not channel_norm and unit_norm:
        channel_norm = unit_norm

    if any(token in channel_norm for token in ("humid", "rh")):
        return "Humidity"
    if any(token in unit_norm for token in ("%", "percent")):
        return "Humidity"

    temp_tokens = ("temp", "°c", "celsius", "degc", "°f", "fahrenheit")
    if any(token in channel_norm for token in temp_tokens):
        return "Temperature"
    if any(token in unit_norm for token in temp_tokens):
        return "Temperature"

    if channel_norm in {"temperature", "humidity"}:
        return channel_norm.capitalize()

    return None


def _apply_serial_override(serial: str, channel: str) -> Optional[str]:
    """Return a forced kind for a serial/channel pair when configured."""

    serial_str = str(serial).strip()
    if not serial_str or serial_str not in SERIAL_KIND_OVERRIDES:
        return None

    overrides = SERIAL_KIND_OVERRIDES[serial_str]
    channel_key = _normalize_channel_key(channel)

    for candidate, kind in overrides.items():
        if _normalize_channel_key(candidate) == channel_key:
            return kind

    return None


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

    grouped: Dict[str, pd.DataFrame] = {}

    measurement_col = "__measurement"
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, serial_col])

    df[serial_col] = df[serial_col].astype(str).str.strip()
    df[data_col] = pd.to_numeric(df[data_col], errors="coerce")

    if channel_col:
        channel_values = df[channel_col]
    else:
        channel_values = pd.Series([""] * len(df), index=df.index)
    if unit_col:
        unit_values = df[unit_col]
    else:
        unit_values = pd.Series([""] * len(df), index=df.index)
    df[measurement_col] = [
        _apply_serial_override(serial, channel)
        or _classify_channel(channel, unit)
        for serial, channel, unit in zip(df[serial_col], channel_values, unit_values)
    ]
    df = df.dropna(subset=[measurement_col])

    if df.empty:
        return {}

    for serial, serial_df in df.groupby(serial_col):
        if serial_df.empty:
            continue
        pivot = (
            serial_df.pivot_table(
                index=timestamp_col,
                columns=measurement_col,
                values=data_col,
                aggfunc="last",
            )
            .sort_index()
            .rename_axis(None, axis=1)
        )

        if pivot.empty:
            continue

        wide = pivot.reset_index().rename(columns={timestamp_col: "DateTime"})
        wide = wide.sort_values("DateTime")

        for column in ("Temperature", "Humidity"):
            if column in wide.columns:
                wide[column] = pd.to_numeric(wide[column], errors="coerce")
            else:
                wide[column] = pd.Series(pd.NA, index=wide.index, dtype="Float64")

        ordered = ["DateTime", "Temperature", "Humidity"]
        wide = wide[ordered]

        serial_str = str(serial)
        alias = serial_alias.get(serial_str, serial_str) if serial_alias else serial_str
        grouped[str(alias)] = wide.reset_index(drop=True)

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
