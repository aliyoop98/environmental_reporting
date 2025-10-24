import io
import re
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

_ZERO_WIDTH_CHARS = ("\ufeff", "\u200b", "\u200c", "\u200d")

DEFAULT_TEMP_RANGE: Tuple[float, float] = (15, 25)
DEFAULT_HUMIDITY_RANGE: Tuple[float, float] = (0, 60)

TEMP_COL_ALIASES = ["Temperature", "Temp", "Temp (°C)", "Temperature (°C)"]
HUMI_COL_ALIASES = ["Humidity", "Humidity (%RH)", "RH", "RH (%)"]

PROFILE_LIMITS: Dict[str, Dict[str, Optional[Tuple[float, float]]]] = {
    "Freezer": {"temp": (-35.0, -5.0), "humi": None},
    "Fridge": {"temp": (2.0, 8.0), "humi": None},
    "Room": {"temp": (15.0, 25.0), "humi": (0.0, 60.0)},
    "Area": {"temp": (15.0, 25.0), "humi": (0.0, 60.0)},
    "Olympus": {"temp": (15.0, 28.0), "humi": (0.0, 80.0)},
    "Olympus Room": {"temp": (15.0, 28.0), "humi": (0.0, 80.0)},
}


def _parse_ts(value: object) -> pd.Timestamp:
    """Return a normalized timestamp using deterministic formats first."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%b-%d %H:%M",
    ):
        try:
            return pd.to_datetime(text, format=fmt)
        except Exception:
            continue

    return pd.to_datetime(text, errors="coerce")


def _infer_profile_from_name(*candidates: Optional[str]) -> Optional[str]:
    """Infer an environmental profile from a collection of text hints."""

    pieces: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        pieces.append(str(candidate))
    if not pieces:
        return None
    text = " ".join(pieces).lower()
    if "freezer" in text:
        return "Freezer"
    if "fridge" in text or "refrigerator" in text:
        return "Fridge"
    if "room" in text or "area" in text:
        return "Room"
    if "olympus room" in text:
        return "Olympus Room"
    if "olympus" in text:
        return "Olympus"
    return None


def _apply_inferred_ranges(
    range_map: Dict[str, Tuple[float, float]], *name_hints: Optional[str]
) -> None:
    """Populate range_map with inferred temperature and humidity ranges."""

    profile = _infer_profile_from_name(*name_hints)
    limits = PROFILE_LIMITS.get(profile or "") if profile else None
    if not limits:
        limits = {"temp": None, "humi": DEFAULT_HUMIDITY_RANGE}

    temp_range = limits.get("temp") or range_map.get("Temperature") or DEFAULT_TEMP_RANGE
    humidity_range = limits.get("humi") or range_map.get("Humidity") or DEFAULT_HUMIDITY_RANGE

    if temp_range:
        for alias in TEMP_COL_ALIASES:
            range_map.setdefault(alias, temp_range)
    if humidity_range:
        for alias in HUMI_COL_ALIASES:
            range_map.setdefault(alias, humidity_range)


def _strip_bom_and_zero_width(text: str) -> str:
    for ch in _ZERO_WIDTH_CHARS:
        text = text.replace(ch, "")
    return text


_DEVICE_ID_RE = re.compile(r"device\s*id\s*:\s*(\d+)", re.IGNORECASE)


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("Temperature", "Humidity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    return df


def _finalize_serial_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    if "DateTime" in result.columns:
        result["DateTime"] = pd.to_datetime(result["DateTime"], errors="coerce")
        result = result.dropna(subset=["DateTime"])  # type: ignore[arg-type]
        result = result.sort_values("DateTime")
        result["Date"] = result["DateTime"].dt.date
        result["Time"] = result["DateTime"].dt.strftime("%H:%M")

    subset = ["DateTime"]
    for col in ("Temperature", "Humidity"):
        if col in result.columns:
            subset.append(col)
    if len(subset) > 1:
        result = result.drop_duplicates(subset=subset, keep="last")

    result = _downcast_numeric(result)

    ordered: List[str] = []
    for col in ["Date", "Time", "DateTime", "Temperature", "Humidity"]:
        if col in result.columns:
            ordered.append(col)
    if ordered:
        result = result[ordered]

    return result.reset_index(drop=True)


def _extract_report_blocks(
    text: str,
) -> Optional[Tuple[Dict[str, Optional[str]], List[Tuple[str, str]]]]:
    """Return metadata and per-channel CSV blocks from a Traceable report."""

    if not text:
        return None

    lines = [line.replace("\ufeff", "") for line in text.splitlines()]
    lines = [_strip_bom_and_zero_width(line) for line in lines]
    if not any(line.lstrip().lower().startswith("device id:") for line in lines):
        return None

    device_id: Optional[str] = None
    device_name: Optional[str] = None
    timezone: Optional[str] = None
    for line in lines[:40]:
        match = _DEVICE_ID_RE.search(line)
        if match:
            device_id = match.group(1)
        lower = line.lower()
        if lower.startswith("device name:"):
            device_name = line.split(":", 1)[1].strip()
        elif lower.startswith("report timezone:"):
            timezone = line.split(":", 1)[1].strip()

    header_indexes = [
        idx
        for idx, line in enumerate(lines)
        if line.strip().lower().startswith("timestamp,") and "data" in line.lower()
    ]
    if not header_indexes:
        return None

    blocks: List[Tuple[str, str]] = []
    for start in header_indexes:
        end = len(lines)
        for pos in range(start + 1, len(lines)):
            stripped = lines[pos].strip()
            lowered = stripped.lower()
            if lowered.startswith("channel data for") and pos > start:
                end = pos
                break
            if pos in header_indexes and pos > start:
                end = pos
                break

        header = lines[start]
        block_lines = [header]
        for row in lines[start + 1 : end]:
            if not row.strip():
                continue
            block_lines.append(row)

        if len(block_lines) <= 1:
            continue

        context_label = ""
        for offset in range(start - 1, -1, -1):
            candidate = lines[offset].strip()
            if not candidate:
                continue
            context_label = candidate
            break

        blocks.append((context_label, "\n".join(block_lines)))

    if not blocks:
        return None

    meta = {"device_id": device_id, "device_name": device_name, "timezone": timezone}
    return meta, blocks


_TEMP_RANGE_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[float, float]], ...] = (
    (("ultra", "low", "freezer"), (-90, -60)),
    (("ultralow",), (-90, -60)),
    (("ult",), (-90, -60)),
    (("minus", "80"), (-90, -60)),
    (("-80",), (-90, -60)),
    (("freezer",), (-35, -5)),
    (("cold", "room"), (2, 8)),
    (("cooler",), (2, 8)),
    (("fridge",), (2, 8)),
    (("refrigerator",), (2, 8)),
)


def _normalise_context_text(*values: Optional[str]) -> str:
    """Return a normalised string used for range inference heuristics."""

    pieces: List[str] = []
    for value in values:
        if not value:
            continue
        cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", value.lower())
        cleaned = cleaned.replace("-", " ")
        pieces.append(" ".join(cleaned.split()))
    return " ".join(piece for piece in pieces if piece)


def _infer_temperature_range(
    context: str, default_range: Tuple[float, float]
) -> Tuple[float, float]:
    """Infer a sensible temperature range from contextual text."""

    if not context:
        return default_range
    for terms, range_values in _TEMP_RANGE_HINTS:
        if all(term in context for term in terms):
            return range_values
    return default_range


def _parse_traceable_report_csv(
    text: str,
    source_name: str,
) -> List[Dict[str, object]]:
    """Parse Traceable Live single-serial report CSVs into the unified schema."""

    extracted = _extract_report_blocks(text)
    if not extracted:
        return []

    meta, blocks = extracted
    frames: List[pd.DataFrame] = []
    for label, csv_text in blocks:
        try:
            frame = pd.read_csv(
                io.StringIO(csv_text),
                engine="python",
                on_bad_lines="skip",
                dtype=str,
            )
        except Exception:
            continue
        if frame.empty:
            continue
        frame = _clean_chunk_columns(frame)
        frame["__block_label"] = label
        frames.append(frame)

    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)

    normalized_cols = {
        _normalize_header(col): col for col in df.columns if isinstance(col, str)
    }

    def _find_column(*candidates: str) -> Optional[str]:
        for candidate in candidates:
            if candidate in normalized_cols:
                return normalized_cols[candidate]
        return None

    timestamp_col = _find_column("timestamp", "time stamp", "date time")
    data_col = _find_column("data", "value", "reading")
    channel_col = _find_column("channel", "sensor", "channel name")
    unit_col = _find_column("unit of measure", "unit", "units")

    if not timestamp_col or not data_col:
        return []

    rename_map = {timestamp_col: "Timestamp", data_col: "Data"}
    if channel_col:
        rename_map[channel_col] = "Channel"
    if unit_col:
        rename_map[unit_col] = "Unit"
    df = df.rename(columns=rename_map)

    range_columns = [
        original
        for normalised, original in normalized_cols.items()
        if normalised.startswith("range")
    ]
    if range_columns:
        df = df.drop(columns=[col for col in range_columns if col in df.columns])

    if "Channel" not in df.columns:
        df["Channel"] = ""
    if "Unit" not in df.columns:
        df["Unit"] = ""

    df["Channel"] = df["Channel"].astype("string").fillna("").str.strip()
    df["Unit"] = df["Unit"].astype("string").fillna("").str.strip()

    data_cleaned = (
        df["Data"].astype("string")
        .str.extract(r"([-+]?\d*\.?\d+)")[0]
        .astype("string")
    )
    df["Value"] = pd.to_numeric(data_cleaned, errors="coerce")
    df["DateTime"] = df["Timestamp"].apply(_parse_ts)

    block_labels = (
        df.pop("__block_label") if "__block_label" in df.columns else pd.Series()
    )
    if block_labels.empty:
        block_labels = pd.Series(["" for _ in range(len(df))], index=df.index)

    def _resolve_measurement(channel: str, unit: str, context: str) -> Optional[str]:
        measurement = _classify_measurement(channel, unit)
        if measurement:
            return measurement
        if context:
            measurement = _classify_measurement(context, unit)
            if measurement:
                return measurement
        if unit:
            return _classify_measurement("", unit)
        return None

    measurement_values = [
        _resolve_measurement(channel, unit, context)
        for channel, unit, context in zip(
            df["Channel"].astype("string"),
            df["Unit"].astype("string"),
            block_labels.astype("string"),
        )
    ]
    measurement_series = pd.Series(measurement_values, index=df.index, dtype="string")
    if measurement_series.notna().sum() == 0:
        measurement_series = pd.Series(
            ["Temperature"] * len(df), index=df.index, dtype="string"
        )
    else:
        known = [value for value in measurement_series.dropna().unique() if value]
        if len(known) == 1:
            measurement_series = measurement_series.fillna(known[0])
    df["Measurement"] = measurement_series

    df = df.dropna(subset=["DateTime", "Value"], how="any")
    df = df[df["Measurement"].astype(str).str.strip() != ""]
    if df.empty:
        return []

    wide = (
        df.pivot_table(
            index="DateTime",
            columns="Measurement",
            values="Value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values("DateTime")
    )

    for column in ("Temperature", "Humidity"):
        if column not in wide.columns:
            wide[column] = pd.NA

    wide["Date"] = wide["DateTime"].dt.date
    wide["Time"] = wide["DateTime"].dt.strftime("%H:%M")
    out = _finalize_serial_dataframe(wide)

    device_id = (meta.get("device_id") or "").strip()
    device_name = (meta.get("device_name") or "").strip()
    serial = device_id or device_name or Path(source_name).stem
    default_label = device_name or serial
    option_label = device_name or Path(source_name).stem

    range_map: Dict[str, Tuple[float, float]] = {}
    _apply_inferred_ranges(range_map, source_name, device_name, device_id)

    channels = [
        column
        for column in ("Temperature", "Humidity")
        if column in out.columns and column in wide.columns
    ]

    return [
        {
            "df": out,
            "serial": str(serial),
            "range_map": range_map,
            "default_label": default_label,
            "option_label": option_label,
            "source_name": source_name,
            "channels": channels,
        }
    ]


def _looks_like_traceable_report(text: str) -> bool:
    if not text:
        return False
    lines = [
        _strip_bom_and_zero_width(line)
        for line in text.splitlines()
        if line.strip()
    ]
    if not lines:
        return False
    has_device = any("device id:" in line.lower() for line in lines[:40])
    if not has_device:
        return False
    return any(
        "timestamp" in line.lower() and "data" in line.lower() for line in lines
    )


def _looks_like_consolidated(text: str) -> bool:
    if not text:
        return False
    for line in text.splitlines()[:50]:
        lowered = _strip_bom_and_zero_width(line).lower()
        if "serial number" in lowered and "channel" in lowered:
            return True
    return False


def _parse_consolidated_serial_csv(
    text: str,
    source_name: str,
) -> List[Dict[str, object]]:
    """Parse Traceable Live consolidated CSVs into the unified schema."""

    try:
        df = pd.read_csv(
            io.StringIO(text),
            engine="python",
            on_bad_lines="skip",
            dtype=str,
            low_memory=False,
        )
    except Exception:
        return []

    if df.empty:
        return []

    df = _clean_chunk_columns(df)

    normalized_cols = {
        _normalize_header(col): col for col in df.columns if isinstance(col, str)
    }

    def _find_column(*candidates: str) -> Optional[str]:
        for candidate in candidates:
            if candidate in normalized_cols:
                return normalized_cols[candidate]
        return None

    timestamp_col = _find_column("timestamp", "time stamp", "date time")
    serial_col = _find_column("serial number", "serial")
    channel_col = _find_column("channel", "sensor", "channel name")
    data_col = _find_column("data", "value", "reading")
    unit_col = _find_column("unit of measure", "unit", "units")

    if not all([timestamp_col, serial_col, channel_col, data_col]):
        return []

    rename_map = {
        timestamp_col: "Timestamp",
        serial_col: "Serial Number",
        channel_col: "Channel",
        data_col: "Data",
    }
    if unit_col:
        rename_map[unit_col] = "Unit of Measure"

    df = df.rename(columns=rename_map)

    for column in ["Timestamp", "Serial Number", "Channel", "Data"]:
        if column not in df.columns:
            df[column] = ""
    if "Unit of Measure" not in df.columns:
        df["Unit of Measure"] = ""

    df["Timestamp"] = df["Timestamp"].astype("string").fillna("").str.strip()
    df["Serial Number"] = df["Serial Number"].astype("string").fillna("").str.strip()
    df["Channel"] = df["Channel"].astype("string").fillna("").str.strip()
    df["Unit of Measure"] = (
        df["Unit of Measure"].astype("string").fillna("").str.strip()
    )
    df["Data"] = df["Data"].astype("string").fillna("")

    df["DateTime"] = df["Timestamp"].apply(_parse_ts)
    df = df.dropna(subset=["DateTime"])
    df = df[df["Serial Number"].astype(str).str.strip() != ""]

    data_cleaned = df["Data"].str.extract(r"([-+]?\d*\.?\d+)")[0]
    df["Value"] = pd.to_numeric(data_cleaned, errors="coerce")
    df = df.dropna(subset=["Value"])

    measurement_series = [
        _classify_measurement(channel, unit)
        or _classify_measurement("", unit)
        for channel, unit in zip(df["Channel"], df["Unit of Measure"])
    ]
    df["Measurement"] = pd.Series(measurement_series, index=df.index, dtype="string")
    df = df[df["Measurement"].astype(str).str.strip() != ""]
    if df.empty:
        return []

    space_name_column = _find_column_by_terms(df.columns, _SPACE_NAME_HINTS)
    space_type_column = _find_column_by_terms(df.columns, _SPACE_TYPE_HINTS)

    results: List[Dict[str, object]] = []
    for serial, group in df.groupby("Serial Number"):
        if not str(serial).strip():
            continue

        wide = (
            group.pivot_table(
                index="DateTime",
                columns="Measurement",
                values="Value",
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(None, axis=1)
            .sort_values("DateTime")
        )

        for column in ("Temperature", "Humidity"):
            if column not in wide.columns:
                wide[column] = pd.NA

        wide["Date"] = wide["DateTime"].dt.date
        wide["Time"] = wide["DateTime"].dt.strftime("%H:%M")
        out = _finalize_serial_dataframe(wide)

        hints: List[Optional[str]] = [source_name, str(serial)]
        if space_name_column and space_name_column in group.columns:
            cleaned = _clean_metadata_series(group[space_name_column])
            non_blank = cleaned[cleaned.astype(str).str.strip().astype(bool)]
            if not non_blank.empty:
                hints.append(non_blank.iloc[0])
        if space_type_column and space_type_column in group.columns:
            cleaned = _clean_metadata_series(group[space_type_column])
            non_blank = cleaned[cleaned.astype(str).str.strip().astype(bool)]
            if not non_blank.empty:
                hints.append(non_blank.iloc[0])

        range_map: Dict[str, Tuple[float, float]] = {}
        _apply_inferred_ranges(range_map, *hints)

        channels = [
            column
            for column in ("Temperature", "Humidity")
            if column in out.columns and column in wide.columns
        ]

        default_label = str(serial)
        option_label = str(serial)

        results.append(
            {
                "df": out,
                "serial": str(serial),
                "range_map": range_map,
                "default_label": default_label,
                "option_label": option_label,
                "source_name": source_name,
                "channels": channels,
            }
        )

    return results

NEW_SCHEMA_ALIASES: Mapping[str, Tuple[str, ...]] = {
    'Timestamp': (
        'timestamp',
        'time stamp',
        'date time',
        'date/time',
        'datetime',
    ),
    'Serial Number': (
        'serial number',
        'serial num',
        'serial no',
        'serial #',
        'serial',
        'serialnumber',
        'serialno',
    ),
    'Channel': (
        'channel',
        'sensor channel',
        'probe channel',
        'sensor',
    ),
    'Data': (
        'data',
        'value',
        'reading',
    ),
    'Unit of Measure': (
        'unit of measure',
        'unit',
        'units',
        'uom',
        'measurement unit',
    ),
}

_SPACE_NAME_HINTS: Tuple[Tuple[str, ...], ...] = (
    ("assignment", "name"),
    ("space", "name"),
    ("location", "name"),
    ("asset", "name"),
    ("subject", "name"),
    ("target", "name"),
    ("environment", "name"),
    ("area", "name"),
    ("zone", "name"),
    ("equipment", "name"),
    ("ambient", "name"),
    ("probe", "name"),
)

_SPACE_TYPE_HINTS: Tuple[Tuple[str, ...], ...] = (
    ("assignment", "type"),
    ("space", "type"),
    ("location", "type"),
    ("asset", "type"),
    ("subject", "type"),
    ("target", "type"),
    ("environment", "type"),
    ("area", "type"),
    ("zone", "type"),
    ("probe", "type"),
    ("space", "category"),
    ("environment", "category"),
)


def _read_csv_flexible(text: str) -> Optional[pd.DataFrame]:
    """Read CSV content supporting multiple delimiters and BOM-bearing headers."""

    # Strip UTF-8 BOM(s) and zero-widths that sometimes stick to the first header.
    if text and text[0] == "\ufeff":
        text = text.lstrip("\ufeff")
    text = _strip_bom_and_zero_width(text)

    for kwargs in ({"sep": None, "engine": "python"}, {}):
        try:
            df = pd.read_csv(
                io.StringIO(text), on_bad_lines="skip", index_col=False, **kwargs
            )
        except Exception:
            continue

        def _clean_column(column: object) -> object:
            if not isinstance(column, str):
                return column
            return _strip_bom_and_zero_width(column).strip()

        return df.rename(columns=_clean_column)
    return None


def _clean_chunk_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        if isinstance(col, str):
            rename_map[col] = _strip_bom_and_zero_width(col).strip()
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _parse_consolidated_serial_text(
    text: str,
    name: str,
    default_temp_range: Tuple[float, float],
    default_humidity_range: Tuple[float, float],
) -> Dict[str, Dict[str, object]]:
    cleaned_text = _strip_bom_and_zero_width(text)
    attempts = ({"sep": None}, {})
    base_read_csv_options = {
        "chunksize": 50_000,
        "index_col": False,
        "engine": "python",
        "on_bad_lines": "skip",
        "skip_blank_lines": True,
        "dtype": {
            "Timestamp": "string",
            "Serial Number": "string",
            "Channel": "string",
            "Data": "string",
            "Unit of Measure": "string",
        },
        "keep_default_na": False,
        "low_memory": False,
    }
    aggregated: Dict[str, Dict[str, object]] = {}

    for kwargs in attempts:
        try:
            reader = pd.read_csv(
                io.StringIO(cleaned_text),
                **{**base_read_csv_options, **kwargs},
            )
            first_chunk = next(reader)
        except StopIteration:
            return {}
        except Exception:
            continue

        for chunk in chain([first_chunk], reader):
            if chunk is None or chunk.empty:
                continue
            chunk = _clean_chunk_columns(chunk)
            try:
                groups = _process_new_schema_df(
                    chunk,
                    name,
                    default_temp_range,
                    default_humidity_range,
                )
            except Exception:
                continue
            for group in groups:
                key = group['key']  # type: ignore[index]
                bucket = aggregated.setdefault(
                    key,
                    {
                        'dfs': [],
                        'range_map': {},
                        'serial': group.get('serial', ''),
                        'default_label': group.get('default_label', ''),
                        'option_label': group.get('option_label', ''),
                        'source_name': group.get('source_name', name),
                        'channels': [],
                    },
                )
                bucket['dfs'].append(group.get('df'))
                bucket['range_map'].update(group.get('range_map', {}))
                if group.get('serial') and not bucket.get('serial'):
                    bucket['serial'] = group.get('serial')
                if group.get('default_label') and not bucket.get('default_label'):
                    bucket['default_label'] = group.get('default_label')
                if group.get('option_label') and not bucket.get('option_label'):
                    bucket['option_label'] = group.get('option_label')
                if group.get('source_name'):
                    bucket['source_name'] = group.get('source_name')
                channels = bucket.setdefault('channels', [])
                for ch in group.get('channels', []):
                    if ch not in channels:
                        channels.append(ch)
        break

    results: Dict[str, Dict[str, object]] = {}
    for key, bucket in aggregated.items():
        dfs = [df for df in bucket.get('dfs', []) if isinstance(df, pd.DataFrame)]
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        finalized = _finalize_serial_dataframe(combined)
        if finalized.empty:
            continue
        range_map = dict(bucket.get('range_map', {}))
        range_map.setdefault('Humidity', DEFAULT_HUMIDITY_RANGE)
        _apply_inferred_ranges(
            range_map,
            bucket.get('source_name', name),
            bucket.get('serial', ''),
            bucket.get('default_label', ''),
            bucket.get('option_label', ''),
        )
        results[key] = {
            'df': finalized,
            'range_map': range_map,
            'serial': bucket.get('serial', ''),
            'default_label': bucket.get('default_label', ''),
            'option_label': bucket.get('option_label', ''),
            'source_name': bucket.get('source_name', name),
            'channels': bucket.get('channels', []),
        }

    return results


def _normalize_header(value: str) -> str:
    """Return a normalized representation of a CSV header for matching."""

    value = _strip_bom_and_zero_width(value)

    cleaned = value.strip().lower()
    cleaned = cleaned.replace('#', ' number ')
    cleaned = cleaned.replace('-', ' ')
    cleaned = cleaned.replace('_', ' ')
    cleaned = ' '.join(cleaned.split())
    return cleaned


def _match_new_schema_columns(columns: Iterable[str]) -> Optional[Dict[str, str]]:
    """Return mapping of original headers to canonical names for the new schema."""

    normalized = {
        col: _normalize_header(col)
        for col in columns
        if isinstance(col, str)
    }

    def _find_match(
        aliases: Tuple[str, ...], used: set[str]
    ) -> Optional[str]:
        return next(
            (
                original
                for original, norm in normalized.items()
                if original not in used and norm in aliases
            ),
            None,
        )

    matches: Dict[str, str] = {}
    used: set[str] = set()

    required_fields = (
        'Timestamp',
        'Serial Number',
        'Channel',
        'Data',
    )
    optional_fields = (
        'Unit of Measure',
    )

    for canonical in required_fields:
        aliases = NEW_SCHEMA_ALIASES.get(canonical, ())
        match = _find_match(aliases, used)
        if match is None:
            return None
        matches[match] = canonical
        used.add(match)

    for canonical in optional_fields:
        aliases = NEW_SCHEMA_ALIASES.get(canonical, ())
        match = _find_match(aliases, used)
        if match is not None:
            matches[match] = canonical
            used.add(match)

    return matches


def _classify_measurement(channel: str, unit: str) -> Optional[str]:
    """Return canonical measurement name based on channel/unit metadata."""

    channel = (channel or '').strip().lower()
    unit = (unit or '').strip().lower()

    sensor_match = re.search(r'sensor\s*0*(\d+)', channel)
    if sensor_match:
        sensor_id = sensor_match.group(1)
        if sensor_id == '1':
            return 'Humidity'
        if sensor_id == '2':
            return 'Temperature'

    # Normalise the unit text so we can reliably inspect the tokens regardless of
    # punctuation, case, or unicode symbols (e.g. "°F").
    unit_normalised = (
        unit.replace('°', ' ')
        .replace('degrees', 'deg')
        .replace('/', ' ')
        .replace('-', ' ')
        .replace('_', ' ')
    )
    unit_tokens = {token for token in unit_normalised.split() if token}

    humidity_tokens = {'%', 'percent', 'humidity', 'humid', 'rh'}
    if (
        any(token in channel for token in ('hum', '%'))
        or '%' in unit
        or unit_tokens & humidity_tokens
    ):
        return 'Humidity'

    temperature_tokens = {
        'temp',
        'temperature',
        'degc',
        'c',
        'celsius',
        'degf',
        'f',
        'fahrenheit',
        'degk',
        'k',
        'kelvin',
    }
    if (
        any(token in channel for token in ('temp', '°c', '°f'))
        or unit_tokens & temperature_tokens
    ):
        return 'Temperature'

    return None


def _find_column_by_terms(columns: Iterable[str], hints: Iterable[Tuple[str, ...]]) -> Optional[str]:
    """Return the first column that matches any group of hint terms."""

    for terms in hints:
        for col in columns:
            if not isinstance(col, str):
                continue
            lower = col.lower()
            if all(term in lower for term in terms):
                return col
    return None


def _clean_metadata_series(series: pd.Series) -> pd.Series:
    """Normalize free-text metadata columns used for grouping."""

    normalized = []
    for value in series.astype(str).tolist():
        text = value.strip()
        normalized.append("" if text.lower() in {"", "nan", "none", "null"} else text)
    return pd.Series(normalized, index=series.index)


def _process_new_schema_df(
    df_full: pd.DataFrame,
    name: str,
    default_temp_range: Tuple[float, float],
    default_humidity_range: Tuple[float, float],
) -> List[Dict[str, object]]:
    canonical_map = _match_new_schema_columns(df_full.columns)
    if not canonical_map:
        return []

    strip_map = {
        col: col.strip()
        for col in df_full.columns
        if isinstance(col, str)
    }
    df_new = df_full.rename(columns=strip_map)
    canonical = {
        strip_map.get(original, original): canonical_name
        for original, canonical_name in canonical_map.items()
    }
    df_new = df_new.rename(columns=canonical)

    for column in ["Timestamp", "Serial Number", "Channel", "Data", "Unit of Measure"]:
        if column not in df_new.columns:
            if column == "Unit of Measure":
                df_new[column] = pd.Series(["" for _ in range(len(df_new))], index=df_new.index, dtype="string")
            else:
                df_new[column] = pd.Series(pd.NA, index=df_new.index, dtype="string")
        df_new[column] = df_new[column].astype("string")

    df_new["Serial Number"] = df_new["Serial Number"].str.strip()
    df_new["Channel"] = df_new["Channel"].str.strip()
    df_new["Unit of Measure"] = df_new["Unit of Measure"].fillna("").str.strip()

    df_new["DateTime"] = df_new["Timestamp"].apply(_parse_ts)
    df_new = df_new.dropna(subset=["DateTime"])
    df_new = df_new[df_new["Serial Number"].astype(str).str.strip() != ""]

    cleaned_values = (
        df_new["Data"]
        .astype("string")
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace({"": pd.NA})
    )
    df_new["Value"] = pd.to_numeric(cleaned_values, errors="coerce")
    df_new = df_new.dropna(subset=["Value"])

    channel_original = df_new["Channel"].fillna("")
    unit_original = df_new["Unit of Measure"].fillna("")
    measurement_series = pd.Series(
        [_classify_measurement(ch, unit) for ch, unit in zip(channel_original, unit_original)],
        index=df_new.index,
        dtype="object",
    )
    sensor_map = {
        "sensor1": "Humidity",
        "sensor 1": "Humidity",
        "sensor01": "Humidity",
        "sensor 01": "Humidity",
        "sensor2": "Temperature",
        "sensor 2": "Temperature",
        "sensor02": "Temperature",
        "sensor 02": "Temperature",
    }
    direct_channel_map = {
        "humidity": "Humidity",
        "relative humidity": "Humidity",
        "rh": "Humidity",
        "temperature": "Temperature",
        "temp": "Temperature",
    }
    channel_lower = channel_original.astype(str).str.lower()
    df_new["Channel"] = channel_lower.map(sensor_map)
    df_new["Channel"] = df_new["Channel"].where(df_new["Channel"].notna(), measurement_series)
    df_new["Channel"] = df_new["Channel"].where(
        df_new["Channel"].notna(), channel_lower.map(direct_channel_map)
    )
    df_new["Channel"] = df_new["Channel"].where(df_new["Channel"].notna(), channel_original)
    df_new["Channel"] = df_new["Channel"].astype("string").str.strip()
    df_new = df_new[df_new["Channel"] != ""]
    if df_new.empty:
        return []

    space_name_col = _find_column_by_terms(df_new.columns, _SPACE_NAME_HINTS)
    space_type_col = _find_column_by_terms(df_new.columns, _SPACE_TYPE_HINTS)
    if space_name_col:
        df_new['_SpaceName'] = _clean_metadata_series(df_new[space_name_col])
    else:
        df_new['_SpaceName'] = ''
    if space_type_col:
        df_new['_SpaceType'] = _clean_metadata_series(df_new[space_type_col])
    else:
        df_new['_SpaceType'] = ''

    group_columns = ['Serial Number']
    if space_name_col:
        group_columns.append('_SpaceName')
    if space_type_col:
        group_columns.append('_SpaceType')

    results: List[Dict[str, object]] = []
    for keys, sdf in df_new.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_columns, keys))
        serial = str(key_map.get('Serial Number', '')).strip()
        space_name = str(key_map.get('_SpaceName', '')).strip()
        space_type = str(key_map.get('_SpaceType', '')).strip()
        sdf = sdf.copy()
        sdf = sdf.dropna(subset=['DateTime', 'Value'])
        if sdf.empty:
            continue
        sdf['Channel'] = sdf['Channel'].astype('string').str.strip()
        sdf = sdf[sdf['Channel'] != '']
        if sdf.empty:
            continue
        channel_lower = sdf['Channel'].str.lower()
        saw_sensor1 = channel_lower.isin({'sensor1', 'sensor 1', 'humidity'}).any()
        saw_sensor2 = channel_lower.isin({'sensor2', 'sensor 2', 'temperature'}).any()
        sensor_alias_map = {
            'sensor1': 'Humidity',
            'sensor 1': 'Humidity',
            'sensor2': 'Temperature',
            'sensor 2': 'Temperature',
        }
        sdf['Channel'] = channel_lower.map(sensor_alias_map).fillna(sdf['Channel'])
        sdf = sdf[sdf['Channel'].astype(str).str.strip() != '']
        if sdf.empty:
            continue
        sdf = sdf.sort_values('DateTime')
        pivot = sdf.pivot_table(
            index='DateTime',
            columns='Channel',
            values='Value',
            aggfunc='mean',
        )
        if pivot.empty:
            continue
        pivot = pivot.rename_axis(None, axis=1).sort_index()
        expected_channels = {
            ch
            for ch in sdf['Channel'].dropna().unique()
            if isinstance(ch, str)
        }
        if saw_sensor1:
            expected_channels.add('Humidity')
        if saw_sensor2:
            expected_channels.add('Temperature')
        for col in ('Temperature', 'Humidity'):
            if col in expected_channels and col not in pivot.columns:
                pivot[col] = pd.NA
        pivot['DateTime'] = pivot.index
        pivot['Date'] = pivot['DateTime'].dt.date
        pivot['Time'] = pivot['DateTime'].dt.strftime('%H:%M')
        ordered_cols = ['Date', 'Time', 'DateTime']
        data_columns = [
            col for col in pivot.columns if col not in {'Date', 'Time', 'DateTime'}
        ]
        ordered_cols.extend(data_columns)
        pivot = pivot[ordered_cols].reset_index(drop=True)
        pivot = _finalize_serial_dataframe(pivot)
        if pivot.empty:
            continue
        display_name = ''
        if space_name and space_type:
            display_name = f"{space_name} ({space_type})"
        elif space_name:
            display_name = space_name
        elif space_type:
            display_name = space_type

        label_prefix = name if not display_name else f"{name} - {display_name}"
        label = f"{label_prefix} [{serial}]" if serial else label_prefix

        context = _normalise_context_text(name, display_name, space_name, space_type)
        inferred_temp_range = _infer_temperature_range(context, default_temp_range)

        range_map: Dict[str, Tuple[float, float]] = {}
        channels: List[str] = []
        if 'Temperature' in pivot.columns:
            range_map['Temperature'] = inferred_temp_range
            channels.append('Temperature')
        if 'Humidity' in pivot.columns:
            range_map['Humidity'] = default_humidity_range
            if pivot['Humidity'].notna().any():
                channels.append('Humidity')

        range_map.setdefault('Humidity', DEFAULT_HUMIDITY_RANGE)
        _apply_inferred_ranges(
            range_map, name, display_name, space_name, space_type, serial
        )

        default_label = display_name or serial or Path(name).stem
        option_label = ''
        if serial and display_name:
            option_label = f"{serial} – {display_name}"
        elif serial:
            option_label = serial
        elif display_name:
            option_label = display_name
        else:
            option_label = label

        results.append(
            {
                'key': label,
                'df': pivot,
                'range_map': range_map,
                'serial': serial,
                'display_name': display_name,
                'default_label': default_label,
                'option_label': option_label,
                'source_name': name,
                'channels': channels,
            }
        )

    return results


def _parse_probe_files(files):
    dfs = {}
    ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    if not files:
        return dfs, ranges
    default_temp_range = DEFAULT_TEMP_RANGE
    default_humidity_range = DEFAULT_HUMIDITY_RANGE
    for f in files:
        f.seek(0)
        name = f.name
        raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
        full_content = ''.join(raw)
        df_full = _read_csv_flexible(full_content)
        if df_full is not None:
            new_schema_groups = _process_new_schema_df(
                df_full, name, default_temp_range, default_humidity_range
            )
            if new_schema_groups:
                for group in new_schema_groups:
                    key = group['key']  # type: ignore[index]
                    dfs[key] = group['df']  # type: ignore[index]
                    range_map = group['range_map']  # type: ignore[index]
                    if range_map:
                        ranges[key] = range_map
                continue
        try:
            idx = max(i for i, line in enumerate(raw) if 'ch1' in line.lower() or 'p1' in line.lower())
        except ValueError:
            continue
        content = ''.join(raw[idx:])
        df = _read_csv_flexible(content)
        if df is None:
            continue
        lname = name.lower()
        has_fridge = any(term in lname for term in ('fridge', 'refrigerator'))
        has_freezer = 'freezer' in lname
        if has_fridge and has_freezer:
            mapping = {'Fridge Temp': 'P1', 'Freezer Temp': 'P2'}
            ranges[name] = {'Fridge Temp': (2, 8), 'Freezer Temp': (-35, -5)}
        elif has_fridge:
            mapping = {'Temperature': 'P1'}
            ranges[name] = {'Temperature': (2, 8)}
        elif has_freezer:
            mapping = {'Temperature': 'P1'}
            ranges[name] = {'Temperature': (-35, -5)}
        else:
            cols_lower = [c.lower() for c in df.columns]
            if any('ch4' in c for c in cols_lower):
                mapping = {'Humidity': 'CH3', 'Temperature': 'CH4'}
            else:
                mapping = {'Humidity': 'CH1', 'Temperature': 'CH2'}
            is_olympus = 'olympus' in lname
            temp_range = (15, 28) if is_olympus else default_temp_range
            humidity_range = (0, 80) if is_olympus else default_humidity_range
            ranges[name] = {'Temperature': temp_range, 'Humidity': humidity_range}
        mapping.update({'Date': 'Date', 'Time': 'Time'})
        col_map = {}
        for new, key in mapping.items():
            col = next((c for c in df.columns if key.lower() in c.lower()), None)
            if col:
                col_map[col] = new
        df = df[list(col_map)].rename(columns=col_map)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['DateTime'] = pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d ') + df['Time'].astype(str),
            errors='coerce'
        )
        for c in df.columns.difference(['Date', 'Time', 'DateTime']):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('+',''), errors='coerce')
        dfs[name] = df.reset_index(drop=True)
    return dfs, ranges



def parse_serial_csv(files):
    serials: Dict[str, Dict[str, object]] = {}
    if not files:
        return serials

    merged_frames: Dict[str, pd.DataFrame] = {}

    for f in files:
        try:
            f.seek(0)
        except Exception:
            pass

        raw_content = f.read()
        if isinstance(raw_content, bytes):
            full_text = raw_content.decode('utf-8', errors='ignore')
        else:
            full_text = str(raw_content)

        source_name = getattr(f, 'name', 'serial.csv') or 'serial.csv'

        parsed_items: List[Dict[str, object]] = []
        if _looks_like_traceable_report(full_text):
            parsed_items = _parse_traceable_report_csv(full_text, source_name)
        elif _looks_like_consolidated(full_text):
            parsed_items = _parse_consolidated_serial_csv(full_text, source_name)
        else:
            parsed_items = _parse_consolidated_serial_csv(full_text, source_name)
            if not parsed_items:
                parsed_items = _parse_traceable_report_csv(full_text, source_name)

        if not parsed_items:
            df_full = _read_csv_flexible(full_text)
            if df_full is not None:
                fallback_groups = _process_new_schema_df(
                    df_full,
                    source_name,
                    DEFAULT_TEMP_RANGE,
                    DEFAULT_HUMIDITY_RANGE,
                )
                parsed_items.extend(
                    {
                        'df': group['df'],
                        'serial': group.get('serial'),
                        'range_map': group.get('range_map', {}),
                        'default_label': group.get('default_label'),
                        'option_label': group.get('option_label'),
                        'source_name': group.get('source_name', source_name),
                        'channels': group.get('channels', []),
                    }
                    for group in fallback_groups
                )

        for item in parsed_items:
            df = item.get('df')
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            serial_value = item.get('serial')
            if serial_value is None or str(serial_value).strip() == '':
                serial_value = item.get('default_label') or Path(source_name).stem
            serial_key = str(serial_value)

            merged_df = merge_serial_data(merged_frames, df, serial_key)

            existing = serials.get(serial_key)
            range_map: Dict[str, Tuple[float, float]] = {}
            channels: List[str] = []
            source_parts: List[str] = []

            if existing:
                if isinstance(existing.get('range_map'), dict):
                    range_map.update(existing['range_map'])  # type: ignore[arg-type]
                channels.extend(
                    [ch for ch in existing.get('channels', []) if isinstance(ch, str)]
                )
                existing_source = existing.get('source_name')
                if isinstance(existing_source, str):
                    source_parts.extend(
                        [part.strip() for part in existing_source.split(',') if part.strip()]
                    )
            if isinstance(item.get('range_map'), dict):
                range_map.update(item['range_map'])  # type: ignore[arg-type]
            channels.extend([ch for ch in item.get('channels', []) if isinstance(ch, str)])
            channels = list(dict.fromkeys(channels))

            item_source = item.get('source_name')
            if isinstance(item_source, str):
                source_parts.extend(
                    [part.strip() for part in item_source.split(',') if part.strip()]
                )
            source_name_combined = ', '.join(dict.fromkeys(source_parts)) if source_parts else item.get('source_name') or source_name

            _apply_inferred_ranges(
                range_map,
                source_name_combined,
                str(serial_value) if serial_value is not None else None,
                item.get('default_label'),
            )

            default_label = (
                (existing.get('default_label') if existing else None)
                or item.get('default_label')
                or serial_key
            )
            option_label = (
                (existing.get('option_label') if existing else None)
                or item.get('option_label')
                or default_label
            )

            serials[serial_key] = {
                'df': merged_df,
                'range_map': range_map,
                'serial': str(serial_value),
                'default_label': default_label,
                'option_label': option_label,
                'source_name': source_name_combined,
                'channels': channels,
            }

    return serials


def serial_data_to_primary(serials: Mapping[str, Dict[str, object]]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Tuple[float, float]]]]:
    """Convert parsed serial datasets into primary probe-style structures.

    Parameters
    ----------
    serials:
        Mapping of serial dataset keys to the metadata produced by
        :func:`parse_serial_csv`.

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Tuple[float, float]]]]
        Two dictionaries mirroring the output of :func:`_parse_probe_files`.
        The first maps dataset keys to processed dataframes suitable for the
        primary probe workflow, ensuring the ``Date`` column uses a
        ``datetime64`` dtype so downstream ``.dt`` accessors are valid.  The
        second maps dataset keys to their channel range configuration.
    """

    primary_dfs: Dict[str, pd.DataFrame] = {}
    primary_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for key, info in serials.items():
        df = info.get('df')
        if df is None:
            continue
        df_converted = df.copy()
        if 'Date' in df_converted.columns:
            df_converted['Date'] = pd.to_datetime(df_converted['Date'], errors='coerce')
        primary_dfs[key] = df_converted
        range_map = info.get('range_map')
        if isinstance(range_map, dict):
            primary_ranges[key] = range_map
        else:
            primary_ranges[key] = {}

    return primary_dfs, primary_ranges


def _parse_tempstick_files(files):
    tempdfs = {}
    if not files:
        return tempdfs
    for f in files:
        f.seek(0)
        raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
        try:
            idx = max(i for i, line in enumerate(raw) if 'timestamp' in line.lower())
        except ValueError:
            continue
        csv_text = ''.join(raw[idx:])
        df_ts = _read_csv_flexible(csv_text)
        if df_ts is None:
            continue
        ts_col = next((c for c in df_ts.columns if 'timestamp' in c.lower()), None)
        if not ts_col:
            continue
        df_ts = df_ts.rename(columns={ts_col: 'DateTime'})
        df_ts['DateTime'] = pd.to_datetime(
            df_ts['DateTime'], errors='coerce'
        )
        temp_col = next((c for c in df_ts.columns if 'temp' in c.lower()), None)
        if temp_col:
            df_ts['Temperature'] = pd.to_numeric(
                df_ts[temp_col].astype(str).str.replace('+', ''), errors='coerce'
            )
        hum_col = next((c for c in df_ts.columns if 'hum' in c.lower()), None)
        if hum_col:
            df_ts['Humidity'] = pd.to_numeric(
                df_ts[hum_col].astype(str).str.replace('+', ''), errors='coerce'
            )
        cols = ['DateTime']
        if 'Temperature' in df_ts.columns:
            cols.append('Temperature')
        if 'Humidity' in df_ts.columns:
            cols.append('Humidity')
        tempdfs[f.name] = df_ts[cols]
    return tempdfs


def merge_serial_data(existing: dict, new_df, serial: str):
    """
    Merge/create a combined DataFrame for a serial.
    If multiple rows share the same DateTime (e.g., one has Temperature and the other has Humidity),
    coalesce columns so we KEEP both channel values instead of dropping one row.
    """
    import pandas as pd

    def _coalesce_series(series: pd.Series):
        valid = series.dropna()
        if valid.empty:
            return pd.NA
        return valid.iloc[-1]

    if serial in existing and isinstance(existing[serial], pd.DataFrame):
        merged = pd.concat([existing[serial], new_df], ignore_index=True)
    else:
        merged = new_df.copy()

    if "DateTime" in merged.columns:
        merged["DateTime"] = pd.to_datetime(merged["DateTime"], errors="coerce")
        merged = merged.dropna(subset=["DateTime"])
        merged = merged.sort_values("DateTime")

        value_cols = [col for col in merged.columns if col != "DateTime"]
        if value_cols:
            agg_map = {col: _coalesce_series for col in value_cols}
            merged = merged.groupby("DateTime", as_index=False, sort=True).agg(agg_map)
        else:
            merged = merged.drop_duplicates(subset=["DateTime"], keep="last")

        if "Date" not in merged.columns:
            merged["Date"] = pd.to_datetime(merged["DateTime"]).dt.date
        if "Time" not in merged.columns:
            merged["Time"] = pd.to_datetime(merged["DateTime"]).dt.strftime("%H:%M")

    result = merged.reset_index(drop=True)
    preferred_order = ["DateTime", "Temperature", "Humidity", "Date", "Time"]
    ordered_cols = [col for col in preferred_order if col in result.columns]
    other_cols = [col for col in result.columns if col not in ordered_cols]
    if ordered_cols and (len(ordered_cols) != len(result.columns)):
        result = result[ordered_cols + other_cols]
    elif ordered_cols:
        result = result[ordered_cols]
    existing[serial] = result
    return result
