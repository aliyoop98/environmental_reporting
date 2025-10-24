import io
import re
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd


def _read_any_csv(file_obj) -> pd.DataFrame:
    """Return a dataframe from a CSV or TSV stream, handling messy inputs."""

    raw = file_obj.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        text = str(raw)

    for sep in (None, ",", "\t"):
        try:
            df = pd.read_csv(
                io.StringIO(text),
                engine="python",
                sep=sep,
                on_bad_lines="skip",
                dtype=str,
                keep_default_na=False,
                index_col=False,
            )
            df.columns = [re.sub(r"\s+", " ", (col or "")).strip() for col in df.columns]
            return df
        except Exception:
            continue
    return pd.DataFrame()


def _read_csv_flexible(text: str) -> Optional[pd.DataFrame]:
    """Backward-compatible CSV reader used by legacy probe parsing code."""

    cleaned = _strip_bom_and_zero_width(text)
    for sep in (None, ",", "\t"):
        try:
            df = pd.read_csv(
                io.StringIO(cleaned),
                engine="python",
                sep=sep,
                on_bad_lines="skip",
                dtype=str,
                index_col=False,
            )
        except Exception:
            continue

        return _clean_chunk_columns(df)
    return None


def _clean_chunk_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        if isinstance(col, str):
            rename_map[col] = _strip_bom_and_zero_width(col).strip()
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _parse_ts_safe(s: str):
    s = (s or "").strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%b-%d %H:%M",
        "%d-%b-%Y %H:%M",
    ):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            continue
    return pd.to_datetime(s, errors="coerce")

_ZERO_WIDTH_CHARS = ("\ufeff", "\u200b", "\u200c", "\u200d")

DEFAULT_TEMP_RANGE: Tuple[float, float] = (15, 25)
DEFAULT_HUMIDITY_RANGE: Tuple[float, float] = (0, 60)

NEW_SCHEMA_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Timestamp": ("timestamp", "time stamp", "date time"),
    "Serial Number": (
        "serial number",
        "serial num",
        "serial",
        "serial no",
        "serial #",
        "serial#",
    ),
    "Channel": (
        "channel",
        "sensor",
        "channel name",
        "measurement",
    ),
    "Data": ("data", "value", "reading"),
    "Unit of Measure": ("unit of measure", "unit", "units"),
}

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


def _extract_report_blocks(text: str) -> Optional[Tuple[Dict[str, Optional[str]], List[pd.DataFrame]]]:
    """Return metadata and all timestamp blocks from a Traceable report text."""

    if not text:
        return None

    lines = [line.replace("\ufeff", "") for line in text.splitlines()]
    if not any("device" in line.lower() and "id" in line.lower() for line in lines):
        return None

    device_id: Optional[str] = None
    device_name: Optional[str] = None
    for line in lines[:50]:
        match = _DEVICE_ID_RE.search(line)
        if match:
            device_id = match.group(1)
        lower = line.lower()
        if lower.startswith("device name:"):
            device_name = line.split(":", 1)[1].strip()

    header_idxs = [
        idx
        for idx, line in enumerate(lines)
        if line.strip().lower().startswith("timestamp")
    ]
    if not header_idxs:
        return None

    blocks: List[pd.DataFrame] = []
    for start in header_idxs:
        end = len(lines)
        for pos in range(start + 1, len(lines)):
            lowered = lines[pos].strip().lower()
            if lowered.startswith("channel data for") or pos in header_idxs:
                end = pos
                break

        rows = [lines[start]]
        for line in lines[start + 1 : end]:
            if line.strip():
                rows.append(line)

        if len(rows) <= 1:
            continue

        try:
            frame = pd.read_csv(
                io.StringIO("\n".join(rows)),
                engine="python",
                on_bad_lines="skip",
                dtype=str,
            )
        except Exception:
            continue
        frame.columns = [re.sub(r"\s+", " ", (col or "")).strip() for col in frame.columns]
        blocks.append(frame)

    if not blocks:
        return None

    meta = {"device_id": device_id, "device_name": device_name}
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




def _parse_consolidated_serial_df(df: pd.DataFrame, source_name: str) -> List[Dict[str, object]]:
    """Return consolidated Traceable serial data grouped by serial identifier."""

    if df is None or df.empty:
        return []

    normalized = {
        re.sub(r"\s+", " ", (str(col) or "")).strip().lower(): col
        for col in df.columns
    }

    def _find_column(*aliases: str) -> Optional[str]:
        for alias in aliases:
            key = re.sub(r"\s+", " ", alias).strip().lower()
            if key in normalized:
                return normalized[key]
        return None

    timestamp_col = _find_column("timestamp", "time stamp", "date time")
    serial_col = _find_column("serial number", "serial num", "serial", "device id")
    channel_col = _find_column("channel", "sensor", "channel name")
    data_col = _find_column("data", "value", "reading")
    unit_col = _find_column("unit of measure", "unit", "units")
    space_name_col = _find_column("space name", "space", "location name")
    space_type_col = _find_column("space type", "location type")

    if not all([timestamp_col, serial_col, channel_col, data_col]):
        return []

    rename_map = {
        timestamp_col: "Timestamp",
        serial_col: "Serial",
        channel_col: "Channel",
        data_col: "Data",
    }
    if unit_col:
        rename_map[unit_col] = "Unit"
    if space_name_col:
        rename_map[space_name_col] = "Space Name"
    if space_type_col:
        rename_map[space_type_col] = "Space Type"
    df = df.rename(columns=rename_map)

    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["Channel"] = df["Channel"].astype(str).str.strip()
    df["Data"] = df["Data"].astype(str).str.strip()
    if "Unit" in df.columns:
        df["Unit"] = df["Unit"].astype(str).str.strip()
    else:
        df["Unit"] = ""
    if "Space Name" in df.columns:
        df["Space Name"] = df["Space Name"].astype(str).str.strip()
    if "Space Type" in df.columns:
        df["Space Type"] = df["Space Type"].astype(str).str.strip()

    df = df[df["Serial"] != ""]
    if df.empty:
        return []

    df["DateTime"] = df["Timestamp"].apply(_parse_ts_safe)
    df = df.dropna(subset=["DateTime"])
    if df.empty:
        return []

    df["Channel"] = (
        df["Channel"]
        .str.lower()
        .replace(
            {
                "sensor1": "Humidity",
                "sensor 1": "Humidity",
                "sensor2": "Temperature",
                "sensor 2": "Temperature",
            }
        )
    )
    df["Channel"] = df["Channel"].replace({"humidity": "Humidity", "temperature": "Temperature"})

    df["Value"] = pd.to_numeric(
        df["Data"].str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce"
    )
    df = df.dropna(subset=["Value"])
    if df.empty:
        return []

    items: List[Dict[str, object]] = []
    for serial, group in df.groupby(df["Serial"].astype(str).str.strip()):
        if not isinstance(serial, str):
            serial = str(serial)
        sub = group.copy()
        pivot = (
            sub.pivot_table(
                index="DateTime",
                columns="Channel",
                values="Value",
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(None, axis=1)
            .sort_values("DateTime")
        )
        if pivot.empty:
            continue

        for col in ("Temperature", "Humidity"):
            if col not in pivot.columns:
                pivot[col] = pd.NA

        pivot["Date"] = pd.to_datetime(pivot["DateTime"]).dt.date
        pivot["Time"] = pd.to_datetime(pivot["DateTime"]).dt.strftime("%H:%M")

        ordered = ["DateTime", "Temperature", "Humidity", "Date", "Time"]
        pivot = pivot[[c for c in ordered if c in pivot.columns]]

        space_name_value = ""
        if "Space Name" in sub.columns:
            space_name_value = next(
                (val for val in sub["Space Name"].astype(str).str.strip() if val),
                "",
            )
        space_type_value = ""
        if "Space Type" in sub.columns:
            space_type_value = next(
                (val for val in sub["Space Type"].astype(str).str.strip() if val),
                "",
            )

        range_map: Dict[str, Tuple[float, float]] = {}
        _apply_inferred_ranges(
            range_map,
            source_name,
            serial,
            space_name_value,
            space_type_value,
        )

        default_label = space_name_value or str(serial)
        option_label = (
            f"{serial} – {space_name_value}"
            if space_name_value and space_name_value != str(serial)
            else default_label
        )

        items.append(
            {
                "df": pivot.reset_index(drop=True),
                "serial": str(serial),
                "range_map": range_map,
                "option_label": option_label,
                "default_label": default_label,
                "source_name": source_name,
            }
        )

    return items


def _parse_traceable_report_text(text: str, source_name: str) -> List[Dict[str, object]]:
    res = _extract_report_blocks(text)
    if not res:
        return []

    meta, frames = res
    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)
    normalized = {
        re.sub(r"\s+", " ", (col or "")).strip().lower(): col
        for col in df.columns
    }

    def _find_column(*aliases: str) -> Optional[str]:
        for alias in aliases:
            key = re.sub(r"\s+", " ", alias).strip().lower()
            if key in normalized:
                return normalized[key]
        return None

    timestamp_col = _find_column("timestamp", "time stamp", "date time")
    data_col = _find_column("data", "value", "reading")
    unit_col = _find_column("unit", "unit of measure", "units")
    if not timestamp_col or not data_col:
        return []

    df = df.rename(columns={timestamp_col: "Timestamp", data_col: "Data"})
    if unit_col:
        df = df.rename(columns={unit_col: "Unit"})
    else:
        df["Unit"] = ""

    df["DateTime"] = df["Timestamp"].apply(_parse_ts_safe)
    df = df.dropna(subset=["DateTime"])
    if df.empty:
        return []

    df["Value"] = pd.to_numeric(
        df["Data"].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce"
    )
    df = df.dropna(subset=["Value"])
    if df.empty:
        return []

    if "Unit" in df.columns:
        unit_series = df["Unit"].astype(str).str.strip()
        df["Kind"] = unit_series.map(lambda u: "Humidity" if "%" in u else "Temperature")
    else:
        df["Kind"] = "Temperature"

    data_text = df["Data"].astype(str)
    df.loc[
        df["Kind"] == "Temperature",
        "Kind",
    ] = df.loc[df["Kind"] == "Temperature", "Kind"].where(
        ~data_text.str.contains("%"), "Humidity"
    )

    pivot = (
        df.pivot_table(
            index="DateTime",
            columns="Kind",
            values="Value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values("DateTime")
    )

    for col in ("Temperature", "Humidity"):
        if col not in pivot.columns:
            pivot[col] = pd.NA

    pivot["Date"] = pd.to_datetime(pivot["DateTime"]).dt.date
    pivot["Time"] = pd.to_datetime(pivot["DateTime"]).dt.strftime("%H:%M")
    ordered = ["DateTime", "Temperature", "Humidity", "Date", "Time"]
    pivot = pivot[[c for c in ordered if c in pivot.columns]]

    serial = meta.get("device_id") or meta.get("device_name") or Path(source_name).stem
    range_map: Dict[str, Tuple[float, float]] = {}
    _apply_inferred_ranges(range_map, source_name, serial, meta.get("device_name"))

    return [
        {
            "df": pivot.reset_index(drop=True),
            "serial": str(serial),
            "range_map": range_map,
            "option_label": str(serial),
            "default_label": str(serial),
            "source_name": source_name,
        }
    ]


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


_SPACE_NAME_HINTS: Tuple[Tuple[str, ...], ...] = (
    ("space", "name"),
    ("assignment", "name"),
    ("location", "name"),
)

_SPACE_TYPE_HINTS: Tuple[Tuple[str, ...], ...] = (
    ("space", "type"),
    ("assignment", "type"),
    ("location", "type"),
)


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
    if not files:
        return {}

    frames: Dict[str, pd.DataFrame] = {}
    metadata: Dict[str, Dict[str, object]] = {}

    for file_obj in files:
        source_name = getattr(file_obj, "name", "serial.csv") or "serial.csv"
        getter = getattr(file_obj, "getvalue", None)
        raw = getter() if callable(getter) else file_obj.read()
        if hasattr(file_obj, "seek"):
            try:
                file_obj.seek(0)
            except Exception:
                pass

        text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else str(raw)

        df = _read_any_csv(io.StringIO(text))
        items: List[Dict[str, object]] = []
        if not df.empty:
            items = _parse_consolidated_serial_df(df, source_name)

        if not items and "device" in text.lower() and "id" in text.lower():
            items = _parse_traceable_report_text(text, source_name)

        if not items:
            continue

        for item in items:
            df_item = item.get("df")
            if not isinstance(df_item, pd.DataFrame) or df_item.empty:
                continue

            serial_value = str(item.get("serial") or source_name).strip() or source_name
            merged_df = merge_serial_data(frames, df_item, serial_value)

            meta = metadata.setdefault(
                serial_value,
                {
                    "range_map": {},
                    "option_label": None,
                    "default_label": None,
                    "source_name": "",
                    "serial": serial_value,
                },
            )

            incoming_range = item.get("range_map")
            if isinstance(incoming_range, dict):
                meta["range_map"].update(incoming_range)

            for key in ("option_label", "default_label"):
                value = item.get(key)
                if value:
                    meta[key] = value

            src_parts: List[str] = []
            existing_source = meta.get("source_name")
            if isinstance(existing_source, str) and existing_source:
                src_parts.extend(part.strip() for part in existing_source.split(",") if part.strip())

            new_source = item.get("source_name") or source_name
            if isinstance(new_source, str):
                src_parts.extend(part.strip() for part in new_source.split(",") if part.strip())

            if src_parts:
                meta["source_name"] = ", ".join(dict.fromkeys(src_parts))
            else:
                meta["source_name"] = new_source

            channels = [col for col in ["Temperature", "Humidity"] if col in merged_df.columns]
            meta["channels"] = channels

            range_map = meta.get("range_map") if isinstance(meta.get("range_map"), dict) else {}
            if isinstance(range_map, dict):
                _apply_inferred_ranges(
                    range_map,
                    meta.get("source_name"),
                    serial_value,
                    meta.get("default_label"),
                )

    output: Dict[str, Dict[str, object]] = {}
    for serial_value, df in frames.items():
        meta = metadata.get(serial_value, {})
        range_map = meta.get("range_map") if isinstance(meta.get("range_map"), dict) else {}
        if isinstance(range_map, dict):
            range_map = dict(range_map)
        else:
            range_map = {}

        if not range_map:
            _apply_inferred_ranges(range_map, meta.get("source_name"), serial_value, meta.get("default_label"))

        channels = meta.get("channels") if isinstance(meta.get("channels"), list) else []
        if not channels:
            channels = [col for col in ["Temperature", "Humidity"] if col in df.columns]

        output[serial_value] = {
            "df": df,
            "range_map": range_map,
            "serial": serial_value,
            "default_label": meta.get("default_label") or serial_value,
            "option_label": meta.get("option_label") or meta.get("default_label") or serial_value,
            "source_name": meta.get("source_name") or serial_value,
            "channels": channels,
        }

    return output


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

    base = existing.get(serial)
    if not isinstance(new_df, pd.DataFrame) or new_df.empty:
        if isinstance(base, pd.DataFrame):
            return base
        existing[serial] = pd.DataFrame()
        return existing[serial]

    if isinstance(base, pd.DataFrame) and not base.empty:
        merged = pd.concat([base, new_df], ignore_index=True)
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
