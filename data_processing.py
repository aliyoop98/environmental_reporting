from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd

CANONICAL_CHANNELS = ("Temperature", "Humidity")
DEFAULT_TEMP_RANGE = (15.0, 25.0)
DEFAULT_HUMIDITY_RANGE = (0.0, 60.0)

HEADER_ALIASES = {
    "timestamp": {"timestamp", "time stamp", "date time", "datetime", "date/time"},
    "serial": {"serial number", "serial num", "serial", "serial no", "serial #", "device id", "device identifier"},
    "channel": {"channel", "sensor", "channel name", "sensor name", "measurement"},
    "data": {"data", "value", "reading", "measurement value"},
    "unit": {"unit of measure", "unit of measurement", "units", "unit", "uom", "engineering units"},
    "space_name": {"space name", "space", "location name", "assignment name"},
    "space_type": {"space type", "location type", "assignment type"},
}

PROFILE_LIMITS = {
    "Room": {"Temperature": (15.0, 25.0), "Humidity": (0.0, 60.0)},
    "Olympus": {"Temperature": (15.0, 28.0), "Humidity": (0.0, 80.0)},
    "Olympus Room": {"Temperature": (15.0, 28.0), "Humidity": (0.0, 80.0)},
    "Fridge": {"Temperature": (2.0, 8.0)},
    "Freezer": {"Temperature": (-35.0, -5.0)},
    "Freezer -80": {"Temperature": (-86.0, -70.0)},
}


@dataclass
class SerialDataset:
    serial: str
    df: pd.DataFrame
    sources: list[str] = field(default_factory=list)
    space_name: str = ""
    space_type: str = ""
    channel_map: Dict[str, str] = field(default_factory=dict)
    range_map: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> list[str]:
        return [
            ch
            for ch in CANONICAL_CHANNELS
            if ch in self.df.columns and pd.to_numeric(self.df[ch], errors="coerce").notna().any()
        ]

    @property
    def label(self) -> str:
        descriptor = self.space_name or self.space_type
        return f"{self.serial} - {descriptor}" if descriptor else self.serial

    def as_dict(self) -> Dict[str, object]:
        return {
            "serial": self.serial,
            "df": self.df,
            "source_name": ", ".join(dict.fromkeys(self.sources)),
            "space_name": self.space_name,
            "space_type": self.space_type,
            "channel_map": dict(self.channel_map),
            "range_map": dict(self.range_map),
            "channels": self.channels,
            "validation": dict(self.validation),
            "option_label": self.label,
            "default_label": self.label,
        }


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("Â°", "°")
    return re.sub(r"\s+", " ", text).strip()


def _header_key(value: object) -> str:
    text = _clean_text(value).lower().replace("_", " ").replace("-", " ")
    text = text.replace("#", " number ")
    return re.sub(r"\s+", " ", text).strip()


def _channel_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_text(value).lower())


def _find_column(columns: Iterable[object], canonical: str) -> Optional[str]:
    aliases = HEADER_ALIASES[canonical]
    for col in columns:
        if _header_key(col) in aliases:
            return str(col)
    return None


def _sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        counts = {sep: sample.count(sep) for sep in (",", "\t", ";")}
        return max(counts, key=counts.get) if max(counts.values(), default=0) else ","


def _read_table(raw: bytes | str) -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else str(raw)
    text = text.replace("\x00", "")
    if not text.strip():
        return pd.DataFrame()
    sep = _sniff_delimiter(text)
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
            skipinitialspace=True,
        )
    except Exception:
        return pd.DataFrame()
    df.columns = [_clean_text(col) for col in df.columns]
    return df


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%b-%d %H:%M:%S",
        "%Y-%b-%d %H:%M",
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y %H:%M",
    )
    for fmt in formats:
        mask = result.isna() & text.notna() & text.ne("")
        if not mask.any():
            break
        parsed = pd.to_datetime(text.loc[mask], format=fmt, errors="coerce")
        result.loc[parsed.index] = parsed
    mask = result.isna() & text.notna() & text.ne("")
    if mask.any():
        try:
            result.loc[mask] = pd.to_datetime(text.loc[mask], format="mixed", errors="coerce")
        except (TypeError, ValueError):
            result.loc[mask] = pd.to_datetime(text.loc[mask], errors="coerce")
    return result


def _numeric_series(series: pd.Series) -> pd.Series:
    extracted = (
        series.astype("string")
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(extracted, errors="coerce")


def _unit_hint(unit: object, data: object = "") -> Optional[str]:
    text = f"{_clean_text(unit)} {_clean_text(data)}".lower().replace(" ", "")
    if any(token in text for token in ("%", "%rh", "humidity", "relativehumidity", "percent", "rh")):
        return "Humidity"
    if any(token in text for token in ("°c", "degc", "celsius", "°f", "degf", "fahrenheit", "kelvin", "degk")):
        return "Temperature"
    if _clean_text(unit).lower() in {"c", "f", "k"}:
        return "Temperature"
    return None


def _channel_text_hint(channel: object) -> Optional[str]:
    text = _clean_text(channel).lower()
    if any(token in text for token in ("humidity", "humid", "relative humidity", " rh", "rh ", "%")):
        return "Humidity"
    if any(token in text for token in ("temperature", "temp", "°c", "°f")):
        return "Temperature"
    return None


def _fallback_sensor_hint(channel: object) -> Optional[str]:
    key = _channel_key(channel)
    if key in {"sensor1", "sensor01", "ch1", "channel1"}:
        return "Humidity"
    if key in {"sensor2", "sensor02", "ch2", "channel2"}:
        return "Temperature"
    return None


def load_serial_overrides(path: str | Path) -> Dict[str, Dict[str, str]]:
    """Load editable serial/channel overrides from a repository CSV."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return {}
    normalized = {_header_key(c): c for c in df.columns}
    serial_col = normalized.get("serial number") or normalized.get("serial")
    channel_col = normalized.get("channel") or normalized.get("sensor")
    measurement_col = normalized.get("measurement") or normalized.get("kind")
    if not serial_col or not channel_col or not measurement_col:
        return {}
    result: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        serial = _clean_text(row.get(serial_col))
        channel = _channel_key(row.get(channel_col))
        measurement = _clean_text(row.get(measurement_col)).title()
        if serial and channel and measurement in CANONICAL_CHANNELS:
            result.setdefault(serial, {})[channel] = measurement
    return result


def _strong_measurement_evidence(group: pd.DataFrame) -> Optional[str]:
    """Return a strong Temperature/Humidity decision without row-wise Python loops.

    Traceable exports normally identify a measurement through ``Unit``.  When
    the unit column is blank, some exports append the unit to the ``Data``
    value instead.  We inspect both columns vectorially and use the dominant
    signal within one physical channel.
    """

    unit = (
        group.get("Unit", pd.Series("", index=group.index))
        .astype("string")
        .fillna("")
        .str.replace("Â°", "°", regex=False)
        .str.strip()
        .str.lower()
    )
    compact_unit = unit.str.replace(r"\s+", "", regex=True)
    humidity_mask = compact_unit.str.contains(r"%|humidity|humid|percent|rh", regex=True, na=False)
    temperature_mask = (
        compact_unit.str.contains(
            r"°c|degc|celsius|°f|degf|fahrenheit|kelvin|degk", regex=True, na=False
        )
        | compact_unit.isin({"c", "f", "k"})
    )

    humidity_count = int(humidity_mask.sum())
    temperature_count = int(temperature_mask.sum())

    # Unit labels are authoritative when they provide a clear majority.
    if humidity_count > temperature_count:
        return "Humidity"
    if temperature_count > humidity_count:
        return "Temperature"

    # If the unit column was silent or tied, inspect unit tokens embedded in Data.
    data = (
        group.get("Data", pd.Series("", index=group.index))
        .astype("string")
        .fillna("")
        .str.replace("Â°", "°", regex=False)
        .str.lower()
    )
    humidity_data = data.str.contains(r"%|%\s*rh|humidity|relative\s*humidity", regex=True, na=False)
    temperature_data = data.str.contains(
        r"°\s*[cf]|deg\s*[cfk]|celsius|fahrenheit|kelvin", regex=True, na=False
    )
    humidity_count = int(humidity_data.sum())
    temperature_count = int(temperature_data.sum())
    if humidity_count > temperature_count:
        return "Humidity"
    if temperature_count > humidity_count:
        return "Temperature"
    return None


def _build_channel_map(
    serial_df: pd.DataFrame,
    serial: str,
    overrides: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Dict[str, str]:
    """Resolve each physical channel to Temperature/Humidity once per serial.

    Evidence priority is deliberately deterministic:

    1. clear unit/data-unit evidence,
    2. repository serial override,
    3. explicit channel words,
    4. sensor-number fallback.

    Exactly two physical channels receive an additional complement check so a
    weak fallback cannot silently fold both channels into the same measurement.
    """

    channel_rows: Dict[str, pd.DataFrame] = {
        str(key): grp for key, grp in serial_df.groupby("ChannelKey", sort=False) if key
    }
    decisions: Dict[str, Tuple[str, int]] = {}
    serial_override = (overrides or {}).get(str(serial), {})

    for key, group in channel_rows.items():
        evidence: list[Tuple[str, int]] = []

        strong = _strong_measurement_evidence(group)
        if strong:
            evidence.append((strong, 120))

        if key in serial_override:
            kind = _clean_text(serial_override[key]).title()
            if kind in CANONICAL_CHANNELS:
                evidence.append((kind, 110))

        channel_text = _clean_text(group["Channel"].iloc[0])
        explicit = _channel_text_hint(channel_text)
        if explicit:
            evidence.append((explicit, 80))

        fallback = _fallback_sensor_hint(channel_text)
        if fallback:
            evidence.append((fallback, 20))

        if evidence:
            best_priority = max(priority for _, priority in evidence)
            top = [kind for kind, priority in evidence if priority == best_priority]
            decisions[key] = (str(pd.Series(top).mode().iloc[0]), best_priority)

    keys = list(channel_rows)
    if len(keys) == 2:
        k1, k2 = keys
        d1 = decisions.get(k1)
        d2 = decisions.get(k2)
        if d1 and not d2:
            decisions[k2] = ("Humidity" if d1[0] == "Temperature" else "Temperature", 60)
        elif d2 and not d1:
            decisions[k1] = ("Humidity" if d2[0] == "Temperature" else "Temperature", 60)
        elif d1 and d2 and d1[0] == d2[0]:
            # If the channels collide only because one decision is weaker,
            # complement the weak side.  Equal strong evidence is left intact
            # so the validation layer can flag the source inconsistency rather
            # than silently rewriting authoritative units.
            if d1[1] > d2[1]:
                decisions[k2] = ("Humidity" if d1[0] == "Temperature" else "Temperature", 60)
            elif d2[1] > d1[1]:
                decisions[k1] = ("Humidity" if d2[0] == "Temperature" else "Temperature", 60)

    return {key: value[0] for key, value in decisions.items()}


def _build_long_validation(
    group: pd.DataFrame,
    channel_map: Mapping[str, str],
    canonical: pd.DataFrame,
) -> Dict[str, Any]:
    """Create parse diagnostics used by the Streamlit preflight table."""

    mapped = group.copy()
    mapped["Kind"] = mapped["ChannelKey"].map(channel_map)
    mapped = mapped[mapped["Kind"].isin(CANONICAL_CHANNELS)]

    temp_source = mapped[mapped["Kind"].eq("Temperature")]
    hum_source = mapped[mapped["Kind"].eq("Humidity")]
    temp_ts = pd.Index(temp_source["DateTime"].dropna().unique())
    hum_ts = pd.Index(hum_source["DateTime"].dropna().unique())

    temp_points = (
        int(pd.to_numeric(canonical["Temperature"], errors="coerce").notna().sum())
        if "Temperature" in canonical.columns
        else 0
    )
    hum_points = (
        int(pd.to_numeric(canonical["Humidity"], errors="coerce").notna().sum())
        if "Humidity" in canonical.columns
        else 0
    )

    mapped_kinds = set(channel_map.values())
    problems: list[str] = []
    if len(channel_map) >= 2 and mapped_kinds != set(CANONICAL_CHANNELS):
        problems.append("physical channels did not resolve to one Temperature and one Humidity channel")
    if len(temp_source) and temp_points == 0:
        problems.append("temperature source rows were found but no Temperature points survived parsing")
    if len(hum_source) and hum_points == 0:
        problems.append("humidity source rows were found but no Humidity points survived parsing")

    shared = temp_ts.intersection(hum_ts)
    return {
        "status": "ERROR" if problems else "PASS",
        "message": "; ".join(problems),
        "raw_rows": int(len(group)),
        "physical_channels": ", ".join(sorted(str(v) for v in group["Channel"].dropna().unique())),
        "temperature_source_rows": int(len(temp_source)),
        "humidity_source_rows": int(len(hum_source)),
        "temperature_unique_source_timestamps": int(len(temp_ts)),
        "humidity_unique_source_timestamps": int(len(hum_ts)),
        "temperature_points": temp_points,
        "humidity_points": hum_points,
        "temperature_duplicate_rows": int(max(0, len(temp_source) - len(temp_ts))),
        "humidity_duplicate_rows": int(max(0, len(hum_source) - len(hum_ts))),
        "shared_timestamps": int(len(shared)),
        "temperature_only_timestamps": int(len(temp_ts.difference(hum_ts))),
        "humidity_only_timestamps": int(len(hum_ts.difference(temp_ts))),
        "unpaired_timestamps": int(len(temp_ts.symmetric_difference(hum_ts))),
    }


def _build_wide_validation(canonical: pd.DataFrame) -> Dict[str, Any]:
    temp_points = int(pd.to_numeric(canonical.get("Temperature"), errors="coerce").notna().sum())
    hum_points = int(pd.to_numeric(canonical.get("Humidity"), errors="coerce").notna().sum())
    return {
        "status": "PASS" if (temp_points or hum_points) else "ERROR",
        "message": "" if (temp_points or hum_points) else "no canonical measurement values were found",
        "raw_rows": int(len(canonical)),
        "physical_channels": "canonical wide",
        "temperature_source_rows": temp_points,
        "humidity_source_rows": hum_points,
        "temperature_unique_source_timestamps": temp_points,
        "humidity_unique_source_timestamps": hum_points,
        "temperature_points": temp_points,
        "humidity_points": hum_points,
        "temperature_duplicate_rows": 0,
        "humidity_duplicate_rows": 0,
        "shared_timestamps": int((canonical["Temperature"].notna() & canonical["Humidity"].notna()).sum()),
        "temperature_only_timestamps": int((canonical["Temperature"].notna() & canonical["Humidity"].isna()).sum()),
        "humidity_only_timestamps": int((canonical["Humidity"].notna() & canonical["Temperature"].isna()).sum()),
        "unpaired_timestamps": int((canonical["Temperature"].notna() ^ canonical["Humidity"].notna()).sum()),
    }


def _infer_profile(*hints: object) -> str:
    text = " ".join(_clean_text(v).lower() for v in hints if _clean_text(v))
    if any(token in text for token in ("-80", "minus 80", "ultra low", "ultra-low", "ult", "ulf")):
        return "Freezer -80"
    if "freezer" in text:
        return "Freezer"
    if any(token in text for token in ("fridge", "refrigerator", "cooler", "cold room")):
        return "Fridge"
    if "olympus room" in text:
        return "Olympus Room"
    if "olympus" in text:
        return "Olympus"
    return "Room"


def _coalesce_timestamp_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical row per timestamp while retaining both channels.

    The common ingest path is already unique by timestamp after ``pivot_table``;
    avoiding an unnecessary Python aggregation here materially speeds up large
    monthly exports.  When uploads overlap, pandas ``groupby().last()`` keeps
    the last non-null value independently for Temperature and Humidity, so a
    row from one file cannot erase the complementary channel from another.
    """

    if df.empty:
        return df
    work = df.copy()
    work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
    work = work.dropna(subset=["DateTime"])
    for ch in CANONICAL_CHANNELS:
        if ch not in work.columns:
            work[ch] = pd.NA
        work[ch] = pd.to_numeric(work[ch], errors="coerce")

    if work["DateTime"].duplicated().any():
        work = (
            work.sort_values("DateTime")
            .groupby("DateTime", as_index=False, sort=True)[list(CANONICAL_CHANNELS)]
            .last()
        )
    else:
        work = work.sort_values("DateTime")[["DateTime", *CANONICAL_CHANNELS]]

    work["Date"] = work["DateTime"].dt.normalize()
    work["Time"] = work["DateTime"].dt.strftime("%H:%M")
    return work[["DateTime", "Temperature", "Humidity", "Date", "Time"]].reset_index(drop=True)


def _modern_traceable(
    df: pd.DataFrame,
    source_name: str,
    overrides: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Dict[str, SerialDataset]:
    columns = {
        key: _find_column(df.columns, key)
        for key in ("timestamp", "serial", "channel", "data", "unit", "space_name", "space_type")
    }
    if not all(columns[key] for key in ("timestamp", "serial", "channel", "data")):
        return {}

    rename = {
        columns["timestamp"]: "Timestamp",
        columns["serial"]: "Serial",
        columns["channel"]: "Channel",
        columns["data"]: "Data",
    }
    if columns["unit"]:
        rename[columns["unit"]] = "Unit"
    if columns["space_name"]:
        rename[columns["space_name"]] = "SpaceName"
    if columns["space_type"]:
        rename[columns["space_type"]] = "SpaceType"

    work = df.rename(columns=rename).copy()
    if "Unit" not in work.columns:
        work["Unit"] = ""
    if "SpaceName" not in work.columns:
        work["SpaceName"] = ""
    if "SpaceType" not in work.columns:
        work["SpaceType"] = ""

    work["Serial"] = work["Serial"].map(_clean_text)
    work["Channel"] = work["Channel"].map(_clean_text)
    work["ChannelKey"] = work["Channel"].map(_channel_key)
    work["DateTime"] = _parse_timestamp_series(work["Timestamp"])
    work["Value"] = _numeric_series(work["Data"])
    work = work[
        work["Serial"].ne("")
        & work["ChannelKey"].ne("")
        & work["DateTime"].notna()
        & work["Value"].notna()
    ].copy()
    if work.empty:
        return {}

    output: Dict[str, SerialDataset] = {}
    for serial, group in work.groupby("Serial", sort=False):
        group = group.copy()
        channel_map = _build_channel_map(group, serial, overrides)
        group["Kind"] = group["ChannelKey"].map(channel_map)
        group = group[group["Kind"].isin(CANONICAL_CHANNELS)]
        if group.empty:
            continue

        pivot = (
            group.pivot_table(
                index="DateTime",
                columns="Kind",
                values="Value",
                aggfunc="last",
            )
            .rename_axis(None, axis=1)
            .reset_index()
        )
        pivot = _coalesce_timestamp_rows(pivot)

        space_name = next((v for v in group["SpaceName"].map(_clean_text) if v), "")
        space_type = next((v for v in group["SpaceType"].map(_clean_text) if v), "")
        profile = _infer_profile(source_name, serial, space_name, space_type)
        range_map = dict(PROFILE_LIMITS.get(profile, PROFILE_LIMITS["Room"]))

        validation = _build_long_validation(group, channel_map, pivot)
        output[str(serial)] = SerialDataset(
            serial=str(serial),
            df=pivot,
            sources=[source_name],
            space_name=space_name,
            space_type=space_type,
            channel_map=channel_map,
            range_map=range_map,
            validation=validation,
        )
    return output



def _canonical_wide(
    df: pd.DataFrame,
    source_name: str,
) -> Dict[str, SerialDataset]:
    """Parse the app's canonical wide CSV format.

    Expected columns are DateTime/Timestamp, Serial Number/Serial, and at least
    one of Temperature or Humidity.  This makes cleaned CSVs exported by the
    app safe to re-upload later.
    """
    timestamp_col = _find_column(df.columns, "timestamp")
    serial_col = _find_column(df.columns, "serial")
    normalized = {_header_key(c): c for c in df.columns}
    temp_col = next((orig for key, orig in normalized.items() if key in {"temperature", "temp", "temperature c", "temp c"}), None)
    hum_col = next((orig for key, orig in normalized.items() if key in {"humidity", "rh", "relative humidity", "humidity rh"}), None)
    if not timestamp_col or not serial_col or not (temp_col or hum_col):
        return {}

    work = pd.DataFrame({
        "Serial": df[serial_col].map(_clean_text),
        "DateTime": _parse_timestamp_series(df[timestamp_col]),
    })
    work["Temperature"] = _numeric_series(df[temp_col]) if temp_col else pd.NA
    work["Humidity"] = _numeric_series(df[hum_col]) if hum_col else pd.NA
    work = work[work["Serial"].ne("") & work["DateTime"].notna()]
    output: Dict[str, SerialDataset] = {}
    for serial, group in work.groupby("Serial", sort=False):
        clean = _coalesce_timestamp_rows(group[["DateTime", "Temperature", "Humidity"]])
        profile = _infer_profile(source_name, serial)
        output[str(serial)] = SerialDataset(
            serial=str(serial),
            df=clean,
            sources=[source_name],
            channel_map={"temperature": "Temperature", "humidity": "Humidity"},
            range_map=dict(PROFILE_LIMITS.get(profile, PROFILE_LIMITS["Room"])),
            validation=_build_wide_validation(clean),
        )
    return output

def parse_serial_bytes(
    source_name: str,
    raw: bytes,
    overrides: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Dict[str, Dict[str, object]]:
    """Parse one Traceable CSV into canonical per-serial datasets."""
    df = _read_table(raw)
    if df.empty:
        return {}
    parsed = _canonical_wide(df, source_name)
    if not parsed:
        parsed = _modern_traceable(df, source_name, overrides)
    return {serial: dataset.as_dict() for serial, dataset in parsed.items()}


def merge_serial_data(existing: dict, new_df: pd.DataFrame, serial: str) -> pd.DataFrame:
    """Merge uploads for one serial while preserving both measurement columns."""
    base = existing.get(serial)
    frames = [f.copy() for f in (base, new_df) if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        existing[serial] = pd.DataFrame()
        return existing[serial]
    # Drop measurement columns that are entirely empty within an individual file
    # before concatenation. The union across files is preserved, while avoiding
    # pandas dtype ambiguity for all-NA columns.
    compact_frames = []
    for frame in frames:
        drop_cols = [
            ch for ch in CANONICAL_CHANNELS
            if ch in frame.columns and frame[ch].isna().all()
        ]
        compact_frames.append(frame.drop(columns=drop_cols))
    merged = _coalesce_timestamp_rows(pd.concat(compact_frames, ignore_index=True, sort=False))
    existing[serial] = merged
    return merged


def parse_serial_csv(
    files,
    overrides: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Dict[str, Dict[str, object]]:
    """Compatibility wrapper for Streamlit UploadedFile objects."""
    output: Dict[str, Dict[str, object]] = {}
    frames: Dict[str, pd.DataFrame] = {}

    for file_obj in files or []:
        name = getattr(file_obj, "name", "serial.csv") or "serial.csv"
        getter = getattr(file_obj, "getvalue", None)
        raw = getter() if callable(getter) else file_obj.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        parsed = parse_serial_bytes(name, raw, overrides)

        for serial, info in parsed.items():
            df = info.get("df")
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            merged = merge_serial_data(frames, df, serial)
            current = output.setdefault(
                serial,
                {
                    "serial": serial,
                    "df": merged,
                    "source_name": "",
                    "space_name": "",
                    "space_type": "",
                    "channel_map": {},
                    "range_map": {},
                    "channels": [],
                    "validation": {},
                    "option_label": serial,
                    "default_label": serial,
                },
            )
            current["df"] = merged
            current["channel_map"].update(info.get("channel_map", {}))
            current["range_map"].update(info.get("range_map", {}))
            for field in ("space_name", "space_type"):
                if info.get(field):
                    current[field] = info[field]
            sources = [s.strip() for s in str(current.get("source_name", "")).split(",") if s.strip()]
            sources.extend([s.strip() for s in str(info.get("source_name", name)).split(",") if s.strip()])
            current["source_name"] = ", ".join(dict.fromkeys(sources))
            current["channels"] = [
                ch for ch in CANONICAL_CHANNELS
                if ch in merged.columns and pd.to_numeric(merged[ch], errors="coerce").notna().any()
            ]

            incoming_validation = info.get("validation") if isinstance(info.get("validation"), dict) else {}
            previous_validation = current.get("validation") if isinstance(current.get("validation"), dict) else {}
            validation = dict(previous_validation)
            if incoming_validation:
                for key in (
                    "raw_rows",
                    "temperature_source_rows",
                    "humidity_source_rows",
                    "temperature_duplicate_rows",
                    "humidity_duplicate_rows",
                ):
                    validation[key] = int(previous_validation.get(key, 0) or 0) + int(incoming_validation.get(key, 0) or 0)
                physical = []
                for value in (previous_validation.get("physical_channels", ""), incoming_validation.get("physical_channels", "")):
                    physical.extend(part.strip() for part in str(value).split(",") if part.strip())
                validation["physical_channels"] = ", ".join(dict.fromkeys(physical))
                if incoming_validation.get("status") == "ERROR" or previous_validation.get("status") == "ERROR":
                    validation["status"] = "ERROR"
                else:
                    validation["status"] = "PASS"
                messages = [m for m in (previous_validation.get("message"), incoming_validation.get("message")) if m]
                validation["message"] = "; ".join(dict.fromkeys(messages))

            validation["temperature_points"] = int(pd.to_numeric(merged["Temperature"], errors="coerce").notna().sum())
            validation["humidity_points"] = int(pd.to_numeric(merged["Humidity"], errors="coerce").notna().sum())
            validation["shared_timestamps"] = int((merged["Temperature"].notna() & merged["Humidity"].notna()).sum())
            validation["temperature_only_timestamps"] = int((merged["Temperature"].notna() & merged["Humidity"].isna()).sum())
            validation["humidity_only_timestamps"] = int((merged["Humidity"].notna() & merged["Temperature"].isna()).sum())
            validation["unpaired_timestamps"] = validation["temperature_only_timestamps"] + validation["humidity_only_timestamps"]
            current["validation"] = validation
            descriptor = current.get("space_name") or current.get("space_type")
            label = f"{serial} - {descriptor}" if descriptor else serial
            current["option_label"] = label
            current["default_label"] = label
    return output


def _parse_probe_files(files):
    """Parse legacy probe exports used as optional comparison overlays."""
    dfs: Dict[str, pd.DataFrame] = {}
    ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for file_obj in files or []:
        name = getattr(file_obj, "name", "probe.csv")
        getter = getattr(file_obj, "getvalue", None)
        raw = getter() if callable(getter) else file_obj.read()
        df = _read_table(raw)
        if df.empty:
            continue
        date_col = next((c for c in df.columns if _header_key(c) == "date"), None)
        time_col = next((c for c in df.columns if _header_key(c) == "time"), None)
        if not date_col or not time_col:
            continue
        out = pd.DataFrame()
        out["DateTime"] = pd.to_datetime(
            df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
            errors="coerce",
        )
        lower = {str(c).lower(): c for c in df.columns}
        hum_col = next((orig for low, orig in lower.items() if "ch1" in low or "ch3" in low or "humid" in low), None)
        temp_col = next((orig for low, orig in lower.items() if "ch2" in low or "ch4" in low or "temp" in low), None)
        if hum_col:
            out["Humidity"] = _numeric_series(df[hum_col])
        if temp_col:
            out["Temperature"] = _numeric_series(df[temp_col])
        out = _coalesce_timestamp_rows(out)
        dfs[name] = out
        ranges[name] = {"Temperature": DEFAULT_TEMP_RANGE, "Humidity": DEFAULT_HUMIDITY_RANGE}
    return dfs, ranges


def _parse_tempstick_files(files):
    result: Dict[str, pd.DataFrame] = {}
    for file_obj in files or []:
        name = getattr(file_obj, "name", "tempstick.csv")
        getter = getattr(file_obj, "getvalue", None)
        raw = getter() if callable(getter) else file_obj.read()
        df = _read_table(raw)
        if df.empty:
            continue
        ts_col = next((c for c in df.columns if "timestamp" in _header_key(c) or _header_key(c) == "datetime"), None)
        if not ts_col:
            continue
        out = pd.DataFrame({"DateTime": _parse_timestamp_series(df[ts_col])})
        temp_col = next((c for c in df.columns if "temp" in _header_key(c)), None)
        hum_col = next((c for c in df.columns if "hum" in _header_key(c) or _header_key(c) == "rh"), None)
        if temp_col:
            out["Temperature"] = _numeric_series(df[temp_col])
        if hum_col:
            out["Humidity"] = _numeric_series(df[hum_col])
        result[name] = _coalesce_timestamp_rows(out)
    return result


__all__ = [
    "CANONICAL_CHANNELS",
    "DEFAULT_HUMIDITY_RANGE",
    "DEFAULT_TEMP_RANGE",
    "PROFILE_LIMITS",
    "load_serial_overrides",
    "merge_serial_data",
    "parse_serial_bytes",
    "parse_serial_csv",
    "_parse_probe_files",
    "_parse_tempstick_files",
]
