from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from data_processing import (
    CANONICAL_CHANNELS,
    PROFILE_LIMITS,
    load_serial_overrides,
    merge_serial_data,
    parse_serial_csv,
    parse_tempstick_csv,
)
from oor import compute_oor_events
from report_metadata import (
    limits_from_metadata,
    metadata_lookup,
    read_metadata_bytes,
    template_workbook_bytes,
)

APP_DIR = Path(__file__).resolve().parent
OVERRIDE_FILE = APP_DIR / "serial_channel_overrides.csv"
PROFILE_FILE = APP_DIR / "environment_profiles.csv"

st.set_page_config(page_title="Environmental Reporting", page_icon="🌡️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      .small-note {color: #667085; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_profile_config() -> Dict[str, Dict[str, Tuple[float, float]]]:
    if not PROFILE_FILE.exists():
        return PROFILE_LIMITS
    try:
        df = pd.read_csv(PROFILE_FILE)
    except Exception:
        return PROFILE_LIMITS
    result: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for _, row in df.iterrows():
        profile = str(row.get("Profile", "")).strip()
        if not profile:
            continue
        limits: Dict[str, Tuple[float, float]] = {}
        tmin, tmax = row.get("Temperature Min"), row.get("Temperature Max")
        hmin, hmax = row.get("Humidity Min"), row.get("Humidity Max")
        if pd.notna(tmin) and pd.notna(tmax):
            limits["Temperature"] = (float(tmin), float(tmax))
        if pd.notna(hmin) and pd.notna(hmax):
            limits["Humidity"] = (float(hmin), float(hmax))
        result[profile] = limits
    return result or PROFILE_LIMITS


SERIAL_OVERRIDES = load_serial_overrides(OVERRIDE_FILE)
PROFILE_CONFIG = _read_profile_config()
PARSER_VERSION = "2026-10-27-v2.4-tempstick-backup-overlay"
CONFIG_SIGNATURE = hashlib.sha1(
    (PARSER_VERSION + json.dumps(SERIAL_OVERRIDES, sort_keys=True)).encode("utf-8")
).hexdigest()[:10]


@st.cache_data(show_spinner=False, max_entries=32)
def _parse_uploaded(name: str, data: bytes, config_signature: str):
    _ = config_signature

    class MemoryUpload:
        def __init__(self, file_name: str, payload: bytes):
            self.name = file_name
            self._payload = payload

        def getvalue(self) -> bytes:
            return self._payload

    return parse_serial_csv([MemoryUpload(name, data)], overrides=SERIAL_OVERRIDES)


@st.cache_data(show_spinner=False, max_entries=32)
def _parse_tempstick_uploaded(name: str, data: bytes, parser_version: str):
    _ = parser_version

    class MemoryUpload:
        def __init__(self, file_name: str, payload: bytes):
            self.name = file_name
            self._payload = payload

        def getvalue(self) -> bytes:
            return self._payload

    return parse_tempstick_csv([MemoryUpload(name, data)])


@st.cache_data(show_spinner=False, max_entries=8)
def _parse_metadata_uploaded(name: str, data: bytes):
    return read_metadata_bytes(name, data)


def _merge_metadata(existing: dict, incoming: dict, merged_df: pd.DataFrame) -> dict:
    result = dict(existing)
    result["df"] = merged_df
    result["serial"] = incoming.get("serial") or existing.get("serial")
    for field in ("space_name", "space_type", "option_label", "default_label"):
        if incoming.get(field):
            result[field] = incoming[field]

    sources = []
    for value in (existing.get("source_name"), incoming.get("source_name")):
        if isinstance(value, str):
            sources.extend(part.strip() for part in value.split(",") if part.strip())
    result["source_name"] = ", ".join(dict.fromkeys(sources))

    range_map = dict(existing.get("range_map") or {})
    range_map.update(incoming.get("range_map") or {})
    result["range_map"] = range_map

    channel_map = dict(existing.get("channel_map") or {})
    channel_map.update(incoming.get("channel_map") or {})
    result["channel_map"] = channel_map

    existing_validation = existing.get("validation") if isinstance(existing.get("validation"), dict) else {}
    incoming_validation = incoming.get("validation") if isinstance(incoming.get("validation"), dict) else {}
    validation = dict(existing_validation)
    if incoming_validation:
        for key in (
            "raw_rows",
            "temperature_source_rows",
            "humidity_source_rows",
            "temperature_duplicate_rows",
            "humidity_duplicate_rows",
        ):
            validation[key] = int(existing_validation.get(key, 0) or 0) + int(incoming_validation.get(key, 0) or 0)
        physical = []
        for value in (existing_validation.get("physical_channels", ""), incoming_validation.get("physical_channels", "")):
            physical.extend(part.strip() for part in str(value).split(",") if part.strip())
        validation["physical_channels"] = ", ".join(dict.fromkeys(physical))
        validation["status"] = (
            "ERROR"
            if "ERROR" in {existing_validation.get("status"), incoming_validation.get("status")}
            else "PASS"
        )
        messages = [m for m in (existing_validation.get("message"), incoming_validation.get("message")) if m]
        validation["message"] = "; ".join(dict.fromkeys(messages))

    if "Temperature" not in merged_df.columns:
        merged_df["Temperature"] = pd.NA
    if "Humidity" not in merged_df.columns:
        merged_df["Humidity"] = pd.NA
    temp_valid = pd.to_numeric(merged_df["Temperature"], errors="coerce").notna()
    hum_valid = pd.to_numeric(merged_df["Humidity"], errors="coerce").notna()
    validation["temperature_points"] = int(temp_valid.sum())
    validation["humidity_points"] = int(hum_valid.sum())
    validation["shared_timestamps"] = int((temp_valid & hum_valid).sum())
    validation["temperature_only_timestamps"] = int((temp_valid & ~hum_valid).sum())
    validation["humidity_only_timestamps"] = int((hum_valid & ~temp_valid).sum())
    validation["unpaired_timestamps"] = int((temp_valid ^ hum_valid).sum())
    validation.setdefault("status", "PASS" if (temp_valid.any() or hum_valid.any()) else "ERROR")
    validation.setdefault("message", "")
    result["validation"] = validation

    result["channels"] = [
        ch for ch in CANONICAL_CHANNELS
        if ch in merged_df.columns and pd.to_numeric(merged_df[ch], errors="coerce").notna().any()
    ]
    return result


def _combine_uploads(uploaded_files) -> Dict[str, dict]:
    combined: Dict[str, dict] = {}
    frames: Dict[str, pd.DataFrame] = {}
    for uploaded in uploaded_files or []:
        parsed = _parse_uploaded(uploaded.name, uploaded.getvalue(), CONFIG_SIGNATURE)
        for serial, info in parsed.items():
            df = info.get("df")
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            merged = merge_serial_data(frames, df, serial)
            combined[serial] = _merge_metadata(combined.get(serial, {}), info, merged)
    return combined


def _combine_tempstick_uploads(uploaded_files) -> Dict[str, dict]:
    combined: Dict[str, dict] = {}
    for uploaded in uploaded_files or []:
        parsed = _parse_tempstick_uploaded(uploaded.name, uploaded.getvalue(), PARSER_VERSION)
        for sensor_id, info in parsed.items():
            incoming_df = info.get("df")
            if not isinstance(incoming_df, pd.DataFrame) or incoming_df.empty:
                continue
            if sensor_id not in combined:
                combined[sensor_id] = dict(info)
                continue

            current = combined[sensor_id]
            merged = pd.concat([current["df"], incoming_df], ignore_index=True, sort=False)
            merged["DateTime"] = pd.to_datetime(merged["DateTime"], errors="coerce")
            merged = merged.dropna(subset=["DateTime"]).sort_values("DateTime")
            for ch in CANONICAL_CHANNELS:
                if ch not in merged.columns:
                    merged[ch] = pd.NA
                merged[ch] = pd.to_numeric(merged[ch], errors="coerce")
            merged = merged.groupby("DateTime", as_index=False, sort=True)[list(CANONICAL_CHANNELS)].last()
            merged["Date"] = merged["DateTime"].dt.normalize()
            merged["Time"] = merged["DateTime"].dt.strftime("%H:%M")
            current["df"] = merged[["DateTime", "Temperature", "Humidity", "Date", "Time"]]
            current["channels"] = [
                ch for ch in CANONICAL_CHANNELS
                if pd.to_numeric(current["df"][ch], errors="coerce").notna().any()
            ]
            sources = []
            for value in (current.get("source_name"), info.get("source_name")):
                if isinstance(value, str):
                    sources.extend(part.strip() for part in value.split(",") if part.strip())
            current["source_name"] = ", ".join(dict.fromkeys(sources))
    return combined


def _series_frame(df: pd.DataFrame, channel: str, source_label: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or channel not in df.columns:
        return pd.DataFrame(columns=["DateTime", channel, "Source"])
    out = df[["DateTime", channel]].copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
    out[channel] = pd.to_numeric(out[channel], errors="coerce")
    out = out.dropna(subset=["DateTime", channel]).sort_values("DateTime")
    out["Source"] = source_label
    return out.drop_duplicates(subset=["DateTime"], keep="last").reset_index(drop=True)


def _combined_oor_series(
    preferred_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    channel: str,
    preferred_label: str,
    fallback_label: str,
    tolerance_minutes: float,
) -> pd.DataFrame:
    """Use the preferred source when readings overlap; use the other source only to fill gaps.

    No values are interpolated. A fallback point is excluded when a preferred-source
    reading exists within +/- tolerance_minutes of that timestamp.
    """
    preferred = _series_frame(preferred_df, channel, preferred_label)
    fallback = _series_frame(fallback_df, channel, fallback_label)
    if preferred.empty:
        return fallback
    if fallback.empty:
        return preferred

    probe = fallback.copy().reset_index(drop=True)
    probe["_row_id"] = range(len(probe))
    preferred_times = preferred[["DateTime"]].rename(columns={"DateTime": "PreferredTime"})
    matched = pd.merge_asof(
        probe.sort_values("DateTime"),
        preferred_times.sort_values("PreferredTime"),
        left_on="DateTime",
        right_on="PreferredTime",
        direction="nearest",
        tolerance=pd.to_timedelta(max(0.0, float(tolerance_minutes)), unit="m"),
    )
    keep_ids = set(matched.loc[matched["PreferredTime"].isna(), "_row_id"].astype(int).tolist())
    fallback_keep = fallback.loc[fallback.index.isin(keep_ids)].copy()
    return (
        pd.concat([preferred, fallback_keep], ignore_index=True, sort=False)
        .sort_values("DateTime")
        .drop_duplicates(subset=["DateTime"], keep="first")
        .reset_index(drop=True)
    )


def _periods(serial_data: Dict[str, dict]) -> list[pd.Period]:
    values: set[pd.Period] = set()
    for info in serial_data.values():
        df = info.get("df")
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        dt = pd.to_datetime(df["DateTime"], errors="coerce").dropna()
        values.update(dt.dt.to_period("M").tolist())
    return sorted(values)


def _filter_period(df: pd.DataFrame, period: pd.Period) -> pd.DataFrame:
    dt = pd.to_datetime(df["DateTime"], errors="coerce")
    return df.loc[dt.dt.to_period("M").eq(period)].copy().sort_values("DateTime")


def _downsample(df: pd.DataFrame, channel: str, max_points: int = 6000) -> pd.DataFrame:
    view = df[["DateTime", channel]].dropna().copy()
    if len(view) <= max_points:
        return view
    stride = max(1, math.ceil(len(view) / max_points))
    sampled = view.iloc[::stride].copy()
    if not sampled.empty and sampled.index[-1] != view.index[-1]:
        sampled = pd.concat([sampled, view.iloc[[-1]]]).drop_duplicates(subset=["DateTime"], keep="last")
    return sampled


def _chart(
    df: pd.DataFrame,
    channel: str,
    limits: Optional[Tuple[float, float]],
    title_line: str,
    detail_line: str,
    series_label: str,
    backup_df: Optional[pd.DataFrame] = None,
    backup_label: str = "",
):
    primary = _downsample(df, channel).copy()
    primary["Series"] = series_label
    primary["Source Type"] = "Traceable"

    series_frames = [primary]
    backup_has_data = False
    if isinstance(backup_df, pd.DataFrame) and not backup_df.empty and channel in backup_df.columns:
        backup = _downsample(backup_df, channel).copy()
        if not backup.empty:
            backup_has_data = True
            backup["Series"] = backup_label or "TempStick backup"
            backup["Source Type"] = "TempStick"
            series_frames.append(backup)

    view = pd.concat(series_frames, ignore_index=True, sort=False)
    legend_domain = [series_label]
    legend_range = ["#1f77b4"]
    if backup_has_data:
        legend_domain.append(backup_label or "TempStick backup")
        legend_range.append("#9467bd")
    if limits:
        legend_domain.extend(["Lower Limit", "Upper Limit"])
        legend_range.extend(["#2ca02c", "#d62728"])
    color_scale = alt.Scale(domain=legend_domain, range=legend_range)

    line = (
        alt.Chart(view)
        .mark_line()
        .encode(
            x=alt.X("DateTime:T", title="Date/Time"),
            y=alt.Y(
                f"{channel}:Q",
                title=("Temperature (°C)" if channel == "Temperature" else "Humidity (%RH)"),
            ),
            color=alt.Color(
                "Series:N",
                scale=color_scale,
                legend=alt.Legend(title="Series", orient="bottom", direction="horizontal"),
            ),
            strokeDash=alt.StrokeDash(
                "Source Type:N",
                scale=alt.Scale(domain=["Traceable", "TempStick"], range=[[1, 0], [7, 4]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("DateTime:T", title="Time"),
                alt.Tooltip("Series:N", title="Source"),
                alt.Tooltip(f"{channel}:Q", format=".2f"),
            ],
        )
    )
    layers = [line]
    if limits:
        limit_df = pd.DataFrame(
            {
                "Limit": [limits[0], limits[1]],
                "Series": ["Lower Limit", "Upper Limit"],
            }
        )
        layers.append(
            alt.Chart(limit_df)
            .mark_rule(strokeDash=[5, 4])
            .encode(
                y="Limit:Q",
                color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title="Series", orient="bottom")),
                tooltip=[alt.Tooltip("Series:N"), alt.Tooltip("Limit:Q", format=".2f")],
            )
        )

    return (
        alt.layer(*layers)
        .properties(
            title=alt.TitleParams(
                text=[title_line, detail_line],
                anchor="start",
                fontSize=16,
                fontWeight="bold",
                lineHeight=20,
                offset=10,
            ),
            height=420,
        )
        .configure(background="white", view=alt.ViewConfig(fill="white", stroke=None))
        .configure_legend(symbolLimit=0)
        .interactive(bind_y=False)
    )


def _metadata_text(mapping: Dict[str, object], key: str, fallback: str = "") -> str:
    value = mapping.get(key)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return fallback
    return text


def _normalized_csv(serial: str, df: pd.DataFrame) -> bytes:
    export = df[["DateTime", "Temperature", "Humidity"]].copy()
    export.insert(1, "Serial Number", serial)
    return export.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode("utf-8")


def _zip_export(
    serial: str,
    df: pd.DataFrame,
    events: Dict[str, pd.DataFrame],
    backup_df: Optional[pd.DataFrame] = None,
    backup_sensor_id: str = "",
) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{serial}_traceable_normalized.csv", _normalized_csv(serial, df))
        if isinstance(backup_df, pd.DataFrame) and not backup_df.empty:
            backup_export = backup_df[["DateTime", "Temperature", "Humidity"]].copy()
            backup_export.insert(1, "Backup Sensor ID", backup_sensor_id or "TempStick")
            zf.writestr(
                f"{serial}_tempstick_backup.csv",
                backup_export.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S"),
            )
        for channel, event_df in events.items():
            zf.writestr(f"{serial}_{channel.lower()}_oor.csv", event_df.to_csv(index=False))
    return mem.getvalue()


st.title("Environmental Reporting")
st.caption("Traceable + TempStick backup overlay, reusable report metadata, monthly charts, selectable OOR source logic, and normalized exports. Parser v2.4.")

with st.sidebar:
    st.header("Upload")
    files = st.file_uploader(
        "Traceable CSV files",
        type=["csv", "txt"],
        accept_multiple_files=True,
        help="Upload Traceable per-device Report CSVs, consolidated/multi-serial exports, or cleaned canonical CSVs. Files for the same serial are merged automatically.",
    )
    tempstick_files = st.file_uploader(
        "TempStick backup CSV files (optional)",
        type=["csv"],
        accept_multiple_files=True,
        help=(
            "Optional backup logger exports. TempStick data are overlaid on the Traceable graph after you map each "
            "TempStick sensor to the appropriate Traceable serial number."
        ),
    )
    metadata_file = st.file_uploader(
        "Report metadata mapping (optional)",
        type=["csv", "xlsx", "xlsm"],
        accept_multiple_files=False,
        help=(
            "Optional reusable serial-number mapping for TempStick ID, Chart Title, Probe ID, Equipment ID, "
            "Materials, profile, and channel limits. CSV and Excel are supported."
        ),
    )
    st.caption(f"Serial override rules loaded: {sum(len(v) for v in SERIAL_OVERRIDES.values())}")
    st.caption(f"Profiles loaded: {len(PROFILE_CONFIG)}")

if not files:
    st.info("Upload Traceable CSV files to begin. Supported formats: per-device Report exports, consolidated/multi-serial exports, and cleaned canonical CSVs.")
    st.stop()

# Include file contents in widget state keys.  Streamlit otherwise preserves an
# old multiselect selection for the same serial/month, which can make a newly
# restored Temperature channel appear to be missing even though parsing worked.
upload_hasher = hashlib.sha1()
for uploaded in files:
    payload = uploaded.getvalue()
    upload_hasher.update(uploaded.name.encode("utf-8", errors="replace"))
    upload_hasher.update(hashlib.sha1(payload).digest())
tempstick_hasher = hashlib.sha1()
for uploaded in tempstick_files or []:
    payload = uploaded.getvalue()
    upload_hasher.update(uploaded.name.encode("utf-8", errors="replace"))
    upload_hasher.update(hashlib.sha1(payload).digest())
    tempstick_hasher.update(uploaded.name.encode("utf-8", errors="replace"))
    tempstick_hasher.update(hashlib.sha1(payload).digest())
UPLOAD_SIGNATURE = f"{CONFIG_SIGNATURE}-{upload_hasher.hexdigest()[:12]}"
TEMPSTICK_SIGNATURE = tempstick_hasher.hexdigest()[:12] if tempstick_files else "none"

with st.spinner("Parsing uploaded data..."):
    serial_data = _combine_uploads(files)
    tempstick_data = _combine_tempstick_uploads(tempstick_files)

if not serial_data:
    st.error("No recognized Traceable data was found. Supported inputs are per-device Report exports with Device id/Channel Data blocks, consolidated files with Timestamp/Serial Number/Channel/Data, or cleaned files with DateTime/Serial Number/Temperature and/or Humidity.")
    st.stop()

if tempstick_files and not tempstick_data:
    st.warning("TempStick backup files were uploaded, but no supported TempStick Timestamp/Temperature/Humidity table was found.")

metadata_df = pd.DataFrame()
metadata_map: Dict[str, Dict[str, object]] = {}
metadata_error = ""
metadata_signature = "none"
if metadata_file is not None:
    try:
        metadata_bytes = metadata_file.getvalue()
        metadata_signature = hashlib.sha1(metadata_bytes).hexdigest()[:12]
        metadata_df = _parse_metadata_uploaded(metadata_file.name, metadata_bytes)
        metadata_map = metadata_lookup(metadata_df)
    except Exception as exc:
        metadata_error = str(exc)
        st.sidebar.error(f"Metadata mapping could not be read: {exc}")

with st.sidebar:
    st.download_button(
        "Download metadata template for detected serials",
        template_workbook_bytes(serial_data),
        file_name="report_metadata_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        help="Pre-filled with the serial numbers detected in the current Traceable upload.",
    )
    if metadata_map:
        matched = sum(1 for serial_key in serial_data if serial_key in metadata_map)
        st.success(f"Metadata matched {matched} of {len(serial_data)} detected serials.")
    elif metadata_file is not None and not metadata_error:
        st.warning("The metadata file was readable but contained no serial mappings.")

tempstick_mapping: Dict[str, list[str]] = {}
if tempstick_data:
    with st.sidebar:
        st.divider()
        st.subheader("TempStick backup mapping")
        st.caption("Map each backup sensor to the Traceable serial whose graph it should overlay.")
        serial_options = [None, *sorted(serial_data)]
        for sensor_id, backup_info in sorted(tempstick_data.items()):
            sensor_label = str(backup_info.get("option_label") or sensor_id)
            metadata_target = next(
                (
                    serial_key
                    for serial_key, row in metadata_map.items()
                    if _metadata_text(row, "TempStick ID").lower() == str(sensor_id).lower()
                ),
                None,
            )
            if metadata_target in serial_data:
                default_index = serial_options.index(metadata_target)
            else:
                default_index = 1 if len(serial_data) == 1 else 0
            target = st.selectbox(
                sensor_label,
                options=serial_options,
                index=default_index,
                format_func=lambda value: "Not mapped" if value is None else str(serial_data[value].get("option_label") or value),
                key=f"tempstick_map::{TEMPSTICK_SIGNATURE}::{sensor_id}",
            )
            if target is not None:
                tempstick_mapping.setdefault(str(target), []).append(str(sensor_id))

METADATA_SIGNATURE = metadata_signature

summary_rows = []
for serial, info in serial_data.items():
    df = info["df"]
    validation = info.get("validation") if isinstance(info.get("validation"), dict) else {}
    summary_rows.append(
        {
            "Status": validation.get("status", "PASS"),
            "Serial": serial,
            "Raw rows": int(validation.get("raw_rows", len(df)) or 0),
            "Temperature source rows": int(validation.get("temperature_source_rows", 0) or 0),
            "Temperature plotted points": int(validation.get("temperature_points", 0) or 0),
            "Humidity source rows": int(validation.get("humidity_source_rows", 0) or 0),
            "Humidity plotted points": int(validation.get("humidity_points", 0) or 0),
            "Duplicate raw rows": int(validation.get("temperature_duplicate_rows", 0) or 0)
            + int(validation.get("humidity_duplicate_rows", 0) or 0),
            "Unpaired timestamps": int(validation.get("unpaired_timestamps", 0) or 0),
            "Format": validation.get("source_format", ""),
            "Physical channels": validation.get("physical_channels", ""),
            "Start": df["DateTime"].min(),
            "End": df["DateTime"].max(),
            "Source": info.get("source_name", ""),
            "Message": validation.get("message", ""),
        }
    )

validation_df = pd.DataFrame(summary_rows).sort_values("Serial").reset_index(drop=True)
parse_errors = validation_df[validation_df["Status"].eq("ERROR")]
if not parse_errors.empty:
    st.error(
        "One or more serials failed ingest validation. Expand Upload validation before relying on charts or OOR results."
    )

with st.expander("Upload validation", expanded=not parse_errors.empty):
    st.dataframe(validation_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download validation CSV",
        validation_df.to_csv(index=False).encode("utf-8"),
        file_name="traceable_upload_validation.csv",
        mime="text/csv",
    )

if tempstick_data:
    tempstick_rows = []
    reverse_mapping = {
        sensor_id: serial_key
        for serial_key, sensor_ids in tempstick_mapping.items()
        for sensor_id in sensor_ids
    }
    for sensor_id, backup_info in sorted(tempstick_data.items()):
        bdf = backup_info.get("df")
        validation = backup_info.get("validation") if isinstance(backup_info.get("validation"), dict) else {}
        tempstick_rows.append(
            {
                "Status": validation.get("status", "PASS"),
                "TempStick ID": sensor_id,
                "Sensor": backup_info.get("sensor_name", ""),
                "Mapped Traceable Serial": reverse_mapping.get(sensor_id, "NOT MAPPED"),
                "Temperature points": int(validation.get("temperature_points", 0) or 0),
                "Humidity points": int(validation.get("humidity_points", 0) or 0),
                "Start": bdf["DateTime"].min() if isinstance(bdf, pd.DataFrame) and not bdf.empty else pd.NaT,
                "End": bdf["DateTime"].max() if isinstance(bdf, pd.DataFrame) and not bdf.empty else pd.NaT,
                "Source": backup_info.get("source_name", ""),
            }
        )
    with st.expander("TempStick backup validation", expanded=False):
        st.dataframe(pd.DataFrame(tempstick_rows), use_container_width=True, hide_index=True)

if metadata_file is not None and not metadata_error:
    with st.expander("Report metadata mapping", expanded=False):
        if metadata_df.empty:
            st.caption("No usable metadata rows were found.")
        else:
            display_metadata = metadata_df.copy()
            display_metadata.insert(
                0,
                "Match Status",
                display_metadata["Serial Number"].map(
                    lambda value: "MATCHED" if str(value) in serial_data else "NOT IN CURRENT UPLOAD"
                ),
            )
            st.dataframe(display_metadata, use_container_width=True, hide_index=True)
            unmatched_data = [serial_key for serial_key in serial_data if serial_key not in metadata_map]
            if unmatched_data:
                st.caption(
                    "No metadata row for: " + ", ".join(unmatched_data)
                    + ". These serials will use parser/app defaults and remain manually editable."
                )

available_periods = _periods(serial_data)
if not available_periods:
    st.error("The uploaded data contains no valid timestamps.")
    st.stop()

left, right = st.columns([1, 1])
with left:
    serial = st.selectbox(
        "Serial number",
        sorted(serial_data),
        format_func=lambda s: serial_data[s].get("option_label") or s,
    )
with right:
    period = st.selectbox(
        "Reporting month",
        available_periods,
        index=len(available_periods) - 1,
        format_func=lambda p: p.strftime("%B %Y"),
    )

info = serial_data[serial]
df_all = info["df"].copy()
df = _filter_period(df_all, period)
if df.empty:
    st.warning("This serial has no readings in the selected month.")
    st.stop()

backup_info: Optional[dict] = None
backup_df = pd.DataFrame()
backup_sensor_id = ""
backup_label = ""
backup_candidates = tempstick_mapping.get(serial, [])
if backup_candidates:
    if len(backup_candidates) == 1:
        backup_sensor_id = backup_candidates[0]
    else:
        backup_sensor_id = st.selectbox(
            "Backup TempStick sensor",
            options=backup_candidates,
            format_func=lambda sid: str(tempstick_data[sid].get("option_label") or sid),
            key=f"active_tempstick::{TEMPSTICK_SIGNATURE}::{serial}::{period}",
        )
    backup_info = tempstick_data.get(backup_sensor_id)
    if backup_info:
        candidate = backup_info.get("df")
        if isinstance(candidate, pd.DataFrame) and not candidate.empty:
            backup_df = _filter_period(candidate, period)
            backup_label = f"TempStick {backup_sensor_id}"

show_backup = False
if backup_info is not None:
    if backup_df.empty:
        st.warning(f"Mapped TempStick {backup_sensor_id} has no readings in {period.strftime('%B %Y')}.")
    else:
        show_backup = st.checkbox(
            f"Overlay {backup_label} on graphs",
            value=True,
            key=f"show_tempstick::{TEMPSTICK_SIGNATURE}::{serial}::{period}",
        )
        st.caption(
            f"Backup overlay: {backup_info.get('sensor_name') or backup_sensor_id} • "
            f"{backup_df['DateTime'].min():%Y-%m-%d %H:%M} to {backup_df['DateTime'].max():%Y-%m-%d %H:%M}"
        )

channel_map = info.get("channel_map") or {}
if channel_map:
    mapping_text = " • ".join(f"{physical} → {kind}" for physical, kind in sorted(channel_map.items()))
    st.caption(f"Detected physical channel mapping: {mapping_text}")

selected_validation = info.get("validation") if isinstance(info.get("validation"), dict) else {}
unpaired = int(selected_validation.get("unpaired_timestamps", 0) or 0)
if unpaired:
    st.caption(
        f"Preflight: {unpaired:,} timestamps contain only one of the two measurements. "
        "They are preserved independently; a missing partner does not remove the available Temperature or Humidity reading."
    )

available_channels = [
    ch for ch in CANONICAL_CHANNELS
    if ch in df.columns and pd.to_numeric(df[ch], errors="coerce").notna().any()
]
if not available_channels:
    st.error("No Temperature or Humidity readings survived parsing for this serial/month.")
    st.stop()

if len(available_channels) == 1 and len(channel_map) >= 2:
    st.warning(
        "Two physical channels were detected but only one measurement has usable values in this month. "
        "Review the channel mapping above or add a rule to serial_channel_overrides.csv."
    )

controls, diagnostics = st.columns([2, 1])
with controls:
    channels = st.multiselect(
        "Channels to graph",
        options=available_channels,
        default=available_channels,
        key=f"channels::{UPLOAD_SIGNATURE}::{serial}::{period}",
    )
with diagnostics:
    st.metric("Rows this month", f"{len(df):,}")

metadata_defaults = metadata_map.get(serial, {})
metadata_ranges = limits_from_metadata(metadata_defaults)
metadata_profile = _metadata_text(metadata_defaults, "Profile")

profile_options = ["Auto"] + list(PROFILE_CONFIG)
profile_default = "Auto"
if metadata_profile:
    profile_match = next(
        (candidate for candidate in profile_options if candidate.lower() == metadata_profile.lower()),
        None,
    )
    if profile_match:
        profile_default = profile_match
    else:
        st.warning(
            f"Metadata profile '{metadata_profile}' is not in environment_profiles.csv. Auto detection is being used."
        )
profile = st.selectbox(
    "Environmental profile",
    profile_options,
    index=profile_options.index(profile_default),
    key=f"profile::{METADATA_SIGNATURE}::{serial}",
)
auto_ranges = dict(info.get("range_map") or {})
base_ranges = auto_ranges if profile == "Auto" else dict(PROFILE_CONFIG.get(profile, {}))
base_ranges.update(metadata_ranges)

with st.expander("Limits and report details", expanded=bool(metadata_defaults)):
    title_default = _metadata_text(metadata_defaults, "Chart Title", str(info.get("option_label") or serial))
    probe_default = _metadata_text(metadata_defaults, "Probe ID")
    equipment_default = _metadata_text(metadata_defaults, "Equipment ID")
    materials_default = _metadata_text(metadata_defaults, "Materials")

    c1, c2 = st.columns(2)
    report_title = c1.text_input(
        "Chart title",
        value=title_default,
        key=f"chart_title::{METADATA_SIGNATURE}::{serial}",
    )
    probe_id = c2.text_input(
        "Probe ID",
        value=probe_default,
        key=f"probe_id::{METADATA_SIGNATURE}::{serial}",
    )
    c3, c4 = st.columns(2)
    equipment_id = c3.text_input(
        "Equipment ID",
        value=equipment_default,
        key=f"equipment_id::{METADATA_SIGNATURE}::{serial}",
    )
    materials = c4.text_input(
        "Materials / contents",
        value=materials_default,
        key=f"materials::{METADATA_SIGNATURE}::{serial}",
    )

    metadata_notes = _metadata_text(metadata_defaults, "Notes")
    if metadata_notes:
        st.caption(f"Metadata note: {metadata_notes}")

    effective_ranges: Dict[str, Tuple[float, float]] = {}
    for ch in available_channels:
        default_range = base_ranges.get(ch)
        limit_key_base = f"{METADATA_SIGNATURE}::{serial}::{profile}::{ch}"
        enabled = st.checkbox(
            f"Evaluate {ch} against limits",
            value=default_range is not None,
            key=f"limit_on::{limit_key_base}",
        )
        if enabled:
            default_low, default_high = default_range or ((15.0, 25.0) if ch == "Temperature" else (0.0, 60.0))
            lc1, lc2 = st.columns(2)
            low = lc1.number_input(
                f"{ch} lower limit",
                value=float(default_low),
                key=f"low::{limit_key_base}",
            )
            high = lc2.number_input(
                f"{ch} upper limit",
                value=float(default_high),
                key=f"high::{limit_key_base}",
            )
            if low >= high:
                st.error(f"{ch}: lower limit must be less than upper limit.")
            else:
                effective_ranges[ch] = (float(low), float(high))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Serial", serial)
k2.metric("Channels", len(available_channels))
k3.metric("Start", df["DateTime"].min().strftime("%Y-%m-%d"))
k4.metric("End", df["DateTime"].max().strftime("%Y-%m-%d"))

events_by_channel: Dict[str, pd.DataFrame] = {}
for channel in channels:
    values = pd.to_numeric(df[channel], errors="coerce").dropna()
    if values.empty:
        st.warning(f"{channel} has no valid values in this month.")
        continue

    st.subheader(channel)
    limits = effective_ranges.get(channel)
    title_line = f"{report_title} - {period.strftime('%B %Y')} - {channel}"
    detail_line = (
        f"Materials: {materials or '—'} | Probe: {probe_id or '—'} | Equipment: {equipment_id or '—'}"
    )
    series_label = f"Probe {probe_id}" if probe_id else str(info.get("option_label") or serial)
    backup_channel_available = (
        isinstance(backup_df, pd.DataFrame)
        and not backup_df.empty
        and channel in backup_df.columns
        and pd.to_numeric(backup_df[channel], errors="coerce").notna().any()
    )
    st.altair_chart(
        _chart(
            df,
            channel,
            limits,
            title_line,
            detail_line,
            series_label,
            backup_df=backup_df if (show_backup and backup_channel_available) else None,
            backup_label=backup_label,
        ),
        use_container_width=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Traceable minimum", f"{values.min():.2f}")
    m2.metric("Traceable maximum", f"{values.max():.2f}")
    m3.metric("Traceable average", f"{values.mean():.2f}")
    if backup_channel_available:
        backup_values = pd.to_numeric(backup_df[channel], errors="coerce").dropna()
        b1, b2, b3 = st.columns(3)
        b1.metric("TempStick minimum", f"{backup_values.min():.2f}")
        b2.metric("TempStick maximum", f"{backup_values.max():.2f}")
        b3.metric("TempStick average", f"{backup_values.mean():.2f}")

    if limits:
        source_df = _series_frame(df, channel, series_label)
        oor_description = "Traceable probe only"
        if backup_channel_available:
            source_mode = st.selectbox(
                f"{channel} OOR data source",
                options=[
                    "Traceable probe only",
                    "TempStick backup only",
                    "Combined - prefer Traceable, fill gaps from TempStick",
                    "Combined - prefer TempStick, fill gaps from Traceable",
                ],
                index=0,
                key=f"oor_source::{UPLOAD_SIGNATURE}::{serial}::{period}::{channel}",
                help=(
                    "Combined modes do not average or interpolate readings. The preferred source wins when a reading "
                    "from the other source is within the matching tolerance; the other source is used only to fill gaps."
                ),
            )
            if source_mode == "TempStick backup only":
                source_df = _series_frame(backup_df, channel, backup_label)
                oor_description = source_mode
            elif source_mode.startswith("Combined"):
                match_tolerance = st.number_input(
                    f"{channel} source matching tolerance (minutes)",
                    min_value=0.0,
                    max_value=60.0,
                    value=5.0,
                    step=0.5,
                    key=f"oor_tolerance::{UPLOAD_SIGNATURE}::{serial}::{period}::{channel}",
                    help="Used only to decide when two readings are close enough to be considered overlapping. No interpolation is performed.",
                )
                if "prefer Traceable" in source_mode:
                    source_df = _combined_oor_series(
                        df,
                        backup_df,
                        channel,
                        series_label,
                        backup_label,
                        match_tolerance,
                    )
                else:
                    source_df = _combined_oor_series(
                        backup_df,
                        df,
                        channel,
                        backup_label,
                        series_label,
                        match_tolerance,
                    )
                oor_description = f"{source_mode} (±{match_tolerance:g} min overlap window)"
            st.caption(f"OOR calculation source: {oor_description}. Effective points: {len(source_df):,}.")

        event_df = compute_oor_events(source_df, channel, limits[0], limits[1], series_label)
        events_by_channel[channel] = event_df
        total_minutes = float(event_df["Duration(min)"].sum()) if not event_df.empty else 0.0
        if event_df.empty:
            st.success("No out-of-range events detected for the selected OOR source policy.")
        else:
            st.warning(f"{len(event_df)} OOR event(s), {total_minutes:.1f} total minutes using the selected OOR source policy.")
            st.dataframe(event_df, use_container_width=True, hide_index=True)
    else:
        events_by_channel[channel] = pd.DataFrame(columns=["Start", "End", "Duration(min)", "Source"])
        st.caption("OOR evaluation disabled for this channel.")

st.divider()
st.subheader("Exports")
normalized = _normalized_csv(serial, df)
export_col1, export_col2 = st.columns(2)
with export_col1:
    st.download_button(
        "Download Traceable normalized CSV",
        normalized,
        file_name=f"{serial}_{period.strftime('%Y_%m')}_normalized.csv",
        mime="text/csv",
        use_container_width=True,
    )
with export_col2:
    st.download_button(
        "Download report ZIP",
        _zip_export(serial, df, events_by_channel, backup_df if not backup_df.empty else None, backup_sensor_id),
        file_name=f"{serial}_{period.strftime('%Y_%m')}_report.zip",
        mime="application/zip",
        use_container_width=True,
    )

backup_source_text = f" • Backup mapped: {backup_label}" if backup_label else ""
st.caption(
    f"Profile: {profile} • Materials: {materials or '—'} • Probe: {probe_id or '—'} • "
    f"Equipment: {equipment_id or '—'} • Traceable source: {info.get('source_name') or '—'}{backup_source_text}"
)
