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
)
from oor import compute_oor_events

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
PARSER_VERSION = "2026-10-05-v2.2-report-and-consolidated-support"
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


def _chart(df: pd.DataFrame, channel: str, limits: Optional[Tuple[float, float]], title: str):
    view = _downsample(df, channel)
    line = (
        alt.Chart(view)
        .mark_line()
        .encode(
            x=alt.X("DateTime:T", title="Date / Time"),
            y=alt.Y(f"{channel}:Q", title=("Temperature (°C)" if channel == "Temperature" else "Humidity (%RH)")),
            tooltip=[alt.Tooltip("DateTime:T", title="Time"), alt.Tooltip(f"{channel}:Q", format=".2f")],
        )
    )
    layers = [line]
    if limits:
        limit_df = pd.DataFrame({"Limit": [limits[0], limits[1]], "Label": ["Lower limit", "Upper limit"]})
        layers.append(
            alt.Chart(limit_df)
            .mark_rule(strokeDash=[5, 4])
            .encode(y="Limit:Q", tooltip=["Label:N", alt.Tooltip("Limit:Q", format=".2f")])
        )
    return alt.layer(*layers).properties(title=title, height=320).interactive(bind_y=False)


def _normalized_csv(serial: str, df: pd.DataFrame) -> bytes:
    export = df[["DateTime", "Temperature", "Humidity"]].copy()
    export.insert(1, "Serial Number", serial)
    return export.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode("utf-8")


def _zip_export(serial: str, df: pd.DataFrame, events: Dict[str, pd.DataFrame]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{serial}_normalized.csv", _normalized_csv(serial, df))
        for channel, event_df in events.items():
            zf.writestr(f"{serial}_{channel.lower()}_oor.csv", event_df.to_csv(index=False))
    return mem.getvalue()


st.title("Environmental Reporting")
st.caption("Traceable Report + consolidated CSV ingestion, channel validation, monthly charts, OOR analysis, and normalized exports. Parser v2.2.")

with st.sidebar:
    st.header("Upload")
    files = st.file_uploader(
        "Traceable CSV files",
        type=["csv", "txt"],
        accept_multiple_files=True,
        help="Upload Traceable per-device Report CSVs, consolidated/multi-serial exports, or cleaned canonical CSVs. Files for the same serial are merged automatically.",
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
UPLOAD_SIGNATURE = f"{CONFIG_SIGNATURE}-{upload_hasher.hexdigest()[:12]}"

with st.spinner("Parsing uploaded data..."):
    serial_data = _combine_uploads(files)

if not serial_data:
    st.error("No recognized Traceable data was found. Supported inputs are per-device Report exports with Device id/Channel Data blocks, consolidated files with Timestamp/Serial Number/Channel/Data, or cleaned files with DateTime/Serial Number/Temperature and/or Humidity.")
    st.stop()

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

profile_options = ["Auto"] + list(PROFILE_CONFIG)
profile = st.selectbox("Environmental profile", profile_options, index=0)
auto_ranges = dict(info.get("range_map") or {})
base_ranges = auto_ranges if profile == "Auto" else dict(PROFILE_CONFIG.get(profile, {}))

with st.expander("Limits and report details", expanded=False):
    c1, c2, c3 = st.columns(3)
    report_title = c1.text_input("Chart title", value=info.get("option_label") or serial)
    materials = c2.text_input("Materials / contents", value="")
    equipment_id = c3.text_input("Equipment ID", value="")

    effective_ranges: Dict[str, Tuple[float, float]] = {}
    for ch in available_channels:
        default_range = base_ranges.get(ch)
        enabled = st.checkbox(f"Evaluate {ch} against limits", value=default_range is not None, key=f"limit_on::{serial}::{ch}")
        if enabled:
            default_low, default_high = default_range or ((15.0, 25.0) if ch == "Temperature" else (0.0, 60.0))
            lc1, lc2 = st.columns(2)
            low = lc1.number_input(f"{ch} lower limit", value=float(default_low), key=f"low::{serial}::{ch}")
            high = lc2.number_input(f"{ch} upper limit", value=float(default_high), key=f"high::{serial}::{ch}")
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
    st.altair_chart(
        _chart(df, channel, limits, f"{report_title} - {period.strftime('%B %Y')} - {channel}"),
        use_container_width=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Minimum", f"{values.min():.2f}")
    m2.metric("Maximum", f"{values.max():.2f}")
    m3.metric("Average", f"{values.mean():.2f}")

    if limits:
        source_df = df[["DateTime", channel]].copy()
        source_df["Source"] = info.get("option_label") or serial
        event_df = compute_oor_events(source_df, channel, limits[0], limits[1], str(info.get("option_label") or serial))
        events_by_channel[channel] = event_df
        total_minutes = float(event_df["Duration(min)"].sum()) if not event_df.empty else 0.0
        if event_df.empty:
            st.success("No out-of-range events detected.")
        else:
            st.warning(f"{len(event_df)} OOR event(s), {total_minutes:.1f} total minutes.")
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
        "Download normalized CSV",
        normalized,
        file_name=f"{serial}_{period.strftime('%Y_%m')}_normalized.csv",
        mime="text/csv",
        use_container_width=True,
    )
with export_col2:
    st.download_button(
        "Download report ZIP",
        _zip_export(serial, df, events_by_channel),
        file_name=f"{serial}_{period.strftime('%Y_%m')}_report.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.caption(
    f"Profile: {profile} • Materials: {materials or '—'} • Equipment: {equipment_id or '—'} • "
    f"Sources: {info.get('source_name') or '—'}"
)
