import streamlit as st
import pandas as pd
import altair as alt
import io
from pandas.tseries.offsets import MonthEnd

st.set_page_config(page_title="Environmental Reporting", layout="wide")

st.title("Environmental Monitoring Dashboard")

# --- SIDEBAR: UPLOADS & CONTROLS ---
st.sidebar.header("Data Upload & Settings")

# 1) Probe CSVs
probe_files = st.sidebar.file_uploader(
    "Upload Probe CSV files",
    type="csv",
    accept_multiple_files=True,
    key="probe_uploader"
)

# 2) Tempstick CSVs (optional)
ts_files = st.sidebar.file_uploader(
    "Upload Tempstick CSV files (optional)",
    type="csv",
    accept_multiple_files=True,
    key="ts_uploader"
)

# Parse Tempsticks into a dict of DataFrames
tempdfs = {}
if ts_files:
    for ts in ts_files:
        raw = ts.getvalue().decode(errors="ignore").splitlines()
        # Find last header row containing "timestamp"
        hdr = max(i for i, L in enumerate(raw) if "timestamp" in L.lower())
        content = "\n".join(raw[hdr:])
        df_ts = pd.read_csv(io.StringIO(content), on_bad_lines="skip", skip_blank_lines=True)
        df_ts.columns = [c.strip() for c in df_ts.columns]
        # Normalize columns
        colmap = {}
        for c in df_ts.columns:
            lc = c.lower()
            if "timestamp" in lc:
                colmap[c] = "DateTime"
            elif "temperature" in lc:
                colmap[c] = "Temperature"
            elif "humidity" in lc:
                colmap[c] = "Humidity"
        df_ts = df_ts.rename(columns=colmap)
        keep = [c for c in ["DateTime", "Temperature", "Humidity"] if c in df_ts.columns]
        df_ts = df_ts[keep].copy()
        df_ts["DateTime"] = pd.to_datetime(df_ts["DateTime"], errors="coerce")
        df_ts["Temperature"] = pd.to_numeric(df_ts["Temperature"], errors="coerce")
        if "Humidity" in df_ts:
            df_ts["Humidity"] = pd.to_numeric(df_ts["Humidity"], errors="coerce")
        tempdfs[ts.name] = df_ts

if tempdfs:
    ts_choice = st.sidebar.selectbox("Select Tempstick to overlay", list(tempdfs.keys()), key="ts_choice")
    ts_df = tempdfs[ts_choice]
else:
    ts_df = None

# --- PARSE PROBE CSVs ---
parsed_probes = {}
for f in probe_files or []:
    raw = f.getvalue().decode(errors="ignore").splitlines()
    # find last header row containing CH1 or P1
    hdr = max(i for i, L in enumerate(raw) if "ch1" in L.lower() or "p1" in L.lower())
    content = "\n".join(raw[hdr:])
    df = pd.read_csv(io.StringIO(content), on_bad_lines="skip", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    # Build a map from raw to standard names
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if "date" == lc or "date" in lc:
            colmap[c] = "Date"
        elif "time" == lc or "time" in lc:
            colmap[c] = "Time"
        elif "p1" in lc:
            colmap[c] = "P1"
        elif "p2" in lc:
            colmap[c] = "P2"
        elif "ch3" in lc:
            colmap[c] = "CH3"
        elif "ch4" in lc:
            colmap[c] = "CH4"
    df = df.rename(columns=colmap)
    # Determine asset type by filename
    name = f.name
    lower = name.lower()
    is_fridge = "fridge" in lower and "freezer" not in lower
    is_freezer = "freezer" in lower and "fridge" not in lower
    is_combo = "fridge" in lower and "freezer" in lower
    is_olympus = "olympus" in lower
    is_room = not (is_fridge or is_freezer or is_combo) or is_olympus

    # Filter & rename columns per type
    if is_combo:
        need = ["P1", "P2", "Date", "Time"]
        df = df[[c for c in need if c in df.columns]]
        df = df.rename(columns={"P1": "Fridge Temp", "P2": "Freezer Temp"})
    elif is_fridge or is_freezer:
        need = ["P1", "Date", "Time"]
        df = df[[c for c in need if c in df.columns]]
        df = df.rename(columns={"P1": "Temperature"})
    else:  # rooms & olympus
        # humidity from CH3, temperature from CH4 or fallback CH1/CH2
        hum_col = "CH3" if "CH3" in df.columns else None
        temp_col = "CH4" if "CH4" in df.columns else None
        if not temp_col and "CH2" in df.columns:
            temp_col = "CH2"
        need = [c for c in [hum_col, temp_col, "Date", "Time"] if c]
        df = df[need]
        df = df.rename(columns={hum_col: "Humidity", temp_col: "Temperature"})

    # Parse Date & DateTime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["DateTime"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
                                     errors="coerce")
    parsed_probes[name] = df

# Stop if no probes
if not parsed_probes:
    st.error("Please upload at least one probe CSV.")
    st.stop()

# --- YEAR & MONTH SELECTION ---
all_dates = pd.concat([df["DateTime"] for df in parsed_probes.values()])
years = sorted(all_dates.dt.year.dropna().unique().astype(int).tolist())
year = st.sidebar.selectbox("Year", years, index=len(years)-1)
months = list(range(1, 13))
month = st.sidebar.selectbox("Month", months, index=year == years[-1] and all_dates.dt.month.max()-1)

# Default ranges per asset type
ranges = {}
for name, df in parsed_probes.items():
    lower = name.lower()
    if "fridge" in lower and "freezer" not in lower:
        ranges[name] = {"Temperature": (2, 8)}
    elif "freezer" in lower and "fridge" not in lower:
        ranges[name] = {"Temperature": (-35, -5)}
    elif "fridge" in lower and "freezer" in lower:
        ranges[name] = {"Fridge Temp": (2, 8), "Freezer Temp": (-35, -5)}
    elif "olympus" in lower:
        ranges[name] = {"Temperature": (15, 28), "Humidity": (0, 60)}
    else:
        ranges[name] = {"Temperature": (15, 25), "Humidity": (0, 60)}

# --- MAIN: DISPLAY & PLOT ---
for name, df in parsed_probes.items():
    with st.expander(name, expanded=True):
        # Metadata inputs
        chart_title = st.text_input("Chart Title", value=f"{name} {pd.Timestamp(year, month, 1).strftime('%B %Y')}",
                                    key=f"{name}_title")
        materials = st.text_area("Materials", key=f"{name}_materials")
        probe_id = st.text_input("Probe ID", key=f"{name}_probe")
        equip_id = st.text_input("Equipment ID", key=f"{name}_equip")

        # Filter to selection
        sel = df[
            (df["DateTime"].dt.year == year) &
            (df["DateTime"].dt.month == month)
        ].copy().reset_index(drop=True)

        # Determine channels to plot
        channels = list(ranges[name].keys())

        # Plot each channel
        for ch in channels:
            # Build probe series
            df_probe = sel[["DateTime", ch]].rename(columns={ch: "Value"}).assign(Source="Probe")
            # Overlay Tempstick if available
            if ts_df is not None and ch in ts_df.columns:
                df_ts_sel = ts_df[
                    (ts_df["DateTime"].dt.year == year) &
                    (ts_df["DateTime"].dt.month == month)
                ][["DateTime", ch]].rename(columns={ch: "Value"}).assign(Source="Tempstick")
                df_chart = pd.concat([df_probe, df_ts_sel], ignore_index=True)
            else:
                df_chart = df_probe

            # Compute Y domain
            low_val, high_val = ranges[name][ch]
            dmin, dmax = df_chart["Value"].min(), df_chart["Value"].max()
            domain_min, domain_max = min(dmin, low_val), max(dmax, high_val)

            # Build base chart
            base = alt.Chart(df_chart).encode(
                x=alt.X("DateTime:T", title="Date/Time",
                        scale=alt.Scale(domain=[
                            pd.Timestamp(year, month, 1),
                            pd.Timestamp(year, month, 1) + MonthEnd(0)
                        ])),
                y=alt.Y("Value:Q", title=f"{ch} ({'°C' if 'Temp' in ch else '%RH'})",
                        scale=alt.Scale(domain=[domain_min, domain_max]))
            )

            # Lines & legend
            line = base.mark_line().encode(color=alt.Color("Source:N", title="Series"))

            # Threshold rules
            rules = alt.Chart(pd.DataFrame({"y": [low_val, high_val]})).mark_rule(
                strokeDash=[5, 5], color="red"
            ).encode(y="y:Q")

            # Final
            chart = (line + rules).properties(
                title=f"{chart_title} – {ch} | Materials: {materials or 'None'} "
                      f"| Probe: {probe_id or '–'} | Equipment: {equip_id or '–'}"
            )
            st.altair_chart(chart, use_container_width=True)
