import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="📈")

# Sidebar
st.sidebar.header("Data Upload & Configuration")
probe_files = st.sidebar.file_uploader(
    "Upload Probe CSV files", type=["csv"], accept_multiple_files=True
)
if not probe_files:
    st.sidebar.info("Upload Probe CSV files to begin.")
    st.stop()

tempstick_files = st.sidebar.file_uploader(
    "Upload Tempstick CSV files (optional)", type=["csv"], accept_multiple_files=True, key="tempstick"
)

tempdfs = {}
# Tempstick parsing
for f in tempstick_files or []:
    raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
    try:
        idx = max(i for i,line in enumerate(raw) if 'timestamp' in line.lower())
    except ValueError:
        continue
    text = ''.join(raw[idx:])
    df_ts = pd.read_csv(io.StringIO(text), on_bad_lines='skip', skip_blank_lines=True)
    df_ts.columns = [c.strip() for c in df_ts.columns]
    # find timestamp col
    ts_col = next((c for c in df_ts.columns if 'timestamp' in c.lower()), None)
    if not ts_col:
        continue
    df_ts['DateTime'] = pd.to_datetime(df_ts[ts_col], errors='coerce')
    # temperature/humidity cols
    for col in ['Temperature','Humidity']:
        if col in df_ts.columns:
            df_ts[col] = pd.to_numeric(df_ts[col], errors='coerce')
    tempdfs[f.name] = df_ts[['DateTime'] + [c for c in ['Temperature','Humidity'] if c in df_ts.columns]]

# Probe parsing
parsed = {}
for f in probe_files:
    raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
    try:
        header = max(i for i,line in enumerate(raw) if 'ch1' in line.lower() or 'p1' in line.lower())
    except ValueError:
        continue
    text = ''.join(raw[header:])
    df = pd.read_csv(io.StringIO(text), on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    # map cols
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if 'date' in lc: col_map[c] = 'Date'
        elif 'time' in lc: col_map[c] = 'Time'
        elif 'p1' in lc: col_map[c] = 'P1'
        elif 'p2' in lc: col_map[c] = 'P2'
        elif 'ch3' in lc: col_map[c] = 'CH3'
        elif 'ch4' in lc: col_map[c] = 'CH4'
        elif 'ch1' in lc: col_map[c] = 'CH3'
        elif 'ch2' in lc: col_map[c] = 'CH4'
    df = df.rename(columns=col_map)
    # datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'], errors='coerce')
    parsed[f.name] = df

dfs = parsed
# define ranges
ranges = {}
for name, df in dfs.items():
    lname = name.lower()
    has_fridge = 'fridge' in lname
    has_freezer = 'freezer' in lname
    is_combo = has_fridge and has_freezer
    is_room = not (has_fridge or has_freezer)
    is_olympus = 'olympus' in lname
    if is_combo:
        ranges[name] = {'Fridge Temp': (2, 8), 'Freezer Temp': (-35, -5)}
    elif has_fridge:
        ranges[name] = {'Temperature': (2, 8)}
    elif has_freezer:
        ranges[name] = {'Temperature': (-35, -5)}
    elif is_room and is_olympus:
        ranges[name] = {'Humidity': (0, 60), 'Temperature': (15, 28)}
    else:
        ranges[name] = {'Humidity': (0, 60), 'Temperature': (15, 25)}

# Year & Month selection
all_years = set()
for df in dfs.values():
    if 'Date' in df.columns:
        all_years.update(df['Date'].dt.year.dropna().astype(int).unique())
all_years = sorted(all_years)
if not all_years:
    st.error("No valid dates found in uploaded data.")
    st.stop()
year = st.sidebar.selectbox("Select Year", all_years)
months = list(range(1, 13))
month = st.sidebar.selectbox("Select Month", months, format_func=lambda m: calendar.month_name[m])

# Main visualization loop
for name, df in dfs.items():
    st.header(name)
    sel = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)].copy().reset_index(drop=True)
    if sel.empty:
        st.warning(f"No data for {name} in {calendar.month_name[month]} {year}.")
        continue
    channels = list(ranges[name].keys())
    for ch in channels:
        df_chart = sel[['DateTime', ch]].rename(columns={ch: 'Value'})
        df_chart['Source'] = 'Probe'
        # Compute domain
        data_min = df_chart['Value'].min()
        data_max = df_chart['Value'].max()
        low_val, high_val = ranges[name][ch]
        domain_min = min(data_min, low_val)
        domain_max = max(data_max, high_val)
        # Build chart
        line = alt.Chart(df_chart).mark_line().encode(
            x=alt.X('DateTime:T', title='Date/Time', scale=alt.Scale(domain=[sel['DateTime'].min(), sel['DateTime'].max()])),
            y=alt.Y('Value:Q', title=f"{ch}", scale=alt.Scale(domain=[domain_min, domain_max]))
        )
        low_rule = alt.Chart(pd.DataFrame({'y': [low_val]})).mark_rule(color='red', strokeDash=[4,4]).encode(y='y:Q')
        high_rule = alt.Chart(pd.DataFrame({'y': [high_val]})).mark_rule(color='red', strokeDash=[4,4]).encode(y='y:Q')
        chart = (line + low_rule + high_rule).properties(
            title=f"{name} - {ch} | {calendar.month_name[month]} {year}"
        )
        st.altair_chart(chart, use_container_width=True)
    # OOR summary
    st.subheader("Out-of-Range Summary")
    # (Assuming OOR events were computed earlier and stored)

