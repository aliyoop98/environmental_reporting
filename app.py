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
for f in tempstick_files or []:
    raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
    try:
        idx = max(i for i, line in enumerate(raw) if 'timestamp' in line.lower())
    except ValueError:
        continue
    text = ''.join(raw[idx:])
    df_ts = pd.read_csv(io.StringIO(text), on_bad_lines='skip', skip_blank_lines=True, dtype=str)
    df_ts.columns = [c.strip() for c in df_ts.columns]
    ts_col = next((c for c in df_ts.columns if 'timestamp' in c.lower()), None)
    if not ts_col:
        continue
    df_ts['DateTime'] = pd.to_datetime(df_ts[ts_col], errors='coerce')
    
    # Detect temperature and humidity columns
    temp_col = next((c for c in df_ts.columns if 'temperature' in c.lower()), None)
    humidity_col = next((c for c in df_ts.columns if 'humidity' in c.lower()), None)
    
    # Standardize column names
    if temp_col:
        df_ts = df_ts.rename(columns={temp_col: 'Temperature'})
        df_ts['Temperature'] = pd.to_numeric(df_ts['Temperature'], errors='coerce')
    if humidity_col:
        df_ts = df_ts.rename(columns={humidity_col: 'Humidity'})
        df_ts['Humidity'] = pd.to_numeric(df_ts['Humidity'], errors='coerce')
    
    tempdfs[f.name] = df_ts[['DateTime'] + [c for c in ['Temperature', 'Humidity'] if c in df_ts.columns]]

# Parse Probe CSVs
parsed_probes = {}
for f in probe_files:
    raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
    try:
        header = max(i for i, line in enumerate(raw) if 'ch1' in line.lower() or 'p1' in line.lower())
    except ValueError:
        continue
    text = ''.join(raw[header:])
    df = pd.read_csv(io.StringIO(text), on_bad_lines='skip', dtype=str)
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if 'date' in lc:
            col_map[c] = 'Date'
        elif 'time' in lc:
            col_map[c] = 'Time'
        elif 'p1' in lc:
            col_map[c] = 'P1'
        elif 'p2' in lc:
            col_map[c] = 'P2'
        elif 'ch3' in lc or 'humidity' in lc:
            col_map[c] = 'Humidity'
        elif 'ch4' in lc or 'temperature' in lc:
            col_map[c] = 'Temperature'
        elif 'ch1' in lc:
            col_map[c] = 'Humidity'
        elif 'ch2' in lc:
            col_map[c] = 'Temperature'
    df = df.rename(columns=col_map)
    
    lname = f.name.lower()
    is_combo = 'fridge' in lname and 'freezer' in lname
    if is_combo:
        df = df.rename(columns={'P1': 'Fridge Temp', 'P2': 'Freezer Temp'})
    
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['DateTime'] = pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        parsed_probes[f.name] = df

# Combine all
dfs = {**parsed_probes, **tempdfs}

# Define ranges
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

# Safe year list
all_years = sorted({
    int(year)
    for df in dfs.values()
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date'])
    for year in df['Date'].dropna().dt.year.unique()
})

if not all_years:
    st.error("No valid years found in uploaded files.")
    st.stop()

year = st.sidebar.selectbox("Year", all_years)
month = st.sidebar.selectbox("Month", list(range(1, 13)), format_func=lambda m: calendar.month_name[m])

# Visualization
for name, df in dfs.items():
    st.header(name)
    if 'Date' not in df.columns:
        st.warning(f"No 'Date' column found for {name}. Skipping...")
        continue
    sel = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)].copy().reset_index(drop=True)
    if sel.empty:
        st.warning(f"No data for {name} in {calendar.month_name[month]} {year}.")
        continue
    channels = list(ranges[name].keys())
    for ch in channels:
        if ch not in sel.columns:
            st.warning(f"Channel {ch} not found in data for {name}. Skipping...")
            continue
        df_chart = sel[['DateTime', ch]].rename(columns={ch: 'Value'})
        df_chart['Value'] = pd.to_numeric(df_chart['Value'], errors='coerce').dropna()
        df_chart = df_chart.dropna(subset=['DateTime'])
        if df_chart.empty:
            st.warning(f"No valid data for {ch} in {name}.")
            continue
        data_min = df_chart['Value'].min()
        data_max = df_chart['Value'].max()
        low_val, high_val = ranges[name][ch]
        domain_min = min(data_min, low_val)
        domain_max = max(data_max, high_val)
        line = alt.Chart(df_chart).mark_line().encode(
            x=alt.X('DateTime:T', title='Date/Time', scale=alt.Scale(domain=[sel['DateTime'].min(), sel['DateTime'].max()])),
            y=alt.Y('Value:Q', title=ch + (" (°C)" if "Temp" in ch else "%RH"), 
                    scale=alt.Scale(domain=[domain_min, domain_max]))
        )
        low_rule = alt.Chart(pd.DataFrame({'y': [low_val]})).mark_rule(color='red', strokeDash=[4, 4]).encode(y='y:Q')
        high_rule = alt.Chart(pd.DataFrame({'y': [high_val]})).mark_rule(color='red', strokeDash=[4, 4]).encode(y='y:Q')
        chart = (line + low_rule + high_rule).properties(
            title=f"{name} - {ch}: {calendar.month_name[month]} {year}"
        )
        st.altair_chart(chart, use_container_width=True)
    st.subheader("Out-of-Range Summary")
