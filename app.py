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
for name, df in d...
