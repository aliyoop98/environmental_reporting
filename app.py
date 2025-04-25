import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="📈")

# Sidebar Uploads
st.sidebar.header("Data Upload & Configuration")
probe_files = st.sidebar.file_uploader(
    "Upload Probe CSV files", type=["csv"], accept_multiple_files=True
)
if not probe_files:
    st.sidebar.info("Upload Probe CSV files to begin.")
    st.stop()

tempstick_files = st.sidebar.file_uploader(
    "Upload Tempstick CSV files (optional)",
    type=["csv"], accept_multiple_files=True, key="tempstick_uploads"
)

# Parse Tempsticks
tempdfs = {}
for f in tempstick_files or []:
    text = f.read().decode('utf-8', errors='ignore')
    df = pd.read_csv(io.StringIO(text), on_bad_lines='skip')
    df = df.rename(columns=str.strip)
    ts_cols = df.columns.str.lower()
    if 'timestamp' in ts_cols:
        df['DateTime'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
        if 'Humidity' in df.columns:
            df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')
        tempdfs[f.name] = df[['DateTime', 'Temperature', 'Humidity']] if 'Humidity' in df.columns else df[['DateTime', 'Temperature']]

# Parse Probes
dfs = {}
ranges = {}
for f in probe_files:
    name = f.name
    raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
    try:
        idx = max(i for i, line in enumerate(raw) if 'ch1' in line.lower() or 'p1' in line.lower())
    except ValueError:
        continue
    content = ''.join(raw[idx:])
    df = pd.read_csv(io.StringIO(content), on_bad_lines='skip').rename(columns=str.strip)
    lname = name.lower()
    has_fridge = 'fridge' in lname
    has_freezer = 'freezer' in lname
    is_combo = has_fridge and has_freezer
    is_room = not (has_fridge or has_freezer)
    is_olympus = 'olympus' in lname
    if is_combo:
        mapping = {'Fridge Temp':'P1', 'Freezer Temp':'P2'}
        ranges[name] = {'Fridge Temp':(2,8), 'Freezer Temp':(-35,-5)}
    elif has_fridge:
        mapping = {'Temperature':'P1'}
        ranges[name] = {'Temperature':(2,8)}
    elif has_freezer:
        mapping = {'Temperature':'P1'}
        ranges[name] = {'Temperature':(-35,-5)}
    else:
        cols = [c.lower() for c in df.columns]
        key_h = 'CH3' if 'ch4' in ''.join(cols) else 'CH1'
        key_t = 'CH4' if 'ch4' in ''.join(cols) else 'CH2'
        mapping = {'Humidity':key_h, 'Temperature':key_t}
        temp_range = (15,28) if is_olympus else (15,25)
        ranges[name] = {'Temperature':temp_range, 'Humidity':(0,60)}
    mapping.update({'Date':'Date', 'Time':'Time'})
    col_map = {}
    for new, key in mapping.items():
        col = next((c for c in df.columns if key.lower() in c.lower()), None)
        if col: col_map[col] = new
    df = df[list(col_map)].rename(columns=col_map)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d ')+df['Time'], errors='coerce')
    for c in ['Fridge Temp','Freezer Temp','Temperature','Humidity']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('+',''), errors='coerce')
    dfs[name] = df.reset_index(drop=True)
# ——————— TEMPSTICK UPLOADER & DEBUG ———————
uploaded_ts = st.sidebar.file_uploader(
    "Upload Tempstick CSV files (optional)",
    type=["csv"],
    accept_multiple_files=True,
    key="tempstick_uploader"
)

# Debug: list out what Streamlit actually sees
if uploaded_ts is not None:
    st.sidebar.write("Tempstick files:", [f.name for f in uploaded_ts])

# Only parse & show selector if we truly have files
tempdfs = {}
if uploaded_ts:
    for ts_file in uploaded_ts:
        # Quick read (no stripping) just to confirm upload
        df = pd.read_csv(ts_file, on_bad_lines="skip")
        tempdfs[ts_file.name] = df

if tempdfs:
    ts_choice = st.sidebar.selectbox(
        "Select Tempstick to overlay",
        options=list(tempdfs.keys()),
        key="tempstick_choice"
    )
    st.sidebar.write(f"Overlaying: {ts_choice}")
    ts_df = tempdfs[ts_choice]
else:
    ts_df = None

# Year & Month Selection
years = sorted({d.year for df in dfs.values() for d in df['Date'].dropna()})
months = sorted({d.month for df in dfs.values() for d in df['Date'].dropna()})
year = st.sidebar.selectbox('Year', years)
month = st.sidebar.selectbox('Month', months, format_func=lambda m: calendar.month_name[m])
start_date = datetime(year,month,1)
end_date = start_date + timedelta(days=calendar.monthrange(year,month)[1]-1)

# Main Loop
for name, df in dfs.items():
    st.header(name)
    # Metadata inputs
    col1, col2 = st.columns(2)
    chart_title = col1.text_input('Chart Title', value=f"{name} {calendar.month_name[month]} {year}", key=f"{name}_title")
    materials = col1.text_area('Materials', key=f"{name}_materials")
    probe_id = col2.text_input('Probe ID', key=f"{name}_probe")
    equipment_id = col2.text_input('Equipment ID', key=f"{name}_equip")
    ts_df = None
    if tempdfs:
        ts_choice = col1.selectbox(
            'Tempstick to overlay',
            options=[None] + list(tempdfs.keys()),
            key=f"{name}_ts"
        )
        ts_df = tempdfs.get(ts_choice)
    else:
        col1.info("Upload Tempstick CSV files to overlay.")

    sel = df[(df['Date'].dt.year==year)&(df['Date'].dt.month==month)].sort_values('DateTime')
    if sel.empty:
        st.warning('No data for this period')
        continue
    channels = list(ranges[name].keys())
    for ch in channels:
        # Prepare data
        probe_sub = sel[['DateTime', ch]].rename(columns={ch:'Value'})
        probe_sub['Source']='Probe'
        df_chart = probe_sub.copy()
        if ts_df is not None and ch in ts_df.columns:
            ts_sub = ts_df[['DateTime', ch]].rename(columns={ch:'Value'})
            ts_sub['Source']='Tempstick'
            df_chart = pd.concat([df_chart, ts_sub], ignore_index=True)
        low, high = ranges[name][ch]
        data_min = df_chart['Value'].min()
        data_max = df_chart['Value'].max()
        domain_min = min(data_min, low)
        domain_max = max(data_max, high)
        pad = (domain_max-domain_min)*0.1
        ymin = domain_min - pad
        ymax = domain_max + pad
        # Build chart
        base = alt.Chart(df_chart).encode(
            x=alt.X('DateTime:T', title='Date/Time', scale=alt.Scale(domain=[start_date, end_date])),
            y=alt.Y('Value:Q', title=f"{ch} ({'°C' if 'Temp' in ch else '%RH'})", scale=alt.Scale(domain=[ymin, ymax])),
            color='Source:N'
        )
        line = base.mark_line()
        rules = [
            alt.Chart(pd.DataFrame({'y':[low]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q'),
            alt.Chart(pd.DataFrame({'y':[high]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q')
        ]
        chart = alt.layer(line, *rules).properties(
            title=f"{chart_title} | Materials: {materials.replace('', ' ') or 'None'} | Probe ID: {probe_id} | Equipment: {equipment_id} | {ch} Range [{low}, {high}]"
        )
        st.altair_chart(chart, use_container_width=True)
