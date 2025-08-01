import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="⛅")
st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(#e3f2fd, #bbdefb); }
        [data-testid="stSidebar"] { background: linear-gradient(#b3e5fc, #e1f5fe); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("🌦️ Data Upload & Configuration")
probe_files = st.sidebar.file_uploader(
    "Upload Probe CSV files",
    accept_multiple_files=True,
    key="probe_uploader"
)
if not probe_files:
    st.sidebar.info("Upload Probe CSV files to begin.")
    st.stop()

tempstick_files = st.sidebar.file_uploader(
    "Upload Tempstick CSV files (optional)",
    accept_multiple_files=True,
    key="tempstick_uploads"
)

# Parse Tempstick CSVs
tempdfs = {}
if tempstick_files:
    for f in tempstick_files:
        raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
        try:
            idx = max(i for i, line in enumerate(raw) if 'timestamp' in line.lower())
        except ValueError:
            continue
        csv_text = ''.join(raw[idx:])
        df_ts = pd.read_csv(io.StringIO(csv_text), on_bad_lines='skip').rename(columns=str.strip)
        ts_col = next((c for c in df_ts.columns if 'timestamp' in c.lower()), None)
        if not ts_col:
            continue
        df_ts = df_ts.rename(columns={ts_col: 'DateTime'})
        df_ts['DateTime'] = pd.to_datetime(df_ts['DateTime'], infer_datetime_format=True, errors='coerce')
        temp_col = next((c for c in df_ts.columns if 'temp' in c.lower()), None)
        if temp_col:
            df_ts['Temperature'] = pd.to_numeric(df_ts[temp_col].astype(str).str.replace('+',''), errors='coerce')
        hum_col = next((c for c in df_ts.columns if 'hum' in c.lower()), None)
        if hum_col:
            df_ts['Humidity'] = pd.to_numeric(df_ts[hum_col].astype(str).str.replace('+',''), errors='coerce')
        cols = ['DateTime']
        if 'Temperature' in df_ts.columns:
            cols.append('Temperature')
        if 'Humidity' in df_ts.columns:
            cols.append('Humidity')
        tempdfs[f.name] = df_ts[cols]

# Parse Probe CSVs
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
        temp_range = (15, 28) if 'olympus' in lname else (15, 25)
        ranges[name] = {'Temperature': temp_range, 'Humidity': (0, 60)}
    mapping.update({'Date': 'Date', 'Time': 'Time'})
    col_map = {}
    for new, key in mapping.items():
        col = next((c for c in df.columns if key.lower() in c.lower()), None)
        if col:
            col_map[col] = new
    df = df[list(col_map)].rename(columns=col_map)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
    df['DateTime'] = pd.to_datetime(
        df['Date'].dt.strftime('%Y-%m-%d ') + df['Time'].astype(str),
        errors='coerce'
    )
    for c in df.columns.difference(['Date', 'Time', 'DateTime']):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace('+',''), errors='coerce')
    dfs[name] = df.reset_index(drop=True)

# Year & Month selection
years = sorted({dt.year for df in dfs.values() for dt in df['Date'].dropna()})
months = sorted({dt.month for df in dfs.values() for dt in df['Date'].dropna()})
year = st.sidebar.selectbox("Year", years)
month = st.sidebar.selectbox("Month", months, format_func=lambda m: calendar.month_name[m])

for name, df in dfs.items():
    st.header(name)
    ts_choice = st.selectbox(f"Match Tempstick for {name}", [None] + list(tempdfs.keys()), key=f"ts_{name}") if tempdfs else None
    title = st.text_input(f"Chart Title", value=f"{name} - {calendar.month_name[month]} {year}", key=f"title_{name}")
    materials = st.text_input("Materials List", key=f"materials_{name}")
    probe_id = st.text_input("Probe ID", key=f"probe_{name}")
    equip_id = st.text_input("Equipment ID", key=f"equip_{name}")
    channel_keys = list(ranges[name].keys())
    channels = st.multiselect("Channels to plot", options=channel_keys, default=channel_keys, key=f"channels_{name}")

    if not st.button(f"Generate {name}", key=f"btn_{name}"):
        continue

    sel = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)].sort_values('DateTime').reset_index(drop=True)
    if sel.empty:
        st.warning("No data for selected period.")
        continue

    ts_df = tempdfs.get(ts_choice) if ts_choice else None
    if ts_df is not None:
        ts_df = ts_df[(ts_df['DateTime'].dt.year == year) & (ts_df['DateTime'].dt.month == month)]

    start_date = datetime(year, month, 1)
    end_date = start_date + timedelta(days=calendar.monthrange(year, month)[1] - 1)

    for ch in channels:
        probe_sub = sel[['DateTime', ch]].rename(columns={ch: 'Value'})
        probe_sub['Legend'] = 'Probe'
        df_chart = probe_sub.copy()
        if ts_df is not None and ch in ts_df.columns:
            ts_sub = ts_df[['DateTime', ch]].rename(columns={ch: 'Value'})
            ts_sub['Legend'] = 'Tempstick'
            df_chart = pd.concat([df_chart, ts_sub], ignore_index=True)
        df_chart = df_chart.dropna(subset=['Value'])
        data_min = df_chart['Value'].min()
        data_max = df_chart['Value'].max()
        lo, hi = ranges[name].get(ch, (data_min, data_max))
        raw_min, raw_max = min(data_min, lo), max(data_max, hi)
        span = (raw_max - raw_min) or 1
        pad = span * 0.1
        ymin, ymax = raw_min - pad, raw_max + pad
        base = alt.Chart(df_chart).encode(
            x=alt.X('DateTime:T', title='Date/Time', scale=alt.Scale(domain=[start_date, end_date])),
            y=alt.Y('Value:Q', title=f"{ch} ({'°C' if 'Temp' in ch else '%RH'})", scale=alt.Scale(domain=[ymin, ymax], nice=False)),
            color=alt.Color('Legend:N')
        )
        line = base.mark_line()
        layers = [line]
        if ch in ranges[name]:
            lo, hi = ranges[name][ch]
            limits_df = pd.DataFrame({'y': [lo, hi], 'Legend': ['Lower Limit', 'Upper Limit']})
            layers.append(
                alt.Chart(limits_df)
                .mark_rule(strokeDash=[4, 4])
                .encode(y='y:Q', color=alt.Color('Legend:N'))
            )
        title_lines = [
            f"{title} - {ch}",
            f"Materials: {materials} | Probe: {probe_id} | Equipment: {equip_id}"
        ]
        chart = alt.layer(*layers).properties(
            title={"text": title_lines, "anchor": "start"}
        ).configure_title(fontSize=14, lineHeight=20, offset=10)
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Out-of-Range Events")
    col_objs = st.columns(len(channels))
    for i, ch in enumerate(channels):
        ch_lo, ch_hi = ranges[name].get(ch, (None, None))
        if ch_lo is None or ch not in sel.columns:
            continue
        df_ch = sel[['DateTime', ch]].copy()
        df_ch['OOR'] = df_ch[ch].apply(lambda v: pd.notna(v) and (v < ch_lo or v > ch_hi))
        df_ch['Group'] = (df_ch['OOR'] != df_ch['OOR'].shift(fill_value=False)).cumsum()
        events = []
        for gid, grp in df_ch.groupby('Group'):
            if not grp['OOR'].iloc[0]:
                continue
            start = grp['DateTime'].iloc[0]
            last_idx = grp.index[-1]
            if last_idx + 1 < len(sel) and not df_ch.loc[last_idx+1, 'OOR']:
                end = sel.loc[last_idx+1, 'DateTime']
            else:
                end = grp['DateTime'].iloc[-1]
            duration = max((end - start).total_seconds() / 60, 0)
            events.append({'Start': start, 'End': end, 'Duration(min)': duration})
        with col_objs[i]:
            st.markdown(f"### {ch} OOR Events")
            if events:
                ev_df = pd.DataFrame(events)
                total = ev_df['Duration(min)'].sum()
                incident = total >= 60 or any(ev_df['Duration(min)'] > 60)
                st.table(ev_df)
                st.write(f"**Total OOR minutes:** {total:.1f}")
                st.write(f"**Incident:** {'YES' if incident else 'No'}")
            else:
                st.info("No out-of-range events detected.")
